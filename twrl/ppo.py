"""PPO self-play on top of VecTeeworldsEnv.

One network plays both agents of every environment, so a vector of N environments gives 2N
samples per step. The network starts from random weights; nothing in here knows how to play.
Rewards are the environment's own (kills, deaths, suicides, small damage terms).

Action heads: direction 3, jump 2, fire 2 and hook 2 are categorical. Aim is either a categorical
over 16 directions (aim="categorical") or a 2D Gaussian vector (aim="gaussian": state-dependent mean,
learned state-independent log std) whose direction becomes the aim target; Teeworlds only uses the
direction of the target vector.
"""

import math
import os

import numpy as np
import torch
from torch import nn

from . import protocol as P

AIM_BINS = 16
DISCRETE = (3, 2, 2, 2)  # direction, jump, fire, hook
HEADS = DISCRETE + (AIM_BINS,)  # with the categorical aim
AIM_RADIUS = 256  # length of the integer target vector sent to the env
LOG_STD_INIT = -0.5
LOG_STD_MIN = -5.0
LOG_STD_MAX = 0.0  # the A/B run's Gaussian arm used 1.0 and its std grew straight to it
_AIM = np.array([[round(256 * math.cos(2 * math.pi * k / AIM_BINS)), round(256 * math.sin(2 * math.pi * k / AIM_BINS))]
                 for k in range(AIM_BINS)], dtype=np.int32)


def to_env_actions(heads):
    """(..., 5) head indices -> (..., 6) int32 env actions."""
    heads = np.asarray(heads)
    out = np.empty(heads.shape[:-1] + (6,), dtype=np.int32)
    out[..., 0] = heads[..., 0] - 1
    out[..., 1:4] = heads[..., 1:4]
    out[..., 4:6] = _AIM[heads[..., 4]]
    return out


def gaussian_env_actions(actions):
    """(..., 6) float actions (4 head indices, aim x, aim y) -> (..., 6) int32 env actions."""
    actions = np.asarray(actions, dtype=np.float64)
    out = np.empty(actions.shape[:-1] + (6,), dtype=np.int32)
    out[..., 0] = actions[..., 0] - 1
    out[..., 1:4] = actions[..., 1:4]
    aim = actions[..., 4:6]
    norm = np.hypot(aim[..., 0], aim[..., 1])[..., None]
    target = np.where(norm > 1e-9, np.rint(AIM_RADIUS * aim / np.maximum(norm, 1e-9)), [AIM_RADIUS, 0])
    out[..., 4:6] = target
    return out


def _init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _conv_size(n, stride):
    return (n - 1) // stride + 1  # 3x3 kernel, padding 1


class Policy(nn.Module):
    """Small CNN over the collision grid plus an MLP over the vector, shared trunk, actor and critic heads.

    Actions are a float tensor (B, action_dim): the four discrete head indices, then the aim (one
    index for the categorical aim, an x, y vector for the Gaussian one). first_stride=2 halves the
    grid in the first convolution, which was most of the network's cost at full resolution."""

    def __init__(self, hidden=256, aim="categorical", first_stride=2, log_std_max=LOG_STD_MAX, aim_mean="raw"):
        super().__init__()
        if aim not in ("categorical", "gaussian"):
            raise ValueError("aim must be 'categorical' or 'gaussian'")
        self.spec = {"hidden": hidden, "aim": aim, "first_stride": first_stride}
        if aim == "gaussian":
            if aim_mean not in ("raw", "bounded"):
                raise ValueError("aim_mean must be 'raw' or 'bounded'")
            self.spec["log_std_max"] = log_std_max
            self.spec["aim_mean"] = aim_mean
        self.gaussian = aim == "gaussian"
        self.action_dim = len(DISCRETE) + (2 if self.gaussian else 1)
        g = math.sqrt(2)
        self.conv = nn.Sequential(
            _init(nn.Conv2d(3, 16, 3, stride=first_stride, padding=1), g), nn.ReLU(),
            _init(nn.Conv2d(16, 32, 3, stride=2, padding=1), g), nn.ReLU(),
            _init(nn.Conv2d(32, 32, 3, stride=2, padding=1), g), nn.ReLU(),
            nn.Flatten(),
        )
        h_out, w_out = P.GRID_HEIGHT, P.GRID_WIDTH
        for stride in (first_stride, 2, 2):
            h_out, w_out = _conv_size(h_out, stride), _conv_size(w_out, stride)
        self.grid_fc = nn.Sequential(_init(nn.Linear(32 * h_out * w_out, 128), g), nn.ReLU())
        self.vec_fc = nn.Sequential(_init(nn.Linear(P.VEC_DIM, 128), g), nn.ReLU())
        self.trunk = nn.Sequential(_init(nn.Linear(256, hidden), g), nn.ReLU(), _init(nn.Linear(hidden, hidden), g), nn.ReLU())
        self.pi = _init(nn.Linear(hidden, sum(DISCRETE) + (2 if self.gaussian else AIM_BINS)), 0.01)
        if self.gaussian:
            self.log_std = nn.Parameter(torch.full((2,), LOG_STD_INIT))
        self.v = _init(nn.Linear(hidden, 1), 1.0)
        self.register_buffer("shifts", torch.arange(3, dtype=torch.uint8).view(1, 3, 1, 1), persistent=False)

    def forward(self, vec, codes):
        """vec float (B, 67), codes uint8 (B, 25, 33) -> actor outputs (B, k), value (B,)

        The actor outputs are the discrete heads' logits followed by the aim's logits or mean."""
        grid = ((codes.unsqueeze(1) >> self.shifts) & 1).float()
        h = torch.cat([self.grid_fc(self.conv(grid)), self.vec_fc(vec)], dim=1)
        h = self.trunk(h)
        return self.pi(h), self.v(h).squeeze(-1)

    def _std(self):
        return self.log_std.clamp(LOG_STD_MIN, self.spec["log_std_max"]).exp()

    def _split(self, out):
        """Discrete heads' logits and the aim's logits (categorical) or mean (Gaussian).

        Only the aim's direction reaches the game. With a raw mean nothing holds its length, which
        grew to several times the std in the first 5M run and made the aim nearly deterministic
        whatever log_std said; "bounded" maps the mean to mu / sqrt(1 + |mu|^2), the same near zero
        but never longer than 1, so the noise relative to the mean stays what log_std sets."""
        sizes = DISCRETE + ((2,) if self.gaussian else (AIM_BINS,))
        chunks = torch.split(out, sizes, dim=-1)
        aim = chunks[-1]
        if self.gaussian and self.spec["aim_mean"] == "bounded":
            aim = aim * torch.rsqrt(1 + (aim * aim).sum(-1, keepdim=True))
        return chunks[:len(DISCRETE)], aim

    def aim_mean(self, vec, codes):
        """The Gaussian aim's mean as the distribution uses it (B, 2)."""
        return self._split(self(vec, codes)[0])[1]

    def act(self, vec, codes, generator=None):
        """Sample actions; returns actions (B, action_dim) float, log prob (B,), value (B,), entropy (B,) and
        the Gaussian aim's part of the log prob (B,; zeros for the categorical aim, whose part is in logp)."""
        out, value = self(vec, codes)
        logits, aim = self._split(out)
        if not self.gaussian:
            logits = logits + (aim,)
        actions, logp, ent = [], 0, 0
        aim_logp = torch.zeros_like(value)
        for chunk in logits:
            probs = torch.softmax(chunk, dim=-1)
            a = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            d = torch.distributions.Categorical(probs=probs)
            actions.append(a.float().unsqueeze(-1))
            logp = logp + d.log_prob(a)
            ent = ent + d.entropy()
        if self.gaussian:
            std = self._std().expand_as(aim)
            a = aim + std * torch.randn(aim.shape, generator=generator)
            d = torch.distributions.Normal(aim, std)
            actions.append(a)
            aim_logp = d.log_prob(a).sum(-1)
            logp = logp + aim_logp
            ent = ent + d.entropy().sum(-1)
        return torch.cat(actions, dim=-1), logp, value, ent, aim_logp

    def evaluate(self, vec, codes, actions, forward=None):
        """Log prob, entropy, value, Gaussian aim entropy and Gaussian aim log prob of `actions`; `forward`
        may be a compiled version of self.forward. logp and the entropy cover the categorical heads (the
        categorical aim included); logp also includes the Gaussian aim, whose log prob and entropy are
        returned on their own as well (zeros for the categorical aim)."""
        out, value = (forward or self)(vec, codes)
        logits, aim = self._split(out)
        n = len(DISCRETE)
        if not self.gaussian:
            logits, n = logits + (aim,), n + 1
        logp, ent = 0, 0
        for chunk, a in zip(logits, actions[:, :n].long().unbind(-1)):
            d = torch.distributions.Categorical(logits=chunk)
            logp = logp + d.log_prob(a)
            ent = ent + d.entropy()
        aim_ent = aim_logp = torch.zeros_like(value)
        if self.gaussian:
            d = torch.distributions.Normal(aim, self._std().expand_as(aim))
            aim_logp = d.log_prob(actions[:, n:]).sum(-1)
            logp = logp + aim_logp
            aim_ent = d.entropy().sum(-1)
        return logp, ent, value, aim_ent, aim_logp

    def env_actions(self, actions):
        """(..., action_dim) actions as numpy -> (..., 6) int32 env actions."""
        if self.gaussian:
            return gaussian_env_actions(actions)
        return to_env_actions(np.asarray(actions).astype(np.int64))


class PPOConfig:
    def __init__(self, **kw):
        self.num_envs = 8
        self.steps = 128  # per env and update
        self.gamma = 0.99
        self.lam = 0.95
        self.epochs = 4
        self.minibatch = 512
        self.lr = 3e-4
        self.clip = 0.2
        self.ent_coef = 0.01  # categorical heads
        self.aim_ent_coef = 0.0  # Gaussian aim: no entropy bonus (it only inflated the std in the A/B run)
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.adv_norm = "batch"  # normalise advantages once over the whole batch ("minibatch": per minibatch)
        # Gaussian aim: "shots" trains the aim only on decisions where a shot actually left the gun (the aim
        # has no effect on the others), with its own clipped ratio; the other heads have their own ratio
        # and see every decision. "shots+hooks": also where a hook was launched, the only moment the aim
        # steers the hook. "all": one joint ratio over all heads on every decision.
        self.aim_mask = "shots"
        # Gaussian aim stabilisation. aim_lr_scale scales the learning rate of the aim's own parameters (its
        # rows of the output layer and log_std; the shared trunk keeps the full rate). aim_target_kl: once a
        # minibatch's aim KL (on shot decisions) passes 1.5x this, the aim loss is switched off for the rest
        # of the update while the other heads and the value keep training. None: never.
        self.aim_lr_scale = 1.0
        self.aim_target_kl = None
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError(f"unknown PPO setting {k}")
            setattr(self, k, v)


def flatten_agents(obs):
    """{"vec": (N, 2, 67), "grid": (N, 2, H, W) codes} -> torch tensors with B = 2N, env-major."""
    vec = torch.from_numpy(obs["vec"].reshape(-1, P.VEC_DIM))
    codes = torch.from_numpy(obs["grid"].reshape(-1, P.GRID_HEIGHT, P.GRID_WIDTH))
    return vec, codes


CHECKPOINT_FORMAT = 3
BEHAVIOUR = ("alive_steps", "visible_steps", "aim_cos_sum", "shots", "damage_dealt", "kills", "fire_presses",
             "shot_decisions", "hook_launch_decisions", "hook_only_decisions", "dry_fires")  # learner sums


class Trainer:
    """Collects rollouts from a VecTeeworldsEnv (grid_format="codes") and runs PPO updates.

    Self-play by default: the policy plays both agents of every env and learns from both. With a
    frozen `opponent` (a Policy that is never updated) the learner plays agent i % 2 of env i and the
    opponent the other one; only the learner's samples are trained on.

    compile=True runs the update's forward/backward through torch.compile (same numbers to ~1e-7,
    about 1.7x faster on this CPU); collection always uses the plain module."""

    def __init__(self, venv, policy, cfg, seed=1, compile=False, opponent=None):
        self.venv, self.policy, self.cfg = venv, policy, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
        # the aim's own parameters as (tensor, index) pairs; Adam's step is proportional to the rate, so
        # scaling these elements' step after opt.step() is the same as giving them their own rate
        self.aim_params = []
        if policy.gaussian:
            rows = slice(sum(DISCRETE), sum(DISCRETE) + 2)
            self.aim_params = [(policy.pi.weight, rows), (policy.pi.bias, rows), (policy.log_std, slice(None))]
        self._forward = torch.compile(policy.forward) if compile else None
        self.gen = torch.Generator().manual_seed(seed)
        self.opponent = opponent
        n = venv.num_envs
        if opponent is None:
            self.side = None
            self.rows = np.arange(2 * n)  # learner rows of the flattened (env, agent) observations
        else:
            opponent.eval()
            self.opp_gen = torch.Generator().manual_seed(seed + 1_000_003)
            self.side = np.arange(n) % 2
            self.rows = 2 * np.arange(n) + self.side
            self.opp_rows = 2 * np.arange(n) + 1 - self.side
        self.update = 0
        self.agent_steps = 0
        self.obs = venv.reset()
        self.prev_agents = None
        self.episode_events = np.zeros((n, 2, len(P.EVENT_NAMES)), dtype=np.int64)
        self.finished = []  # per finished episode: (2, EVENT_NAMES) totals, learner first when there is an opponent
        self.behaviour = np.zeros(len(BEHAVIOUR))

    def _learner_first(self, events, i):
        return events if self.side is None or self.side[i] == 0 else events[::-1]

    def _track(self, vec, env_actions, infos, ended):
        """Learner behaviour measurements (no effect on training): visibility, aim alignment, shots."""
        v = vec[self.rows]
        a = env_actions.reshape(-1, 6)[self.rows, 4:6].astype(np.float64)
        alive = v[:, P.VEC_INDEX["alive"]] > 0
        vis = alive & (v[:, P.VEC_INDEX["opp_visible"]] > 0)
        d = np.stack([v[:, P.VEC_INDEX["opp_dx"]] * P.VIEW_HALF_WIDTH, v[:, P.VEC_INDEX["opp_dy"]] * P.VIEW_HALF_HEIGHT], 1)
        cos = (a * d).sum(1) / np.maximum(np.hypot(*a.T) * np.hypot(*d.T), 1e-6)
        self.behaviour[0] += alive.sum()
        self.behaviour[1] += vis.sum()
        self.behaviour[2] += cos[vis].sum()
        events = infos["events"].reshape(-1, len(P.EVENT_NAMES))[self.rows]
        self.behaviour[4] += events[:, P.EVENT_NAMES.index("damage_dealt")].sum()
        self.behaviour[5] += events[:, P.EVENT_NAMES.index("kills")].sum()
        shot = events[:, P.EVENT_NAMES.index("shots")] > 0
        launch = events[:, P.EVENT_NAMES.index("hook_launches")] > 0
        self.behaviour[6] += (alive & (env_actions.reshape(-1, 6)[self.rows, 2] == 1)).sum()
        self.behaviour[7] += shot.sum()
        self.behaviour[8] += launch.sum()
        self.behaviour[9] += (launch & ~shot).sum()
        self.behaviour[10] += events[:, P.EVENT_NAMES.index("dry_fires")].sum()
        agents = infos.get("agents")
        if agents is not None:
            ammo, live = agents["ammo"].reshape(-1)[self.rows], agents["alive"].reshape(-1)[self.rows]
            if self.prev_agents is not None:
                same = ~np.repeat(ended, 2)[self.rows]  # an env that was reset starts a new episode
                prev_ammo, prev_live = self.prev_agents
                self.behaviour[3] += ((prev_ammo - ammo > 0) & (prev_live == 1) & (live == 1) & same).sum()
            self.prev_agents = (ammo.copy(), live.copy())

    def collect(self):
        cfg, n = self.cfg, self.venv.num_envs
        rows = torch.from_numpy(self.rows)
        b = len(self.rows)
        T = cfg.steps
        vec = torch.zeros(T, b, P.VEC_DIM)
        codes = torch.zeros(T, b, P.GRID_HEIGHT, P.GRID_WIDTH, dtype=torch.uint8)
        actions = torch.zeros(T, b, self.policy.action_dim)
        logp = torch.zeros(T, b)
        aim_logp = torch.zeros(T, b)
        aim_mask = torch.zeros(T, b)
        values = torch.zeros(T, b)
        rewards = torch.zeros(T, b)
        dones = torch.zeros(T, b)
        ent = torch.zeros(T, b)
        env_actions = np.zeros((2 * n, 6), dtype=np.int32)
        for t in range(T):
            v_all, c_all = flatten_agents(self.obs)
            v, c = v_all[rows], c_all[rows]
            with torch.no_grad():
                a, lp, val, e, alp = self.policy.act(v, c, self.gen)
                env_actions[self.rows] = self.policy.env_actions(a.numpy())
                if self.opponent is not None:
                    opp = torch.from_numpy(self.opp_rows)
                    oa = self.opponent.act(v_all[opp], c_all[opp], self.opp_gen)[0]
                    env_actions[self.opp_rows] = self.opponent.env_actions(oa.numpy())
            vec[t], codes[t], actions[t], logp[t], values[t], ent[t], aim_logp[t] = v, c, a, lp, val, e, alp
            self.obs, r, term, trunc, infos = self.venv.step(env_actions.reshape(n, 2, 6))
            ended = term | trunc
            self._track(v_all.numpy(), env_actions, infos, ended)
            r = torch.from_numpy(r.reshape(-1)[self.rows].copy())
            # the shots (and hook launches) of the ticks this decision covered: where the aim actually mattered
            ev = infos["events"].reshape(-1, len(P.EVENT_NAMES))[self.rows]
            used = ev[:, P.EVENT_NAMES.index("shots")] > 0
            if cfg.aim_mask == "shots+hooks":
                used |= ev[:, P.EVENT_NAMES.index("hook_launches")] > 0
            aim_mask[t] = torch.from_numpy(used.astype(np.float32))
            done = np.repeat(ended, 2)[self.rows]
            self.episode_events += infos["events"]
            if "final_index" in infos:
                # truncation is a time limit, not a real end: bootstrap from the last observation
                idx = infos["final_index"]
                fv = torch.from_numpy(infos["final_vec"].reshape(-1, P.VEC_DIM))
                fc = torch.from_numpy(infos["final_grid"].reshape(-1, P.GRID_HEIGHT, P.GRID_WIDTH))
                # learner rows among the final observations, and where those agents sit in the batch
                final_rows = np.flatnonzero(np.isin(np.repeat(idx, 2) * 2 + np.tile([0, 1], len(idx)), self.rows))
                batch_rows = np.searchsorted(self.rows, (np.repeat(idx, 2) * 2 + np.tile([0, 1], len(idx)))[final_rows])
                with torch.no_grad():
                    _, fval = self.policy(fv[final_rows], fc[final_rows])
                trunc_rows = torch.from_numpy(np.repeat(trunc[idx], 2)[final_rows])
                r[torch.from_numpy(batch_rows)] += cfg.gamma * fval * trunc_rows.float()
                for i in idx:
                    totals = self.episode_events[i].copy()
                    score = P.EVENT_NAMES.index("score")
                    totals[:, score] = infos["events"][i, :, score]  # the score is a running total, not a count
                    self.finished.append(self._learner_first(totals, i))
                    self.episode_events[i] = 0
            rewards[t] = r
            dones[t] = torch.from_numpy(done.astype(np.float32))
        v_all, c_all = flatten_agents(self.obs)
        with torch.no_grad():
            _, last_value = self.policy(v_all[rows], c_all[rows])
        adv = torch.zeros(T, b)
        last = torch.zeros(b)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * next_value * nonterminal - values[t]
            last = delta + cfg.gamma * cfg.lam * nonterminal * last
            adv[t] = last
        self.agent_steps += T * b
        return {"vec": vec.reshape(T * b, -1), "codes": codes.reshape(T * b, P.GRID_HEIGHT, P.GRID_WIDTH),
                "actions": actions.reshape(T * b, -1), "logp": logp.reshape(-1), "values": values.reshape(-1),
                "aim_logp": aim_logp.reshape(-1), "aim_mask": aim_mask.reshape(-1),
                "adv": adv.reshape(-1), "returns": (adv + values).reshape(-1), "rewards": rewards, "entropy": ent.mean().item()}

    def masked_aim(self):
        return self.policy.gaussian and self.cfg.aim_mask in ("shots", "shots+hooks")

    def losses(self, batch, mb, adv, aim_weight=1.0):
        """PPO loss terms for minibatch `mb` with advantages `adv`; also used by tests and audits.
        aim_weight 0 leaves the masked aim term out of the loss (aim KL early stop)."""
        cfg = self.cfg
        logp, ent, value, aim_ent, aim_logp = self.policy.evaluate(batch["vec"][mb], batch["codes"][mb], batch["actions"][mb], self._forward)
        old = batch["logp"][mb]
        out = {}
        if self.masked_aim():
            # the other heads: their own ratio on every decision; the aim: its own ratio, only where a shot left
            old_aim, mask = batch["aim_logp"][mb], batch["aim_mask"][mb]
            ratio = torch.exp((logp - aim_logp) - (old - old_aim))
            aim_ratio = torch.exp(aim_logp - old_aim)
            aim_pg = -(mask * torch.min(aim_ratio * adv, torch.clamp(aim_ratio, 1 - cfg.clip, 1 + cfg.clip) * adv)).sum() / len(mb)
            aim_ent_term = (mask * aim_ent).sum() / len(mb)
            out.update(aim_ratio=aim_ratio, aim_mask=mask, aim_logp=aim_logp, old_aim=old_aim)
        else:
            ratio = torch.exp(logp - old)
            aim_pg = torch.zeros(())
            aim_ent_term = aim_ent.mean()
        pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        vl = 0.5 * ((value - batch["returns"][mb]) ** 2).mean()
        el = ent.mean()
        loss = pg + aim_weight * aim_pg + cfg.vf_coef * vl - cfg.ent_coef * el - cfg.aim_ent_coef * aim_ent_term
        out.update(loss=loss, pg=pg, aim_pg=aim_pg, vl=vl, ent=el, aim_ent=aim_ent.mean(), ratio=ratio,
                   log_ratio=torch.log(ratio))
        return out

    def train(self, batch):
        cfg = self.cfg
        total = batch["adv"].shape[0]
        keys = ("policy_loss", "aim_policy_loss", "value_loss", "entropy", "aim_entropy", "approx_kl", "clip_frac", "grad_norm")
        stats = {k: [] for k in keys + (("aim_approx_kl", "aim_clip_frac", "aim_mask_frac", "aim_off") if self.masked_aim() else ())}
        aim_weight = 1.0
        scale = cfg.aim_lr_scale if self.policy.gaussian else 1.0
        all_adv = batch["adv"]
        if cfg.adv_norm == "batch":
            all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
        for _ in range(cfg.epochs):
            perm = torch.randperm(total, generator=self.gen)
            for start in range(0, total, cfg.minibatch):
                mb = perm[start:start + cfg.minibatch]
                adv = all_adv[mb]
                if cfg.adv_norm == "minibatch":
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                L = self.losses(batch, mb, adv, aim_weight)
                loss = L["loss"]
                self.opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                if not (torch.isfinite(loss) and torch.isfinite(gn)):
                    raise FloatingPointError(f"non-finite loss {loss.item()} or grad norm {gn.item()} at update {self.update}")
                # with the aim loss off its own parameters stay put (Adam's momentum would still move them)
                step = scale * aim_weight
                before = [p.detach()[i].clone() for p, i in self.aim_params] if step != 1.0 else None
                self.opt.step()
                if before is not None:
                    with torch.no_grad():
                        for (p, i), old in zip(self.aim_params, before):
                            p[i] = old + step * (p[i] - old)
                with torch.no_grad():
                    ratio, log_ratio = L["ratio"], L["log_ratio"]
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["clip_frac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                    if self.masked_aim():
                        m = L["aim_mask"]
                        n = m.sum().clamp(min=1)
                        ar = L["aim_ratio"]
                        stats["aim_approx_kl"].append((m * ((ar - 1) - torch.log(ar))).sum().item() / n.item())
                        stats["aim_clip_frac"].append((m * ((ar - 1).abs() > cfg.clip).float()).sum().item() / n.item())
                        stats["aim_mask_frac"].append(m.mean().item())
                        stats["aim_off"].append(float(aim_weight == 0.0))
                        if cfg.aim_target_kl is not None and stats["aim_approx_kl"][-1] > 1.5 * cfg.aim_target_kl:
                            aim_weight = 0.0
                stats["policy_loss"].append(L["pg"].item())
                stats["aim_policy_loss"].append(float(L["aim_pg"].detach()))
                stats["value_loss"].append(L["vl"].item())
                stats["entropy"].append(L["ent"].item())
                stats["aim_entropy"].append(L["aim_ent"].item())
                stats["grad_norm"].append(gn.item())
        self.update += 1
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["grad_norm_max"] = float(np.max(stats["grad_norm"]))
        if self.policy.gaussian:
            out["log_std_x"], out["log_std_y"] = (float(x) for x in self.policy.log_std.detach())
        return out

    def state_dict(self):
        return {"format": CHECKPOINT_FORMAT, "model": dict(self.policy.spec), "policy": self.policy.state_dict(), "optimizer": self.opt.state_dict(),
                "update": self.update, "agent_steps": self.agent_steps, "config": vars(self.cfg),
                # random streams: action sampling and minibatch order, torch's global stream, and how many
                # episodes each env has started, so a resumed run does not replay the same starts
                "generator": self.gen.get_state(), "torch_rng": torch.get_rng_state(),
                "opponent_generator": self.opp_gen.get_state() if self.opponent is not None else None,
                "env_episodes": list(getattr(self.venv, "_episodes", []))}

    def load_state_dict(self, state):
        if model_spec(state) != self.policy.spec:
            raise ValueError(f"checkpoint model {model_spec(state)} differs from this policy {self.policy.spec}")
        self.policy.load_state_dict(state["policy"])
        self.opt.load_state_dict(state["optimizer"])
        self.update = state["update"]
        self.agent_steps = state["agent_steps"]
        self.gen.set_state(state["generator"])
        if self.opponent is not None and state.get("opponent_generator") is not None:
            self.opp_gen.set_state(state["opponent_generator"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"])
        if state.get("env_episodes") and hasattr(self.venv, "_episodes"):
            if len(state["env_episodes"]) != len(self.venv._episodes):
                raise ValueError("checkpoint was made with a different number of environments")
            self.venv._episodes = list(state["env_episodes"])
            # episodes in flight at save time are gone; start fresh ones from the saved seed stream
            self.obs = self.venv.reset()
            self.episode_events[:] = 0


def save_checkpoint(path, trainer, extra=None):
    """Write atomically: an interrupted save never leaves a truncated checkpoint behind."""
    state = trainer.state_dict()
    if extra:
        state.update(extra)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def model_spec(state):
    """The Policy settings a checkpoint was made with; checkpoints before format 3 used the first
    network (categorical aim, stride-1 first convolution). Gaussian checkpoints made before log_std_max
    existed keep the cap they were trained with (1.0), and those made before aim_mean the raw mean."""
    spec = dict(state.get("model", {"hidden": 256, "aim": "categorical", "first_stride": 1}))
    if spec["aim"] == "gaussian":
        spec.setdefault("log_std_max", 1.0)
        spec.setdefault("aim_mean", "raw")
    return spec


def load_policy(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    policy = Policy(**model_spec(state))
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy, state
