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
LOG_STD_RANGE = (-5.0, 1.0)
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

    def __init__(self, hidden=256, aim="categorical", first_stride=2):
        super().__init__()
        if aim not in ("categorical", "gaussian"):
            raise ValueError("aim must be 'categorical' or 'gaussian'")
        self.spec = {"hidden": hidden, "aim": aim, "first_stride": first_stride}
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
        return self.log_std.clamp(*LOG_STD_RANGE).exp()

    def _split(self, out):
        sizes = DISCRETE + ((2,) if self.gaussian else (AIM_BINS,))
        chunks = torch.split(out, sizes, dim=-1)
        return chunks[:len(DISCRETE)], chunks[-1]

    def act(self, vec, codes, generator=None):
        """Sample actions; returns actions (B, action_dim) float, log prob (B,), value (B,), entropy (B,)."""
        out, value = self(vec, codes)
        logits, aim = self._split(out)
        if not self.gaussian:
            logits = logits + (aim,)
        actions, logp, ent = [], 0, 0
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
            logp = logp + d.log_prob(a).sum(-1)
            ent = ent + d.entropy().sum(-1)
        return torch.cat(actions, dim=-1), logp, value, ent

    def evaluate(self, vec, codes, actions, forward=None):
        """Log prob, entropy and value of `actions`; `forward` may be a compiled version of self.forward."""
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
        if self.gaussian:
            d = torch.distributions.Normal(aim, self._std().expand_as(aim))
            logp = logp + d.log_prob(actions[:, n:]).sum(-1)
            ent = ent + d.entropy().sum(-1)
        return logp, ent, value

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
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
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


class Trainer:
    """Collects rollouts from a VecTeeworldsEnv (grid_format="codes") and runs PPO updates.

    compile=True runs the update's forward/backward through torch.compile (same numbers to ~1e-7,
    about 1.7x faster on this CPU); collection always uses the plain module."""

    def __init__(self, venv, policy, cfg, seed=1, compile=False):
        self.venv, self.policy, self.cfg = venv, policy, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
        self._forward = torch.compile(policy.forward) if compile else None
        self.gen = torch.Generator().manual_seed(seed)
        self.update = 0
        self.agent_steps = 0
        self.obs = venv.reset()
        self.episode_events = np.zeros((venv.num_envs, 2, 5), dtype=np.int64)
        self.finished = []  # per finished episode: (2, 5) event totals

    def collect(self):
        cfg, n = self.cfg, self.venv.num_envs
        b = 2 * n
        T = cfg.steps
        vec = torch.zeros(T, b, P.VEC_DIM)
        codes = torch.zeros(T, b, P.GRID_HEIGHT, P.GRID_WIDTH, dtype=torch.uint8)
        actions = torch.zeros(T, b, self.policy.action_dim)
        logp = torch.zeros(T, b)
        values = torch.zeros(T, b)
        rewards = torch.zeros(T, b)
        dones = torch.zeros(T, b)
        ent = torch.zeros(T, b)
        for t in range(T):
            v, c = flatten_agents(self.obs)
            with torch.no_grad():
                a, lp, val, e = self.policy.act(v, c, self.gen)
            vec[t], codes[t], actions[t], logp[t], values[t], ent[t] = v, c, a, lp, val, e
            self.obs, r, term, trunc, infos = self.venv.step(self.policy.env_actions(a.numpy().reshape(n, 2, -1)))
            r = torch.from_numpy(r.reshape(-1).copy())
            done = np.repeat(term | trunc, 2)
            self.episode_events += infos["events"][..., :5]
            if "final_index" in infos:
                # truncation is a time limit, not a real end: bootstrap from the last observation
                idx = infos["final_index"]
                fv = torch.from_numpy(infos["final_vec"].reshape(-1, P.VEC_DIM))
                fc = torch.from_numpy(infos["final_grid"].reshape(-1, P.GRID_HEIGHT, P.GRID_WIDTH))
                with torch.no_grad():
                    _, fval = self.policy(fv, fc)
                rows = torch.from_numpy(np.stack([2 * idx, 2 * idx + 1], axis=1).reshape(-1))
                trunc_rows = torch.from_numpy(np.repeat(trunc[idx], 2))
                r[rows] += cfg.gamma * fval * trunc_rows.float()
                for i in idx:
                    self.finished.append(self.episode_events[i].copy())
                    self.episode_events[i] = 0
            rewards[t] = r
            dones[t] = torch.from_numpy(done.astype(np.float32))
        v, c = flatten_agents(self.obs)
        with torch.no_grad():
            _, last_value = self.policy(v, c)
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
                "adv": adv.reshape(-1), "returns": (adv + values).reshape(-1), "rewards": rewards, "entropy": ent.mean().item()}

    def train(self, batch):
        cfg = self.cfg
        total = batch["adv"].shape[0]
        stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm")}
        for _ in range(cfg.epochs):
            perm = torch.randperm(total, generator=self.gen)
            for start in range(0, total, cfg.minibatch):
                mb = perm[start:start + cfg.minibatch]
                logp, ent, value = self.policy.evaluate(batch["vec"][mb], batch["codes"][mb], batch["actions"][mb], self._forward)
                adv = batch["adv"][mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = torch.exp(logp - batch["logp"][mb])
                pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                vl = 0.5 * ((value - batch["returns"][mb]) ** 2).mean()
                el = ent.mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * el
                self.opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                if not (torch.isfinite(loss) and torch.isfinite(gn)):
                    raise FloatingPointError(f"non-finite loss {loss.item()} or grad norm {gn.item()} at update {self.update}")
                self.opt.step()
                with torch.no_grad():
                    log_ratio = logp - batch["logp"][mb]
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["clip_frac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                stats["policy_loss"].append(pg.item())
                stats["value_loss"].append(vl.item())
                stats["entropy"].append(el.item())
                stats["grad_norm"].append(gn.item())
        self.update += 1
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def state_dict(self):
        return {"format": CHECKPOINT_FORMAT, "model": dict(self.policy.spec), "policy": self.policy.state_dict(), "optimizer": self.opt.state_dict(),
                "update": self.update, "agent_steps": self.agent_steps, "config": vars(self.cfg),
                # random streams: action sampling and minibatch order, torch's global stream, and how many
                # episodes each env has started, so a resumed run does not replay the same starts
                "generator": self.gen.get_state(), "torch_rng": torch.get_rng_state(),
                "env_episodes": list(getattr(self.venv, "_episodes", []))}

    def load_state_dict(self, state):
        if model_spec(state) != self.policy.spec:
            raise ValueError(f"checkpoint model {model_spec(state)} differs from this policy {self.policy.spec}")
        self.policy.load_state_dict(state["policy"])
        self.opt.load_state_dict(state["optimizer"])
        self.update = state["update"]
        self.agent_steps = state["agent_steps"]
        self.gen.set_state(state["generator"])
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
    network (categorical aim, stride-1 first convolution)."""
    return state.get("model", {"hidden": 256, "aim": "categorical", "first_stride": 1})


def load_policy(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    policy = Policy(**model_spec(state))
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy, state
