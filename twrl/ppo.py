"""PPO self-play on top of VecTeeworldsEnv.

One network plays both agents of every environment, so a vector of N environments gives 2N
samples per step. The network starts from random weights; nothing in here knows how to play.
Rewards are the environment's own (kills, deaths, suicides, small damage terms).

Action heads (factored categorical): direction 3, jump 2, fire 2, hook 2, aim 16 directions.
"""

import math

import numpy as np
import torch
from torch import nn

from . import protocol as P

AIM_BINS = 16
HEADS = (3, 2, 2, 2, AIM_BINS)  # direction, jump, fire, hook, aim
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


def _init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Policy(nn.Module):
    """Small CNN over the collision grid plus an MLP over the vector, shared trunk, actor and critic heads."""

    def __init__(self, hidden=256):
        super().__init__()
        g = math.sqrt(2)
        self.conv = nn.Sequential(
            _init(nn.Conv2d(3, 16, 3, padding=1), g), nn.ReLU(),
            _init(nn.Conv2d(16, 32, 3, stride=2, padding=1), g), nn.ReLU(),
            _init(nn.Conv2d(32, 32, 3, stride=2, padding=1), g), nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 32 * math.ceil(math.ceil(P.GRID_HEIGHT / 2) / 2) * math.ceil(math.ceil(P.GRID_WIDTH / 2) / 2)
        self.grid_fc = nn.Sequential(_init(nn.Linear(conv_out, 128), g), nn.ReLU())
        self.vec_fc = nn.Sequential(_init(nn.Linear(P.VEC_DIM, 128), g), nn.ReLU())
        self.trunk = nn.Sequential(_init(nn.Linear(256, hidden), g), nn.ReLU(), _init(nn.Linear(hidden, hidden), g), nn.ReLU())
        self.pi = _init(nn.Linear(hidden, sum(HEADS)), 0.01)
        self.v = _init(nn.Linear(hidden, 1), 1.0)
        self.register_buffer("shifts", torch.arange(3, dtype=torch.uint8).view(1, 3, 1, 1), persistent=False)

    def forward(self, vec, codes):
        """vec float (B, 67), codes uint8 (B, 25, 33) -> logits (B, sum(HEADS)), value (B,)"""
        grid = ((codes.unsqueeze(1) >> self.shifts) & 1).float()
        h = torch.cat([self.grid_fc(self.conv(grid)), self.vec_fc(vec)], dim=1)
        h = self.trunk(h)
        return self.pi(h), self.v(h).squeeze(-1)

    @staticmethod
    def dists(logits):
        return [torch.distributions.Categorical(logits=chunk) for chunk in torch.split(logits, HEADS, dim=-1)]

    def act(self, vec, codes, generator=None):
        """Sample actions; returns heads (B, 5), log prob (B,), value (B,), entropy (B,)."""
        logits, value = self(vec, codes)
        heads, logp, ent = [], 0, 0
        for chunk in torch.split(logits, HEADS, dim=-1):
            probs = torch.softmax(chunk, dim=-1)
            a = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            d = torch.distributions.Categorical(probs=probs)
            heads.append(a)
            logp = logp + d.log_prob(a)
            ent = ent + d.entropy()
        return torch.stack(heads, dim=-1), logp, value, ent

    def evaluate(self, vec, codes, heads):
        logits, value = self(vec, codes)
        logp, ent = 0, 0
        for d, a in zip(self.dists(logits), heads.unbind(-1)):
            logp = logp + d.log_prob(a)
            ent = ent + d.entropy()
        return logp, ent, value


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


class Trainer:
    """Collects rollouts from a VecTeeworldsEnv (grid_format="codes") and runs PPO updates."""

    def __init__(self, venv, policy, cfg, seed=1):
        self.venv, self.policy, self.cfg = venv, policy, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
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
        heads = torch.zeros(T, b, len(HEADS), dtype=torch.long)
        logp = torch.zeros(T, b)
        values = torch.zeros(T, b)
        rewards = torch.zeros(T, b)
        dones = torch.zeros(T, b)
        ent = torch.zeros(T, b)
        for t in range(T):
            v, c = flatten_agents(self.obs)
            with torch.no_grad():
                a, lp, val, e = self.policy.act(v, c, self.gen)
            vec[t], codes[t], heads[t], logp[t], values[t], ent[t] = v, c, a, lp, val, e
            self.obs, r, term, trunc, infos = self.venv.step(to_env_actions(a.numpy().reshape(n, 2, -1)))
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
                "heads": heads.reshape(T * b, -1), "logp": logp.reshape(-1), "values": values.reshape(-1),
                "adv": adv.reshape(-1), "returns": (adv + values).reshape(-1), "rewards": rewards, "entropy": ent.mean().item()}

    def train(self, batch):
        cfg = self.cfg
        total = batch["adv"].shape[0]
        stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm")}
        for _ in range(cfg.epochs):
            perm = torch.randperm(total, generator=self.gen)
            for start in range(0, total, cfg.minibatch):
                mb = perm[start:start + cfg.minibatch]
                logp, ent, value = self.policy.evaluate(batch["vec"][mb], batch["codes"][mb], batch["heads"][mb])
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
        return {"policy": self.policy.state_dict(), "optimizer": self.opt.state_dict(), "update": self.update,
                "agent_steps": self.agent_steps, "generator": self.gen.get_state(), "config": vars(self.cfg)}

    def load_state_dict(self, state):
        self.policy.load_state_dict(state["policy"])
        self.opt.load_state_dict(state["optimizer"])
        self.update = state["update"]
        self.agent_steps = state["agent_steps"]
        self.gen.set_state(state["generator"])


def save_checkpoint(path, trainer):
    torch.save(trainer.state_dict(), path)


def load_policy(path, hidden=256):
    state = torch.load(path, map_location="cpu", weights_only=False)
    policy = Policy(hidden)
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy, state
