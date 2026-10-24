#!/usr/bin/env python3
"""How much of the aim's advantage comes from the dry-fire penalty (measurement only).

    adv_decompose.py --server-dir ../teeworlds-0.7.5/build-v4 --run-dir runs/x --checkpoint ckpt_0489.pt \
        --dry-fire-penalty 0.0005 --batches 16 --out d.json

GAE is linear in the rewards, so with the run's values the advantage splits exactly into the part from the
dry-fire penalty, sum_k (gamma lambda)^k (prod of nonterminal) * penalty_{t+k}, and the rest (game reward and
value terms). On the shot decisions, the ones the aim learns from, this gives each part's spread and, per
batch, each part's push on the aim mean towards the opponent (as in aim_consistency.py: adv x the rotational
part of d logp / d mu on visible shot decisions). If the penalty part is noise for the aim, the push from the
rest has a better signal-to-noise ratio than the push from the whole advantage.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aim_consistency import opponent_dir  # noqa: E402
from twrl import protocol as P  # noqa: E402
from twrl.env import DEFAULT_REWARD_WEIGHTS, VecTeeworldsEnv  # noqa: E402
from twrl.ppo import PPOConfig, Trainer, load_policy  # noqa: E402

V, EV = P.VEC_INDEX, {n: i for i, n in enumerate(P.EVENT_NAMES)}


class Recorder:
    def __init__(self, venv):
        self._venv, self.events, self.ended = venv, [], []

    def step(self, actions):
        out = self._venv.step(actions)
        self.events.append(out[4]["events"].copy())
        self.ended.append(np.logical_or(out[2], out[3]).copy())
        return out

    def __getattr__(self, name):
        return getattr(self._venv, name)


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(ddof=1)
    return {"mean": float(x.mean()), "sd": float(sd), "snr_per_batch": float(x.mean() / sd) if sd > 0 else None,
            "share_positive": float(np.mean(x > 0))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dry-fire-penalty", type=float, required=True)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(3)
    path = os.path.join(a.run_dir, a.checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=False)
    policy, _ = load_policy(path)
    opponent, _ = load_policy(os.path.join(a.run_dir, "ckpt_0000.pt"))
    cfg = PPOConfig(**state["config"])
    weights = {**DEFAULT_REWARD_WEIGHTS, "dry_fires": -a.dry_fire_penalty}
    push = {"total": [], "penalty": [], "rest": []}
    spread = {"total": [], "penalty": [], "rest": []}
    with VecTeeworldsEnv(cfg.num_envs, a.server_dir, seed=a.seed, map="lms1", grid_format="codes", raw_info=True,
                         timeout=600, reward_weights=weights) as venv:
        rec = Recorder(venv)
        trainer = Trainer(rec, policy, cfg, seed=a.seed, opponent=opponent)
        for _ in range(a.batches):
            rec.events, rec.ended = [], []
            batch = trainer.collect()
            T = len(rec.events)
            ev = np.stack(rec.events).reshape(T, -1, len(P.EVENT_NAMES))[:, trainer.rows]
            done = np.stack([np.repeat(e, 2)[trainer.rows] for e in rec.ended]).astype(np.float64)
            pen = -a.dry_fire_penalty * ev[..., EV["dry_fires"]].astype(np.float64)
            adv_pen = np.zeros_like(pen)
            last = np.zeros(pen.shape[1])
            for t in reversed(range(T)):
                last = pen[t] + cfg.gamma * cfg.lam * (1.0 - done[t]) * last
                adv_pen[t] = last
            adv = batch["adv"].numpy().astype(np.float64).reshape(T, -1)
            parts = {"total": adv, "penalty": adv_pen, "rest": adv - adv_pen}
            sd_all = adv.std()
            shot = ev[..., EV["shots"]] > 0
            vec = batch["vec"]
            with torch.no_grad():
                mu = trainer.policy.aim_mean(vec, batch["codes"]).double()
                std = trainer.policy._std().double()
            x = batch["actions"][:, -2:].double()
            u = opponent_dir(vec).double()
            mhat = mu / mu.norm(dim=1, keepdim=True).clamp(min=1e-6)
            rot = u - (u * mhat).sum(1, keepdim=True) * mhat
            g = (((x - mu) / std ** 2) * rot).sum(1).numpy().reshape(T, -1)
            vis = (vec[:, V["alive"]] > 0).numpy().reshape(T, -1) & (vec[:, V["opp_visible"]] > 0).numpy().reshape(T, -1)
            m = shot & vis
            for k, v in parts.items():
                # normalised as in train(): by the whole advantage's spread over the batch
                push[k].append(float((v[m] / sd_all * g[m]).sum() / v.size))
                spread[k].append(float(v[shot].std()))
            print(json.dumps({k: round(push[k][-1], 5) for k in push}), flush=True)
    out = {"checkpoint": path, "penalty": a.dry_fire_penalty, "batches": a.batches,
           "push_on_visible_shots": {k: stats(v) for k, v in push.items()},
           "adv_sd_on_shots": {k: float(np.mean(v)) for k, v in spread.items()}}
    print(json.dumps(out, indent=1))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
