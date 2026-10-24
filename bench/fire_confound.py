#!/usr/bin/env python3
"""Is the empty-gun press/no-press advantage gap causal or borrowed from nearby hits? (measurement only)

    fire_confound.py --server-dir ../teeworlds-0.7.5/build-v4 --run-dir runs/x --checkpoint ckpt_1221.pt --batches 16

On an empty gun a press can do nothing but a dry fire, which restarts the 500 ms ammo refill. Its advantage
should then be slightly negative. GAE also gives every decision a share of the rewards that follow within
its horizon, so presses that sit next to a hit (a volley that emptied the gun, then more presses while the
bullet flies) borrow that hit's reward. Here the empty-gun decisions are split by whether damage is dealt or
taken within --window decisions after them, and the press/no-press gap is given within each part. Also: the
decisions from an empty gun to its next round, for refills after a run with and without dry fires.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aim_probe import Recorder  # noqa: E402
from twrl import protocol as P  # noqa: E402
from twrl.env import VecTeeworldsEnv  # noqa: E402
from twrl.ppo import PPOConfig, Trainer, load_policy  # noqa: E402

V, EV = P.VEC_INDEX, {n: i for i, n in enumerate(P.EVENT_NAMES)}


def gap(adv, press, m):
    a, b = adv[m & press], adv[m & ~press]
    if len(a) < 2 or len(b) < 2:
        return None
    g = a.mean() - b.mean()
    return {"gap": float(g), "t": float(g / math.sqrt(a.var() / len(a) + b.var() / len(b))), "n_press": int(len(a)), "n_not": int(len(b))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(3)
    path = os.path.join(a.run_dir, a.checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=False)
    policy, _ = load_policy(path)
    opponent, _ = load_policy(os.path.join(a.run_dir, "ckpt_0000.pt"))
    cfg = PPOConfig(**state["config"])
    ADV, PRESS, NEAR, EMPTY = [], [], [], []
    waits = {"with_dry": [], "without_dry": []}
    with VecTeeworldsEnv(cfg.num_envs, a.server_dir, seed=a.seed, map="lms1", grid_format="codes", raw_info=True, timeout=600) as venv:
        rec = Recorder(venv)
        trainer = Trainer(rec, policy, cfg, seed=a.seed, opponent=opponent)
        for _ in range(a.batches):
            rec.events = []
            batch = trainer.collect()
            T = len(rec.events)
            ev = np.stack(rec.events).reshape(T, -1, len(P.EVENT_NAMES))[:, trainer.rows]
            raw = batch["adv"]
            adv = ((raw - raw.mean()) / (raw.std() + 1e-8)).numpy().reshape(T, -1)
            vec = batch["vec"].numpy().reshape(T, -1, batch["vec"].shape[-1])
            press = batch["actions"][:, 2].numpy().reshape(T, -1) == 1
            alive = vec[..., V["alive"]] > 0
            empty = alive & (vec[..., V["ammo"]] <= 0)
            dmg = (ev[..., EV["damage_dealt"]] > 0) | (ev[..., EV["damage_taken"]] > 0)
            near = np.zeros_like(dmg)
            for t in range(T):
                near[t] = dmg[t:t + a.window + 1].any(0)
            keep = np.arange(T) < T - a.window
            ADV.append(adv[keep]); PRESS.append(press[keep]); NEAR.append(near[keep]); EMPTY.append(empty[keep])
            # runs of an empty gun ending in a round: length, and whether a dry fire happened in the run
            dry = ev[..., EV["dry_fires"]] > 0
            for j in range(empty.shape[1]):
                start, had_dry = None, False
                for t in range(T):
                    if empty[t, j] and start is None:
                        start, had_dry = t, False
                    if start is not None:
                        had_dry |= bool(dry[t, j])
                        if not empty[t, j]:
                            if alive[t, j] and start > 0:
                                waits["with_dry" if had_dry else "without_dry"].append(t - start)
                            start = None
    adv, press, near, empty = (np.concatenate(x).reshape(-1) for x in (ADV, PRESS, NEAR, EMPTY))
    out = {"checkpoint": path, "window": a.window, "batches": a.batches,
           "empty_all": gap(adv, press, empty), "empty_damage_near": gap(adv, press, empty & near),
           "empty_quiet": gap(adv, press, empty & ~near),
           "press_share_near": float(press[empty & near].mean()), "press_share_quiet": float(press[empty & ~near].mean()),
           "adv_near_minus_quiet": float(adv[empty & near].mean() - adv[empty & ~near].mean()),
           "share_near": float(near[empty].mean()),
           "empty_run_to_round": {k: {"n": len(v), "mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None}
                                  for k, v in waits.items()}}
    print(json.dumps(out, indent=1))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
