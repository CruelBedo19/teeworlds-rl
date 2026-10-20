#!/usr/bin/env python3
"""Larger evaluation of one checkpoint: the fixed protocol repeated over several evaluation seeds.

    strong_eval.py --server-dir ../teeworlds-0.7.5/build-release --checkpoint runs/x/ckpt_1221.pt \
        --opponents runs/x/ckpt_0000.pt random runs/x/ckpt_0245.pt --seeds 8 --out strong.json

Each seed is an independent set of EVAL_ENVS x EVAL_ROUNDS starts played on both sides (64 episodes), so
--seeds 8 is 512 episodes per opponent. Per opponent it reports pooled numbers and, over the seeds,
the mean and standard error of the reward and of the damage difference, which says whether the
result holds beyond one set of starts.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import twrl.evaluation as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponents", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--first-seed", type=int, default=9101)  # not the 9001 of the standard protocol
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(E.EVAL_THREADS)
    out = {"checkpoint": a.checkpoint, "seeds": list(range(a.first_seed, a.first_seed + a.seeds)), "opponents": {}}
    keys = ("episodes", "kills", "deaths", "suicides", "damage_dealt", "damage_taken", "shots", "hit_rate", "reward", "win_rate",
            "draw_rate", "aim_cos_when_visible", "opp_visible_frac")
    for opp in a.opponents:
        per_seed = []
        for s in out["seeds"]:
            E.EVAL_SEED = s
            me, them = E.both_sides(a.server_dir, a.checkpoint, opp)
            per_seed.append({"seed": s, "current": me, "opponent": them})
            print(opp, s, json.dumps(E.round_floats(me)), flush=True)
        n = np.array([p["current"]["episodes"] for p in per_seed])
        pooled = {k: float(np.average([p["current"][k] for p in per_seed], weights=n)) for k in keys if k != "episodes"}
        pooled["episodes"] = int(n.sum())
        pooled["opp_win_rate"] = float(np.average([p["opponent"]["win_rate"] for p in per_seed], weights=n))
        pooled["opp_hit_rate"] = float(np.average([p["opponent"]["hit_rate"] for p in per_seed], weights=n))
        pooled["damage_ratio"] = pooled["damage_dealt"] / max(pooled["damage_taken"], 1e-9)
        pooled["wins"] = int(round(pooled["win_rate"] * n.sum()))
        pooled["losses"] = int(round(pooled["opp_win_rate"] * n.sum()))
        rew = np.array([p["current"]["reward"] for p in per_seed])
        diff = np.array([p["current"]["damage_dealt"] - p["current"]["damage_taken"] for p in per_seed])
        se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")  # noqa: E731
        spread = {"reward_mean": float(rew.mean()), "reward_se": se(rew), "damage_diff_mean": float(diff.mean()), "damage_diff_se": se(diff),
                  "seeds_with_positive_reward": int((rew > 0).sum()), "seeds_with_ratio_ge_1": int((diff >= 0).sum())}
        out["opponents"][opp] = E.round_floats({"pooled": pooled, "over_seeds": spread, "per_seed": per_seed})
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
