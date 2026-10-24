#!/usr/bin/env python3
"""Side-by-side fire / aim diagnostics (bench/diagnose.py output) and training statistics for A/B runs.

    fire_ab_table.py --diag control.json:ckpt_0245.pt:runs/x/metrics.jsonl exp.json:ckpt_0245.pt:runs/y/metrics.jsonl ...
"""
import argparse
import json

import numpy as np

KEYS = ("p_fire_given_los_ready", "p_fire_given_invisible_ready", "p_fire_given_visible", "fire_frac", "gun_empty_frac",
        "weapon_ready_frac", "line_of_fire_frac", "combat_opportunity_frac", "shots_per_episode", "dry_fires_per_episode",
        "wasted_fire_share", "hit_rate", "hit_decisions_per_episode", "damage_dealt_per_episode", "damage_taken_per_episode",
        "kills_per_episode", "deaths_per_episode", "aim_cos_visible", "visible_frac")
TRAIN = ("approx_kl", "aim_approx_kl", "aim_off", "log_std_x", "aim_cos", "entropy_fire", "p_fire_los_ready",
         "p_fire_invisible_ready", "p_fire_empty", "grad_norm_max", "dry_fires", "shots")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", nargs="+", required=True, help="diag.json:checkpoint-suffix[:metrics.jsonl[:first-update:last-update]]")
    a = ap.parse_args()
    cols = []
    for spec in a.diag:
        parts = spec.split(":")
        d = json.load(open(parts[0]))
        ck = next(k for k in d["checkpoints"] if k.endswith(parts[1]))
        x = dict(d["checkpoints"][ck])
        x["damage_ratio"] = x["damage_dealt_per_episode"] / max(x["damage_taken_per_episode"], 1e-9)
        if len(parts) > 2:
            rows = [json.loads(l) for l in open(parts[2]) if '"update"' in l]
            lo, hi = (int(parts[3]), int(parts[4])) if len(parts) > 4 else (len(rows) * 3 // 4, len(rows))
            w = [r for r in rows if lo < r["update"] <= hi]
            for k in TRAIN:
                v = [r[k] for r in w if k in r and r[k] is not None]
                x["train_" + k] = float(np.mean(v)) if v else float("nan")
            x["train_nonfinite"] = sum(1 for r in rows for v in r.values() if isinstance(v, float) and not np.isfinite(v))
        cols.append((parts[0].split("/")[-1].replace(".json", "") + ":" + parts[1], x))
    keys = KEYS + ("damage_ratio",) + tuple("train_" + k for k in TRAIN) + ("train_nonfinite",)
    print(f"{'':32}" + "".join(f"{c[0][-28:]:>30}" for c in cols))
    for k in keys:
        print(f"{k:32}" + "".join(f"{c[1].get(k, float('nan')):30.4f}" for c in cols))


if __name__ == "__main__":
    main()
