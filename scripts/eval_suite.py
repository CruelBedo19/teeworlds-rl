#!/usr/bin/env python3
"""Run the fixed evaluation protocol (twrl/evaluation.py) for one checkpoint of a run.

    eval_suite.py --server-dir ../teeworlds-0.7.5/build-release --run-dir runs/x --checkpoint ckpt_0400.pt

Compares against uniform random, the run's ckpt_0000.pt and the checkpoint before this one, each
with sides swapped, and writes runs/x/eval/<checkpoint>.json. --previous overrides the automatic
choice of the previous checkpoint.
"""

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl.evaluation import evaluate, round_floats  # noqa: E402


def update_of(path):
    return torch.load(path, map_location="cpu", weights_only=False)["update"]


def previous_checkpoint(run_dir, current):
    """The latest numbered checkpoint from an earlier update than `current` (ckpt_final may equal the last one)."""
    target = update_of(current)
    earlier = [p for p in glob.glob(os.path.join(run_dir, "ckpt_[0-9][0-9][0-9][0-9].pt")) if 0 < update_of(p) < target]
    return max(earlier, key=update_of) if earlier else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--previous")
    a = ap.parse_args()
    current = os.path.join(a.run_dir, a.checkpoint)
    initial = os.path.join(a.run_dir, "ckpt_0000.pt")
    previous = a.previous or previous_checkpoint(a.run_dir, current)
    if previous and os.path.abspath(previous) in (os.path.abspath(current), os.path.abspath(initial)):
        previous = None
    report = round_floats(evaluate(a.server_dir, current, initial if os.path.abspath(initial) != os.path.abspath(current) else None, previous))
    out_dir = os.path.join(a.run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.splitext(a.checkpoint)[0] + ".json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    s = report["self"]
    print(f"self-play: kills {s['kills']} damage {s['damage_dealt']} hit_rate {s['hit_rate']} aim_cos {s['aim_cos_when_visible']}")
    for name, r in report["vs"].items():
        c, o = r["current"], r["opponent"]
        print(f"vs {name:8s}: win {c['win_rate']} / {o['win_rate']} draw {c['draw_rate']}  kills {c['kills']} / {o['kills']}"
              f"  damage {c['damage_dealt']} / {o['damage_dealt']}  reward {c['reward']} / {o['reward']}")
    print("written to", out)


if __name__ == "__main__":
    main()
