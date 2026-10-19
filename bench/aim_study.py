#!/usr/bin/env python3
"""Compare aim representations for PPO with the real gun geometry of Teeworlds 0.7.5.

1. Situations: relative opponent positions seen in real rollouts (random actions, lms1), only
   where the opponent is in view and nothing solid is in between.
2. Coverage: for each representation, could ANY of its aim choices hit a static target there?
   Uses the gun's own path: start 21 units out (PHYS_SIZE*0.75), speed 2200, curvature 1.25
   (CalcPos), and a hit within 34 units of the centre (proximity 28 + radius 6).
3. Learning: a contextual bandit with exactly that hit test as reward (1 hit, 0 miss), trained with
   the same clipped policy-gradient update as PPO, for each representation and three seeds.

This isolates the aim head; it says nothing about learning the rest of the game.

    aim_study.py --server-dir ../teeworlds-0.7.5/build-release --out aim.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import VecTeeworldsEnv, random_actions  # noqa: E402
from twrl import protocol as P  # noqa: E402

V = P.VEC_INDEX
SPEED, CURVATURE, LIFETIME, START, HIT = 2200.0, 1.25, 2.0, 28 * 0.75, 28 + 6


def situations(server_dir, count, seed):
    """(dx, dy) of the opponent in world units where it is visible with a clear line of fire."""
    rng = np.random.default_rng(seed)
    out = []
    with VecTeeworldsEnv(8, server_dir, seed=seed, map="lms1", grid_format="codes") as venv:
        obs = venv.reset()
        while len(out) < count:
            v = obs["vec"].reshape(-1, P.VEC_DIM)
            ok = (v[:, V["opp_visible"]] > 0) & (v[:, V["opp_line_of_fire"]] > 0)
            for row in v[ok]:
                out.append((row[V["opp_dx"]] * P.VIEW_HALF_WIDTH, row[V["opp_dy"]] * P.VIEW_HALF_HEIGHT))
            obs = venv.step(random_actions(rng, (8, 2)))[0]
    return np.array(out[:count], dtype=np.float64)


def miss_distance(target, angles):
    """Closest approach of a gun projectile fired at `angles` (radians) to a static target. Vectorised."""
    t = np.linspace(0.0, LIFETIME, 801)[None, None, :] * SPEED  # path length
    dx, dy = np.cos(angles)[..., None], np.sin(angles)[..., None]
    px = START * dx + dx * t
    py = START * dy + dy * t + CURVATURE / 10000 * t * t
    d = np.hypot(px - target[:, 0, None, None], py - target[:, 1, None, None])
    return d.min(axis=-1)


def coverage(targets, bins_list):
    res = {}
    for bins in bins_list:
        angles = np.tile(2 * np.pi * np.arange(bins) / bins, (len(targets), 1))
        best = np.concatenate([miss_distance(targets[i:i + 500], angles[i:i + 500]).min(axis=1) for i in range(0, len(targets), 500)])
        res[f"categorical_{bins}"] = float((best < HIT).mean())
    fine = np.tile(np.linspace(-np.pi, np.pi, 1441), (len(targets), 1))  # 0.25 degrees, as precise as the client's aim
    best = np.concatenate([miss_distance(targets[i:i + 200], fine[i:i + 200]).min(axis=1) for i in range(0, len(targets), 200)])
    res["continuous"] = float((best < HIT).mean())
    return res


class Bandit(nn.Module):
    def __init__(self, kind, bins=16):
        super().__init__()
        self.kind, self.bins = kind, bins
        out = bins if kind == "categorical" else 2
        self.body = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Linear(64, out)
        nn.init.orthogonal_(self.head.weight, 0.01)
        nn.init.zeros_(self.head.bias)
        if kind != "categorical":
            self.log_std = nn.Parameter(torch.full((out if kind == "gauss_vector" else 1,), -0.5))

    def dist(self, x):
        h = self.head(self.body(x))
        if self.kind == "categorical":
            return torch.distributions.Categorical(logits=h)
        if self.kind == "gauss_vector":  # 2D mean, aim along the sampled vector
            return torch.distributions.Independent(torch.distributions.Normal(h, self.log_std.exp()), 1)
        mean = torch.atan2(h[:, 1], h[:, 0] + 1e-6)  # gauss_angle: angle from a 2D output
        return torch.distributions.Normal(mean, self.log_std.exp())

    def angle(self, action):
        if self.kind == "categorical":
            return action.double() * 2 * math.pi / self.bins
        if self.kind == "gauss_vector":
            return torch.atan2(action[:, 1], action[:, 0]).double()
        return action.double()


def learn(targets, kind, bins, seed, samples, batch=512, epochs=4, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Bandit(kind, bins)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    seen = 0
    while seen < samples:
        idx = rng.integers(0, len(targets), batch)
        tgt = targets[idx]
        x = torch.tensor(tgt / [P.VIEW_HALF_WIDTH, P.VIEW_HALF_HEIGHT], dtype=torch.float32)
        with torch.no_grad():
            d = model.dist(x)
            a = d.sample()
            old_logp = d.log_prob(a)
        hit = miss_distance(tgt, model.angle(a).numpy()[:, None])[:, 0] < HIT
        r = torch.tensor(hit, dtype=torch.float32)
        adv = (r - r.mean()) / (r.std() + 1e-8)
        for _ in range(epochs):
            ratio = torch.exp(model.dist(x).log_prob(a) - old_logp)
            loss = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        seen += batch
        curve.append((seen, float(hit.mean())))
    # final stochastic hit rate on fresh situations
    idx = rng.integers(0, len(targets), 4096)
    with torch.no_grad():
        x = torch.tensor(targets[idx] / [P.VIEW_HALF_WIDTH, P.VIEW_HALF_HEIGHT], dtype=torch.float32)
        final = float((miss_distance(targets[idx], model.angle(model.dist(x).sample()).numpy()[:, None])[:, 0] < HIT).mean())
    return curve, final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--situations", type=int, default=8000)
    ap.add_argument("--samples", type=int, default=150000, help="bandit training samples per run")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(6)

    targets = situations(a.server_dir, a.situations, 7)
    dist = np.hypot(targets[:, 0], targets[:, 1])
    report = {"situations": len(targets), "distance_percentiles": {p: float(np.percentile(dist, p)) for p in (10, 50, 90)}}
    report["coverage"] = coverage(targets, (8, 16, 32, 64))
    by_dist = {}
    for lo, hi in ((0, 300), (300, 600), (600, 1100)):
        sel = targets[(dist >= lo) & (dist < hi)]
        if len(sel):
            by_dist[f"{lo}-{hi}"] = {"n": len(sel), **coverage(sel[:2000], (16, 32, 64))}
    report["coverage_by_distance"] = by_dist
    print(json.dumps({k: report[k] for k in ("situations", "distance_percentiles", "coverage", "coverage_by_distance")}, indent=1), flush=True)

    learning = {}
    for kind, bins in (("categorical", 16), ("categorical", 32), ("categorical", 64), ("gauss_vector", 0), ("gauss_angle", 0)):
        name = f"{kind}_{bins}" if bins else kind
        runs = [learn(targets, kind, bins, s, a.samples) for s in range(a.seeds)]
        at = lambda n: float(np.mean([np.mean([h for seen, h in c if n - 10000 < seen <= n]) for c, _ in runs]))  # noqa: E731
        learning[name] = {"hit_at_25k": at(25000), "hit_at_75k": at(75000), "hit_at_end": at(a.samples),
                          "final_stochastic_hit": float(np.mean([f for _, f in runs])),
                          "final_spread_over_seeds": float(np.ptp([f for _, f in runs]))}
        print(name, json.dumps({k: round(v, 3) for k, v in learning[name].items()}), flush=True)
    report["bandit_learning"] = learning
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
