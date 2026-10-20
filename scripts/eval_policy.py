#!/usr/bin/env python3
"""Measure behaviour of policies on fixed starts. Measurements only: none of these numbers is a reward.

An actor is "random" (uniform over the PPO action heads) or a checkpoint path. Every matchup
plays the same seeded starts, and each side plays as agent 0 and as agent 1 equally often.

    eval_policy.py --server-dir ../teeworlds-0.7.5/build-release \\
        --actors runs/x/ckpt_0000.pt runs/x/ckpt_final.pt random --episodes 16
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import VecTeeworldsEnv  # noqa: E402
from twrl import protocol as P  # noqa: E402
from twrl.ppo import HEADS, flatten_agents, load_policy, to_env_actions  # noqa: E402

V = P.VEC_INDEX
EV = {n: i for i, n in enumerate(P.EVENT_NAMES)}


class Actor:
    def __init__(self, spec, seed):
        self.spec = spec
        self.gen = torch.Generator().manual_seed(seed)
        self.policy = None if spec == "random" else load_policy(spec)[0]

    def env_actions(self, vec, codes):
        if self.policy is None:
            return to_env_actions(torch.stack([torch.randint(0, n, (vec.shape[0],), generator=self.gen) for n in HEADS], dim=-1).numpy())
        with torch.no_grad():
            return self.policy.env_actions(self.policy.act(vec, codes, self.gen)[0].numpy())


def play(server_dir, map_name, actors, episodes, envs, seed, episode_ticks):
    """actors = (actor for agent 0, actor for agent 1). Returns per-agent metric sums and counts."""
    m = {k: np.zeros(2) for k in ("steps", "alive_steps", "kills", "deaths", "suicides", "damage_dealt", "damage_taken",
                                   "shots", "speed", "grounded", "hook", "jump", "fire", "visible", "line_of_fire",
                                   "aim_cos", "opp_dist", "travel")}
    rounds = math.ceil(episodes / envs)
    with VecTeeworldsEnv(envs, server_dir, seed=seed, map=map_name, grid_format="codes", raw_info=True,
                         episode_ticks=episode_ticks) as venv:
        for _ in range(rounds):
            obs = venv.reset()
            prev = None
            while True:
                vec, codes = flatten_agents(obs)
                vec, codes = vec.view(envs, 2, -1), codes.view(envs, 2, P.GRID_HEIGHT, P.GRID_WIDTH)
                actions = np.stack([actors[i].env_actions(vec[:, i], codes[:, i]) for i in range(2)], axis=1)
                obs_before = obs
                obs, _, _, truncated, infos = venv.step(actions)
                ag = infos["agents"]
                ev = infos["events"]
                if truncated.all():
                    ag_now = None
                else:
                    ag_now = ag
                for i in range(2):
                    m["steps"][i] += envs
                    for k in ("kills", "deaths", "suicides", "damage_dealt", "damage_taken"):
                        m[k][i] += ev[:, i, EV[k]].sum()
                    v0 = obs_before["vec"][:, i]
                    alive = v0[:, V["alive"]] > 0
                    m["alive_steps"][i] += alive.sum()
                    m["hook"][i] += (actions[:, i, 3] * alive).sum()
                    m["jump"][i] += (actions[:, i, 1] * alive).sum()
                    m["fire"][i] += (actions[:, i, 2] * alive).sum()
                    m["grounded"][i] += v0[:, V["grounded"]].sum()
                    m["speed"][i] += np.abs(v0[:, V["vel_x"]] * 32).sum()
                    vis = v0[:, V["opp_visible"]] > 0
                    m["visible"][i] += vis.sum()
                    m["line_of_fire"][i] += v0[vis, V["opp_line_of_fire"]].sum()
                    d = np.stack([v0[:, V["opp_dx"]] * P.VIEW_HALF_WIDTH, v0[:, V["opp_dy"]] * P.VIEW_HALF_HEIGHT], 1)
                    dist = np.hypot(d[:, 0], d[:, 1])
                    m["opp_dist"][i] += dist[vis].sum()
                    aim = actions[:, i, 4:6].astype(np.float64)
                    aim /= np.maximum(np.hypot(aim[:, 0], aim[:, 1]), 1e-6)[:, None]
                    cos = (aim * d).sum(1) / np.maximum(dist, 1e-6)
                    m["aim_cos"][i] += cos[vis].sum()
                    if ag_now is not None and prev is not None:
                        both = (prev["alive"][:, i] == 1) & (ag_now["alive"][:, i] == 1)
                        m["shots"][i] += ((prev["ammo"][:, i] - ag_now["ammo"][:, i] > 0) & both).sum()
                        step = np.hypot(*(ag_now["pos"][:, i] - prev["pos"][:, i]).T)
                        m["travel"][i] += step[both & (step < 200)].sum()
                prev = ag_now
                if truncated.all():
                    break
    return m


EPISODE_STEPS = 1500  # set from --episode-ticks in main


def summarize(m, i):
    ep = m["steps"][i] / EPISODE_STEPS if m["steps"][i] else 1
    alive = max(m["alive_steps"][i], 1)
    vis = max(m["visible"][i], 1)
    return {
        "episodes": round(ep, 1),
        "kills_per_ep": m["kills"][i] / ep, "deaths_per_ep": m["deaths"][i] / ep, "suicides_per_ep": m["suicides"][i] / ep,
        "damage_dealt_per_ep": m["damage_dealt"][i] / ep, "damage_taken_per_ep": m["damage_taken"][i] / ep,
        "shots_per_ep": m["shots"][i] / ep, "hit_rate": m["damage_dealt"][i] / max(m["shots"][i], 1),
        "mean_speed": m["speed"][i] / alive, "travel_per_ep": m["travel"][i] / ep,
        "grounded_frac": m["grounded"][i] / alive, "hook_frac": m["hook"][i] / alive, "jump_frac": m["jump"][i] / alive,
        "fire_frac": m["fire"][i] / alive, "opp_visible_frac": m["visible"][i] / alive,
        "line_of_fire_when_visible": m["line_of_fire"][i] / vis, "aim_cos_when_visible": m["aim_cos"][i] / vis,
        "opp_dist_when_visible": m["opp_dist"][i] / vis,
    }


def merge(a, b):
    """Metrics of one actor that played agent 0 in run a and agent 1 in run b."""
    return {k: np.array([a[k][0] + b[k][1]]) for k in a}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--actors", nargs="+", required=True)
    ap.add_argument("--map", default="lms1")
    ap.add_argument("--episodes", type=int, default=16, help="per matchup and side")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--episode-ticks", type=int, default=3000)
    ap.add_argument("--head-to-head", nargs=2, metavar=("A", "B"), action="append", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    global EPISODE_STEPS
    EPISODE_STEPS = a.episode_ticks // 2  # action_ticks is 2
    torch.set_num_threads(4)

    def run(x, y):
        return play(a.server_dir, a.map, (Actor(x, a.seed), Actor(y, a.seed + 1)), a.episodes, a.envs, a.seed, a.episode_ticks)

    report = {"self_play": {}, "head_to_head": []}
    for spec in a.actors:
        m = run(spec, spec)
        both = {k: np.array([m[k].sum()]) for k in m}
        report["self_play"][spec] = summarize(both, 0)
        print(spec, json.dumps({k: round(float(v), 3) for k, v in report["self_play"][spec].items()}), flush=True)
    for x, y in a.head_to_head:
        ab, ba = run(x, y), run(y, x)
        sx, sy = summarize(merge(ab, ba), 0), summarize(merge(ba, ab), 0)
        row = {"a": x, "b": y, "a_kills_per_ep": sx["kills_per_ep"], "b_kills_per_ep": sy["kills_per_ep"],
               "a_damage_dealt_per_ep": sx["damage_dealt_per_ep"], "b_damage_dealt_per_ep": sy["damage_dealt_per_ep"],
               "a_suicides_per_ep": sx["suicides_per_ep"], "b_suicides_per_ep": sy["suicides_per_ep"], "episodes": sx["episodes"]}
        report["head_to_head"].append(row)
        print("h2h", json.dumps({k: (round(float(v), 3) if not isinstance(v, str) else v) for k, v in row.items()}), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, default=float)


if __name__ == "__main__":
    main()
