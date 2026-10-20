#!/usr/bin/env python3
"""Behaviour diagnostics of checkpoints against a frozen opponent (measurement only, nothing trains).

    diagnose.py --server-dir ../teeworlds-0.7.5/build-release --opponent runs/x/ckpt_0000.pt \
        --checkpoints runs/x/ckpt_0000.pt runs/x/ckpt_0123.pt ... --out diag.json

Plays the fixed evaluation seeds (twrl/evaluation.py) with each checkpoint on both sides against the
opponent and counts, over the checkpoint's own decisions while alive:
  fire / ammo   fire presses, presses with the gun ready (ammo > 0, no reload), presses on an empty gun
                or during reload, shots; how often the gun is empty; whether the policy fires more when the
                opponent is in the line of fire than when it is not visible
  sparsity      opponent visible, open line of fire, a combat opportunity (visible, line of fire, gun
                ready), decisions with any reward event, damage and kills
  aim           aim cosine to the opponent on all visible decisions, on shots with a line of fire
  hook          hook presses and new hook launches; how many launches fall on decisions without a shot
                (with the shot-masked aim loss the aim gets no gradient from those)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import protocol as P  # noqa: E402
from twrl.env import VecTeeworldsEnv  # noqa: E402
from twrl.evaluation import EVAL_ENVS, EVAL_EPISODE_TICKS, EVAL_MAP, EVAL_ROUNDS, EVAL_SEED, Actor, round_floats  # noqa: E402

V = P.VEC_INDEX
EV = {n: i for i, n in enumerate(P.EVENT_NAMES)}
REWARD = np.array([1.0, -1.0, -2.0, 0.02, -0.02])

KEYS = ("decisions", "visible", "los", "opportunity", "fire", "fire_ready", "fire_empty", "fire_reload", "shots", "empty",
        "fire_when_los_ready", "fire_when_invisible_ready", "los_ready", "invisible_ready", "reward_events", "damage_dealt",
        "damage_taken", "kills", "deaths", "suicides", "aim_cos_visible", "aim_cos_los_shot", "los_shots", "hook", "hook_launch",
        "hook_launch_no_shot", "hook_launch_visible", "hook_cos_launch_visible", "episodes")


def play(server_dir, specs, learner):
    """One pass over the evaluation seeds; counts for agent `learner`."""
    actors = (Actor(specs[0], 11), Actor(specs[1], 12))
    m = dict.fromkeys(KEYS, 0.0)
    with VecTeeworldsEnv(EVAL_ENVS, server_dir, seed=EVAL_SEED, map=EVAL_MAP, grid_format="codes", raw_info=True,
                         episode_ticks=EVAL_EPISODE_TICKS) as venv:
        for _ in range(EVAL_ROUNDS):
            obs = venv.reset()
            prev_hook = np.zeros(EVAL_ENVS, dtype=bool)
            ammo = None
            while True:
                vec, codes = torch.from_numpy(obs["vec"]), torch.from_numpy(obs["grid"])
                actions = np.stack([actors[i](vec[:, i], codes[:, i]) for i in range(2)], axis=1)
                v = obs["vec"][:, learner]
                if ammo is None:
                    ammo = venv._agents["ammo"][:, learner].copy() if hasattr(venv, "_agents") else None
                obs, _, _, truncated, infos = venv.step(actions)
                ev = infos["events"][:, learner]
                ag = infos["agents"]
                a = actions[:, learner]
                alive = v[:, V["alive"]] > 0
                vis = (v[:, V["opp_visible"]] > 0) & alive
                los = vis & (v[:, V["opp_line_of_fire"]] > 0)
                ready = alive & (ammo > 0) & (v[:, V["reload"]] <= 0)
                fire = alive & (a[:, 2] == 1)
                shot = ev[:, EV["shots"]] > 0
                hook = alive & (a[:, 3] == 1)
                launch = hook & ~prev_hook
                d = np.stack([v[:, V["opp_dx"]] * P.VIEW_HALF_WIDTH, v[:, V["opp_dy"]] * P.VIEW_HALF_HEIGHT], 1)
                aim = a[:, 4:6].astype(np.float64)
                cos = (aim * d).sum(1) / np.maximum(np.hypot(*aim.T) * np.hypot(*d.T), 1e-6)
                m["decisions"] += alive.sum()
                m["visible"] += vis.sum()
                m["los"] += los.sum()
                m["opportunity"] += (los & ready).sum()
                m["fire"] += fire.sum()
                m["fire_ready"] += (fire & ready).sum()
                m["fire_empty"] += (fire & alive & (ammo <= 0)).sum()
                m["fire_reload"] += (fire & (ammo > 0) & (v[:, V["reload"]] > 0)).sum()
                m["shots"] += (shot & alive).sum()
                m["empty"] += (alive & (ammo <= 0)).sum()
                m["los_ready"] += (los & ready).sum()
                m["fire_when_los_ready"] += (los & ready & fire).sum()
                m["invisible_ready"] += (~vis & ready).sum()
                m["fire_when_invisible_ready"] += (~vis & ready & fire).sum()
                m["reward_events"] += (alive & (np.abs(ev[:, :5]) @ np.abs(REWARD) > 0)).sum()
                for k in ("damage_dealt", "damage_taken", "kills", "deaths", "suicides"):
                    m[k] += ev[:, EV[k]].sum()
                m["aim_cos_visible"] += cos[vis].sum()
                m["aim_cos_los_shot"] += cos[los & shot].sum()
                m["los_shots"] += (los & shot).sum()
                m["hook"] += hook.sum()
                m["hook_launch"] += launch.sum()
                m["hook_launch_no_shot"] += (launch & ~shot).sum()
                m["hook_launch_visible"] += (launch & vis).sum()
                m["hook_cos_launch_visible"] += cos[launch & vis].sum()
                prev_hook = hook
                ammo = ag["ammo"][:, learner].copy()
                if truncated.all():
                    m["episodes"] += EVAL_ENVS
                    break
    return m


def report(m):
    n = max(m["decisions"], 1)
    r = lambda a, b: m[a] / max(m[b], 1)  # noqa: E731
    return {
        "episodes": m["episodes"], "alive_decisions": m["decisions"],
        "visible_frac": m["visible"] / n, "line_of_fire_frac": m["los"] / n, "combat_opportunity_frac": m["opportunity"] / n,
        "reward_event_frac": m["reward_events"] / n, "damage_per_1000": 1000 * m["damage_dealt"] / n,
        "kills_per_episode": r("kills", "episodes"), "deaths_per_episode": r("deaths", "episodes"),
        "suicides_per_episode": r("suicides", "episodes"),
        "damage_dealt_per_episode": r("damage_dealt", "episodes"), "damage_taken_per_episode": r("damage_taken", "episodes"),
        "fire_frac": m["fire"] / n, "shot_frac": m["shots"] / n, "actual_shot_rate": r("shots", "fire"),
        "fire_ready_share": r("fire_ready", "fire"), "fire_empty_share": r("fire_empty", "fire"),
        "fire_reload_share": r("fire_reload", "fire"), "gun_empty_frac": m["empty"] / n,
        "p_fire_given_los_ready": r("fire_when_los_ready", "los_ready"),
        "p_fire_given_invisible_ready": r("fire_when_invisible_ready", "invisible_ready"),
        "aim_cos_visible": r("aim_cos_visible", "visible"), "aim_cos_los_shots": r("aim_cos_los_shot", "los_shots"),
        "hit_rate": r("damage_dealt", "shots"),
        "hook_frac": m["hook"] / n, "hook_launches_per_episode": r("hook_launch", "episodes"),
        "hook_launch_without_shot_share": r("hook_launch_no_shot", "hook_launch"),
        "hook_launch_visible_share": r("hook_launch_visible", "hook_launch"),
        "hook_aim_cos_on_visible_launches": r("hook_cos_launch_visible", "hook_launch_visible"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    out = {"opponent": a.opponent, "protocol": {"seed": EVAL_SEED, "envs": EVAL_ENVS, "rounds": EVAL_ROUNDS, "sides": 2}, "checkpoints": {}}
    for c in a.checkpoints:
        m0 = play(a.server_dir, (c, a.opponent), 0)
        m1 = play(a.server_dir, (a.opponent, c), 1)
        rep = report({k: m0[k] + m1[k] for k in KEYS})
        out["checkpoints"][c] = round_floats(rep)
        print(c, json.dumps(out["checkpoints"][c]), flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
