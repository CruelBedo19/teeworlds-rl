#!/usr/bin/env python3
"""Behaviour diagnostics of checkpoints against a frozen opponent (measurement only, nothing trains).

    diagnose.py --server-dir ../teeworlds-0.7.5/build-release --opponent runs/x/ckpt_0000.pt \
        --checkpoints runs/x/ckpt_0000.pt runs/x/ckpt_0123.pt ... --out diag.json

Plays the fixed evaluation seeds (twrl/evaluation.py) with each checkpoint on both sides against the
opponent and counts, over the checkpoint's own decisions while alive:
  fire / ammo   the weapon state at each decision (ready: ammo and no reload timer; reload timer running
                with or without ammo; empty with the timer run out) and what a press does in each: a shot,
                a dry fire (counted by the server: an empty gun, which restarts the 125 ms delay and with it
                the 500 ms ammo regeneration) or nothing (the timer is running); ammo refills, how long the
                gun stays empty; whether the policy fires more when the opponent is in the line of fire
                than when it is not visible
  sparsity      opponent visible, open line of fire, a combat opportunity (visible, line of fire, gun
                ready), decisions with any reward event, damage and kills
  aim           aim cosine to the opponent on all visible decisions, on shots with a line of fire
  hook          hook presses and hook launches (counted by the server); how many launches fall on decisions
                without a shot (with the shot-masked aim loss the aim gets no gradient from those)

--mask-empty-fire turns the checkpoint's fire off whenever its gun is empty at the decision (the opponent
is untouched): a counterfactual that separates the policy's pressing from the weapon's rules. Measurement
only; nothing trains with it.
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
        "hook_launch_no_shot", "hook_launch_visible", "hook_cos_launch_visible", "episodes",
        "ready", "reload_loaded", "reload_empty", "empty_idle", "fire_visible", "fire_los", "dry_fires", "fire_lost",
        "fire_dry_decisions", "hits", "refills", "empty_streaks", "empty_streak_decisions", "fire_shot")


def play(server_dir, specs, learner, mask_empty_fire=False):
    """One pass over the evaluation seeds; counts for agent `learner`."""
    actors = (Actor(specs[0], 11), Actor(specs[1], 12))
    m = dict.fromkeys(KEYS, 0.0)
    with VecTeeworldsEnv(EVAL_ENVS, server_dir, seed=EVAL_SEED, map=EVAL_MAP, grid_format="codes", raw_info=True,
                         episode_ticks=EVAL_EPISODE_TICKS) as venv:
        for _ in range(EVAL_ROUNDS):
            obs = venv.reset()
            ammo = np.full(EVAL_ENVS, 10)
            streak = np.zeros(EVAL_ENVS, dtype=int)
            while True:
                vec, codes = torch.from_numpy(obs["vec"]), torch.from_numpy(obs["grid"])
                actions = np.stack([actors[i](vec[:, i], codes[:, i]) for i in range(2)], axis=1)
                v = obs["vec"][:, learner]
                if mask_empty_fire:
                    actions[(ammo <= 0) & (v[:, V["alive"]] > 0), learner, 2] = 0
                obs, _, _, truncated, infos = venv.step(actions)
                ev = infos["events"][:, learner]
                ag = infos["agents"]
                a = actions[:, learner]
                alive = v[:, V["alive"]] > 0
                vis = (v[:, V["opp_visible"]] > 0) & alive
                los = vis & (v[:, V["opp_line_of_fire"]] > 0)
                timer = v[:, V["reload"]] > 0
                ready = alive & (ammo > 0) & ~timer
                fire = alive & (a[:, 2] == 1)
                shot = ev[:, EV["shots"]] > 0
                hook = alive & (a[:, 3] == 1)
                launch = alive & (ev[:, EV["hook_launches"]] > 0)
                m["reload_loaded"] += (alive & (ammo > 0) & timer).sum()
                m["reload_empty"] += (alive & (ammo <= 0) & timer).sum()
                m["empty_idle"] += (alive & (ammo <= 0) & ~timer).sum()
                m["ready"] += ready.sum()
                m["fire_visible"] += (fire & vis).sum()
                m["fire_los"] += (fire & los).sum()
                m["dry_fires"] += ev[:, EV["dry_fires"]][alive].sum()
                m["fire_dry_decisions"] += (fire & (ev[:, EV["dry_fires"]] > 0)).sum()
                m["fire_lost"] += (fire & timer).sum()
                m["fire_shot"] += (fire & shot).sum()
                m["hits"] += (alive & (ev[:, EV["damage_dealt"]] > 0)).sum()
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
                new_ammo = ag["ammo"][:, learner].copy()
                now_alive = ag["alive"][:, learner] > 0
                # a refill: more ammo than before minus this decision's shots, while staying alive
                m["refills"] += (alive & now_alive & (new_ammo > ammo - ev[:, EV["shots"]])).sum()
                empty = alive & (ammo <= 0)
                ended = (streak > 0) & ~empty
                m["empty_streaks"] += ended.sum()
                m["empty_streak_decisions"] += streak[ended].sum()
                streak = np.where(empty, streak + 1, 0)
                ammo = np.where(now_alive, new_ammo, 10)
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
        # weapon state at the decision, shares of alive decisions (they add up to 1)
        "weapon_ready_frac": m["ready"] / n, "reload_timer_with_ammo_frac": m["reload_loaded"] / n,
        "reload_timer_empty_frac": m["reload_empty"] / n, "empty_timer_done_frac": m["empty_idle"] / n,
        # what the presses did, shares of fire presses (shot + dry + lost = 1)
        "fire_to_shot": r("fire_shot", "fire"), "fire_to_dry_fire": r("fire_dry_decisions", "fire"),
        "fire_lost_in_timer": r("fire_lost", "fire"), "wasted_fire_share": 1 - r("fire_shot", "fire"),
        "p_fire_given_visible": r("fire_visible", "visible"), "p_fire_given_los": r("fire_los", "los"),
        "p_fire_given_ready": r("fire_ready", "ready"),
        "shots_per_episode": r("shots", "episodes"), "dry_fires_per_episode": r("dry_fires", "episodes"),
        "hit_decisions_per_episode": r("hits", "episodes"), "refills_per_episode": r("refills", "episodes"),
        "mean_empty_streak_decisions": r("empty_streak_decisions", "empty_streaks"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-empty-fire", action="store_true", help="counterfactual: no fire on an empty gun (the checkpoint's side only)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    out = {"opponent": a.opponent, "mask_empty_fire": a.mask_empty_fire, "protocol": {"seed": EVAL_SEED, "envs": EVAL_ENVS, "rounds": EVAL_ROUNDS, "sides": 2}, "checkpoints": {}}
    for c in a.checkpoints:
        m0 = play(a.server_dir, (c, a.opponent), 0, a.mask_empty_fire)
        m1 = play(a.server_dir, (a.opponent, c), 1, a.mask_empty_fire)
        rep = report({k: m0[k] + m1[k] for k in KEYS})
        out["checkpoints"][c] = round_floats(rep)
        print(c, json.dumps(out["checkpoints"][c]), flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
