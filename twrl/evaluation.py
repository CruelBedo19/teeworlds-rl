"""Fixed evaluation protocol for comparing checkpoints over a long run.

Every evaluation plays the same seeded starts (EVAL_SEED, EVAL_ENVS x EVAL_ROUNDS episodes per
matchup and side), samples actions from fixed torch generator seeds and uses fixed torch threads,
so evaluating the same checkpoint twice gives the same numbers. All numbers are measurements;
none of them is fed back as a reward. Chase is never an opponent here.

Matchups for a checkpoint `current`:
    self      current controls both agents (behaviour)
    random    current vs uniform random over the action heads, both sides
    initial   current vs ckpt_0000 (the untrained random-weight policy), both sides
    previous  current vs the checkpoint before it, both sides
"""

import math

import numpy as np
import torch

from . import protocol as P
from .env import VecTeeworldsEnv, episode_seed
from .ppo import HEADS, load_policy, to_env_actions

EVAL_SEED = 9001
EVAL_ENVS = 8
EVAL_ROUNDS = 4  # episodes per matchup and side = EVAL_ENVS * EVAL_ROUNDS
EVAL_MAP = "lms1"
EVAL_EPISODE_TICKS = 3000
EVAL_THREADS = 4
REWARD = np.array([1.0, -1.0, -2.0, 0.02, -0.02])  # the env's weights, for reporting only

V = P.VEC_INDEX
EV = {n: i for i, n in enumerate(P.EVENT_NAMES)}


def seed_set():
    """The episode seeds every evaluation plays, in order."""
    return [[episode_seed(EVAL_SEED, i, k) for i in range(EVAL_ENVS)] for k in range(1, EVAL_ROUNDS + 1)]


class Actor:
    """Maps observations of one side to env actions (B, 6). spec: "random" or a checkpoint path."""

    def __init__(self, spec, seed):
        self.spec = spec
        self.gen = torch.Generator().manual_seed(seed)
        self.policy = None if spec == "random" else load_policy(spec)[0]

    def __call__(self, vec, codes):
        if self.policy is None:
            heads = torch.stack([torch.randint(0, n, (vec.shape[0],), generator=self.gen) for n in HEADS], dim=-1)
        else:
            with torch.no_grad():
                heads = self.policy.act(vec, codes, self.gen)[0]
        return to_env_actions(heads.numpy())


def play(server_dir, spec0, spec1):
    """Agent 0 played by spec0, agent 1 by spec1, on the fixed seed set. Per agent metric sums."""
    actors = (Actor(spec0, 11), Actor(spec1, 12))
    keys = ("episodes", "steps", "alive_steps", "kills", "deaths", "suicides", "damage_dealt", "damage_taken", "shots",
            "visible", "encounters", "aim_cos", "reward", "wins", "draws")
    m = {k: np.zeros(2) for k in keys}
    with VecTeeworldsEnv(EVAL_ENVS, server_dir, seed=EVAL_SEED, map=EVAL_MAP, grid_format="codes", raw_info=True,
                         episode_ticks=EVAL_EPISODE_TICKS) as venv:
        for _ in range(EVAL_ROUNDS):
            obs = venv.reset()
            prev_ammo = prev_alive = None
            was_visible = np.zeros((EVAL_ENVS, 2), dtype=bool)
            score = np.zeros((EVAL_ENVS, 2))
            while True:
                vec = torch.from_numpy(obs["vec"])
                codes = torch.from_numpy(obs["grid"])
                actions = np.stack([actors[i](vec[:, i], codes[:, i]) for i in range(2)], axis=1)
                v = obs["vec"]
                obs, _, _, truncated, infos = venv.step(actions)
                ev = infos["events"]
                for i in range(2):
                    alive = v[:, i, V["alive"]] > 0
                    vis = v[:, i, V["opp_visible"]] > 0
                    m["steps"][i] += EVAL_ENVS
                    m["alive_steps"][i] += alive.sum()
                    m["visible"][i] += vis.sum()
                    m["encounters"][i] += (vis & ~was_visible[:, i]).sum()
                    d = np.stack([v[:, i, V["opp_dx"]] * P.VIEW_HALF_WIDTH, v[:, i, V["opp_dy"]] * P.VIEW_HALF_HEIGHT], 1)
                    aim = actions[:, i, 4:6].astype(np.float64)
                    cos = (aim * d).sum(1) / np.maximum(np.hypot(*aim.T) * np.hypot(*d.T), 1e-6)
                    m["aim_cos"][i] += cos[vis].sum()
                    for k in ("kills", "deaths", "suicides", "damage_dealt", "damage_taken"):
                        m[k][i] += ev[:, i, EV[k]].sum()
                    m["reward"][i] += (ev[:, i, :5] @ REWARD).sum()
                was_visible = v[:, :, V["opp_visible"]] > 0
                if truncated.all():
                    final = ev[:, :, EV["score"]]
                    m["episodes"] += EVAL_ENVS
                    m["wins"][0] += (final[:, 0] > final[:, 1]).sum()
                    m["wins"][1] += (final[:, 1] > final[:, 0]).sum()
                    m["draws"] += (final[:, 0] == final[:, 1]).sum()
                    break
                ag = infos["agents"]
                if prev_ammo is not None:
                    both = (prev_alive == 1) & (ag["alive"] == 1)
                    m["shots"] += ((prev_ammo - ag["ammo"] > 0) & both).sum(axis=0)
                prev_ammo, prev_alive = ag["ammo"].copy(), ag["alive"].copy()
    return m


def side(m, i):
    eps = max(m["episodes"][i], 1)
    return {
        "episodes": int(m["episodes"][i]),
        "kills": m["kills"][i] / eps, "deaths": m["deaths"][i] / eps, "suicides": m["suicides"][i] / eps,
        "damage_dealt": m["damage_dealt"][i] / eps, "damage_taken": m["damage_taken"][i] / eps,
        "shots": m["shots"][i] / eps, "hit_rate": m["damage_dealt"][i] / max(m["shots"][i], 1),
        "opp_visible_frac": m["visible"][i] / max(m["alive_steps"][i], 1), "encounters": m["encounters"][i] / eps,
        "aim_cos_when_visible": m["aim_cos"][i] / max(m["visible"][i], 1),
        "reward": m["reward"][i] / eps, "win_rate": m["wins"][i] / eps, "draw_rate": m["draws"][i] / eps,
    }


def both_sides(server_dir, a, b):
    """a vs b with sides swapped; metrics of a, metrics of b, each over 2 * EVAL_ENVS * EVAL_ROUNDS episodes."""
    ab, ba = play(server_dir, a, b), play(server_dir, b, a)
    merged_a = {k: np.array([ab[k][0] + ba[k][1]]) for k in ab}
    merged_b = {k: np.array([ab[k][1] + ba[k][0]]) for k in ab}
    return side(merged_a, 0), side(merged_b, 0)


def evaluate(server_dir, current, initial=None, previous=None):
    torch.set_num_threads(EVAL_THREADS)
    report = {"protocol": {"seed": EVAL_SEED, "envs": EVAL_ENVS, "rounds": EVAL_ROUNDS, "map": EVAL_MAP,
                           "episode_ticks": EVAL_EPISODE_TICKS, "threads": EVAL_THREADS, "seeds": seed_set()},
              "current": current, "initial": initial, "previous": previous}
    m = play(server_dir, current, current)
    merged = {k: np.array([m[k].sum()]) for k in m}
    report["self"] = side(merged, 0)
    report["vs"] = {}
    for name, opp in (("random", "random"), ("initial", initial), ("previous", previous)):
        if opp is None:
            continue
        me, them = both_sides(server_dir, current, opp)
        report["vs"][name] = {"current": me, "opponent": them}
    return report


def round_floats(x, digits=4):
    if isinstance(x, dict):
        return {k: round_floats(v, digits) for k, v in x.items()}
    if isinstance(x, list):
        return [round_floats(v, digits) for v in x]
    if isinstance(x, float) and math.isfinite(x):
        return round(x, digits)
    return x
