#!/usr/bin/env python3
"""Audit the reward / credit-assignment pipeline of the PPO trainer on recorded rollouts.

Nothing here trains: the Trainer only collects (its update is never called). Every rollout step
is recorded from the env's side (observations, raw agent state, the env actions really sent,
events, done flags) and the trainer's batch is then checked against an independent re-computation:

  1. mapping   batch rows <-> (env, agent): observations, actions, rewards, done flags
  2. rewards   event -> reward weights, truncation bootstrap value, agent 0/1 event consistency
  3. GAE       advantages and returns recomputed per agent with a plain loop
  4. timing    with 2 ticks per action: does a fire press at decision t show up as a shot in obs t+1,
               and never before; does a damage event land after the shot that caused it
  5. aim       network sample -> integer target (exact), target -> projectile direction in the
               world (from the observed own projectile one decision later), sign conventions of
               opp_dx/opp_dy against raw positions
  6. hits      every damage event linked back to the shot that caused it by projectile geometry;
               delay, aim error, and the advantage that shot's decision received
  7. signal    advantage vs aim alignment on shot decisions; hit vs miss; how much of the aim
               policy gradient comes from decisions where the aim mattered at all

    credit_audit.py --server-dir ../teeworlds-0.7.5/build-release --run-dir runs/learn-gauss-vs-frozen
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
from twrl.ppo import PPOConfig, Trainer, flatten_agents, gaussian_env_actions, load_policy  # noqa: E402

V = P.VEC_INDEX
EV = {n: i for i, n in enumerate(P.EVENT_NAMES)}
GUN = P.WEAPONS.index("gun")
GUN_SPEED, GUN_CURVATURE, PHYS_SIZE = 2200.0, 1.25, 28.0


class Recorder:
    """Wraps a VecTeeworldsEnv and keeps what the env saw and answered at every step."""

    def __init__(self, venv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.steps = []

    def __getattr__(self, name):
        return getattr(self.venv, name)

    def reset(self):
        obs = self.venv.reset()
        self.agents = self.venv._agents.copy()
        self.obs = obs
        return obs

    def step(self, actions):
        before = {"vec": self.obs["vec"].copy(), "grid": self.obs["grid"].copy(), "agents": self.agents}
        obs, r, term, trunc, infos = self.venv.step(actions)
        rec = {**before, "actions": np.array(actions, copy=True), "events": infos["events"].copy(), "rewards": r.copy(),
               "term": term.copy(), "trunc": trunc.copy(), "next_agents": infos["agents"].copy()}
        if "final_index" in infos:
            rec["final_index"] = infos["final_index"].copy()
            rec["final_vec"], rec["final_grid"] = infos["final_vec"].copy(), infos["final_grid"].copy()
        self.steps.append(rec)
        self.agents = infos["agents"].copy()
        self.obs = obs
        return obs, r, term, trunc, infos


def calc_pos(start, direction, tau):
    t = tau * GUN_SPEED
    return np.stack([start[..., 0] + direction[..., 0] * t,
                     start[..., 1] + direction[..., 1] * t + GUN_CURVATURE / 10000 * t * t], -1)


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def collect(server_dir, learner_path, opponent_path, envs, rollouts, seed, episode_ticks=3000):
    torch.set_num_threads(4)
    learner = load_policy(learner_path)[0]
    opponent = load_policy(opponent_path)[0] if opponent_path else None
    cfg = PPOConfig(num_envs=envs, steps=128)
    with VecTeeworldsEnv(envs, server_dir, seed=seed, map="lms1", grid_format="codes", raw_info=True,
                         episode_ticks=episode_ticks) as venv:
        rec = Recorder(venv)
        trainer = Trainer(rec, learner, cfg, seed=seed, opponent=opponent)
        batches = []
        for _ in range(rollouts):
            start = len(rec.steps)
            batch = trainer.collect()
            # the obs after the last step, for the independent bootstrap
            batch["after_vec"], batch["after_grid"] = rec.obs["vec"].copy(), rec.obs["grid"].copy()
            batch["first_step"] = start
            batches.append(batch)
        return trainer, rec, batches, learner, cfg


def check_pipeline(trainer, rec, batches, policy, cfg):
    """Sections 1-3: compare the trainer's batch with an independent re-computation. Returns max errors."""
    rows = trainer.rows
    env_of, agent_of = rows // 2, rows % 2
    w = rec.venv.reward_weights
    T = cfg.steps
    err = {k: 0.0 for k in ("obs", "actions_to_env", "reward", "adv", "returns")}
    counts = {"truncations": 0, "terminations": 0, "learner_rows": len(rows)}
    for batch in batches:
        s0 = batch["first_step"]
        steps = rec.steps[s0:s0 + T]
        b = len(rows)
        vec = batch["vec"].view(T, b, -1).numpy()
        acts = batch["actions"].view(T, b, -1).numpy()
        rewards = np.zeros((T, b))
        dones = np.zeros((T, b))
        for t, st in enumerate(steps):
            err["obs"] = max(err["obs"], np.abs(vec[t] - st["vec"][env_of, agent_of]).max())
            sent = st["actions"][env_of, agent_of]
            err["actions_to_env"] = max(err["actions_to_env"], np.abs(policy.env_actions(acts[t]) - sent).max())
            r = st["events"][env_of, agent_of, :5].astype(np.float64) @ w.astype(np.float64)
            done = (st["term"] | st["trunc"])[env_of].astype(float)
            counts["truncations"] += int(st["trunc"].sum())
            counts["terminations"] += int(st["term"].sum())
            if "final_index" in st:
                for j, (e, k) in enumerate(zip(env_of, agent_of)):
                    hit = np.flatnonzero(st["final_index"] == e)
                    if len(hit) and st["trunc"][e]:
                        fv = torch.from_numpy(st["final_vec"][hit[0], k][None])
                        fc = torch.from_numpy(st["final_grid"][hit[0], k][None])
                        with torch.no_grad():
                            r[j] += cfg.gamma * policy(fv, fc)[1].item()
            rewards[t], dones[t] = r, done
        err["reward"] = max(err["reward"], np.abs(rewards - batch["rewards"].numpy()).max())
        # independent GAE, one agent at a time, with the values the trainer stored
        values = batch["values"].view(T, b).numpy().astype(np.float64)
        after_v = torch.from_numpy(batch["after_vec"][env_of, agent_of])
        after_c = torch.from_numpy(batch["after_grid"][env_of, agent_of])
        with torch.no_grad():
            last_value = policy(after_v, after_c)[1].numpy().astype(np.float64)
        adv = np.zeros((T, b))
        for j in range(b):
            gae = 0.0
            for t in reversed(range(T)):
                nxt = last_value[j] if t == T - 1 else values[t + 1, j]
                nonterminal = 1.0 - dones[t, j]
                delta = rewards[t, j] + cfg.gamma * nxt * nonterminal - values[t, j]
                gae = delta + cfg.gamma * cfg.lam * nonterminal * gae
                adv[t, j] = gae
        err["adv"] = max(err["adv"], np.abs(adv - batch["adv"].view(T, b).numpy()).max())
        err["returns"] = max(err["returns"], np.abs(adv + values - batch["returns"].view(T, b).numpy()).max())
    return err, counts


def check_events(rec):
    """Agent 0/1 consistency of the events: what one agent deals the other takes, kills are deaths."""
    ev = np.stack([s["events"] for s in rec.steps])  # (S, N, 2, 6)
    dealt_vs_taken = np.abs(ev[..., 0, EV["damage_dealt"]] - ev[..., 1, EV["damage_taken"]]).max(), \
        np.abs(ev[..., 1, EV["damage_dealt"]] - ev[..., 0, EV["damage_taken"]]).max()
    kills_vs_deaths = np.abs(ev[..., 0, EV["kills"]] - ev[..., 1, EV["deaths"]]).max(), \
        np.abs(ev[..., 1, EV["kills"]] - ev[..., 0, EV["deaths"]]).max()
    rewards = np.stack([s["rewards"] for s in rec.steps])
    recomputed = ev[..., :5].astype(np.float64) @ rec.venv.reward_weights.astype(np.float64)
    return {"damage_dealt_minus_other_taken_max": [int(x) for x in dealt_vs_taken],
            "kills_minus_other_deaths_max": [int(x) for x in kills_vs_deaths],
            "env_reward_vs_events_max_err": float(np.abs(rewards - recomputed).max()),
            "steps_with_damage": int((ev[..., EV["damage_dealt"]] > 0).sum()),
            "steps_with_kill": int((ev[..., EV["kills"]] > 0).sum()),
            "steps_with_suicide": int((ev[..., EV["suicides"]] > 0).sum())}


def shots_and_presses(rec):
    """Per (step, env, agent): did a fire press happen at this decision, did the ammo drop by the next obs."""
    S = len(rec.steps)
    N = rec.num_envs
    fire = np.stack([s["actions"][..., 2] for s in rec.steps])  # (S, N, 2)
    prev_fire = np.zeros_like(fire)
    prev_fire[1:] = fire[:-1]
    reset_before = np.zeros((S, N), bool)  # the decision at s is the first of an episode
    for s in range(1, S):
        reset_before[s] = rec.steps[s - 1]["term"] | rec.steps[s - 1]["trunc"]
    # protocol v3: fire=1 is a new press at every decision (v2: only after a decision with fire=0)
    press = (fire == 1) if P.VERSION >= 3 else (fire == 1) & (prev_fire == 0)
    a_now = np.stack([s["agents"] for s in rec.steps])
    a_next = np.stack([s["next_agents"] for s in rec.steps])
    ended = np.stack([s["term"] | s["trunc"] for s in rec.steps])
    valid = (a_now["alive"] == 1) & (a_next["alive"] == 1) & ~ended[..., None] & ~reset_before[..., None]
    ammo_drop = valid & (a_now["ammo"] - a_next["ammo"] > 0)
    if "shots" in P.EVENT_NAMES:  # counted by the server, exact even when a refill hides the ammo drop
        shot = np.stack([s["events"][..., P.EVENT_NAMES.index("shots")] for s in rec.steps]) > 0
    else:
        shot = ammo_drop
    gun = a_now["weapon"] == GUN
    ready = valid & gun & (a_now["ammo"] > 0) & (a_now["vec"][..., V["reload"]] == 0)
    return {"press": press, "shot": shot, "ammo_drop": ammo_drop, "valid": valid, "ready": ready, "a_now": a_now, "a_next": a_next,
            "fire": fire, "prev_fire": prev_fire}


def timing_report(sp):
    press, shot, valid, ready = sp["press"], sp["shot"], sp["valid"], sp["ready"]
    out = {
        "protocol": P.VERSION,
        "shots_per_decision": float(shot.sum() / valid.sum()),
        "ammo_drop_without_shot_event": int((sp["ammo_drop"] & ~shot).sum()),
        "shot_event_without_ammo_drop": int((shot & valid & ~sp["ammo_drop"]).sum()),
        "fire1_decisions": int((valid & (sp["fire"] == 1)).sum()),
        "fire1_ready_decisions": int((ready & (sp["fire"] == 1)).sum()),
        "fire1_ready_that_shot": int((ready & (sp["fire"] == 1) & shot).sum()),
        "decisions_valid": int(valid.sum()),
        "shots": int(shot.sum()),
        "shots_without_press_at_that_decision": int((shot & ~press).sum()),
        "ready_and_press": int((ready & press).sum()),
        "ready_and_press_that_shot": int((ready & press & shot).sum()),
        "ready_without_press_that_shot": int((ready & ~press & shot).sum()),
        # fire=1 while the gun is ready, but fire was already held at the previous decision: no new press,
        # no shot (the gun is not automatic). The observation does not include the previous fire state.
        "ready_fire1_after_fire1_no_shot": int((ready & (sp["fire"] == 1) & (sp["prev_fire"] == 1) & ~shot).sum()),
        "ready_fire1_after_fire0": int((ready & (sp["fire"] == 1) & (sp["prev_fire"] == 0)).sum()),
        "weapon_counts_alive": {P.WEAPONS[w]: int(c) for w, c in zip(*np.unique(sp["a_now"]["weapon"][sp["a_now"]["alive"] == 1], return_counts=True))},
    }
    # a shot visible one decision late would show up as press at t, ammo drop between t+1 and t+2
    late = np.zeros_like(shot)
    late[1:] = shot[1:] & press[:-1] & ~press[1:]
    out["shots_one_decision_after_a_press_only"] = int(late.sum())
    return out


def aim_report(rec, sp, policy, batches, trainer):
    """Network sample -> integer target, target -> world direction of the new projectile, sign conventions."""
    out = {}
    # 1. float aim sample -> sent integer target
    acts = torch.cat([b["actions"] for b in batches]).numpy()
    conv = gaussian_env_actions(acts)
    a = acts[:, 4:6]
    n = np.linalg.norm(a, axis=1)
    expect = np.where(n[:, None] > 1e-9, np.rint(256 * a / np.maximum(n, 1e-9)[:, None]), [256, 0])
    out["target_rounding_max_err"] = float(np.abs(conv[:, 4:6] - expect).max())
    out["target_vs_sample_angle_deg"] = {"max": float(angle_deg(conv[:, 4:6], a).max()),
                                         "p99": float(np.percentile(angle_deg(conv[:, 4:6], a), 99))}
    # 2. sign conventions of the opponent offset in the observation vs raw world positions
    a_now = sp["a_now"]
    vis = a_now["vec"][..., V["opp_visible"]] > 0
    other = a_now[:, :, ::-1]
    raw = other["pos"] - a_now["pos"]
    obs_off = np.stack([a_now["vec"][..., V["opp_dx"]] * P.VIEW_HALF_WIDTH, a_now["vec"][..., V["opp_dy"]] * P.VIEW_HALF_HEIGHT], -1)
    out["opp_offset_vs_raw_pos_max_err_units"] = float(np.abs(obs_off[vis] - raw[vis]).max())
    # 3. the sent target vs the direction of the projectile that shot created (own projectile, next obs)
    shot = sp["shot"]
    errs = []
    nxt = sp["a_next"]
    acts_env = np.stack([s["actions"] for s in rec.steps])
    for s, e, k in zip(*np.nonzero(shot)):
        v = nxt[s, e, k]["vec"]
        best = None
        for p in range(P.NUM_PROJECTILES):
            base = V["proj%d_present" % p]
            if v[base] > 0 and v[base + 5] == 0:  # present, own
                d = np.hypot(v[base + 1] * P.VIEW_HALF_WIDTH, v[base + 2] * P.VIEW_HALF_HEIGHT)
                if best is None or d < best[0]:
                    best = (d, np.array([v[base + 3], v[base + 4]]))
        if best is not None and best[0] < 150:
            errs.append(angle_deg(best[1], acts_env[s, e, k, 4:6]))
    errs = np.array(errs)
    out["projectile_dir_vs_target"] = {"shots_matched": int(len(errs)), "of_shots": int(shot.sum()),
                                       "median_deg": float(np.median(errs)), "p95_deg": float(np.percentile(errs, 95)),
                                       "max_deg": float(errs.max())}
    return out


def link_hits(rec, sp, max_delay=50):
    """Each damage event (step t, env e, dealer k) -> the dealer's shot s <= t whose projectile passes
    closest to the victim at that time. Returns a list of dicts."""
    a_now, a_next = sp["a_now"], sp["a_next"]
    shot = sp["shot"]
    acts = np.stack([s["actions"] for s in rec.steps])
    ev = np.stack([s["events"] for s in rec.steps])
    ended = np.stack([s["term"] | s["trunc"] for s in rec.steps])
    hits = []
    for t, e, k in zip(*np.nonzero(ev[..., EV["damage_dealt"]] > 0)):
        victim = 1 - k
        cands = []
        for s in range(t, max(-1, t - max_delay), -1):
            if s < t and ended[s, e]:
                break  # an earlier episode
            if not shot[s, e, k]:
                continue
            d = acts[s, e, k, 4:6].astype(float)
            d /= np.linalg.norm(d)
            start = a_now[s, e, k]["pos"] + d * PHYS_SIZE * 0.75
            # the projectile's first tick tests the segment from one tick *behind* the start point
            # (CProjectile::Tick, Pt = (Tick-StartTick-1)/TickSpeed), so it can hit behind the shooter
            taus = np.arange(-1, (t - s + 1) * 2 + 1) / 50.0
            path = calc_pos(start[None], d[None], taus)
            vpos = np.stack([a_now[t, e, victim]["pos"], a_next[t, e, victim]["pos"]])
            miss = np.min(np.linalg.norm(path[:, None, :] - vpos[None], axis=-1))
            cands.append((miss, s))
        if not cands:
            hits.append({"t": int(t), "env": int(e), "agent": int(k), "linked": False})
            continue
        # the most recent shot whose path passes within reach of the victim (34 units + a tick of
        # movement); only if none does, the closest path
        near = [c for c in cands if c[0] <= 45.0]
        miss, s = near[0] if near else min(cands)
        off = a_now[s, e, victim]["pos"] - a_now[s, e, k]["pos"]
        hits.append({"t": int(t), "env": int(e), "agent": int(k), "linked": True, "shot_step": int(s), "delay": int(t - s),
                     # a press on a ready gun with no ammo drop: a shot hidden by an ammo refill in the same decision
                     "undetected_press_within_5": int(any(sp["ready"][u, e, k] and sp["press"][u, e, k] and not shot[u, e, k]
                                                          for u in range(max(0, t - 5), t + 1))),
                     "dealer_weapon": P.WEAPONS[a_now[t, e, k]["weapon"]], "dealer_alive": int(a_now[t, e, k]["alive"]),
                     "dealer_died_since_shot": int(any(a_now[u, e, k]["alive"] == 0 for u in range(s, t + 1))),
                     "victim_dist_at_hit": float(np.linalg.norm(a_now[t, e, victim]["pos"] - a_now[t, e, k]["pos"])),
                     "victim_offset_at_shot": [float(x) for x in a_now[s, e, victim]["pos"] - a_now[s, e, k]["pos"]],
                     "victim_vel_at_hit": [float(x) for x in a_now[t, e, victim]["vel"]],
                     "path_miss_units": float(miss), "candidates": len(cands),
                     "dist_at_shot": float(np.linalg.norm(off)),
                     "aim_error_deg": float(angle_deg(acts[s, e, k, 4:6], off)),
                     "damage": int(ev[t, e, k, EV["damage_dealt"]])})
    return hits


def signal_report(rec, sp, hits, batches, trainer, policy, cfg):
    """Advantages at the decisions where a shot left the gun, vs the aim alignment of that shot."""
    rows = trainer.rows
    env_of, agent_of = rows // 2, rows % 2
    T, b = cfg.steps, len(rows)
    adv = np.zeros((len(rec.steps), rec.num_envs, 2)) * np.nan
    adv_norm = adv.copy()
    for batch in batches:
        s0 = batch["first_step"]
        a = batch["adv"].view(T, b).numpy()
        an = (a - a.mean()) / (a.std() + 1e-8)  # what the update uses (adv_norm="batch")
        adv[s0:s0 + T, env_of, agent_of] = a
        adv_norm[s0:s0 + T, env_of, agent_of] = an
    learner = ~np.isnan(adv)
    a_now = sp["a_now"]
    acts = np.stack([s["actions"] for s in rec.steps])
    other = a_now[:, :, ::-1]
    off = other["pos"] - a_now["pos"]
    vis = a_now["vec"][..., V["opp_visible"]] > 0
    cos = ((acts[..., 4:6] * off).sum(-1) / np.maximum(np.linalg.norm(acts[..., 4:6], axis=-1) * np.linalg.norm(off, axis=-1), 1e-9))
    shot = sp["shot"] & learner
    hit_shots = {(h["shot_step"], h["env"], h["agent"]) for h in hits if h["linked"]}
    is_hit = np.zeros_like(shot)
    for s, e, k in hit_shots:
        is_hit[s, e, k] = True
    sel = shot & vis
    out = {"learner_decisions": int(learner.sum()), "learner_shots": int(shot.sum()), "learner_shots_opp_visible": int(sel.sum()),
           "learner_hit_shots": int((is_hit & learner).sum())}

    def summary(x):
        x = x[np.isfinite(x)]
        if not len(x):
            return None
        return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "sem": float(x.std() / math.sqrt(len(x)))}

    out["adv_shot_hit"] = summary(adv_norm[shot & is_hit])
    out["adv_shot_miss_visible"] = summary(adv_norm[sel & ~is_hit])
    out["adv_no_shot"] = summary(adv_norm[learner & ~shot])
    out["corr_adv_cos_on_visible_shots"] = float(np.corrcoef(adv_norm[sel], cos[sel])[0, 1]) if sel.sum() > 2 else None
    out["corr_adv_cos_on_visible_non_shots"] = float(np.corrcoef(adv_norm[learner & vis & ~shot], cos[learner & vis & ~shot])[0, 1])
    bins = [-1, 0.0, 0.9, 0.99, 1.0001]
    out["adv_by_cos_on_visible_shots"] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = sel & (cos >= lo) & (cos < hi)
        out["adv_by_cos_on_visible_shots"].append({"cos": [lo, min(hi, 1.0)], "hit_rate": float(is_hit[m].mean()) if m.sum() else None,
                                                   **(summary(adv_norm[m]) or {"n": 0})})
    # the aim part of the policy gradient at ratio 1: adv * (a - mu) / sigma^2, projected on the
    # direction that turns the aim towards the opponent; split by where it comes from
    vec = np.stack([s["vec"] for s in rec.steps])
    grid = np.stack([s["grid"] for s in rec.steps])
    idx = np.nonzero(learner & vis)
    with torch.no_grad():
        mu = policy.aim_mean(torch.from_numpy(vec[idx]), torch.from_numpy(grid[idx])).numpy().astype(float)
    std = policy._std().detach().numpy().astype(float)
    aim_sample = np.full(acts.shape[:3] + (2,), np.nan)
    for batch in batches:
        s0 = batch["first_step"]
        aim_sample[s0:s0 + T, env_of, agent_of] = batch["actions"].view(T, b, -1).numpy()[..., 4:6]
    score = (aim_sample[idx] - mu) / std ** 2 * adv_norm[idx][:, None]  # gradient of logp * adv w.r.t. mu
    o = off[idx]
    # unit vector that rotates the mean towards the opponent (perpendicular to mu, towards o)
    mu_u = mu / np.maximum(np.linalg.norm(mu, axis=1, keepdims=True), 1e-9)
    perp = np.stack([-mu_u[:, 1], mu_u[:, 0]], 1)
    turn = np.sign((perp * o).sum(1))[:, None] * perp
    towards = (score * turn).sum(1)
    shot_i = shot[idx]
    # per update (4096 learner samples): expected push towards the opponent vs the noise of the sum
    per_batch = 4096 / learner.sum()

    def part(m):
        x = towards[m]
        return {"n": int(m.sum()), "mean": float(x.mean()), "rms": float(np.sqrt((x ** 2).mean())),
                "per_update_signal": float(x.sum() * per_batch), "per_update_noise_sd": float(x.std() * math.sqrt(m.sum() * per_batch))}
    # hook launches also use the target vector: counted by the server (protocol 4), else hook pressed while idle
    if "hook_launches" in P.EVENT_NAMES:
        hook_launch = (np.stack([s["events"][..., P.EVENT_NAMES.index("hook_launches")] for s in rec.steps]) > 0)[idx]
    else:
        hook_launch = ((acts[..., 3] == 1) & (a_now["hook_state"] == 0))[idx]
    out["aim_gradient_towards_opponent"] = {"from_shot_decisions": part(shot_i), "from_other_decisions": part(~shot_i),
                                            "from_hook_launch_non_shot": part(~shot_i & hook_launch),
                                            "from_rest_non_shot": part(~shot_i & ~hook_launch)}
    allx = towards
    out["aim_gradient_towards_opponent"]["all_per_update_snr"] = float(allx.sum() * per_batch / (allx.std() * math.sqrt(len(allx) * per_batch)))
    out["aim_gradient_towards_opponent"]["shot_only_per_update_snr"] = float(towards[shot_i].sum() * per_batch / (towards[shot_i].std() * math.sqrt(shot_i.sum() * per_batch)))
    used = shot_i | hook_launch
    out["aim_gradient_towards_opponent"]["shot_or_hook_per_update_snr"] = float(towards[used].sum() * per_batch / (towards[used].std() * math.sqrt(used.sum() * per_batch)))
    return out, adv, adv_norm, cos, is_hit


def traces(rec, sp, hits, adv, adv_norm, trainer, batches, cfg, count):
    """A few hits by the learner, decision by decision from the shot to the reward."""
    a_now, a_next = sp["a_now"], sp["a_next"]
    acts = np.stack([s["actions"] for s in rec.steps])
    ev = np.stack([s["events"] for s in rec.steps])
    rows = trainer.rows
    learner_hits = [h for h in hits if h["linked"] and not np.isnan(adv[h["t"], h["env"], h["agent"]])
                    and not np.isnan(adv[h["shot_step"], h["env"], h["agent"]])]
    out = []
    w = rec.venv.reward_weights
    for h in learner_hits[:count]:
        e, k, s, t = h["env"], h["agent"], h["shot_step"], h["t"]
        lines = []
        for u in range(s - 1, t + 2):
            if u < 0 or u >= len(rec.steps):
                continue
            me, op = a_now[u, e, k], a_now[u, e, 1 - k]
            off = op["pos"] - me["pos"]
            r = float(ev[u, e, k, :5] @ w)
            lines.append({
                "decision": u - s, "episode_tick_of_obs": None,
                "my_pos": [round(float(x), 1) for x in me["pos"]], "opp_offset": [round(float(x), 1) for x in off],
                "obs_opp_dx_dy_world": [round(float(me["vec"][V["opp_dx"]] * P.VIEW_HALF_WIDTH), 1),
                                        round(float(me["vec"][V["opp_dy"]] * P.VIEW_HALF_HEIGHT), 1)],
                "opp_visible": int(me["vec"][V["opp_visible"]] > 0), "line_of_fire": int(me["vec"][V["opp_line_of_fire"]] > 0),
                "ammo": int(me["ammo"]), "ammo_next": int(a_next[u, e, k]["ammo"]), "fire": int(acts[u, e, k, 2]),
                "target": [int(x) for x in acts[u, e, k, 4:6]],
                "aim_error_deg": round(float(angle_deg(acts[u, e, k, 4:6], off)), 1),
                "opp_health_next": int(a_next[u, e, 1 - k]["health"]), "opp_armor_next": int(a_next[u, e, 1 - k]["armor"]),
                "events": {n: int(ev[u, e, k, i]) for i, n in enumerate(P.EVENT_NAMES[:5]) if ev[u, e, k, i]},
                "reward": round(r, 3), "adv": round(float(adv[u, e, k]), 4), "adv_norm": round(float(adv_norm[u, e, k]), 3),
            })
        out.append({"hit": h, "steps": lines})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--learner", default="ckpt_final.pt")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--rollouts", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--traces", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    learner = os.path.join(a.run_dir, a.learner)
    initial = os.path.join(a.run_dir, "ckpt_0000.pt")
    report = {"learner": learner, "opponent": initial}

    # self-play: only the mapping / reward / GAE checks, with short episodes so that truncations
    # (and their bootstrap) fall inside the rollouts
    trainer, rec, batches, policy, cfg = collect(a.server_dir, learner, None, 4, 3, a.seed, episode_ticks=150)
    err, counts = check_pipeline(trainer, rec, batches, policy, cfg)
    report["self_play_pipeline"] = {"max_abs_err": err, **counts}
    print("self-play pipeline:", json.dumps(report["self_play_pipeline"]), flush=True)

    trainer, rec, batches, policy, cfg = collect(a.server_dir, learner, initial, a.envs, a.rollouts, a.seed)
    err, counts = check_pipeline(trainer, rec, batches, policy, cfg)
    report["vs_frozen_pipeline"] = {"max_abs_err": err, **counts}
    print("vs frozen pipeline:", json.dumps(report["vs_frozen_pipeline"]), flush=True)
    report["events"] = check_events(rec)
    print("events:", json.dumps(report["events"]), flush=True)
    sp = shots_and_presses(rec)
    report["timing"] = timing_report(sp)
    print("timing:", json.dumps(report["timing"]), flush=True)
    report["aim"] = aim_report(rec, sp, policy, batches, trainer)
    print("aim:", json.dumps(report["aim"]), flush=True)
    hits = link_hits(rec, sp)
    linked = [h for h in hits if h["linked"]]
    contact = [h for h in linked if h["dist_at_shot"] < 70]
    ranged = [h for h in linked if h["dist_at_shot"] >= 70]
    report["hits_by_range"] = {
        "contact_lt70": {"n": len(contact), "aim_error_deg_median": float(np.median([h["aim_error_deg"] for h in contact])) if contact else None,
                         "delay_0_or_1": sum(h["delay"] <= 1 for h in contact)},
        "ranged_ge70": {"n": len(ranged), "aim_error_deg_median": float(np.median([h["aim_error_deg"] for h in ranged])) if ranged else None,
                        "aim_error_deg_p90": float(np.percentile([h["aim_error_deg"] for h in ranged], 90)) if ranged else None,
                        "path_miss_median": float(np.median([h["path_miss_units"] for h in ranged])) if ranged else None,
                        "path_miss_max": float(np.max([h["path_miss_units"] for h in ranged])) if ranged else None,
                        "delay": {str(d): int(c) for d, c in zip(*np.unique([h["delay"] for h in ranged], return_counts=True))} if ranged else {}},
    }
    print("hits by range:", json.dumps(report["hits_by_range"]), flush=True)
    longd = [h for h in linked if h["delay"] > 10]
    report["long_delay_examples"] = longd[:12]
    report["long_delay_links"] = {"n": len(longd), "with_undetected_press_within_5": sum(h["undetected_press_within_5"] for h in longd),
                                  "shot_reused_by_an_earlier_hit": sum(any(o is not h and o["shot_step"] == h["shot_step"] and o["env"] == h["env"]
                                                                           and o["agent"] == h["agent"] and o["t"] < h["t"] for o in linked) for h in longd)}
    short = [h for h in linked if h["delay"] <= 10]
    report["short_delay_links"] = {"n": len(short), "with_undetected_press_within_5": sum(h["undetected_press_within_5"] for h in short)}
    print("long-delay links:", json.dumps(report["long_delay_links"]), "short:", json.dumps(report["short_delay_links"]), flush=True)
    report["hits"] = {
        "damage_events": len(hits), "linked_to_a_shot": len(linked),
        "path_miss_units": {"median": float(np.median([h["path_miss_units"] for h in linked])),
                            "p95": float(np.percentile([h["path_miss_units"] for h in linked], 95)),
                            "max": float(np.max([h["path_miss_units"] for h in linked]))},
        "delay_decisions": {str(d): int(c) for d, c in zip(*np.unique([h["delay"] for h in linked], return_counts=True))},
        "aim_error_deg_of_hitting_shots": {"median": float(np.median([h["aim_error_deg"] for h in linked])),
                                           "p95": float(np.percentile([h["aim_error_deg"] for h in linked], 95))},
        "dist_at_shot": {"median": float(np.median([h["dist_at_shot"] for h in linked]))},
    }
    print("hits:", json.dumps(report["hits"]), flush=True)
    sig, adv, adv_norm, cos, is_hit = signal_report(rec, sp, hits, batches, trainer, policy, cfg)
    report["signal"] = sig
    print("signal:", json.dumps(sig, indent=1), flush=True)
    report["traces"] = traces(rec, sp, hits, adv, adv_norm, trainer, batches, cfg, a.traces)
    for tr in report["traces"]:
        print("trace", json.dumps(tr["hit"]))
        for line in tr["steps"]:
            print("   ", json.dumps(line))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
