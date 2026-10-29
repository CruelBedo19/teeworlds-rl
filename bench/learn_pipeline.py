#!/usr/bin/env python3
"""Unattended learnability run: train, watch, evaluate, judge.

    learn_pipeline.py --run-dir runs/x --eval-at 500000,1000000,2000000 -- <train_ppo.py arguments>

Starts scripts/train_ppo.py (its --run-dir is added), then every --poll seconds:
  - reads the new metrics.jsonl rows and stops the run when a watchdog rule trips (the rules and the
    reason are written to <run-dir>/pipeline.json);
  - evaluates (scripts/eval_suite.py, one at a time) ckpt_0000 and the first checkpoint at or past
    each --eval-at step count, written as eval/at_<steps>.json like the earlier learnability runs.
At the end it writes <run-dir>/summary.json: per evaluation point the matchup against ckpt_0000 and
the training statistics around it, and the four pre-registered criteria at the last point.
"""

import argparse
import glob
import json
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PY = sys.executable

# watchdog rules, fixed before the run
RULES = {
    "grad_norm_max_above": 20.0,  # v3 never passed 4.5 (clipped to 0.5 before the step)
    "aim_kl_window": 20,  # updates
    "aim_kl_mean_above": 0.05,  # v3 reached 0.05 at about 1.5M without stabilisation
    "aim_kl_after_steps": 500000,
    # the exact full-batch Gaussian KL; the minibatch k3 estimate aim_approx_kl (about 30 shot decisions per
    # minibatch) spikes to 5-12 on single updates whose exact KL is 0.15, and stopped two runs that way
    "aim_kl_metric": "aim_kl_exact",
    "entropy_below": 1.0,  # categorical heads, start 3.17, v3 never under 2.1
    "log_std_below": -4.5,  # floor is -5
    "no_shots_updates": 50,
    "stale_minutes": 10,  # no new metrics row
}


def finite(row):
    return all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float)) and not isinstance(v, bool))


def watchdog(rows, minutes_since_row):
    if not rows:
        return "no metrics after %d minutes" % RULES["stale_minutes"] if minutes_since_row > RULES["stale_minutes"] else None
    last = rows[-1]
    if not finite(last):
        return f"non-finite metric at update {last['update']}"
    if last["grad_norm_max"] > RULES["grad_norm_max_above"]:
        return f"grad norm {last['grad_norm_max']:.2f} at update {last['update']}"
    w = rows[-RULES["aim_kl_window"]:]
    # runs logged before aim_kl_exact existed fall back to the estimate
    key = RULES["aim_kl_metric"] if any(RULES["aim_kl_metric"] in r for r in w) else "aim_approx_kl"
    kls = [r[key] for r in w if key in r]  # (an update without shot decisions has no exact KL)
    if kls and last["agent_steps"] >= RULES["aim_kl_after_steps"] and len(w) == RULES["aim_kl_window"]:
        kl = np.mean(kls)
        if kl > RULES["aim_kl_mean_above"]:
            return f"aim KL {kl:.4f} ({key}, mean of {len(w)} updates) at update {last['update']}"
    if np.mean([r["entropy"] for r in w]) < RULES["entropy_below"]:
        return f"categorical entropy collapse at update {last['update']}"
    if "log_std_x" in last and min(last["log_std_x"], last["log_std_y"]) < RULES["log_std_below"]:
        return f"aim log_std collapse at update {last['update']}"
    n = RULES["no_shots_updates"]
    if len(rows) >= n and sum(r["shots"] for r in rows[-n:]) == 0:
        return f"no shots for {n} updates at update {last['update']}"
    if minutes_since_row > RULES["stale_minutes"]:
        return f"no new metrics for {minutes_since_row:.0f} minutes (bridge or env stall)"
    return None


_steps = {}


def steps_of(path):
    if path not in _steps:
        _steps[path] = torch.load(path, map_location="cpu", weights_only=False)["agent_steps"]
    return _steps[path]


def window_stats(rows, steps, width=250000):
    """Means of the training statistics over the updates in (steps - width, steps]."""
    w = [r for r in rows if steps - width < r["agent_steps"] <= steps]
    keys = ("reward_per_step", "entropy", "aim_entropy", "log_std_x", "log_std_y", "approx_kl", "clip_frac", "aim_approx_kl",
            "aim_kl_exact", "aim_kl_mean_part", "aim_kl_std_part", "aim_mu_shift", "floor_hit_x", "floor_hit_y",
            "aim_clip_frac", "aim_off", "aim_mask_frac", "policy_loss", "aim_policy_loss", "value_loss", "grad_norm",
            "grad_norm_max", "visible_frac", "aim_cos", "hit_rate", "shots", "win_rate", "loss_rate")
    out = {"updates": len(w)}
    for k in keys:
        v = [r[k] for r in w if k in r]
        if v:
            out[k] = float(np.max(v) if k == "grad_norm_max" else np.mean(v))
    return out


def criteria(points, baseline_hit):
    """The four pre-registered criteria against ckpt_0000 at the last evaluation point.
    Trend (4): the damage ratio and the aim cosine are higher at the end than at the first point and
    the damage ratio's rank correlation with the step count is positive."""
    last = points[-1]
    c = last["vs_initial"]
    ratio = c["ratio"]
    out = {
        "1_damage_ratio_ge_1.25": ratio >= 1.25,
        "2_reward_pos_and_wins_gt_losses": c["reward"] > 0 and c["wins"] > c["losses"],
        "3_aim_cos_ge_0.10_or_hit_ge_1.5x": c["aim_cos"] >= 0.10 or c["hit_rate"] >= 1.5 * baseline_hit,
    }
    if len(points) >= 3:
        r = [p["vs_initial"]["ratio"] for p in points]
        x = list(range(len(r)))
        tau = np.mean([np.sign(r[j] - r[i]) * np.sign(x[j] - x[i]) for i in x for j in x if i < j])
        out["trend_kendall_tau_ratio"] = float(tau)
        out["4_consistent_trend"] = bool(tau > 0 and r[-1] > r[0] and c["aim_cos"] > points[0]["vs_initial"]["aim_cos"])
    else:
        out["4_consistent_trend"] = False
    out["all_four"] = all(v for k, v in out.items() if k[0].isdigit())
    return out


def matchup(report):
    c, o = report["vs"]["initial"]["current"], report["vs"]["initial"]["opponent"]
    n = c["episodes"]
    return {"ratio": c["damage_dealt"] / max(c["damage_taken"], 1e-9), "reward": c["reward"],
            "wins": round(c["win_rate"] * n), "losses": round(o["win_rate"] * n), "draws": round(c["draw_rate"] * n),
            "kills": c["kills"], "deaths": c["deaths"], "suicides": c["suicides"], "damage_dealt": c["damage_dealt"],
            "damage_taken": c["damage_taken"], "hit_rate": c["hit_rate"], "aim_cos": c["aim_cos_when_visible"], "shots": c["shots"],
            "encounters": c["encounters"], "visible_frac": c["opp_visible_frac"], "opp_hit_rate": o["hit_rate"]}


def summarize(run_dir, targets, rows, state):
    base_path = os.path.join(run_dir, "eval", "ckpt_0000.json")
    base = json.load(open(base_path)) if os.path.exists(base_path) else None
    points = []
    for t in targets:
        p = os.path.join(run_dir, "eval", f"at_{t}.json")
        if not os.path.exists(p):
            continue
        rep = json.load(open(p))
        s = steps_of(rep["current"])
        points.append({"target": t, "checkpoint": os.path.basename(rep["current"]), "agent_steps": s, "vs_initial": matchup(rep),
                       "vs_random": rep["vs"]["random"]["current"], "self": rep["self"], "training": window_stats(rows, s)})
    summary = {"pipeline": state, "baseline_ckpt_0000": base and {"self": base["self"], "vs_random": base["vs"]["random"]["current"]},
               "points": points}
    if points and base:
        summary["criteria_at_last_point"] = criteria(points, base["self"]["hit_rate"])
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary.get("criteria_at_last_point"), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--server-dir", default="../teeworlds-0.7.5/build-release")
    ap.add_argument("--eval-at", required=True, help="comma-separated agent step counts")
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--summarize", action="store_true", help="only rebuild summary.json from a finished run's files")
    ap.add_argument("--aim-kl-limit", type=float, help="replaces the aim_kl_mean_above rule (long runs: single spikes trip 0.05)")
    ap.add_argument("train_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.aim_kl_limit is not None:
        RULES["aim_kl_mean_above"] = a.aim_kl_limit
    os.chdir(ROOT)
    targets = [int(x) for x in a.eval_at.split(",")]
    if a.summarize:
        with open(os.path.join(a.run_dir, "metrics.jsonl")) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(a.run_dir, "pipeline.json")) as f:
            summarize(a.run_dir, targets, rows, json.load(f))
        return
    train_args = [x for x in a.train_args if x != "--"]
    os.makedirs(a.run_dir, exist_ok=True)
    state = {"rules": RULES, "train_args": train_args, "eval_at": targets, "started": time.strftime("%F %T"), "stop_reason": None}

    def save():
        with open(os.path.join(a.run_dir, "pipeline.json"), "w") as f:
            json.dump(state, f, indent=1)

    save()
    train_log = open(os.path.join(a.run_dir, "train.log"), "w")
    train = subprocess.Popen([PY, "scripts/train_ppo.py", "--server-dir", a.server_dir, "--run-dir", a.run_dir] + train_args,
                             stdout=train_log, stderr=subprocess.STDOUT, start_new_session=True)
    state["train_pid"] = train.pid
    save()
    metrics = os.path.join(a.run_dir, "metrics.jsonl")
    rows, last_row_time = [], time.time()
    queue = ["ckpt_0000"]  # evaluation names still to do; at_<steps> are added as checkpoints appear
    pending = list(targets)
    done = {}
    ev = None  # (name, Popen)
    eval_log = open(os.path.join(a.run_dir, "eval.log"), "a")

    def start_eval(name):
        if name == "ckpt_0000":
            ckpt, out = "ckpt_0000.pt", "ckpt_0000"
        else:
            ckpt, out = done[name], name
        return name, subprocess.Popen([PY, "scripts/eval_suite.py", "--server-dir", a.server_dir, "--run-dir", a.run_dir,
                                       "--checkpoint", ckpt, "--out-name", out], stdout=eval_log, stderr=subprocess.STDOUT)

    while True:
        running = train.poll() is None
        if os.path.exists(metrics):
            with open(metrics) as f:
                new = [json.loads(line) for line in f.readlines()[len(rows):] if line.strip()]
            if new:
                rows += new
                last_row_time = time.time()
        if running and state["stop_reason"] is None and os.path.exists(os.path.join(a.run_dir, "ckpt_0000.pt")):
            reason = watchdog(rows, (time.time() - last_row_time) / 60)
            if reason:
                state["stop_reason"] = reason
                state["stopped_at"] = time.strftime("%F %T")
                save()
                os.killpg(train.pid, signal.SIGINT)
                try:
                    train.wait(120)
                except subprocess.TimeoutExpired:
                    os.killpg(train.pid, signal.SIGKILL)
        # checkpoints that reached a target
        ckpts = sorted(glob.glob(os.path.join(a.run_dir, "ckpt_[0-9][0-9][0-9][0-9].pt")))
        for t in list(pending):
            hits = [(s, p) for p in ckpts if (s := steps_of(p)) >= t]
            if hits:
                done[f"at_{t}"] = os.path.basename(min(hits)[1])
                queue.append(f"at_{t}")
                pending.remove(t)
        if ev is not None and ev[1].poll() is not None:
            state.setdefault("evals", {})[ev[0]] = {"exit": ev[1].returncode, "checkpoint": done.get(ev[0], "ckpt_0000.pt"),
                                                    "finished": time.strftime("%F %T")}
            save()
            ev = None
        # (ckpt_0000.pt only exists once the trainer has started)
        if ev is None and queue and (queue[0] != "ckpt_0000" or os.path.exists(os.path.join(a.run_dir, "ckpt_0000.pt"))):
            ev = start_eval(queue.pop(0))
        if not running and ev is None and not queue:
            break
        time.sleep(a.poll)

    state["train_exit"] = train.returncode
    state["finished"] = time.strftime("%F %T")
    state["updates"] = rows[-1]["update"] if rows else 0
    state["agent_steps"] = rows[-1]["agent_steps"] if rows else 0
    state["train_minutes"] = rows[-1]["minutes"] if rows else 0
    state["not_reached"] = pending
    save()

    summarize(a.run_dir, targets, rows, state)

if __name__ == "__main__":
    main()
