#!/usr/bin/env python3
"""Drive N environments from one process with random (or scripted chase) actions, the way a PPO
trainer will, and report stability, event consistency, memory and throughput as JSON.

Both policies are diagnostics, not AI: random is uniform noise and chase is a scripted helper.

    random_rollout.py --server-dir ../teeworlds-0.7.5/build-release --envs 8 --steps 20000
"""

import argparse
import json
import os
import random
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import VecTeeworldsEnv, random_actions  # noqa: E402
from twrl import protocol as P  # noqa: E402
from twrl.testing import chase_actions  # noqa: E402

K = {n: i for i, n in enumerate(P.EVENT_NAMES)}


def proc_status(pid, key):
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    return None


def proc_cpu_seconds(pid):
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--envs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10000, help="vector steps, each one step in every env")
    ap.add_argument("--map", default="lms1")
    ap.add_argument("--mode", type=int, default=2)
    ap.add_argument("--policy", choices=("random", "chase"), default="random")
    ap.add_argument("--action-ticks", type=int, default=2)
    ap.add_argument("--episode-ticks", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rss-every", type=int, default=5000)
    ap.add_argument("--out")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    prng = random.Random(a.seed)
    totals = np.zeros(5, dtype=np.int64)
    violations = dict.fromkeys(("score", "damage_pair", "kill_pair", "not_finite", "out_of_range", "reset_state"), 0)
    latencies, rss = [], []
    episodes = 0
    with VecTeeworldsEnv(a.envs, a.server_dir, seed=a.seed, map=a.map, mode=a.mode, action_ticks=a.action_ticks,
                         episode_ticks=a.episode_ticks, raw_info=True) as venv:
        pids = [env.server_pid for env in venv.envs]
        obs = venv.reset()
        agents = None  # raw state of the latest obs, only the chase helper reads it
        score = np.zeros((a.envs, 2), dtype=np.int64)
        step_in_episode = 0
        start = time.perf_counter()
        for step in range(a.steps):
            if a.policy == "chase" and agents is not None:
                actions = np.array([chase_actions(agents[i], prng, step_in_episode) for i in range(a.envs)], dtype=np.int32)
            else:
                actions = random_actions(rng, (a.envs, 2))
            t0 = time.perf_counter()
            obs, rewards, terminated, truncated, infos = venv.step(actions)
            latencies.append(time.perf_counter() - t0)
            agents = infos["agents"]
            step_in_episode += 1
            ev = infos["events"]
            totals += ev[..., :5].sum(axis=(0, 1))
            violations["score"] += int((ev[..., K["score"]] - score != ev[..., K["kills"]] - ev[..., K["suicides"]]).sum())
            violations["damage_pair"] += int((ev[..., K["damage_dealt"]] != ev[:, ::-1, K["damage_taken"]]).sum())
            violations["kill_pair"] += int((ev[..., K["kills"]] != ev[:, ::-1, K["deaths"]]).sum())
            if not np.isfinite(obs["vec"]).all():
                violations["not_finite"] += 1
            if np.abs(obs["vec"]).max() > 5.0 or obs["grid"].max() > 1:
                violations["out_of_range"] += 1
            score = ev[..., K["score"]].astype(np.int64)
            done = terminated | truncated
            if done.any():
                episodes += int(done.sum())
                score[done] = 0
                if (agents["alive"][done] != 1).any() or (agents["health"][done] != 10).any():
                    violations["reset_state"] += 1
                step_in_episode = 0
            if step % a.rss_every == 0 or step == a.steps - 1:
                rss.append((step, [proc_status(pid, "VmRSS") for pid in pids]))
        wall = time.perf_counter() - start
        server_cpu = [proc_cpu_seconds(pid) for pid in pids]
        action_ticks = venv.action_ticks

    ticks = a.steps * a.envs * action_ticks
    latencies.sort()
    summary = {
        "config": vars(a),
        "load1": os.getloadavg()[0],
        "wall_s": round(wall, 3),
        "episodes": episodes,
        "vector_steps": a.steps,
        "ticks": ticks,
        "total_ticks_per_s": round(ticks / wall),
        "ticks_per_s_per_env": round(ticks / wall / a.envs),
        "agent_steps_per_s": round(a.steps * a.envs * 2 / wall),
        "vector_step_latency_us_p50": round(latencies[len(latencies) // 2] * 1e6, 1),
        "vector_step_latency_us_p99": round(latencies[int(len(latencies) * 0.99)] * 1e6, 1),
        "server_us_per_tick": round(sum(server_cpu) / ticks * 1e6, 3),
        "obs_payload_bytes": P.OBS_SIZE,
        "totals": dict(zip(P.EVENT_NAMES[:5], totals.tolist())),
        "violations": violations,
        "server_rss_kb": rss,
        "python_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    text = json.dumps(summary, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    brief = {k: v for k, v in summary.items() if k not in ("config", "server_rss_kb")}
    print(json.dumps(brief, indent=1))
    print("server rss kb:", rss[0][1], "->", rss[-1][1])
    sys.exit(1 if any(violations.values()) else 0)


if __name__ == "__main__":
    main()
