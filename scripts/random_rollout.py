#!/usr/bin/env python3
"""Run random (or scripted chase) agents in N parallel environments and report stability,
event consistency, memory and throughput as JSON.

    random_rollout.py --server-dir ../teeworlds-0.7.5/build-release --envs 8 --episodes 200
"""

import argparse
import json
import multiprocessing
import os
import random
import resource
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import TeeworldsEnv, random_action  # noqa: E402
from twrl.testing import chase_action  # noqa: E402


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


def worker(args):
    a, index, port = args
    rng = random.Random(a.seed * 1000 + index)
    totals = {k: 0 for k in ("kills", "deaths", "suicides", "damage_dealt", "damage_taken")}
    violations = {"score": 0, "damage_pair": 0, "kill_pair": 0, "not_running": 0, "reset_state": 0}
    rss, latencies = [], []
    steps = ticks = 0
    start = time.perf_counter()
    with TeeworldsEnv(a.server_dir, map=a.map, mode=a.mode, action_ticks=a.action_ticks,
                      episode_ticks=a.episode_ticks, port=port) as env:
        for episode in range(a.episodes):
            obs, info = env.reset(seed=a.seed + episode)
            ev = info["events"]
            if any(e["score"] != 0 for e in ev) or not (obs[0][0] and obs[1][0]):
                violations["reset_state"] += 1
            scores = [0, 0]
            step = 0
            while True:
                if a.policy == "chase":
                    actions = [chase_action(obs[i], rng, step) for i in range(2)]
                else:
                    actions = [random_action(rng), random_action(rng)]
                t0 = time.perf_counter()
                obs, rewards, terminated, truncated, info = env.step(actions)
                if step % 16 == 0:
                    latencies.append(time.perf_counter() - t0)
                step += 1
                ev = info["events"]
                for i in range(2):
                    for k in totals:
                        totals[k] += ev[i][k]
                    if ev[i]["score"] - scores[i] != ev[i]["kills"] - ev[i]["suicides"]:
                        violations["score"] += 1
                    scores[i] = ev[i]["score"]
                    # no self damage with gun and hammer, so every point dealt is taken by the other agent
                    if ev[i]["damage_dealt"] != ev[1 - i]["damage_taken"]:
                        violations["damage_pair"] += 1
                    if ev[i]["kills"] != ev[1 - i]["deaths"]:
                        violations["kill_pair"] += 1
                if info["flags"]:
                    violations["not_running"] += 1
                if truncated:
                    break
            steps += step
            ticks += step * env.action_ticks
            if episode % a.rss_every == 0 or episode == a.episodes - 1:
                rss.append((episode, proc_status(env.server_pid, "VmRSS")))
        wall = time.perf_counter() - start
        server_cpu = proc_cpu_seconds(env.server_pid)
    latencies.sort()
    return {
        "env": index, "episodes": a.episodes, "steps": steps, "ticks": ticks, "wall_s": round(wall, 3),
        "steps_per_s": round(steps / wall), "ticks_per_s": round(ticks / wall),
        "step_latency_us_p50": round(latencies[len(latencies) // 2] * 1e6, 1),
        "step_latency_us_p99": round(latencies[int(len(latencies) * 0.99)] * 1e6, 1),
        "server_cpu_s": round(server_cpu, 3), "server_us_per_tick": round(server_cpu / ticks * 1e6, 3),
        "server_rss_kb": rss, "totals": totals, "violations": violations,
        "python_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--envs", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=10, help="per environment")
    ap.add_argument("--map", default="dm1")
    ap.add_argument("--mode", type=int, default=2)
    ap.add_argument("--policy", choices=("random", "chase"), default="random")
    ap.add_argument("--action-ticks", type=int, default=2)
    ap.add_argument("--episode-ticks", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rss-every", type=int, default=100)
    ap.add_argument("--out")
    a = ap.parse_args()

    ports = [int(p) for p in subprocess.run(["freeport", "-u", "-n", str(a.envs)], check=True,
                                            capture_output=True, text=True).stdout.split()]
    start = time.perf_counter()
    try:
        with multiprocessing.Pool(a.envs) as pool:
            results = pool.map(worker, [(a, i, ports[i]) for i in range(a.envs)])
    finally:
        subprocess.run(["freeport", "--release", ",".join(map(str, ports)), "--proto", "udp"], capture_output=True)
    wall = time.perf_counter() - start

    summary = {
        "config": vars(a),
        "load1": os.getloadavg()[0],
        "wall_s": round(wall, 3),
        "episodes": sum(r["episodes"] for r in results),
        "steps": sum(r["steps"] for r in results),
        "ticks": sum(r["ticks"] for r in results),
        "total_ticks_per_s": sum(r["ticks_per_s"] for r in results),
        "total_steps_per_s": sum(r["steps_per_s"] for r in results),
        "median_step_latency_us_p50": statistics.median(r["step_latency_us_p50"] for r in results),
        "median_step_latency_us_p99": statistics.median(r["step_latency_us_p99"] for r in results),
        "median_server_us_per_tick": statistics.median(r["server_us_per_tick"] for r in results),
        "totals": {k: sum(r["totals"][k] for r in results) for k in results[0]["totals"]},
        "violations": {k: sum(r["violations"][k] for r in results) for k in results[0]["violations"]},
        "per_env": results,
    }
    text = json.dumps(summary, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    brief = {k: v for k, v in summary.items() if k not in ("per_env", "config")}
    print(json.dumps(brief, indent=1))
    print("server rss kb per env:", [r["server_rss_kb"][0][1] for r in results], "->",
          [r["server_rss_kb"][-1][1] for r in results])
    sys.exit(1 if any(summary["violations"].values()) else 0)


if __name__ == "__main__":
    main()
