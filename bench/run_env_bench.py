#!/usr/bin/env python3
"""Throughput of the Phase 2 environment with uniformly random actions (a diagnostic, not a policy).

Two layouts per env count N:
  single  one Python process steps a VecTeeworldsEnv of N servers (what a simple PPO loop does)
  multi   N Python processes, each stepping one env (the Phase 1 rollout layout, for comparison)

    run_env_bench.py --server-dir ../teeworlds-0.7.5/build-release --envs 1 4 8 12 --out results.csv
"""

import argparse
import csv
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import VecTeeworldsEnv, random_actions  # noqa: E402
from twrl.env import free_udp_ports, release_udp_ports  # noqa: E402
from twrl import protocol as P  # noqa: E402


def cpu_seconds(pid):
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def run(server_dir, envs, steps, map_name, seed, start_at=None, ports=None, grid_format="channels"):
    """Step a vector env of `envs` servers `steps` times; return ticks, wall time, latencies, server cpu."""
    rng = np.random.default_rng(seed)
    with VecTeeworldsEnv(envs, server_dir, seed=seed, map=map_name, ports=ports, grid_format=grid_format) as venv:
        venv.reset()
        actions = random_actions(rng, (steps, envs, 2))
        cpu0 = [cpu_seconds(e.server_pid) for e in venv.envs]
        if start_at:
            time.sleep(max(0.0, start_at - time.time()))
        lat = np.empty(steps)
        t0 = time.perf_counter()
        for i in range(steps):
            s = time.perf_counter()
            venv.step(actions[i])
            lat[i] = time.perf_counter() - s
        wall = time.perf_counter() - t0
        cpu = sum(cpu_seconds(e.server_pid) - c for e, c in zip(venv.envs, cpu0))
        return envs * steps * venv.action_ticks, wall, lat, cpu


def multi_worker(args):
    return run(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 4, 8, 12])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--map", default="lms1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid-format", choices=("channels", "codes"), default="channels")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = []
    for n in a.envs:
        ticks, wall, lat, cpu = run(a.server_dir, n, a.steps, a.map, a.seed, grid_format=a.grid_format)
        rows.append(dict(layout="single", grid=a.grid_format, envs=n, total_ticks_per_s=round(ticks / wall), ticks_per_s_per_env=round(ticks / wall / n),
                         step_latency_us_p50=round(np.percentile(lat, 50) * 1e6, 1), step_latency_us_p99=round(np.percentile(lat, 99) * 1e6, 1),
                         server_us_per_tick=round(cpu / ticks * 1e6, 2), load1=round(os.getloadavg()[0], 2)))
        print(rows[-1], flush=True)
        start_at = time.time() + 3.0  # let every worker set up before the clock starts
        ports, reserved = free_udp_ports(n)  # one call: concurrent freeport calls can collide
        try:
            with multiprocessing.Pool(n) as pool:
                results = pool.map(multi_worker, [(a.server_dir, 1, a.steps, a.map, a.seed + i, start_at, [ports[i]], a.grid_format)
                                                  for i in range(n)])
        finally:
            if reserved:
                release_udp_ports(ports)
        lat = np.concatenate([r[2] for r in results])
        rows.append(dict(layout="multi", grid=a.grid_format, envs=n, total_ticks_per_s=round(sum(r[0] / r[1] for r in results)),
                         ticks_per_s_per_env=round(sum(r[0] / r[1] for r in results) / n),
                         step_latency_us_p50=round(np.percentile(lat, 50) * 1e6, 1), step_latency_us_p99=round(np.percentile(lat, 99) * 1e6, 1),
                         server_us_per_tick=round(sum(r[3] for r in results) / sum(r[0] for r in results) * 1e6, 2),
                         load1=round(os.getloadavg()[0], 2)))
        print(rows[-1], flush=True)
    print("obs payload bytes per env step:", P.OBS_SIZE, "+ 16 header")
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
