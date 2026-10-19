#!/usr/bin/env python3
"""Rollout collection with W worker processes, each with its own envs and a copy of the policy.

Measures collection only (the PPO update would still run in one process). Answers whether
splitting the envs over processes speeds up collecting on this machine.

    worker_collect.py --server-dir ../teeworlds-0.7.5/build-release --envs 8 16 --workers 1 2 4
"""

import argparse
import json
import multiprocessing
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import VecTeeworldsEnv  # noqa: E402
from twrl.env import free_udp_ports, release_udp_ports  # noqa: E402
from twrl.ppo import Policy, PPOConfig, Trainer  # noqa: E402


def worker(args):
    server_dir, envs, steps, threads, ports, state, start_at, rollouts, seed = args
    torch.set_num_threads(threads)
    policy = Policy()
    policy.load_state_dict(state)
    with VecTeeworldsEnv(envs, server_dir, seed=seed, map="lms1", grid_format="codes", ports=ports) as venv:
        trainer = Trainer(venv, policy, PPOConfig(num_envs=envs, steps=steps), seed=seed)
        trainer.collect()  # warm up
        time.sleep(max(0.0, start_at - time.time()))
        t0 = time.perf_counter()
        for _ in range(rollouts):
            trainer.collect()
        return rollouts * steps * envs * 2, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--envs", type=int, nargs="+", default=[8])
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--cores", type=int, default=6, help="torch threads shared out over the workers")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    state = Policy().state_dict()
    rows = []
    for envs in a.envs:
        for w in a.workers:
            if envs % w:
                continue
            ports, reserved = free_udp_ports(envs)
            try:
                start_at = time.time() + 4.0
                jobs = [(a.server_dir, envs // w, a.steps, max(1, a.cores // w), ports[i * envs // w:(i + 1) * envs // w],
                         state, start_at, a.rollouts, 1 + i) for i in range(w)]
                with multiprocessing.get_context("spawn").Pool(w) as pool:
                    res = pool.map(worker, jobs)
            finally:
                if reserved:
                    release_udp_ports(ports)
            samples = sum(r[0] for r in res)
            wall = max(r[1] for r in res)
            rows.append({"envs": envs, "workers": w, "threads_per_worker": max(1, a.cores // w),
                         "collect_agent_steps_per_s": round(samples / wall), "load1": round(os.getloadavg()[0], 2)})
            print(json.dumps(rows[-1]), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
