#!/usr/bin/env python3
"""Where the time of a PPO update goes: simulation, bridge, inference, update.

    profile_ppo.py --server-dir ../teeworlds-0.7.5/build-release --envs 8 --threads 4 --updates 5

Splits a rollout into policy inference, env.step (send, wait for the servers, decode) and the
rest, measures the PPO update, and reads the CPU time of the Python process and the servers
from /proc while collecting. Uses an untrained policy; the numbers are about speed only.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import VecTeeworldsEnv  # noqa: E402
from twrl.ppo import Policy, PPOConfig, Trainer  # noqa: E402


def cpu(pid):
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


class Timed:
    """Wraps a callable and adds its wall time to a bucket."""

    def __init__(self, fn, buckets, name):
        self.fn, self.buckets, self.name = fn, buckets, name

    def __call__(self, *a, **kw):
        t = time.perf_counter()
        try:
            return self.fn(*a, **kw)
        finally:
            self.buckets[self.name] += time.perf_counter() - t


def profile(server_dir, envs, steps, threads, updates, minibatch, epochs, seed=1):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    b = {k: 0.0 for k in ("inference", "env_send", "env_wait_recv", "env_decode", "env_step_total")}
    with VecTeeworldsEnv(envs, server_dir, seed=seed, map="lms1", grid_format="codes") as venv:
        policy = Policy()
        trainer = Trainer(venv, policy, PPOConfig(num_envs=envs, steps=steps, minibatch=minibatch, epochs=epochs), seed=seed)
        policy.act = Timed(policy.act, b, "inference")
        for env in venv.envs:
            env.send_step = Timed(env.send_step, b, "env_send")
            env.recv_obs_into = Timed(env.recv_obs_into, b, "env_wait_recv")
        venv.step = Timed(venv.step, b, "env_step_total")
        trainer.collect()  # warm up
        trainer.train(trainer.collect())
        for k in b:
            b[k] = 0.0
        pids = [e.server_pid for e in venv.envs]
        collect_s = train_s = 0.0
        srv0 = sum(cpu(p) for p in pids)
        py0 = cpu(os.getpid())
        for _ in range(updates):
            t0 = time.perf_counter()
            batch = trainer.collect()
            collect_s += time.perf_counter() - t0
        srv_cpu = sum(cpu(p) for p in pids) - srv0
        py_cpu_collect = cpu(os.getpid()) - py0
        py1 = cpu(os.getpid())
        for _ in range(updates):
            t0 = time.perf_counter()
            trainer.train(batch)
            train_s += time.perf_counter() - t0
        py_cpu_train = cpu(os.getpid()) - py1
    samples = updates * steps * envs * 2
    b["env_decode"] = b["env_step_total"] - b["env_send"] - b["env_wait_recv"]
    collect_other = collect_s - b["inference"] - b["env_step_total"]
    per = lambda x: round(x / samples * 1e6, 1)  # noqa: E731  microseconds per agent step
    return {
        "envs": envs, "threads": threads, "minibatch": minibatch, "epochs": epochs, "samples_per_update": steps * envs * 2,
        "us_per_agent_step": {"inference": per(b["inference"]), "env_send": per(b["env_send"]),
                              "env_wait_recv": per(b["env_wait_recv"]), "env_decode": per(b["env_decode"]),
                              "collect_other": per(collect_other), "collect_total": per(collect_s), "update": per(train_s)},
        "agent_steps_per_s": round(samples / (collect_s + train_s)),
        "collect_agent_steps_per_s": round(samples / collect_s),
        "cpu_cores_during_collect": {"servers": round(srv_cpu / collect_s, 2), "python": round(py_cpu_collect / collect_s, 2)},
        "cpu_cores_during_update": round(py_cpu_train / train_s, 2),
        "server_us_per_tick": round(srv_cpu / (samples / 2 * 2) * 1e6, 2),
        "load1": round(os.getloadavg()[0], 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--envs", type=int, nargs="+", default=[8])
    ap.add_argument("--threads", type=int, nargs="+", default=[4])
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for envs in a.envs:
        for threads in a.threads:
            rows.append(profile(a.server_dir, envs, a.steps, threads, a.updates, a.minibatch, a.epochs))
            print(json.dumps(rows[-1]), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
