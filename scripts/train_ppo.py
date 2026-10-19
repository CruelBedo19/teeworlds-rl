#!/usr/bin/env python3
"""Short, bounded PPO self-play run from random weights. Not a long training job.

    train_ppo.py --server-dir ../teeworlds-0.7.5/build-release --updates 200 --max-minutes 20

Writes runs/<name>/metrics.jsonl, ckpt_0000.pt (the untrained random-weight policy, saved before
any update), periodic checkpoints and ckpt_final.pt. Stops at --updates or --max-minutes,
whichever comes first.
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
from twrl.ppo import Policy, PPOConfig, Trainer, flatten_agents, load_policy, save_checkpoint  # noqa: E402

EVENTS = ("kills", "deaths", "suicides", "damage_dealt", "damage_taken")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--map", default="lms1")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    run_dir = a.run_dir or os.path.join("runs", time.strftime("ppo-%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    cfg = PPOConfig(num_envs=a.envs, steps=a.steps)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(a), "ppo": vars(cfg)}, f, indent=1)

    policy = Policy()
    start = time.time()
    with VecTeeworldsEnv(a.envs, a.server_dir, seed=a.seed, map=a.map, grid_format="codes") as venv:
        trainer = Trainer(venv, policy, cfg, seed=a.seed)
        save_checkpoint(os.path.join(run_dir, "ckpt_0000.pt"), trainer)
        log = open(os.path.join(run_dir, "metrics.jsonl"), "w")
        while trainer.update < a.updates and time.time() - start < a.max_minutes * 60:
            t0 = time.time()
            batch = trainer.collect()
            t1 = time.time()
            stats = trainer.train(batch)
            t2 = time.time()
            finished = np.array(trainer.finished) if trainer.finished else np.zeros((0, 2, 5))
            trainer.finished = []
            row = {"update": trainer.update, "agent_steps": trainer.agent_steps, "minutes": round((time.time() - start) / 60, 2),
                   "collect_s": round(t1 - t0, 2), "train_s": round(t2 - t1, 2),
                   "reward_per_step": batch["rewards"].mean().item(), "rollout_entropy": batch["entropy"],
                   "episodes": len(finished), **stats}
            if len(finished):
                per_agent = finished.reshape(-1, 5).mean(axis=0)
                row.update({"ep_" + k: float(v) for k, v in zip(EVENTS, per_agent)})
            log.write(json.dumps(row) + "\n")
            log.flush()
            if trainer.update % 10 == 0 or trainer.update == 1:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            if trainer.update % a.checkpoint_every == 0:
                save_checkpoint(os.path.join(run_dir, f"ckpt_{trainer.update:04d}.pt"), trainer)
        final = os.path.join(run_dir, "ckpt_final.pt")
        save_checkpoint(final, trainer)

        # the saved checkpoint must give back exactly the network that was trained
        probe = flatten_agents(venv.reset())
        loaded, state = load_policy(final)
        with torch.no_grad():
            same = all(torch.equal(x, y) for x, y in zip(policy.eval()(*probe), loaded(*probe)))
        print(json.dumps({"final_update": trainer.update, "agent_steps": trainer.agent_steps,
                          "minutes": round((time.time() - start) / 60, 2), "checkpoint_roundtrip_identical": same,
                          "run_dir": run_dir}))
        if not same:
            sys.exit("checkpoint does not reproduce the trained policy")


if __name__ == "__main__":
    main()
