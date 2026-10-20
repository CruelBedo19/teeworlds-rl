#!/usr/bin/env python3
"""How often the current, unchanged reward says anything, and what GAE makes of it.

Collects PPO rollouts without training for each actor (uniform random over the PPO action heads,
or a checkpoint) on fixed seeds and reports event and reward frequencies and the advantage
distribution. The rewards are exactly the environment's; nothing is shaped.

    reward_sparsity.py --server-dir ../teeworlds-0.7.5/build-release --actors random runs/x/ckpt_0000.pt runs/x/ckpt_final.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import VecTeeworldsEnv  # noqa: E402
from twrl.ppo import HEADS, PPOConfig, Trainer, load_policy, to_env_actions  # noqa: E402


class UniformPolicy(torch.nn.Module):
    """Uniform over the PPO heads with a zero critic, so GAE is just discounted reward."""

    action_dim = len(HEADS)

    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(1))  # the trainer builds an optimiser it never uses here

    def forward(self, vec, codes):
        return torch.zeros(vec.shape[0], sum(HEADS)), torch.zeros(vec.shape[0])

    def act(self, vec, codes, generator=None):
        heads = torch.stack([torch.randint(0, n, (vec.shape[0],), generator=generator) for n in HEADS], dim=-1)
        return heads.float(), torch.zeros(vec.shape[0]), torch.zeros(vec.shape[0]), torch.zeros(vec.shape[0])

    def env_actions(self, actions):
        return to_env_actions(np.asarray(actions).astype(np.int64))


def measure(server_dir, spec, envs, steps, rollouts, seed):
    policy = UniformPolicy() if spec == "random" else load_policy(spec)[0]
    with VecTeeworldsEnv(envs, server_dir, seed=seed, map="lms1", grid_format="codes") as venv:
        trainer = Trainer(venv, policy, PPOConfig(num_envs=envs, steps=steps), seed=seed)
        rewards, advs, events_steps = [], [], []
        orig_step = venv.step

        def step(actions):
            out = orig_step(actions)
            events_steps.append(out[4]["events"][..., :5].copy())
            return out
        venv.step = step
        for _ in range(rollouts):
            batch = trainer.collect()
            rewards.append(batch["rewards"].numpy())  # (T, 2N), includes truncation bootstrap terms
            advs.append(batch["adv"].numpy())
        finished = np.array(trainer.finished)  # (episodes, 2, 5)
    ev = np.concatenate(events_steps).reshape(-1, 5)  # one row per agent step
    r = np.concatenate([x.reshape(-1) for x in rewards])
    # reward from events only (no bootstrap terms), per agent step
    weights = np.array([1.0, -1.0, -2.0, 0.02, -0.02])
    er = ev @ weights
    seg = np.stack([x != 0 for x in rewards])  # (rollouts, T, 2N)
    ev_seg = np.concatenate(events_steps).reshape(rollouts, steps, -1, 5)
    adv = np.concatenate([x.reshape(-1) for x in advs])
    eps = len(finished) * 2
    nz_event = (np.abs(er) > 0)
    return {
        "actor": spec, "agent_steps": int(len(er)), "agent_episodes": int(eps),
        "per_agent_step": {
            "kill": float((ev[:, 0] > 0).mean()), "death": float((ev[:, 1] > 0).mean()), "suicide": float((ev[:, 2] > 0).mean()),
            "damage_dealt": float((ev[:, 3] > 0).mean()), "damage_taken": float((ev[:, 4] > 0).mean()),
            "nonzero_event_reward": float(nz_event.mean()),
        },
        "per_agent_episode": {
            "kills": float(finished[..., 0].mean()) if eps else None,
            "damage_dealt": float(finished[..., 3].mean()) if eps else None,
            "nonzero_reward_steps": float(nz_event.sum() / max(eps, 1)) if eps else None,
            "return_mean": float((finished.reshape(-1, 5) @ weights).mean()) if eps else None,
            "return_std": float((finished.reshape(-1, 5) @ weights).std()) if eps else None,
            "episodes_with_any_reward": float((np.abs(finished.reshape(-1, 5)).sum(1) > 0).mean()) if eps else None,
            "episodes_with_a_kill": float((finished.reshape(-1, 5)[:, 0] > 0).mean()) if eps else None,
        },
        "rollouts": {
            "with_any_event_reward": float((np.abs(ev_seg @ weights) > 0).any(axis=(1, 2)).mean()),
            "agent_segments_with_any_event_reward": float((np.abs(ev_seg @ weights) > 0).any(axis=1).mean()),
            "agent_segments_with_a_kill": float((ev_seg[..., 0] > 0).any(axis=1).mean()),
            "steps_with_nonzero_reward_incl_bootstrap": float(seg.mean()),
        },
        "advantage": {
            "mean": float(adv.mean()), "std": float(adv.std()),
            "abs_percentiles": {p: float(np.percentile(np.abs(adv), p)) for p in (50, 90, 99, 99.9)},
            "frac_abs_below_1e-3": float((np.abs(adv) < 1e-3).mean()),
            "frac_abs_above_0.1": float((np.abs(adv) > 0.1).mean()),
        },
        "reward_incl_bootstrap": {"mean": float(r.mean()), "std": float(r.std())},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--actors", nargs="+", required=True)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--rollouts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(6)
    report = [measure(a.server_dir, s, a.envs, a.steps, a.rollouts, a.seed) for s in a.actors]
    for r in report:
        print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
