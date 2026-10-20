#!/usr/bin/env python3
"""Bounded PPO self-play run from random weights.

    train_ppo.py --server-dir ../teeworlds-0.7.5/build-release --updates 200 --max-minutes 20
    train_ppo.py --server-dir ... --aim gaussian --envs 16 --compile --checkpoint-steps 250000 --updates 1221
    train_ppo.py --server-dir ... --run-dir runs/x --resume runs/x/ckpt_final.pt --updates 400

Writes runs/<name>/metrics.jsonl, ckpt_0000.pt (the untrained random-weight policy, saved before
any update), periodic checkpoints (every --checkpoint-every updates, or each time the agent step
count passes a multiple of --checkpoint-steps) and ckpt_final.pt. Stops at --updates (a total, counted across
resumes) or after --max-minutes of this invocation, whichever comes first.

--opponent makes it a learner-vs-frozen-opponent run instead of self-play: "initial" is this run's
own ckpt_0000.pt (the untrained random-weight network), or give a checkpoint path. The opponent is
never updated and only the learner's samples are trained on and counted as agent steps.

--resume continues from a checkpoint: weights, optimiser, update and step counters, random
streams and the env seed stream. Episodes that were running when it was saved are not resumed;
the environments start fresh episodes, so a resumed run is not bit-identical to an uninterrupted one.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import VecTeeworldsEnv  # noqa: E402
from twrl.ppo import BEHAVIOUR, Policy, PPOConfig, Trainer, flatten_agents, load_policy, save_checkpoint  # noqa: E402
from twrl.protocol import EVENT_NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--map", default="lms1")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--checkpoint-every", type=int, default=50, help="updates between checkpoints")
    ap.add_argument("--checkpoint-steps", type=int, default=0, help="agent steps between checkpoints; replaces --checkpoint-every")
    ap.add_argument("--aim", choices=("categorical", "gaussian"), default="categorical")
    ap.add_argument("--first-stride", type=int, default=2, help="stride of the first convolution (1 = the first network)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--compile", action="store_true", help="torch.compile the update (about 1.7x faster here)")
    ap.add_argument("--opponent", help='frozen opponent: "initial" (this run\'s ckpt_0000.pt) or a checkpoint path')
    ap.add_argument("--resume", help="checkpoint to continue from")
    ap.add_argument("--aim-lr-scale", type=float, default=1.0, help="Gaussian aim: learning rate of the aim's own parameters, relative to the PPO rate")
    ap.add_argument("--aim-target-kl", type=float, default=None,
                    help="Gaussian aim: switch the aim loss off for the rest of an update once a minibatch's aim KL passes 1.5x this")
    a = ap.parse_args()

    run_dir = a.run_dir or os.path.join("runs", time.strftime("ppo-%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    cfg = PPOConfig(num_envs=a.envs, steps=a.steps, aim_lr_scale=a.aim_lr_scale, aim_target_kl=a.aim_target_kl)
    resume = torch.load(a.resume, map_location="cpu", weights_only=False) if a.resume else None
    if resume:
        saved = resume.get("args", {})
        for key in ("map", "envs", "steps", "seed", "aim", "first_stride", "opponent"):
            if key in saved and saved[key] != getattr(a, key):
                sys.exit(f"--{key} {getattr(a, key)} does not match the checkpoint ({saved[key]})")
        # settings added after the checkpoint was made count as their defaults
        if {**vars(PPOConfig()), **resume["config"]} != vars(cfg):
            sys.exit(f"PPO settings differ from the checkpoint: {resume['config']}")
    else:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"args": vars(a), "ppo": vars(cfg)}, f, indent=1)
    extra = {"args": {k: getattr(a, k) for k in ("map", "envs", "steps", "seed", "aim", "first_stride", "opponent")}}

    policy = Policy(aim=a.aim, first_stride=a.first_stride)
    start = time.time()
    initial = os.path.join(run_dir, "ckpt_0000.pt")
    if a.opponent and not resume and a.opponent != "initial":
        sys.exit("a new run takes --opponent initial; other checkpoints only make sense with --resume or by path")
    with VecTeeworldsEnv(a.envs, a.server_dir, seed=a.seed, map=a.map, grid_format="codes", raw_info=True) as venv:
        opponent = None
        if a.opponent == "initial" and not resume:
            opponent = copy.deepcopy(policy)  # exactly the learner's starting network, which is ckpt_0000.pt
        elif a.opponent:
            opponent = load_policy(initial if a.opponent == "initial" else a.opponent)[0]
        if opponent is not None:
            for p in opponent.parameters():
                p.requires_grad_(False)
        trainer = Trainer(venv, policy, cfg, seed=a.seed, compile=a.compile, opponent=opponent)
        if resume:
            trainer.load_state_dict(resume)
            print(f"resumed at update {trainer.update}, {trainer.agent_steps} agent steps", flush=True)
        else:
            save_checkpoint(initial, trainer, extra)
        log = open(os.path.join(run_dir, "metrics.jsonl"), "a" if resume else "w")
        while trainer.update < a.updates and time.time() - start < a.max_minutes * 60:
            t0 = time.time()
            batch = trainer.collect()
            t1 = time.time()
            stats = trainer.train(batch)
            t2 = time.time()
            finished = np.array(trainer.finished) if trainer.finished else np.zeros((0, 2, len(EVENT_NAMES)))
            trainer.finished = []
            beh = dict(zip(BEHAVIOUR, trainer.behaviour))
            trainer.behaviour[:] = 0
            row = {"update": trainer.update, "agent_steps": trainer.agent_steps, "minutes": round((time.time() - start) / 60, 2),
                   "collect_s": round(t1 - t0, 2), "train_s": round(t2 - t1, 2),
                   "reward_per_step": batch["rewards"].mean().item(), "rollout_entropy": batch["entropy"],
                   "episodes": len(finished), **stats,
                   "visible_frac": beh["visible_steps"] / max(beh["alive_steps"], 1),
                   "aim_cos": beh["aim_cos_sum"] / max(beh["visible_steps"], 1),
                   "shots": beh["shots"], "hit_rate": beh["damage_dealt"] / max(beh["shots"], 1), "step_kills": beh["kills"]}
            if len(finished):
                if opponent is None:
                    row.update({"ep_" + k: float(v) for k, v in zip(EVENT_NAMES[:5], finished.reshape(-1, len(EVENT_NAMES))[:, :5].mean(axis=0))})
                else:  # learner first, then the frozen opponent
                    row.update({"ep_" + k: float(v) for k, v in zip(EVENT_NAMES[:5], finished[:, 0, :5].mean(axis=0))})
                    row.update({"opp_ep_" + k: float(v) for k, v in zip(EVENT_NAMES[:5], finished[:, 1, :5].mean(axis=0))})
                    score = EVENT_NAMES.index("score")
                    row["win_rate"] = float((finished[:, 0, score] > finished[:, 1, score]).mean())
                    row["loss_rate"] = float((finished[:, 0, score] < finished[:, 1, score]).mean())
            log.write(json.dumps(row) + "\n")
            log.flush()
            if trainer.update % 10 == 0 or trainer.update == 1:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            if a.checkpoint_steps:
                due = trainer.agent_steps // a.checkpoint_steps > (trainer.agent_steps - batch["adv"].shape[0]) // a.checkpoint_steps
            else:
                due = trainer.update % a.checkpoint_every == 0
            if due:
                save_checkpoint(os.path.join(run_dir, f"ckpt_{trainer.update:04d}.pt"), trainer, extra)
        final = os.path.join(run_dir, "ckpt_final.pt")
        save_checkpoint(final, trainer, extra)

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
