#!/usr/bin/env python3
"""How the Gaussian aim's update behaves at a checkpoint (measurement only, nothing is trained or saved).

    aim_probe.py --server-dir ../teeworlds-0.7.5/build-v4 --run-dir runs/x --checkpoints ckpt_0123.pt ... \
        --batches 4 --kl-batches 2 --out probe.json

For each checkpoint the trainer is restored with its optimiser and random state, and training batches are
collected against the run's ckpt_0000 exactly as in the run (its own PPOConfig). Per checkpoint:

- state: aim mean length |mu| (bounded mean, at most 1) and std; how many decisions per batch carry a
  shot, a hook launch without a shot; the advantage's spread on those decisions.
- gradients, at the start of the update (ratio 1, advantages normalised over the batch as in train()):
  the aim loss's gradient from the shot decisions alone and from the hook-launch-only decisions alone,
  on the aim's own parameters (its output rows and log_std) and on the shared trunk; their norms and
  cosines within a batch and across batches (across batches no noise is shared, so a cross-batch cosine
  between shot and hook gradients measures how their expected directions agree). Also the cosine on
  the trunk between the aim gradient and the rest of the loss (discrete heads, value, entropy).
- the KL control: one update replayed from the checkpoint on the same batch three ways, each from the
  same restored state: as in the run (aim_target_kl), aim loss off from the first minibatch (what the
  shared trunk alone does to the aim), and no KL stop. Per minibatch the aim KL train() logs (measured
  before that minibatch's step) and, after the update, the aim KL on the batch for shot and hook-only
  decisions.
"""

import argparse
import copy
import json
import math
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import protocol as P  # noqa: E402
from twrl.env import DEFAULT_REWARD_WEIGHTS, VecTeeworldsEnv  # noqa: E402
from twrl.ppo import PPOConfig, Trainer, load_policy  # noqa: E402

SHOT, LAUNCH = P.EVENT_NAMES.index("shots"), P.EVENT_NAMES.index("hook_launches")


class Recorder:
    """The vector env, with the events of every step kept for the probe."""

    def __init__(self, venv):
        self._venv, self.events = venv, []

    def step(self, actions):
        out = self._venv.step(actions)
        self.events.append(out[4]["events"].copy())
        return out

    def __getattr__(self, name):
        return getattr(self._venv, name)


def masks(trainer, events):
    """Shot and hook-launch-only masks in batch order (T * learner rows)."""
    ev = np.stack(events)[:, :, :, :].reshape(len(events), -1, len(P.EVENT_NAMES))[:, trainer.rows]
    shot = torch.from_numpy((ev[..., SHOT] > 0).reshape(-1))
    launch = torch.from_numpy((ev[..., LAUNCH] > 0).reshape(-1))
    return shot, launch & ~shot


def param_groups(policy):
    head_rows = slice(policy.pi.weight.shape[0] - 2, policy.pi.weight.shape[0])
    trunk = [p for n, p in policy.named_parameters() if not n.startswith(("pi.", "v.", "log_std"))]
    return head_rows, trunk


def flat_grads(policy, loss):
    head_rows, trunk = param_groups(policy)
    policy.zero_grad()
    loss.backward()
    head = torch.cat([policy.pi.weight.grad[head_rows].reshape(-1), policy.pi.bias.grad[head_rows], policy.log_std.grad])
    tr = torch.cat([p.grad.reshape(-1) for p in trunk])
    return head.clone(), tr.clone()


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def aim_kl(trainer, batch, m):
    with torch.no_grad():
        _, _, _, _, alp = trainer.policy.evaluate(batch["vec"], batch["codes"], batch["actions"])
        r = torch.exp(alp - batch["aim_logp"])
        k = (r - 1) - torch.log(r)
    return float(k[m].mean()) if m.any() else None


def gradients(trainer, batch, shot, hook):
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pol = trainer.policy
    _, _, _, _, alp = pol.evaluate(batch["vec"], batch["codes"], batch["actions"])
    n = len(adv)
    g = {}
    for name, m in (("shot", shot), ("hook", hook)):
        loss = -(m.float() * adv * alp).sum() / n
        g[name] = flat_grads(pol, loss)
        _, _, _, _, alp = pol.evaluate(batch["vec"], batch["codes"], batch["actions"])
    # the rest of the loss at ratio 1 on the trunk: discrete heads' policy gradient, value, entropy
    logp, ent, value, _, alp = pol.evaluate(batch["vec"], batch["codes"], batch["actions"])
    cfg = trainer.cfg
    rest = -(adv * (logp - alp)).mean() + cfg.vf_coef * 0.5 * ((value - batch["returns"]) ** 2).mean() - cfg.ent_coef * ent.mean()
    g["rest"] = flat_grads(pol, rest)
    pol.zero_grad()
    return g


def replay_update(trainer, batch, state, mode):
    """One update from the restored checkpoint state, as train() does it. mode: "run" (the run's KL stop),
    "aim_off" (aim loss off from the first minibatch), "no_stop" (no KL stop)."""
    trainer.policy.load_state_dict(state["policy"])
    trainer.opt.load_state_dict(state["optimizer"])
    trainer.gen.set_state(state["generator"])
    cfg = trainer.cfg
    total = batch["adv"].shape[0]
    adv_all = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
    aim_weight = 0.0 if mode == "aim_off" else 1.0
    per_mb, off_at = [], None
    for epoch in range(cfg.epochs):
        perm = torch.randperm(total, generator=trainer.gen)
        for start in range(0, total, cfg.minibatch):
            mb = perm[start:start + cfg.minibatch]
            L = trainer.losses(batch, mb, adv_all[mb], aim_weight)
            trainer.opt.zero_grad()
            L["loss"].backward()
            nn.utils.clip_grad_norm_(trainer.policy.parameters(), cfg.max_grad_norm)
            step = cfg.aim_lr_scale * aim_weight
            before = [p.detach()[i].clone() for p, i in trainer.aim_params] if step != 1.0 else None
            trainer.opt.step()
            if before is not None:
                with torch.no_grad():
                    for (p, i), old in zip(trainer.aim_params, before):
                        p[i] = old + step * (p[i] - old)
            with torch.no_grad():
                m, ar = L["aim_mask"], L["aim_ratio"]
                kl = float((m * ((ar - 1) - torch.log(ar))).sum() / m.sum().clamp(min=1))
            per_mb.append(kl)
            if mode == "run" and cfg.aim_target_kl is not None and kl > 1.5 * cfg.aim_target_kl and aim_weight:
                aim_weight, off_at = 0.0, len(per_mb)
    return per_mb, off_at


def probe(server_dir, run_dir, checkpoint, batches, kl_batches, seed, dry_fire_penalty=0.0):
    state = torch.load(os.path.join(run_dir, checkpoint), map_location="cpu", weights_only=False)
    policy, _ = load_policy(os.path.join(run_dir, checkpoint))
    opponent, _ = load_policy(os.path.join(run_dir, "ckpt_0000.pt"))
    cfg = PPOConfig(**state["config"])
    weights = {**DEFAULT_REWARD_WEIGHTS, "dry_fires": -dry_fire_penalty} if dry_fire_penalty else None
    with VecTeeworldsEnv(cfg.num_envs, server_dir, seed=seed, map="lms1", grid_format="codes", raw_info=True,
                         timeout=600, reward_weights=weights) as venv:
        rec = Recorder(venv)
        trainer = Trainer(rec, policy, cfg, seed=seed, opponent=opponent)
        trainer.opt.load_state_dict(state["optimizer"])
        out = {"update": state["update"], "agent_steps": state["agent_steps"], "batches": []}
        grads = []
        for b in range(batches):
            rec.events = []
            # every batch from the checkpoint's own weights (the replays below move them)
            trainer.policy.load_state_dict(state["policy"])
            batch = trainer.collect()
            shot, hook = masks(trainer, rec.events)
            vec = batch["vec"]
            alive = vec[:, P.VEC_INDEX["alive"]] > 0
            with torch.no_grad():
                mu = trainer.policy.aim_mean(vec, batch["codes"])
            mlen = mu.norm(dim=-1)
            adv = batch["adv"]
            row = {"shot_decisions": int(shot.sum()), "hook_only_decisions": int(hook.sum()),
                   "mu_len_alive": float(mlen[alive].mean()), "mu_len_shot": float(mlen[shot].mean()),
                   "mu_len_hook": float(mlen[hook].mean()) if hook.any() else None,
                   "std": [float(x) for x in trainer.policy._std().detach()],
                   "adv_std_all": float(adv.std()), "adv_std_shot": float(adv[shot].std()),
                   "adv_std_hook": float(adv[hook].std()), "return_std": float(batch["returns"].std()),
                   "reward_events": int((batch["rewards"].abs() > 1e-9).sum())}
            g = gradients(trainer, batch, shot, hook)
            grads.append(g)
            row.update({
                "g_shot_head": float(g["shot"][0].norm()), "g_hook_head": float(g["hook"][0].norm()),
                "g_shot_trunk": float(g["shot"][1].norm()), "g_hook_trunk": float(g["hook"][1].norm()),
                "g_rest_trunk": float(g["rest"][1].norm()),
                "cos_shot_hook_head": cos(g["shot"][0], g["hook"][0]), "cos_shot_hook_trunk": cos(g["shot"][1], g["hook"][1]),
                "cos_aim_rest_trunk": cos(g["shot"][1] + g["hook"][1], g["rest"][1]),
                "cos_shot_rest_trunk": cos(g["shot"][1], g["rest"][1]), "cos_hook_rest_trunk": cos(g["hook"][1], g["rest"][1]),
            })
            if b < kl_batches:
                for mode in ("run", "aim_off", "no_stop"):
                    per_mb, off_at = replay_update(trainer, batch, state, mode)
                    row[f"kl_{mode}"] = {"per_minibatch": per_mb, "off_at_minibatch": off_at,
                                         "end_shot": aim_kl(trainer, batch, shot), "end_hook": aim_kl(trainer, batch, hook),
                                         "end_masked": aim_kl(trainer, batch, shot | hook)}
            out["batches"].append(row)
        # across batches: noise is independent, so these measure agreement of the expected directions
        cross = {"shot_shot": [], "hook_hook": [], "shot_hook": []}
        for part, idx in (("head", 0), ("trunk", 1)):
            for i in range(len(grads)):
                for j in range(len(grads)):
                    if i < j:
                        cross["shot_shot"].append((part, cos(grads[i]["shot"][idx], grads[j]["shot"][idx])))
                        cross["hook_hook"].append((part, cos(grads[i]["hook"][idx], grads[j]["hook"][idx])))
                    if i != j:
                        cross["shot_hook"].append((part, cos(grads[i]["shot"][idx], grads[j]["hook"][idx])))
        out["cross_batch_cos"] = {f"{k}_{part}": float(np.mean([c for p, c in v if p == part]))
                                  for k, v in cross.items() for part in ("head", "trunk")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="file names inside --run-dir")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--kl-batches", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--dry-fire-penalty", type=float, default=0.0, help="as the run was trained (train_ppo.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = {"run_dir": a.run_dir, "seed": a.seed, "checkpoints": {}}
    for c in a.checkpoints:
        out["checkpoints"][c] = probe(a.server_dir, a.run_dir, c, a.batches, a.kl_batches, a.seed, a.dry_fire_penalty)
        b = out["checkpoints"][c]["batches"]
        print(c, json.dumps({k: round(float(np.mean([r[k] for r in b])), 4) for k in
                             ("mu_len_shot", "g_shot_head", "g_hook_head", "cos_shot_hook_head", "cos_shot_hook_trunk")}),
              json.dumps({k: round(v, 3) for k, v in out["checkpoints"][c]["cross_batch_cos"].items()}), flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
