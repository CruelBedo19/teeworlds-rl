#!/usr/bin/env python3
"""How many batches the aim's gradient needs before its direction is reliable (measurement only).

    aim_consistency.py --server-dir ../teeworlds-0.7.5/build-v4 --run-dir runs/x --checkpoint ckpt_1099.pt \
        --batches 32 --out consistency.json

The checkpoint's weights are fixed; `--batches` training batches are collected against the run's ckpt_0000
exactly as in the run (its own PPOConfig, 4096 decisions each). Per batch, at ratio 1 with the advantages
normalised over the batch, the aim loss's gradient is taken separately from the shot decisions, from the
hook-launch-only decisions and from both (the "shots+hooks" mask), on the aim's own parameters (head) and
on the shared trunk. Nothing is stepped.

Consistency, per mask and part:

- half vs half: for k = 1, 2, 4, 8, 16 batches, the cosine between the mean gradient of k batches and
  that of k other batches, over random disjoint draws. With N batches the expected cosine is
  N s2 / (N s2 + v) (s2 = squared length of the expected gradient, v = noise variance of one batch's
  gradient); s2 and v are estimated from all batches and give the curve at any N, 32 and 64 included,
  and N_half = v / s2, the batches for which half-vs-half reaches 0.5.
- towards the opponent: per batch, the push the aim loss gives the aim mean towards the opponent on the
  decisions where the opponent is visible, d logp / d mu (rotational part only) times the advantage, and
  the projection of the parameter gradient onto the gradient of the mean aim cosine on those decisions.
  Their mean over batches and its noise give the batches needed for the sum to be positive 95 % of the
  time, (1.645 sd / mean)^2.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aim_probe import Recorder, flat_grads, masks  # noqa: E402
from twrl import protocol as P  # noqa: E402
from twrl.env import DEFAULT_REWARD_WEIGHTS, VecTeeworldsEnv  # noqa: E402
from twrl.ppo import PPOConfig, Trainer, load_policy  # noqa: E402

V = P.VEC_INDEX
KS = (1, 2, 4, 8, 16)


def opponent_dir(vec):
    d = torch.stack([vec[:, V["opp_dx"]] * P.VIEW_HALF_WIDTH, vec[:, V["opp_dy"]] * P.VIEW_HALF_HEIGHT], 1)
    return d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)


def batch_measures(trainer, batch, shot, hook):
    pol = trainer.policy
    vec, codes, actions = batch["vec"], batch["codes"], batch["actions"]
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(adv)
    alive = vec[:, V["alive"]] > 0
    vis = alive & (vec[:, V["opp_visible"]] > 0)
    u = opponent_dir(vec)
    g, push = {}, {}
    for name, m in (("shot", shot), ("hook", hook)):
        _, _, _, _, alp = pol.evaluate(vec, codes, actions)
        g[name] = flat_grads(pol, -(m.float() * adv * alp).sum() / n)
    # the aim mean's own push towards the opponent: d logp / d mu = (x - mu) / std^2, its rotational part
    with torch.no_grad():
        mu = pol.aim_mean(vec, codes)
        std = pol._std()
        x = actions[:, -2:]
        dlogp = (x - mu) / std ** 2
        mhat = mu / mu.norm(dim=1, keepdim=True).clamp(min=1e-6)
        rot = u - (u * mhat).sum(1, keepdim=True) * mhat
        per = adv * (dlogp * rot).sum(1)
        for name, m in (("shot", shot), ("hook", hook)):
            push[name] = float(per[m & vis].sum() / n)
    # the direction in parameter space that raises the mean aim cosine on visible decisions
    mu = pol.aim_mean(vec, codes)
    c = (mu * u).sum(1) / mu.norm(dim=1).clamp(min=1e-6)
    target = flat_grads(pol, -c[vis].mean() + 0 * pol.log_std.sum())  # log_std plays no part
    target = (-target[0], -target[1])
    pol.zero_grad()
    info = {"shot": int(shot.sum()), "hook_only": int(hook.sum()), "shot_visible": int((shot & vis).sum()),
            "hook_visible": int((hook & vis).sum()), "reward_events": int((batch["rewards"].abs() > 1e-9).sum()),
            "aim_cos_visible": float(c[vis].mean()), "push_shot": push["shot"], "push_hook": push["hook"]}
    return g, target, info


def consistency(G, rng, draws):
    """G (B, D) per-batch gradients -> half-vs-half cosines, norms, signal and noise."""
    B = len(G)
    mean = G.mean(0)
    v = float(((G - mean) ** 2).sum(1).sum() / (B - 1))  # noise variance of one batch's gradient (trace)
    s2 = float(mean @ mean) - v / B  # unbiased squared length of the expected gradient
    half = {}
    for k in KS:
        if 2 * k > B:
            continue
        cs = []
        for _ in range(draws):
            p = rng.permutation(B)
            a, b = G[p[:k]].mean(0), G[p[k:2 * k]].mean(0)
            cs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20)))
        half[k] = {"mean": float(np.mean(cs)), "sd": float(np.std(cs)), "p_positive": float(np.mean(np.array(cs) > 0))}
    model = {N: (N * s2 / (N * s2 + v) if s2 > 0 else 0.0) for N in (1, 4, 16, 32, 64, 128)}
    norms = np.linalg.norm(G, axis=1)
    return {"half_vs_half": half, "model_cos": model, "signal_sq": s2, "noise_var": v,
            "n_half": (v / s2) if s2 > 0 else None, "norm_mean": float(norms.mean()), "norm_sd": float(norms.std()),
            "mean_grad_norm": float(np.linalg.norm(mean))}


def positivity(x):
    x = np.asarray(x, dtype=np.float64)
    m, sd = float(x.mean()), float(x.std(ddof=1))
    return {"mean": m, "sd": sd, "snr_per_batch": m / sd if sd > 0 else None, "share_positive": float(np.mean(x > 0)),
            "batches_for_95pct_positive": (1.645 * sd / m) ** 2 if m > 0 else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", required=True, help="file name inside --run-dir")
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--dry-fire-penalty", type=float, default=0.0, help="as the run was trained (train_ppo.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-grads", help="also write the per-batch gradients to this .npz")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    path = os.path.join(a.run_dir, a.checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=False)
    policy, _ = load_policy(path)
    opponent, _ = load_policy(os.path.join(a.run_dir, "ckpt_0000.pt"))
    cfg = PPOConfig(**state["config"])
    grads = {"shot": ([], []), "hook": ([], [])}
    targets, infos = ([], []), []
    weights = {**DEFAULT_REWARD_WEIGHTS, "dry_fires": -a.dry_fire_penalty} if a.dry_fire_penalty else None
    with VecTeeworldsEnv(cfg.num_envs, a.server_dir, seed=a.seed, map="lms1", grid_format="codes", raw_info=True,
                         timeout=600, reward_weights=weights) as venv:
        rec = Recorder(venv)
        trainer = Trainer(rec, policy, cfg, seed=a.seed, opponent=opponent)
        for b in range(a.batches):
            rec.events = []
            batch = trainer.collect()
            shot, hook = masks(trainer, rec.events)
            g, target, info = batch_measures(trainer, batch, shot, hook)
            for name in grads:
                for part in (0, 1):
                    grads[name][part].append(g[name][part].numpy().astype(np.float64))
            for part in (0, 1):
                targets[part].append(target[part].numpy().astype(np.float64))
            infos.append(info)
            print(b, json.dumps(info), flush=True)
    if a.save_grads:
        np.savez_compressed(a.save_grads, **{f"{nm}_{lab}": np.stack(grads[nm][part]) for nm in grads
                                             for part, lab in ((0, "head"), (1, "trunk"))},
                            target_head=np.stack(targets[0]), target_trunk=np.stack(targets[1]))
    rng = np.random.default_rng(a.seed)
    out = {"run_dir": a.run_dir, "checkpoint": a.checkpoint, "update": state["update"],
           "agent_steps": state["agent_steps"], "batches": a.batches, "per_batch": infos, "masks": {}}
    for name in ("shot", "hook", "both"):
        res = {}
        for part, label in ((0, "head"), (1, "trunk"), (None, "all")):
            def stack(nm):
                if part is None:
                    return np.concatenate([np.stack(grads[nm][0]), np.stack(grads[nm][1])], 1)
                return np.stack(grads[nm][part])
            G = stack("shot") + stack("hook") if name == "both" else stack(name)
            T = (np.concatenate([np.stack(targets[0]), np.stack(targets[1])], 1) if part is None
                 else np.stack(targets[part])).mean(0)
            T = T / (np.linalg.norm(T) + 1e-20)
            res[label] = consistency(G, rng, a.draws)
            # ascent direction is minus the loss gradient; its projection on the aim-cosine direction
            res[label]["towards_target"] = positivity(-(G @ T))
            res[label]["cos_mean_grad_target"] = float(-(G.mean(0) @ T) / (np.linalg.norm(G.mean(0)) + 1e-20))
        res["push_mu"] = positivity([i["push_shot"] + i["push_hook"] if name == "both" else i[f"push_{name}"]
                                     for i in infos])
        out["masks"][name] = res
    # does adding the hook gradient help or dilute the shot direction? cosine of each batch's both-mask
    # gradient with the shot gradient's mean over the other batches (no shared noise)
    for label, part in (("head", 0), ("trunk", 1)):
        S, H = np.stack(grads["shot"][part]), np.stack(grads["hook"][part])
        tot = S.sum(0)
        cs_shot, cs_both = [], []
        for i in range(len(S)):
            ref = (tot - S[i]) / (len(S) - 1)
            for lst, gi in ((cs_shot, S[i]), (cs_both, S[i] + H[i])):
                lst.append(float(gi @ ref / (np.linalg.norm(gi) * np.linalg.norm(ref) + 1e-20)))
        out[f"vs_shot_reference_{label}"] = {"shot_only": float(np.mean(cs_shot)), "shots_plus_hooks": float(np.mean(cs_both))}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for name, res in out["masks"].items():
        print(name, {lab: {"half": {k: round(v["mean"], 3) for k, v in res[lab]["half_vs_half"].items()},
                           "n_half": res[lab]["n_half"] and round(res[lab]["n_half"], 1),
                           "n95": res[lab]["towards_target"]["batches_for_95pct_positive"]} for lab in ("head", "trunk", "all")},
              "push_mu", {k: round(v, 4) if isinstance(v, float) else v for k, v in res["push_mu"].items()})


if __name__ == "__main__":
    main()
