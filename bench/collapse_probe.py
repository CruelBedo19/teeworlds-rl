#!/usr/bin/env python3
"""Why the long runs collapse to a uniform policy (measurement only, nothing is trained or saved).

    collapse_probe.py --server-dir ../teeworlds-0.7.5/build-v4 --run-dir runs/phase3-long/seed1 \
        --checkpoints ckpt_3663.pt ckpt_4883.pt ... --batches 4 --out probe.json

Per checkpoint the weights are fixed and training batches are collected against the run's ckpt_0000 as in
the run (its own PPOConfig and GAE). Per checkpoint:

- activations: for every ReLU layer, the share of units that are zero on every decision of the batches
  (dead), the mean share of active units per decision, and the mean activation;
- outputs: per discrete head, the spread of its logits across decisions (std of the centred logits) and
  its mean entropy; the aim mean's length; the value's spread;
- credit: the raw advantage's and return's spread, the reward per decision;
- gradient at ratio 1, on the discrete heads' rows of the output layer and on the trunk: the policy
  gradient -(adv * logp_discrete).mean() with the advantage normalised as in train(), and the entropy
  bonus's gradient -ent_coef * H.mean(); their norms and the cosine between them;
- weights: norms of each layer, and from the saved Adam state the median of sqrt(exp_avg_sq) per layer
  (a step is about lr * m / sqrt(v)).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl.env import VecTeeworldsEnv  # noqa: E402
from twrl.ppo import DISCRETE, PPOConfig, Trainer, load_policy  # noqa: E402

HEADS = ("direction", "jump", "fire", "hook")


def relus(policy):
    named = [("conv1", policy.conv[1]), ("conv2", policy.conv[3]), ("conv3", policy.conv[5]), ("grid_fc", policy.grid_fc[1]),
             ("vec_fc", policy.vec_fc[1]), ("trunk1", policy.trunk[1]), ("trunk2", policy.trunk[3])]
    return named


def grads(policy, batch, ent_coef):
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    logp, ent, _, _, aim_logp = policy.evaluate(batch["vec"], batch["codes"], batch["actions"])
    params = {"pi_discrete": policy.pi.weight, "trunk2": policy.trunk[2].weight, "trunk1": policy.trunk[0].weight}
    out = {}
    pg = torch.autograd.grad(-(adv * (logp - aim_logp)).mean(), list(params.values()), retain_graph=True)
    eg = torch.autograd.grad(-ent_coef * ent.mean(), list(params.values()))
    n = sum(DISCRETE)
    for (k, _), g, e in zip(params.items(), pg, eg):
        if k == "pi_discrete":
            g, e = g[:n], e[:n]
        g, e = g.flatten(), e.flatten()
        out[k] = {"pg": float(g.norm()), "ent": float(e.norm()), "ent_over_pg": float(e.norm() / (g.norm() + 1e-12)),
                  "cos": float(torch.dot(g, e) / (g.norm() * e.norm() + 1e-12))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(3)
    opponent, _ = load_policy(os.path.join(a.run_dir, "ckpt_0000.pt"))
    result = {}
    for name in a.checkpoints:
        path = os.path.join(a.run_dir, name)
        state = torch.load(path, map_location="cpu", weights_only=False)
        policy, _ = load_policy(path)
        cfg = PPOConfig(**state["config"])
        acts = {k: [] for k, _ in relus(policy)}
        hooks = [m.register_forward_hook(lambda m, i, o, k=k: acts[k].append(o.detach().flatten(1) > 0) if record[0] else None)
                 for k, m in relus(policy)]
        record = [False]
        rec = {"acts_mean": {k: [] for k in acts}, "logit_std": {h: [] for h in HEADS}, "entropy": {h: [] for h in HEADS},
               "aim_len": [], "value_sd": [], "adv_sd": [], "ret_sd": [], "reward_per_decision": [], "grads": []}
        alive_any = {k: None for k in acts}
        with VecTeeworldsEnv(cfg.num_envs, a.server_dir, seed=a.seed, map="lms1", grid_format="codes", timeout=600) as venv:
            trainer = Trainer(venv, policy, cfg, seed=a.seed, opponent=opponent)
            for _ in range(a.batches):
                batch = trainer.collect()
                for k in acts:
                    acts[k].clear()
                record[0] = True
                with torch.no_grad():
                    out, value = policy(batch["vec"], batch["codes"])
                record[0] = False
                for k, v in acts.items():
                    on = torch.cat(v)
                    alive_any[k] = on.any(0) if alive_any[k] is None else alive_any[k] | on.any(0)
                    rec["acts_mean"][k].append(float(on.float().mean()))
                logits, aim = policy._split(out)
                for h, z in zip(HEADS, logits):
                    rec["logit_std"][h].append(float((z - z.mean(1, keepdim=True)).std()))
                    rec["entropy"][h].append(float(torch.distributions.Categorical(logits=z).entropy().mean()))
                rec["aim_len"].append(float(aim.norm(dim=1).mean()))
                rec["value_sd"].append(float(value.std()))
                rec["adv_sd"].append(float(batch["adv"].std()))
                rec["ret_sd"].append(float(batch["returns"].std()))
                rec["reward_per_decision"].append(float(batch["rewards"].mean()))
                rec["grads"].append(grads(policy, batch, cfg.ent_coef))
        for h in hooks:
            h.remove()
        mean = lambda x: float(np.mean(x))  # noqa: E731
        opt = state["optimizer"]["state"]
        names = [n for n, _ in policy.named_parameters()]
        adam = {names[i]: float(s["exp_avg_sq"].sqrt().median()) for i, s in opt.items() if "exp_avg_sq" in s and names[i].endswith("weight")}
        r = {"update": state["update"], "agent_steps": state["agent_steps"],
             "dead_share": {k: float(1 - v.float().mean()) for k, v in alive_any.items()},
             "active_share": {k: mean(v) for k, v in rec["acts_mean"].items()},
             "logit_std": {h: mean(v) for h, v in rec["logit_std"].items()},
             "entropy": {h: mean(v) for h, v in rec["entropy"].items()},
             "aim_len": mean(rec["aim_len"]), "value_sd": mean(rec["value_sd"]), "adv_sd": mean(rec["adv_sd"]),
             "ret_sd": mean(rec["ret_sd"]), "reward_per_decision": mean(rec["reward_per_decision"]),
             "grads": {k: {m: mean([g[k][m] for g in rec["grads"]]) for m in rec["grads"][0][k]} for k in rec["grads"][0]},
             "weight_norm": {n: float(p.norm()) for n, p in policy.named_parameters() if n.endswith("weight")},
             "adam_sqrt_v_median": adam, "log_std": policy.log_std.tolist()}
        result[name] = r
        print(name, json.dumps({k: r[k] for k in ("update", "dead_share", "logit_std", "adv_sd", "grads")}), flush=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
