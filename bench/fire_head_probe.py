#!/usr/bin/env python3
"""What pushes the fire head: the policy gradient against the entropy bonus (measurement only).

    fire_head_probe.py --server-dir ../teeworlds-0.7.5/build-v4 --opponent runs/x/ckpt_0000.pt \
        --checkpoints runs/x/ckpt_0489.pt ... --batches 20 --out probe.json

Collects training batches exactly as the trainer does (32 envs x 128 steps against the frozen opponent,
advantages normalised over the batch) and, for the fire head's logit difference z = logit(fire) -
logit(no fire) with p = sigmoid(z), compares two per-decision derivatives of the PPO objective at ratio 1:

  policy gradient   adv * (a - p)              (a = 1 when the decision pressed fire)
  entropy bonus     ent_coef * dH/dz = -ent_coef * z * p * (1 - p)

split by situation: opponent in the line of fire with the gun ready, gun ready and opponent out of view,
gun empty. For each it reports the mean of both and the policy gradient's signal-to-noise ratio per
update (its sum over one batch's decisions in that situation divided by the noise of that sum). A
positive policy gradient in a situation means the advantages favour pressing there.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import protocol as P  # noqa: E402
from twrl.env import VecTeeworldsEnv  # noqa: E402
from twrl.ppo import PPOConfig, Trainer, load_policy  # noqa: E402

V = P.VEC_INDEX


def situations(vec):
    alive = vec[:, V["alive"]] > 0
    ready = alive & (vec[:, V["ammo"]] > 0) & (vec[:, V["reload"]] <= 0)
    visible = alive & (vec[:, V["opp_visible"]] > 0)
    los = visible & (vec[:, V["opp_line_of_fire"]] > 0)
    return {"all_alive": alive, "los_ready": los & ready, "invisible_ready": ~visible & ready & alive,
            "visible_no_los_ready": visible & ~los & ready, "empty": alive & (vec[:, V["ammo"]] <= 0),
            "timer_with_ammo": alive & (vec[:, V["ammo"]] > 0) & (vec[:, V["reload"]] > 0)}


def probe(server_dir, checkpoint, opponent_path, batches, seed):
    policy, _ = load_policy(checkpoint)
    opponent, _ = load_policy(opponent_path)
    cfg = PPOConfig(num_envs=32, steps=128)
    parts = {}
    with VecTeeworldsEnv(32, server_dir, seed=seed, map="lms1", grid_format="codes", raw_info=True) as venv:
        trainer = Trainer(venv, policy, cfg, seed=seed, opponent=opponent)
        for _ in range(batches):
            batch = trainer.collect()
            adv = batch["adv"]
            adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).numpy().astype(np.float64)
            p_fire = trainer.policy.head_probs(batch["vec"], batch["codes"])[2][:, 1].numpy().astype(np.float64)
            p_fire = np.clip(p_fire, 1e-7, 1 - 1e-7)
            z = np.log(p_fire / (1 - p_fire))
            a = batch["actions"][:, 2].numpy().astype(np.float64)
            pg = adv * (a - p_fire)
            ent = -cfg.ent_coef * z * p_fire * (1 - p_fire)
            for name, m in situations(batch["vec"]).items():
                m = m.numpy()
                d = parts.setdefault(name, {"n": [], "pg_sum": [], "pg": [], "ent": [], "p": [], "adv_fire": [], "adv_nofire": []})
                d["n"].append(int(m.sum()))
                d["pg_sum"].append(float(pg[m].sum()))
                d["pg"].append(pg[m])
                d["ent"].append(ent[m])
                d["p"].append(p_fire[m])
                d["adv_fire"].append(adv[m & (a == 1)])
                d["adv_nofire"].append(adv[m & (a == 0)])
    out = {}
    for name, d in parts.items():
        pg, ent, p = np.concatenate(d["pg"]), np.concatenate(d["ent"]), np.concatenate(d["p"])
        af, an = np.concatenate(d["adv_fire"]), np.concatenate(d["adv_nofire"])
        n_per = float(np.mean(d["n"]))
        if len(pg) < 2:
            out[name] = {"n_per_update": n_per}
            continue
        out[name] = {
            "n_per_update": n_per, "share_of_batch": n_per / (cfg.num_envs * cfg.steps),
            "p_fire_mean": float(p.mean()), "p_fire_sd_over_states": float(p.std()),
            "pg_mean": float(pg.mean()), "ent_mean": float(ent.mean()),
            "pg_per_update_snr": float(pg.mean() * n_per / (pg.std() * math.sqrt(n_per))),
            "pg_sign_consistency": float(np.mean(np.sign(d["pg_sum"]) == np.sign(np.sum(d["pg_sum"])))),
            "adv_when_fired": float(af.mean()) if len(af) else None,
            "adv_when_not_fired": float(an.mean()) if len(an) else None,
            "adv_gap_sem": float(math.sqrt(af.var() / max(len(af), 1) + an.var() / max(len(an), 1))) if len(af) and len(an) else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    out = {"opponent": a.opponent, "batches": a.batches, "seed": a.seed, "ent_coef": PPOConfig().ent_coef, "checkpoints": {}}
    for c in a.checkpoints:
        out["checkpoints"][c] = probe(a.server_dir, c, a.opponent, a.batches, a.seed)
        print(c, json.dumps({k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                             for k, v in out["checkpoints"][c].items()}), flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
