#!/usr/bin/env python3
"""Fire credit audit: what the fire head's PPO update sees (measurement only, nothing is trained or saved).

    fire_audit.py --server-dir ../teeworlds-0.7.5/build-v4 --run-dir runs/x --checkpoints ckpt_0000.pt ... \
        --batches 24 --out audit.json

Per checkpoint the weights are fixed and training batches are collected against the run's ckpt_0000 exactly
as in the run (its own PPOConfig, GAE, advantages normalised over the batch). Per batch:

- credit: the advantage (normalised as in train(), and raw) and the return of every decision, split by the
  situation at the decision (line of fire with a ready gun, ready gun with the opponent out of view, ready
  gun visible without line of fire, empty gun, reload timer with ammo) and by what the decision did:
  no press, a press that shot and was followed by damage dealt within --hit-window decisions (useful
  fire), a shot without damage, a dry fire, a press lost in the timer. Per situation the gap between the
  advantage after pressing and after not pressing, and the batches needed for its sign to be right 95 %
  of the time.
- gradient: on the fire head's own logits (its two rows of the actor output) and on the shared trunk, the
  fire policy gradient -(adv * logp_fire).sum() / n and the entropy bonus's gradient on the fire head
  -ent_coef * H_fire.mean() (the run's ent_coef; its loss adds the entropy of every discrete head).
  Their norms, their ratio, the cosine between them, and across batches the consistency of the policy
  gradient's direction: rho = |expected gradient|^2 / one batch's noise variance, N_half = 1 / rho, with
  a sign-flip p-value.
- the fire logit's own derivatives as in fire_head_probe.py: policy gradient adv * (a - p) against the
  entropy bonus -ent_coef * z * p * (1 - p), per situation.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aim_probe import Recorder  # noqa: E402
from consistency_stats import rho  # noqa: E402
from twrl import protocol as P  # noqa: E402
from twrl.env import VecTeeworldsEnv  # noqa: E402
from twrl.ppo import DISCRETE, PPOConfig, Trainer, load_policy  # noqa: E402

V, EV = P.VEC_INDEX, {n: i for i, n in enumerate(P.EVENT_NAMES)}
FIRE = slice(sum(DISCRETE[:2]), sum(DISCRETE[:3]))  # rows of the fire logits in the actor output
OUTCOMES = ("no_press", "useful", "shot_miss", "dry", "lost_in_timer")


def situations(vec):
    alive = vec[:, V["alive"]] > 0
    ready = alive & (vec[:, V["ammo"]] > 0) & (vec[:, V["reload"]] <= 0)
    visible = alive & (vec[:, V["opp_visible"]] > 0)
    los = visible & (vec[:, V["opp_line_of_fire"]] > 0)
    return {"los_ready": los & ready, "invisible_ready": ~visible & ready, "visible_no_los_ready": visible & ~los & ready,
            "empty": alive & (vec[:, V["ammo"]] <= 0), "timer_with_ammo": alive & (vec[:, V["ammo"]] > 0) & (vec[:, V["reload"]] > 0),
            "all_alive": alive}


def outcomes(fire, ev, hit_window):
    """ev (T, b, events) for the learner rows; fire (T, b) bool -> outcome masks (T*b,) and a validity mask
    (a shot too close to the batch end to see its hit)."""
    T = ev.shape[0]
    shot = ev[..., EV["shots"]] > 0
    dry = ev[..., EV["dry_fires"]] > 0
    dmg = ev[..., EV["damage_dealt"]] > 0
    later = np.zeros_like(dmg)
    for t in range(T):
        later[t] = dmg[t:t + hit_window + 1].any(0)
    valid = np.ones_like(shot)
    valid[T - hit_window:] = ~shot[T - hit_window:]
    out = {"no_press": ~fire, "useful": fire & shot & later, "shot_miss": fire & shot & ~later,
           "dry": fire & dry & ~shot, "lost_in_timer": fire & ~shot & ~dry}
    return {k: torch.from_numpy(v.reshape(-1)) for k, v in out.items()}, torch.from_numpy(valid.reshape(-1)), \
        {"shots": int(shot.sum()), "hit_shots": int((shot & later & valid).sum()), "valid_shots": int((shot & valid).sum()),
         "damage_events": int(dmg.sum()), "dry_fires": int(dry.sum())}


def fire_grads(policy, batch, adv, ent_coef):
    """Fire policy gradient and fire entropy-bonus gradient, on the fire rows and the trunk."""
    trunk = [p for n, p in policy.named_parameters() if not n.startswith(("pi.", "v.", "log_std"))]
    out, _ = policy(batch["vec"], batch["codes"])
    d = torch.distributions.Categorical(logits=out[:, FIRE])
    a = batch["actions"][:, 2].long()
    res = {}
    for name, loss in (("pg", -(adv * d.log_prob(a)).sum() / len(adv)), ("ent", -ent_coef * d.entropy().mean())):
        policy.zero_grad()
        loss.backward(retain_graph=True)
        head = torch.cat([policy.pi.weight.grad[FIRE].reshape(-1), policy.pi.bias.grad[FIRE]])
        res[name] = (head.clone().double().numpy(), torch.cat([p.grad.reshape(-1) for p in trunk]).clone().double().numpy())
    policy.zero_grad()
    return res


def gap_stats(rows, a, b):
    """Per-batch advantage gap between outcome sets a and b; rows: list of per-batch dicts."""
    gaps, ts = [], []
    for r in rows:
        na, nb = sum(r[k]["n"] for k in a), sum(r[k]["n"] for k in b)
        if na < 2 or nb < 2:
            continue
        ma = sum(r[k]["s"] for k in a) / na
        mb = sum(r[k]["s"] for k in b) / nb
        va = sum(r[k]["ss"] for k in a) / na - ma ** 2
        vb = sum(r[k]["ss"] for k in b) / nb - mb ** 2
        gaps.append(ma - mb)
        ts.append((ma - mb) / math.sqrt(max(va, 1e-12) / na + max(vb, 1e-12) / nb))
    if not gaps:
        return None
    g = np.array(gaps)
    sd = g.std(ddof=1) if len(g) > 1 else float("nan")
    m = float(g.mean())
    return {"gap_mean": m, "gap_sd_over_batches": float(sd), "t_pooled": m / (sd / math.sqrt(len(g))) if sd > 0 else None,
            "t_per_batch_mean": float(np.mean(ts)), "share_positive": float(np.mean(g > 0)),
            "batches_for_95pct_sign": (1.645 * sd / abs(m)) ** 2 if m != 0 and sd > 0 else None, "batches": len(g)}


def probe(server_dir, run_dir, checkpoint, batches, seed, hit_window):
    state = torch.load(os.path.join(run_dir, checkpoint), map_location="cpu", weights_only=False)
    policy, _ = load_policy(os.path.join(run_dir, checkpoint))
    opponent, _ = load_policy(os.path.join(run_dir, "ckpt_0000.pt"))
    cfg = PPOConfig(**state["config"])
    rows, grads, logit = [], {"pg": ([], []), "ent": ([], [])}, {}
    with VecTeeworldsEnv(cfg.num_envs, server_dir, seed=seed, map="lms1", grid_format="codes", raw_info=True,
                         timeout=600) as venv:
        rec = Recorder(venv)
        trainer = Trainer(rec, policy, cfg, seed=seed, opponent=opponent)
        for _ in range(batches):
            rec.events = []
            batch = trainer.collect()
            T = len(rec.events)
            ev = np.stack(rec.events).reshape(T, -1, len(P.EVENT_NAMES))[:, trainer.rows]
            fire = batch["actions"][:, 2].numpy().reshape(T, -1) == 1
            outs, valid, counts = outcomes(fire, ev, hit_window)
            raw = batch["adv"]
            adv = (raw - raw.mean()) / (raw.std() + 1e-8)
            sit = situations(batch["vec"])
            row = {"counts": counts, "reward_events": int((batch["rewards"].abs() > 1e-9).sum()),
                   "return_sd": float(batch["returns"].std()), "raw_adv_sd": float(raw.std()), "sit": {}}
            for sname, sm in sit.items():
                cell = {}
                for oname, om in outs.items():
                    m = sm & om & valid
                    x, xr, ret = adv[m].double(), raw[m].double(), batch["returns"][m].double()
                    cell[oname] = {"n": int(m.sum()), "s": float(x.sum()), "ss": float((x * x).sum()), "raw_s": float(xr.sum()),
                                   "raw_ss": float((xr * xr).sum()), "ret_s": float(ret.sum())}
                row["sit"][sname] = cell
            with torch.no_grad():
                p = torch.softmax(trainer.policy(batch["vec"], batch["codes"])[0][:, FIRE], -1)[:, 1].double().clamp(1e-7, 1 - 1e-7)
            z = torch.log(p / (1 - p))
            a = batch["actions"][:, 2].double()
            pg = adv.double() * (a - p)
            ent = -cfg.ent_coef * z * p * (1 - p)
            for sname, sm in sit.items():
                d = logit.setdefault(sname, {"pg_sum": [], "pg_abs": [], "ent_sum": [], "p": [], "n": []})
                d["pg_sum"].append(float(pg[sm].sum()))
                d["ent_sum"].append(float(ent[sm].sum()))
                d["p"].append(float(p[sm].mean()) if sm.any() else float("nan"))
                d["n"].append(int(sm.sum()))
            g = fire_grads(trainer.policy, batch, adv, cfg.ent_coef)
            for k in grads:
                grads[k][0].append(g[k][0])
                grads[k][1].append(g[k][1])
            row["grad"] = {f"{k}_{part}": float(np.linalg.norm(g[k][i])) for k in g for i, part in ((0, "head"), (1, "trunk"))}
            row["grad"]["cos_pg_ent_head"] = float(g["pg"][0] @ g["ent"][0] / (np.linalg.norm(g["pg"][0]) * np.linalg.norm(g["ent"][0]) + 1e-20))
            rows.append(row)
            print(checkpoint, json.dumps({"shots": counts["shots"], "hit_shots": counts["hit_shots"], "dry": counts["dry_fires"],
                                          "rew": row["reward_events"], **{k: round(v, 4) for k, v in row["grad"].items()}}), flush=True)
    out = {"update": state["update"], "agent_steps": state["agent_steps"], "batches": batches, "per_batch": rows}
    # credit: pooled means per situation and outcome, and gaps
    pooled = {}
    for sname in rows[0]["sit"]:
        pooled[sname] = {}
        for oname in OUTCOMES:
            n = sum(r["sit"][sname][oname]["n"] for r in rows)
            if n == 0:
                pooled[sname][oname] = {"n_per_batch": 0.0}
                continue
            s = sum(r["sit"][sname][oname]["s"] for r in rows)
            ss = sum(r["sit"][sname][oname]["ss"] for r in rows)
            rs = sum(r["sit"][sname][oname]["raw_s"] for r in rows)
            pooled[sname][oname] = {"n_per_batch": n / batches, "adv_mean": s / n, "adv_sd": math.sqrt(max(ss / n - (s / n) ** 2, 0)),
                                    "raw_adv_mean": rs / n, "return_mean": sum(r["sit"][sname][oname]["ret_s"] for r in rows) / n}
        flat = [r["sit"][sname] for r in rows]
        pooled[sname]["gap_press_vs_not"] = gap_stats(flat, ("useful", "shot_miss", "dry", "lost_in_timer"), ("no_press",))
        pooled[sname]["gap_useful_vs_not"] = gap_stats(flat, ("useful",), ("no_press",))
        pooled[sname]["gap_shot_vs_not"] = gap_stats(flat, ("useful", "shot_miss"), ("no_press",))
        pooled[sname]["gap_dry_vs_not"] = gap_stats(flat, ("dry",), ("no_press",))
        pooled[sname]["gap_useful_vs_useless"] = gap_stats(flat, ("useful",), ("shot_miss", "dry", "lost_in_timer"))
    out["credit"] = pooled
    c = [r["counts"] for r in rows]
    out["shots_per_batch"] = float(np.mean([x["shots"] for x in c]))
    out["hit_shot_share"] = sum(x["hit_shots"] for x in c) / max(sum(x["valid_shots"] for x in c), 1)
    out["rewarded_fire_per_batch"] = float(np.mean([x["hit_shots"] for x in c]))
    out["reward_events_per_batch"] = float(np.mean([r["reward_events"] for r in rows]))
    # logit level
    out["logit"] = {}
    for sname, d in logit.items():
        pgs, ents = np.array(d["pg_sum"]), np.array(d["ent_sum"])
        sd = pgs.std(ddof=1)
        out["logit"][sname] = {"n_per_batch": float(np.mean(d["n"])), "p_fire": float(np.nanmean(d["p"])),
                               "pg_per_batch_mean": float(pgs.mean()), "pg_per_batch_sd": float(sd),
                               "ent_per_batch_mean": float(ents.mean()), "pg_snr_per_batch": float(pgs.mean() / sd) if sd > 0 else None,
                               "ent_over_pg_noise": float(abs(ents.mean()) / sd) if sd > 0 else None,
                               "ent_over_pg_mean": float(abs(ents.mean()) / abs(pgs.mean())) if pgs.mean() != 0 else None}
    # parameter level
    out["gradient"] = {}
    rng = np.random.default_rng(0)
    for part, i in (("head", 0), ("trunk", 1)):
        G = np.stack(grads["pg"][i])
        E = np.stack(grads["ent"][i])
        K = G @ G.T
        r = rho(K)
        null = [rho(K * np.outer(s, s)) for s in rng.choice([-1.0, 1.0], size=(2000, len(G)))]
        mean_g, mean_e = G.mean(0), E.mean(0)
        out["gradient"][part] = {
            "pg_norm_per_batch": float(np.linalg.norm(G, axis=1).mean()), "pg_norm_sd": float(np.linalg.norm(G, axis=1).std()),
            "pg_mean_norm": float(np.linalg.norm(mean_g)), "ent_norm_per_batch": float(np.linalg.norm(E, axis=1).mean()),
            "ent_mean_norm": float(np.linalg.norm(mean_e)),
            "ent_over_pg_per_batch": float(np.linalg.norm(E, axis=1).mean() / np.linalg.norm(G, axis=1).mean()),
            "ent_mean_over_pg_mean": float(np.linalg.norm(mean_e) / (np.linalg.norm(mean_g) + 1e-20)),
            "cos_mean_pg_mean_ent": float(mean_g @ mean_e / (np.linalg.norm(mean_g) * np.linalg.norm(mean_e) + 1e-20)),
            "rho": float(r), "p_sign_flip": float(np.mean(np.array(null) >= r)), "n_half": float(1 / r) if r > 0 else None,
            "signal_norm": float(math.sqrt(max(np.sum(mean_g ** 2) - (np.trace(K) - K.sum() / len(G)) / (len(G) - 1) / len(G), 0))),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="file names inside --run-dir")
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--hit-window", type=int, default=15, help="decisions after a shot in which damage dealt counts as its hit")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = {"run_dir": a.run_dir, "seed": a.seed, "hit_window": a.hit_window, "checkpoints": {}}
    for c in a.checkpoints:
        out["checkpoints"][c] = probe(a.server_dir, a.run_dir, c, a.batches, a.seed, a.hit_window)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
