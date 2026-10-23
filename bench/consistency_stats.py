#!/usr/bin/env python3
"""Significance of aim_consistency.py results from the saved per-batch gradients (.npz).

    consistency_stats.py a.npz b.npz ... --out stats.json

Per mask (shot, hook-only, both) and part (head, trunk, all):

- rho = s2 / v, the expected gradient's squared length over one batch's noise variance; N_half = 1 / rho
  batches give a half-vs-half cosine of 0.5. Its p-value is from a sign-flip test (under no common
  direction a batch's gradient is as likely negated), its 90 % interval from a bootstrap over batches.
- the projection of the ascent direction (minus the loss gradient) on the direction that raises the mean
  aim cosine on visible decisions: mean, t and the batches for the sum to be positive 95 % of the time.
- the cosine between adjacent batches and between batches at least 4 apart: batches that share episodes
  would make adjacent pairs agree more.
"""

import argparse
import json

import numpy as np

FLIPS, BOOT = 2000, 2000


def rho(K):
    """From the Gram matrix K = G G^T of the per-batch gradients."""
    B = len(K)
    msq = K.sum() / B ** 2
    v = (np.trace(K) - K.sum() / B) / (B - 1)
    return (msq - v / B) / v


def pair_cos(G, close):
    n = np.linalg.norm(G, axis=1)
    C = G @ G.T / np.outer(n, n)
    i, j = np.triu_indices(len(G), 1)
    sel = (j - i == 1) if close else (j - i >= 4)
    return float(C[i[sel], j[sel]].mean())


def stats(G, T, rng):
    B = len(G)
    K = G @ G.T
    r = rho(K)
    null = []
    for _ in range(FLIPS):
        s = rng.choice([-1.0, 1.0], size=B)
        null.append(rho(K * np.outer(s, s)))
    null = np.array(null)
    boot = []
    for _ in range(BOOT):
        i = rng.integers(0, B, B)
        boot.append(rho(K[np.ix_(i, i)]))
    boot = np.array(boot)
    t = T.mean(0)
    t = t / np.linalg.norm(t)
    proj = -(G @ t)
    m, sd = float(proj.mean()), float(proj.std(ddof=1))
    return {"rho": float(r), "p_sign_flip": float(np.mean(null >= r)), "rho_ci90": [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
            "n_half": float(1 / r) if r > 0 else None, "model_cos": {N: float(N * r / (N * r + 1)) if r > 0 else 0.0 for N in (1, 4, 16, 32, 64)},
            "norm_mean": float(np.linalg.norm(G, axis=1).mean()), "norm_sd": float(np.linalg.norm(G, axis=1).std()),
            "target_proj_mean": m, "target_t": m / (sd / np.sqrt(B)), "target_snr_per_batch": m / sd,
            "batches_for_95pct_positive": (1.645 * sd / m) ** 2 if m > 0 else None,
            "cos_mean_grad_target": float(-(G.mean(0) @ t) / np.linalg.norm(G.mean(0))),
            "cos_adjacent": pair_cos(G, True), "cos_far": pair_cos(G, False)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {}
    for f in a.files:
        z = np.load(f)
        parts = {"head": lambda nm: z[f"{nm}_head"], "trunk": lambda nm: z[f"{nm}_trunk"],
                 "all": lambda nm: np.concatenate([z[f"{nm}_head"], z[f"{nm}_trunk"]], 1)}
        tg = {"head": z["target_head"], "trunk": z["target_trunk"], "all": np.concatenate([z["target_head"], z["target_trunk"]], 1)}
        res = {}
        for mask in ("shot", "hook", "both"):
            for part, get in parts.items():
                G = get("shot") + get("hook") if mask == "both" else get(mask)
                res[f"{mask}_{part}"] = stats(G, tg[part], rng)
                s = res[f"{mask}_{part}"]
                print(f"{f.split('/')[-1]:32} {mask:5}{part:6} rho={s['rho']:+.4f} p={s['p_sign_flip']:.3f} "
                      f"ci=[{s['rho_ci90'][0]:+.4f},{s['rho_ci90'][1]:+.4f}] Nhalf={s['n_half'] and round(s['n_half'], 1)} "
                      f"tgt t={s['target_t']:+.2f} n95={s['batches_for_95pct_positive'] and round(s['batches_for_95pct_positive'], 1)} "
                      f"cosMT={s['cos_mean_grad_target']:+.2f} adj={s['cos_adjacent']:+.3f} far={s['cos_far']:+.3f}", flush=True)
        out[f] = res
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
