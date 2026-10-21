"""Length of the Gaussian aim's mean and the angular noise it leaves, over 160 states from random play
(env seed 77). Only the direction reaches the game, so a mean much longer than the std makes the aim
nearly deterministic whatever log_std says.

    mu_probe.py ../teeworlds-0.7.5/build-release runs/x/ckpt_0489.pt runs/x/ckpt_1221.pt
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from twrl.env import VecTeeworldsEnv
from twrl.evaluation import EVAL_MAP
from twrl.ppo import load_policy
torch.set_num_threads(4)
srv, cks = sys.argv[1], sys.argv[2:]
pols = [load_policy(c)[0] for c in cks]
rng = np.random.default_rng(0); V, C = [], []
with VecTeeworldsEnv(8, srv, seed=77, map=EVAL_MAP, grid_format="codes") as env:
    obs = env.reset()
    for t in range(40):
        if t % 2 == 0: V.append(obs["vec"][:, 0].copy()); C.append(obs["grid"][:, 0].copy())
        a = np.zeros((8, 2, 6), np.float32); a[..., 0] = rng.integers(-1, 2, (8, 2)); a[..., 2] = rng.integers(0, 2, (8, 2))
        a[..., 4:6] = rng.normal(size=(8, 2, 2)) * 200
        obs, *_ = env.step(a)
vec, codes = torch.from_numpy(np.concatenate(V)), torch.from_numpy(np.concatenate(C))
for c, p in zip(cks, pols):
    with torch.no_grad():
        mu = p.aim_mean(vec, codes).numpy(); std = p.log_std.exp().numpy()
    n = np.hypot(*mu.T)
    ang = np.degrees(np.arctan(std.mean() / np.maximum(n, 1e-6)))
    print(c.split("/")[-2][-8:], c.split("/")[-1], "mean", mu.mean(0).round(3), "median|mu| %.3f std %s median angular-noise %.0f deg" % (np.median(n), std.round(3), np.median(ang)))
