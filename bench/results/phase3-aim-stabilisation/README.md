# Gaussian aim stabilisation (2026-10-20/21)

Problem: in the v3 learnability run (2M steps vs frozen ckpt_0000) the shot-masked Gaussian aim's KL per
update rose from 0.009 to 0.072 and its clip fraction from 0.11 to 0.35, while the other heads stayed
at about 0.01 KL. Goal: aim KL about 0.01-0.02 without killing the aim gradient or changing the other heads.

Two settings were added to `PPOConfig` (both off by default; with defaults the first 30 updates are
bit-identical to v3, see `determinism_vs_v3_first30.jsonl`):

- `aim_lr_scale`: the aim's own parameters (its two rows of the output layer and `log_std`) step this
  fraction of the normal Adam step. The shared trunk keeps the full rate.
- `aim_target_kl`: once a minibatch's aim KL (on shot decisions) passes 1.5x this, the aim loss is left out
  and the aim's own parameters are held still for the rest of that update. The other heads and the value
  keep training. `aim_off` in the metrics is the fraction of minibatches with the aim off.

## Short trials

Each trial: 1.5M learner steps, same protocol as v3 (lms1, 32 envs, seed 1, stride 2, compile, frozen
ckpt_0000, fire v3, shot mask). The three ran in parallel, about 14 minutes each. Means per 250k-step
window are in `tune_windows.txt`; 1.25-1.5M:

| setting | aim KL | aim clip | aim off | log_std | train aim cos | train hit |
|---|---|---|---|---|---|---|
| v3 (none) | 0.048, rising | 0.30 | - | -0.81 | +0.094 | 0.014 |
| aim_target_kl 0.015 | 0.015, flat | 0.15 | 0.43 | -0.68 | +0.064 | 0.012 |
| aim_lr_scale 0.3 | 0.008 | 0.09 | 0 | -0.64 | +0.051 | 0.017 |
| scale 0.5 + target 0.02 | 0.008 | 0.09 | 0.10 | -0.58 | -0.055 | 0.008 |

Chosen: **aim_target_kl 0.015** (aim_lr_scale 1). It is the only setting that holds the aim KL inside
the 0.01-0.02 target over the whole trial, and its training aim cosine turns positive at the same point
as v3. The aim still trains on roughly 57 % of the minibatches. aim_lr_scale 0.3 undershoots the target
(0.008), and the combination was the weakest on aim cosine. These are single-seed 1.5M runs, so the
differences in aim cosine and hit rate between the settings are within noise; the choice rests on the KL behaviour.
