# Dry-fire penalty: short A/B (2026-10-24)

## The change

The one variable changed is `--dry-fire-penalty c`. Each dry fire, which the server reports when a press lands on an empty gun, now gives a reward of -c.

Everything else is the shots-only baseline of `phase3-ab-mask`:

- bounded aim mean, aim mask shots, aim_target_kl 0.015;
- frozen ckpt_0000 opponent, build-v4, env-timeout 600;
- the same PPO settings, observations, actions and map.

With the default `c = 0`, the reward is computed exactly as before, and the first 8 updates of `phase3-ab-mask/seed1` were reproduced bit for bit. So the control is the existing shots-only run with the same seed.

The scale came from `phase3-fire-audit`:

- The value-function bias on empty-gun presses is up to about 4e-4. The penalty has to be above it to give the empty-gun signal a known sign.
- In the control, the damage terms add up to about 0.1-0.35 per batch, and dry fires come to 500-870 per batch. With c = 5e-4 the penalty starts at about 0.3 per batch, about the size of the damage terms, and falls as the dry fires go away.

## Results (`bench/diagnose.py`, 64 matches per point, against the run's ckpt_0000)

### Seed 1 at 1M, two values of c

| | control | c = 5e-4 | c = 2.5e-4 |
|---|---|---|---|
| dry fires per match | 180 | 47 | 96 |
| gun empty | 0.70 | 0.30 | 0.47 |
| combat opportunity | 0.049 | 0.093 | 0.050 |
| hit rate | 0.012 | 0.015 | 0.014 |
| damage ratio | 1.36 | 1.58 | 1.07 |
| aim cos | +0.07 | -0.01 | -0.05 |

The smaller value was weaker on every count and was dropped.

### c = 5e-4, five seeds, at 2M

Seeds 1-3 were extended to 2M from their 1M checkpoints (`extend.sh`). Seeds 4 and 5 were run from scratch in both arms (`replicate.sh`).

Each cell is control → penalty:

| seed | dry fires per match | gun empty | combat opportunity | p(fire), line of fire, ready gun | p(fire), out of view, ready gun |
|---|---|---|---|---|---|
| 1 | 284 → 57 | 0.85 → 0.41 | 0.025 → 0.077 | 0.47 → 0.24 | 0.48 → 0.13 |
| 2 | 276 → 22 | 0.84 → 0.22 | 0.029 → 0.105 | 0.45 → 0.16 | 0.48 → 0.11 |
| 3 | 264 → 80 | 0.82 → 0.49 | 0.059 → 0.057 | 0.37 → 0.27 | 0.40 → 0.14 |
| 4 | 267 → 58 | 0.83 → 0.40 | 0.037 → 0.080 | 0.41 → 0.22 | 0.44 → 0.14 |
| 5 | 243 → 40 | 0.81 → 0.27 | 0.038 → 0.087 | 0.45 → 0.21 | 0.38 → 0.10 |

| seed | hit rate | damage dealt / taken | damage ratio | aim cos, visible | aim cos, shots in line of fire |
|---|---|---|---|---|---|
| 1 | 0.016 → 0.017 | 1.56 / 1.08 → 2.02 / 1.55 | 1.45 → 1.30 | +0.14 → -0.09 | +0.16 → -0.13 |
| 2 | 0.011 → 0.016 | 1.13 / 1.28 → 1.88 / 1.39 | 0.88 → 1.35 | +0.09 → -0.09 | +0.11 → -0.01 |
| 3 | 0.035 → 0.013 | 3.44 / 2.64 → 1.53 / 1.22 | 1.30 → 1.26 | +0.18 → +0.04 | +0.25 → +0.01 |
| 4 | 0.015 → 0.017 | 1.50 / 1.53 → 1.95 / 1.69 | 0.98 → 1.16 | -0.10 → +0.10 | -0.09 → +0.04 |
| 5 | 0.021 → 0.019 | 2.11 / 1.67 → 2.27 / 1.48 | 1.26 → 1.53 | +0.10 → +0.11 | +0.09 → +0.04 |

In all five seeds:

- dry fires fall by 70-92 %;
- the gun is empty about half as often;
- shots per match rise from 96-101 to 115-118.

The selectivity ratio, p(fire) in the line of fire with a ready gun over p(fire) out of view with a ready gun, goes from 0.9-1.2 to 1.5-2.1. p(fire) falls in both situations, more out of view.

Damage ratio, mean over the seeds: 1.17 → 1.32, better in 3 of 5.

Aim cosine on visible decisions, mean difference -0.07. It is lower in seeds 1-3 and higher or equal in seeds 4-5. On shots in the line of fire it is lower in 4 of 5, mean -0.11.

No run had non-finite values, and none was stopped.

## Why the aim moved (seeds 1-3, which were worse at 2M)

Three mechanisms were measured at 2M and none explains the drop (`aimcheck/`, `adv_decompose.py`):

- **Trunk drift:** the aim KL with the aim loss switched off is the same in both arms, 0.008-0.011.
- **The penalty leaking into the aim's advantage:** on shot decisions the penalty's part of the advantage has sd 3e-4, against 0.011-0.020 for the whole advantage, and its push on the aim is about 0.
- **Gradient clipping:** the grad norm per update is the same in both arms, 0.55-0.97.

The aim's own signal at 2M is lower in seed 3 (push signal over noise 0.46 → 0.05) and similar in seed 1 (0.65 → 0.52).

With seeds 4-5 the difference is not consistent, so it is treated as possibly seed noise, to be settled by the 5M validation (`phase3-fire-5m`).

## Counterfactual without training (`cf_maskempty_s*.json`)

The control's own 2M and 5M checkpoints were evaluated with their fire switched off on an empty gun:

- shots per match: 86-104 → 124;
- damage dealt: +4 % to +62 %;
- hit rate per shot: about the same.

That is the gain available from stopping dry fires alone, with the aim unchanged.
