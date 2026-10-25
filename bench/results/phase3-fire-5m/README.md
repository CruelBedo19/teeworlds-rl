# Dry-fire penalty: 5M validation, seeds 1-3 (2026-10-24)

## Setup

`seq.sh` trains the shots-only baseline of `phase3-ab-mask` with one change, `--dry-fire-penalty 0.0005`. Code is at commit `68651f2` (`commit`).

The control is the existing `phase3-ab-mask` seeds 1-3, with the same seeds and settings and no penalty. It was not retrained.

How the runs went:

- Seeds were run one after another, each to 1221 updates (5.0M agent steps).
- Each run took 30-45 minutes.
- Every run exited with 0. None was stopped, and there were no non-finite values or infrastructure errors.

Evaluation:

- `learn_pipeline.py` evaluated each run at 0.5, 1, 1.5, 2, 3, 4 and 5M, with 64 matches per point against the run's ckpt_0000.
- `diagnose.py` measured fire and ammo behaviour at 0, 1, 2, 3, 4 and 5M (`diag_seed*.json`), with 64 matches each.

## Results at 5M, control → penalty

Fire and ammo:

| seed | dry fires per match | gun empty | combat opportunity | p(fire), line of fire, ready gun | p(fire), out of view, ready gun | selectivity |
|---|---|---|---|---|---|---|
| 1 | 228 → 79 | 0.79 → 0.55 | 0.066 → 0.052 | 0.30 → 0.32 | 0.44 → 0.18 | 0.68 → 1.8 |
| 2 | 326 → 36 | 0.88 → 0.32 | 0.035 → 0.093 | 0.49 → 0.21 | 0.61 → 0.12 | 0.80 → 1.7 |
| 3 | 241 → 88 | 0.81 → 0.58 | 0.056 → 0.055 | 0.36 → 0.34 | 0.45 → 0.17 | 0.80 → 2.0 |

Selectivity is p(fire) in the line of fire with a ready gun, divided by p(fire) out of view with a ready gun.

Match outcome:

| seed | criteria | damage ratio | damage dealt / taken | hit rate | kills / deaths | W/L | aim cos, visible | aim cos, shots in line of fire |
|---|---|---|---|---|---|---|---|---|
| 1 | 3/4 → 4/4 | 1.53 → 2.18 | 2.77 / 1.81 → 3.14 / 1.44 | 0.027 → 0.027 | 0.016 / 0 → 0.062 / 0.016 | 1/0 → 4/1 | +0.20 → +0.27 | +0.11 → +0.36 |
| 2 | 3/4 → 4/4 | 1.34 → 1.47 | 2.39 / 1.78 → 2.36 / 1.61 | 0.028 → 0.020 | 0 / 0 → 0.047 / 0 | 0/0 → 3/0 | +0.17 → -0.01 | +0.15 → +0.04 |
| 3 | 4/4 → 3/4 | 1.58 → 1.37 | 3.69 / 2.33 → 1.95 / 1.42 | 0.036 → 0.017 | 0.062 / 0 → 0.031 / 0.031 | 4/0 → 2/2 | +0.14 → +0.01 | +0.07 → +0.00 |
| mean | | 1.48 → 1.68 | 2.95 / 1.97 → 2.48 / 1.49 | 0.030 → 0.021 | 0.026 / 0 → 0.047 / 0.016 | | +0.17 → +0.09 | |

Shots per match: 97 → 116.

Aim cos on visible decisions, at 0.5, 1, 1.5, 2, 3, 4 and 5M:

| | control | penalty |
|---|---|---|
| seed 1 | +0.05 +0.07 +0.10 +0.14 +0.26 +0.15 +0.20 | +0.02 -0.01 -0.17 +0.25 +0.10 +0.06 +0.27 |
| seed 2 | -0.06 +0.01 +0.10 +0.09 +0.21 +0.19 +0.17 | -0.08 -0.05 +0.11 -0.02 -0.02 +0.09 -0.01 |
| seed 3 | -0.05 +0.08 +0.14 +0.18 +0.16 +0.14 +0.14 | -0.07 -0.02 +0.11 +0.03 -0.04 +0.17 +0.01 |

Mean over 3, 4 and 5M: control +0.18, penalty +0.07.

## Reading

### The fire change holds at 5M in every seed

- Dry fires fall by 63-89 %.
- The gun is empty less often.
- Selectivity rises from 0.7-0.8, which is anti-selective, to 1.7-2.0.
- More rounds are fired, 116 against 97 per match.

Part of the gain fades after 2M. In seeds 1 and 3, dry fires rise again from their lowest point: 19 → 79 and 26 → 88 per match. The gun is empty more often again: 0.17 → 0.55 and 0.22 → 0.58. Seed 2 stays low.

### The aim is worse in two of three seeds

- At 5M the aim is lower in seeds 2 and 3. Their aim cosine stays near 0 from 2M to 5M, against +0.14 to +0.21 for the control.
- The hit rate per shot is lower in seeds 2 and 3 as well.
- Seed 1 is better on every measure.

Counting the 2M A/B (`phase3-fire-ab`), the aim is lower with the penalty in 5 of 8 seed comparisons. Seeds 4 and 5 were not continued to 5M.

### Match outcome is mixed

- The damage ratio is higher on average: 1.48 → 1.68, better in 2 of 3 seeds.
- Most of that comes from taking less damage, 1.97 → 1.49, not from dealing more: 2.95 → 2.48.
- Kills went up on average, but a kill happens in fewer than 1 match in 15, so the kill counts here are too small to read.

### Verdict

By the success rule, a serious loss of aim means the experiment fails, and here the aim is clearly worse in 2 of 3 seeds at 5M. So the dry-fire penalty is **not adopted as the new baseline**, and no checkpoint is marked best.

Fire did become learnable with this change; the problem is the cost to the aim.

## Why the aim moved

`aimcheck/run.sh` probes each 5M checkpoint of both arms without training, using `adv_decompose.py`:

- 16 batches per checkpoint;
- the run's own PPOConfig, against the run's ckpt_0000.

It measures the aim's push towards the opponent on visible shot decisions.

| seed | control: signal / noise per batch | penalty: signal / noise per batch | penalty: share of the advantage spread on shots |
|---|---|---|---|
| 1 | 0.98 | 0.36 | sd 4e-4 of 0.018 |
| 2 | 0.29 | 0.55 | sd 3e-4 of 0.009 |
| 3 | 0.38 | 0.60 | sd 4e-4 of 0.009 |

- The penalty's own share of the advantage on shot decisions stays tiny, as it was at 2M.
- The aim's learning signal is **not** weaker in the two seeds whose aim is worse. Seeds 2 and 3 have a higher signal over noise than their controls, and seed 1, the one with the better aim, has a lower one.

So at 5M, the lower aim in seeds 2 and 3 is not explained by:

- a weaker aim gradient;
- trunk drift;
- the penalty in the aim's advantage;
- gradient clipping.

The last three were checked at 2M (`phase3-fire-ab`).

What is left is the random walk with a weak drift seen before in the aim (`phase3-seed-divergence`, `phase3-aim-consistency`). With three seeds per arm, that random walk cannot be told apart from a real but indirect effect of the penalty. One candidate for such an effect is that the penalty changes which states the shots come from. That is not measured here.

## Seeds 4 and 5 to 5M: the aim loss is real (2026-10-25)

Seeds 4 and 5 of both arms of `phase3-fire-ab` were continued from 2M to 5M (`seeds45/extend5m.sh`), one run at a time, and all four exited with 0.

Two of the four 5M diagnoses first failed: they ran in parallel and both tried to bind the same server port. That was an infrastructure failure, not a learning one. They were re-run one at a time (`seeds45/rediag5m.sh`).

Every number below comes from `diagnose.py` at 3, 4 and 5M, 64 matches per point against the run's ckpt_0000 (`seeds45/five_seed_table.txt`).

| seed | aim cos, visible, mean of 3-5M: control | penalty | difference |
|---|---|---|---|
| 1 | +0.20 | +0.14 | -0.06 |
| 2 | +0.19 | +0.02 | -0.17 |
| 3 | +0.15 | +0.04 | -0.10 |
| 4 | +0.25 | +0.02 | -0.24 |
| 5 | +0.22 | -0.36 | -0.58 |

The aim is lower with the penalty in 5 of 5 seeds. The mean difference is -0.23, with a standard error of 0.09.

In seed 5 the penalty arm aims away from the opponent at 3, 4 and 5M: -0.32, -0.31, -0.46.

Means over the five seeds at 5M, control → penalty:

- dry fires per match 263 → 53;
- gun empty 0.83 → 0.45;
- selectivity 0.80 → 1.59;
- shots per match 98 → 117;
- hit rate 0.030 → 0.016;
- damage dealt / taken 2.90 / 2.02 → 1.87 / 1.33;
- damage ratio 1.47 → 1.35;
- kills per match 0.028 → 0.028.

## Verdict

The aim cost is real. The dry-fire penalty fixes the fire behaviour, but it costs about half the hit rate and most of the aim, and it does not raise the damage ratio. **It is rejected.** The shots-only control (`phase3-ab-mask`) stays the baseline.

The cause is still not known. At 5M the aim's learning signal was not weaker in the worse seeds, and the penalty's own part of the aim's advantage is tiny. The aim loss uses shot decisions only, so the remaining candidate is the change in which states the shots come from. That has not been measured yet.
