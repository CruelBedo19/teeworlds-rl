# ReDo test: seed 1 resumed from 15M (2026-10-28)

## Change

The one change is ReDo dormant-unit recycling: `--redo-every 25 --redo-tau 0`, at commit `3a723df`.

- Every 25 updates, each fully connected ReLU unit (grid_fc, vec_fc, trunk1, trunk2) that is zero on every decision of the batch gets fresh incoming weights.
- Its outgoing weights and its Adam moments are zeroed.
- The network output is unchanged by a recycle; this was checked to 0.0.

Seed 1 was resumed from `runs/phase3-long/seed1/ckpt_3663.pt` (15M) with otherwise the same settings and watchdog as long-run attempt 3. That run is the control: same start, no ReDo.

## What happened

- The run went from 15M to 27.8M (update 6786), in 106 minutes. The box was loaded by another user's CI.
- It was stopped by the watchdog: aim KL 0.40, mean of 20 updates, against the limit of 0.25.
- Exit 0, no infrastructure error.
- The 30M evaluation was not reached.

## Dead units: fixed

`probe_seed1.json`:

| | control trunk2 dead share | ReDo trunk2 dead share |
|---|---|---|
| 20M | 61 % | 0.8 % |
| 25M | 100 % | 0.4 % |

- Per 25 updates, 0-14 trunk2 units were recycled.
- Entropy stayed at 2.35-2.5, against the uniform 3.17 that the control reached at 23M.
- The policy never went uniform.

## Training metrics, means per 500 updates (control → ReDo)

| updates | entropy | aim cos | hit rate | kills per batch | aim KL |
|---|---|---|---|---|---|
| 3700 | 2.68 → 2.49 | .26 → .30 | .047 → .054 | .29 → .36 | .050 → .063 |
| 4700 | 2.75 → 2.43 | .19 → .30 | .034 → .051 | .11 → .28 | .053 → .075 |
| 5700 | 3.16 → 2.47 | .00 → .23 | .018 → .046 | .03 → .28 | .001 → .078 |
| 6700 | 3.16 → 2.35 | -.03 → .30 | .020 → .047 | .04 → .15 | .001 → .175 |

## Evaluation against ckpt_0000 (64 matches, control → ReDo)

| | kills / deaths | damage dealt / taken | hit rate | aim cos | wins |
|---|---|---|---|---|---|
| 20M | 0 / 0 → .016 / 0 | 1.94 / 1.52 → 2.84 / 1.59 | .019 → .028 | .13 → .12 | 0 → 1 |
| 25M | 0 / 0 → .062 / .016 | 1.73 / 1.34 → 3.78 / 2.56 | .017 → .038 | .01 → -.03 | 0 → 4 |
| 15M start, both arms | .078 / 0 | 5.25 / 2.70 | .052 | .37 | 5 |

## Reading

- ReDo removes the collapse: there are no dead units, no uniform policy, and training keeps learning to 27.8M.
- The 15M row was corrected on 2026-10-29. The first version had the uniform-random numbers; see the correction in `phase3-long/README.md`.
- It does **not** improve on the 15M start. The 20M and 25M evaluations are below 15M on every measure; the training metrics hold flat at about the 15M level.
- The aim's KL per update grows steadily (0.06 → 0.17 → 0.40) as log_std_y narrows (-2.1 → -2.6). This is a second instability, separate from the dead units.
- This is one seed only, so no best-model claim is made.
