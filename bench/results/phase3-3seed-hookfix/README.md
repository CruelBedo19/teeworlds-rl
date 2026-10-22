# 3 seeds: bounded aim mean + hook launches in the aim loss (2026-10-21/22)

Settings, identical for every seed (`seq.sh`):

- 5M learner steps (1221 updates of 32 envs x 128 decisions);
- `--aim gaussian --aim-mean bounded --aim-mask shots+hooks --aim-target-kl 0.015`;
- protocol 4 server (`build-v4`), `--env-timeout 600`;
- each seed against its own frozen ckpt_0000;
- evaluations at 0.5/1/1.5/2/3/4/5M, the pre-registered criteria and the watchdog of `bench/learn_pipeline.py`.

Seeds 1, 2 and 3 ran one after another, each about 33 min of training and 36 min with its evaluations. Then `bench/diagnose.py` ran on ckpt_0000 and the 0.5/1/2/3/4/5M checkpoints of each seed (`diag_seed*.json`, 64 episodes per checkpoint).

Earlier starts of the same experiment are kept under `runs/phase3-3seed/` and are not part of these results:

- `crashed-timeout/`: three seeds in parallel overloaded the box. One update took 30-34 s, past the bridge's 30 s timeout, so the servers closed the socket. Hence `--env-timeout`, which only decides when a stalled bridge gives up.
- `killed-slow/` and `aborted-parallel/`: still in parallel, at about 1.3 updates/min each; stopped in favour of running the seeds one after another.

## Against ckpt_0000 (64 matches per point)

| seed | point | damage ratio | reward | W/L/D | kills / deaths | damage dealt / taken | hit rate | aim cos |
|---|---|---|---|---|---|---|---|---|
| 1 | 2M | 1.25 | 0.056 | 3/0/61 | 0.047 / 0 | 2.20 / 1.77 | 0.022 | 0.03 |
| 1 | 3M | 1.34 | -0.004 | 0/1/63 | 0 / 0.016 | 2.34 / 1.75 | 0.024 | 0.18 |
| 1 | 4M | 1.49 | 0.020 | 0/0/64 | 0 / 0 | 3.11 / 2.09 | 0.032 | 0.14 |
| 1 | 5M | 1.56 | 0.128 | 6/0/58 | 0.094 / 0 | 4.73 / 3.03 | 0.047 | 0.18 |
| 2 | 3M | 1.24 | 0.023 | 1/0/63 | 0.016 / 0 | 1.84 / 1.48 | 0.019 | 0.12 |
| 2 | 4M | 1.16 | -0.025 | 0/2/62 | 0 / 0.031 | 2.28 / 1.97 | 0.023 | 0.04 |
| 2 | 5M | 0.98 | -0.017 | 0/1/63 | 0 / 0.016 | 2.16 / 2.20 | 0.022 | 0.08 |
| 3 | 3M | 1.09 | 0.033 | 2/0/62 | 0.031 / 0 | 1.14 / 1.05 | 0.012 | -0.09 |
| 3 | 4M | 1.11 | 0.003 | 0/0/64 | 0 / 0 | 1.59 / 1.44 | 0.016 | 0.09 |
| 3 | 5M | 1.18 | 0.020 | 1/0/63 | 0.016 / 0 | 1.52 / 1.28 | 0.015 | -0.07 |

ckpt_0000 hit rate is 0.010-0.014. Criteria at 5M:

| | 1 damage >= 1.25 | 2 reward > 0, W > L | 3 aim cos >= 0.10 or hit >= 1.5x | 4 trend | all four |
|---|---|---|---|---|---|
| seed 1 | yes (1.56) | yes | yes | yes (tau 0.71) | **yes** |
| seed 2 | no (0.98) | no | yes (hit 1.6x) | yes (tau 0.24) | no |
| seed 3 | no (1.18) | yes | no | no (aim cos fell) | no |

For comparison, the shots-only bounded run (seed 1, `phase3-learnability-v5-bounded`) reached, at 5M:

- damage ratio 1.17, reward 0.005, W/L 1/1;
- hit rate 0.018, aim cos 0.11;
- 2 of 4 criteria.

So 1 of 3 seeds passes. Seed 1 is the best run of Phase 3 so far, but the result does not replicate across seeds.

## Training

Means over the last quarter (updates 977-1221):

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| aim KL | 0.026 | 0.027 | 0.024 |
| aim off (share of minibatches) | 0.57 | 0.58 | 0.56 |
| log_std x | -1.05 | -1.11 | -1.11 |
| training aim cos | 0.21 | 0.03 | 0.03 |
| fire entropy | 0.55 | 0.56 | 0.56 |
| max 20-update mean aim KL (watchdog 0.05) | 0.03 | 0.034 | 0.032 |
| max grad norm (watchdog 20) | 5.99 | 4.96 | 8.48 |

There were no non-finite values in any seed and the watchdog never stopped a seed. In all three seeds the aim KL rises to about 1.7x its target over training, and the aim loss is switched off in more than half the minibatches.

## Fire / ammo (`diag_seed*.json`, ckpt_0000 to 5M)

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| p(fire), line of fire, gun ready | 0.51 to 0.33 | 0.51 to 0.49 | 0.50 to 0.28 |
| p(fire), opponent not visible, gun ready | 0.50 to 0.51 | 0.50 to 0.41 | 0.50 to 0.45 |
| gun empty | 0.85 to 0.82 | 0.85 to 0.84 | 0.85 to 0.80 |
| wasted presses | 0.87 to 0.86 | 0.87 to 0.87 | 0.87 to 0.85 |
| shots / dry fires per episode | 99/276 to 100/255 | 99/274 to 98/271 | 99/275 to 102/238 |
| line of fire | 0.22 to 0.39 | 0.18 to 0.30 | 0.21 to 0.19 |
| combat opportunity | 0.028 to 0.077 | 0.023 to 0.037 | 0.027 to 0.047 |
| aim cos on visible hook launches | 0.01 to 0.19 | 0.01 to 0.07 | 0.00 to -0.09 |

Firing does not become selective in any seed. If anything it drifts the wrong way: two seeds press less often with the opponent in the line of fire and the gun ready than with the opponent out of view. The empty gun and the wasted presses are unchanged, as expected from the ammo rules (see `phase3-hook-fire`).

Seed 1's gain comes from aim and position:

- aim cosine 0.18, hit rate 3.4x ckpt_0000;
- more time in the line of fire (0.39) and about 2.7x the combat opportunities.

It does not come from when it fires. In seed 1 the aim also points at the opponent on hook launches (0.19). In seed 3 the aim never turns towards the opponent, whether shooting or hooking.
