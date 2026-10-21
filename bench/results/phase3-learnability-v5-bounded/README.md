# Learnability v5: bounded aim mean, plus a second seed of v4 (2026-10-21)

Both runs: 5M learner steps against the frozen ckpt_0000, same protocol as v4 (lms1, 32 envs, stride 2,
compile, fire v3, shot mask, aim_target_kl 0.015). They ran in parallel (about 112 min each). The
regressions passed first (PPO 28, env 40, relay 2, GTest 30), and with default settings the new code
reproduces the first 30 updates of v4 bit for bit.

- **v5 bounded** (this folder): seed 1, `--aim-mean bounded`.
- **v4 seed 2** (`../phase3-learnability-v4-seed2`): seed 2, raw mean, i.e. the v4 configuration again.

Why the bounded mean: in v4 the raw aim mean grew from |mu| 0.002 to 1.7 at 2M and 3.2 at 5M, while
the std fell only from 0.61 to 0.45. Only the direction reaches the game, so the length is unidentified;
the aim became nearly deterministic (about 8 degrees of angular noise) and locked to one quadrant, and
the aim KL rose to 0.033 although the aim loss was off in 60 % of the minibatches. The bounded mean
maps mu to mu / sqrt(1 + |mu|^2).

## Each checkpoint against ckpt_0000 (both sides, 64 episodes)

| | v5 bounded | | | | | v4 seed 2 | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| steps | ratio | reward | W/L/D | hit | aim cos | ratio | reward | W/L/D | hit | aim cos |
| 0.5M | 1.03 | -0.015 | 1/2/61 | 0.0150 | +0.027 | 0.47 | -0.012 | 0/0/64 | 0.0076 | -0.177 |
| 1M | 1.14 | +0.002 | 0/0/64 | 0.0080 | -0.041 | 1.20 | +0.005 | 0/0/64 | 0.0154 | -0.125 |
| 1.5M | 1.04 | +0.001 | 0/0/64 | 0.0090 | +0.022 | 1.06 | +0.002 | 0/0/64 | 0.0141 | +0.014 |
| 2M | 0.88 | +0.012 | 1/0/63 | 0.0161 | +0.059 | 1.12 | +0.003 | 0/0/64 | 0.0162 | -0.160 |
| 3M | 1.22 | +0.004 | 0/0/64 | 0.0124 | +0.015 | 1.29 | +0.009 | 0/0/64 | 0.0193 | -0.005 |
| 4M | 1.23 | +0.006 | 0/0/64 | 0.0178 | +0.163 | 1.19 | +0.004 | 0/0/64 | 0.0133 | -0.152 |
| 5M | 1.17 | +0.005 | 1/1/62 | 0.0179 | +0.115 | 1.52 | +0.013 | 0/0/64 | 0.0181 | -0.117 |

Criteria at 5M: v5 bounded **2/4** (aim, trend: tau +0.43); v4 seed 2 **3/4** (ratio, aim through hit
rate 0.0181 >= 1.5 x 0.0102, trend: tau +0.62). Both fail criterion 2: no wins (kills are 0-2 in 64 episodes).

## Training (means per window, 0.5-1M / 2.75-3M / 4.75-5M)

| | aim KL | aim off | log_std | aim entropy |
|---|---|---|---|---|
| v4 seed 1 | 0.014 / 0.024 / 0.033 | 0.38 / 0.60 / 0.60 | -0.55 / -0.70 / -0.81 | 1.75 / 1.45 / 1.21 |
| v5 bounded | 0.011 / 0.016 / 0.021 | 0.26 / 0.46 / 0.56 | -0.58 / -0.91 / -1.05 | 1.69 / 1.03 / 0.74 |
| v4 seed 2 | 0.016 / 0.019 / 0.024 | 0.44 / 0.53 / 0.57 | -0.56 / -0.82 / -0.96 | 1.72 / 1.21 / 0.92 |

The other heads stay at KL 0.010-0.014; grad norm 0.6-0.85; no watchdog stop.

## Aim mean length (`mu_probe.txt`, `bench/mu_probe.py`)

The bounded mean stays at |mu| about 0.5 with about 35 degrees of angular noise and no locked quadrant
(mean over states near zero). Seed 2 with the raw mean repeats the v4 drift more slowly (|mu| 0.7 at
2M, 1.25 at 5M). v5's aim cosine on visible decisions is positive from 2M on and +0.16/+0.12 at 4M/5M,
the only run whose late aim does not turn negative; this is one seed.

## Reading

The bounded mean fixes the drift it was meant to fix and keeps the aim KL near the band (0.021 at the
end against 0.033). It does not make the agent win: reward stays near zero and kills are rare in every
run. Fire behaviour is unchanged (gun empty about 82-86 % of alive decisions, about 85 % of presses on an
empty gun, p(fire | line of fire, ready) no higher than p(fire | not visible, ready)), combat opportunities
remain 2-5 % of decisions, and about 93 % of hook launches carry no aim gradient.
