# How many batches the aim gradient needs (2026-10-23)

This is a measurement only. Nothing was trained and no setting was changed.

`bench/aim_consistency.py` was run on three checkpoints of `phase3-3seed-hookfix`:

| checkpoint | phase | aim cosine in the probe |
|---|---|---|
| seed 1 `ckpt_1099` (4.5M) | successful phase | +0.15 / +0.17 |
| seed 3 `ckpt_0611` (2.5M) | just before the collapse | +0.05 / +0.05 |
| seed 3 `ckpt_0794` (3.25M) | in the collapse | -0.18 / -0.12 |

Each checkpoint was probed twice, with environment seeds 4242 (`run.sh`) and 5151 (`run2.sh`):

- 32 training batches per checkpoint and seed, against the run's ckpt_0000, with the run's own PPOConfig;
- the weights are fixed, and each batch holds 4096 decisions.

The aim loss's gradient at ratio 1 is taken separately for:

- the shot decisions (shot);
- the hook-launch-only decisions (hook);
- both together (both, the run's `shots+hooks` mask).

Each is taken on the aim's own parameters (head) and on the shared trunk.

`bench/consistency_stats.py` turns the per-batch gradients of the second probe into the statistics below. The gradients themselves (`runs/phase3-3seed/consistency/*.npz`, about 90 MB each) are not committed.

## Measures

- **rho = s2 / v**: the squared length of the expected gradient divided by one batch's noise variance.
  - The half-vs-half cosine of two means of N batches is N rho / (N rho + 1).
  - N_half = 1 / rho is the number of batches at which that cosine reaches 0.5.
  - The p-value comes from a sign-flip test.
  - The bootstrap interval in `stats_s5151.json` is biased upwards, because a resample repeats batches; it is not used here.
- **push**: the push the aim loss gives the aim mean, in the mean's own coordinates, towards the opponent.
  - It is taken on visible decisions as adv x (d logp / d mu), using only the rotational part.
  - It ignores how the network shares the change between states.
  - n95 is the number of batches for the summed push to be positive 95 % of the time, (1.645 sd / mean)^2.

## Findings

### 1. In parameter space the aim gradient shows no direction that replicates

With 32 batches, the sign-flip null puts the 95th percentile of rho at about 0.03 on the head and 0.012 on the trunk.

The first probe showed two clear directions:

- seed 3 at 2.5M, shot on the head: rho 0.042, pointing away from the opponent;
- seed 3 at 3.25M, hook on the head: rho 0.035, pointing towards the opponent.

The second probe did not reproduce either:

- the first fell to 0.003 (p 0.34);
- the second fell to 0.010 (p 0.20), and its direction flipped.

The second probe has one p < 0.05 out of 27 tests (seed 1, shot on the head, rho 0.045, p 0.025), which is about what chance gives. Adjacent batches agree no more than distant ones: the cosine is 0.00-0.12, and mostly below 0.05.

So rho is below about 0.03 on the head and 0.01 on the trunk. Half-vs-half would reach 0.5 only at more than about 30 batches on the head and more than about 80 on the trunk, so 32 batches is not enough to measure it. At 1, 4, 16 and 32 batches the cosine is 0 within noise for every mask, on all three checkpoints and in both probes.

Gradient norms per batch, mean ± sd, from the first probe:

| | head | trunk |
|---|---|---|
| shot | 0.06-0.08 ± 0.03 | 0.09-0.14 ± 0.05 |
| hook | 0.07-0.14 ± 0.03 | 0.12-0.24 ± 0.03-0.07 |

The norm of the mean over 32 batches is 17-29 % of a single batch's norm. Pure noise would give 1/√32 = 18 %.

### 2. In the aim mean's own coordinates the shot signal is consistent

The push from shot decisions towards the opponent is positive in all 6 probes:

| | push per batch, signal / noise | n95 |
|---|---|---|
| seed 1, 4.5M | 0.54 / 0.48 | 9 / 12 batches |
| seed 3, 2.5M | 0.65 / 0.40 | 7 / 17 batches |
| seed 3, 3.25M | 0.44 / 0.53 | 14 / 10 batches |

The hook push is -0.35 to +0.25 and is not consistent.

Shot decisions do carry information about where the opponent is, and about 10-17 batches (40k-70k decisions, 2.5k-4.5k shots) are needed before that information has a reliable sign. The difference from section 1 is that, after the network, this per-decision push spreads over many parameter directions, and none of them stands out within 32 batches.

There is no difference between seed 1's successful phase and seed 3's failing phases at this level. The shot push is about as strong in all three.

### 3. shots+hooks dilutes the shot signal

Push signal / noise per batch, shot vs both:

| | first probe | second probe |
|---|---|---|
| seed 1, 4.5M | 0.54 vs 0.01 | 0.48 vs 0.60 |
| seed 3, 2.5M | 0.65 vs 0.54 | 0.40 vs 0.28 |
| seed 3, 3.25M | 0.44 vs 0.32 | 0.53 vs 0.44 |

In 5 of 6 probes, adding the hooks lowers it. Measured against the shot gradient's mean over the other batches, a batch's both-mask gradient is no better aligned than its shot-only gradient (`vs_shot_reference_*`).

## Reading

The aim's update has a real but weak shot signal. Per batch it is about half its noise, so a single update moves the aim's direction towards the opponent only slightly more often than away from it. For the sign to be right 95 % of the time takes about 10-17 batches' worth of shot decisions. The hooks add noise rather than signal.

This fits the random walk with a weak positive drift seen across checkpoints (`phase3-seed-divergence`). It does not explain why seed 1 rather than seed 3 succeeded: at these checkpoints the signal is no stronger in seed 1.
