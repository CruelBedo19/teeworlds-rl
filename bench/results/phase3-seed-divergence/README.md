# Why seed 1 learned and seeds 2 and 3 did not (2026-10-23)

This is a measurement on the existing runs of `phase3-3seed-hookfix`. Nothing was trained.

## Data

- **`diag_fill_seed*.json`**: `bench/diagnose.py` on the 14 checkpoints per seed that were missing from `phase3-3seed-hookfix/diag_seed*.json`. Together they give all 21 checkpoints, one every 250k steps, with 64 episodes each against the seed's ckpt_0000.
- **`aim_probe_seed*.json`**: `bench/aim_probe.py` on all 21 checkpoints per seed. Each checkpoint was restored with its optimiser and random state, and 6 training batches were collected against ckpt_0000. The probe reports:
  - aim gradients from shot decisions and from hook-launch-only decisions, on the aim's own parameters (head) and on the shared trunk; their norms and their cosines within a batch and across batches;
  - one update replayed three ways on 3 of the batches: as in the run, with the aim loss off from the first minibatch, and with no KL stop. For each it records the aim KL per minibatch and after the update.
- **`summarise_probe.py`**: prints the per-checkpoint table of a probe file.

## Findings

### 1. The aim does not improve steadily; it swings from checkpoint to checkpoint

Aim cosine on visible decisions, per 250k steps:

- The standard deviation of the step-to-step change is 0.08-0.13.
- Seed 1 went down to -0.27 at 1.25M.
- Seed 3 went from -0.20 at 3.25M to +0.21 at 3.75M.
- Seed 2 went from +0.13 at 3.25M to -0.14 at 4.25M.

Mean over the second half (2.75M-5M):

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| aim cosine | 0.14 | 0.06 | 0.01 |
| line of fire | 0.31 | 0.30 | 0.21 |

### 2. Line of fire sets how many hits there are; aim sets who gets the better of them

Over the 21 checkpoints of each seed:

- r(hit rate, line of fire) is 0.94 / 0.89 / 0.83 for seeds 1 / 2 / 3.
- r(hit rate, aim cosine) is 0.67 / 0.58 / 0.37.

Pooled over all 63 checkpoints, the log damage ratio (dealt / taken) correlates with aim cosine at 0.65 and with line of fire at 0.45. Line of fire is close to symmetric: it raises the damage taken too.

### 3. Order of events

| seed | line of fire | aim |
|---|---|---|
| 1 | jumps at 1.75M (0.17 to 0.28) and stays at 0.22-0.39 | first consistently positive at 2.75-3M; 0.14-0.27 from 4M |
| 2 | jumps at 1.75M as well | rises from 2.25M, collapses at 4.25M (-0.14) |
| 3 | 0.26-0.33 at 2.25-2.5M (combat opportunity 0.11 / 0.09) | at 2.75M turns negative (-0.14 to -0.20) while visibility and line of fire fall back to 0.39-0.43 and 0.16-0.18 |

In seed 1 the hit rate rose at 1.75M (0.008 to 0.021) before the aim did, with aim cosine -0.01. Lagged correlations between the aim and the line of fire are weak and inconsistent in sign (-0.40 to +0.10), so the data do not show which of the two drives the other.

### 4. The aim gradient is noise-dominated at every checkpoint after ckpt_0000

- Across independent batches, the cosine between two batches' gradients is 0 ± 0.1, on the head and on the trunk. This holds for shot vs shot, hook vs hook and shot vs hook.
- Within a batch, the shot and hook gradients' cosine scatters between -0.37 and +0.44 around 0.
- The hook gradient is 1.2-2.5x the shot gradient in norm.
- A batch of 4096 decisions holds 5-11 rewarded decisions. In the second half seed 1 had 10.6 per batch, seed 2 8.5 and seed 3 7.3.

There is no measurable conflict between shot and hook gradients, and no measurable agreement either.

### 5. The aim KL is above target largely because of the shared trunk

Replaying an update with the aim loss off from the first minibatch, so the aim's output rows and log_std are fixed:

- The aim KL still reaches 0.006-0.010 by 0.5M and 0.013-0.027 from 3M on, on trunk drift alone.
- As run, the end-of-update aim KL is 0.03-0.06. With no stop it is 0.02-0.06, about the same.
- In the first epoch the per-minibatch KL is 0.002-0.01, with no jump at the start. The KL grows over the epochs and passes 1.5x target at minibatch 6-31 of 32.

The stop works as written: it acts on KL measured before the minibatch's step, and switches off the aim's own parameters. It cannot hold back the change the trunk makes to the aim through the other heads' and the value's gradients. The logged `aim_approx_kl` also averages the minibatches after the stop, so it sits above the threshold by construction.

## Reading

The aim's update has almost no signal per batch: there are a handful of rewarded decisions, and the gradient directions are uncorrelated across batches. Meanwhile the trunk moves the aim by about as much as the aim's own step.

Over 5M steps the aim behaves like a random walk with a weak positive drift. Where it ends up, together with whether positioning raised the line of fire, decides the final evaluation. Seed 1 had the highest line of fire and the most rewarded decisions late in training, which is consistent with, but does not prove, a feedback loop: more line of fire gives more hit events, which give more aim signal.
