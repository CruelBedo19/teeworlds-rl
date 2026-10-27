# Overnight long run: shots-only baseline past 5M (2026-10-26/27)

See `PLAN.md` for the plan and the three attempts. The training settings are those of `phase3-ab-mask`; only the length changed.

The code is at:

- `5450545`, the plan;
- `e06513f`, `learn_pipeline.py --aim-kl-limit`;
- `d60d419`, attempt 3.

## What ran

Seeds 3 and 1 of `phase3-ab-mask` were resumed from 5M, one at a time.

| | stopped | where | reason |
|---|---|---|---|
| attempt 1 | seed 3 | 7M | the watchdog's aim-KL rule (20-update mean above 0.05), after one spike |
| attempt 2 | seed 3 | 8.9M | the same rule, limit 0.10 |
| attempt 2 | seed 1 | 15.6M | the same rule, limit 0.10 |
| attempt 3 | seed 3 | 30.7M (update 7507) | "aim log_std collapse": log_std_y below -4.5 |
| attempt 3 | seed 1 | 31.0M (update 7564) | the same |

Attempt 3 used a limit of 0.25. Every resume started from the last checkpoint, with the metrics cut back to it. All processes exited with 0, and there was no infrastructure failure.

## Evaluation against ckpt_0000 (64 matches per point, `learn_pipeline.py`)

| seed | point | kills / deaths | damage dealt / taken | hit rate | aim cos | wins | draws |
|---|---|---|---|---|---|---|---|
| 1 | 5M (`phase3-ab-mask`) | 0.016 / 0 | 2.77 / 1.81 | 0.027 | +0.20 | 1 | |
| 1 | 10M | 0.125 / 0.031 | 4.44 / 2.64 | 0.047 | +0.24 | 7 | 0.88 |
| 1 | **15M** | **0.125 / 0.047** | **5.75 / 3.41** | **0.057** | **+0.35** | 8 | 0.83 |
| 1 | 20M | 0.016 / 0 | 2.62 / 1.62 | 0.026 | +0.14 | 1 | 0.98 |
| 1 | 25M | 0.016 / 0.016 | 1.89 / 1.59 | 0.018 | +0.01 | 1 | 0.97 |
| 1 | 30M | 0 / 0 | 2.05 / 1.25 | 0.019 | +0.01 | 0 | 1.00 |
| 3 | 5M (`phase3-ab-mask`) | 0.062 / 0 | 3.69 / 2.33 | 0.036 | +0.14 | 4 | |
| 3 | 10M | 0.016 / 0 | 2.34 / 2.08 | 0.024 | +0.04 | 1 | 0.98 |
| 3 | 15M | 0.062 / 0 | 3.16 / 2.19 | 0.031 | +0.12 | 4 | 0.94 |
| 3 | **20M** | 0 / 0.016 | **3.62 / 1.80** | 0.037 | **+0.29** | 0 | 0.98 |
| 3 | 25M | 0 / 0.016 | 1.44 / 1.16 | 0.015 | -0.10 | 0 | 0.98 |
| 3 | 30M | 0.016 / 0 | 2.67 / 1.42 | 0.026 | 0.00 | 1 | 0.98 |

## Reading

### 1. Past 5M the agent keeps improving, for a while

- Seed 1 peaks at 10-15M:
  - kills 8×;
  - hit rate 2×;
  - aim cos 0.20 → 0.35;
  - damage dealt 2×.
- Seed 3 peaks at 20M:
  - aim cos 0.14 → 0.29;
  - damage ratio 2.0;
  - hit rate the same, and no kills.
- In training, both seeds' aim cos rises to 0.3-0.38.

### 2. Then both seeds collapse, at about 22-25M

The same sequence happens in both seeds (`metrics.jsonl`, updates 5000-6500):

1. The aim KL climbs and spikes:
   - single updates of 0.9-1.3;
   - grad_norm_max up to 43 (it is clipped to 0.5 before the step).
2. Within about 300-500 updates, every discrete head drifts to the maximum entropy:
   - direction ln 3 = 1.10;
   - jump, fire and hook ln 2 = 0.69.
3. The policy's KL per update falls to 0.0000, the clip fraction to 0, and the aim's KL to 0. Learning stops.
4. log_std_y then falls steadily to the -4.5 watchdog line.
5. The aim cosine is 0 and the agent plays like the random ckpt_0000.

This is real instability, not a false alarm. The aim-KL spikes that stopped attempts 1 and 2 were its early sign. Raising the limit let both runs reach their peaks at 15-20M, and also let them run on past the point where it started.

The mechanism is not measured. The reward per step falls just before the drift. A uniform policy with zero KL means the policy gradient no longer moves anything and the entropy bonus wins. Two explanations fit that, and both are hypotheses:

- the advantage loses its signal;
- the trunk saturates after the large-gradient updates.

### 3. Best checkpoint

`runs/phase3-long/seed1/ckpt_3663.pt` (15M) is the best checkpoint measured so far. Against ckpt_0000, at 5M → 15M:

- kills 0.016 → 0.125;
- hit rate 0.027 → 0.057;
- aim cos 0.20 → 0.35;
- damage ratio 1.69.

Seed 3's peak (20M) is better than its 5M point on aim and damage ratio, but not on kills or hit rate. So the gain is in two seeds for the aim, but in one seed for kills. It is reported as the best checkpoint, not as an established multi-seed result.

## Next step (proposed, not started)

1. **Measure the collapse, without training.** Probe the checkpoints just before and after the drift in both seeds, at about 20, 22, 23 and 24M (all saved):
   - the advantage's spread and signal;
   - the trunk's activations, for saturation or dead units;
   - the per-head policy gradient against the entropy gradient.
2. **Only then make one change.** The candidates depend on step 1:
   - a learning-rate decay;
   - a tighter aim KL target;
   - a lower entropy coefficient once the policy is uniform.

   The test would resume from 15-20M, one seed at a time.
