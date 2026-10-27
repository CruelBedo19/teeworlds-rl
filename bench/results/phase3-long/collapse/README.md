# Why the long runs collapse (2026-10-27)

This is a measurement only. `bench/collapse_probe.py` fixes each checkpoint's weights and collects training batches against the run's ckpt_0000, as the trainer does:

- 4 batches per checkpoint, 2 for seed 1's early checkpoints;
- seed 1 at 0, 5, 10, 15, 20, 21, 22, 23, 24, 25 and 30M;
- seed 3 at 15, 20, 22, 23, 24, 25, 26 and 30M.

## Finding: the last trunk layer dies unit by unit

A unit counts as dead when its ReLU output is zero on every decision of the batches.

| seed 1 | 0 | 5M | 10M | 15M | 20M | 21M | 22M | 23M | 24M |
|---|---|---|---|---|---|---|---|---|---|
| trunk2 dead | 0.01 | 0.02 | 0.10 | 0.27 | 0.61 | 0.70 | 0.73 | 0.88 | **1.00** |
| trunk2 units active per decision | 0.50 | 0.24 | 0.10 | 0.028 | 0.007 | 0.004 | 0.004 | 0.001 | 0 |
| trunk1 dead | 0.02 | 0.03 | 0.06 | 0.08 | 0.11 | 0.13 | 0.15 | 0.17 | 0.28 |

| seed 3 | 15M | 20M | 22M | 23M | 24M | 25M | 26M |
|---|---|---|---|---|---|---|---|
| trunk2 dead | 0.21 | 0.48 | 0.61 | 0.61 | 0.68 | 0.86 | **1.00** |

The convolutions, `grid_fc` and `vec_fc` stay alive, with at most 2 % dead.

## How this produces the collapse

- With every unit of the last hidden layer dead, the actor's output is its bias alone. The output is the same in every state.
- The policy gradient on the actor and on the trunk is exactly 0 from that point: 0.0000 at 24M in seed 1 and at 26M in seed 3.
- The value's spread across states is also 0.
- The output bias is close to its initial zero, so every discrete head sits at maximum entropy. That is the "uniform policy" in the training log.
- Adam's second-moment estimate decays towards 0, from 1e-4 to 1e-21.
- log_std drifts down to the watchdog line, because nothing else is moving.

The decline is gradual, not a single event:

- At 15M, the best checkpoint of seed 1, a quarter of trunk2 is already dead, and only 3 % of its units fire on a typical decision.
- The aim and kills peak while the layer is narrowing. They fall once too few units are left.

## What is ruled out

The entropy bonus's gradient on the discrete heads is 1-4 % of the policy gradient's at every checkpoint before the death. So it is not the cause: it wins only once the policy gradient is 0.

## Next step (proposed, not started)

The phenomenon is known as dormant or dead neurons in deep RL (Sokar et al. 2023, ReDo). Two fixes fit:

1. **Recycle dormant units, without an architecture change.** Every N updates, re-initialise the input weights of trunk units that are dormant and zero their output weights, as ReDo does. The network's output stays the same.
   - This can resume from seed 1's 15M checkpoint (27 % dead) and run to 30M, about 1.5 hours per seed.
   - It is one variable: the recycling.
2. **A non-dying activation (LeakyReLU) or LayerNorm in the trunk.**
   - This is an architecture change. It needs training from scratch, so the existing checkpoints cannot be resumed.
