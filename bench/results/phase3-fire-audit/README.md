# Fire credit audit on the shots-only runs (2026-10-24)

This is a measurement only; nothing was trained.

## What was measured

`bench/fire_audit.py` was run on `phase3-ab-mask` seeds 1, 2 and 3, at four checkpoints each: ckpt_0000, 1M (`ckpt_0245`), 3M (`ckpt_0733`) and 5M (`ckpt_1221`). For each checkpoint:

- 24 training batches were collected, as the trainer does, against ckpt_0000;
- every decision was split by its situation and by what it did, and its advantage compared;
- the fire head's policy gradient and its entropy-bonus gradient were measured on the fire logits and on the trunk.

`bench/fire_confound.py` then checked the empty-gun gaps on two checkpoints, seed 1 and seed 3 at 5M. The fire and ammo behaviour of every checkpoint comes from `phase3-ab-mask/diag_seed*.json`.

The audit samples use one policy at a time. Advantages are normalised over the batch, as in `train()`, unless they are called raw.

## The rule that makes fire hard

In `CCharacter::FireWeapon` and `HandleWeapons` (`character.cpp`):

- A press on an empty gun is a dry fire. It sets the 125 ms reload timer.
- Any shot or dry fire sets the timer, and that resets the ammo refill: `m_AmmoRegenStart = -1`.
- A round comes back only after 500 ms, about 12.5 decisions, with the timer at zero.
- Presses while the timer runs are lost.
- The bridge turns every `fire=1` decision into a new press.

So every press on an empty gun pushes the next round back by up to 500 ms. That cost shows up seconds later, as an empty gun at the next combat opportunity.

## Findings

### 1. Most fire decisions are made on an empty gun

Per batch of 4096 decisions:

| situation | decisions |
|---|---|
| gun empty | 2850-3550 |
| line of fire with a ready gun | 83-275 |
| ready gun with the opponent out of view | 208-599 |

Each batch also holds 500-870 dry fires.

From ckpt_0000 to 5M the gun is empty 79-88 % of the time, and there are 228-326 dry fires per match.

Firing is not selective. At 5M, p(fire) is:

| | line of fire, ready gun | opponent out of view, ready gun |
|---|---|---|
| seed 1 | 0.30 | 0.44 |
| seed 2 | 0.49 | 0.61 |
| seed 3 | 0.36 | 0.45 |

In every seed the agent fires less often when it can hit than when it cannot.

### 2. The entropy bonus is not what holds p(fire) near 0.5

On the fire head's own parameters, per batch, the entropy gradient is 0.2-10 % of the policy gradient's norm, and 0.2-34 % of the norm of the policy gradient's mean over 24 batches.

On the fire logit itself, the bonus summed over a batch is at most 16 % of the batch-to-batch noise of the policy gradient, in every situation. This repeats `phase3-hook-fire/fire_head_probe.json` on the new runs.

**The hypothesis that entropy regularisation suppresses the fire signal is rejected.** Changing the fire entropy coefficient is therefore not tested.

### 3. Firing in the line of fire has a weak positive signal; firing out of view has none

Advantage after a shot minus after no press, per situation:

| situation | normalised gap | batches for the sign to be right 95 % of the time |
|---|---|---|
| line of fire, ready gun (1M-5M) | +0.01 to +0.35, positive at all 9 checkpoints | 5-1520 |
| ready gun, opponent out of view | -0.04 to +0.04 | no consistent sign |

In raw reward units, the gap in the line of fire is about 1e-3, the size of a 5 % chance of a 0.02 hit.

Firing out of view costs a round that would be missing at a later opportunity. That cost is delayed and smaller than the return noise, so the advantage does not see it.

### 4. On an empty gun the press/no-press gap is value-function bias

The empty-gun gap is large but has no consistent sign:

| checkpoint | normalised gap | t |
|---|---|---|
| seed 1, 5M | -0.058 | -7.3 |
| seed 3, 5M | +0.079 | +7.6 |
| seed 3, 1M | +0.077 | +5.4 |
| seed 2, 1M | -0.057 | -5.9 |

`fire_confound.py` kept only the empty-gun decisions with no damage dealt or taken within 20 decisions. The gap stayed: -0.056 (t -11.7) for seed 1 and +0.086 (t +11.4) for seed 3. So it is not the reward of nearby hits.

With no reward nearby, the advantage is made only of value differences, gamma V(s') - V(s). A dry fire changes the next observation: the reload timer runs. The value function rates that state up or down depending on the seed, so its error becomes the advantage for pressing.

In raw units this bias is up to about 4e-4 per decision. The real cost of a dry fire is small:

- The counterfactual in `phase3-hook-fire` showed that taking away every dry fire adds about 25 shots per match.
- So one dry fire costs about 0.1 round.
- A round is worth about 3-5 % x 0.02, so a dry fire costs about 1e-4.

Empty-gun decisions are 80 % of all fire decisions, and they get an advantage whose sign comes from the value function's error, not from their effect. That signal and its noise dominate the fire head's update. They swamp the small line-of-fire signal and leave the dry fires where they are.

### 5. Parameter-space consistency of the fire gradient is low

On the fire head, rho (the squared length of the expected gradient over one batch's noise) is significant only at seed 1's ckpt_0000, 1M and 3M:

- rho 0.10-0.14;
- N_half 7-10 batches.

Elsewhere it is about 0, and on the trunk it is below 0.06.

## Conclusion and next step

The fire head does not learn because:

- 80 % of its decisions sit on an empty gun, where the only real effect of a press, a later round, is about 1e-4;
- that effect is far below the value-function bias, about 4e-4, and the return noise.

The smallest change that targets this is to make a real game event, the dry fire, carry its own small immediate cost:

- `--dry-fire-penalty`, with the default off;
- a value above the measured value bias, so the empty-gun signal has a known sign.

Firing out of view and the line-of-fire signal are not touched. Neither is the aim. See `phase3-fire-ab`.
