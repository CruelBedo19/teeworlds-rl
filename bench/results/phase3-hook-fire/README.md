# Hook aim gradient and fire/ammo diagnostics (2026-10-21)

Needs the protocol 4 server (`teeworlds-0.7.5` a2d001e, built in `build-v4`). The older `build-release`
binary speaks protocol 3 and the Python side now refuses it.

## Hook aim gradient

The hook takes its direction from the aim at one moment only: in `CCharacterCore::Tick` the hook leaves
`HOOK_IDLE` when the button is down, with `m_HookDir = TargetDirection`; after that the aim plays no part
(flying, grabbed, retracted, and a held button never launches again). The bridge had no launch signal
(the core's `COREEVENTFLAG_HOOK_LAUNCH` is commented out in vanilla), and with `aim_mask "shots"` the
aim learned only where a shot left the gun. About 92-94 % of the 370 launches per episode fall on
decisions without a shot, so the aim got no gradient from them.

Change:

- Server (protocol 4): `CCharacter::Tick` reports a launch when the core's hook state leaves `HOOK_IDLE`
  (nothing else does), and `FireWeapon` reports dry fires (a press on an empty weapon). Both are new
  events, `hook_launches` and `dry_fires`.
- Trainer: `aim_mask "shots+hooks"` (`--aim-mask shots+hooks`) also trains the aim where the server
  counted a launch. The default stays `"shots"`; with it the first 30 updates of the bounded 5M run
  are reproduced bit for bit on the new server.
- Tests: launches counted once per launch; over 300 random decisions a launch happens exactly when the
  hook is idle at the decision and the button is down; dry fires only on an empty gun; under both masks
  the same rollout, with `"shots+hooks"` adding exactly the launch decisions without a shot.

## Smoke runs (`smoke.sh`)

Same seed and settings as the bounded 5M run (seed 1, bounded mean, aim_target_kl 0.015, frozen
ckpt_0000), 250 updates (1.02M learner steps) each, the two masks in parallel. Means per quarter, then
the credit audit of each final checkpoint (20 rollouts):

| per update (4096 decisions) | shots | shots+hooks |
|---|---|---|
| decisions in the aim loss | ~275 (6.7 %) | ~1100 (25-29 %) |
| of which hook launches without a shot | 0 | ~760-940 |
| aim KL | 0.006 / 0.008 / 0.011 / 0.013 | 0.007 / 0.009 / 0.009 / 0.010 |
| aim clip | 0.07 / 0.10 / 0.12 / 0.15 | 0.09 / 0.11 / 0.10 / 0.11 |
| aim off | 0.06 / 0.14 / 0.26 / 0.30 | 0.04 / 0.08 / 0.07 / 0.10 |
| log_std (end) | -0.60 | -0.64 |
| other heads KL / clip | 0.011 / 0.16 | 0.012 / 0.17 |
| grad norm max | 1.09 | 1.15 |
| non-finite values | 0 | 0 |
| fire presses / dry fires | ~1970 / ~690 | ~1970 / ~690 |
| hook launches | ~890 | ~910 |

Credit audit (pipeline errors <= 2.4e-7 in both): the aim gradient's push towards the opponent per
update, signal over noise: shot decisions 0.82 (shots run) and 0.56 (hooks run); hook launches without
a shot -0.50 and +0.15 of their own noise. Hook launches do not on average point the aim at the
opponent, which is expected (the hook is also used on walls and ceilings), so their gradient is a
different target sharing the same head, not added signal for shooting. After 1M steps neither run
aims at the opponent better than the other (training aim cosine -0.06 vs -0.07); that would need the
longer runs.

## Fire / ammo (`fire_diag.json`, `fire_diag_masked.json`)

`bench/diagnose.py` on the eval seeds, both sides, 64 episodes per checkpoint, against ckpt_0000 of the
bounded run. Shares of alive decisions for the weapon state, shares of presses for what a press did.

| | bounded 0 | bounded 2M | bounded 4M | bounded 5M | v4 2M | v4 5M |
|---|---|---|---|---|---|---|
| weapon ready | 0.13 | 0.15 | 0.12 | 0.17 | 0.22 | 0.14 |
| reload timer, ammo left | 0.02 | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 |
| reload timer, empty | 0.48 | 0.48 | 0.51 | 0.45 | 0.41 | 0.48 |
| empty, timer done | 0.37 | 0.36 | 0.36 | 0.37 | 0.36 | 0.36 |
| fire pressed | 0.50 | 0.52 | 0.58 | 0.46 | 0.41 | 0.53 |
| press -> shot | 0.13 | 0.13 | 0.11 | 0.15 | 0.17 | 0.12 |
| press -> dry fire | 0.37 | 0.35 | 0.35 | 0.36 | 0.35 | 0.35 |
| press -> nothing (timer running) | 0.50 | 0.52 | 0.54 | 0.49 | 0.48 | 0.53 |
| p(fire given visible / line of fire) | 0.50 / 0.50 | 0.53 / 0.57 | 0.57 / 0.59 | 0.45 / 0.43 | 0.37 / 0.38 | 0.53 / 0.52 |
| p(fire given line of fire and ready) | 0.51 | 0.47 | 0.51 | 0.37 | 0.28 | 0.40 |
| p(fire given not visible and ready) | 0.50 | 0.46 | 0.55 | 0.45 | 0.39 | 0.49 |
| line of fire | 0.22 | 0.20 | 0.18 | 0.21 | 0.21 | 0.21 |
| combat opportunity (line of fire and ready) | 0.028 | 0.028 | 0.023 | 0.039 | 0.054 | 0.035 |
| shots / dry fires per episode | 99 / 276 | 97 / 274 | 94 / 298 | 101 / 246 | 106 / 212 | 96 / 277 |
| hits (= damage) per episode | 1.4 | 1.6 | 1.7 | 1.8 | 2.6 | 1.1 |

Counterfactual with the checkpoint's fire turned off on an empty gun (`--mask-empty-fire`, bounded 0 /
4M / 5M): the gun is still empty 0.82 / 0.83 / 0.79 of the time, presses turn into shots 0.91 / 0.89 /
0.91 of the time, and shots per episode rise from 94-101 to 124 in all three.

Reading:

- The empty gun is mostly the ammo rules. The gun refills one round per 500 ms, at most about 120
  rounds plus the first 10 in a 3000-tick episode. A policy that presses on half its decisions
  empties the gun at once and then waits for every refill; with no dry fires at all it still reaches
  124 shots per episode, the refill limit, with the gun empty about 80 % of the time.
- Dry fires cost about 20 % of the shots (99 against 124): each one restarts the 125 ms delay and with
  it the refill timer (`CCharacter::HandleWeapons`). That is the game's rule, not a bridge bug.
- The "wasted" presses (about 87 %) are half presses into a running timer, which do nothing, and a
  third dry fires.
- The policy's firing is still close to its initial coin flip (p(fire) about 0.5) and is not selective:
  it fires as often with the opponent out of view as in its line of fire, so the rounds are gone when
  an opportunity comes. That, not the environment, is why only 2-5 % of decisions are a combat
  opportunity while the opponent is in the line of fire on about 21 % of them.
- A gun hit does 1 damage, and the policy cannot switch weapons, so a kill takes 10 hits on a player
  with full health and no armour. At 1-3 hits per episode kills, and with them wins, are out of reach
  for these policies; the reward is almost entirely the +-0.02 damage terms.
