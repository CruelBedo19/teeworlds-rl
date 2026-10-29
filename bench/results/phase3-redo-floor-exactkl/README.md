# ReDo + aim log_std floor, exact-KL watchdog: seed 1 from 15M to 30M (2026-10-29)

## Change

Training is that of `phase3-redo-floor`: ReDo `--redo-every 25 --redo-tau 0`, `--aim-log-std-floor -2.3`, shots-only aim mask, the same reward, LR, PPO settings, KL brake and opponent.

The one change is in the watchdog, at commit `1bb9a25`. Its 20-update aim KL mean now reads `aim_kl_exact`, the exact full-batch Gaussian KL, instead of the minibatch k3 estimate `aim_approx_kl`. The limit stays 0.25.

The run resumed from the same 15M checkpoint (`ckpt_3663`, sha256 `8ef34121…`). It was checked to be bit-identical to `phase3-redo-floor` over the updates the two runs share.

## What happened

- The run went from 15M to 30.0M (update 7325), in 94 minutes, with exit 0.
- No watchdog stop, no NaN, no infrastructure error.
- The 20M, 25M and 30M evaluations are done.
- The old rule would have stopped the run at update 5108, as in `phase3-redo-floor`: `aim_approx_kl` was 12.41 there while the exact KL was 0.145.
- A second large outlier came at update 5836: `aim_approx_kl` 9.03, exact 0.165.

### A watchdog gap that predates this change

The pipeline checks only the newest metrics row at each 30 s poll, so roughly ten updates go unchecked between polls.

- Update 5246 had `grad_norm_max` 23.6, which is above the `grad_norm_max_above` rule of 20, and the watchdog did not see it.
- That norm is measured before clipping. The step itself was clipped to `max_grad_norm`.
- On the same update, exact aim KL was 0.082 and approx_kl 0.017; both are normal.
- Replayed row by row, the rules would have stopped this run at 21.4M.
- The control `phase3-redo` had the same gap: a grad norm of 48.2 at update 6664 went unseen.

## Evaluation against ckpt_0000 (64 matches)

| | kills / deaths | W/L/D | damage dealt / taken | ratio | hit rate | aim cos |
|---|---|---|---|---|---|---|
| 15M start | .078 / 0 | 5/0/59 | 5.25 / 2.70 | 1.94 | .052 | .37 |
| 20M (floor not yet active) | .016 / 0 | 1/0/63 | 2.84 / 1.59 | 1.78 | .028 | .12 |
| 25M | .078 / .016 | 5/1/58 | 4.62 / 2.44 | 1.90 | .048 | .29 |
| 30M | .109 / .016 | 7/1/56 | 3.39 / 1.86 | 1.82 | .033 | .19 |
| control 25M (no floor) | .062 / .016 | 4/1/59 | 3.78 / 2.56 | 1.48 | .038 | -.03 |

All four pre-registered criteria pass at 30M; the Kendall tau of the ratio is 0.33.

## Training metrics per 1M steps

| M | exact KL mean | roll20 exact max | approx KL mean / max | mu shift | log_std x / y | floor-hit updates | aim_off | hit | aim cos |
|---|---|---|---|---|---|---|---|---|---|
| 15-16 | .072 | .088 | .058 / 0.61 | .0512 | -1.39 / -2.10 | 0 | .72 | .049 | .30 |
| 19-20 | .081 | .099 | .065 / 0.37 | .0497 | -1.44 / -2.26 | 0 | .73 | .046 | .30 |
| 20-21 | .090 | .108 | .135 / 12.41 | .0524 | -1.44 / -2.29 | 8 % | .75 | .049 | .29 |
| 22-23 | .099 | .117 | .078 / 0.53 | .0530 | -1.45 / -2.30 | 18 % | .75 | .054 | .29 |
| 25-26 | .095 | .110 | .085 / 0.66 | .0483 | -1.47 / -2.30 | 32 % | .74 | .051 | .24 |
| 27-28 | .099 | .140 | .087 / 2.18 | .0471 | -1.48 / -2.30 | 39 % | .72 | .047 | .23 |
| 29-30 | .103 | .123 | .082 / 0.88 | .0468 | -1.51 / -2.30 | 20 % | .74 | .037 | .18 |

- `aim_kl_std_part` stays at about 1e-6.
- The highest 20-update mean of the exact KL over the whole run was 0.140, against the limit of 0.25.
- Across all updates, the correlation between approx and exact KL is 0.14.

## Dead units (`probe_seed1.json`)

| | conv1 | conv2 | conv3 | grid_fc | vec_fc | trunk1 | trunk2 |
|---|---|---|---|---|---|---|---|
| 20M | 0.5 % | 0.7 % | 0.9 % | 0 | 0 | 0 | 0.8 % |
| 25M | 0.0 % | 3.8 % | 1.6 % | 0 | 0 | 0 | 0 |
| 30M | 0.1 % | 3.3 % | 0.8 % | 0 | 0 | 0 | 0 |

ReDo recycled the following over the run: trunk2 787, trunk1 103, vec_fc 7, grid_fc 1. The conv layers are not recycled.

## Reading

- With the floor active and the false watchdog stops removed, the run reached 30M stable: the exact KL never got near the limit, and there was no collapse.
- It did not pass the 15M start. At 25M the policy is close to 15M on most measures, and well above the control's 25M. At 30M kills and wins are a little higher, but damage, hit rate and aim cos are lower, and the training hit rate and aim cos fall over 28-30M.
- After the floor engaged, exact KL still crept from 0.090 to 0.103. In the same stretch the mean shift fell (0.052 → 0.047) and log_std_y was pinned, while log_std_x narrowed (-1.44 → -1.51). Larger mean updates do not explain the rise: the mean |dmu| fell by 10 %. Narrowing x accounts for at most part of it. The rest must come from how the shift is spread, either more of it along y (where sigma is smaller) or heavier tails. The per-axis split is not logged, so this is not settled.
- This is one seed only, so no best-model claim is made.
