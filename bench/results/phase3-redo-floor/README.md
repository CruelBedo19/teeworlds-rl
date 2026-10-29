# ReDo + aim log_std floor: seed 1 resumed from 15M (2026-10-29)

## Change

The one change against `phase3-redo` (the control) is `--aim-log-std-floor -2.3`, at commit `e8a7601`, with the port retry from `9e48b64`. ReDo, optimizer state, opponent, reward, LR, PPO settings, KL brake and the watchdog limit of 0.25 are the same.

Seed 1 was resumed from `runs/phase3-redo-floor/seed1/ckpt_3663.pt`, the same 15M checkpoint as the control.

## What happened

- The run went from 15M to 21.0M (update 5123), in 41 minutes.
- It was stopped by the watchdog: aim KL 0.68, mean of 20 updates, against the limit of 0.25.
- Exit 0, no NaN and no infrastructure error.
- The 25M and 30M evaluations were not reached.
- No checkpoint exists after the floor engaged: the last one is `ckpt_4883` (20M), and the floor first acted at update 5000.

The run is identical to the control up to update 4999: every metric matches, as does the 20M evaluation (`ckpt_4883`). The two runs diverge at update 5000, the first update where log_std_y reached -2.3.

## The stop was one outlier update

The watchdog reads `aim_approx_kl`. That is the k3 estimator `(r-1) - log r`, computed per 512-decision minibatch on the shot decisions only. A minibatch has about 30 of them, so one tail sample can dominate the estimate.

- Update 5108: `aim_approx_kl` = 12.41, but the exact full-batch Gaussian KL (`aim_kl_exact`) was 0.145.
- Last 20 updates: the mean is 0.682, the median 0.069, and the mean without that one update 0.065.
- The control's stop at 27.8M has the same shape. Update 6784 had `aim_approx_kl` = 5.75; over the last 20 updates the mean is 0.398, the median 0.107, and the mean without that update 0.116.
- Across this run, the per-update correlation between `aim_approx_kl` and `aim_kl_exact` is 0.14.

## Aim KL decomposition

| updates | log_std_y | mu shift | exact KL (mean / median) | approx KL (mean / median) | grad norm |
|---|---|---|---|---|---|
| 15.0-16.0M | -2.095 | .0511 | .072 / .068 | .058 / .052 | .99 |
| 19.0-20.0M | -2.257 | .0497 | .081 / .078 | .065 / .056 | 1.07 |
| 20.0-20.5M | -2.286 | .0502 | .087 / .083 | .085 / .060 | 1.16 |
| 20.5-21.0M (floor) | -2.298 | .0547 | .093 / .088 | .184 / .071 | 1.15 |

- `aim_kl_std_part` is about 0 throughout. log_std is a state-independent parameter that moves by about 1e-3 per update, so its own change adds almost no KL.
- Narrowing std still matters, because the mean part is `dmu² / (2 sigma²)`. From 15M to 20M the mean shift stayed flat at 0.050, while exact KL rose by about 13 % as sigma_y narrowed. The upper bound from sigma_y alone is `exp(2 x 0.16)` = 1.38, but sigma_x moved less and also carries part of the KL.
- After the floor engaged, log_std_y stayed in [-2.300, -2.294] and log_std_x was about -1.44 and free.
- The floor was hit on 2.1 % of the y optimizer steps, and in 16 % of the 124 updates; it was never hit on x.
- In those 124 updates exact KL did not fall: 0.093 against 0.087 before, with the mean shift slightly up (0.055 against 0.050). That is too short a stretch to call a trend.

## Dead units

`probe_seed1.json` covers only `ckpt_4883` (20M), which is shared with the control: trunk2 0.8 %, trunk1 0 %, conv 0.5-0.9 %.

ReDo recycled the following over the run: trunk2 429, trunk1 72, vec_fc 3, grid_fc 0. In the 124 floor updates it recycled trunk2 30 and trunk1 5.

## Evaluation against ckpt_0000 (64 matches)

| | kills / deaths | W/L/D | damage dealt / taken | ratio | hit rate | aim cos |
|---|---|---|---|---|---|---|
| 15M start | .078 / 0 | 5/0/59 | 5.25 / 2.70 | 1.94 | .052 | .37 |
| 20M (= control 20M) | .016 / 0 | 1/0/63 | 2.84 / 1.59 | 1.78 | .028 | .12 |

There is no evaluation with the floor active.

## Reading

- The experiment is **inconclusive**. The floor held log_std_y at -2.3 as designed, but only for 124 updates, and there is no checkpoint or evaluation with it active.
- Both watchdog stops, this run's and the control's, came from a single update in which the minibatch k3 estimate spiked. They were not sustained KL growth.
- Up to 20M the mean shift per update did not grow (0.051 → 0.050), and exact KL rose only 13 %. The narrowing sigma accounts for that rise; larger mean updates do not.
- Performance had already fallen from 15M to 20M before the floor engaged, so the 15M→20M drop has a cause other than the KL stop.
