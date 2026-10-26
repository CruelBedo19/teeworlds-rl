# Overnight long run (2026-10-26), at most 12 hours

The user asked for one training session of at most 12 hours, planned to make the best use of the time.

## Question

Does the shots-only baseline keep improving past 5M, or does the aim stay stuck at about 0.2?

So far every run stopped at 5M. Between 3M and 5M the aim cosine was flat, about 0.14-0.25.

## What changes

Only the length of training. Everything else is the shots-only baseline of `phase3-ab-mask`:

- bounded aim mean, aim mask shots, aim_target_kl 0.015;
- frozen ckpt_0000 opponent (the run's own random-weight network);
- build-v4, env-timeout 600;
- the same PPO settings, observations, actions, map and spawn.

There is no dry-fire penalty: it was rejected (`bench/results/phase3-fire-5m`). There is no self-play and no new reward.

## Runs, one after another

| run | starts from | why |
|---|---|---|
| A | control seed 3 at 5M (`phase3-ab-mask/seed3/ckpt_1221.pt`) | best control at 5M: 4/4 criteria, damage ratio 1.58, kills 0.062 |
| B | control seed 1 at 5M | the second seed, to tell a trend from seed luck |

- Each run is resumed from 5M to 40M (9766 updates).
- 40M keeps the update count below 10000, the 4-digit checkpoint names the pipeline looks for.

## Evaluation

`learn_pipeline.py` evaluates the run at 5, 10, 15, 20, 25, 30, 35 and 40M, with 64 matches against ckpt_0000 at each point. The 5M point re-measures the starting checkpoint.

After both runs, `diagnose.py` measures fire and aim at 10, 20, 30 and 40M, one run at a time.

## Limits

- Each training run is killed after 330 minutes.
- Run B starts only if at least 5.5 hours of the 12 are left.
- The whole script is killed after 715 minutes.
- An infrastructure failure, such as a bridge timeout or a port clash, is recorded as such in `status` and is not counted as a learning result.

## Reading

| result | conclusion |
|---|---|
| aim cos and hit rate rise past 5M in both runs | length is a lever; next step is a longer run |
| both runs stay flat | the aim needs a method change before more steps (see `phase3-fire-5m` README) |
| one rises, one stays flat | seed luck; needs more seeds before any conclusion |
