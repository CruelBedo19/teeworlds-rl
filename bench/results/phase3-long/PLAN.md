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

## Attempt 2 (23:44)

Attempt 1 stopped seed 3 at 7M (update 1705). The cause was the pipeline's watchdog, not instability.

- The rule is "aim KL, mean of 20 updates, above 0.05".
- The aim KL had been steady at a mean of about 0.03 in both seeds since 4M.
- Single updates spike to 0.2-0.3: the maximum was 0.08-0.12 from 4M to 5M, and 0.23-0.29 after.
- One spike took the 20-update mean just over the limit.

The fix:

- `learn_pipeline.py --aim-kl-limit` makes the limit settable. The default is unchanged, at 0.05.
- This run uses 0.10. That still catches a sustained climb, such as v3's, but not a single spike.

Both seeds restart from their 6M checkpoint (`ckpt_1465.pt`), with metrics cut back to update 1465. Seed 1 had reached 6.3M when it was stopped.

Attempt 1's files are in `attempt1/`, including the 5M re-evaluation of seed 3. The deadline is still 11:13.

## Attempt 3 (01:05)

Attempt 2 stopped again on the same watchdog rule:

- seed 3 at 8.9M, with a 20-update mean of 0.156;
- seed 1 at 15.6M, with a 20-update mean of 0.104.

Neither stop was instability.

- Both times the cause was one or two single updates of 0.9-1.3. Most updates were 0.02-0.06.
- These spikes grow as the aim's std narrows (log_std_y fell from -1.6 to -2.1), because the same mean shift costs more KL.
- Over the same updates, seed 1's training aim cos rose from 0.22 to 0.34-0.38 and its hit rate from 0.03 to 0.05.
- Seed 1's 15M evaluation against ckpt_0000 was well above its 5M point: kills 0.125, deaths 0.047, damage 5.75 / 3.41, hit rate 0.057, aim cos 0.35.

The fix:

- The limit is raised to 0.25. That is about four such spikes within 20 updates, so a sustained climb still trips it.
- The other watchdog rules are unchanged: grad norm, entropy, log_std, shots, stale.
- Each seed resumes from its last checkpoint (seed 3 at 8M, seed 1 at 15M), with metrics cut back to it.
- Attempt 2's logs are in `attempt2/`.
