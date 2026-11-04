# Handoff note (for a new session or a new account)

This repo is the output of an analysis done in a Claude Code session. In a new session, continue in this order.

> **Development is paused (as of 2026-11-02).** Work can resume at any time from the state below. Everything lives on the `phase3` branch. The next step is the offline aim-brake probe described in "Where we stopped and next step".

## Where we are (2026-11-02, branch `phase3`)

- Phases 0, 1 and 2 are done (see [PLAN.md](PLAN.md)). Phase 3 (PPO) is in progress on the `phase3` branch. Nothing has been merged into main.
- Current baseline: shots-only aim mask, dry-fire penalty, ReDo (`--redo-every 25 --redo-tau 0`) and an aim log_std floor of `-2.3`. Full arguments: `runs/phase3-redo-floor-exactkl/seq.sh`.
- The source of truth for performance comparisons is `vs.initial` (= ckpt_0000) in the raw eval files. Do not confuse it with `vs.random`.
- Seed 1, 15M, vs.ckpt_0000: K/D .078/0, W/L/D 5/0/59, damage 5.25/2.70 (ratio 1.94), hit .052, aim cos .37.

### Latest experiments (results in `bench/results/`)

- **`phase3-redo-floor`:** run with floor -2.3. The watchdog stopped it at 21.0M. The cause was a single 12.41 outlier in the minibatch k3 estimate (`aim_approx_kl`); the exact KL in the same update was 0.145. The 0.40 that stopped the control (`phase3-redo`) was the same kind of single outlier.
- **Commit `1bb9a25`:** the watchdog now reads `aim_kl_exact`. The 20-update mean and the 0.25 limit are unchanged. Tests: `tests/test_learn_pipeline.py`.
- **`phase3-redo-floor-exactkl`:** went from 15M to 30M without a stop.
  - Exact KL at most 0.14. Dead neurons: 0 in the FC layers, 3-4% in conv2.
  - 25M: ratio 1.90, hit .048, aim .29. Control at 25M: ratio 1.48, aim -.03.
  - 30M: ratio 1.82, hit .033, aim .19.
  - Stable, but it did not pass the 15M result.

### Open findings

- **aim_off ≈ 72-76%:** the KL brake fires on the same noisy k3 estimate, and the aim loss is switched off in about three quarters of the minibatches.
- **KL growth after the floor:** exact KL rose from 0.090 to 0.103 while the mu shift fell. The x/y per-axis split is not logged.
- **Watchdog gap:** at each poll the watchdog looks only at the last line. A grad_norm_max of 23.6 (update 5246) passed unseen. Not fixed; fixing it could stop runs, so the decision is the user's.

## Where we stopped and next step

The user approved this, but it has not been started: **offline aim_off and x/y KL measurement.** No training, nothing is written to checkpoints.

1. Checkpoints: 3663, 4883, 5372, 6104, 6836 and 7325 under `runs/phase3-redo-floor-exactkl/seed1/`. Load each with `Trainer.load_state_dict`, including optimizer and generator. As in training, collect 8 batches against ckpt_0000 (see `bench/collapse_probe.py`).
2. On each batch, run a single update on a copy of the checkpoint. Three variants of the brake signal:
   - k3 (current);
   - minibatch exact Gaussian KL (relative to the pre-rollout mu0/s0);
   - no brake.
3. Control first: the replica of the `train()` loop must give exactly the same result as the real `Trainer.train` in the k3 variant.
4. To measure: aim_off, the minibatch where the brake first fires, mean-part KL for the x and y axes separately, RMS Δμ, effect on σ, k3/exact difference.
5. Output goes to `bench/results/phase3-aim-brake-probe/`; local commit.

The user's rules:
- Do not start new training, or change LR, reward or the watchdog limit, without approval.
- Do not use self-play or a scripted opponent for this probe.
- Do not delete checkpoints, logs or runs; archive unfinished runs under `failed-*` / `aborted-*`.
- Do not touch the chase, relay and spectator processes.
- Ask for approval before push and PR.
- Reports are in Turkish.

(Checkpoints, `runs/` and training live on the author's remote server and are not in this repo. Only code and result files are here.)

## Working environment

- Teeworlds source: `git clone --branch 0.7.5 https://github.com/teeworlds/teeworlds`, then `git -c url."https://github.com/".insteadOf="git://github.com/" submodule update --init` (`.gitmodules` uses the old `git://` addresses). Apply the patches with `git am patches/*.patch`.
- Build (with CMake 4): `CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DCLIENT=OFF`. For the benchmark with dummies, also build `build-bench` with `-DCMAKE_CXX_FLAGS=-DCONF_DEBUG`.
- Line numbers come from sub-agent reports and were not verified one by one. Check them in the relevant files before writing code.
- The location of the `SERVER_TICK_SPEED` definition is not verified (its value is 50).
- The `seq.sh` / `run.sh` scripts under `bench/results/` and `runs/` expect the environment variable `REPO_DIR` to point to the repo checkout (see [SANITIZATION.md](SANITIZATION.md)).

## First message for the new session

> Read README.md, PLAN.md and HANDOFF.md in the teeworlds-rl repo (branch `phase3`). Summarize where we stopped and propose the next step. Do not start training or write code until I approve.

## Opening the same session on another account

The session is not tied to the account; it is a local file:

```
~/.claude/projects/<project-dir>/<session-id>.jsonl
```

On the same computer: `/logout`, `/login` (new account), then in the project folder run
`claude --resume <session-id>`.
On another computer: copy the file to the same path and run the same command.
You can also continue without the session: this repo carries enough context.

The Python environment needs numpy: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Tests: `TWRL_SERVER_DIR=<build-release> .venv/bin/python tests/test_env.py -v`.
