# Phase plan

Status: **Phases 0, 1 and 2 complete (2026-10-19).** Phase 3: PPO infrastructure and a short trial run exist (2026-10-20), and later experiments run on the `phase3` branch (see [HANDOFF.md](HANDOFF.md) for the latest state; **development is paused as of 2026-11-02**). See [README.md](README.md) for the architecture and rationale.

Goal: two AIs learn aim, movement, dodging and combat through 1v1 Deathmatch self-play in Teeworlds 0.7.5.

## Phase 0: Speed measurement (decision gate)

Purpose: measure how many ticks/s the server can run faster than real time. This number determines the rest of the architecture.

- [x] Build Teeworlds 0.7.5 as Release (`teeworlds_srv`, client off).
- [x] `sv_rl_mode` (0 = real time, 1 = fast ticks, 2 = fast ticks + minimal snapshot/network). Patch: [patches/](patches/), teeworlds branch `rl/phase0`, only `engine/server/server.cpp`, `server.h`, `engine/shared/config_variables.h`.
- [x] `sv_rl_seed`, `sv_rl_bench_ticks`, `sv_rl_hash_interval`.
- [x] Benchmark matrix and determinism test: [bench/run_bench.py](bench/run_bench.py), results in [bench/results/](bench/results/).

### Result (AMD Ryzen 5 3600, 6 cores / 12 threads, `dm1`)

Median ticks/s, 3 repetitions. Fast modes ran 5,000,000 ticks, mode 0 ran 3000 ticks.

| Case | ticks/s/process | total |
|---|---|---|
| mode 0, no players / 2 dummies | 50 | 50 |
| mode 1, no players | 870,726 | |
| mode 2, no players | 2,952,218 | |
| mode 1, 2 dummies | 409,449 | |
| mode 2, 2 dummies | 633,979 | |
| mode 2, 2 dummies, 2 processes | 625,771 | 1,252,108 |
| mode 2, 2 dummies, 4 processes | 615,991 | 2,464,768 |
| mode 2, 2 dummies, 8 processes | 505,802 | 4,056,516 |
| mode 2, 2 dummies, 12 processes | 374,895 | 4,489,251 |

Determinism: two runs with the same config, across modes 0/1/2, and 4 parallel processes all produced the same state hash sequence.

Limits: the debug dummies only walk (no fire, jump or hook), so the numbers are an upper bound without combat load. The seed does not change the result in this scenario (the path that uses `rand()` in DM is not triggered). Both must be re-measured in Phase 1 with bridge actions.

Decision: per-process speed is ~300 times the 2000 ticks/s decision threshold. A separate light simulator is not needed; the bottleneck will be the Python side.

## Phase 1: Bridge and Python environment

- [x] If `sv_rl_socket` is set, the server connects to a Unix socket Python listens on and adds two agents (`OnClientConnected(id, true)`, works in Release, no `CONF_DEBUG` needed). Patch: [patches/0002-*](patches/), teeworlds branch `rl/phase1`.
- [x] Lockstep: every `sv_rl_action_ticks` (default 2) ticks the server sends OBS and waits for an action; this happens at the start of `CGameContext::OnTick`, before `m_World.Tick()`.
- [x] Input goes through `CPlayer::OnDirectInput` + `OnPredictedInput`, with the fire counter like a client.
- [x] Events come from two observer hooks inside `CCharacter::TakeDamage` and `Die`; reward weights live in Python.
- [x] Reset takes the same path as the `restart` command (`DoWarmup(0)` → `StartMatch`); the process does not restart.
- [x] Fixed-size binary protocol v1: [twrl/protocol.py](twrl/protocol.py) and `src/game/server/rlenv.h`.
- [x] Python environment (stdlib only): [twrl/env.py](twrl/env.py); tests [tests/test_env.py](tests/test_env.py); long run and measurement [scripts/random_rollout.py](scripts/random_rollout.py).
- [ ] N-process vectorized wrapper: moved to Phase 2 (for now the rollout script runs in parallel with `multiprocessing`).

Observation v1 (23 floats, raw units): own alive, pos, vel, health, armor, weapon, ammo, reload, grounded, jumped, hook state and relative hook position; opponent alive, relative pos, vel, weapon, hook state; episode progress. The opponent's health/armor/ammo are not included because a human player cannot see them either. Action v1: direction, jump, fire, hook, target_x/y. No weapon switching.

### Result

- 25 integration tests passed: movement, jumping, firing, hook, damage/kill/score consistency, suicide via debug kill, reset, determinism, error cases (malformed message, wrong seq, timeout, disconnect, server kill, clean shutdown).
- Long run: 3200 episodes, 9.6M ticks, 8 parallel envs. No crash, protocol error or timeout. At every step `Δscore == kill − suicide`, damage dealt == damage the opponent took, kill == opponent death: 0 violations. Server RSS stable (~15-16 MB, at most +8 KB over 250 episodes).
- Reset time p50 0.05 ms (about 28 ticks).
- Determinism: with the same seed, episodes are identical both in the same process and in a fresh process; a different seed differs; modes 0, 1, 2 are identical. With the bridge off, the Phase 0 hash sequence is identical.
- ASan/UBSan: no reports from the new code. The double `pthread_join` in the vanilla `CJobPool` shutdown gives exit 1 under ASan (also with the bridge off); out of scope.
- Throughput (`bench/results/2026-10-19-phase1-scaling.csv`), total ticks/s, K=2:

| env | random dm1 total | chase lms1 total | step latency p50 / p99 (µs) |
|---|---|---|---|
| 1 | 45,521 | 56,542 | 36 / 60 |
| 2 | 82,635 | 97,602 | 38 / 64 |
| 4 | 140,851 | 167,994 | 45 / 75 |
| 8 | 240,130 | 279,064 | 54 / 89 |
| 12 | 269,590 | 320,044 | 58 / 269 |

As expected in Phase 0, the bottleneck is Python and IPC: the server alone does ~630k ticks/s, with the bridge ~45k ticks/s per env.

Note: on dm1, random agents almost never meet (3 kills in 6M ticks). Combat tests ran on the small `lms1` map, only with a test-only chasing script ([twrl/testing.py](twrl/testing.py)). Choosing a small map matters for Phase 2.

## Phase 2: Observation and environment layer

Detailed and exact specification: [ENVIRONMENT.md](ENVIRONMENT.md). Patch: [patches/0004-*](patches/), teeworlds branch `rl/phase2`.

- [x] Protocol v2: the action is unchanged; OBS is a fixed 2340 bytes, and HELLO sends the map's collision once.
- [x] Spatial observation: a 33×25-tile egocentric collision grid centered on the agent's tile (solid, death, nohook).
- [x] 67-dimensional vector: own state, opponent (only inside the rectangle a human would see), line-of-fire bit, nearest 4 projectiles, episode progress. No opponent health/armor/ammo.
- [x] Seeded spawn variety: the first spawn of each episode uses two different random map spawn points (RL mode only).
- [x] `VecTeeworldsEnv`: N servers from a single Python process, auto-reset, numpy arrays, `grid_format="codes"` option.
- [x] Tests (38 integration + 2 relay), long stress test, comparative benchmark under the same load as Phase 1.

### Result

- 38/38 integration tests and 2/2 relay tests passed. The grid matches the expected window computed from the map itself exactly at every step (including death and nohook tiles on dm6); opponent relative position, velocity, field of view and line of fire were verified against raw state; projectile slots are ordered and within bounds.
- Determinism: with the same seed, episodes are identical in the same process and in a fresh process; a different seed gives a different start and trace; modes 0, 1, 2 are identical; the vector env is identical to a single env. With the bridge off, the Phase 0 hash sequence is still identical.
- Stress: 8 envs, 3200 episodes in total and 9.6M ticks (lms1 random, dm6 random, lms1 chase). No crash, desync, NaN/Inf, out-of-range value or event inconsistency. 636 suicides (death tile) occurred naturally on dm6. Server RSS is stable after warm-up, Python ~45 MB.
- Payload: 2340 + 16 bytes per env step (Phase 1: 252 + 16). BuildObs takes 1.4-3 µs per step on the server.
- Throughput, back to back under the same load (`bench/results/2026-10-19-phase1-vs-phase2.csv`, load 10-15):

| env | Phase 1, process per env | Phase 2, process per env | Phase 2, single process |
|---|---|---|---|
| 1 | 37,636 | 30,036 | 29,944 |
| 4 | 119,538 | 94,653 | 54,670 |
| 8 | 183,594 | 141,377 | 84,947 |
| 12 | 212,709 | 164,822 | 90,660 |

With the same layout, Phase 2 is about 20-23% slower: ~20 µs per step of decoding in Python (grid bits, copies) and ~2 µs per tick on the server. A single Python process saturates at ~90k ticks/s regardless of env count; Phase 3 will need several worker processes.

## Phase 3: PPO (start of training)

- [x] PPO self-play infrastructure: [twrl/ppo.py](twrl/ppo.py), [scripts/train_ppo.py](scripts/train_ppo.py). One network plays both agents and starts from random weights. A small CNN for the grid, an MLP for the vector, a shared trunk, actor/critic heads. Action heads: direction 3, jump 2, fire 2, hook 2, aim 16 directions.
- [x] Rewards are the env's own weights, unchanged (kill +1, death −1, suicide −2, damage ±0.02). Chase is not used in training.
- [x] For episodes ending at the time limit, bootstrap from the last observation (truncation is not a real end).
- [x] Finite loss/gradient check (stops on NaN/Inf), checkpoint save/load and resume tests: [tests/test_ppo.py](tests/test_ppo.py).
- [x] Behavior measurement: [scripts/eval_policy.py](scripts/eval_policy.py). Measurements do not enter the reward.
- [x] Design analysis (2026-10-20, `bench/results/phase3-design/`): bottleneck, workers, aim representation, reward sparsity, fixed evaluation, checkpoint/resume. Details below.
- [ ] Long training, memory (GRU), opponent pool.

### Short trial (2026-10-20, `bench/results/phase3-short/`)

lms1, 8 envs, 400 updates, 819,200 agent steps (~410k env steps, ~2.3 hours of game time), 12.5 minutes, CPU. **Not evidence of learning**; it only shows the loop works.

- All loss, gradient and KL values are finite; grad norm at most 1.04 (clip 0.5), approx KL at most 0.018. After a checkpoint is restored, the network is bit-identical.
- Entropy fell from 5.95 (uniform) to ~5.0-5.3: the policy moves away from randomness (jump 50% → 62%, fire 50% → 43%).
- Evaluation, 64 agent-episodes with a fixed seed, stochastic sampling:

| | start (ckpt_0000) | final | uniform random |
|---|---|---|---|
| kills / episode | 0.000 | 0.094 | 0.000 |
| damage dealt / episode | 2.36 | 4.44 | 1.19 |
| hit rate (damage/shot) | 0.021 | 0.039 | 0.011 |
| opponent visible | 0.62 | 0.68 | 0.63 |
| line of fire clear while visible | 0.47 | 0.55 | 0.38 |
| aim–opponent direction cos | 0.002 | 0.014 | 0.005 |

- Head-to-head match (64 episodes on each side): the final policy is **not superior** to the start or to random (damage 1.52 vs 1.66, 1.19 vs 1.36; kills 0 vs 0.016, 0.016 vs 0). The difference is at noise level.
- Interpretation: in self-play, encounters and damage may have risen slightly, but it has not learned to aim (cos ≈ 0) and has no advantage over the opponents. The sample is small and no significance test was run. The reward is very sparse: the mean reward per step is ~0.

### Design analysis (2026-10-20)

- **The bottleneck is the neural network computation.** Per agent step with 8 envs: inference ~107 µs, env (simulation + bridge) ~10 µs, PPO update ~555 µs. During collection, the 8 servers use 0.12 cores in total. 63% of the network is the first two convolutions at full resolution (25×33).
- **Worker processes bring no benefit:** on the same 6 cores, collection speeds up by at most 12% (`worker_collect.json`), the update not at all. Not implemented. Increasing the env count (the inference batch) is more effective.
- **`torch.compile`** speeds up the update 1.7×, with numbers equal to eager within ~1e-7; added as `--compile`. 6 torch threads is best (SMT slows it down at 12).
- **End to end** (`throughput.csv`): 8 envs eager 1523, 16 envs compile 2344, 32 envs compile 2503 agent steps/s.
- **Aim** (`aim_study.json`): even with perfect aim in real states, the fraction of hittable shots is 40% with 16 directions, 69% with 32, 96% with 64, and 100% with continuous aim. In a bandit learning test, a 2D Gaussian vector reaches 98% hits at 150k samples, categoricals 14-21%, a Gaussian over the angle 31%. Recommendation: a 2D Gaussian aim vector (the decision is the user's, not changed yet).
- **Reward sparsity** (`reward_sparsity.json`, reward unchanged): a nonzero reward occurs in 0.24% of agent steps (random) up to 0.77% (end of the short training); kills are 1 per 50k to 1 per 7.7k agent steps; 66-84% of 128-step agent segments carry no signal; the untrained critic fills the advantage with noise.
- **Fixed evaluation:** [twrl/evaluation.py](twrl/evaluation.py), [scripts/eval_suite.py](scripts/eval_suite.py). A fixed seed set, two-sided against random / ckpt_0000 / the previous checkpoint, self-play behavior; evaluating the same checkpoint twice gives identical JSON. ~130 s per checkpoint.
- **Checkpoint/resume:** model, optimizer, update, agent steps, RNGs (sampling, minibatch, torch global), the env seed stream, PPO settings and run arguments; atomic writes. Resume after `SIGKILL` with `--resume` was tested. Episodes cut in half do not continue (a new episode starts), so a resumed run is not bit-identical to an uninterrupted one.

## Phase 4: Full game and self-play

- [ ] Weapon switching (action + current weapon features), pickups, larger maps.
- [ ] Checkpoint pool: opponent sampling (80% latest, 20% older), Elo tracking, champion.
- [ ] Memory for an out-of-view opponent: GRU or frame stacking.
- [ ] Measure the effect of the auxiliary damage reward (on/off comparison).

Acceptance criterion: a new checkpoint beats the old pool consistently, with no policy collapse.

## Phase 5: Watching and evaluation

- [x] Live spectator (SSH relay) and demo recording: [WATCH.md](WATCH.md).
- [ ] A watch mode that plays the policy at normal speed, and play against a human.
- [ ] Match recordings, score and behavior metrics (hit rate, hook usage, dodging).

## Open risks

- The ticks/s number from Phase 0 (the decision gate above).
- Reward sparsity: tmlp (the earlier project) most likely struggled because of this. No behavior-teaching shaping is used; the first measures are a small map, a single weapon and the auxiliary damage reward. If that is not enough, a curriculum (map, spawn distance) is tried first; shaping would be a separate decision.
- Discrete aim bins may limit fine accuracy. If needed, switch to a continuous action.
- The tmlp patch cannot be ported directly to 0.7.5 (tmlp is based on 0.6).
