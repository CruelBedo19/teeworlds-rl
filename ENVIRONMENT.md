# The RL environment (Phase 2)

Teeworlds is the simulator. A patched `teeworlds_srv` runs two server-side players and talks to Python over a
unix socket in lockstep. Python turns the answers into fixed-size numpy arrays that batch straight into
PyTorch. Nothing on the server decides how to play.

```
VecTeeworldsEnv (one Python process)
  ├─ TeeworldsEnv ── unix socket ── teeworlds_srv  (sv_rl_mode 2, two agents AI-1/AI-2)
  ├─ TeeworldsEnv ── unix socket ── teeworlds_srv
  └─ ...                                              one server process per environment
```

**Neither the random policy nor the chase helper is the AI.** `random_action(s)` is uniform noise and
`twrl.testing.chase_actions` is a scripted diagnostic that reads privileged positions and shoots into
walls. Both exist to exercise the bridge, tests and benchmarks only. The learned policy will start
from random network weights (Phase 3).

## Server side

Patches in `patches/` on top of Teeworlds 0.7.5 (`git am patches/*.patch`):

| Patch | What |
|---|---|
| 0001 | `sv_rl_mode` 0/1/2: realtime, fast ticks, fast ticks without snapshots |
| 0002 | the bridge (`src/game/server/rlenv.*`), protocol v1 |
| 0003 | agent names AI-1/AI-2, demos finished on close |
| 0004 | protocol v2: spatial observation, projectiles, field of view, seeded spawns |

Without `sv_rl_socket` none of the bridge code runs and the server behaves like vanilla 0.7.5
(Phase 0 state hashes are unchanged, see `bench/results/`).

## Protocol v2

Little endian, 16 byte header `magic "TWRL", u16 version=2, u16 type, u32 seq, u32 size`, then the payload.
Requests carry an increasing `seq`; the answer repeats it. The trainer sends `RESET`, `STEP`, `CLOSE`
(and `DEBUG_KILL` in tests); the server answers each with `OBS`, or with `ERROR` right before shutting
down. `HELLO` is sent once on connect.

- `HELLO` (76 bytes + map): num_agents, vec_dim=67, grid_width=33, grid_height=25, action_size=24,
  action_ticks, tick_speed=50, episode_ticks, map_width, map_height, num_spawn_points, map name; then
  the map's whole collision grid, `map_width*map_height` bytes. Sent once, used by tests and tools.
- `RESET` (8 bytes): u32 seed, u32 reserved.
- `STEP` (48 bytes): two actions.
- `OBS` (2340 bytes): u32 episode_id, u32 episode_tick, u32 server_tick, u8 terminated, u8 truncated,
  u16 flags, u32 reserved, then per agent 1160 bytes:
  - `events` i32[6]: kills, deaths, suicides, damage_dealt, damage_taken since the previous OBS, and score
  - raw info (40 bytes): pos f32[2], vel f32[2], alive, health, armor, weapon, ammo, hook_state.
    **Privileged, for logging and tests. Never a policy input.**
  - `vec` f32[67]: the policy vector
  - `grid` u8[828]: 33×25 collision bytes, row by row from the top, 3 bytes padding

`twrl/protocol.py` holds the same layout as numpy dtypes (`OBS_DTYPE`).

## Observation

Per agent, both fixed size:

```
obs["vec"]   float32 (2, 67)                        single env
obs["grid"]  uint8   (2, 3, 25, 33)  channels       single env
VecTeeworldsEnv: a leading env axis, (N, 2, 67) and (N, 2, 3, 25, 33); with grid_format="codes"
the grid is (N, 2, 25, 33) collision bytes (bit 0 solid, bit 1 death, bit 2 no-hook), which is
cheaper to move and can be expanded on the GPU.
```

### Grid

An egocentric window of 33×25 tiles (1 tile = 32 world units) around the agent's own tile, which is
column 16, row 12. It covers ±16 tiles horizontally and ±12 vertically; the hook reaches about 12 tiles
(380 units). Channels:

| Channel | Meaning |
|---|---|
| 0 solid | collision for movement, bullets and hooks |
| 1 death | kills on touch |
| 2 nohook | solid that the hook does not grab (always also solid) |

These are the only collision kinds the 0.7.5 game layer has after `CCollision::Init`. Outside the map
the border tiles repeat, exactly like the game's own collision lookup. Only static geometry is in the
grid; players, projectiles and pickups are in the vector. A dead agent's grid is all zero.

### Vector

Everything is scaled to roughly [-1, 1] and clamped; the largest possible magnitude is 5. A dead
agent's vector is zero except `episode_progress`.

| Index | Name | Scaling |
|---|---|---|
| 0 | alive | 0/1 |
| 1–2 | vel_x, vel_y | units per tick / 32, clamped to ±5 |
| 3–4 | tile_offset_x, tile_offset_y | position inside the own tile, [0, 1) |
| 5 | health | /10 |
| 6 | armor | /10 |
| 7–12 | weapon one-hot | hammer, gun, shotgun, grenade, laser, ninja |
| 13 | ammo | /10, 1 for unlimited |
| 14 | reload | ticks /50, clamped to 1 |
| 15 | grounded | 0/1 |
| 16–17 | jumped, air_jumped | the two `m_Jumped` bits |
| 18–21 | hook one-hot | idle, flying, grabbed, retracting |
| 22–23 | hook_dx, hook_dy | hook position − own position, /380, clamped to ±1.5; 0 when idle |
| 24 | hooking_opp | own hook holds the opponent |
| 25 | opp_visible | opponent alive and inside the view rectangle |
| 26–27 | opp_dx, opp_dy | opponent − own position, /1000 and /800 |
| 28–29 | opp_rel_vel_x, opp_rel_vel_y | opponent − own velocity, /32, clamped to ±5 |
| 30–35 | opp weapon one-hot | |
| 36–39 | opp hook one-hot | |
| 40 | opp_hooking_me | |
| 41 | opp_line_of_fire | no solid tile between the two (`CCollision::IntersectLine`) |
| 42–65 | 4 projectile slots × (present, dx/1000, dy/800, vel_x/32, vel_y/32, hostile) | nearest first |
| 66 | episode_progress | episode_tick / episode_ticks |

Exact names are `twrl.protocol.VEC_NAMES`; `VEC_INDEX` maps them to indices.

Not in the vector: absolute map coordinates (the grid gives local geometry), the opponent's health,
armor and ammo (a human player cannot see them either), pickups and lasers.

### Field of view

Opponent and projectile features are filled only inside the rectangle a human client is sent entities
for: |dx| ≤ 1000, |dy| ≤ 800 and distance ≤ 1100 (`CEntity::NetworkClipped`). Outside it they read as
zero. This is the same information a human has, no more.

Walls do not hide players in Teeworlds: a human sees an opponent behind a wall on screen. Hiding them
from the agent would make it weaker than a human, so visibility is not occluded. Whether a shot would
get through is a separate bit, `opp_line_of_fire`, computed with the same line test bullets use.
The grid only covers part of the view, so the network could not work this out by itself.

### Projectiles

Up to 4 projectiles in view, nearest first, slots filled from the front and zero after. Position and
velocity come from the projectile's own trajectory (`CProjectile::GetPos`) at the last simulated tick,
so curved grenade paths are right too. `hostile` is 1 unless the agent fired it. Lasers are not
included: agents cannot switch to the laser yet.

## Action

Unchanged from Phase 1, two per step, `int32[2, 6]`:

| Field | Values |
|---|---|
| direction | -1 left, 0 none, 1 right |
| jump | 0/1 |
| fire | 0/1, held; a press counts on 0→1 like the client |
| hook | 0/1 |
| target_x, target_y | aim relative to the character, like the mouse; (0, 0) means up |

Each action is held for `action_ticks` ticks (default 2, so 25 decisions per second at 50 ticks/s).
Inputs go through `CPlayer::OnDirectInput` and `OnPredictedInput` like a network client.

Weapon switching is not in the action yet. Agents spawn with hammer and gun, with the gun active, and
pickups do not switch weapons. This is fine for learning movement and gun fights. It belongs in the phase
that trains full combat, together with the weapon features that are already in the vector.

## Rewards

The server sends event counts, Python weights them (`DEFAULT_REWARD_WEIGHTS` in `twrl/env.py`):

| Event | Weight |
|---|---|
| kill | +1 |
| death (killed by the opponent) | −1 |
| suicide (self or world) | −2 |
| damage dealt, per health/armor point | +0.02 |
| damage taken, per point | −0.02 |

No shaping: nothing rewards approaching, aiming or positions. The weights are a starting point for
Phase 3. Kill, death and the damage terms are symmetric between the agents, but suicide only affects
one of them, so the total is not zero-sum.

## Episodes, resets and seeds

- An episode lasts `episode_ticks` (default 3000 = 60 s of game time) and ends with `truncated`. Teeworlds
  DM has no terminal state, so `terminated` is always false. Dying is not the end: the agent respawns
  after the normal delay.
- `reset(seed)` restarts the match the way the `restart` command does: world, projectiles, pickups,
  scores and respawn timers. The server process keeps running. The first observation comes once both
  agents have spawned, about 28 ticks later.
- **Spawns:** natively DM picks the spawn point farthest from the opponent, which is the same every
  episode. In RL mode the first spawn of each episode puts the agents on two different map spawn
  points chosen by the env's own seeded generator. Later respawns use the native rule. Map spawn points
  are placed by the map author, so they are valid by construction (the tests also check that they are
  not solid or death tiles). lms1 has 18 spawn points, dm1 has 8, dm6 has 9.
- **Seeds:** `reset(seed=s)` with s > 0 reseeds the spawn generator (and `rand()`); the same seed gives
  the same start and, with the same actions, the same episode, in the same process or a fresh one.
  `reset()` / seed 0 continues the previous stream. `VecTeeworldsEnv(seed=...)` derives a distinct seed
  for every env and episode with `episode_seed(seed, env, episode)`.

## Python API

```python
from twrl import TeeworldsEnv, VecTeeworldsEnv, Action, random_actions

env = TeeworldsEnv(build_dir, map="lms1")        # mode=2, action_ticks=2, episode_ticks=3000
obs, info = env.reset(seed=1)
obs, rewards, terminated, truncated, info = env.step([Action(direction=1), Action(fire=1)])
env.close()

venv = VecTeeworldsEnv(8, build_dir, seed=1, map="lms1")   # grid_format="channels" | "codes"
obs = venv.reset()                                   # {"vec": (8,2,67), "grid": (8,2,3,25,33)}
obs, rewards, terminated, truncated, infos = venv.step(actions)   # actions (8,2,6) int32, rewards (8,2)
venv.close()
```

The vector env sends all steps before it reads any answer, so the servers simulate in parallel. It
resets finished envs itself: the returned obs then starts the new episode, and
`infos["final_vec"] / ["final_grid"] / ["final_index"]` hold the last obs of the finished episodes (for
bootstrapping truncated returns). `infos["events"]` is (N, 2, 6). `raw_info=True` adds the privileged
raw state as `infos["agents"]`.

Self-play maps naturally onto the agent axis: both agents of every env can be one batch of 2N
for one network, or split between a learner and an opponent.

## Live viewing and demos

`scripts/watch_match.py` runs the same environment at `sv_rl_mode 0` (50 ticks/s) with a relay socket for
a human spectator (see [WATCH.md](WATCH.md)), or records a demo at full speed. Training does not need it.
Spectators only ever connect to a watch server, never to training servers, which run in mode 2
without snapshots.

## Training (Phase 3, early)

`twrl/ppo.py` trains one network for both agents with PPO, from random weights, on the env's own
rewards. `scripts/train_ppo.py` runs a bounded trial and saves `ckpt_0000.pt` (untrained) before
the first update; `--compile` speeds up the update, `--resume` continues from a checkpoint.
`scripts/eval_suite.py` runs the fixed evaluation protocol of `twrl/evaluation.py` for a
checkpoint (seeded starts, uniform random / ckpt_0000 / previous checkpoint, sides swapped).
Needs `torch` (CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
Design measurements: `bench/profile_ppo.py`, `bench/worker_collect.py`, `bench/aim_study.py`,
`bench/reward_sparsity.py`, results in `bench/results/phase3-design/`.

## Tests and benchmarks

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt     # numpy
TWRL_SERVER_DIR=../teeworlds-0.7.5/build-release .venv/bin/python tests/test_env.py -v
.venv/bin/python tests/test_relay.py -v
TWRL_SERVER_DIR=../teeworlds-0.7.5/build-release .venv/bin/python tests/test_ppo.py -v
.venv/bin/python scripts/random_rollout.py --server-dir ../teeworlds-0.7.5/build-release --envs 8 --steps 100000
.venv/bin/python bench/run_env_bench.py --server-dir ../teeworlds-0.7.5/build-release --envs 1 4 8 12
```

## Limitations

- One Python process tops out at about 90k ticks/s however many envs it steps (syscalls and numpy per
  env). One process per env reaches about 165k ticks/s on 12 envs. See `PLAN.md` for the numbers.
- Opponent perception has no memory: once the opponent leaves the view rectangle, the vector says
  nothing about it. A recurrent policy or frame stacking can add memory in Phase 3.
- No pickups, lasers or weapon switching in the observation or action.
- The grid shows static geometry only.
- Determinism holds for one binary on one machine; float results may differ on other compilers or CPUs.
- Changing the map while the bridge runs closes the bridge.
