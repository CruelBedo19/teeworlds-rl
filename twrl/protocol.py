"""Wire format of the teeworlds_srv RL bridge (src/game/server/rlenv.h, protocol version 2).

Every message is a 16 byte header followed by a payload, all little endian. The trainer sends
RESET, STEP, CLOSE and DEBUG_KILL; the server answers each of them except CLOSE with OBS, or
with ERROR right before it shuts down. HELLO is sent once on connect and carries the map's
collision grid. OBS has a fixed size and is read with numpy without copying.
"""

import struct

import numpy as np

MAGIC = 0x4C525754  # "TWRL"
VERSION = 2

MSG_HELLO = 1
MSG_RESET = 2
MSG_STEP = 3
MSG_CLOSE = 4
MSG_OBS = 5
MSG_ERROR = 6
MSG_DEBUG_KILL = 7

ERROR_PROTOCOL = 1
ERROR_STATE = 2
ERROR_DEBUG_DISABLED = 3

FLAG_GAME_NOT_RUNNING = 1

NUM_AGENTS = 2
NUM_PROJECTILES = 4
VEC_DIM = 67
GRID_WIDTH = 33  # tiles, the agent's own tile is column 16, row 12
GRID_HEIGHT = 25
GRID_CELLS = GRID_WIDTH * GRID_HEIGHT
GRID_BYTES = (GRID_CELLS + 3) // 4 * 4
TILE_SIZE = 32  # world units per tile

# collision bits of a grid cell, as CCollision stores them
TILE_SOLID = 1
TILE_DEATH = 2
TILE_NOHOOK = 4  # always together with TILE_SOLID
GRID_CHANNELS = ("solid", "death", "nohook")

HEADER = struct.Struct("<IHHII")  # magic, version, type, seq, payload size
HELLO = struct.Struct("<11I32s")  # see CRlEnv::CHello, followed by map_width*map_height bytes
RESET = struct.Struct("<II")  # seed (0 = continue the previous random stream), reserved
ACTION = struct.Struct("<6i")  # direction, jump, fire, hook, target_x, target_y
ERROR = struct.Struct("<I64s")  # code, message
DEBUG_KILL = struct.Struct("<I")  # agent index

EVENT_NAMES = ("kills", "deaths", "suicides", "damage_dealt", "damage_taken", "score")

AGENT_DTYPE = np.dtype([
    ("events", "<i4", (6,)),  # EVENT_NAMES, counted since the previous OBS; score is the total
    # privileged raw state for logging and tests, never an input to the policy
    ("pos", "<f4", (2,)), ("vel", "<f4", (2,)),
    ("alive", "<i4"), ("health", "<i4"), ("armor", "<i4"), ("weapon", "<i4"), ("ammo", "<i4"), ("hook_state", "<i4"),
    ("vec", "<f4", (VEC_DIM,)),
    ("grid", "u1", (GRID_BYTES,)),
])
OBS_DTYPE = np.dtype([
    ("episode_id", "<u4"), ("episode_tick", "<u4"), ("server_tick", "<u4"),
    ("terminated", "u1"), ("truncated", "u1"), ("flags", "<u2"), ("reserved", "<u4"),
    ("agents", AGENT_DTYPE, (NUM_AGENTS,)),
])
OBS_SIZE = OBS_DTYPE.itemsize

assert HEADER.size == 16 and HELLO.size == 76 and ACTION.size == 24 and ERROR.size == 68
assert AGENT_DTYPE.itemsize == 24 + 40 + 4 * VEC_DIM + GRID_BYTES and OBS_SIZE == 20 + 2 * AGENT_DTYPE.itemsize

WEAPONS = ("hammer", "gun", "shotgun", "grenade", "laser", "ninja")
HOOK_STATES = ("idle", "flying", "grabbed", "retracting")

# The policy vector of one agent, see CRlEnv::BuildAgentObs. Everything is scaled to roughly
# [-1, 1] and clamped. A dead agent's vector is zero apart from episode_progress; opponent and
# projectile values are zero unless they are within the view rectangle of a human player.
VEC_NAMES = (
    ("alive", "vel_x", "vel_y", "tile_offset_x", "tile_offset_y", "health", "armor")
    + tuple("weapon_" + w for w in WEAPONS)
    + ("ammo", "reload", "grounded", "jumped", "air_jumped")
    + tuple("hook_" + h for h in HOOK_STATES)
    + ("hook_dx", "hook_dy", "hooking_opp")
    + ("opp_visible", "opp_dx", "opp_dy", "opp_rel_vel_x", "opp_rel_vel_y")
    + tuple("opp_weapon_" + w for w in WEAPONS)
    + tuple("opp_hook_" + h for h in HOOK_STATES)
    + ("opp_hooking_me", "opp_line_of_fire")
    + tuple("proj%d_%s" % (i, f) for i in range(NUM_PROJECTILES)
            for f in ("present", "dx", "dy", "vel_x", "vel_y", "hostile"))
    + ("episode_progress",)
)
assert len(VEC_NAMES) == VEC_DIM
VEC_INDEX = {name: i for i, name in enumerate(VEC_NAMES)}

# scale used for each feature, to turn a value back into game units in tests and tools
VIEW_HALF_WIDTH = 1000.0  # CEntity::NetworkClipped
VIEW_HALF_HEIGHT = 800.0
VIEW_RADIUS = 1100.0
VEL_SCALE = 32.0
HOOK_SCALE = 380.0
