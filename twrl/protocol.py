"""Wire format of the teeworlds_srv RL bridge (src/game/server/rlenv.h, protocol version 1).

Every message is a 16 byte header followed by a fixed size payload, all little endian.
The trainer sends RESET, STEP, CLOSE and DEBUG_KILL; the server answers each of them except
CLOSE with OBS, or with ERROR right before it shuts down. HELLO is sent once on connect.
"""

import struct

MAGIC = 0x4C525754  # "TWRL"
VERSION = 1

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
OBS_DIM = 23

HEADER = struct.Struct("<IHHII")  # magic, version, type, seq, payload size
HELLO = struct.Struct("<6I32s")  # num_agents, obs_dim, action_size, action_ticks, tick_speed, episode_ticks, map
RESET = struct.Struct("<II")  # seed (0 = keep), reserved
ACTION = struct.Struct("<6i")  # direction, jump, fire, hook, target_x, target_y
OBS_HEAD = struct.Struct("<IIIBBHI")  # episode_id, episode_tick, server_tick, terminated, truncated, flags, reserved
AGENT = struct.Struct(f"<6i{OBS_DIM}f")  # kills, deaths, suicides, damage_dealt, damage_taken, score, obs
ERROR = struct.Struct("<I64s")  # code, message
DEBUG_KILL = struct.Struct("<I")  # agent index

OBS_SIZE = OBS_HEAD.size + NUM_AGENTS * AGENT.size

assert HEADER.size == 16 and HELLO.size == 56 and ACTION.size == 24
assert OBS_HEAD.size == 20 and AGENT.size == 24 + 4 * OBS_DIM and ERROR.size == 68

EVENT_NAMES = ("kills", "deaths", "suicides", "damage_dealt", "damage_taken", "score")

# Index of each value in an agent's observation. Positions and velocities are in world units
# (32 per tile) and units per tick, everything else is the raw game value. A dead agent's own
# values are all zero, and the opponent's position is zero whenever either of them is dead.
OBS_NAMES = (
    "alive", "pos_x", "pos_y", "vel_x", "vel_y", "health", "armor", "weapon", "ammo",
    "reload_timer", "grounded", "jumped", "hook_state", "hook_dx", "hook_dy",
    "opp_alive", "opp_dx", "opp_dy", "opp_vel_x", "opp_vel_y", "opp_weapon", "opp_hook_state",
    "episode_progress",
)
assert len(OBS_NAMES) == OBS_DIM
OBS_INDEX = {name: i for i, name in enumerate(OBS_NAMES)}
