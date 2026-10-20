"""Two agent Teeworlds environments backed by teeworlds_srv processes.

    env = TeeworldsEnv("/path/to/build-release", map="lms1")
    obs, info = env.reset(seed=1)
    obs, rewards, terminated, truncated, info = env.step(actions)   # actions: (2, 6) int
    env.close()

    venv = VecTeeworldsEnv(8, "/path/to/build-release", map="lms1", seed=1)
    obs = venv.reset()
    obs, rewards, terminated, truncated, infos = venv.step(actions)  # actions: (8, 2, 6) int
    venv.close()

Each environment listens on a unix socket, starts the server with sv_rl_socket pointing at it
and drives it in lockstep: a step holds the actions for action_ticks server ticks.

Observations are numpy arrays with a leading agent axis (and an env axis for the vector env):
    obs["vec"]   float32 (2, VEC_DIM)               see protocol.VEC_NAMES
    obs["grid"]  uint8   (2, 3, GRID_HEIGHT, GRID_WIDTH)  solid, death, nohook around the agent
Everything else in info (raw positions, health, ...) is for logging and tests, not for policies.
"""

import math
import os
import shutil
import socket
import subprocess
import tempfile
from typing import NamedTuple

import numpy as np

from . import protocol as P

# kill +1, death -1, suicide -2, damage +-0.02 per health/armor point, in protocol.EVENT_NAMES order
DEFAULT_REWARD_WEIGHTS = {"kills": 1.0, "deaths": -1.0, "suicides": -2.0, "damage_dealt": 0.02, "damage_taken": -0.02}

_GRID_SHIFTS = np.arange(len(P.GRID_CHANNELS), dtype=np.uint8).reshape(-1, 1, 1)


class EnvError(Exception):
    pass


class ProtocolError(EnvError):
    """The server rejected a message (it shuts down after sending the error)."""

    def __init__(self, code, message):
        super().__init__(f"server error {code}: {message}")
        self.code = code


class EnvTimeout(EnvError):
    pass


class EnvCrashed(EnvError):
    """The server went away without an error message."""


class Action(NamedTuple):
    direction: int = 0  # -1 left, 0 none, 1 right
    jump: int = 0
    fire: int = 0  # held; a press is counted when it goes from 0 to 1
    hook: int = 0
    target_x: int = 1  # aim, relative to the character like the client's mouse position
    target_y: int = 0


def random_action(rng):
    """One uniformly random action from a random.Random. Diagnostic only, not a policy."""
    angle = rng.uniform(-math.pi, math.pi)
    return Action(
        direction=rng.choice((-1, 0, 1)),
        jump=rng.randrange(2),
        fire=rng.randrange(2),
        hook=rng.randrange(2),
        target_x=int(round(math.cos(angle) * 256)),
        target_y=int(round(math.sin(angle) * 256)),
    )


def random_actions(rng, shape):
    """Uniformly random actions as an int32 array of shape (*shape, 6) from a numpy Generator."""
    out = np.empty(tuple(shape) + (6,), dtype=np.int32)
    out[..., 0] = rng.integers(-1, 2, size=shape)
    out[..., 1:4] = rng.integers(0, 2, size=tuple(shape) + (3,))
    angle = rng.uniform(-math.pi, math.pi, size=shape)
    out[..., 4] = np.rint(np.cos(angle) * 256)
    out[..., 5] = np.rint(np.sin(angle) * 256)
    return out


def unpack_grid(grid_bytes):
    """(..., GRID_BYTES) uint8 collision bytes -> (..., 3, GRID_HEIGHT, GRID_WIDTH) uint8 channels."""
    cells = grid_bytes[..., :P.GRID_CELLS].reshape(grid_bytes.shape[:-1] + (1, P.GRID_HEIGHT, P.GRID_WIDTH))
    return (cells >> _GRID_SHIFTS) & 1


def free_udp_ports(count):
    """Ports from a single freeport call: concurrent freeport calls can hand out the same port."""
    try:
        out = subprocess.run(["freeport", "-u", "-n", str(count)], check=True, capture_output=True, text=True).stdout
        return [int(p) for p in out.split()], True
    except (OSError, subprocess.CalledProcessError):
        socks = []
        for _ in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("127.0.0.1", 0))
            socks.append(s)
        ports = [s.getsockname()[1] for s in socks]
        for s in socks:
            s.close()
        return ports, False


def release_udp_ports(ports):
    subprocess.run(["freeport", "--release", ",".join(map(str, ports)), "--proto", "udp"], capture_output=True)


class TeeworldsEnv:
    def __init__(self, server_dir, map="dm1", mode=2, action_ticks=2, episode_ticks=3000, timeout=30.0,
                 debug=False, extra_commands=(), reward_weights=None, server_binary="teeworlds_srv", port=None):
        weights = DEFAULT_REWARD_WEIGHTS if reward_weights is None else reward_weights
        self.reward_weights = np.array([weights.get(k, 0.0) for k in P.EVENT_NAMES[:5]], dtype=np.float32)
        self.timeout = timeout
        self._seq = 0
        self._header = memoryview(bytearray(P.HEADER.size))
        self._stage = memoryview(bytearray(P.HEADER.size + P.OBS_SIZE))
        self._step_msg = bytearray(P.HEADER.size + P.NUM_AGENTS * P.ACTION.size)
        self._sock = None
        self._proc = None
        self._closed = False
        self._dir = tempfile.mkdtemp(prefix="twrl-")
        self.socket_path = os.path.join(self._dir, "env.sock")
        self.log_path = os.path.join(self._dir, "server.log")
        if port is None:
            ports, self._port_reserved = free_udp_ports(1)
            self.port = ports[0]
        else:
            self.port, self._port_reserved = port, False

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(self.socket_path)
            listener.listen(1)
            listener.settimeout(1.0)
            commands = [
                f"sv_port {self.port}", "bindaddr 127.0.0.1", "sv_register 0", f"sv_map {map}",
                "sv_scorelimit 0", "sv_timelimit 0", f"sv_rl_mode {mode}",
                f"sv_rl_socket {self.socket_path}", f"sv_rl_action_ticks {action_ticks}",
                f"sv_rl_episode_ticks {episode_ticks}", f"sv_rl_timeout {int(timeout * 1000)}",
                f"sv_rl_debug {1 if debug else 0}",
            ] + list(extra_commands)
            self._log = open(self.log_path, "w")
            self._proc = subprocess.Popen(
                [os.path.join(os.path.abspath(server_dir), server_binary), "; ".join(commands)],
                cwd=server_dir, stdout=self._log, stderr=subprocess.STDOUT,
            )
            waited = 0.0
            while self._sock is None:
                try:
                    self._sock, _ = listener.accept()
                except socket.timeout:
                    waited += 1.0
                    if self._proc.poll() is not None:
                        raise EnvCrashed(f"server exited with {self._proc.returncode} before connecting\n{self.log_tail()}")
                    if waited >= 10.0:
                        raise EnvTimeout(f"server did not connect within 10 s\n{self.log_tail()}")
            self._sock.settimeout(timeout)
            msg_type, _, payload = self._recv()
            if msg_type != P.MSG_HELLO:
                raise EnvError(f"expected HELLO, got message type {msg_type}")
            hello = P.HELLO.unpack_from(payload)
            (num_agents, vec_dim, grid_w, grid_h, action_size, self.action_ticks, self.tick_speed,
             self.episode_ticks, map_w, map_h, self.num_spawn_points, map_name) = hello
            if (num_agents, vec_dim, grid_w, grid_h, action_size) != (P.NUM_AGENTS, P.VEC_DIM, P.GRID_WIDTH, P.GRID_HEIGHT, P.ACTION.size):
                raise EnvError(f"server protocol does not match this client: {hello}")
            self.map = map_name.split(b"\0", 1)[0].decode()
            # the map's collision, one byte per tile like the grid cells; for tests and tools
            self.map_collision = np.frombuffer(payload, dtype=np.uint8, count=map_w * map_h,
                                               offset=P.HELLO.size).reshape(map_h, map_w)
        except BaseException:
            self.close()
            raise
        finally:
            listener.close()

    @property
    def server_pid(self):
        return self._proc.pid if self._proc else None

    def log_tail(self, lines=50):
        try:
            with open(self.log_path) as f:
                return "".join(f.readlines()[-lines:])
        except OSError:
            return ""

    # -- one environment --

    def reset(self, seed=None):
        """seed > 0 reseeds the spawn choice (and rand()); None/0 continues the previous stream."""
        self.send_reset(seed)
        return self.recv_obs()

    def step(self, actions):
        self.send_step(actions)
        obs, info = self.recv_obs()
        return obs, self.rewards(info["events"]), info["terminated"], info["truncated"], info

    def debug_kill(self, agent):
        """Kill an agent like the in-game kill command. Needs debug=True; the world does not tick."""
        self._send(P.MSG_DEBUG_KILL, P.DEBUG_KILL.pack(agent))
        return self.recv_obs()

    def rewards(self, events):
        """events (..., 2, EVENT_NAMES) -> rewards (..., 2) with the env's weights; score is not rewarded."""
        return events[..., :5].astype(np.float32) @ self.reward_weights

    # -- split send/receive, so a vector of environments can step in parallel --

    def send_reset(self, seed=None):
        self._send(P.MSG_RESET, P.RESET.pack(seed or 0, 0))

    def send_step(self, actions):
        data = np.ascontiguousarray(actions, dtype="<i4")
        if data.shape != (P.NUM_AGENTS, 6):
            raise ValueError(f"actions must have shape ({P.NUM_AGENTS}, 6), got {data.shape}")
        if self._sock is None:
            raise EnvError("environment is closed")
        self._seq += 1
        msg = self._step_msg
        P.HEADER.pack_into(msg, 0, P.MAGIC, P.VERSION, P.MSG_STEP, self._seq, len(msg) - P.HEADER.size)
        msg[P.HEADER.size:] = data.tobytes()
        try:
            self._sock.sendall(msg)
        except OSError as e:
            raise self._gone(f"send failed: {e}") from e

    def recv_obs_into(self, out):
        """Receive the next OBS into `out`, a writable buffer of protocol.OBS_SIZE bytes.

        Header and payload usually arrive in one recv, which matters when stepping many envs."""
        if self._sock is None:
            raise EnvError("environment is closed")
        stage = self._stage
        got = 0
        while got < P.HEADER.size:
            got += self._recv_some(stage[got:])
        magic, version, msg_type, seq, size = P.HEADER.unpack_from(stage)
        if magic != P.MAGIC or version != P.VERSION:
            raise EnvError(f"bad header from server: magic={magic:#x} version={version}")
        if msg_type != P.MSG_OBS or size != P.OBS_SIZE:
            if msg_type == P.MSG_ERROR and size == P.ERROR.size:
                while got < P.HEADER.size + size:
                    got += self._recv_some(stage[got:P.HEADER.size + size])
                code, message = P.ERROR.unpack_from(stage, P.HEADER.size)
                raise ProtocolError(code, message.split(b"\0", 1)[0].decode(errors="replace"))
            raise EnvError(f"expected OBS, got type {msg_type} with {size} bytes")
        if seq != self._seq:
            raise EnvError(f"answer for request {seq}, expected {self._seq}")
        while got < len(stage):
            got += self._recv_some(stage[got:])
        out[:] = stage[P.HEADER.size:]

    def _recv_some(self, view):
        try:
            count = self._sock.recv_into(view)
        except socket.timeout as e:
            raise EnvTimeout(f"no answer from the server within {self.timeout} s") from e
        except OSError as e:
            raise self._gone(f"receive failed: {e}") from e
        if not count:
            raise self._gone("server closed the connection")
        return count

    def recv_raw(self):
        """The next OBS as a numpy record of protocol.OBS_DTYPE, plus its payload bytes."""
        msg_type, seq, payload = self._recv()
        if msg_type != P.MSG_OBS or len(payload) != P.OBS_SIZE:
            raise EnvError(f"expected OBS, got type {msg_type} with {len(payload)} bytes")
        if seq != self._seq:
            raise EnvError(f"answer for request {seq}, expected {self._seq}")
        return np.frombuffer(payload, dtype=P.OBS_DTYPE)[0], payload

    def recv_obs(self):
        raw, payload = self.recv_raw()
        agents = raw["agents"]
        obs = {"vec": agents["vec"].copy(), "grid": unpack_grid(agents["grid"])}
        info = {
            "episode_id": int(raw["episode_id"]), "episode_tick": int(raw["episode_tick"]),
            "server_tick": int(raw["server_tick"]), "terminated": bool(raw["terminated"]),
            "truncated": bool(raw["truncated"]), "flags": int(raw["flags"]),
            "events": agents["events"].copy(), "agents": agents, "payload": payload,
        }
        return obs, info

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._sock is not None:
            try:
                self._send(P.MSG_CLOSE, b"")
            except (OSError, EnvError):
                pass
            self._sock.close()
            self._sock = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if hasattr(self, "_log"):
            self._log.close()
        if self._port_reserved:
            release_udp_ports([self.port])
        shutil.rmtree(self._dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- wire --

    def _send(self, msg_type, payload):
        if self._sock is None:
            raise EnvError("environment is closed")
        self._seq += 1
        try:
            self._sock.sendall(P.HEADER.pack(P.MAGIC, P.VERSION, msg_type, self._seq, len(payload)) + payload)
        except OSError as e:
            raise self._gone(f"send failed: {e}") from e

    def _recv_exact_into(self, view, n):
        view = memoryview(view)
        got = 0
        while got < n:
            try:
                count = self._sock.recv_into(view[got:], n - got)
            except socket.timeout as e:
                raise EnvTimeout(f"no answer from the server within {self.timeout} s") from e
            except OSError as e:
                raise self._gone(f"receive failed: {e}") from e
            if not count:
                raise self._gone("server closed the connection")
            got += count

    def _recv_exact(self, n):
        buf = bytearray(n)
        self._recv_exact_into(buf, n)
        return bytes(buf)

    def _recv_header(self):
        if self._sock is None:
            raise EnvError("environment is closed")
        self._recv_exact_into(self._header, P.HEADER.size)
        header = P.HEADER.unpack(self._header)
        if header[0] != P.MAGIC or header[1] != P.VERSION:
            raise EnvError(f"bad header from server: magic={header[0]:#x} version={header[1]}")
        return header

    def _check_error(self, msg_type, size):
        if msg_type == P.MSG_ERROR:
            code, message = P.ERROR.unpack(self._recv_exact(size))
            raise ProtocolError(code, message.split(b"\0", 1)[0].decode(errors="replace"))

    def _recv(self):
        _, _, msg_type, seq, size = self._recv_header()
        self._check_error(msg_type, size)
        return msg_type, seq, self._recv_exact(size)

    def _gone(self, reason):
        code = None
        if self._proc is not None:
            try:
                code = self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        return EnvCrashed(f"{reason}; server exit code {code}\n{self.log_tail()}")


def episode_seed(base_seed, env_index, episode):
    """Deterministic, distinct reset seeds for env `env_index`'s `episode`-th episode."""
    x = (base_seed * 0x9E3779B1 + env_index * 0x85EBCA77 + episode * 0xC2B2AE3D) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    return x or 1


class VecTeeworldsEnv:
    """N environments (N server processes) driven from one Python process.

    step() sends every env its actions before waiting for any answer, so the servers simulate in
    parallel. Finished episodes are reset automatically: the returned obs is then the first obs
    of the new episode and infos["final_vec"/"final_grid"] hold the last obs of the old one for
    the envs where truncated or terminated is set.
    """

    def __init__(self, num_envs, server_dir, seed=1, ports=None, grid_format="channels", raw_info=False, **env_kwargs):
        """grid_format "channels" gives (N, 2, 3, H, W) 0/1 planes; "codes" gives the (N, 2, H, W) collision
        bytes (solid 1, death 2, no hook 4) and leaves expanding them to the model, which is cheaper.
        raw_info adds infos["agents"], the privileged raw state, for logging and diagnostics."""
        if grid_format not in ("channels", "codes"):
            raise ValueError("grid_format must be 'channels' or 'codes'")
        self.num_envs = num_envs
        self.seed = seed
        self.grid_format = grid_format
        self.raw_info = raw_info
        self._episodes = [0] * num_envs
        # the servers' answers land here directly, one OBS record per env
        self._buf = np.zeros(num_envs, dtype=P.OBS_DTYPE)
        self._rows = [memoryview(row) for row in self._buf.view(np.uint8).reshape(num_envs, P.OBS_SIZE)]
        # field views are slow to make, so make them once
        agents = self._buf["agents"]
        self._agents, self._vec, self._grid, self._events = agents, agents["vec"], agents["grid"], agents["events"]
        self._codes = self._grid[..., :P.GRID_CELLS].reshape(num_envs, P.NUM_AGENTS, P.GRID_HEIGHT, P.GRID_WIDTH)
        self._terminated, self._truncated = self._buf["terminated"], self._buf["truncated"]
        if ports is None:
            self._ports, self._ports_reserved = free_udp_ports(num_envs)
        else:
            self._ports, self._ports_reserved = list(ports), False
        self.envs = []
        try:
            for port in self._ports:
                self.envs.append(TeeworldsEnv(server_dir, port=port, **env_kwargs))
        except BaseException:
            self.close()
            raise
        self.action_ticks = self.envs[0].action_ticks
        self.reward_weights = self.envs[0].reward_weights

    def _reset_seed(self, i):
        self._episodes[i] += 1
        return episode_seed(self.seed, i, self._episodes[i])

    def _reset_envs(self, indices):
        for i in indices:
            self.envs[i].send_reset(self._reset_seed(i))
        for i in indices:
            self.envs[i].recv_obs_into(self._rows[i])

    def _grids(self, index=slice(None)):
        if self.grid_format == "codes":
            return self._codes[index].copy()
        return unpack_grid(self._grid[index])

    def reset(self):
        self._reset_envs(range(self.num_envs))
        return {"vec": self._vec.copy(), "grid": self._grids()}

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int32)
        for env, a in zip(self.envs, actions):
            env.send_step(a)
        for env, row in zip(self.envs, self._rows):
            env.recv_obs_into(row)
        events = self._events.copy()
        terminated = self._terminated != 0
        truncated = self._truncated != 0
        infos = {"events": events}
        ended = terminated | truncated
        if ended.any():
            done = np.flatnonzero(ended)
            infos["final_vec"], infos["final_grid"], infos["final_index"] = self._vec[done], self._grids(done), done
            self._reset_envs(done)
        if self.raw_info:
            infos["agents"] = self._agents.copy()  # belongs to the returned obs
        obs = {"vec": self._vec.copy(), "grid": self._grids()}
        return obs, events[..., :5].astype(np.float32) @ self.reward_weights, terminated, truncated, infos

    def close(self):
        for env in self.envs:
            env.close()
        if self._ports_reserved:
            release_udp_ports(self._ports)
        self.envs = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def trace_key(payload):
    """The parts of an OBS payload that should repeat across episodes: drops episode id and server tick."""
    return payload[4:8] + payload[12:]
