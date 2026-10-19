"""Two agent Teeworlds environment backed by one teeworlds_srv process.

    env = TeeworldsEnv("/path/to/build-release")
    obs, info = env.reset(seed=1)
    obs, rewards, terminated, truncated, info = env.step([Action(direction=1), Action(fire=1)])
    env.close()

The environment listens on a unix socket, starts the server with sv_rl_socket pointing at it
and then drives it in lockstep: every step holds the actions for action_ticks server ticks.
Observations are tuples of protocol.OBS_DIM floats, one per agent.
"""

import math
import os
import shutil
import socket
import subprocess
import tempfile
from typing import NamedTuple

from . import protocol as P

# kill +1, death -1, suicide -2, damage +-0.02 per health/armor point; placeholders for phase 2
DEFAULT_REWARD_WEIGHTS = {"kills": 1.0, "deaths": -1.0, "suicides": -2.0, "damage_dealt": 0.02, "damage_taken": -0.02}


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
    angle = rng.uniform(-math.pi, math.pi)
    return Action(
        direction=rng.choice((-1, 0, 1)),
        jump=rng.randrange(2),
        fire=rng.randrange(2),
        hook=rng.randrange(2),
        target_x=int(round(math.cos(angle) * 256)),
        target_y=int(round(math.sin(angle) * 256)),
    )


def _free_udp_port():
    try:
        out = subprocess.run(["freeport", "-u"], check=True, capture_output=True, text=True).stdout
        return int(out.split()[0]), True
    except (OSError, subprocess.CalledProcessError):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port, False


class TeeworldsEnv:
    def __init__(self, server_dir, map="dm1", mode=2, action_ticks=2, episode_ticks=3000, timeout=30.0,
                 debug=False, extra_commands=(), reward_weights=None, server_binary="teeworlds_srv", port=None):
        self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS if reward_weights is None else reward_weights)
        self.timeout = timeout
        self._seq = 0
        self._sock = None
        self._proc = None
        self._closed = False
        self._dir = tempfile.mkdtemp(prefix="twrl-")
        self.socket_path = os.path.join(self._dir, "env.sock")
        self.log_path = os.path.join(self._dir, "server.log")
        # parallel callers should pass ports from one `freeport -u -n N` call: concurrent
        # freeport calls can hand out the same port
        if port is None:
            self.port, self._port_reserved = _free_udp_port()
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
            hello = P.HELLO.unpack(payload)
            if hello[0] != P.NUM_AGENTS or hello[1] != P.OBS_DIM or hello[2] != P.ACTION.size:
                raise EnvError(f"server protocol does not match this client: {hello}")
            self.action_ticks, self.tick_speed, self.episode_ticks = hello[3], hello[4], hello[5]
            self.map = hello[6].split(b"\0", 1)[0].decode()
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

    def reset(self, seed=None):
        self._send(P.MSG_RESET, P.RESET.pack(seed or 0, 0))
        obs, info = self._recv_obs()
        return obs, info

    def step(self, actions):
        if len(actions) != P.NUM_AGENTS:
            raise ValueError(f"need {P.NUM_AGENTS} actions")
        self._send(P.MSG_STEP, b"".join(P.ACTION.pack(*a) for a in actions))
        obs, info = self._recv_obs()
        rewards = [sum(w * ev[k] for k, w in self.reward_weights.items()) for ev in info["events"]]
        return obs, rewards, info["terminated"], info["truncated"], info

    def debug_kill(self, agent):
        """Kill an agent like the in-game kill command. Needs debug=True; the world does not tick."""
        self._send(P.MSG_DEBUG_KILL, P.DEBUG_KILL.pack(agent))
        return self._recv_obs()

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
            subprocess.run(["freeport", "--release", str(self.port), "--proto", "udp"], capture_output=True)
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

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self._sock.recv(n - len(buf))
            except socket.timeout as e:
                raise EnvTimeout(f"no answer from the server within {self.timeout} s") from e
            except OSError as e:
                raise self._gone(f"receive failed: {e}") from e
            if not chunk:
                raise self._gone("server closed the connection")
            buf += chunk
        return bytes(buf)

    def _recv(self):
        magic, version, msg_type, seq, size = P.HEADER.unpack(self._recv_exact(P.HEADER.size))
        if magic != P.MAGIC or version != P.VERSION:
            raise EnvError(f"bad header from server: magic={magic:#x} version={version}")
        payload = self._recv_exact(size)
        if msg_type == P.MSG_ERROR:
            code, message = P.ERROR.unpack(payload)
            raise ProtocolError(code, message.split(b"\0", 1)[0].decode(errors="replace"))
        return msg_type, seq, payload

    def _recv_obs(self):
        msg_type, seq, payload = self._recv()
        if msg_type != P.MSG_OBS or len(payload) != P.OBS_SIZE:
            raise EnvError(f"expected OBS, got type {msg_type} with {len(payload)} bytes")
        if seq != self._seq:
            raise EnvError(f"answer for request {seq}, expected {self._seq}")
        episode_id, episode_tick, server_tick, terminated, truncated, flags, _ = P.OBS_HEAD.unpack_from(payload)
        obs, events = [], []
        for i in range(P.NUM_AGENTS):
            values = P.AGENT.unpack_from(payload, P.OBS_HEAD.size + i * P.AGENT.size)
            events.append(dict(zip(P.EVENT_NAMES, values[:6])))
            obs.append(values[6:])
        info = {
            "episode_id": episode_id, "episode_tick": episode_tick, "server_tick": server_tick,
            "terminated": bool(terminated), "truncated": bool(truncated), "flags": flags,
            "events": events, "payload": payload,
        }
        return obs, info

    def _gone(self, reason):
        code = None
        if self._proc is not None:
            try:
                code = self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        return EnvCrashed(f"{reason}; server exit code {code}\n{self.log_tail()}")


def trace_key(payload):
    """The parts of an OBS payload that should repeat across episodes: drops episode id and server tick."""
    return payload[4:8] + payload[12:]
