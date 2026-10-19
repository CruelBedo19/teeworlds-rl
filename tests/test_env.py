"""Integration tests for the RL bridge. Need a built server:

    TWRL_SERVER_DIR=../teeworlds-0.7.5/build-release python3 tests/test_env.py -v
"""

import os
import random
import socket
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import Action, EnvCrashed, EnvTimeout, ProtocolError, TeeworldsEnv, random_action  # noqa: E402
from twrl import protocol as P  # noqa: E402
from twrl.env import trace_key  # noqa: E402
from twrl.protocol import OBS_INDEX as I  # noqa: E402
from twrl.testing import chase_action  # noqa: E402

SERVER_DIR = os.environ.get("TWRL_SERVER_DIR")
EVENTS = ("kills", "deaths", "suicides", "damage_dealt", "damage_taken")


def make_env(**kw):
    return TeeworldsEnv(SERVER_DIR, **kw)


def chase_trace(env, seed, steps):
    """Run the scripted chase for `steps` steps after a reset and return the per step trace keys."""
    rng = random.Random(seed)
    obs, info = env.reset(seed=seed)
    trace = [trace_key(info["payload"])]
    for step in range(steps):
        obs, _, _, _, info = env.step([chase_action(obs[i], rng, step) for i in range(2)])
        trace.append(trace_key(info["payload"]))
    return trace


def wait_exit(env, timeout=5):
    return env._proc.wait(timeout=timeout)


@unittest.skipUnless(SERVER_DIR, "set TWRL_SERVER_DIR to a build directory with teeworlds_srv")
class TestBasics(unittest.TestCase):
    def setUp(self):
        self.env = make_env(debug=True)

    def tearDown(self):
        self.env.close()

    def test_hello(self):
        self.assertEqual((self.env.action_ticks, self.env.tick_speed, self.env.episode_ticks, self.env.map), (2, 50, 3000, "dm1"))

    def test_reset_state(self):
        obs, info = self.env.reset(seed=1)
        self.assertEqual((info["episode_id"], info["episode_tick"], info["flags"]), (1, 0, 0))
        for i in range(2):
            self.assertEqual(obs[i][I["alive"]], 1.0)
            self.assertEqual(obs[i][I["opp_alive"]], 1.0)
            self.assertEqual(obs[i][I["health"]], 10.0)
            self.assertEqual(obs[i][I["weapon"]], 1.0)  # gun
            self.assertEqual(obs[i][I["ammo"]], 10.0)
            self.assertEqual(info["events"][i], dict(kills=0, deaths=0, suicides=0, damage_dealt=0, damage_taken=0, score=0))
        # the opponent's relative position mirrors
        self.assertEqual(obs[0][I["opp_dx"]], -obs[1][I["opp_dx"]])

    def test_episode_ticks_and_truncation(self):
        env = make_env(episode_ticks=20)
        try:
            env.reset()
            ticks = []
            for _ in range(10):
                _, _, terminated, truncated, info = env.step([Action(), Action()])
                ticks.append(info["episode_tick"])
            self.assertEqual(ticks, list(range(2, 21, 2)))
            self.assertTrue(truncated)
            self.assertFalse(terminated)
            with self.assertRaises(ProtocolError) as cm:
                env.step([Action(), Action()])
            self.assertEqual(cm.exception.code, P.ERROR_STATE)
        finally:
            env.close()

    def test_movement(self):
        self.env.reset(seed=1)
        vx = [[], []]
        for _ in range(25):
            obs, *_ = self.env.step([Action(direction=1), Action(direction=-1)])
            vx[0].append(obs[0][I["vel_x"]])
            vx[1].append(obs[1][I["vel_x"]])
        self.assertGreater(max(vx[0]), 1.0)
        self.assertLess(min(vx[1]), -1.0)

    def test_jump(self):
        obs, _ = self.env.reset(seed=1)
        for _ in range(100):
            if obs[0][I["grounded"]]:
                break
            obs, *_ = self.env.step([Action(), Action()])
        self.assertEqual(obs[0][I["grounded"]], 1.0)
        obs, *_ = self.env.step([Action(jump=1), Action()])
        self.assertLess(obs[0][I["vel_y"]], -1.0)
        self.assertEqual(obs[0][I["grounded"]], 0.0)

    def test_fire_uses_ammo(self):
        obs, _ = self.env.reset(seed=1)
        for step in range(20):
            obs, *_ = self.env.step([Action(fire=step % 2, target_x=0, target_y=1), Action()])
        self.assertLess(obs[0][I["ammo"]], 10.0)
        self.assertEqual(obs[1][I["ammo"]], 10.0)

    def test_hook(self):
        self.env.reset(seed=1)
        states = []
        for _ in range(5):
            obs, *_ = self.env.step([Action(hook=1, target_x=0, target_y=-256), Action()])
            states.append(obs[0][I["hook_state"]])
        self.assertTrue(any(s != 0 for s in states), states)
        self.assertEqual(obs[1][I["hook_state"]], 0.0)

    def test_debug_kill_is_a_suicide(self):
        self.env.reset(seed=1)
        self.env.step([Action(), Action()])
        obs, info = self.env.debug_kill(0)
        self.assertEqual(obs[0][I["alive"]], 0.0)
        self.assertEqual(obs[1][I["opp_alive"]], 0.0)
        self.assertEqual(info["events"][0]["suicides"], 1)
        self.assertEqual(info["events"][0]["score"], -1)
        self.assertEqual(info["events"][1]["kills"], 0)
        # respawns after the usual delay (3 s for WEAPON_SELF)
        for _ in range(100):
            obs, *_ = self.env.step([Action(), Action()])
            if obs[0][I["alive"]]:
                break
        self.assertEqual(obs[0][I["alive"]], 1.0)


@unittest.skipUnless(SERVER_DIR, "set TWRL_SERVER_DIR to a build directory with teeworlds_srv")
class TestCombat(unittest.TestCase):
    def test_damage_kills_and_score_agree(self):
        # lms1 is small enough for the scripted chase to produce fights; the game type stays dm
        with make_env(map="lms1", episode_ticks=6000) as env:
            rng = random.Random(3)
            obs, _ = env.reset(seed=1)
            totals = [dict.fromkeys(EVENTS, 0) for _ in range(2)]
            score = [0, 0]
            for step in range(3000):
                obs, rewards, _, truncated, info = env.step([chase_action(obs[i], rng, step) for i in range(2)])
                ev = info["events"]
                for i in range(2):
                    for k in EVENTS:
                        totals[i][k] += ev[i][k]
                    self.assertEqual(ev[i]["damage_dealt"], ev[1 - i]["damage_taken"])
                    self.assertEqual(ev[i]["kills"], ev[1 - i]["deaths"])
                    self.assertEqual(ev[i]["score"] - score[i], ev[i]["kills"] - ev[i]["suicides"])
                    score[i] = ev[i]["score"]
                    expected = ev[i]["kills"] - ev[i]["deaths"] - 2 * ev[i]["suicides"] + 0.02 * (ev[i]["damage_dealt"] - ev[i]["damage_taken"])
                    self.assertAlmostEqual(rewards[i], expected)
                if truncated:
                    break
            self.assertGreater(totals[0]["kills"] + totals[1]["kills"], 0, totals)
            self.assertGreater(totals[0]["damage_dealt"] + totals[1]["damage_dealt"], 0, totals)

    def test_reset_after_fights(self):
        with make_env(map="lms1", episode_ticks=6000) as env:
            rng = random.Random(3)
            obs, _ = env.reset(seed=1)
            kills = 0
            for step in range(3000):
                obs, _, _, _, info = env.step([chase_action(obs[i], rng, step) for i in range(2)])
                kills += info["events"][0]["kills"] + info["events"][1]["kills"]
                if kills >= 2:
                    break
            self.assertGreaterEqual(kills, 2)
            obs, info = env.reset(seed=1)
            self.assertEqual(info["episode_id"], 2)
            self.assertEqual(info["episode_tick"], 0)
            for i in range(2):
                self.assertEqual(info["events"][i]["score"], 0)
                self.assertEqual(obs[i][I["alive"]], 1.0)
                self.assertEqual(obs[i][I["health"]], 10.0)
                self.assertEqual(obs[i][I["ammo"]], 10.0)


@unittest.skipUnless(SERVER_DIR, "set TWRL_SERVER_DIR to a build directory with teeworlds_srv")
class TestDeterminism(unittest.TestCase):
    STEPS = 1500

    def test_episodes_repeat_in_one_process_and_across_processes(self):
        # a fresh episode must not see anything left over from the previous one
        with make_env(map="lms1") as env:
            first = chase_trace(env, 5, self.STEPS)
            second = chase_trace(env, 5, self.STEPS)
        with make_env(map="lms1") as env:
            fresh = chase_trace(env, 5, self.STEPS)
        self.assertEqual(first, second)
        self.assertEqual(first, fresh)

    def test_seed_is_used(self):
        with make_env(map="lms1") as env:
            a = chase_trace(env, 5, self.STEPS)
            b = chase_trace(env, 6, self.STEPS)
        self.assertNotEqual(a, b)

    def test_mode_1_and_2_match(self):
        traces = []
        for mode in (1, 2):
            with make_env(map="lms1", mode=mode) as env:
                traces.append(chase_trace(env, 5, self.STEPS))
        self.assertEqual(traces[0], traces[1])

    def test_realtime_mode_matches(self):
        # mode 0 runs at 50 ticks/s, so keep this one short: 300 steps are 12 s
        traces = []
        for mode in (0, 2):
            with make_env(map="lms1", mode=mode) as env:
                traces.append(chase_trace(env, 5, 300))
        self.assertEqual(traces[0], traces[1])


@unittest.skipUnless(SERVER_DIR, "set TWRL_SERVER_DIR to a build directory with teeworlds_srv")
class TestFailures(unittest.TestCase):
    def raw(self, env, magic=P.MAGIC, version=P.VERSION, msg_type=P.MSG_STEP, seq=None, payload=b""):
        env._seq += 1
        env._sock.sendall(P.HEADER.pack(magic, version, msg_type, env._seq if seq is None else seq, len(payload)) + payload)

    def expect_error(self, env, code):
        with self.assertRaises(ProtocolError) as cm:
            env._recv()
        self.assertEqual(cm.exception.code, code)
        self.assertIsNotNone(wait_exit(env))

    def test_bad_magic(self):
        with make_env() as env:
            self.raw(env, magic=0x1234, payload=bytes(48))
            self.expect_error(env, P.ERROR_PROTOCOL)

    def test_wrong_sequence(self):
        with make_env() as env:
            env.reset()
            self.raw(env, seq=99, payload=bytes(48))
            self.expect_error(env, P.ERROR_PROTOCOL)

    def test_bad_size(self):
        with make_env() as env:
            env.reset()
            self.raw(env, payload=bytes(10))
            self.expect_error(env, P.ERROR_PROTOCOL)

    def test_unknown_type(self):
        with make_env() as env:
            self.raw(env, msg_type=99)
            self.expect_error(env, P.ERROR_PROTOCOL)

    def test_step_before_reset(self):
        with make_env() as env:
            with self.assertRaises(ProtocolError) as cm:
                env.step([Action(), Action()])
            self.assertEqual(cm.exception.code, P.ERROR_STATE)
            self.assertIsNotNone(wait_exit(env))

    def test_debug_kill_needs_debug(self):
        with make_env() as env:
            env.reset()
            with self.assertRaises(ProtocolError) as cm:
                env.debug_kill(0)
            self.assertEqual(cm.exception.code, P.ERROR_DEBUG_DISABLED)

    def test_server_times_out_on_a_silent_trainer(self):
        with make_env(timeout=1.0) as env:
            env.reset()
            t0 = time.monotonic()
            self.assertIsNotNone(wait_exit(env, timeout=5))
            self.assertLess(time.monotonic() - t0, 2.5)
            with self.assertRaises(EnvCrashed):
                env.step([Action(), Action()])

    def test_trainer_disconnect_stops_server(self):
        env = make_env()
        try:
            env.reset()
            env._sock.shutdown(socket.SHUT_RDWR)
            env._sock.close()
            env._sock = None
            self.assertIsNotNone(wait_exit(env, timeout=3))
        finally:
            env.close()

    def test_killed_server_is_reported(self):
        with make_env() as env:
            env.reset()
            env._proc.kill()
            env._proc.wait()
            with self.assertRaises(EnvCrashed) as cm:
                env.step([Action(), Action()])
            self.assertIn("exit code -9", str(cm.exception))

    def test_client_timeout(self):
        with make_env(timeout=1.0, mode=0) as env:
            env.reset()
            env._proc.send_signal(19)  # SIGSTOP: the server hangs without closing the socket
            try:
                with self.assertRaises(EnvTimeout):
                    env.step([Action(), Action()])
            finally:
                env._proc.send_signal(18)  # SIGCONT

    def test_clean_close(self):
        env = make_env()
        env.reset()
        env.step([random_action(random.Random(1)), Action()])
        env.close()
        self.assertEqual(env._proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
