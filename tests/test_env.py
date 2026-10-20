"""Integration tests for the RL bridge and the observation layer. Need a built server:

    TWRL_SERVER_DIR=../teeworlds-0.7.5/build-release .venv/bin/python tests/test_env.py -v
"""

import math
import os
import random
import socket
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import (Action, EnvCrashed, EnvTimeout, ProtocolError, TeeworldsEnv, VecTeeworldsEnv,  # noqa: E402
                  episode_seed, random_action, random_actions)
from twrl import protocol as P  # noqa: E402
from twrl.env import trace_key  # noqa: E402
from twrl.testing import chase_actions  # noqa: E402

SERVER_DIR = os.environ.get("TWRL_SERVER_DIR")
V = P.VEC_INDEX
NEEDS_SERVER = unittest.skipUnless(SERVER_DIR, "set TWRL_SERVER_DIR to a build directory with teeworlds_srv")


def make_env(**kw):
    return TeeworldsEnv(SERVER_DIR, **kw)


def chase_trace(env, seed, steps):
    """Reset with `seed`, run the scripted chase and return the per step trace keys."""
    rng = random.Random(seed)
    obs, info = env.reset(seed=seed)
    trace = [trace_key(info["payload"])]
    for step in range(steps):
        obs, _, _, _, info = env.step(chase_actions(info["agents"], rng, step))
        trace.append(trace_key(info["payload"]))
    return trace


def random_run(env, seed, steps):
    """Yield (obs, info) for a reset and `steps` uniformly random steps."""
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    yield obs, info
    for _ in range(steps):
        obs, _, _, truncated, info = env.step(random_actions(rng, (2,)))
        yield obs, info
        if truncated:
            obs, info = env.reset()
            yield obs, info


def wait_exit(env, timeout=5):
    return env._proc.wait(timeout=timeout)


def round_to_int(f):
    return int(f + np.float32(0.5)) if f > 0 else int(f - np.float32(0.5))


def tile_at(collision, x, y):
    """Collision byte at a world position, the way CCollision::GetCollisionAt reads it."""
    h, w = collision.shape
    tx = min(max(int(round_to_int(x) / 32), 0), w - 1)  # C division truncates towards zero
    ty = min(max(int(round_to_int(y) / 32), 0), h - 1)
    return collision[ty, tx]


def expected_grid(collision, pos):
    h, w = collision.shape
    tx, ty = math.floor(pos[0] / 32), math.floor(pos[1] / 32)
    xs = np.clip(np.arange(tx - P.GRID_WIDTH // 2, tx + P.GRID_WIDTH // 2 + 1), 0, w - 1)
    ys = np.clip(np.arange(ty - P.GRID_HEIGHT // 2, ty + P.GRID_HEIGHT // 2 + 1), 0, h - 1)
    cells = collision[np.ix_(ys, xs)]
    return np.stack([(cells >> c) & 1 for c in range(3)]).astype(np.uint8)


def line_of_fire(collision, a, b):
    """CCollision::IntersectLine(a, b) finds no solid tile, in the same float32 steps."""
    f = np.float32
    a = np.asarray(a, dtype=f)
    b = np.asarray(b, dtype=f)
    d = a - b
    end = int(np.sqrt(d[0] * d[0] + d[1] * d[1]) + f(1))
    for i in range(end + 1):
        t = f(i) / f(end)
        pos = a + (b - a) * t
        if tile_at(collision, pos[0], pos[1]) & P.TILE_SOLID:
            return False
    return True


def in_view(dx, dy):
    return abs(dx) <= P.VIEW_HALF_WIDTH and abs(dy) <= P.VIEW_HALF_HEIGHT and math.hypot(dx, dy) <= P.VIEW_RADIUS


@NEEDS_SERVER
class TestBasics(unittest.TestCase):
    def setUp(self):
        self.env = make_env(debug=True)

    def tearDown(self):
        self.env.close()

    def test_hello(self):
        env = self.env
        self.assertEqual((env.action_ticks, env.tick_speed, env.episode_ticks, env.map), (2, 50, 3000, "dm1"))
        self.assertEqual(env.map_collision.shape, (50, 60))
        self.assertGreaterEqual(env.num_spawn_points, 2)

    def test_agent_names(self):
        self.env.reset(seed=1)
        with open(self.env.log_path) as f:
            log = f.read()
        self.assertIn("team_join player='63:AI-1'", log)
        self.assertIn("team_join player='62:AI-2'", log)
        self.assertNotIn("(invalid)", log)

    def test_reset_state(self):
        obs, info = self.env.reset(seed=1)
        self.assertEqual((info["episode_id"], info["episode_tick"], info["flags"]), (1, 0, 0))
        a = info["agents"]
        np.testing.assert_array_equal(a["alive"], [1, 1])
        np.testing.assert_array_equal(a["health"], [10, 10])
        np.testing.assert_array_equal(a["weapon"], [1, 1])  # gun
        np.testing.assert_array_equal(a["ammo"], [10, 10])
        np.testing.assert_array_equal(info["events"], np.zeros((2, len(P.EVENT_NAMES))))
        for i in range(2):
            self.assertEqual(obs["vec"][i, V["alive"]], 1.0)
            self.assertEqual(obs["vec"][i, V["weapon_gun"]], 1.0)
            self.assertEqual(obs["vec"][i, V["episode_progress"]], 0.0)

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

    def test_agents_move_independently(self):
        # only one agent gets input at a time; turn around halfway in case a wall is close
        for mover in (0, 1):
            self.env.reset(seed=1)
            vx = [[], []]
            for step in range(40):
                actions = [Action(), Action()]
                actions[mover] = Action(direction=1 if step < 20 else -1)
                _, _, _, _, info = self.env.step(actions)
                for i in range(2):
                    vx[i].append(info["agents"]["vel"][i][0])
            self.assertGreater(max(vx[mover]), 1.0)
            self.assertLess(min(vx[mover]), -1.0)
            self.assertLess(max(abs(v) for v in vx[1 - mover]), 0.5)

    def test_jump(self):
        obs, info = self.env.reset(seed=1)
        for _ in range(100):
            if obs["vec"][0, V["grounded"]]:
                break
            obs, _, _, _, info = self.env.step([Action(), Action()])
        self.assertEqual(obs["vec"][0, V["grounded"]], 1.0)
        obs, _, _, _, info = self.env.step([Action(jump=1), Action()])
        self.assertLess(info["agents"]["vel"][0][1], -1.0)
        self.assertLess(obs["vec"][0, V["vel_y"]], 0.0)
        self.assertEqual(obs["vec"][0, V["grounded"]], 0.0)
        self.assertEqual(obs["vec"][0, V["jumped"]], 1.0)

    def test_fire_uses_ammo(self):
        self.env.reset(seed=1)
        for step in range(20):
            obs, _, _, _, info = self.env.step([Action(fire=step % 2, target_x=0, target_y=1), Action()])
        self.assertLess(info["agents"]["ammo"][0], 10)
        self.assertEqual(info["agents"]["ammo"][1], 10)
        self.assertAlmostEqual(obs["vec"][0, V["ammo"]], info["agents"]["ammo"][0] / 10.0)

    def test_fire_held_over_decisions_keeps_shooting(self):
        # fire=1 at every decision is a new press each time: the (non-automatic) gun shoots whenever it is ready
        obs, info = self.env.reset(seed=1)
        shots, prev = [], info["agents"]["ammo"][0]
        S = P.EVENT_NAMES.index("shots")
        for _ in range(40):
            obs, _, _, _, info = self.env.step([Action(fire=1, target_x=0, target_y=1), Action()])
            ammo = info["agents"]["ammo"][0]
            shots.append(info["events"][0, S])
            self.assertLessEqual(info["events"][0, S], 1)
            self.assertLessEqual(ammo, prev - info["events"][0, S] + 1)  # at most one refill per decision
            prev = ammo
        self.assertGreaterEqual(sum(shots), 8)  # the full magazine of 10 and more, not a single press
        self.assertEqual(info["events"][1, S], 0)

    def test_shots_count_only_real_shots(self):
        obs, info = self.env.reset(seed=1)
        S = P.EVENT_NAMES.index("shots")
        total = 0
        for step in range(140):
            fire = 1 if step < 100 else 0
            ammo_before = info["agents"]["ammo"][0]
            obs, _, _, _, info = self.env.step([Action(fire=fire, target_x=256, target_y=0), Action()])
            n = info["events"][0, S]
            if n:
                self.assertGreater(ammo_before, 0)  # an empty gun does not shoot
            if not fire:
                self.assertEqual(n, 0)
            total += n
            if step == 99:
                # the magazine, and no refill: like a player clicking an empty gun, every press restarts
                # the 125 ms no-ammo delay, which holds the ammo regeneration back (CCharacter::HandleWeapons)
                self.assertEqual(total, 10)
                self.assertEqual(info["agents"]["ammo"][0], 0)
        self.assertGreater(info["agents"]["ammo"][0], 0)  # refills once the trigger is left alone

    def test_hook(self):
        self.env.reset(seed=1)
        flying = False
        for _ in range(5):
            obs, _, _, _, info = self.env.step([Action(hook=1, target_x=0, target_y=-256), Action()])
            flying |= info["agents"]["hook_state"][0] != 0
            self.assertEqual(obs["vec"][0, V["hook_idle"]:V["hook_retracting"] + 1].sum(), 1.0)
        self.assertTrue(flying)
        self.assertEqual(obs["vec"][1, V["hook_idle"]], 1.0)
        self.assertLess(obs["vec"][0, V["hook_dy"]], 0.0)

    def test_debug_kill_is_a_suicide(self):
        self.env.reset(seed=1)
        self.env.step([Action(), Action()])
        obs, info = self.env.debug_kill(0)
        self.assertEqual(info["agents"]["alive"][0], 0)
        self.assertEqual(info["events"][0][P.EVENT_NAMES.index("suicides")], 1)
        self.assertEqual(info["events"][0][P.EVENT_NAMES.index("score")], -1)
        self.assertEqual(info["events"][1][P.EVENT_NAMES.index("kills")], 0)
        # a dead agent sees only the clock, and is not visible to the other one
        dead = obs["vec"][0].copy()
        dead[V["episode_progress"]] = 0
        self.assertFalse(dead.any())
        self.assertFalse(obs["grid"][0].any())
        self.assertEqual(obs["vec"][1, V["opp_visible"]], 0.0)
        for _ in range(100):
            obs, _, _, _, info = self.env.step([Action(), Action()])
            if info["agents"]["alive"][0]:
                break
        self.assertEqual(info["agents"]["alive"][0], 1)


@NEEDS_SERVER
class TestObservation(unittest.TestCase):
    """The policy observation against the map and the raw state. dm6 has death and no-hook tiles."""

    STEPS = 1500

    @classmethod
    def setUpClass(cls):
        cls.runs = {}
        for name in ("dm6", "lms1"):
            env = make_env(map=name)
            try:
                cls.runs[name] = (env.map_collision.copy(), list(random_run(env, 11, cls.STEPS)))
            finally:
                env.close()
        with make_env(map="lms1", episode_ticks=6000) as env:
            rng = random.Random(3)
            obs, info = env.reset(seed=3)
            steps = [(obs, info)]
            for step in range(cls.STEPS):
                obs, _, _, _, info = env.step(chase_actions(info["agents"], rng, step))
                steps.append((obs, info))
            cls.runs["chase"] = (env.map_collision.copy(), steps)

    def all_steps(self):
        for collision, steps in self.runs.values():
            for obs, info in steps:
                yield collision, obs, info

    def test_fixed_shapes_finite_and_bounded(self):
        onehots = [("weapon_hammer", 6), ("opp_weapon_hammer", 6), ("hook_idle", 4), ("opp_hook_idle", 4)]
        unit = [V[n] for n in ("alive", "health", "armor", "ammo", "reload", "grounded", "jumped", "air_jumped",
                               "hooking_opp", "opp_visible", "opp_hooking_me", "opp_line_of_fire", "episode_progress",
                               "tile_offset_x", "tile_offset_y")]
        for _, obs, _ in self.all_steps():
            vec, grid = obs["vec"], obs["grid"]
            self.assertEqual((vec.shape, vec.dtype), ((2, P.VEC_DIM), np.float32))
            self.assertEqual((grid.shape, grid.dtype), ((2, 3, P.GRID_HEIGHT, P.GRID_WIDTH), np.uint8))
            self.assertTrue(np.isfinite(vec).all())
            self.assertLessEqual(np.abs(vec).max(), 5.0)
            self.assertTrue(((vec[:, unit] >= 0) & (vec[:, unit] <= 1)).all())
            for first, size in onehots:
                sums = vec[:, V[first]:V[first] + size].sum(axis=1)
                self.assertTrue(np.isin(sums, (0.0, 1.0)).all())
            self.assertLessEqual(grid.max(), 1)
            # no-hook tiles are always solid
            self.assertFalse((grid[:, 2] & (1 - grid[:, 0])).any())

    def test_grid_matches_map_geometry(self):
        seen = np.zeros(3, dtype=int)
        for collision, obs, info in self.all_steps():
            for i in range(2):
                if not info["agents"]["alive"][i]:
                    self.assertFalse(obs["grid"][i].any())
                    continue
                pos = info["agents"]["pos"][i]
                np.testing.assert_array_equal(obs["grid"][i], expected_grid(collision, pos))
                seen += obs["grid"][i].reshape(3, -1).any(axis=1)
                tile = np.array(pos) / 32
                np.testing.assert_allclose(obs["vec"][i, [V["tile_offset_x"], V["tile_offset_y"]]], tile - np.floor(tile), atol=1e-5)
        # dm6 puts death and no-hook tiles in view at some point
        self.assertTrue((seen > 0).all(), seen)

    def test_grid_follows_movement(self):
        collision, steps = self.runs["lms1"]
        checked = 0
        for (o1, i1), (o2, i2) in zip(steps, steps[1:]):
            if not (i1["agents"]["alive"][0] and i2["agents"]["alive"][0]):
                continue
            d = (np.floor(np.array(i2["agents"]["pos"][0]) / 32) - np.floor(np.array(i1["agents"]["pos"][0]) / 32)).astype(int)
            if d.any() and abs(d).max() < 5:
                dx, dy = d
                g1, g2 = o1["grid"][0], o2["grid"][0]
                h, w = P.GRID_HEIGHT, P.GRID_WIDTH
                np.testing.assert_array_equal(g2[:, max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)],
                                              g1[:, max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)])
                checked += 1
        self.assertGreater(checked, 20)

    def test_opponent_features(self):
        visible = hidden = clear = blocked = 0
        for collision, obs, info in self.all_steps():
            a = info["agents"]
            for i in range(2):
                v = obs["vec"][i]
                opp = v[V["opp_visible"]:V["opp_line_of_fire"] + 1]
                if not (a["alive"][i] and a["alive"][1 - i]):
                    self.assertFalse(opp.any())
                    continue
                dx, dy = a["pos"][1 - i] - a["pos"][i]
                if not in_view(dx, dy):
                    self.assertFalse(opp.any(), (dx, dy))
                    hidden += 1
                    continue
                visible += 1
                self.assertEqual(v[V["opp_visible"]], 1.0)
                self.assertAlmostEqual(v[V["opp_dx"]] * P.VIEW_HALF_WIDTH, dx, places=2)
                self.assertAlmostEqual(v[V["opp_dy"]] * P.VIEW_HALF_HEIGHT, dy, places=2)
                rel = (a["vel"][1 - i] - a["vel"][i]) / P.VEL_SCALE
                np.testing.assert_allclose(v[[V["opp_rel_vel_x"], V["opp_rel_vel_y"]]], np.clip(rel, -5, 5), atol=1e-5)
                self.assertEqual(v[V["opp_weapon_hammer"] + a["weapon"][1 - i]], 1.0)
                expected = line_of_fire(collision, a["pos"][i], a["pos"][1 - i])
                self.assertEqual(v[V["opp_line_of_fire"]], 1.0 if expected else 0.0)
                clear += expected
                blocked += not expected
            if a["alive"].all() and obs["vec"][0, V["opp_visible"]]:
                # symmetric
                self.assertAlmostEqual(obs["vec"][0, V["opp_dx"]], -obs["vec"][1, V["opp_dx"]], places=5)
        self.assertGreater(visible, 50)
        self.assertGreater(hidden, 50)
        self.assertGreater(clear, 10)
        self.assertGreater(blocked, 10)

    def test_projectiles(self):
        present = hostile = own = 0
        for _, obs, info in self.all_steps():
            for i in range(2):
                v = obs["vec"][i]
                slots = v[V["proj0_present"]:V["proj0_present"] + 6 * P.NUM_PROJECTILES].reshape(P.NUM_PROJECTILES, 6)
                flags = slots[:, 0]
                # filled from the front, nearest first
                self.assertTrue((np.diff(flags) <= 0).all())
                n = int(flags.sum())
                self.assertFalse(slots[n:].any())
                dist = np.hypot(slots[:n, 1] * P.VIEW_HALF_WIDTH, slots[:n, 2] * P.VIEW_HALF_HEIGHT)
                self.assertTrue((np.diff(dist) >= -1e-3).all())
                self.assertTrue((np.abs(slots[:n, 1:3]) <= 1.0 + 1e-6).all())
                present += n
                hostile += int(slots[:n, 5].sum())
                own += n - int(slots[:n, 5].sum())
        self.assertGreater(present, 100)
        self.assertGreater(hostile, 10)
        self.assertGreater(own, 10)

    def test_fresh_shot_is_an_own_projectile(self):
        with make_env(map="lms1") as env:
            env.reset(seed=4)
            for _ in range(3):
                env.step([Action(), Action()])
            obs, _, _, _, info = env.step([Action(fire=1, target_x=256, target_y=0), Action()])
            for _ in range(3):
                if obs["vec"][0, V["proj0_present"]]:
                    break
                obs, _, _, _, info = env.step([Action(fire=1, target_x=256, target_y=0), Action()])
            self.assertEqual(obs["vec"][0, V["proj0_present"]], 1.0)
            self.assertEqual(obs["vec"][0, V["proj0_hostile"]], 0.0)
            self.assertGreater(obs["vec"][0, V["proj0_vel_x"]], 0.0)


@NEEDS_SERVER
class TestSpawns(unittest.TestCase):
    def test_spawns_valid_distinct_and_diverse(self):
        for name in ("lms1", "dm1", "dm6"):
            with make_env(map=name) as env:
                starts = set()
                for seed in range(1, 41):
                    obs, info = env.reset(seed=seed)
                    pos = info["agents"]["pos"]
                    self.assertTrue(info["agents"]["alive"].all())
                    for p in pos:
                        self.assertFalse(tile_at(env.map_collision, *p) & (P.TILE_SOLID | P.TILE_DEATH), (name, seed, p))
                    self.assertGreater(np.hypot(*(pos[0] - pos[1])), 28)  # never overlapping (PHYS_SIZE)
                    starts.add(tuple(np.round(pos.ravel())))
                self.assertGreater(len(starts), 15, name)

    def test_seed_reproducible(self):
        with make_env(map="lms1") as env:
            a = env.reset(seed=5)[1]["agents"]["pos"].copy()
            b = env.reset()[1]["agents"]["pos"].copy()
            c = env.reset(seed=5)[1]["agents"]["pos"].copy()
            d = env.reset()[1]["agents"]["pos"].copy()
        with make_env(map="lms1") as env:
            e = env.reset(seed=5)[1]["agents"]["pos"].copy()
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(a, e)
        np.testing.assert_array_equal(b, d)
        self.assertFalse(np.array_equal(a, b))


@NEEDS_SERVER
class TestCombat(unittest.TestCase):
    def test_damage_kills_and_score_agree(self):
        # lms1 is small enough for the scripted chase to produce fights; the game type stays dm
        with make_env(map="lms1", episode_ticks=6000) as env:
            rng = random.Random(10)
            obs, info = env.reset(seed=10)
            totals = np.zeros((2, len(P.EVENT_NAMES)), dtype=int)
            score = [0, 0]
            k = {n: j for j, n in enumerate(P.EVENT_NAMES)}
            for step in range(3000):
                obs, rewards, _, truncated, info = env.step(chase_actions(info["agents"], rng, step))
                ev = info["events"]
                totals += ev
                for i in range(2):
                    self.assertEqual(ev[i][k["damage_dealt"]], ev[1 - i][k["damage_taken"]])
                    self.assertEqual(ev[i][k["kills"]], ev[1 - i][k["deaths"]])
                    self.assertEqual(ev[i][k["score"]] - score[i], ev[i][k["kills"]] - ev[i][k["suicides"]])
                    score[i] = ev[i][k["score"]]
                    expected = (ev[i][k["kills"]] - ev[i][k["deaths"]] - 2 * ev[i][k["suicides"]]
                                + 0.02 * (ev[i][k["damage_dealt"]] - ev[i][k["damage_taken"]]))
                    self.assertAlmostEqual(float(rewards[i]), expected, places=5)
                if truncated:
                    break
            self.assertGreater(totals[:, k["kills"]].sum(), 2, totals)
            self.assertGreater(totals[:, k["damage_dealt"]].sum(), 0, totals)

    def test_reset_after_fights(self):
        with make_env(map="lms1", episode_ticks=6000) as env:
            rng = random.Random(10)
            obs, info = env.reset(seed=10)
            kills = 0
            for step in range(3000):
                obs, _, _, _, info = env.step(chase_actions(info["agents"], rng, step))
                kills += info["events"][:, 0].sum()
                if kills >= 2:
                    break
            self.assertGreaterEqual(kills, 2)
            obs, info = env.reset(seed=1)
            self.assertEqual((info["episode_id"], info["episode_tick"]), (2, 0))
            np.testing.assert_array_equal(info["events"][:, P.EVENT_NAMES.index("score")], [0, 0])
            np.testing.assert_array_equal(info["agents"]["alive"], [1, 1])
            np.testing.assert_array_equal(info["agents"]["health"], [10, 10])
            np.testing.assert_array_equal(info["agents"]["ammo"], [10, 10])


@NEEDS_SERVER
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
        self.assertNotEqual(a[0], b[0])  # different spawns already
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


@NEEDS_SERVER
class TestVecEnv(unittest.TestCase):
    def run_vec(self, seed, steps=60, n=3, episode_ticks=40):
        rng = np.random.default_rng(99)
        out = []
        with VecTeeworldsEnv(n, SERVER_DIR, seed=seed, map="lms1", episode_ticks=episode_ticks) as venv:
            obs = venv.reset()
            out.append(obs)
            for _ in range(steps):
                obs, rewards, terminated, truncated, infos = venv.step(random_actions(rng, (n, 2)))
                out.append((obs, rewards, terminated, truncated, infos))
        return out

    def test_shapes_and_auto_reset(self):
        out = self.run_vec(1)
        obs0 = out[0]
        self.assertEqual(obs0["vec"].shape, (3, 2, P.VEC_DIM))
        self.assertEqual(obs0["grid"].shape, (3, 2, 3, P.GRID_HEIGHT, P.GRID_WIDTH))
        resets = 0
        for step, (obs, rewards, terminated, truncated, infos) in enumerate(out[1:], 1):
            self.assertEqual(rewards.shape, (3, 2))
            self.assertEqual(infos["events"].shape, (3, 2, len(P.EVENT_NAMES)))
            if step % 20 == 0:  # 40 ticks = 20 steps
                self.assertTrue(truncated.all())
                self.assertEqual(infos["final_vec"].shape, (3, 2, P.VEC_DIM))
                self.assertTrue((infos["final_vec"][:, :, V["episode_progress"]] == 1.0).all())
                # the returned obs starts the next episode
                self.assertTrue((obs["vec"][:, :, V["episode_progress"]] == 0.0).all())
                resets += 1
            else:
                self.assertFalse(truncated.any())
                self.assertNotIn("final_vec", infos)
        self.assertEqual(resets, 3)

    def test_deterministic_per_seed(self):
        a, b, c = self.run_vec(1), self.run_vec(1), self.run_vec(2)
        np.testing.assert_array_equal(a[0]["vec"], b[0]["vec"])
        for x, y in zip(a[1:], b[1:]):
            np.testing.assert_array_equal(x[0]["vec"], y[0]["vec"])
            np.testing.assert_array_equal(x[0]["grid"], y[0]["grid"])
        self.assertFalse(np.array_equal(a[0]["vec"], c[0]["vec"]))
        # the envs inside one vector env start differently too
        self.assertFalse(np.array_equal(a[0]["vec"][0], a[0]["vec"][1]))

    def test_matches_single_env(self):
        rng = np.random.default_rng(5)
        actions = random_actions(rng, (30, 2, 2))
        with VecTeeworldsEnv(2, SERVER_DIR, seed=7, map="lms1") as venv:
            vec_obs = [venv.reset()]
            for a in actions:
                vec_obs.append(venv.step(a)[0])
        with make_env(map="lms1") as env:
            single = [env.reset(seed=episode_seed(7, 1, 1))[0]]
            for a in actions:
                single.append(env.step(a[1])[0])
        for v, s in zip(vec_obs, single):
            np.testing.assert_array_equal(v["vec"][1], s["vec"])
            np.testing.assert_array_equal(v["grid"][1], s["grid"])


@NEEDS_SERVER
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

    def test_old_protocol_version(self):
        with make_env() as env:
            self.raw(env, version=1, msg_type=P.MSG_RESET, payload=bytes(8))
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
