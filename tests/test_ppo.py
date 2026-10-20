"""PPO plumbing tests. The unit tests need only torch; the integration tests need a built server:

    TWRL_SERVER_DIR=../teeworlds-0.7.5/build-release .venv/bin/python tests/test_ppo.py -v
"""

import glob
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import VecTeeworldsEnv  # noqa: E402
from twrl import protocol as P  # noqa: E402
from twrl.ppo import (AIM_BINS, DISCRETE, HEADS, Policy, PPOConfig, Trainer, gaussian_env_actions, load_policy,  # noqa: E402
                      save_checkpoint, to_env_actions)

SERVER_DIR = os.environ.get("TWRL_SERVER_DIR")


def fake_batch(n=256, seed=0, **policy_kw):
    g = torch.Generator().manual_seed(seed)
    policy = Policy(**policy_kw)
    vec = torch.randn(n, P.VEC_DIM, generator=g)
    codes = torch.randint(0, 8, (n, P.GRID_HEIGHT, P.GRID_WIDTH), generator=g, dtype=torch.uint8) & 7
    with torch.no_grad():
        actions, logp, values, _, aim_logp = policy.act(vec, codes, g)
    adv = torch.randn(n, generator=g)
    aim_mask = (torch.rand(n, generator=g) < 0.3).float()  # decisions where a shot left the gun
    return policy, {"vec": vec, "codes": codes, "actions": actions, "logp": logp, "values": values, "adv": adv,
                    "aim_logp": aim_logp, "aim_mask": aim_mask,
                    "returns": adv + values}


class NoEnv:
    num_envs = 1

    def reset(self):
        return {"vec": np.zeros((1, 2, P.VEC_DIM), np.float32), "grid": np.zeros((1, 2, P.GRID_HEIGHT, P.GRID_WIDTH), np.uint8)}


class TestPolicy(unittest.TestCase):
    def test_shapes_and_initial_entropy(self):
        policy, batch = fake_batch()
        logits, value = policy(batch["vec"], batch["codes"])
        self.assertEqual(logits.shape, (256, sum(HEADS)))
        self.assertEqual(value.shape, (256,))
        _, ent, _, aim_ent, _ = policy.evaluate(batch["vec"], batch["codes"], batch["actions"])
        self.assertTrue(torch.equal(aim_ent, torch.zeros(256)))  # the categorical aim is in ent
        # the actor head starts almost uniform: entropy close to the maximum
        self.assertAlmostEqual(ent.mean().item(), sum(np.log(HEADS)), delta=0.01)

    def test_env_action_mapping(self):
        heads = np.array([[0, 0, 0, 0, 0], [2, 1, 1, 1, AIM_BINS // 4], [1, 0, 1, 0, AIM_BINS // 2]])
        out = to_env_actions(heads)
        np.testing.assert_array_equal(out[:, :4], [[-1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 0]])
        np.testing.assert_array_equal(out[:, 4:], [[256, 0], [0, 256], [-256, 0]])

    def test_update_is_finite_and_changes_weights(self):
        policy, batch = fake_batch()
        trainer = Trainer(NoEnv(), policy, PPOConfig(epochs=2, minibatch=64))
        before = [p.detach().clone() for p in policy.parameters()]
        stats = trainer.train(batch)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()), stats)
        self.assertGreater(stats["grad_norm"], 0.0)
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, policy.parameters())))
        self.assertEqual(trainer.update, 1)

    def test_non_finite_loss_is_caught(self):
        policy, batch = fake_batch()
        batch["returns"] = batch["returns"].clone()
        batch["returns"][0] = float("nan")
        trainer = Trainer(NoEnv(), policy, PPOConfig(epochs=1, minibatch=256))
        with self.assertRaises(FloatingPointError):
            trainer.train(batch)

    def test_checkpoint_roundtrip_and_resume(self):
        policy, batch = fake_batch()
        trainer = Trainer(NoEnv(), policy, PPOConfig(epochs=1, minibatch=128))
        trainer.train(batch)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            save_checkpoint(path, trainer)
            loaded, state = load_policy(path)
            with torch.no_grad():
                for x, y in zip(policy.eval()(batch["vec"], batch["codes"]), loaded(batch["vec"], batch["codes"])):
                    self.assertTrue(torch.equal(x, y))
            # resuming continues the same optimisation: same update, same next step
            twin = Trainer(NoEnv(), Policy(), PPOConfig(epochs=1, minibatch=128))
            twin.load_state_dict(state)
            self.assertEqual(twin.update, 1)
            policy.train()
            twin.policy.train()
            trainer.train(batch)
            twin.train(batch)
            for a, b in zip(trainer.policy.parameters(), twin.policy.parameters()):
                self.assertTrue(torch.allclose(a, b, atol=1e-6))
            self.assertEqual(os.listdir(d), ["ckpt.pt"])  # nothing half-written left behind
            for key in ("format", "policy", "optimizer", "update", "agent_steps", "config", "generator", "torch_rng", "env_episodes"):
                self.assertIn(key, state)

    def test_compiled_update_matches_eager(self):
        for aim in ("categorical", "gaussian"):
            torch.manual_seed(0)
            policy, batch = fake_batch(aim=aim)
            twin = Policy(aim=aim)
            twin.load_state_dict(policy.state_dict())
            eager = Trainer(NoEnv(), policy, PPOConfig(epochs=1, minibatch=256), seed=3)
            compiled = Trainer(NoEnv(), twin, PPOConfig(epochs=1, minibatch=256), seed=3, compile=True)
            s1, s2 = eager.train(batch), compiled.train(batch)
            for k in s1:
                self.assertAlmostEqual(s1[k], s2[k], places=4, msg=(aim, k))
            for a, b in zip(policy.parameters(), twin.parameters()):
                self.assertTrue(torch.allclose(a, b, atol=1e-5), aim)

    def test_first_stride(self):
        for stride, cells in ((1, 7 * 9), (2, 4 * 5)):
            policy = Policy(first_stride=stride)
            self.assertEqual(policy.grid_fc[0].in_features, 32 * cells)
            _, batch = fake_batch(n=8)
            logits, value = policy(batch["vec"], batch["codes"])
            self.assertEqual(logits.shape, (8, sum(HEADS)))

    def test_old_checkpoints_load_as_the_first_network(self):
        policy = Policy(first_stride=1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.pt")
            torch.save({"format": 2, "policy": policy.state_dict(), "update": 0, "agent_steps": 0}, path)
            loaded, _ = load_policy(path)
        self.assertEqual(loaded.spec, {"hidden": 256, "aim": "categorical", "first_stride": 1})
        for a, b in zip(policy.state_dict().values(), loaded.state_dict().values()):
            self.assertTrue(torch.equal(a, b))


class TestGaussianAim(unittest.TestCase):
    def test_act_and_evaluate_agree(self):
        policy, batch = fake_batch(aim="gaussian")
        self.assertEqual(batch["actions"].shape, (256, len(DISCRETE) + 2))
        logp, ent, value, aim_ent, aim_logp = policy.evaluate(batch["vec"], batch["codes"], batch["actions"])
        self.assertTrue(torch.allclose(logp, batch["logp"], atol=1e-5))
        self.assertTrue(torch.allclose(aim_logp, batch["aim_logp"], atol=1e-5))
        self.assertTrue(torch.allclose(value, batch["values"]))
        # discrete heads near uniform; aim std exp(-0.5) per axis at the start, reported on its own
        self.assertAlmostEqual(ent.mean().item(), sum(np.log(DISCRETE)), delta=0.01)
        gauss = 2 * (0.5 * math.log(2 * math.pi * math.e) - 0.5)
        self.assertAlmostEqual(aim_ent.mean().item(), gauss, delta=1e-4)

    def test_log_std_is_capped_at_zero(self):
        policy, batch = fake_batch(aim="gaussian")
        with torch.no_grad():
            policy.log_std.fill_(2.0)
        self.assertTrue(torch.equal(policy._std(), torch.ones(2)))
        self.assertEqual(policy.spec["log_std_max"], 0.0)

    def test_no_entropy_bonus_on_the_aim(self):
        # with zero advantages and exact value targets only the entropy bonus could move log_std
        policy, batch = fake_batch(aim="gaussian")
        batch["adv"] = torch.zeros_like(batch["adv"])
        batch["returns"] = batch["values"].clone()
        before = policy.log_std.detach().clone()
        stats = Trainer(NoEnv(), policy, PPOConfig(epochs=2, minibatch=64)).train(batch)
        self.assertTrue(torch.equal(policy.log_std.detach(), before))
        self.assertEqual(stats["log_std_x"], before[0].item())

    def test_gaussian_checkpoints_before_the_cap_keep_theirs(self):
        policy = Policy(aim="gaussian", log_std_max=1.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ab.pt")
            torch.save({"format": 3, "model": {"hidden": 256, "aim": "gaussian", "first_stride": 2},
                        "policy": policy.state_dict(), "update": 0, "agent_steps": 0}, path)
            loaded, _ = load_policy(path)
        self.assertEqual(loaded.spec["log_std_max"], 1.0)

    def test_initial_aim_has_no_preferred_direction(self):
        policy, batch = fake_batch(n=4096, aim="gaussian")
        aim = batch["actions"][:, 4:].numpy()
        angle = np.arctan2(aim[:, 1], aim[:, 0])
        counts, _ = np.histogram(angle, bins=8, range=(-np.pi, np.pi))
        self.assertLess(counts.max() / counts.min(), 1.3)

    def test_env_action_mapping(self):
        actions = np.array([[0, 0, 0, 0, 3.0, 4.0], [2, 1, 1, 1, -0.001, 0.0], [1, 0, 1, 0, 0.0, 0.0]])
        out = gaussian_env_actions(actions)
        np.testing.assert_array_equal(out[:, :4], [[-1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 0]])
        np.testing.assert_array_equal(out[:, 4:], [[154, 205], [-256, 0], [256, 0]])  # a zero vector aims right

    def test_update_moves_aim_towards_rewarded_direction(self):
        torch.manual_seed(0)
        policy, batch = fake_batch(n=2048, aim="gaussian")
        aim = batch["actions"][:, 4:]
        batch["adv"] = aim[:, 0] / aim.norm(dim=1)  # reward aiming to the right
        trainer = Trainer(NoEnv(), policy, PPOConfig(epochs=4, minibatch=256, lr=1e-3))
        before = policy(batch["vec"], batch["codes"])[0][:, -2].mean().item()
        stats = trainer.train(batch)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()), stats)
        after = policy(batch["vec"], batch["codes"])[0][:, -2].mean().item()
        self.assertGreater(after, before + 0.01)

    def grads(self, policy, batch, mask_value, cfg=None):
        """Gradients of one loss evaluation with every decision's aim mask set to mask_value."""
        b = dict(batch, aim_mask=torch.full_like(batch["aim_mask"], mask_value))
        trainer = Trainer(NoEnv(), policy, cfg or PPOConfig())
        policy.zero_grad()
        trainer.losses(b, torch.arange(len(b["adv"])), b["adv"])["loss"].backward()
        return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for n, p in policy.named_parameters()}

    def test_no_aim_gradient_without_a_shot(self):
        policy, batch = fake_batch(aim="gaussian")
        g = self.grads(policy, batch, 0.0)
        self.assertEqual(g["pi.weight"][-2:].abs().max().item(), 0.0)
        self.assertEqual(g["pi.bias"][-2:].abs().max().item(), 0.0)
        self.assertEqual(g["log_std"].abs().max().item(), 0.0)

    def test_aim_gradient_with_a_shot(self):
        policy, batch = fake_batch(aim="gaussian")
        g = self.grads(policy, batch, 1.0)
        self.assertGreater(g["pi.weight"][-2:].abs().max().item(), 0.0)
        self.assertGreater(g["log_std"].abs().max().item(), 0.0)
        # only the shot decisions count: masking half the decisions changes the aim gradient
        half = dict(batch, aim_mask=(torch.arange(len(batch["adv"])) % 2).float())
        trainer = Trainer(NoEnv(), policy, PPOConfig())
        policy.zero_grad()
        trainer.losses(half, torch.arange(len(half["adv"])), half["adv"])["loss"].backward()
        self.assertFalse(torch.equal(policy.pi.weight.grad[-2:], g["pi.weight"][-2:]))

    def test_other_heads_do_not_see_the_mask(self):
        policy, batch = fake_batch(aim="gaussian")
        g0, g1 = self.grads(policy, batch, 0.0), self.grads(policy, batch, 1.0)
        k = sum(DISCRETE)
        self.assertTrue(torch.equal(g0["pi.weight"][:k], g1["pi.weight"][:k]))
        self.assertTrue(torch.equal(g0["pi.bias"][:k], g1["pi.bias"][:k]))
        self.assertTrue(torch.equal(g0["v.weight"], g1["v.weight"]))
        # a categorical aim is not masked at all
        policy, batch = fake_batch(aim="categorical")
        g0, g1 = self.grads(policy, batch, 0.0), self.grads(policy, batch, 1.0)
        self.assertTrue(all(torch.equal(g0[n], g1[n]) for n in g0))

    def test_unmasked_mode_is_the_joint_ratio(self):
        policy, batch = fake_batch(aim="gaussian")
        g0 = self.grads(policy, batch, 0.0, PPOConfig(aim_mask="all"))
        g1 = self.grads(policy, batch, 1.0, PPOConfig(aim_mask="all"))
        self.assertTrue(all(torch.equal(g0[n], g1[n]) for n in g0))
        self.assertGreater(g0["pi.weight"][-2:].abs().max().item(), 0.0)

    def test_aim_weight_zero_leaves_only_the_aim_gradient_out(self):
        policy, batch = fake_batch(aim="gaussian")
        b = dict(batch, aim_mask=torch.ones_like(batch["aim_mask"]))
        trainer = Trainer(NoEnv(), policy, PPOConfig())
        grads = []
        for w in (1.0, 0.0):
            policy.zero_grad()
            trainer.losses(b, torch.arange(len(b["adv"])), b["adv"], w)["loss"].backward()
            grads.append({n: p.grad.clone() for n, p in policy.named_parameters()})
        k = sum(DISCRETE)
        self.assertEqual(grads[1]["pi.weight"][k:].abs().max().item(), 0.0)
        self.assertEqual(grads[1]["log_std"].abs().max().item(), 0.0)
        self.assertGreater(grads[0]["pi.weight"][k:].abs().max().item(), 0.0)
        self.assertTrue(torch.equal(grads[0]["pi.weight"][:k], grads[1]["pi.weight"][:k]))
        self.assertTrue(torch.equal(grads[0]["v.weight"], grads[1]["v.weight"]))

    def trained(self, cfg, seed=0):
        torch.manual_seed(seed)
        policy, batch = fake_batch(n=1024, aim="gaussian")
        batch["aim_mask"] = torch.ones_like(batch["aim_mask"])
        start = {n: p.detach().clone() for n, p in policy.named_parameters()}
        stats = Trainer(NoEnv(), policy, cfg).train(batch)
        return start, {n: p.detach().clone() for n, p in policy.named_parameters()}, stats

    def test_aim_lr_scale_scales_only_the_aim_step(self):
        base = dict(epochs=1, minibatch=1024)
        s0, full, _ = self.trained(PPOConfig(**base))
        s1, scaled, _ = self.trained(PPOConfig(aim_lr_scale=0.25, **base))
        k = sum(DISCRETE)
        # one step: the aim rows and log_std move a quarter as far, everything else exactly as far
        for n, i in (("pi.weight", slice(k, None)), ("pi.bias", slice(k, None)), ("log_std", slice(None))):
            self.assertTrue(torch.allclose(scaled[n][i] - s1[n][i], 0.25 * (full[n][i] - s0[n][i]), atol=1e-7), n)
        self.assertTrue(torch.equal(scaled["pi.weight"][:k], full["pi.weight"][:k]))
        self.assertTrue(torch.equal(scaled["trunk.0.weight"], full["trunk.0.weight"]))

    def test_aim_kl_early_stop(self):
        k = sum(DISCRETE)
        base = dict(epochs=8, minibatch=256, lr=3e-3)
        _, _, free = self.trained(PPOConfig(**base))
        self.assertEqual(free["aim_off"], 0.0)
        start, end, stats = self.trained(PPOConfig(aim_target_kl=1e-6, **base))
        # the first minibatch measures zero KL and trains the aim; it is off for (almost) all the rest
        self.assertGreater(stats["aim_off"], 0.8)
        self.assertLess(stats["aim_approx_kl"], free["aim_approx_kl"])
        # the other heads keep training
        self.assertFalse(torch.equal(start["pi.weight"][:k], end["pi.weight"][:k]))
        _, frozen, _ = self.trained(PPOConfig(aim_target_kl=1e-6, aim_lr_scale=0.0, **base))
        self.assertTrue(torch.equal(frozen["log_std"], start["log_std"]))

    def test_checkpoint_keeps_the_aim_kind(self):
        policy, batch = fake_batch(aim="gaussian")
        trainer = Trainer(NoEnv(), policy, PPOConfig(epochs=1, minibatch=128))
        trainer.train(batch)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            save_checkpoint(path, trainer)
            loaded, state = load_policy(path)
            self.assertEqual(loaded.spec["aim"], "gaussian")
            self.assertTrue(torch.equal(loaded.log_std, policy.log_std))
            with self.assertRaises(ValueError):
                Trainer(NoEnv(), Policy(), PPOConfig()).load_state_dict(state)


@unittest.skipUnless(SERVER_DIR, "set TWRL_SERVER_DIR to a build directory with teeworlds_srv")
class TestTrainingLoop(unittest.TestCase):
    def test_collect_and_update_on_real_envs(self):
        for aim in ("categorical", "gaussian"):
            with self.subTest(aim=aim):
                self.collect_and_update(aim)

    def collect_and_update(self, aim):
        torch.manual_seed(0)
        with VecTeeworldsEnv(2, SERVER_DIR, seed=3, map="lms1", grid_format="codes", episode_ticks=100) as venv:
            trainer = Trainer(venv, Policy(aim=aim), PPOConfig(num_envs=2, steps=64, epochs=1, minibatch=128))
            for _ in range(2):
                batch = trainer.collect()
                self.assertEqual(batch["vec"].shape, (64 * 4, P.VEC_DIM))
                self.assertEqual(batch["codes"].shape, (64 * 4, P.GRID_HEIGHT, P.GRID_WIDTH))
                for k in ("adv", "returns", "logp", "values"):
                    self.assertTrue(torch.isfinite(batch[k]).all(), k)
                stats = trainer.train(batch)
                self.assertTrue(all(np.isfinite(v) for v in stats.values()))
            # 2 x 64 steps of 50-step episodes: both envs finished twice, events were summed
            self.assertEqual(len(trainer.finished), 4)
            self.assertEqual(trainer.agent_steps, 2 * 64 * 4)

    def test_frozen_opponent(self):
        torch.manual_seed(0)
        with VecTeeworldsEnv(4, SERVER_DIR, seed=3, map="lms1", grid_format="codes", episode_ticks=100, raw_info=True) as venv:
            learner = Policy(aim="gaussian")
            opponent = Policy(aim="gaussian")
            opponent.load_state_dict(learner.state_dict())
            frozen = [p.detach().clone() for p in opponent.parameters()]
            trainer = Trainer(venv, learner, PPOConfig(num_envs=4, steps=64, epochs=1, minibatch=128), opponent=opponent)
            np.testing.assert_array_equal(trainer.rows, [0, 3, 4, 7])  # learner is agent 0 in even envs, 1 in odd
            for _ in range(2):
                batch = trainer.collect()
                self.assertEqual(batch["vec"].shape, (64 * 4, P.VEC_DIM))  # learner samples only
                for k in ("adv", "returns", "logp", "values"):
                    self.assertTrue(torch.isfinite(batch[k]).all(), k)
                trainer.train(batch)
            self.assertEqual(trainer.agent_steps, 2 * 64 * 4)
            self.assertEqual(len(trainer.finished), 8)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(frozen, opponent.parameters())))
            self.assertFalse(all(torch.equal(a, b) for a, b in zip(frozen, learner.parameters())))
            # the aim mask marks exactly the learner decisions with a shot, and a shot needs fire=1
            self.assertGreater(batch["aim_mask"].sum().item(), 0)
            self.assertTrue(torch.all(batch["actions"][batch["aim_mask"] == 1, 2] == 1))
            alive, visible, _, shots, _, _ = trainer.behaviour
            self.assertGreater(alive, 0)
            self.assertLessEqual(visible, alive)
            self.assertGreater(shots, 0)


@unittest.skipUnless(SERVER_DIR, "set TWRL_SERVER_DIR to a build directory with teeworlds_srv")
class TestResume(unittest.TestCase):
    SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "train_ppo.py")

    def train(self, run_dir, *extra):
        cmd = [sys.executable, "-I", self.SCRIPT, "--server-dir", SERVER_DIR, "--run-dir", run_dir, "--envs", "2",
               "--steps", "32", "--threads", "2", *extra]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    def test_resume_after_a_hard_kill(self):
        with tempfile.TemporaryDirectory() as run_dir:
            cmd = [sys.executable, "-I", self.SCRIPT, "--server-dir", SERVER_DIR, "--run-dir", run_dir, "--envs", "2",
                   "--steps", "32", "--threads", "2", "--updates", "1000", "--checkpoint-every", "1"]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                deadline = time.time() + 300
                while not os.path.exists(os.path.join(run_dir, "ckpt_0003.pt")) and time.time() < deadline:
                    time.sleep(0.2)
            finally:
                os.killpg(proc.pid, signal.SIGKILL)  # the trainer and its servers, mid-run
                proc.wait()
            ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_[0-9]*.pt")))
            latest = ckpts[-1]
            state = torch.load(latest, map_location="cpu", weights_only=False)
            done = state["update"]
            self.assertGreaterEqual(done, 3)
            r = self.train(run_dir, "--resume", latest, "--updates", str(done + 2), "--checkpoint-every", "1000")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            final = torch.load(os.path.join(run_dir, "ckpt_final.pt"), map_location="cpu", weights_only=False)
            self.assertEqual(final["update"], done + 2)
            self.assertEqual(final["agent_steps"], state["agent_steps"] + 2 * 32 * 4)
            # the env seed stream moved on instead of replaying the first starts
            self.assertTrue(all(b > a for a, b in zip(state["env_episodes"], final["env_episodes"])))
            rows = [json.loads(line) for line in open(os.path.join(run_dir, "metrics.jsonl"))]
            self.assertEqual([r["update"] for r in rows][-2:], [done + 1, done + 2])

    def test_resume_refuses_other_settings(self):
        with tempfile.TemporaryDirectory() as run_dir:
            self.assertEqual(self.train(run_dir, "--updates", "1").returncode, 0)
            r = self.train(run_dir, "--resume", os.path.join(run_dir, "ckpt_final.pt"), "--updates", "2", "--map", "dm1")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("does not match", r.stderr)


if __name__ == "__main__":
    unittest.main()
