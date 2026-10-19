"""PPO plumbing tests. The unit tests need only torch; the integration tests need a built server:

    TWRL_SERVER_DIR=../teeworlds-0.7.5/build-release .venv/bin/python tests/test_ppo.py -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twrl import VecTeeworldsEnv  # noqa: E402
from twrl import protocol as P  # noqa: E402
from twrl.ppo import AIM_BINS, HEADS, Policy, PPOConfig, Trainer, load_policy, save_checkpoint, to_env_actions  # noqa: E402

SERVER_DIR = os.environ.get("TWRL_SERVER_DIR")


def fake_batch(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    policy = Policy()
    vec = torch.randn(n, P.VEC_DIM, generator=g)
    codes = torch.randint(0, 8, (n, P.GRID_HEIGHT, P.GRID_WIDTH), generator=g, dtype=torch.uint8) & 7
    with torch.no_grad():
        heads, logp, values, _ = policy.act(vec, codes, g)
    adv = torch.randn(n, generator=g)
    return policy, {"vec": vec, "codes": codes, "heads": heads, "logp": logp, "values": values, "adv": adv,
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
        _, ent, _ = policy.evaluate(batch["vec"], batch["codes"], batch["heads"])
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


@unittest.skipUnless(SERVER_DIR, "set TWRL_SERVER_DIR to a build directory with teeworlds_srv")
class TestTrainingLoop(unittest.TestCase):
    def test_collect_and_update_on_real_envs(self):
        torch.manual_seed(0)
        with VecTeeworldsEnv(2, SERVER_DIR, seed=3, map="lms1", grid_format="codes", episode_ticks=100) as venv:
            trainer = Trainer(venv, Policy(), PPOConfig(num_envs=2, steps=64, epochs=1, minibatch=128))
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


if __name__ == "__main__":
    unittest.main()
