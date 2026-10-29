"""Watchdog tests for bench/learn_pipeline.py; need only numpy and torch:

    .venv/bin/python tests/test_learn_pipeline.py -v
"""

import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))

import learn_pipeline as lp  # noqa: E402
from tests.test_ppo import NoEnv, fake_batch  # noqa: E402
from twrl.ppo import PPOConfig, Trainer  # noqa: E402

LIMIT = 0.25  # the long runs' --aim-kl-limit


def rows(n=40, exact=0.09, approx=0.07, over=None):
    """n healthy metrics rows past aim_kl_after_steps; over maps an index to fields to replace."""
    out = []
    for i in range(n):
        r = {"update": 5000 + i, "agent_steps": 20_000_000 + 4096 * i, "grad_norm_max": 2.0, "entropy": 2.4,
             "log_std_x": -1.44, "log_std_y": -2.3, "shots": 250.0, "aim_approx_kl": approx, "aim_kl_exact": exact}
        r.update((over or {}).get(i, {}))
        out.append(r)
    return out


class Watchdog(unittest.TestCase):
    def setUp(self):
        self.saved = dict(lp.RULES)
        lp.RULES["aim_kl_mean_above"] = LIMIT

    def tearDown(self):
        lp.RULES.clear()
        lp.RULES.update(self.saved)

    def test_healthy_run_passes(self):
        self.assertIsNone(lp.watchdog(rows(), 0))

    def test_one_approx_outlier_does_not_stop(self):
        # update 5108 of runs/phase3-redo-floor: k3 estimate 12.41, exact 0.145
        r = rows(over={35: {"aim_approx_kl": 12.41, "aim_kl_exact": 0.145}})
        self.assertGreater(np.mean([x["aim_approx_kl"] for x in r[-20:]]), LIMIT)
        self.assertIsNone(lp.watchdog(r, 0))

    def test_exact_mean_above_the_limit_stops(self):
        r = rows(exact=0.26)
        reason = lp.watchdog(r, 0)
        self.assertIsNotNone(reason)
        self.assertIn("aim_kl_exact", reason)
        # a single large exact KL is enough when it lifts the 20-update mean over the limit
        r = rows(over={39: {"aim_kl_exact": 0.09 + 20 * (LIMIT - 0.09) + 0.01}})
        self.assertIsNotNone(lp.watchdog(r, 0))
        self.assertIsNone(lp.watchdog(rows(over={39: {"aim_kl_exact": 0.09 + 20 * (LIMIT - 0.09) - 0.01}}), 0))

    def test_window_and_warmup_are_unchanged(self):
        self.assertEqual(lp.RULES["aim_kl_window"], 20)
        self.assertIsNone(lp.watchdog(rows(n=19, exact=1.0), 0))
        early = rows(exact=1.0)
        for i, r in enumerate(early):
            r["agent_steps"] = 4096 * i
        self.assertIsNone(lp.watchdog(early, 0))

    def test_updates_without_shots_are_skipped(self):
        r = rows(exact=0.26)
        for x in r[-5:]:
            del x["aim_kl_exact"]
        self.assertIn("aim_kl_exact", lp.watchdog(r, 0))
        self.assertIsNone(lp.watchdog(rows(), 0))

    def test_old_runs_fall_back_to_the_estimate(self):
        r = rows(approx=0.3)
        for x in r:
            del x["aim_kl_exact"]
        self.assertIn("aim_approx_kl", lp.watchdog(r, 0))

    def test_non_finite_still_stops(self):
        for field in ("aim_kl_exact", "aim_approx_kl", "grad_norm_max", "log_std_y"):
            for bad in (float("nan"), float("inf")):
                r = rows(over={39: {field: bad}})
                self.assertIn("non-finite", lp.watchdog(r, 0), (field, bad))

    def test_other_rules_unchanged(self):
        self.assertIn("grad norm", lp.watchdog(rows(over={39: {"grad_norm_max": 25.0}}), 0))
        self.assertIn("log_std", lp.watchdog(rows(over={39: {"log_std_y": -4.6}}), 0))
        self.assertIn("no new metrics", lp.watchdog(rows(), 11))


class BothKLsLogged(unittest.TestCase):
    def test_train_reports_approx_and_exact(self):
        torch.manual_seed(0)
        policy, batch = fake_batch(n=1024, aim="gaussian")
        stats = Trainer(NoEnv(), policy, PPOConfig(epochs=2, minibatch=256, aim_target_kl=0.015,
                                                   aim_log_std_floor=-2.3)).train(batch)
        for k in ("aim_approx_kl", "aim_kl_exact", "aim_kl_mean_part", "aim_kl_std_part", "aim_mu_shift", "aim_off",
                  "floor_hit_x", "floor_hit_y"):
            self.assertIn(k, stats)
            self.assertTrue(np.isfinite(stats[k]), k)

    def test_train_ppo_logs_both_rolling_means(self):
        with open(os.path.join(ROOT, "scripts", "train_ppo.py")) as f:
            src = f.read()
        self.assertIn('stats["aim_kl_roll20"]', src)
        self.assertIn('stats["aim_kl_exact_roll20"]', src)

    def test_window_stats_carry_both(self):
        s = lp.window_stats(rows(), 20_000_000 + 4096 * 39)
        self.assertIn("aim_approx_kl", s)
        self.assertIn("aim_kl_exact", s)


if __name__ == "__main__":
    unittest.main()
