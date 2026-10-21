"""Visible decisions of a checkpoint vs ckpt_0000 (one side, eval seeds): does it move away from the
opponent, and does its aim follow its movement rather than the opponent?"""
import json, sys
import numpy as np, torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
from twrl import protocol as P
from twrl.env import VecTeeworldsEnv
from twrl.evaluation import EVAL_ENVS, EVAL_SEED, EVAL_MAP, EVAL_EPISODE_TICKS, Actor
V = P.VEC_INDEX
torch.set_num_threads(4)
srv, opp, out = sys.argv[1], sys.argv[2], {}
for ck in sys.argv[3:]:
    actors = (Actor(ck, 11), Actor(opp, 12))
    m = {"vis": 0, "away": 0, "towards": 0, "aim_x_with_move": 0, "moving": 0, "aim_x_towards": 0, "aim_up": 0, "opp_above": 0}
    with VecTeeworldsEnv(EVAL_ENVS, srv, seed=EVAL_SEED, map=EVAL_MAP, grid_format="codes", raw_info=True, episode_ticks=EVAL_EPISODE_TICKS) as venv:
        for _ in range(2):
            obs = venv.reset()
            while True:
                vec, codes = torch.from_numpy(obs["vec"]), torch.from_numpy(obs["grid"])
                acts = np.stack([actors[i](vec[:, i], codes[:, i]) for i in range(2)], axis=1)
                v = obs["vec"][:, 0]; a = acts[:, 0]
                vis = (v[:, V["opp_visible"]] > 0) & (v[:, V["alive"]] > 0)
                dx, dy = np.sign(v[:, V["opp_dx"]]), np.sign(v[:, V["opp_dy"]])
                mv = a[:, 0]
                ax, ay = np.sign(a[:, 4]), np.sign(a[:, 5])
                m["vis"] += vis.sum()
                m["moving"] += (vis & (mv != 0)).sum()
                m["away"] += (vis & (mv != 0) & (mv == -dx)).sum()
                m["towards"] += (vis & (mv != 0) & (mv == dx)).sum()
                m["aim_x_with_move"] += (vis & (mv != 0) & (ax == mv)).sum()
                m["aim_x_towards"] += (vis & (ax == dx)).sum()
                m["aim_up"] += (vis & (ay < 0)).sum()
                m["opp_above"] += (vis & (dy < 0)).sum()
                obs, _, _, trunc, _ = venv.step(acts)
                if trunc.all(): break
    n = max(m["vis"], 1); mo = max(m["moving"], 1)
    out[ck] = {"visible_decisions": int(m["vis"]), "moving_share": m["moving"] / n, "move_away_share": m["away"] / mo,
               "move_towards_share": m["towards"] / mo, "aim_x_same_side_as_move": m["aim_x_with_move"] / mo,
               "aim_x_towards_opp": m["aim_x_towards"] / n, "aim_up_share": m["aim_up"] / n, "opp_above_share": m["opp_above"] / n}
    print(ck, json.dumps({k: round(float(v), 4) for k, v in out[ck].items()}), flush=True)
