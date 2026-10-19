"""Scripted test helpers. Not a policy: only used to force fights in bridge tests."""

from .env import Action
from .protocol import OBS_INDEX as I


def chase_action(obs, rng, step):
    """Run and hook towards the opponent and fire when roughly level with it."""
    if not obs[I["alive"]] or not obs[I["opp_alive"]]:
        return Action(fire=step % 2)  # fire also asks for an early respawn, like a human
    dx, dy = obs[I["opp_dx"]], obs[I["opp_dy"]]
    direction = 0 if abs(dx) < 48 else (1 if dx > 0 else -1)
    hook = 1 if (step // 10) % 3 != 2 else 0
    jump = 1 if rng.random() < 0.15 else 0
    fire = step % 2 if (abs(dy) < 64 and abs(dx) < 700) else 0
    return Action(direction=direction, jump=jump, fire=fire, hook=hook, target_x=int(dx) or 1, target_y=int(dy))
