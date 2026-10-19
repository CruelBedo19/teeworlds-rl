"""Scripted diagnostic helpers. Not a policy and never part of training: they read privileged
raw state (info["agents"]) to force fights in bridge tests. The chase happily shoots into walls."""

from .env import Action


def chase_actions(agents, rng, step):
    """Both agents run and hook towards each other and fire when roughly level."""
    actions = []
    for i in range(2):
        me, opp = agents[i], agents[1 - i]
        if not me["alive"] or not opp["alive"]:
            actions.append(Action(fire=step % 2))  # fire also asks for an early respawn, like a human
            continue
        dx = float(opp["pos"][0] - me["pos"][0])
        dy = float(opp["pos"][1] - me["pos"][1])
        direction = 0 if abs(dx) < 48 else (1 if dx > 0 else -1)
        hook = 1 if (step // 10) % 3 != 2 else 0
        jump = 1 if rng.random() < 0.15 else 0
        fire = step % 2 if (abs(dy) < 64 and abs(dx) < 700) else 0
        actions.append(Action(direction=direction, jump=jump, fire=fire, hook=hook, target_x=int(dx) or 1, target_y=int(dy)))
    return actions
