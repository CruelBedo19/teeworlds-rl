#!/usr/bin/env python3
"""Let a human watch two agents play, live or from a recorded demo.

Live, at the normal 50 ticks/s, reachable only through the relay socket (see udp_tcp_relay.py):
    watch_match.py live --server-dir ../teeworlds-0.7.5/build-release

Live with the learned policy of a training run, in its own realtime server next to the training
(the training itself keeps running at full speed). Both agents play the newest checkpoint of the
run, which is reloaded when a newer one appears (checked every --refresh-seconds):
    watch_match.py live --server-dir ../teeworlds-0.7.5/build-release --policy ppo --run-dir runs/x \
        --relay-socket ~/.twrl-watch/ppo.sock

Demo, recorded at full speed into ~/.local/share/teeworlds/demos/<name>.demo:
    watch_match.py record --server-dir ../teeworlds-0.7.5/build-release --name twrl-random --episodes 5

The agents take both player slots (sv_player_slots 2), so anyone who connects is a spectator.
"""

import argparse
import glob
import os
import random
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from twrl import TeeworldsEnv, random_action  # noqa: E402
from twrl import protocol as P  # noqa: E402
from twrl.protocol import EVENT_NAMES  # noqa: E402
from twrl.testing import chase_actions  # noqa: E402
import udp_tcp_relay  # noqa: E402

DEMO_DIR = os.path.expanduser("~/.local/share/teeworlds/demos")


class CheckpointFollower:
    """Both agents sample from the newest checkpoint of a run directory, reloaded when it changes.

    Checkpoints are written atomically (temp file + rename), so a listed ckpt_*.pt is always complete."""

    def __init__(self, run_dir, refresh_seconds, seed):
        import torch  # only this mode needs torch
        from twrl.ppo import load_policy, to_env_actions
        self.torch, self.load_policy, self.to_env_actions = torch, load_policy, to_env_actions
        torch.set_num_threads(1)  # 25 decisions/s; leave the cores to the training
        self.run_dir = run_dir
        self.refresh_seconds = refresh_seconds
        self.gen = torch.Generator().manual_seed(seed)
        self.policy = self.current = None
        self.checked = 0.0
        self.refresh(force=True)

    def newest(self):
        paths = glob.glob(os.path.join(self.run_dir, "ckpt_*.pt"))
        return max(paths, key=os.path.getmtime) if paths else None

    def refresh(self, force=False):
        if not force and time.time() - self.checked < self.refresh_seconds:
            return
        self.checked = time.time()
        path = self.newest()
        if path is None:
            if self.policy is None:
                sys.exit("no ckpt_*.pt in %s yet" % self.run_dir)
            return
        key = (path, os.path.getmtime(path))
        if key == self.current:
            return
        policy, state = self.load_policy(path)
        self.policy, self.current = policy, key
        print("policy: %s (update %d, %d agent steps)" % (os.path.basename(path), state["update"], state["agent_steps"]),
              flush=True)

    def __call__(self, agents):
        self.refresh()
        vec = self.torch.from_numpy(agents["vec"].copy())
        codes = self.torch.from_numpy(agents["grid"][:, :P.GRID_CELLS].reshape(P.NUM_AGENTS, P.GRID_HEIGHT, P.GRID_WIDTH).copy())
        with self.torch.no_grad():
            heads = self.policy.act(vec, codes, self.gen)[0]
        return self.to_env_actions(heads.numpy())


def play(env, args, episodes):
    rng = random.Random(args.seed)
    ppo = CheckpointFollower(args.run_dir, args.refresh_seconds, args.seed) if args.policy == "ppo" else None
    episode = 0
    while episodes == 0 or episode < episodes:
        obs, info = env.reset(seed=args.seed + episode)
        step = 0
        while True:
            if ppo is not None:
                actions = ppo(info["agents"])
            # chase and random are diagnostics, not the learned AI
            elif args.policy == "chase":
                actions = chase_actions(info["agents"], rng, step)
            else:
                actions = [random_action(rng), random_action(rng)]
            obs, _, _, truncated, info = env.step(actions)
            step += 1
            if truncated:
                break
        episode += 1
        score = info["events"][:, EVENT_NAMES.index("score")]
        print("episode %d done: AI-1 score %d, AI-2 score %d" % (info["episode_id"], score[0], score[1]), flush=True)


def run_live(args):
    port = int(subprocess.run(["freeport", "-u"], check=True, capture_output=True, text=True).stdout.split()[0])
    sock_path = os.path.abspath(os.path.expanduser(args.relay_socket))
    if udp_tcp_relay.socket_in_use(sock_path):
        sys.exit("a match is already being served on %s; stop it first or pass --relay-socket" % sock_path)
    relay_args = argparse.Namespace(listen_unix=sock_path, listen_tcp=None, udp="127.0.0.1:%d" % port)
    threading.Thread(target=udp_tcp_relay.run_server, args=(relay_args,), daemon=True).start()
    time.sleep(0.2)
    try:
        with TeeworldsEnv(args.server_dir, map=args.map, mode=0, episode_ticks=args.episode_ticks, port=port,
                          extra_commands=["sv_player_slots 2", "sv_max_clients 4", "sv_name twrl-watch"]) as env:
            print("server: udp 127.0.0.1:%d (pid %d), relay socket %s" % (port, env.server_pid, sock_path))
            print("on the Mac: ssh -N -L 127.0.0.1:8304:%s <remote-host>" % sock_path)
            print("            python3 udp_tcp_relay.py client --listen-udp 127.0.0.1:8303 --connect-tcp 127.0.0.1:8304")
            print("            Teeworlds: connect to localhost:8303. Ctrl+C here stops the match.", flush=True)
            play(env, args, args.episodes)
    except KeyboardInterrupt:
        pass
    finally:
        subprocess.run(["freeport", "--release", str(port), "--proto", "udp"], capture_output=True)
        try:
            os.unlink(sock_path)
        except OSError:
            pass


def run_record(args):
    path = os.path.join(DEMO_DIR, args.name + ".demo")
    # mode 1: full speed but with snapshots, which is what the demo recorder writes
    with TeeworldsEnv(args.server_dir, map=args.map, mode=1, episode_ticks=args.episode_ticks,
                      extra_commands=["record %s" % args.name]) as env:
        play(env, args, args.episodes)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    seconds = args.episodes * args.episode_ticks / 50
    print("demo: %s (%d KB, %d episodes, %.0f s of play)" % (path, size // 1024, args.episodes, seconds))
    print("on the Mac: scp <remote-host>:%s ." % path)
    if not size:
        sys.exit("the demo file is missing or empty")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("live", "record"):
        p = sub.add_parser(name)
        p.add_argument("--server-dir", required=True)
        p.add_argument("--map", default="lms1")
        p.add_argument("--policy", choices=("random", "chase", "ppo"), default="random")
        p.add_argument("--run-dir", help="--policy ppo: the training run whose newest checkpoint plays")
        p.add_argument("--refresh-seconds", type=float, default=60.0, help="--policy ppo: how often to look for a newer checkpoint")
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--episode-ticks", type=int, default=3000)
    sub.choices["live"].add_argument("--episodes", type=int, default=0, help="0 = until Ctrl+C")
    sub.choices["live"].add_argument("--relay-socket", default="~/.twrl-watch/relay.sock")
    sub.choices["record"].add_argument("--episodes", type=int, default=5)
    sub.choices["record"].add_argument("--name", default="twrl-random")
    args = ap.parse_args()
    if getattr(args, "policy", None) == "ppo" and not args.run_dir:
        ap.error("--policy ppo needs --run-dir")
    if args.cmd == "live":
        run_live(args)
    elif args.cmd == "record":
        run_record(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
