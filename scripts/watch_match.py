#!/usr/bin/env python3
"""Let a human watch two agents play, live or from a recorded demo.

Live, at the normal 50 ticks/s, reachable only through the relay socket (see udp_tcp_relay.py):
    watch_match.py live --server-dir ../teeworlds-0.7.5/build-release

Demo, recorded at full speed into ~/.local/share/teeworlds/demos/<name>.demo:
    watch_match.py record --server-dir ../teeworlds-0.7.5/build-release --name twrl-random --episodes 5

The agents take both player slots (sv_player_slots 2), so anyone who connects is a spectator.
"""

import argparse
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
from twrl.testing import chase_action  # noqa: E402
import udp_tcp_relay  # noqa: E402

DEMO_DIR = os.path.expanduser("~/.local/share/teeworlds/demos")


def play(env, args, episodes):
    rng = random.Random(args.seed)
    episode = 0
    while episodes == 0 or episode < episodes:
        obs, info = env.reset(seed=args.seed + episode)
        step = 0
        while True:
            if args.policy == "chase":
                actions = [chase_action(obs[i], rng, step) for i in range(2)]
            else:
                actions = [random_action(rng), random_action(rng)]
            obs, _, _, truncated, info = env.step(actions)
            step += 1
            if truncated:
                break
        episode += 1
        print("episode %d done: AI-1 score %d, AI-2 score %d" % (
            info["episode_id"], info["events"][0]["score"], info["events"][1]["score"]), flush=True)


def run_live(args):
    port = int(subprocess.run(["freeport", "-u"], check=True, capture_output=True, text=True).stdout.split()[0])
    sock_path = os.path.abspath(os.path.expanduser(args.relay_socket))
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
        p.add_argument("--policy", choices=("random", "chase"), default="random")
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--episode-ticks", type=int, default=3000)
    sub.choices["live"].add_argument("--episodes", type=int, default=0, help="0 = until Ctrl+C")
    sub.choices["live"].add_argument("--relay-socket", default="~/.twrl-watch/relay.sock")
    sub.choices["record"].add_argument("--episodes", type=int, default=5)
    sub.choices["record"].add_argument("--name", default="twrl-random")
    args = ap.parse_args()
    if args.cmd == "live":
        run_live(args)
    elif args.cmd == "record":
        run_record(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
