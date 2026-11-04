# Watching an agent match

The server is not exposed to the internet. Live watching goes through the existing SSH connection; demo watching works by copying files.

```
Mac: Teeworlds ─UDP→ 127.0.0.1:8303 [udp_tcp_relay client] ─TCP→ 127.0.0.1:8304 ═ ssh -L ═▶
box: ~/.twrl-watch/relay.sock (0700 folder) [udp_tcp_relay server] ─UDP→ 127.0.0.1:<sv_port> teeworlds_srv
```

- The agents fill the two player slots (`sv_player_slots 2`); everyone who connects becomes a spectator.
- On the client, the agents appear as `AI-1` and `AI-2`. The server browser shows 0 players, because server-side bots do not enter the server info.
- The relay carries each UDP packet with a 2-byte length prefix. It uses only the Python standard library.
- The default map is `lms1`. Both random and `--policy chase` are diagnostic tools, not AI: chase is only a scripted chasing script for testing and it shoots at walls.

## Live (box)

```bash
cd <REPO_DIR>
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once, the env needs numpy
.venv/bin/python scripts/watch_match.py live --server-dir ../teeworlds-0.7.5/build-release
```

Only one match can use a relay socket at a time. A second `live` refuses to start instead of taking over the socket; use `--relay-socket` for a different path.

Ctrl+C stops the match and the relay.

## Live, the PPO policy in training (box)

Training runs at full speed on its own servers. Watching starts a separate real-time (50 ticks/s) server and does not slow training down: one torch thread, 25 decisions per second.

```bash
.venv/bin/python scripts/watch_match.py live --server-dir ../teeworlds-0.7.5/build-release \
    --policy ppo --run-dir runs/<run> --relay-socket ~/.twrl-watch/ppo.sock
```

- Both agents play the run's newest `ckpt_*.pt` file.
- Every `--refresh-seconds` (default 60) seconds the script looks for a new checkpoint. If there is one, it loads it without cutting the episode and prints `policy: ckpt_0350.pt (update 350, 716800 agent steps)` to the terminal.
- Checkpoints are written atomically, so a half-written file is never read.
- The policy samples actions as in training (stochastic).
- On the Mac side only the socket path changes: `~/.twrl-watch/ppo.sock`.

## Live (Mac)

```bash
scp <remote-host>:<REPO_DIR>/scripts/udp_tcp_relay.py ~/udp_tcp_relay.py   # once
ssh -N -L 127.0.0.1:8304:/home/<user>/.twrl-watch/relay.sock <remote-host>              # terminal 1
python3 ~/udp_tcp_relay.py client --listen-udp 127.0.0.1:8303 --connect-tcp 127.0.0.1:8304   # terminal 2
```

In the Teeworlds client, open the F1 console: `connect localhost:8303`.

## Demo

```bash
# box: records 5 episodes (5 minutes of game time) in a few seconds
.venv/bin/python scripts/watch_match.py record --server-dir ../teeworlds-0.7.5/build-release --name twrl-random --episodes 5
# Mac
scp <remote-host>:/home/<user>/.local/share/teeworlds/demos/twrl-random.demo ~/Library/Application\ Support/Teeworlds/demos/
```

Open it from the Demos tab in the Teeworlds client.

## Tests

```bash
.venv/bin/python tests/test_relay.py -v
TWRL_SERVER_DIR=../teeworlds-0.7.5/build-release .venv/bin/python tests/test_env.py -v
```
