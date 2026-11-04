# Handoff note (for a new session or a new account)

This repo is the output of an analysis done in a Claude Code session. In a new session, continue in this order.

## Where we are

- Analysis is done: Teeworlds 0.7.5 server ([README.md](README.md) Part 1) and tmlp (Part 2).
- The architecture proposal is ready (Part 3), phase plan in [PLAN.md](PLAN.md).
- Phases 0, 1 and 2 are done: patches in [patches/](patches/) (`git am patches/*.patch`), environment spec in [ENVIRONMENT.md](ENVIRONMENT.md), results in [PLAN.md](PLAN.md). Get explicit approval from the user before moving to Phase 3 (PPO).

## Working environment

- Teeworlds source: `git clone --branch 0.7.5 https://github.com/teeworlds/teeworlds`, then `git -c url."https://github.com/".insteadOf="git://github.com/" submodule update --init` (`.gitmodules` uses the old `git://` addresses). Apply the patches with `git am patches/*.patch`.
- Build (with CMake 4): `CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DCLIENT=OFF`. For the benchmark with dummies, also build `build-bench` with `-DCMAKE_CXX_FLAGS=-DCONF_DEBUG`.
- Line numbers come from sub-agent reports and were not verified one by one. Check them in the relevant files before writing code.
- The location of the `SERVER_TICK_SPEED` definition is not verified (its value is 50).

## First message for the new session

> Read README.md, PLAN.md and HANDOFF.md in the teeworlds-rl repo. Produce a file-level plan for Phase 0 (speed measurement). Do not write code until I approve.

## Opening the same session on another account

The session is not tied to the account; it is a local file:

```
~/.claude/projects/<project-dir>/<session-id>.jsonl
```

On the same computer: `/logout`, `/login` (new account), then in the project folder run
`claude --resume <session-id>`.
On another computer: copy the file to the same path and run the same command.
You can also continue without the session: this repo carries enough context.

## Next step

Match watching is ready: [WATCH.md](WATCH.md). Next: Phase 3 ([PLAN.md](PLAN.md)). Do not start without user approval.

The Python environment needs numpy: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Tests: `TWRL_SERVER_DIR=<build-release> .venv/bin/python tests/test_env.py -v`.
