# Sanitization note

Before publishing, personal and infrastructure details were replaced with placeholders. Nothing here changes how the code works. If you use or fork this repo, read this once so the placeholders do not surprise you.

## What was changed and why

| Removed | Replaced with | Why |
|---|---|---|
| SSH host alias of the author's dev server (`WATCH.md`, `tr/WATCH.md`, `scripts/watch_match.py`, `scripts/udp_tcp_relay.py`) | `<remote-host>` | Reveals the name of a private machine. Useless to others, and it is an extra target for scanning or social engineering. |
| Linux user name and home path on that server (`/home/<name>/...`) | `/home/<user>/...` | The user name is a login name for the private machine. |
| Absolute repo path on the server (a path under the server's home or dev directory) | `<REPO_DIR>` | Exposes the server's directory layout. |
| Local Claude Code session path and session id (`HANDOFF.md`, `tr/HANDOFF.md`) | `<project-dir>/<session-id>` | Contains the author's local user name and project folder. The id points to a private conversation file that is not in this repo anyway. |
| Author name and personal e-mail in the `From:` line of `patches/*.patch` | `teeworlds-rl <noreply@users.noreply.github.com>` | A personal e-mail in a public repo invites spam and phishing. |

## What was checked and kept

- **No API keys, tokens, passwords or private keys** exist in the files or in the git history (searched for key/secret/token/password/bearer patterns and common key formats).
- **IP addresses:** the only addresses are `127.0.0.1` (loopback). They are required for the code to run: the game server binds to loopback only, and the relay tunnels through SSH. They are not private and were kept. No public or LAN IP was ever in the repo.
- `ec_password` inside `patches/0001-*` is Teeworlds' own config definition with an empty default. It is not a secret.

## How to use the placeholders

Nothing to edit in the code. Only the printed hints and the docs use them:

- `<remote-host>`: the SSH host (or `~/.ssh/config` alias) of the machine that runs the server.
- `<user>`: your login name on that machine.
- `<REPO_DIR>`: where you cloned this repo on that machine.
- `<project-dir>/<session-id>`: only in the handoff note; replace with your own Claude Code session file if you use one.

`git am patches/*.patch` still applies the patches. The changed `From:` line only affects the commit author shown after applying them; use `git am --committer-date-is-author-date` or set your own author afterwards if you want your name on them.

## Git history

Older commits still carry the original author name and e-mail. Removing them needs a history rewrite (`git filter-repo`) and a force push, which breaks existing clones, so it was not done here. If that matters, do it deliberately before the repo is shared.
