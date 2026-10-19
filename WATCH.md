# Agent maçını izlemek

Server internete açılmaz. Canlı izleme mevcut SSH bağlantısının içinden, demo izleme dosya kopyalayarak yapılır.

```
Mac: Teeworlds ─UDP→ 127.0.0.1:8303 [udp_tcp_relay client] ─TCP→ 127.0.0.1:8304 ═ ssh -L ═▶
box: ~/.twrl-watch/relay.sock (0700 klasör) [udp_tcp_relay server] ─UDP→ 127.0.0.1:<sv_port> teeworlds_srv
```

- Agent'lar iki oyuncu slotunu doldurur (`sv_player_slots 2`), bağlanan herkes spectator olur.
- Agent'lar client'ta `AI-1` ve `AI-2` olarak görünür. Server browser'da oyuncu sayısı 0 görünür, çünkü server tarafı botlar server info'ya girmez.
- Relay her UDP paketini 2 byte uzunluk önekiyle taşır. Yalnız Python standart kütüphanesi kullanır.
- Varsayılan harita `lms1`. `--policy chase` yalnız test amaçlı kovalama script'idir, AI değildir.

## Canlı (box)

```bash
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
python3 scripts/watch_match.py live --server-dir ../teeworlds-0.7.5/build-release
```

Ctrl+C maçı ve relay'i kapatır.

## Canlı (Mac)

```bash
scp <remote-host>:<REPO_DIR>/scripts/udp_tcp_relay.py ~/udp_tcp_relay.py   # bir kez
ssh -N -L 127.0.0.1:8304:/home/<user>/.twrl-watch/relay.sock <remote-host>              # terminal 1
python3 ~/udp_tcp_relay.py client --listen-udp 127.0.0.1:8303 --connect-tcp 127.0.0.1:8304   # terminal 2
```

Teeworlds client'ta F1 konsolu: `connect localhost:8303`.

## Demo

```bash
# box: 5 episode'u (5 dakikalık oyun) birkaç saniyede kaydeder
python3 scripts/watch_match.py record --server-dir ../teeworlds-0.7.5/build-release --name twrl-random --episodes 5
# Mac
scp <remote-host>:/home/<user>/.local/share/teeworlds/demos/twrl-random.demo ~/Library/Application\ Support/Teeworlds/demos/
```

Teeworlds client'ta Demos sekmesinden açılır.

## Testler

```bash
python3 tests/test_relay.py -v
TWRL_SERVER_DIR=../teeworlds-0.7.5/build-release python3 tests/test_env.py -v
```
