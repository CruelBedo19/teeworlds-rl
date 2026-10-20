# Agent maçını izlemek

Server internete açılmaz. Canlı izleme mevcut SSH bağlantısının içinden, demo izleme dosya kopyalayarak yapılır.

```
Mac: Teeworlds ─UDP→ 127.0.0.1:8303 [udp_tcp_relay client] ─TCP→ 127.0.0.1:8304 ═ ssh -L ═▶
box: ~/.twrl-watch/relay.sock (0700 klasör) [udp_tcp_relay server] ─UDP→ 127.0.0.1:<sv_port> teeworlds_srv
```

- Agent'lar iki oyuncu slotunu doldurur (`sv_player_slots 2`), bağlanan herkes spectator olur.
- Agent'lar client'ta `AI-1` ve `AI-2` olarak görünür. Server browser'da oyuncu sayısı 0 görünür, çünkü server tarafı botlar server info'ya girmez.
- Relay her UDP paketini 2 byte uzunluk önekiyle taşır. Yalnız Python standart kütüphanesi kullanır.
- Varsayılan harita `lms1`. Random de `--policy chase` de tanı araçlarıdır, AI değildir: chase yalnız test amaçlı kovalama script'idir ve duvarlara ateş eder.

## Canlı (box)

```bash
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # bir kez, env numpy istiyor
.venv/bin/python scripts/watch_match.py live --server-dir ../teeworlds-0.7.5/build-release
```

Bir relay socket'ini aynı anda tek maç kullanabilir. İkinci bir `live` socket'i devralmak yerine başlamayı reddeder; başka bir yol için `--relay-socket`.

Ctrl+C maçı ve relay'i kapatır.

## Canlı, eğitimdeki PPO policy'si (box)

Eğitim tam hızda kendi server'larında sürer. İzleme ayrı bir gerçek zamanlı (50 tick/s) server açar ve eğitimi yavaşlatmaz: tek torch thread'i, saniyede 25 karar.

```bash
.venv/bin/python scripts/watch_match.py live --server-dir ../teeworlds-0.7.5/build-release \
    --policy ppo --run-dir runs/<koşu> --relay-socket ~/.twrl-watch/ppo.sock
```

- İki agent da koşunun en yeni `ckpt_*.pt` dosyasını oynar.
- Script her `--refresh-seconds` (varsayılan 60) saniyede yeni checkpoint arar. Yenisi varsa episode'u kesmeden yükler ve terminale `policy: ckpt_0350.pt (update 350, 716800 agent steps)` yazar.
- Checkpoint'ler atomik yazıldığı için yarım dosya okunmaz.
- Policy aksiyonları eğitimdeki gibi örnekler (stokastik).
- Mac tarafında yalnız socket yolu değişir: `~/.twrl-watch/ppo.sock`.

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
.venv/bin/python scripts/watch_match.py record --server-dir ../teeworlds-0.7.5/build-release --name twrl-random --episodes 5
# Mac
scp <remote-host>:/home/<user>/.local/share/teeworlds/demos/twrl-random.demo ~/Library/Application\ Support/Teeworlds/demos/
```

Teeworlds client'ta Demos sekmesinden açılır.

## Testler

```bash
.venv/bin/python tests/test_relay.py -v
TWRL_SERVER_DIR=../teeworlds-0.7.5/build-release .venv/bin/python tests/test_env.py -v
```
