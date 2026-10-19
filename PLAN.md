# Faz planı

Durum: **Faz 0 ve Faz 1 tamamlandı (2026-10-19).** Faz 2 başlamadı, kullanıcı onayı bekleniyor. Mimari ve gerekçeler için [README.md](README.md).

Hedef: iki AI'ın Teeworlds 0.7.5'te 1v1 Deathmatch self-play ile aim, movement, dodge ve combat öğrenmesi.

## Faz 0: Hız ölçümü (karar kapısı)

Amaç: server'ın gerçek zamandan hızlı kaç tick/sn atabildiğini ölç. Bu sayı mimarinin geri kalanını belirler.

- [x] Teeworlds 0.7.5'i Release olarak derle (`teeworlds_srv`, client kapalı).
- [x] `sv_rl_mode` (0 = gerçek zaman, 1 = hızlı tick, 2 = hızlı tick + yalın snapshot/network). Patch: [patches/](patches/), teeworlds branch `rl/phase0`, yalnız `engine/server/server.cpp`, `server.h`, `engine/shared/config_variables.h`.
- [x] `sv_rl_seed`, `sv_rl_bench_ticks`, `sv_rl_hash_interval`.
- [x] Benchmark matrisi ve determinizm testi: [bench/run_bench.py](bench/run_bench.py), sonuçlar [bench/results/](bench/results/).

### Sonuç (AMD Ryzen 5 3600, 6 çekirdek / 12 thread, `dm1`)

Medyan tick/s, 3 tekrar. Hızlı modlar 5 000 000 tick, mod 0 3000 tick.

| Durum | tick/s/süreç | toplam |
|---|---|---|
| mod 0, oyuncusuz / 2 dummy | 50 | 50 |
| mod 1, oyuncusuz | 870 726 | |
| mod 2, oyuncusuz | 2 952 218 | |
| mod 1, 2 dummy | 409 449 | |
| mod 2, 2 dummy | 633 979 | |
| mod 2, 2 dummy, 2 süreç | 625 771 | 1 252 108 |
| mod 2, 2 dummy, 4 süreç | 615 991 | 2 464 768 |
| mod 2, 2 dummy, 8 süreç | 505 802 | 4 056 516 |
| mod 2, 2 dummy, 12 süreç | 374 895 | 4 489 251 |

Determinizm: aynı config iki çalıştırma, mod 0/1/2 arası ve 4 paralel süreç aynı state hash dizisini üretti.

Sınırlar: debug dummy'ler yalnız yürür (ateş, zıplama, hook yok), bu yüzden sayılar savaş yükü olmadan üst sınırdır. Seed bu senaryoda sonucu değiştirmiyor (DM'de `rand()` kullanan yol tetiklenmiyor). İkisi de Faz 1'de bridge aksiyonlarıyla yeniden ölçülmeli.

Karar: süreç başına hız 2000 tick/s karar noktasının ~300 katı. Ayrı bir hafif simülatöre gerek yok; darboğaz Python tarafı olacak.

## Faz 1: Köprü ve Python ortamı

- [x] `sv_rl_socket` ayarlıysa server Python'un dinlediği Unix socket'e bağlanır ve iki agent ekler (`OnClientConnected(id, true)`, Release'te çalışır, `CONF_DEBUG` gerekmez). Patch: [patches/0002-*](patches/), teeworlds branch `rl/phase1`.
- [x] Lockstep: her `sv_rl_action_ticks` (varsayılan 2) tick'te OBS gönderilir, aksiyon beklenir; `CGameContext::OnTick` başında, `m_World.Tick()` öncesinde.
- [x] Girdi `CPlayer::OnDirectInput` + `OnPredictedInput` ile, fire sayacı client gibi.
- [x] Event'ler `CCharacter::TakeDamage` ve `Die` içindeki iki gözlemci hook'tan; reward ağırlıkları Python'da.
- [x] Reset `restart` komutuyla aynı yol (`DoWarmup(0)` → `StartMatch`), process yeniden başlamaz.
- [x] Sabit boyutlu binary protokol v1: [twrl/protocol.py](twrl/protocol.py) ve `src/game/server/rlenv.h`.
- [x] Python ortamı (yalnız stdlib): [twrl/env.py](twrl/env.py); testler [tests/test_env.py](tests/test_env.py); uzun koşu ve ölçüm [scripts/random_rollout.py](scripts/random_rollout.py).
- [ ] N süreçli vektörlü sarmalayıcı: Faz 2'ye (şimdilik rollout script'i `multiprocessing` ile paralel koşuyor).

Gözlem v1 (23 float, ham birim): kendi alive, pos, vel, health, armor, weapon, ammo, reload, grounded, jumped, hook state ve göreli hook konumu; rakip alive, göreli pos, vel, weapon, hook state; episode ilerlemesi. Rakibin health/armor/ammo değeri insan oyuncu da göremediği için yok. Aksiyon v1: direction, jump, fire, hook, target_x/y. Silah değişimi yok.

### Sonuç

- 25 entegrasyon testi geçti: hareket, zıplama, ateş, hook, hasar/kill/skor tutarlılığı, debug kill ile suicide, reset, determinizm, hata durumları (bozuk mesaj, yanlış seq, timeout, kopma, server kill, temiz kapanış).
- Uzun koşu: 3200 episode, 9.6M tick, 8 paralel env. Çökme, protokol hatası, timeout yok. Her step'te `Δskor == kill − suicide`, verilen hasar == rakibin aldığı hasar, kill == rakibin ölümü: 0 ihlal. Server RSS sabit (~15-16 MB, 250 episode'da en fazla +8 KB).
- Reset süresi p50 0.05 ms (yaklaşık 28 tick).
- Determinizm: aynı seed ile episode'lar hem aynı süreçte hem taze süreçte birebir aynı; farklı seed farklı; mod 0, 1, 2 aynı. Bridge kapalıyken Faz 0 hash dizisi birebir aynı.
- ASan/UBSan: yeni koddan rapor yok. Vanilla `CJobPool` kapanışındaki çift `pthread_join` ASan'da exit 1 veriyor (bridge kapalıyken de), kapsam dışı.
- Throughput (`bench/results/2026-10-19-phase1-scaling.csv`), toplam tick/s, K=2:

| env | random dm1 toplam | chase lms1 toplam | step latency p50 / p99 (µs) |
|---|---|---|---|
| 1 | 45 521 | 56 542 | 36 / 60 |
| 2 | 82 635 | 97 602 | 38 / 64 |
| 4 | 140 851 | 167 994 | 45 / 75 |
| 8 | 240 130 | 279 064 | 54 / 89 |
| 12 | 269 590 | 320 044 | 58 / 269 |

Darboğaz Faz 0'da beklendiği gibi Python ve IPC: server tek başına ~630k tick/s, bridge ile env başına ~45k tick/s.

Not: dm1'de random ajanlar neredeyse hiç karşılaşmıyor (6M tick'te 3 kill). Savaş testleri küçük `lms1` haritasında, yalnız test amaçlı bir kovalama script'iyle ([twrl/testing.py](twrl/testing.py)) yapıldı. Faz 2 için küçük harita seçimi önemli.

## Faz 2: Basit senaryo ve PPO

- [ ] Tek silah (tabanca veya lazer), düz harita, pickup yok.
- [ ] Gözlem: vektör (kendi hız, konum, can, hook; rakibe göreli konum ve hız) + küçük tile ızgarası.
- [ ] Aksiyon: hareket (3), zıpla, hook, ateş, ayrık nişan açısı (32-64 kova).
- [ ] Ödül: öldürme +1, ölme -1, verilen hasar küçük +, alınan hasar küçük −, suicide daha büyük −. Yakınlık ve nişan ödülü yok.
- [ ] PyTorch PPO + GAE, grad norm 0.5-1, adaptif entropi.
- [ ] Rakip: kendisi (iki ajan random weights ile başlar). Kural tabanlı bot zorunlu değil; gerekirse yalnızca baseline/değerlendirme veya ayrı bir curriculum deneyi.

Kabul ölçütü: politika random politikayı ve kendi eski checkpoint'lerini belirgin farkla yener. İsabet oranı yalnızca ölçülür, ödüle girmez.

## Faz 3: Tam oyun ve self-play

- [ ] Tam harita, tüm silahlar, pickup'lar (can, zırh, silah).
- [ ] Checkpoint havuzu: rakip örnekleme (%80 son, %20 eski), Elo takibi.
- [ ] İsteğe bağlı GRU (hook ve mermi dinamiği için).
- [ ] Hasar yardımcı ödülünün etkisini ölç (açık/kapalı karşılaştırma).

Kabul ölçütü: yeni checkpoint eski havuza karşı kararlı biçimde kazanır, politika çökmesi yok.

## Faz 4: İzleme ve değerlendirme

- [ ] `sv_rl_mode` kapalı, normal hızda server; insan client ile bağlanıp politikayı izle.
- [ ] Maç kayıtları, skor ve davranış metrikleri (isabet oranı, hook kullanımı, kaçınma).

## Açık riskler

- Faz 0'daki tick/sn sayısı (yukarıdaki karar kapısı).
- Ödül seyrekliği: tmlp (eski proje) büyük olasılıkla bu yüzden zorlandı. Davranış öğreten shaping kullanılmıyor; ilk önlemler küçük harita, tek silah ve hasar yardımcı ödülü. Yetmezse önce curriculum (harita, spawn mesafesi) denenir; shaping ayrı bir karar olur.
- Ayrık nişan kovaları ince isabeti sınırlayabilir. Gerekirse sürekli aksiyona geçilir.
- 0.7.5 sürümünde tmlp yaması doğrudan taşınamaz (tmlp 0.6 üzerine).
