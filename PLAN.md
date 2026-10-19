# Faz planı

Durum: **Faz 0, 1 ve 2 tamamlandı (2026-10-19).** Faz 3: PPO altyapısı ve kısa bir deneme koşusu var (2026-10-20); uzun eğitim başlamadı. Mimari ve gerekçeler için [README.md](README.md).

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

## Faz 2: Gözlem ve ortam katmanı

Ayrıntılı ve kesin şartname: [ENVIRONMENT.md](ENVIRONMENT.md). Patch: [patches/0004-*](patches/), teeworlds branch `rl/phase2`.

- [x] Protokol v2: aksiyon değişmedi; OBS sabit 2340 byte, HELLO haritanın collision'ını bir kez gönderir.
- [x] Mekânsal gözlem: agent'ın tile'ı merkezde 33×25 tile egosentrik collision ızgarası (solid, death, nohook).
- [x] 67 boyutlu vektör: kendi durum, rakip (yalnız insanın göreceği dikdörtgen içinde), atış hattı biti, en yakın 4 mermi, episode ilerlemesi. Rakibin can/zırh/ammo değeri yok.
- [x] Seed'li spawn çeşitliliği: her episode'un ilk spawn'ı iki farklı rastgele harita spawn noktası (yalnız RL modunda).
- [x] `VecTeeworldsEnv`: N server tek Python process'ten, otomatik reset, numpy dizileri, `grid_format="codes"` seçeneği.
- [x] Testler (38 entegrasyon + 2 relay), uzun stres testi, Faz 1 ile aynı yük altında karşılaştırmalı benchmark.

### Sonuç

- 38/38 entegrasyon testi ve 2/2 relay testi geçti. Grid, haritanın kendisinden hesaplanan beklenen pencereyle her step'te birebir aynı (dm6'da death ve nohook tile'ları dahil); rakip göreli konumu, hızı, görüş alanı ve atış hattı ham durumla doğrulandı; mermi slotları sıralı ve sınırlar içinde.
- Determinizm: aynı seed ile episode'lar aynı process'te ve taze process'te birebir aynı; farklı seed farklı başlangıç ve iz; mod 0, 1, 2 aynı; vector env tek env ile birebir aynı. Bridge kapalıyken Faz 0 hash dizisi hâlâ birebir aynı.
- Stres: 8 env, toplam 3200 episode ve 9.6M tick (lms1 random, dm6 random, lms1 chase). Çökme, desync, NaN/Inf, aralık dışı değer, olay tutarsızlığı yok. dm6'da 636 suicide (death tile) doğal olarak üretildi. Server RSS ısınmadan sonra sabit, Python ~45 MB.
- ASan/UBSan: yeni koddan rapor yok (yalnız vanilla `CJobPool` kapanışındaki bilinen çift `pthread_join`).
- Payload: env step başına 2340 + 16 byte (Faz 1: 252 + 16). BuildObs server'da step başına 1.4-3 µs.
- Throughput, aynı yük altında arka arkaya (`bench/results/2026-10-19-phase1-vs-phase2.csv`, load 10-15):

| env | Faz 1, env başına process | Faz 2, env başına process | Faz 2, tek process |
|---|---|---|---|
| 1 | 37 636 | 30 036 | 29 944 |
| 4 | 119 538 | 94 653 | 54 670 |
| 8 | 183 594 | 141 377 | 84 947 |
| 12 | 212 709 | 164 822 | 90 660 |

Faz 2 aynı düzende yaklaşık %20-23 yavaş: Python'da step başına ~20 µs çözme (grid bitleri, kopyalar), server'da tick başına ~2 µs. Tek Python process'i env sayısından bağımsız ~90k tick/s'de doyuyor; Faz 3'te birden fazla worker process gerekecek.

## Faz 3: PPO (eğitim başlangıcı)

- [x] PPO self-play altyapısı: [twrl/ppo.py](twrl/ppo.py), [scripts/train_ppo.py](scripts/train_ppo.py). Tek ağ iki agent'ı da oynar, rastgele ağırlıklardan başlar. Grid için küçük CNN, vektör için MLP, ortak gövde, actor/critic başları. Aksiyon başları: yön 3, zıplama 2, ateş 2, hook 2, nişan 16 yön.
- [x] Ödüller env'in kendi ağırlıkları, değiştirilmedi (kill +1, death −1, suicide −2, hasar ±0.02). Chase eğitimde yok.
- [x] Zaman sınırıyla biten episode'larda son gözlemden bootstrap (truncation gerçek son değil).
- [x] Sonlu loss/gradyan kontrolü (NaN/Inf'te durur), checkpoint kaydetme/yükleme ve devam ettirme testleri: [tests/test_ppo.py](tests/test_ppo.py).
- [x] Davranış ölçümü: [scripts/eval_policy.py](scripts/eval_policy.py). Ölçümler ödüle girmez.
- [ ] Uzun eğitim, worker process'leri, ayrık nişanın yeterliliği, hafıza (GRU), rakip havuzu.

### Kısa deneme (2026-10-20, `bench/results/phase3-short/`)

lms1, 8 env, 400 güncelleme, 819 200 agent step (~410k env step, ~2.3 saat oyun zamanı), 12.5 dakika, CPU. **Öğrenme kanıtı değil**, yalnız döngünün çalıştığını gösterir.

- Bütün loss, gradyan ve KL değerleri sonlu; grad norm en fazla 1.04 (clip 0.5), approx KL en fazla 0.018. Checkpoint geri yüklenince ağ birebir aynı.
- Entropi 5.95'ten (düzgün dağılım) ~5.0-5.3'e indi: politika rastgelelikten uzaklaşıyor (zıplama %50 → %62, ateş %50 → %43).
- Değerlendirme, sabit seed'li 64 agent-episode, stokastik örnekleme:

| | başlangıç (ckpt_0000) | son | düzgün rastgele |
|---|---|---|---|
| kill / episode | 0.000 | 0.094 | 0.000 |
| verilen hasar / episode | 2.36 | 4.44 | 1.19 |
| isabet oranı (hasar/atış) | 0.021 | 0.039 | 0.011 |
| rakip görünür | 0.62 | 0.68 | 0.63 |
| görünürken atış hattı açık | 0.47 | 0.55 | 0.38 |
| nişan–rakip yönü cos | 0.002 | 0.014 | 0.005 |

- Karşılıklı maç (her iki tarafta 64 episode): son politika başlangıca ve rastgeleye karşı **üstün değil** (hasar 1.52 vs 1.66, 1.19 vs 1.36; kill 0 vs 0.016, 0.016 vs 0). Fark gürültü düzeyinde.
- Yorum: self-play'de karşılaşma ve hasar biraz artmış olabilir, ama nişan almayı öğrenmemiş (cos ≈ 0) ve rakiplere karşı avantaj yok. Örnek küçük, anlamlılık testi yapılmadı. Ödül çok seyrek: adım başına ortalama ödül ~0.

## Faz 4: Tam oyun ve self-play

- [ ] Silah değişimi (aksiyon + mevcut silah özellikleri), pickup'lar, daha büyük haritalar.
- [ ] Checkpoint havuzu: rakip örnekleme (%80 son, %20 eski), Elo takibi, champion.
- [ ] Görüş dışı rakip için hafıza: GRU veya frame stacking.
- [ ] Hasar yardımcı ödülünün etkisini ölç (açık/kapalı karşılaştırma).

Kabul ölçütü: yeni checkpoint eski havuza karşı kararlı biçimde kazanır, politika çökmesi yok.

## Faz 5: İzleme ve değerlendirme

- [x] Canlı spectator (SSH relay) ve demo kaydı: [WATCH.md](WATCH.md).
- [ ] Politikayı normal hızda oynatan izleme modu, insan karşısında oyun.
- [ ] Maç kayıtları, skor ve davranış metrikleri (isabet oranı, hook kullanımı, kaçınma).

## Açık riskler

- Faz 0'daki tick/sn sayısı (yukarıdaki karar kapısı).
- Ödül seyrekliği: tmlp (eski proje) büyük olasılıkla bu yüzden zorlandı. Davranış öğreten shaping kullanılmıyor; ilk önlemler küçük harita, tek silah ve hasar yardımcı ödülü. Yetmezse önce curriculum (harita, spawn mesafesi) denenir; shaping ayrı bir karar olur.
- Ayrık nişan kovaları ince isabeti sınırlayabilir. Gerekirse sürekli aksiyona geçilir.
- 0.7.5 sürümünde tmlp yaması doğrudan taşınamaz (tmlp 0.6 üzerine).
