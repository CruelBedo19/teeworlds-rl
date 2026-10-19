# Faz planı

Durum: **Faz 0 tamamlandı (2026-10-19), karar için kullanıcı onayı bekleniyor.** Faz 1 başlamadı. Mimari ve gerekçeler için [README.md](README.md).

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

- [ ] Dummy ajanları Release'te de açan ayar (`CONF_DEBUG` kapısı yerine yeni config).
- [ ] `IRLBridge` arayüzü: her K tick'te gözlem yaz, aksiyon oku; `CGameContext::OnTick` içinden, `m_World.Tick()` öncesinde.
- [ ] Girdi enjeksiyonu: `OnPredictedInput` **ve** `OnDirectInput` (ateş yalnız `m_LatestInput` okur).
- [ ] `ResetEpisode()`: `ResetGame` ile skor, karakter, pickup ve mermileri sıfırla, süreç yeniden başlamasın.
- [ ] İkili protokol (Unix socket). `step(actions) -> obs, reward, done`.
- [ ] Python çok ajanlı Gymnasium/PettingZoo benzeri ortam, N süreçli vektörlü sarmalayıcı.
- [ ] Rastgele politika ile uçtan uca test: 1000 bölüm, çökme yok, sızıntı yok.

Kabul ölçütü: rastgele politika N paralel ortamda kararlı çalışır, bölüm sıfırlama < 10 ms.

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
