# Faz planı

Durum: **hiçbir faz başlamadı.** Kod yazılmadı, Teeworlds kaynağına dokunulmadı. Mimari ve gerekçeler için [README.md](README.md).

Hedef: iki AI'ın Teeworlds 0.7.5'te 1v1 Deathmatch self-play ile aim, movement, dodge ve combat öğrenmesi.

## Faz 0: Hız ölçümü (karar kapısı)

Amaç: server'ın gerçek zamandan hızlı kaç tick/sn atabildiğini ölç. Bu sayı mimarinin geri kalanını belirler.

- [ ] Teeworlds 0.7.5'i Release olarak derle (`teeworlds_srv`, client kapalı).
- [ ] `CServer::Run` içindeki `time_get()` kapısını (`engine/server/server.cpp:~1383`) ve `m_NetServer.Wait` uykusunu (~1420) atlayan `sv_rl_mode` ayarı ekle.
- [ ] Snapshot ve ağ yazımını (`DoSnapshot`, `PumpNetwork`) bu modda kapat.
- [ ] İki dummy client ile haritada tick/sn ölç (tek süreç, sonra N süreç).
- [ ] Sabit `srand` ile iki çalıştırmanın aynı sonucu verdiğini doğrula (determinizm).

Kabul ölçütü: ölçülen tick/sn raporlanır.
- 2000 tick/sn/süreç ve üstü: mimari aynen devam eder.
- 2000'in altı: fiziği hafif bir C++ simülatöre veya Python'a taşımayı yeniden değerlendir (`game/gamecore.cpp`, `game/collision.cpp`).

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
- [ ] Ödül: öldürme +1, ölme -1, hasar farkı yardımcı, küçük yakınlık ödülü (sonra azalt).
- [ ] PyTorch PPO + GAE, grad norm 0.5-1, adaptif entropi.
- [ ] Rakip: basit sabit bot (hareket eden, rastgele nişanlı).

Kabul ölçütü: politika basit botu belirgin farkla yener ve rastgele politikadan ölçülebilir iyi nişan isabeti gösterir.

## Faz 3: Tam oyun ve self-play

- [ ] Tam harita, tüm silahlar, pickup'lar (can, zırh, silah).
- [ ] Checkpoint havuzu: rakip örnekleme (%80 son, %20 eski), Elo takibi.
- [ ] İsteğe bağlı GRU (hook ve mermi dinamiği için).
- [ ] Yardımcı ödüllerin aşamalı kapatılması.

Kabul ölçütü: yeni checkpoint eski havuza karşı kararlı biçimde kazanır, politika çökmesi yok.

## Faz 4: İzleme ve değerlendirme

- [ ] `sv_rl_mode` kapalı, normal hızda server; insan client ile bağlanıp politikayı izle.
- [ ] Maç kayıtları, skor ve davranış metrikleri (isabet oranı, hook kullanımı, kaçınma).

## Açık riskler

- Faz 0'daki tick/sn sayısı (yukarıdaki karar kapısı).
- Ödül seyrekliği: tmlp (eski proje) büyük olasılıkla bu yüzden zorlandı. Yardımcı ödüller gerekli.
- Ayrık nişan kovaları ince isabeti sınırlayabilir. Gerekirse sürekli aksiyona geçilir.
- 0.7.5 sürümünde tmlp yaması doğrudan taşınamaz (tmlp 0.6 üzerine).
