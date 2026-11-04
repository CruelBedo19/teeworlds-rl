# Devir notu (yeni oturum veya yeni hesap için)

Bu repo bir Claude Code oturumunda yapılan analizin çıktısıdır. Yeni bir oturumda şu sırayla devam et.

## Şu an nerede (2026-11-02, branch `phase3`)

- Faz 0, 1 ve 2 bitti (bkz. [PLAN.md](PLAN.md)). Faz 3 (PPO) sürüyor, `phase3` branch'inde. Main'e merge yok.
- Geçerli baseline: shots-only aim mask, dry-fire penalty, ReDo (`--redo-every 25 --redo-tau 0`) ve aim log_std floor `-2.3`. Tam argümanlar: `runs/phase3-redo-floor-exactkl/seq.sh`.
- Performans karşılaştırmasında source of truth, ham eval dosyalarındaki `vs.initial` (= ckpt_0000). `vs.random` ile karıştırma.
- Seed 1, 15M, vs.ckpt_0000: K/D .078/0, W/L/D 5/0/59, damage 5.25/2.70 (ratio 1.94), hit .052, aim cos .37.

### Son deneyler (sonuçlar `bench/results/`)

- **`phase3-redo-floor`:** floor -2.3 ile koşu. 21.0M'de watchdog durdurdu. Sebep, minibatch k3 tahmininde (`aim_approx_kl`) tek bir 12.41'lik outlier; aynı update'te exact KL 0.145. Control'ü (`phase3-redo`) durduran 0.40 da aynı türden tek bir outlier'dı.
- **Commit `1bb9a25`:** watchdog artık `aim_kl_exact` okuyor. 20-update mean ve 0.25 limiti aynı. Testler: `tests/test_learn_pipeline.py`.
- **`phase3-redo-floor-exactkl`:** 15M'den 30M'e stop olmadan ulaştı.
  - Exact KL en fazla 0.14. Dead neuron FC katmanlarında 0, conv2'de %3–4.
  - 25M: ratio 1.90, hit .048, aim .29. Control 25M: ratio 1.48, aim -.03.
  - 30M: ratio 1.82, hit .033, aim .19.
  - Stabil, ama 15M'i geçmedi.

### Açık bulgular

- **aim_off ≈ %72–76:** KL brake aynı gürültülü k3 tahminiyle tetikleniyor, aim loss minibatch'lerin yaklaşık dörtte üçünde kapalı.
- **Floor sonrası KL artışı:** exact KL 0.090'dan 0.103'e çıktı, mu shift ise düştü. Eksen bazlı (x/y) ayrım loglanmıyor.
- **Watchdog boşluğu:** watchdog her poll'da yalnızca son satıra bakıyor. grad_norm_max 23.6 (update 5246) görülmeden geçti. Düzeltilmedi; düzeltilirse run'ları durdurabilir, karar kullanıcının.

## Kaldığımız yer ve sonraki adım

Kullanıcı onayladı, ama henüz başlanmadı: **offline aim_off ve x/y KL ölçümü.** Training yok, checkpoint'lere yazılmıyor.

1. Checkpoint'ler: `runs/phase3-redo-floor-exactkl/seed1/` altındaki 3663, 4883, 5372, 6104, 6836 ve 7325. Her birini `Trainer.load_state_dict` ile optimizer ve generator dahil yükle. Training'deki gibi ckpt_0000'a karşı 8 batch topla (bkz. `bench/collapse_probe.py`).
2. Her batch'te, checkpoint'in kopyası üzerinde tek bir update koş. Brake sinyalinin üç varyantı:
   - k3 (mevcut);
   - minibatch exact Gaussian KL (rollout öncesi mu0/s0'a göre);
   - brake yok.
3. Önce kontrol: `train()` döngüsünün replica'sı k3 varyantında gerçek `Trainer.train` ile birebir aynı sonucu vermeli.
4. Ölçülecekler: aim_off, brake'in ilk tetiklendiği minibatch, x ve y eksenleri ayrı mean-part KL, RMS Δμ, σ etkisi, k3/exact farkı.
5. Çıktı `bench/results/phase3-aim-brake-probe/` altına; local commit.

Kullanıcının kuralları:
- Onay olmadan yeni training başlatma, LR, reward veya watchdog limitini değiştirme.
- Self-play ya da scripted opponent kullanma.
- Checkpoint, log veya run silme; yarım kalan run'ları `failed-*` / `aborted-*` altına arşivle.
- Chase, relay ve spectator process'lerine dokunma.
- Push ve PR için onay iste.
- Raporlar Türkçe.

## Çalışma ortamı

- Teeworlds kaynağı: `git clone --branch 0.7.5 https://github.com/teeworlds/teeworlds`, sonra `git -c url."https://github.com/".insteadOf="git://github.com/" submodule update --init` (`.gitmodules` eski `git://` adreslerini kullanıyor). Patch'i `git am patches/*.patch` ile uygula.
- Build (CMake 4 ile): `CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DCLIENT=OFF`. Dummy'li benchmark için ayrıca `-DCMAKE_CXX_FLAGS=-DCONF_DEBUG` ile `build-bench`.
- Satır numaraları alt ajan raporlarından alındı, tek tek doğrulanmadı. Kod yazmadan önce ilgili dosyalarda kontrol et.
- `SERVER_TICK_SPEED` tanımının yeri doğrulanmadı (değeri 50).

## Yeni oturuma verilecek ilk mesaj

> teeworlds-rl reposundaki README.md, PLAN.md ve HANDOFF.md dosyalarını oku. Faz 0 (hız ölçümü) için dosya düzeyinde plan çıkar. Ben onaylamadan kod yazma.

## Aynı oturumu başka hesapta açmak

Oturum hesaba bağlı değil, yerel dosya:

```
~/.claude/projects/<project-dir>/<session-id>.jsonl
```

Aynı bilgisayarda: `/logout`, `/login` (yeni hesap), sonra proje klasöründe
`claude --resume <session-id>`.
Başka bilgisayarda: dosyayı aynı yola kopyala, aynı komutu çalıştır.
Oturum olmadan da devam edilebilir: bu repo yeterli bağlamı taşır.

Python ortamı numpy istiyor: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Testler: `TWRL_SERVER_DIR=<build-release> .venv/bin/python tests/test_env.py -v`.
