# Devir notu (yeni oturum veya yeni hesap için)

Bu repo bir Claude Code oturumunda yapılan analizin çıktısıdır. Yeni bir oturumda şu sırayla devam et.

## Şu an nerede

- Analiz bitti: Teeworlds 0.7.5 server ([README.md](README.md) Bölüm 1) ve tmlp (Bölüm 2).
- Mimari önerisi hazır (Bölüm 3), faz planı [PLAN.md](PLAN.md).
- **Kod yazılmadı.** Kullanıcı ilk istekte "şimdilik KOD DEĞİŞTİRME, implementation yapma" demişti. Kodlamaya geçmeden önce kullanıcıdan açık onay al.

## Çalışma ortamı

- Teeworlds kaynağı: `/Users/bedo/Documents/ikinci-beyin/teeworlds` (0.7.5). Yeni makinede ayrıca klonlanmalı.
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

## Sonraki adım

Faz 0: `sv_rl_mode` ile tick kapısını kaldır, tick/sn ölç, determinizmi doğrula. Sonuca göre mimari kilitlenir.
