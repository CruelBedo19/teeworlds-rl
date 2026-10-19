# Devir notu (yeni oturum veya yeni hesap için)

Bu repo bir Claude Code oturumunda yapılan analizin çıktısıdır. Yeni bir oturumda şu sırayla devam et.

## Şu an nerede

- Analiz bitti: Teeworlds 0.7.5 server ([README.md](README.md) Bölüm 1) ve tmlp (Bölüm 2).
- Mimari önerisi hazır (Bölüm 3), faz planı [PLAN.md](PLAN.md).
- Faz 0 bitti: `sv_rl_mode` patch'i ([patches/](patches/)), benchmark ve determinizm sonuçları [PLAN.md](PLAN.md) Faz 0 bölümünde. Faz 1'e geçmeden önce kullanıcıdan açık onay al.

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

## Sonraki adım

Faz 1: köprü ve Python ortamı ([PLAN.md](PLAN.md)). Kullanıcı onayı olmadan başlama.
