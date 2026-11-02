# 00 — Ortak Kurallar (her faz için geçerli)

Bu dosyayı her fazdan önce oku. Buradaki kurallar tüm faz dosyalarında geçerlidir.

## Ortam

- Proje kökü: `D:\Projeler\rag-fin`
- Platform: Windows (PowerShell)
- Python: **her zaman** `.external\venv311\Scripts\python.exe`, proje kökünden
- Frontend komutları: `cd frontend` sonrası çalıştır, sonra `cd ..`
- Node/npm sürümleri repodaki `frontend/package-lock.json` ile sabit; `npm install` çalıştırma

## Başlangıç kontrolü

```powershell
git status --short
git log --oneline -3
```

Beklenen: sadece `?? rag-fin-baslangic-tam-boy.png` görünür. Başka değişiklik varsa **dur ve sor**.

## Test politikası

- Geniş suite **yasak**: çıplak `pytest`, çıplak `npm test` çalıştırma.
- Sadece ilgili faz dosyasındaki hedefli komutları çalıştır.
- Bir test kırmızıysa önce nedenini raporla; testi "geçirmek için" değiştirme.
  Test gerçekten yanlışsa bunu açıkça gerekçelendir.

## Commit formatı (AGENTS.md)

```
<type>(<scope>): <short summary>
```

- Küçük harf, emir kipi, en fazla ~72 karakter.
- Tipler: `feat`, `fix`, `chore`, `refactor`, `docs`, `test`, `perf`, `build`, `ci`, `style`.
- Her faz = bir commit. Faz içinde ara commit atma.

## YAPMA listesi (tüm fazlar)

1. **`_MEMORY_CACHE.clear()` çağrılarını kaldırma veya seyreltme.** Bayat fiyat okumasına yol açar.
2. **`rows.length`'i `tableResetDeps`'e ekleme.** Canlı veride tabloyu sürekli başa sarar.
3. **`overscan` değerini düşürme.** Mobil kayma için tek tampon.
4. **Tatil tarihi uydurma.** Özellikle 2027 dini bayramları.
5. **Flow-head TTL'ini 3 s'de bırakma.** Faz 3'ün amacı bu uyumsuzluğu gidermek.
6. **`_kap_post_json`'a retry ekleme.** Mevcut test bilinçli davranışı koruyor.
7. **Bütün test suite'ini çalıştırma.**
8. **"orjson 100 ms kazandırıyor" iddiasını tekrarlama.** Ölçüm ~4-30 ms.
9. **`wrangler.toml` (development) içindeki `EDGE_CACHE_ENABLED = "false"` değerini değiştirme.**
   Production ayarı `wrangler.production.toml` içinde.
10. **Yeni bağımlılık ekleme.** Ne Python ne npm.
11. **Plan dışı refactor yapma.** "Bu arada şunu da düzelttim" yasak.
12. **`git push` veya deploy yapma.** Sadece yerel commit.

## Çelişki durumunda

Plan ile kod uyuşmuyorsa (satır numarası kaymış, fonksiyon yeniden adlandırılmış, test zaten var):

1. Dur.
2. Ne beklediğini ve ne bulduğunu yaz.
3. Kullanıcıya sor.
4. Tahminle ilerleme.

## Faz sonunda raporlanacaklar

- Değişen dosyaların listesi
- Çalıştırılan doğrulama komutları ve **gerçek çıktı**
- DoD maddelerinin durumu (✅ / ❌ / atlandı + gerekçe)
- Commit SHA'sı
- Bilinen risk veya açık kalan nokta
