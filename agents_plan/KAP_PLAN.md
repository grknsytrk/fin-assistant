# KAP + Extractor Genellestirme Plani (Revize)

## Amac
Mevcut Overview akisina hic dokunmadan, resmi finansallari ayri bir KAP sekmesinde gostermek ve KAP tarafini tamamen bagimsiz bir modul olarak calistirmak.

## Senin Istegine Gore Net Kapsam
- Mevcut mavi KPI kartlari kalkmayacak, aynen kalacak.
- KAP verisi ayri bir sekme olarak gelecek (ikame degil, tamamlayici kaynak).
- Cozum TAVHL-ozel olmayacak; BIM, MIGROS, SOK, TAVHL gibi farkli formatlarda genel calisacak.
- Eski RAG extractor/validator koduna MVP asamasinda dokunulmayacak.

## Kapsam Disi (MVP)
- Ucretli API entegrasyonu yok (yalnizca KAP HTML + cache).

## Mimari Yaklasim
- PDF-RAG akis aynen kalir.
- KAP verisi ek kaynak olarak gelir (ayri `Overview (KAP)` sekmesi).
- HTTP katmaninda `urllib` kullanilir (yeni bagimlilik yok).
- Sirket secimine gore sadece ilgili sirket + son 4 ceyrek cekilir.
- Sonuclar disk cache'e yazilir (TTL ile).
- Sekme goruntuleme ve veri cekme `kap.enabled` feature-flag ile ac/kapa yapilir.
- Mevcut Overview kod yolu ve gorunumu degistirilmez.
- KAP verisi, mevcut KPI hesaplama fonksiyonlarina baglanmaz (ayri kod yolu).

## Uygulama Adimlari (Oncelik Sirali)
1. **Config genisletmesi**
   - `src/config.py` ve `config.yaml` icine `kap` blogu ekle:
     - `enabled: true`
     - `timeout_seconds: 10`
     - `cache_ttl_hours: 24`
     - `user_agent`
2. **KAP fetch modulu**
   - `src/kap_fetcher.py` olustur:
     - company->ticker esleme
     - KAP HTML sayfasi cekme
     - gelir tablosu/bilanco/nakit akisi parse
     - normalize edilmis ceyrek bazli cikti
3. **Cache katmani**
   - `data/processed/kap_cache/{COMPANY}.json`
   - format: `{ fetched_at, source, data[] }`
   - stale kontrol: `cache_ttl_hours`
4. **UI entegrasyonu (ayri sekme)**
    - `app/ui.py` icinde mevcut `Overview` yanina yeni sekme/sayfa:
       - `Overview (KAP)` veya `KAP Finansallari`
       - son 4 ceyrek tablo
       - kaynak ve guncelleme zamani
   - Ana `Overview` davranisi ve icerigi degismez.
5. **Fallback ve hata yonetimi**
    - ticker yoksa KAP sekmesi bos durum mesaji gosterir
    - network/parse hatasinda sadece KAP sekmesi etkilenir, ana UI kirilmaz
   - cache varsa stale olsa bile "son bilinen veri" secenegiyle gosterilebilir
6. **Smoke test**
   - BIM, MIGROS, SOK, TAVHL ile:
    - KAP sekmesi gorunur/guncel
     - farkli donemlerde tablo degerleri duzgun gorunur
     - ana Overview ekraninda regression olmaz

## Dogrulama Kriterleri
- Ana Overview acilisinda uygulama davranisi degismez.
- KAP sekmesi acilisinda performans kabul edilebilir kalir (cache hit senaryosu).
- KAP veri yoksa veya hata varsa ana UI calismaya devam eder.
- Sirket secimi degistiginde sadece ilgili sirket verisi guncellenir.
- Eski RAG KPI cikti/hesaplama davranisi birebir korunur.

## Izolasyon ve Geri Alma Plani
- `kap.enabled = false` ile ozellik tek ayarla kapanir.
- Gerekirse navigation'dan KAP sekmesi kaldirilir; extractor disindaki KAP katmani ayrik kalir.
- Ana Overview kod yoluna temas edilmez.

## Faz 2 (Opsiyonel - Ayrica Onaylanir)
- `src/metrics_extractor.py` genellestirme (multi-currency, synonym, table parsing)
- `src/validators.py` unit-aware esikler
- Bu faz, MVP bagimsiz KAP sekmesi stabil olduktan sonra ve ayrica onayla uygulanir.

## Riskler ve Azaltma
- KAP HTML degisirse parse kirilabilir -> parser'i izole et, fallback uygula.
- Rate limit -> cache + kisitli istek.
- Sirket kodu uyumsuzlugu -> merkezi ticker mapping tablosu.

## Sonraki Opsiyon
MVP tamamlaninca resmi veri saglayici/API entegrasyonu (opsiyonel) ile parser bagimliligini azalt.
