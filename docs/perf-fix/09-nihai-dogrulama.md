# Faz 9 — Nihai doğrulama

**Öncelik:** Zorunlu (son adım)
**Bağımlılık:** Uygulanan tüm fazlar bitmiş olmalı
**Commit:** yok (doğrulama fazı)

Bu faz kod değiştirmez. Sadece kanıt toplar. Bir şey kırmızıysa **commit atma**, ilgili faza dön.

---

## 1. Çalışma ağacı kontrolü

```powershell
git status --short
git log --oneline -12
git diff --check
```

Beklenen:

- `git status` sadece `?? rag-fin-baslangic-tam-boy.png` gösterir (veya o da commit'lenmemişse).
- Her faz için bir commit görünür.
- `git diff --check` çıktısı boş (whitespace hatası yok).

---

## 2. Backend testleri

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_etag.py tests/test_financial_snapshot_store.py tests/test_http_client_pooling.py -q
```

Beklenen: tümü geçer. Hata varsa çıktıyı **olduğu gibi** rapora koy.

Faz 6'da eklenen test:

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_fund_service.py -q -k "memory_cache"
```

---

## 3. Frontend testleri

```powershell
cd frontend
npm test -- src/hooks/useTableVirtualizer.test.ts src/components/MarketFlowPanel.test.ts src/workerApiCache.test.ts src/utils/marketHours.test.ts
npm run build
cd ..
```

Beklenen: 4 dosya, tüm testler geçer; build başarılı.

> Uygulanmayan faz varsa ilgili test dosyasını komuttan çıkar. Örneğin Faz 4 atlandıysa
> `src/utils/marketHours.test.ts` henüz yoktur; komuta eklemek "no test files found" hatası verir.
> Faz 03 atlanmadıysa `workerApiCache.test.ts` her hâlükârda çalışır.

---

## 4. Regresyon kanıtı (Faz 1)

Faz 1'in testinin gerçekten hatayı yakaladığını bir kez daha doğrula:

1. `computeRange`'i geçici olarak eski haline getir.
2. `npm test -- src/hooks/useTableVirtualizer.test.ts` çalıştır → yeni test **kırmızı** olmalı.
3. Düzeltmeyi geri koy → test **yeşil** olmalı.

Bu adımın çıktısını rapora ekle. "Test geçti" demek yetmez; testin hatayı yakaladığını kanıtla.

---

## 5. Manuel QA listesi

| # | Senaryo | Beklenen |
|---|---|---|
| 1 | `/fonlar`: en alta kaydır → arama `zzzz` | Tablo boş kalmıyor, "sonuç yok" görünüyor |
| 2 | `/fonlar`: aramayı temizle → filtre değiştir | Tablo baştan render ediyor |
| 3 | `/piyasalar`: hisse tablosu en alta kaydır → sıralama değiştir | Boş kalmıyor |
| 4 | `/piyasalar` mobil: endeks detayı | 95 px satır hizası bozulmuyor |
| 5 | KAP flow paneli, seans içi | Network'te ~5 s'de bir istek |
| 6 | KAP flow paneli, gece/hafta sonu | ~60 s'de bir istek |
| 7 | Resmî tatil (29 Ekim) | ~60 s'de bir istek |
| 8 | Chrome DevTools → Elements | `aria-rowcount` / `aria-rowindex` doğru |
| 9 | Deploy sonrası `curl -I /api/market/flow/head` | `max-age=10` header'ı |

Deploy bu planın kapsamında değil; 9. madde sadece deploy edilirse kontrol edilir.

---

## 6. Faz raporu formatı

Ajanın final mesajı şu formatta olmalı:

```markdown
## Uygulanan fazlar
- [x] 01 — clamp düzeltmesi
- [x] 02 — ARIA
- [ ] 03 — atlandı (gerekçe)

## Doğrulama çıktıları
<komut>
<gerçek çıktı>

## Regresyon kanıtı
<eski kod + kırmızı test çıktısı>

## Commitler
<hash> <mesaj>

## Açık noktalar / riskler
- ...
```

---

## Definition of Done

- [ ] `git diff --check` temiz
- [ ] Backend hedefli testler yeşil
- [ ] Frontend hedefli testler yeşil
- [ ] `npm run build` yeşil
- [ ] Faz 1 regresyon kanıtı rapora eklendi
- [ ] Manuel QA tablosu dolduruldu
- [ ] Faz raporu yukarıdaki formatta yazıldı

---

## Bu fazı ajana verirken kullan

```
docs/perf-fix/09-nihai-dogrulama.md dosyasını uygula.
Kod değiştirme. Sadece komutları çalıştır, çıktıları topla ve raporu yaz.
Kırmızı bir şey varsa hangi faza dönülmesi gerektiğini söyle.
```
