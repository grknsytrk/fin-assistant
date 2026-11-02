# Commit planı ve geri alma

**Öncelik:** Son adım
**Bağımlılık:** İlgili fazlar tamamlanmış olmalı
**Commit:** bu dosya commit üretmez; sırayı ve mesajları tanımlar

---

## Sıra

| # | Commit mesajı | Faz | Dosyalar |
|---|---|---|---|
| 1 | `fix(ui): clamp virtual table range when rows shrink` | 01 | `useTableVirtualizer.ts`, `useTableVirtualizer.test.ts`, `FundsPage.tsx` |
| 2 | `feat(ui): add aria row metadata to virtualized tables` | 02 | `FundsPage.tsx`, `MarketsView.tsx` |
| 3 | `fix(edge): align flow-head cache ttl with poll cadence` | 03 | `worker/api-cache.ts`, `workerApiCache.test.ts` |
| 4 | `fix(ui): skip bist polling on full-day holidays` | 04 | `marketHours.ts`, `marketHours.test.ts` |
| 5 | `chore(config): make snapshot cache ttl configurable` | 05 | `financial_snapshot_store.py`, `test_financial_snapshot_store.py`, `README.md` |
| 6 | `fix(funds): bound process-local fund cache entries` | 06 | `fund_service.py`, `test_fund_service.py` |
| 7 | `fix(etag): serialize none content as json null` | 07 | `etag.py`, `test_etag.py`, `api.py` |
| 8 | `refactor(api): stop refresh executor before closing clients` | 08 (opsiyonel) | `kap_api_service.py`, `api.py` |

Kurallar:

- Her commit kendi başına derlenebilir ve test edilebilir olmalı.
- Faz içinde ara commit atma; commit'i faz bitince at.
- `git add -A` kullanma. Sadece o fazın dosyalarını ekle.
- Commit mesajında gövde (body) kullanacaksan 72 karakterden sonra satır kır.

Örnek:

```powershell
git add frontend/src/hooks/useTableVirtualizer.ts frontend/src/hooks/useTableVirtualizer.test.ts frontend/src/pages/FundsPage.tsx
git commit -m "fix(ui): clamp virtual table range when rows shrink"
```

---

## Faz 1 + 2 aynı dosyaya dokunuyor

`FundsPage.tsx` iki fazda da değişiyor. İki commit'te de aynı dosya görünecek; bu normal.
Commit 2'yi commit 1'den sonra atmak zorunlu.

## Faz 7 + 8 aynı dosyaya dokunuyor

`api.py` iki fazda da değişiyor. Sıra: 7 → 8.

---

## Geri alma

Tek faz geri alınacaksa:

```powershell
git revert <commit-sha>
```

Faz 1 geri alınırsa Faz 2 de gözden geçirilmeli: ARIA `startIndex`'e bağlı ve `FundsPage.tsx`
değişiklikleri iç içe. İkisini birlikte revert etmek daha temiz:

```powershell
git revert <faz-2-sha> <faz-1-sha>
```

Faz 8 opsiyonel olduğu için tek başına güvenle revert edilebilir.

---

## Commit öncesi son kontrol

```powershell
git diff --check
git status --short
```

- Whitespace hatası yok.
- Plan dışı dosya yok (`rag-fin-baslangic-tam-boy.png` hariç).
- Her commit'in mesajı AGENTS.md formatına uygun.
