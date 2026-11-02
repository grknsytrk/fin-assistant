# Faz 1 — Sanal tablo aralığı hatası

**Öncelik:** P0 (en kritik)
**Bağımlılık:** yok
**Sonraki faz:** `02-faz-2-aria.md` (bu faz bitmeden başlamaz)
**Commit:** `fix(ui): clamp virtual table range when rows shrink`

---

## Bağlam

`useTableVirtualizer` hook'u, `totalRows` küçüldüğünde `startIndex > endIndex` üretebiliyor.
Bu durumda `rows.slice(startIndex, endIndex)` boş döner ve tablo, kullanıcı scroll oynatana kadar
**tamamen boş** kalır. Hata latent: kullanıcı aşağı kaydırılmışken arama/filtre uygularsa tetiklenir.

Etkilenen üç kullanım:

1. `frontend/src/pages/FundsPage.tsx` — `FundsTable` (rowHeight 48)
2. `frontend/src/pages/MarketsView.tsx` — hisse tablosu (rowHeight 38)
3. `frontend/src/pages/MarketsView.tsx` — endeks detay tablosu (rowHeight 38/95)

Mevcut test (`useTableVirtualizer.test.ts:227`) bu hatayı **görmüyor**, çünkü testin içinde
`scrollTop = 0` yapılıyor; hatanın oluşma koşulu scrollTop'un yüksek kalması.

**Kapsam dışı:** ARIA, edge cache, backend. Sadece bu dosyalar:

- `frontend/src/hooks/useTableVirtualizer.ts`
- `frontend/src/hooks/useTableVirtualizer.test.ts`
- `frontend/src/pages/FundsPage.tsx`

---

## Adım 1.1 — `computeRange` düzeltmesi

Dosya: `frontend/src/hooks/useTableVirtualizer.ts`

Mevcut kod (satır 53-66):

```ts
    const computeRange = useCallback(
        (sTop: number, vHeight: number, total: number): VirtualRange => {
            const safeTotal = Math.max(0, total);
            if (safeTotal === 0) {
                return { startIndex: 0, endIndex: 0 };
            }
            const startRow = Math.floor(sTop / rowHeight);
            const visibleRowCount = Math.ceil(vHeight / rowHeight);
            const startIndex = Math.max(0, startRow - overscan);
            const endIndex = Math.min(safeTotal, startRow + visibleRowCount + overscan);
            return { startIndex, endIndex };
        },
        [rowHeight, overscan]
    );
```

Yeni kod — **birebir** bunu kullan:

```ts
    const computeRange = useCallback(
        (sTop: number, vHeight: number, total: number): VirtualRange => {
            const safeTotal = Math.max(0, total);
            if (safeTotal === 0) {
                return { startIndex: 0, endIndex: 0 };
            }
            const safeHeight = Math.max(0, vHeight);
            // Row count can shrink while the user is scrolled far down (search,
            // filter, live data refresh). Clamp the stale scrollTop to the new
            // content height so the window never inverts and renders nothing.
            const maxScrollTop = Math.max(0, safeTotal * rowHeight - safeHeight);
            const effectiveScrollTop = Math.min(Math.max(0, sTop), maxScrollTop);
            const startRow = Math.floor(effectiveScrollTop / rowHeight);
            const visibleRowCount = Math.ceil(safeHeight / rowHeight);
            const startIndex = Math.max(0, Math.min(startRow - overscan, safeTotal - 1));
            const endIndex = Math.min(
                safeTotal,
                Math.max(startIndex + 1, startRow + visibleRowCount + overscan),
            );
            return { startIndex, endIndex };
        },
        [rowHeight, overscan]
    );
```

**Neden bu hali:**

- `maxScrollTop` sayesinde eski/yüksek `scrollTop` yeni veri boyutuna çekilir.
- `startIndex` en fazla `safeTotal - 1` → her zaman geçerli son satır.
- `endIndex` en az `startIndex + 1` → `total > 0` iken asla boş aralık dönmez.
- Mevcut sayısal beklentiler korunur: `total=580, scrollTop=3800, clientHeight=600` için
  hâlâ `startIndex=90`, `endIndex=126`.

---

## Adım 1.2 — Regresyon testleri

Dosya: `frontend/src/hooks/useTableVirtualizer.test.ts`

Dosyanın sonuna ekle. **Kritik:** `scrollTop`'u sıfırlama; hatanın koşulu tam olarak budur.

```ts
    it('keeps a non-empty range when totalRows shrinks while scrolled far down', () => {
        const div = document.createElement('div');
        Object.defineProperty(div, 'clientHeight', { value: 600, configurable: true });
        let currentScrollTop = 5000;
        Object.defineProperty(div, 'scrollTop', {
            get: () => currentScrollTop,
            set: (val: number) => { currentScrollTop = val; },
            configurable: true,
        });
        const containerRef = { current: div };

        const { result, rerender } = renderHook(
            ({ totalRows }: { totalRows: number }) =>
                useTableVirtualizer({ totalRows, rowHeight: 38, overscan: 10, containerRef }),
            { initialProps: { totalRows: 580 } },
        );

        expect(result.current.startIndex).toBeGreaterThan(0);

        // No resetDeps change and scrollTop stays at 5000 on purpose: this is
        // the stale state that used to produce startIndex > endIndex.
        rerender({ totalRows: 8 });

        expect(result.current.startIndex).toBe(0);
        expect(result.current.endIndex).toBe(8);
        expect(result.current.endIndex).toBeGreaterThan(result.current.startIndex);
        expect(
            result.current.topSpacerHeight
            + (result.current.endIndex - result.current.startIndex) * 38
            + result.current.bottomSpacerHeight,
        ).toBe(8 * 38);
    });

    it.each([1, 2, 7, 50])('never returns an inverted range for totalRows=%i', (totalRows) => {
        const div = document.createElement('div');
        Object.defineProperty(div, 'clientHeight', { value: 600, configurable: true });
        Object.defineProperty(div, 'scrollTop', { value: 100000, configurable: true });
        const containerRef = { current: div };

        const { result } = renderHook(() =>
            useTableVirtualizer({ totalRows, rowHeight: 38, overscan: 10, containerRef })
        );

        expect(result.current.startIndex).toBeLessThan(result.current.endIndex);
    });
```

**Kanıt zorunluluğu:** Düzeltmeyi geçici olarak geri al (sadece `computeRange`'i eski haline getir),
`npm test -- src/hooks/useTableVirtualizer.test.ts` çalıştır ve yeni testin **kırmızı** olduğunu
göster. Sonra düzeltmeyi geri koy. Bu adımı atlama; testin gerçekten hatayı yakaladığının kanıtıdır.

---

## Adım 1.3 — `FundsTable` reset bağımlılıkları

Dosya: `frontend/src/pages/FundsPage.tsx`

**Kural:** `rows.length`'i reset deps'e **ekleme**. Clamp düzeltmesi doğruluğu zaten sağlıyor;
satır sayısı değişiminde tabloyu başa sarmak istemiyoruz.

Satır ~1030:

```ts
        resetDeps: resetDeps ?? [sortKey, sortOrder],
```

Satır ~6106:

```ts
    const tableResetDeps = useMemo(
        () => [searchTerm, deferredSearchTerm, fundTypeFilter, riskFilter, sortKey, sortOrder],
        [searchTerm, deferredSearchTerm, fundTypeFilter, riskFilter, sortKey, sortOrder],
    );
```

`deferredSearchTerm` eklenmesinin nedeni: filtre gerçekte bu değerle uygulanıyor; kullanıcı yazmayı
bıraktığında tablonun başa dönmesi beklenir.

---

## Doğrulama

```powershell
cd frontend
npm test -- src/hooks/useTableVirtualizer.test.ts
cd ..
```

Beklenen: mevcut testler + yeni testler geçer. Eski
`'clamps range safely when totalRows shrinks while scrolled down'` testi **değiştirilmeden** geçmeli.

Manuel:

1. `npm run dev`, `/fonlar`.
2. Tabloyu en alta kaydır.
3. Arama kutusuna `zzzzzz` yaz.
4. Tablo boş kalmamalı; "sonuç yok" görünmeli.
5. Aramayı temizle, tekrar en alta kaydır, filtre sekmesini değiştir; tablo baştan render etmeli.

---

## Definition of Done

- [ ] `computeRange` yeni kodla değiştirildi
- [ ] İki yeni test eklendi
- [ ] Düzeltme geri alındığında yeni testin kırmızı olduğu kanıtlandı
- [ ] `FundsTable` varsayılan resetDeps'ten `rows.length` çıkarıldı
- [ ] `tableResetDeps`'e `deferredSearchTerm` eklendi
- [ ] `npm test -- src/hooks/useTableVirtualizer.test.ts` yeşil
- [ ] Manuel QA yapıldı
- [ ] Tek commit atıldı

## Commit

```
fix(ui): clamp virtual table range when rows shrink
```

---

## Bu fazı ajana verirken kullan

```
docs/perf-fix/01-faz-1-virtualizer.md dosyasını uygula.
Önce docs/perf-fix/00-kurallar.md oku.
Sadece bu fazı yap; ARIA ve edge cache'e dokunma.
Kanıt adımını (düzeltmeyi geri al → test kırmızı → geri koy) atlama.
```
