# RAG-FIN — Performans Düzeltme Planı (Tek Dosya)

Bu dosya, Gemini'nin uyguladığı yedi performans değişikliğinin incelenmesi sonucu ortaya çıkan
**tüm düzeltme işlerini tek yerde** toplar. Kurallar, fazlar, birebir kod, testler, doğrulama
komutları, Definition of Done listeleri ve commit sırası bu dosyanın içindedir.

> Faz bazında ayrılmış sürüm de mevcut: `docs/perf-fix/`. İçerik aynıdır.
> Bir fazı tek başına bir ajana vermek isterseniz o klasördeki dosyayı kullanın.

---

## İçindekiler

- [0. Nasıl kullanılır](#0-nasıl-kullanılır)
- [1. Ortam](#1-ortam)
- [2. Değişmez kurallar](#2-değişmez-kurallar)
- [3. Tespit özeti](#3-tespit-özeti)
- [4. Genel bakış](#4-genel-bakış)
- [5. Faz 1 — Sanal tablo aralığı hatası (P0)](#faz-1--sanal-tablo-aralığı-hatası-p0)
- [6. Faz 2 — ARIA satır bilgisi](#faz-2--aria-satır-bilgisi)
- [7. Faz 3 — Flow-head edge cache](#faz-3--flow-head-edge-cache-tutarlılığı)
- [8. Faz 4 — BIST tatil takvimi](#faz-4--bist-tatil-takvimi)
- [9. Faz 5 — Snapshot TTL env](#faz-5--snapshot-cache-ttlini-yapılandırılabilir-yap)
- [10. Faz 6 — `_MEMORY_CACHE` sınırı](#faz-6--_memory_cache-sınırı)
- [11. Faz 7 — orjson `None` + ölçüm notu](#faz-7--orjson-none-tutarlılığı-ve-ölçüm-notu)
- [12. Faz 8 — Shutdown sırası (opsiyonel)](#faz-8--shutdown-sırasında-client-kapatma-opsiyonel)
- [13. Nihai doğrulama](#faz-9--nihai-doğrulama)
- [14. Commit planı ve geri alma](#commit-planı-ve-geri-alma)
- [15. Ajana verilecek promptlar](#ajana-verilecek-promptlar)

---

## 0. Nasıl kullanılır

İki mod var. İkisi de bu dosyayla çalışır.

### Mod A — Tek faz, tek oturum (önerilen)

Her faz ayrı bir ajan oturumuna verilir. En iyi sonucu bu verir:

- Ajanın kapsamı tek dosya grubuyla sınırlı kalır.
- Tek DoD, tek commit, tek doğrulama komutu olur.
- Bir şey kırılırsa hangi fazın bozduğu bellidir.

Sıra:

```
01 → 02 → (03 ∥ 04 ∥ 05 ∥ 06) → 07 → 08 → 09
```

### Mod B — Tek oturum, tüm plan

Tüm plan tek seferde verilir. Daha hızlıdır ama risklidir:

- Ajan 8 fazı birden görünce kapsamı genişletir.
- DoD'leri tek tek kanıtlamaz.
- Commit'ler karışır.
- Doğrulama sonda tek seferde yapılır; kırmızı çıkınca kaynağı belli olmaz.

Bu modu seçerseniz ajana **"her faz için ayrı commit, her faz sonunda doğrulama"** kuralını
açıkça yazın ve Faz 1 ile Faz 2'yi yine de ayrı vermeyi düşünün (ikisi de `FundsPage.tsx`).

---

## 1. Ortam

- Proje kökü: `D:\Projeler\rag-fin`
- Platform: Windows (PowerShell)
- Python: **her zaman** `.external\venv311\Scripts\python.exe`, proje kökünden çalıştırılır
- Frontend komutları: `cd frontend` → komut → `cd ..`
- `npm install` **çalıştırılmaz**; sürümler `frontend/package-lock.json` ile sabit
- Yeni bağımlılık eklenmez (ne Python ne npm)

### Başlangıç kontrolü

```powershell
git status --short
git log --oneline -3
```

Beklenen: sadece `?? rag-fin-baslangic-tam-boy.png` görünür. Başka değişiklik varsa **dur ve sor.**

### Test politikası

- Geniş suite **yasak**: çıplak `pytest`, çıplak `npm test` çalıştırılmaz.
- Sadece ilgili fazdaki hedefli komutlar çalıştırılır.
- Test kırmızıysa önce nedeni raporlanır; testi "geçirmek için" değiştirmek yasak.
  Test gerçekten yanlışsa bu açıkça gerekçelendirilir.

---

## 2. Değişmez kurallar

Aşağıdakiler **tüm fazlarda** geçerlidir. İhlali geri alınır.

1. **`_MEMORY_CACHE.clear()` çağrılarını kaldırma veya seyreltme.** Bayat fiyat okumasına yol açar.
2. **`rows.length`'i `tableResetDeps`'e ekleme.** Canlı veride tabloyu sürekli başa sarar.
3. **`overscan` değerini düşürme.** Mobil kayma için tek tampon.
4. **Tatil tarihi uydurma.** Özellikle 2027 dini bayramları.
5. **Flow-head TTL'ini 3 s'de bırakma.** Faz 3'ün amacı bu uyumsuzluğu gidermek.
6. **`_kap_post_json`'a retry ekleme.** Mevcut test bilinçli davranışı koruyor.
7. **Bütün test suite'ini çalıştırma.**
8. **"orjson 100 ms kazandırıyor" iddiasını tekrarlama.** Ölçüm ~4–30 ms.
9. **`wrangler.toml` içindeki `EDGE_CACHE_ENABLED = "false"` değerini değiştirme.**
   Production ayarı `wrangler.production.toml` içinde.
10. **Yeni bağımlılık ekleme.** Ne Python ne npm.
11. **Plan dışı refactor yapma.** "Bu arada şunu da düzelttim" yasak.
12. **`git push` veya deploy yapma.** Sadece yerel commit.

### Çelişki durumunda

Plan ile kod uyuşmuyorsa (satır numarası kaymış, fonksiyon yeniden adlandırılmış, test zaten var):

1. Dur.
2. Ne beklendiğini ve ne bulunduğunu yaz.
3. Kullanıcıya sor.
4. Tahminle ilerleme.

### Commit formatı (AGENTS.md)

```
<type>(<scope>): <short summary>
```

- Küçük harf, emir kipi, en fazla ~72 karakter.
- Tipler: `feat`, `fix`, `chore`, `refactor`, `docs`, `test`, `perf`, `build`, `ci`, `style`.
- Her faz = bir commit. Faz içinde ara commit atılmaz.
- `git add -A` kullanılmaz; sadece o fazın dosyaları eklenir.

---

## 3. Tespit özeti

Yedi maddenin kodda karşılığı var; ancak bazı gerekçeler yanlış veya abartılıydı.
Bu plan, doğru olanları korur ve hatalı olanları düzeltir.

| # | Gemini'nin iddiası | Gerçek durum | Bu plandaki karşılık |
|---|---|---|---|
| 1 | Sanal tablo clamp | Kodda **gerçek hata**: `startIndex > endIndex` olabiliyor, tablo tamamen boş kalıyor | Faz 1 (P0) |
| 2 | ARIA eksik | Doğru; `aria-rowcount` / `aria-rowindex` yok | Faz 2 |
| 3 | Flow-head TTL 3 s | Doğru ama gerekçe abartılı: cron dakikada bir çalışıyor, 3 s TTL tek istemcide %0 fayda sağlıyor | Faz 3 |
| 4 | Tatil takvimi | Hafta sonu var, resmî tatiller yok | Faz 4 |
| 5 | Snapshot TTL "%80" | "%80" iddiası **desteksiz**; gerçek sorun çok süreçli yazmada 30 dk bayatlama | Faz 5 |
| 6 | `_MEMORY_CACHE` sınırı | Doğru; ayrıca overlay TTL 60→900 olunca baskı arttı | Faz 6 |
| 7 | orjson "100 ms" | Ölçüm ~4–30 ms; ayrıca `render(None)` → `b""` sessiz sözleşme farkı | Faz 7 |
| — | — | Shutdown sırasında executor client'lardan sonra kapanıyor | Faz 8 (opsiyonel) |

---

## 4. Genel bakış

| Faz | Öncelik | Bağımlılık | Dokunduğu dosyalar | Commit |
|---|---|---|---|---|
| 1 — Sanal tablo clamp | **P0** | — | `useTableVirtualizer.ts`, `useTableVirtualizer.test.ts`, `FundsPage.tsx` | `fix(ui): clamp virtual table range when rows shrink` |
| 2 — ARIA | P2 | **01** | `FundsPage.tsx`, `MarketsView.tsx` | `feat(ui): add aria row metadata to virtualized tables` |
| 3 — Flow-head cache | P1 | — | `worker/api-cache.ts`, `workerApiCache.test.ts` | `fix(edge): align flow-head cache ttl with poll cadence` |
| 4 — Tatil takvimi | P2 | — | `marketHours.ts`, `marketHours.test.ts` | `fix(ui): skip bist polling on full-day holidays` |
| 5 — Snapshot TTL | P1 | — | `financial_snapshot_store.py`, test, `README.md` | `chore(config): make snapshot cache ttl configurable` |
| 6 — `_MEMORY_CACHE` | P1 | — | `fund_service.py`, `test_fund_service.py` | `fix(funds): bound process-local fund cache entries` |
| 7 — orjson | P3 | — | `etag.py`, `test_etag.py`, `api.py` | `fix(etag): serialize none content as json null` |
| 8 — Shutdown | P3 | **07** | `kap_api_service.py`, `api.py` | `refactor(api): stop refresh executor before closing clients` |

### Bağımlılık grafiği

```
01 ──► 02

03 ─┐
04 ─┤
05 ─┼──► 09 (doğrulama) ──► commit planı
06 ─┤
07 ──► 08 (opsiyonel)
```

- `01` bitmeden `02` başlamaz: ikisi de `FundsPage.tsx`'e dokunuyor ve `02`, `startIndex`'e bağlı.
- `07` ve `08` ikisi de `app/api.py`'ye dokunduğu için sıralı.
- `03`, `04`, `05`, `06` birbirinden bağımsız; farklı dosyalar. Paralel verilebilir.

### Çalıştırma partileri

| Parti | Fazlar | Not |
|---|---|---|
| A | `01` | Tek başına. En kritik iş. |
| B | `02` | `01` commit'lendikten sonra. |
| C | `03` + `04` + `05` + `06` | Dört ayrı oturumla paralel çalışabilir. |
| D | `07` → `08` | Sıralı. `08` opsiyonel. |
| E | `09` | Hepsi bittikten sonra tek oturum. |

---

# Faz 1 — Sanal tablo aralığı hatası (P0)

**Öncelik:** P0 (en kritik)
**Bağımlılık:** yok
**Sonraki faz:** Faz 2 (bu faz bitmeden başlamaz)
**Commit:** `fix(ui): clamp virtual table range when rows shrink`

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

## Definition of Done

- [ ] `computeRange` yeni kodla değiştirildi
- [ ] İki yeni test eklendi
- [ ] Düzeltme geri alındığında yeni testin kırmızı olduğu kanıtlandı
- [ ] `FundsTable` varsayılan resetDeps'ten `rows.length` çıkarıldı
- [ ] `tableResetDeps`'e `deferredSearchTerm` eklendi
- [ ] `npm test -- src/hooks/useTableVirtualizer.test.ts` yeşil
- [ ] Manuel QA yapıldı
- [ ] Tek commit atıldı

---

# Faz 2 — ARIA satır bilgisi

**Öncelik:** P2
**Bağımlılık:** **Faz 1 bitmiş olmalı** (`startIndex` düzeltmesi ve `FundsPage` değişikliği)
**Commit:** `feat(ui): add aria row metadata to virtualized tables`

## Bağlam

Sanallaştırma nedeniyle görünmeyen satırlar DOM'dan çıkıyor. Bu yüzden:

- Ekran okuyucu toplam satır sayısını bilmiyor.
- Tarayıcı içi arama (Ctrl+F) ve "sayfada bul" sadece görünür satırları buluyor.

Çözüm: tabloya `aria-rowcount`, her gerçek veri satırına `aria-rowindex` eklemek.
Spacer (`stocks-spacer-row` / `funds-spacer-row`) satırları `aria-hidden="true"` olarak kalacak.

**ARIA kuralı:** Başlık satırı 1. satırdır. İlk veri satırı 2'dir. `aria-rowcount` = veri satırı
sayısı + 1.

**Kapsam dışı:** Satır yükseklikleri, CSS, edge cache, backend.

Dosyalar:

- `frontend/src/pages/FundsPage.tsx`
- `frontend/src/pages/MarketsView.tsx`

## Adım 2.1 — FundsPage

### Tablo etiketi

`FundsTable` içindeki `<table className="funds-table">` (satır ~1055):

```tsx
            <table className="funds-table" aria-rowcount={rows.length + 1}>
```

### `FundTableRow` prop'u

Bileşen imzası (satır ~950):

```tsx
const FundTableRow = memo(function FundTableRow({
    row,
    rowIndex,
    onOpenFund,
}: {
    row: FundSummary;
    rowIndex: number;
    onOpenFund: (fundCode: string, tab?: FundTab) => void;
}) {
```

`<tr` etiketinin en başına (satır ~962):

```tsx
        <tr
            aria-rowindex={rowIndex}
            onClick={(event: ReactMouseEvent<HTMLTableRowElement>) => {
```

### Render döngüsü

Satır ~1074:

```tsx
                    {visibleRows.map((row, index) => (
                        <FundTableRow
                            key={row.fund_code}
                            row={row}
                            rowIndex={startIndex + index + 2}
                            onOpenFund={onOpenFund}
                        />
                    ))}
```

## Adım 2.2 — MarketsView hisse tablosu

### Tablo etiketi

Satır ~4815:

```tsx
                                        <table
                                            className={`stocks-table${stockTableMode === 'fundamentals' ? ' stocks-fundamental-table' : ''}`}
                                            aria-rowcount={sortedStocks.length + 1}
                                        >
```

### `StockTableRow`

Bileşen `<tr>` etiketi (satır ~1301). Yeni prop gerekmiyor; `rank` zaten
`stockStartIndex + index + 1` olarak geçiliyor.

```tsx
        <tr
            aria-rowindex={rank + 1}
            className={flashClass}
```

`+ 1` açıklaması: `rank` 1 tabanlı veri sırası; başlık satırı nedeniyle ARIA indeksi bir fazla.

## Adım 2.3 — MarketsView endeks detay tablosu

### Tablo etiketi

Satır ~5206:

```tsx
                                    <table
                                        className="stocks-table indices-constituent-table"
                                        aria-rowcount={sortedIndexConstituents.length + 1}
                                    >
```

### Render döngüsü

Satır ~5250:

```tsx
                                            {visibleIndexConstituents.map((row, index) => (
                                                <tr
                                                    key={row.symbol}
                                                    aria-rowindex={indexConstituentStartIndex + index + 2}
```

## Doğrulama

```powershell
cd frontend
npm run build
cd ..
```

Manuel (Chrome DevTools → Elements):

- `/fonlar` tablosunda `<table aria-rowcount="...">` görünmeli.
- Görünür `<tr>`'lerde `aria-rowindex` artan sırada olmalı.
- Spacer satırlarında `aria-hidden="true"` korunmalı, `aria-rowindex` **olmamalı**.
- `/piyasalar` hisse tablosu ve endeks detay tablosunda aynı kontrol.

Ekran okuyucu testi zorunlu değil.

## Definition of Done

- [ ] FundsPage tablosuna `aria-rowcount` eklendi
- [ ] `FundTableRow` `rowIndex` prop'u aldı ve `aria-rowindex` yazıyor
- [ ] FundsPage render döngüsü `startIndex + index + 2` geçiyor
- [ ] Hisse tablosuna `aria-rowcount` eklendi, `StockTableRow` `rank + 1` yazıyor
- [ ] Endeks detay tablosuna `aria-rowcount` eklendi, satırlar `indexConstituentStartIndex + index + 2` yazıyor
- [ ] Spacer satırları değişmedi
- [ ] `npm run build` yeşil
- [ ] Tek commit atıldı

---

# Faz 3 — Flow-head edge cache tutarlılığı

**Öncelik:** P1
**Bağımlılık:** yok
**Commit:** `fix(edge): align flow-head cache ttl with poll cadence`

## Bağlam

Üç ayrı sorun var:

1. **TTL / poll uyumsuzluğu:** Edge TTL 3 s, istemci poll aralığı 5 s. Tek istemci için her istek
   cache'i ıskalıyor → sağlanan fayda **%0**.
2. **`refresh_status` kontrolü yok:** `isCacheableListResponse`, flow-head için boş/hata durumunu
   da cache'lenebilir kabul ediyor.
3. **304 belirsizliği:** Gövdesiz 304 yanıtı cache'lenebilir sayılıyor; Workers Cache'in bunu nasıl
   sakladığı test edilmemiş.

**Neden 10 s TTL güvenli:** KAP flow verisi Worker cron ile **dakikada bir** yenileniyor
(`wrangler.production.toml` → `crons = ["* * * * *", ...]`). Yani `latest_cursor` en fazla 60 s'de
bir değişir. 5 s'lik poll zaten verinin değişme hızından 12 kat hızlı. 10 s'lik cache, veri tazeliği
açısından ölçülebilir kayıp yaratmaz.

**Kapsam dışı:** Backend `kap_flow_store`, frontend poll aralığı, `wrangler.*.toml` ayarları.

Dosyalar:

- `worker/api-cache.ts`
- `frontend/src/workerApiCache.test.ts`

## Adım 3.1 — TTL'i poll aralığıyla uyumlu yap

Dosya: `worker/api-cache.ts`, satır 205-207.

Mevcut:

```ts
    if (route === "market-flow-head") {
        return "public, max-age=3, stale-while-revalidate=15, stale-if-error=0";
    }
```

Yeni:

```ts
    if (route === "market-flow-head") {
        // The KAP flow is refreshed by cron once per minute, and the panel
        // polls every 5s. A 10s edge TTL is the smallest value that can serve
        // at least every other poll for a single client.
        return "public, max-age=10, stale-while-revalidate=10, stale-if-error=0";
    }
```

Beklenen etki: tek istemci ~%50, 4+ eşzamanlı istemci ~%87 origin azalması. En kötü durumda yeni
KAP satırı 20 s gecikir; bu, 60 s'lik ingest kadansının altında kalır.

> **Ürün kararı:** "5 s'den fazla gecikme olmasın" deniyorsa alternatif, flow-head'i cache'ten
> tamamen çıkarmaktır (`return "no-store";` + `getCacheableListRoute`'tan kaldırma). Varsayılan
> olarak 10 s uygulanır. Bu alternatifi ancak kullanıcı açıkça isterse yap.

## Adım 3.2 — `refresh_status` ve cursor kontrolü

Dosya: `worker/api-cache.ts`, `isCacheableListResponse` içinde.

Mevcut satır 248 civarı:

```ts
    if (route === "funds" && status && status !== "ok") return false;
```

Bunun **hemen altına** ekle:

```ts
    if (route === "market-flow-head") {
        const refreshStatus = String(payload.refresh_status || "").trim().toLowerCase();
        if (["empty", "error", "failed", "unavailable"].includes(refreshStatus)) return false;
        if (!payload.latest_cursor) return false;
    }
```

`stale` bilinçli olarak serbest: son geçerli cursor hâlâ kullanılabilir.

## Adım 3.3 — 304 yanıtını olduğu gibi geçir

### Kaldır

`isCacheableListResponse` başındaki şu satırı **sil** (satır 230):

```ts
    if (response.status === 304) return true;
```

### Ekle

`applyListCachePolicy` fonksiyonunun en başına (satır ~260):

```ts
export async function applyListCachePolicy(
    response: Response,
    request: Request,
    route: CacheableListRoute,
    now: Date = new Date(),
): Promise<Response> {
    // A 304 has no body; caching it as a standalone entry is not meaningful and
    // would make the edge serve an empty response without revalidation.
    if (response.status === 304) return response;

    const cacheable = await isCacheableListResponse(response, route);
```

## Adım 3.4 — Testler

Dosya: `frontend/src/workerApiCache.test.ts`

### Mevcut assertion'ı güncelle (satır ~192-193)

```ts
        expect(flowResponse.headers.get('Cloudflare-CDN-Cache-Control'))
            .toBe('public, max-age=10, stale-while-revalidate=10, stale-if-error=0');
```

### Yeni testler

```ts
    it('does not cache an empty market flow head', async () => {
        const response = await applyListCachePolicy(
            new Response(JSON.stringify({ latest_cursor: null, refresh_status: 'empty' }), {
                headers: { 'Content-Type': 'application/json' },
            }),
            makeRequest('/api/market/flow/head'),
            'market-flow-head',
        );
        expect(response.headers.get('Cache-Control')).toBe('private, no-store');
        expect(response.headers.get('Cloudflare-CDN-Cache-Control')).toBeNull();
    });

    it('passes 304 responses through without edge cache headers', async () => {
        const response = await applyListCachePolicy(
            new Response(null, { status: 304, headers: { ETag: 'W/"abc"' } }),
            makeRequest('/api/funds'),
            'funds',
        );
        expect(response.status).toBe(304);
        expect(response.headers.get('Cloudflare-CDN-Cache-Control')).toBeNull();
        expect(response.headers.get('Cache-Tag')).toBeNull();
    });
```

## Doğrulama

```powershell
cd frontend
npm test -- src/workerApiCache.test.ts
npm run build
cd ..
```

Deploy sonrası manuel (deploy bu fazın kapsamı **değil**, sadece bilgi):

```powershell
curl.exe -sI "https://<worker-host>/api/market/flow/head" | Select-String -Pattern "cache-control|cloudflare-cdn-cache-control|cache-tag"
```

Beklenen: `Cloudflare-CDN-Cache-Control: public, max-age=10, stale-while-revalidate=10, stale-if-error=0`
ve `Cache-Tag: market-flow-head`.

## Definition of Done

- [ ] `cacheControlForListRequest` flow-head değeri 10/10/0 oldu
- [ ] `isCacheableListResponse` flow-head için `refresh_status` ve `latest_cursor` kontrol ediyor
- [ ] 304 için `return true` kaldırıldı
- [ ] `applyListCachePolicy` 304'ü değiştirmeden döndürüyor
- [ ] Mevcut test assertion'ı güncellendi
- [ ] İki yeni test eklendi ve geçiyor
- [ ] `npm run build` yeşil
- [ ] Tek commit atıldı

---

# Faz 4 — BIST tatil takvimi

**Öncelik:** P2
**Bağımlılık:** yok
**Commit:** `fix(ui): skip bist polling on full-day holidays`

## Bağlam

`isBistTradingHours` sadece hafta sonu ve saat kontrolü yapıyor. Resmî tatillerde panel 5 s'de bir
poll etmeye devam ediyor.

Bu fonksiyon **iki yerde** kullanılıyor:

1. `frontend/src/components/MarketFlowPanel.tsx` → flow-head poll aralığı (5 s / 60 s)
2. `worker/api-cache.ts` → `cacheControlForListRequest` içinde market-stocks TTL'i

İkinci kullanım için not: tatilde `isBistTradingHours` false dönecek ve market-stocks cache'i
300 s'ye çıkacak. Bu **istenen** davranış. `secondsUntilNextBistSession` tatilleri bilmiyor ama
`MAX_MARKET_CACHE_AGE_SECONDS = 300` üst sınırı sayesinde bayat veri riski yok. Bu davranışı
değiştirme.

**Kapsam dışı:** Yarım günler, backend `_TURKEY_MARKET_FULL_DAY_HOLIDAYS`, edge TTL formülü.

Dosyalar:

- `frontend/src/utils/marketHours.ts`
- `frontend/src/utils/marketHours.test.ts` (yeni)

## Adım 4.1 — Saat parçalarına tarih ekle

Dosya: `frontend/src/utils/marketHours.ts`

### Formatter

Mevcut:

```ts
const istanbulPartsFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: ISTANBUL_TIMEZONE,
    weekday: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
});
```

Yeni:

```ts
const istanbulPartsFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: ISTANBUL_TIMEZONE,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    weekday: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
});
```

### Interface

```ts
export interface IstanbulClockParts {
    year: number;
    month: number;
    day: number;
    weekday: string;
    hour: number;
    minute: number;
    minutesOfDay: number;
    /** YYYY-MM-DD in Europe/Istanbul. */
    isoDate: string;
}
```

### `getIstanbulClockParts`

```ts
export function getIstanbulClockParts(now: Date = new Date()): IstanbulClockParts {
    const parts = istanbulPartsFormatter.formatToParts(now);
    let weekday = 'Mon';
    let hour = 0;
    let minute = 0;
    let year = 1970;
    let month = 1;
    let day = 1;

    for (const part of parts) {
        if (part.type === 'weekday') weekday = part.value;
        else if (part.type === 'year') year = Number(part.value);
        else if (part.type === 'month') month = Number(part.value);
        else if (part.type === 'day') day = Number(part.value);
        else if (part.type === 'hour') hour = Number(part.value) % 24;
        else if (part.type === 'minute') minute = Number(part.value);
    }

    const isoDate = `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;

    return {
        year,
        month,
        day,
        weekday,
        hour,
        minute,
        minutesOfDay: hour * 60 + minute,
        isoDate,
    };
}
```

`getIstanbulClockParts`'ın diğer kullanıcıları (`isGlobalFxTradingHours`, `getAdaptiveRefreshMs`)
ek alanlardan etkilenmez; interface genişlemesi geriye dönük uyumlu.

## Adım 4.2 — Tatil listesi

`marketHours.ts` içine, `istanbulPartsFormatter` tanımından sonra ekle:

```ts
/**
 * BIST full-day closures, Europe/Istanbul calendar dates.
 * Source: Borsa İstanbul official holiday calendar.
 * Keep in sync with app/fund_service.py::_TURKEY_MARKET_FULL_DAY_HOLIDAYS.
 * Half-days are intentionally NOT listed: polling a few extra hours is harmless.
 */
const BIST_FULL_DAY_HOLIDAYS: ReadonlySet<string> = new Set([
    // 2026 — mirrors app/fund_service.py:3776-3786
    '2026-01-01',
    '2026-03-20',
    '2026-04-23',
    '2026-05-01',
    '2026-05-19',
    '2026-05-27',
    '2026-05-28',
    '2026-05-29',
    '2026-07-15',
    '2026-08-30',
    '2026-10-29',
    // TODO(2027): add verified BIST holiday dates before 2027-01-01.
]);
```

### Veri doğrulama kuralı — bunu atlama

2026 tarihleri `app/fund_service.py:3776-3786` ile birebir aynı; **değiştirme**.

**2027 tarihlerini uydurma.** Ajanda 2027 için resmî BIST takvimi (`borsaistanbul.com` → Tatil
Günleri) doğrulanabiliyorsa ekle. Doğrulanamıyorsa listeye **ekleme** ve yukarıdaki
`TODO(2027)` yorumunu bırak. Dini bayram tarihlerini tahminle yazmak, tatil gününde yanlış
davranışa yol açar.

## Adım 4.3 — `isBistTradingHours` güncelle

Mevcut:

```ts
export function isBistTradingHours(now: Date = new Date()): boolean {
    const { weekday, minutesOfDay } = getIstanbulClockParts(now);
    if (weekday === 'Sat' || weekday === 'Sun') {
        return false;
    }
    const sessionStart = 9 * 60 + 50; // 09:50
    const sessionEnd = 18 * 60 + 20;  // 18:20
    return minutesOfDay >= sessionStart && minutesOfDay <= sessionEnd;
}
```

Yeni:

```ts
export function isBistTradingHours(now: Date = new Date()): boolean {
    const { weekday, minutesOfDay, isoDate } = getIstanbulClockParts(now);
    if (weekday === 'Sat' || weekday === 'Sun') {
        return false;
    }
    if (BIST_FULL_DAY_HOLIDAYS.has(isoDate)) {
        return false;
    }
    const sessionStart = 9 * 60 + 50; // 09:50
    const sessionEnd = 18 * 60 + 20;  // 18:20
    return minutesOfDay >= sessionStart && minutesOfDay <= sessionEnd;
}
```

## Adım 4.4 — Testler

Yeni dosya: `frontend/src/utils/marketHours.test.ts`

```ts
import { describe, expect, it } from 'vitest';
import { isBistTradingHours } from './marketHours';

describe('isBistTradingHours', () => {
    it('is open on a regular weekday during the session', () => {
        // Wednesday 2026-11-04 12:00 Istanbul (09:00 UTC)
        expect(isBistTradingHours(new Date('2026-11-04T09:00:00.000Z'))).toBe(true);
    });

    it('is closed on weekends', () => {
        expect(isBistTradingHours(new Date('2026-10-25T11:00:00.000Z'))).toBe(false);
    });

    it('is closed outside session hours', () => {
        expect(isBistTradingHours(new Date('2026-11-04T19:00:00.000Z'))).toBe(false);
    });

    it('is closed on a full-day BIST holiday', () => {
        // 2026-10-29 Republic Day, 12:00 Istanbul
        expect(isBistTradingHours(new Date('2026-10-29T09:00:00.000Z'))).toBe(false);
    });
});
```

## Doğrulama

```powershell
cd frontend
npm test -- src/utils/marketHours.test.ts src/components/MarketFlowPanel.test.ts src/workerApiCache.test.ts
npm run build
cd ..
```

`workerApiCache.test.ts` de çalıştırılıyor çünkü Worker aynı fonksiyonu import ediyor; mevcut testler
bozulmamalı.

## Definition of Done

- [ ] Formatter'a `year/month/day` eklendi
- [ ] `IstanbulClockParts` `year/month/day/isoDate` içeriyor
- [ ] `getIstanbulClockParts` `isoDate` üretiyor
- [ ] `BIST_FULL_DAY_HOLIDAYS` 2026 tarihleriyle eklendi
- [ ] 2027 için doğrulanmamış tarih **eklenmedi**, TODO yorumu bırakıldı
- [ ] `isBistTradingHours` tatil kontrolü yapıyor
- [ ] Yeni test dosyası eklendi ve geçiyor
- [ ] `MarketFlowPanel` ve `workerApiCache` testleri bozulmadı
- [ ] `npm run build` yeşil
- [ ] Tek commit atıldı

---

# Faz 5 — Snapshot cache TTL'ini yapılandırılabilir yap

**Öncelik:** P1
**Bağımlılık:** yok
**Commit:** `chore(config): make snapshot cache ttl configurable`

## Bağlam

`app/financial_snapshot_store.py` içinde `_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS = 1800.0` sabit.
Sorunlar:

1. Çok süreçli kurulumda (ikinci uvicorn worker, ayrı cron süreci, elle SQL yazımı) invalidation
   diğer süreçlere ulaşmıyor; veri **30 dakikaya kadar** bayat görünebiliyor.
2. Bunu düşürmek için env yok.

`_invalidate_memory_cache` sadece aynı süreçte çalışıyor; bu tasarım değişmiyor. Yapılan iş:
TTL'i env'e taşımak + riski kod içinde belgelemek.

**Kapsam dışı:** Cross-process invalidation mekanizması, Redis pub/sub, DB polling.
Bunları bu fazda **yapma**.

Dosyalar:

- `app/financial_snapshot_store.py`
- `tests/test_financial_snapshot_store.py`
- `README.md`

## Adım 5.1 — Env okuma yardımcısı

Dosya: `app/financial_snapshot_store.py`

`LOGGER = logging.getLogger(__name__)` satırının hemen altına, sabitlerden önce ekle:

```python
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        LOGGER.warning("invalid %s=%r; using default %s", name, raw, default)
        return default
```

## Adım 5.2 — Sabiti değiştir

Mevcut (satır 31):

```python
_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS = 1800.0
```

Yeni:

```python
# Upper bound only. Writes invalidate this cache inside the same process; a
# separate writer process (another uvicorn worker, a cron job, a manual SQL
# update) is NOT invalidated and can stay stale until this TTL expires. Keep
# this low (e.g. 60) when multiple processes write to the same database.
_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS = max(
    1.0,
    _env_float("RAGFIN_FINANCIAL_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS", 1800.0),
)
```

`LOGGER` satır 19'da tanımlı; `_env_float` ondan sonra geldiği için sorun yok.

## Adım 5.3 — Test

Dosya: `tests/test_financial_snapshot_store.py`

Mevcut `test_financial_snapshot_memory_cache_lives_1800_seconds_and_expires` testine **dokunma**;
varsayılanın 1800 olduğunu doğrulamaya devam eder.

Dosyanın sonuna ekle:

```python
def test_env_float_parses_snapshot_cache_ttl_override(monkeypatch) -> None:
    monkeypatch.setenv("RAGFIN_FINANCIAL_SNAPSHOT_CACHE_TEST", "90")
    assert store._env_float("RAGFIN_FINANCIAL_SNAPSHOT_CACHE_TEST", 1800.0) == 90.0

    monkeypatch.setenv("RAGFIN_FINANCIAL_SNAPSHOT_CACHE_TEST", "not-a-number")
    assert store._env_float("RAGFIN_FINANCIAL_SNAPSHOT_CACHE_TEST", 1800.0) == 1800.0
```

> Not: Test dosyasında `from app import financial_snapshot_store as store` import'u mevcut.
> Yeni import stili icat etme.

## Adım 5.4 — README

Dosya: `README.md`, "Önemli değişkenler" listesi (satır ~149 civarı).

Şu satırı ekle:

```markdown
- `RAGFIN_FINANCIAL_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS` (varsayılan 1800; çok süreçli yazma varsa düşürün)
```

## Doğrulama

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_financial_snapshot_store.py -q
```

Beklenen: mevcut testler + yeni test geçer (toplam 14 civarı).

Manuel env kontrolü (opsiyonel):

```powershell
$env:RAGFIN_FINANCIAL_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS="60"
.external\venv311\Scripts\python.exe -c "from app import financial_snapshot_store as s; print(s._SNAPSHOT_MEMORY_CACHE_TTL_SECONDS)"
Remove-Item Env:RAGFIN_FINANCIAL_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS
```

Beklenen: `60.0`.

## Definition of Done

- [ ] `_env_float` eklendi ve hatalı değerde uyarı verip varsayılana dönüyor
- [ ] TTL sabiti env'den okunuyor, minimum 1.0
- [ ] Çok süreçli bayatlama riski yorumda belgelendi
- [ ] Yeni test eklendi, mevcut 1800 testi değişmedi
- [ ] README'ye env değişkeni eklendi
- [ ] Hedefli pytest yeşil
- [ ] Tek commit atıldı

---

# Faz 6 — `_MEMORY_CACHE` sınırı

**Öncelik:** P1
**Bağımlılık:** yok
**Commit:** `fix(funds): bound process-local fund cache entries`

## Bağlam

`app/fund_service.py` içindeki `_MEMORY_CACHE` düz bir `dict` ve **sınırsız**. Anahtar yapıları:

- `latest-fund-price-points:{db_path}:{end_date}:{code1,code2,...}` → her farklı arama/filtre
  kombinasyonu yeni girdi
- snapshot dosya cache anahtarları

Overlay TTL'i 60 → 900'e çıkarıldığı için girdilerin ortalama ömrü arttı; bellek baskısı da arttı.
Çözüm: FIFO tabanlı üst sınır.

**Kapsam dışı ve kritik:** `_MEMORY_CACHE.clear()` çağrılarının semantiğini **değiştirme**.
Onlar fiyat/snapshot yazımında tüm cache'i geçersiz kılıyor; dokunursan bayat fiyat okuması olur.

Dosyalar:

- `app/fund_service.py`
- `tests/test_fund_service.py`

## Adım 6.1 — Sabit ve yardımcı fonksiyon

Dosya: `app/fund_service.py`

Mevcut (satır ~210):

```python
_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}
```

Bunun **hemen altına** ekle:

```python
try:
    _MEMORY_CACHE_MAX_ENTRIES = max(
        16,
        int(os.getenv("RAGFIN_FUNDS_MEMORY_CACHE_MAX_ENTRIES", "256")),
    )
except ValueError:
    _MEMORY_CACHE_MAX_ENTRIES = 256


def _memory_cache_put(cache_key: str, value: Dict[str, Any]) -> None:
    """Insert into the process-local cache, evicting oldest entries first.

    This is FIFO, not LRU: reads do not refresh insertion order. 256 entries is
    far above the number of distinct fund list/detail views a single process
    serves between deploys, so eviction should stay rare.
    """
    _MEMORY_CACHE.pop(cache_key, None)
    _MEMORY_CACHE[cache_key] = value
    overflow = len(_MEMORY_CACHE) - _MEMORY_CACHE_MAX_ENTRIES
    if overflow <= 0:
        return
    for stale_key in list(_MEMORY_CACHE.keys())[:overflow]:
        _MEMORY_CACHE.pop(stale_key, None)
```

`os`, `Dict`, `Any` zaten import edilmiş; yeni import gerekmiyor.

## Adım 6.2 — Yazma noktalarını değiştir

Sadece **iki** yer.

### 1) `read_latest_fund_price_points` (satır ~3155)

Mevcut:

```python
    _MEMORY_CACHE[cache_key] = {"cached_at": time.time(), "points": points}
```

Yeni:

```python
    _memory_cache_put(cache_key, {"cached_at": time.time(), "points": points})
```

### 2) Snapshot dosya cache'i (satır ~6531)

Mevcut:

```python
            if stat:
                _MEMORY_CACHE[cache_key] = {
                    "file_signature": file_signature,
                    "mtime": stat.st_mtime,
                    "payload": payload,
                }
```

Yeni:

```python
            if stat:
                _memory_cache_put(cache_key, {
                    "file_signature": file_signature,
                    "mtime": stat.st_mtime,
                    "payload": payload,
                })
```

Başka `_MEMORY_CACHE[...] = ...` ataması **yok**; varsa `grep` ile doğrula ve aynı helper'a çevir.
Okuma (`_MEMORY_CACHE.get`) ve `clear()` çağrılarına dokunma.

## Adım 6.3 — Test

Dosya: `tests/test_fund_service.py`

```python
def test_memory_cache_evicts_oldest_entries(monkeypatch) -> None:
    from app import fund_service

    fund_service.reset_fund_caches_for_tests()
    monkeypatch.setattr(fund_service, "_MEMORY_CACHE_MAX_ENTRIES", 4)

    for index in range(6):
        fund_service._memory_cache_put(f"key-{index}", {"value": index})

    assert len(fund_service._MEMORY_CACHE) == 4
    assert "key-0" not in fund_service._MEMORY_CACHE
    assert "key-1" not in fund_service._MEMORY_CACHE
    assert "key-5" in fund_service._MEMORY_CACHE

    fund_service.reset_fund_caches_for_tests()
```

`reset_fund_caches_for_tests` mevcut (`fund_service.py:434`); yeni fixture yazma.

## Adım 6.4 — Opsiyonel: overlay TTL testi

Bu adım **zorunlu değil**. Zaman varsa yap; yoksa atla ve raporla.

Yeni dosya: `tests/test_fund_overlay_cache.py`

Amaç: `read_latest_fund_price_points` TTL içinde ikinci kez DB'ye gitmiyor, TTL sonrası gidiyor.

İskelet:

```python
from __future__ import annotations

from pathlib import Path

from app import fund_service


def test_latest_price_overlay_cache_respects_ttl(tmp_path: Path, monkeypatch) -> None:
    fund_service.reset_fund_caches_for_tests()
    processed_dir = tmp_path
    fund_service.ensure_fund_prices_schema(processed_dir)

    # TODO: insert one fund_prices row using the same helper pattern as the
    # existing tests in tests/test_fund_service.py. Then monkeypatch
    # fund_service.time.time to a fake clock, count calls to
    # fund_service._connect_fund_prices_db, and assert:
    #   call 1 -> DB read
    #   call 2 within 900s -> no DB read
    #   call 3 after 901s -> DB read
    #
    # If wiring this takes more than ~20 lines, skip the test entirely and say
    # so in the final report. Do not invent a new DB fixture.
```

## Doğrulama

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_fund_service.py -q -k "memory_cache"
```

Beklenen: yeni test geçer.

Regresyon kontrolü (helper'a çevirdiğin iki yol için):

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_fund_service.py -q -k "snapshot or overlay or latest"
```

Beklenen: mevcut testler bozulmaz. Çok uzun sürerse `-k` filtresini daralt; tüm dosyayı çalıştırmak
zorunda değilsin.

## Definition of Done

- [ ] `_MEMORY_CACHE_MAX_ENTRIES` env'den okunuyor, hatalı değerde 256'ya düşüyor
- [ ] `_memory_cache_put` FIFO eviction yapıyor
- [ ] İki yazma noktası helper'a çevrildi
- [ ] Başka doğrudan atama kalmadı (grep ile doğrulandı)
- [ ] `_MEMORY_CACHE.clear()` çağrıları değişmedi
- [ ] Eviction testi eklendi ve geçiyor
- [ ] Opsiyonel overlay testi yapıldı veya atlandığı raporlandı
- [ ] Hedefli pytest yeşil
- [ ] Tek commit atıldı

---

# Faz 7 — orjson `None` tutarlılığı ve ölçüm notu

**Öncelik:** P3 (düşük risk)
**Bağımlılık:** yok
**Sonraki faz:** Faz 8 (ikisi de `app/api.py`)
**Commit:** `fix(etag): serialize none content as json null`

## Bağlam

### Sorun 1: `None` gövdesi

`ORJSONResponse(content=None)` şu an gövdeyi `b""` döndürüyor. FastAPI'nin standart `JSONResponse`'ı
`b"null"` döndürür. Repoda `None` dönen route yok, yani pratik regresyon yok — ama sessiz bir
sözleşme farkı.

Mevcut test bu davranışı açıkça doğruluyor (`tests/test_etag.py:386-387`), yani değişiklik
**bilinçli** bir karar.

### Sorun 2: yanlış performans iddiası

"orjson 100 ms kazandırıyor" iddiası ölçümle desteklenmiyor. 1.500 satırlık payload ile ölçüm:

| Yol | Süre |
|---|---|
| stdlib `json.dumps` | 4,4 ms |
| `jsonable_encoder` + `json.dumps` | 30,3 ms |
| orjson | 0,47 ms |

Kazanç ~4 ms (sadece serializer) veya ~30 ms (`jsonable_encoder` atlanıyorsa). FastAPI normal
route'larda `jsonable_encoder`'ı hâlâ çalıştırıyor, yani gerçek kazanç ~4 ms mertebesinde.
Bu fazda kod optimize edilmiyor; sadece yanlış iddia kod yorumunda düzeltiliyor.

**Kapsam dışı:** `orjson_default`'a yeni tip ekleme, `jsonable_encoder`'ı bypass etme denemesi.

Dosyalar:

- `app/etag.py`
- `tests/test_etag.py`
- `app/api.py` (sadece yorum)

## Adım 7.1 — `render(None)` davranışı

Dosya: `app/etag.py`, satır 127-132.

Mevcut:

```python
    def render(self, content: Any) -> bytes:
        if content is None:
            return b""
        if isinstance(content, (bytes, memoryview)):
            return bytes(content)
        return orjson_dumps(content)
```

Yeni:

```python
    def render(self, content: Any) -> bytes:
        if content is None:
            return b"null"
        if isinstance(content, (bytes, memoryview)):
            return bytes(content)
        return orjson_dumps(content)
```

## Adım 7.2 — Testi güncelle

Dosya: `tests/test_etag.py`, satır 386-387.

Mevcut:

```python
    empty_resp = ORJSONResponse(content=None)
    assert empty_resp.body == b""
```

Yeni:

```python
    null_resp = ORJSONResponse(content=None)
    assert null_resp.body == b"null"
```

**Dikkat:** `tests/test_etag.py:406` civarındaki `not_modified.body == b""` assertion'ı 304 içindir;
ona **dokunma**. 304 gövdesiz olmalı.

## Adım 7.3 — `app/api.py`'ye ölçüm notu

Dosya: `app/api.py`, satır 107-112.

Mevcut:

```python
app = FastAPI(
    title="RAG-Fin API",
    version="0.10.0",
    lifespan=_lifespan,
    default_response_class=ORJSONResponse,
)
```

Yeni:

```python
# orjson only accelerates the final serialization step; FastAPI still runs
# jsonable_encoder for normal routes. Measured saving on a 1,500-row payload is
# ~4 ms, or ~30 ms on the conditional_json_response path which skips the encoder.
app = FastAPI(
    title="RAG-Fin API",
    version="0.10.0",
    lifespan=_lifespan,
    default_response_class=ORJSONResponse,
)
```

"100 ms" gibi bir iddiayı kod yorumuna veya commit mesajına yazma.

## Doğrulama

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_etag.py -q
```

Beklenen: tüm testler geçer (13 civarı).

Hızlı manuel kontrol:

```powershell
.external\venv311\Scripts\python.exe -c "from app.etag import ORJSONResponse; print(ORJSONResponse(content=None).body)"
```

Beklenen: `b'null'`.

## Definition of Done

- [ ] `ORJSONResponse.render(None)` `b"null"` döndürüyor
- [ ] `tests/test_etag.py` içindeki `None` assertion'ı güncellendi
- [ ] 304 testi (`not_modified.body == b""`) değişmedi
- [ ] `app/api.py`'ye ölçüm yorumu eklendi
- [ ] `tests/test_etag.py` yeşil
- [ ] Tek commit atıldı

> **Ürün kararı:** `b""` davranışı bilinçli olarak isteniyorsa bu fazı atla. O durumda
> `render`'a şu yorumu ekle ve commit atma:
> `# Intentionally empty: no route returns None today. JSONResponse would return b"null".`

---

# Faz 8 — Shutdown sırasında client kapatma (OPSİYONEL)

**Öncelik:** P3
**Bağımlılık:** Faz 7 (ikisi de `app/api.py`)
**Commit:** `refactor(api): stop refresh executor before closing clients`

> **Bu faz opsiyonel.** Zaman/risk kısıtı varsa atla ve raporla. Doğruluk hatası değil; kapanış
> sırasında görülebilecek hata logu riski.

## Bağlam

`app/api.py` lifespan `finally` bloğu:

```python
    finally:
        await _stop_fund_price_collector()
        close_postgres_pool()
        close_shared_http_clients()
```

`_stop_fund_price_collector()` sadece fon fiyat collector'ını durduruyor. KAP flow refresh
executor'ı (`_KAP_FLOW_REFRESH_EXECUTOR`) ve statement refresh job'ları hâlâ çalışıyor olabilir.
Paylaşılan httpx client'ları onların altından kapatılırsa `RuntimeError: Cannot send a request, as
the client has been closed` logları görülebilir.

**Kapsam dışı:** `_kap_post_json` retry davranışı, thread pool boyutu, yeni job iptal mekanizması.

Dosyalar:

- `app/kap_api_service.py`
- `app/api.py`

## Adım 8.1 — Executor shutdown fonksiyonu

Dosya: `app/kap_api_service.py`

`_KAP_FLOW_REFRESH_EXECUTOR` tanımının (satır ~40) altına veya modülün uygun bir yerine ekle:

```python
def shutdown_background_refresh_executor() -> None:
    """Stop accepting new KAP refresh jobs during application shutdown."""
    _KAP_FLOW_REFRESH_EXECUTOR.shutdown(wait=False, cancel_futures=True)
```

`cancel_futures=True` Python 3.9+ gerektirir; proje 3.11 kullanıyor, sorun yok.

## Adım 8.2 — Lifespan'e ekle

Dosya: `app/api.py`

Import ekle:

```python
from app.kap_api_service import shutdown_background_refresh_executor
```

> Döngüsel import hatası alırsan import'u `_lifespan` fonksiyonunun içine taşı:
> ```python
> from app.kap_api_service import shutdown_background_refresh_executor
> ```
> Import'u başka bir modüle taşıma; fonksiyon `kap_api_service`'te kalacak.

`finally` bloğunu şu hale getir:

```python
    finally:
        await _stop_fund_price_collector()
        shutdown_background_refresh_executor()
        close_postgres_pool()
        close_shared_http_clients()
```

Sıra önemli: executor, client'lardan **önce** durdurulmalı.

## Doğrulama

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_http_client_pooling.py -q
```

Beklenen: mevcut lifespan ve client pooling testleri bozulmadan geçer.

Manuel (opsiyonel): uygulamayı başlat, bir KAP flow yenilemesi tetikle, hemen Ctrl+C ile kapat.
Loglarda "client has been closed" hatası görünmemeli.

## Definition of Done

- [ ] `shutdown_background_refresh_executor` eklendi
- [ ] Lifespan `finally` bloğunda client kapatmadan önce çağrılıyor
- [ ] Döngüsel import yok (veya fonksiyon içi import ile çözüldü)
- [ ] `tests/test_http_client_pooling.py` yeşil
- [ ] Tek commit atıldı

---

# Faz 9 — Nihai doğrulama

**Öncelik:** Zorunlu (son adım)
**Bağımlılık:** Uygulanan tüm fazlar bitmiş olmalı
**Commit:** yok (doğrulama fazı)

Bu faz kod değiştirmez. Sadece kanıt toplar. Bir şey kırmızıysa **commit atma**, ilgili faza dön.

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

## 2. Backend testleri

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_etag.py tests/test_financial_snapshot_store.py tests/test_http_client_pooling.py -q
```

Beklenen: tümü geçer. Hata varsa çıktıyı **olduğu gibi** rapora koy.

Faz 6'da eklenen test:

```powershell
.external\venv311\Scripts\python.exe -m pytest tests/test_fund_service.py -q -k "memory_cache"
```

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

## 4. Regresyon kanıtı (Faz 1)

Faz 1'in testinin gerçekten hatayı yakaladığını bir kez daha doğrula:

1. `computeRange`'i geçici olarak eski haline getir.
2. `npm test -- src/hooks/useTableVirtualizer.test.ts` çalıştır → yeni test **kırmızı** olmalı.
3. Düzeltmeyi geri koy → test **yeşil** olmalı.

Bu adımın çıktısını rapora ekle. "Test geçti" demek yetmez; testin hatayı yakaladığını kanıtla.

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

## Definition of Done

- [ ] `git diff --check` temiz
- [ ] Backend hedefli testler yeşil
- [ ] Frontend hedefli testler yeşil
- [ ] `npm run build` yeşil
- [ ] Faz 1 regresyon kanıtı rapora eklendi
- [ ] Manuel QA tablosu dolduruldu
- [ ] Faz raporu yukarıdaki formatta yazıldı

---

# Commit planı ve geri alma

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
- Commit mesajında gövde kullanacaksan 72 karakterden sonra satır kır.

Örnek:

```powershell
git add frontend/src/hooks/useTableVirtualizer.ts frontend/src/hooks/useTableVirtualizer.test.ts frontend/src/pages/FundsPage.tsx
git commit -m "fix(ui): clamp virtual table range when rows shrink"
```

## Aynı dosyaya dokunan fazlar

- **01 + 02:** `FundsPage.tsx` iki fazda da değişiyor. Commit 2, commit 1'den sonra atılmalı.
- **07 + 08:** `api.py` iki fazda da değişiyor. Sıra: 7 → 8.

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

## Commit öncesi son kontrol

```powershell
git diff --check
git status --short
```

- Whitespace hatası yok.
- Plan dışı dosya yok (`rag-fin-baslangic-tam-boy.png` hariç).
- Her commit'in mesajı AGENTS.md formatına uygun.

---

# Ajana verilecek promptlar

## Tek faz verme (önerilen)

```
Sana tek bir faz dosyası veriyorum: docs/perf-fix/<dosya>.md
Alternatif: bu plan dosyasının "<Faz N>" bölümü.

1. Önce ortak kuralları oku: docs/perf-fix/00-kurallar.md
   (veya bu dosyanın "2. Değişmez kurallar" bölümü)
2. Sadece bu fazdaki adımları uygula. Sonraki fazlara geçme, kapsam genişletme.
3. Dosyadaki kod bloklarını birebir uygula; "daha iyi yol" gerekçesiyle sapma.
4. Faz sonunda "Doğrulama" komutlarını çalıştır ve gerçek çıktıyı raporla.
5. "Definition of Done" listesini tek tek işaretle.
6. Son olarak fazda yazan commit mesajıyla tek commit at.

Bir şey uyuşmuyorsa dur ve sor; tahminle ilerleme.
```

## Faz 1 için hazır metin

```
Proje dizini: D:\Projeler\rag-fin

Uygula: docs/perf-fix/01-faz-1-virtualizer.md
Önce oku: docs/perf-fix/00-kurallar.md

Bağlam: 8 fazlık performans planının 1. fazı (P0, en kritik).
- Sana bağlı olan: 02 (ARIA) — sen bitmeden başlamaz.
- Bağımsız olanlar: 03, 04, 05, 06
- Sıralı olanlar: 07 → 08

Sadece 01'i bitir. Kod bloklarını birebir uygula.
"Doğrulama" komutlarını çalıştır, gerçek çıktıyı raporla.
"Kanıt zorunluluğu" adımını atlama (düzeltmeyi geri al → test kırmızı → geri koy).
"DoD" listesini tek tek işaretle.
Sonda tek commit at: fix(ui): clamp virtual table range when rows shrink
Bir şey uyuşmuyorsa dur ve sor.
```

## Tüm planı tek oturumda verme

```
Proje dizini: D:\Projeler\rag-fin

Uygula: docs/perf-fix-plan.md — Faz 1'den Faz 8'e kadar sırayla.

Zorunlu kurallar:
- Her faz için AYRI commit at. Fazları tek commit'te birleştirme.
- Her fazın sonunda o fazın "Doğrulama" komutlarını çalıştır. Hepsi bitince değil.
- Her fazın "Definition of Done" listesini ayrı ayrı işaretle.
- Faz 1 bitmeden Faz 2'ye geçme. Faz 7 bitmeden Faz 8'e geçme.
- Faz 8 opsiyonel; zaman/risk varsa atla ve raporla.
- Kapsam genişletme, plan dışı refactor yapma, yeni bağımlılık ekleme.
- "2. Değişmez kurallar" bölümündeki 12 maddeye uy.

Bir şey uyuşmuyorsa dur ve sor; tahminle ilerleme.
```

> Bu mod daha hızlıdır ama risklidir. Faz 1 ve Faz 2'yi yine de ayrı oturumda vermen önerilir:
> ikisi de `FundsPage.tsx`'e dokunuyor ve `02`, `01`'in `startIndex` çıktısına bağlı.

---

## Ek: Faz dosyaları ile ilişki

| Bu dosyadaki bölüm | Ayrı dosya |
|---|---|
| Faz 1 | `docs/perf-fix/01-faz-1-virtualizer.md` |
| Faz 2 | `docs/perf-fix/02-faz-2-aria.md` |
| Faz 3 | `docs/perf-fix/03-faz-3-flow-head-cache.md` |
| Faz 4 | `docs/perf-fix/04-faz-4-tatil-takvimi.md` |
| Faz 5 | `docs/perf-fix/05-faz-5-snapshot-ttl.md` |
| Faz 6 | `docs/perf-fix/06-faz-6-memory-cache.md` |
| Faz 7 | `docs/perf-fix/07-faz-7-orjson.md` |
| Faz 8 | `docs/perf-fix/08-faz-8-shutdown.md` |
| Faz 9 | `docs/perf-fix/09-nihai-dogrulama.md` |
| Commit planı | `docs/perf-fix/10-commit-plani.md` |
| Ortak kurallar | `docs/perf-fix/00-kurallar.md` |

İki sürüm çelişirse **bu dosya** geçerlidir.
