# Faz 2 — Sanal tablolara ARIA satır bilgisi

**Öncelik:** P2
**Bağımlılık:** **Faz 1 bitmiş olmalı** (`startIndex` düzeltmesi ve `FundsPage` değişikliği)
**Commit:** `feat(ui): add aria row metadata to virtualized tables`

---

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

---

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

---

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

---

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

---

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

---

## Definition of Done

- [ ] FundsPage tablosuna `aria-rowcount` eklendi
- [ ] `FundTableRow` `rowIndex` prop'u aldı ve `aria-rowindex` yazıyor
- [ ] FundsPage render döngüsü `startIndex + index + 2` geçiyor
- [ ] Hisse tablosuna `aria-rowcount` eklendi, `StockTableRow` `rank + 1` yazıyor
- [ ] Endeks detay tablosuna `aria-rowcount` eklendi, satırlar `indexConstituentStartIndex + index + 2` yazıyor
- [ ] Spacer satırları değişmedi
- [ ] `npm run build` yeşil
- [ ] Tek commit atıldı

## Commit

```
feat(ui): add aria row metadata to virtualized tables
```

---

## Bu fazı ajana verirken kullan

```
docs/perf-fix/02-faz-2-aria.md dosyasını uygula.
Önce docs/perf-fix/00-kurallar.md oku.
Faz 1'in uygulandığını varsay; startIndex değerini değiştirme.
Sadece aria-rowcount ve aria-rowindex ekle; CSS ve satır yüksekliğine dokunma.
```
