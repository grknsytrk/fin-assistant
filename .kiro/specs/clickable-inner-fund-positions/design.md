# Design Document

## Overview

Bu özellik, bir fonun "Fon İçeriği" panelinde (`FundHoldingsPanel`) ve fon ile ilgili diğer alt fon listelerinde, `Fund_Position_Category` değeri `"fund"` olan satırları kullanıcı için tıklanabilir hâle getirir. Tıklanabilirlik kararı backend tarafından üretilen `tefas_tradable` boolean bayrağına göre verilir; frontend bu bayrağı okur, satırı aktif veya devre dışı render eder, hover sırasında ilgili fund detail / fund performance verisini arka planda prefetch eder ve tıklamada `onOpenFund(asset_code, "overview")` aracılığıyla mevcut Fund_Detail_Page'e geçer.

Tasarımın hedefleri:

1. **Tek kaynak doğruluğu (Backend):** `tefas_tradable` türetimi, halihazırda holdings response oluşturulurken bir kez yüklenen Fund_Snapshot_Index referansı üzerinden yapılır; pozisyon başına ayrı bir TEFAS sorgusu yapılmaz (Req 2.5).
2. **Tek render kuralı (Frontend):** Alt fon satırı render mantığı paylaşılan bir bileşen + custom hook üzerine indirilir; `FundHoldingsPanel` ve diğer alt fon listeleri (rank panel, group table, highlights) bu paylaşılan bileşeni kullanır (Req 4).
3. **Hızlı detay açılışı:** Hover üzerine 150 ms debounce sonrası react-query `prefetchQuery` çağrıları ile `Fund_Detail_Query_Key` ve `Fund_Performance_Query_Key` (period `"1Y"`) doldurulur; 60 sn cache-key throttle ve 2 concurrent in-flight üst sınırı uygulanır (Req 5).
4. **Geriye dönük uyum:** Eski cache yanıtlarında `tefas_tradable` alanı yoksa, frontend o satırı Req 3'teki devre dışı görünümle render eder; backend Holdings_Response_Schema_Version değerini bir artırarak yeni alanı içermeyen response cache girdilerini kendiliğinden geçersiz kılar (Req 6, Req 9).
5. **Erişilebilirlik:** Tıklanabilir satırlar klavye sırasına dahildir, devre dışı satırlar `aria-disabled="true"` taşır ve tooltip mesajı `aria-describedby` ile ekran okuyuculara sunulur (Req 7).
6. **Restricted_Data_Notice:** Fund_Detail_Page açıldığında kritik alanlar eksikse oturum boyunca kapatılabilir bir uyarı şeridi gösterilir (Req 8).

### Kapsam Dışı

- Yeni bir TEFAS adapter veya yeni bir backend endpoint'i eklenmez. Mevcut `/funds/{code}/holdings` ve `/funds` cevapları zenginleştirilir.
- Fund_Snapshot_Index'in türetim mantığı (TEFAS_OPEN_ONLY filtresi) değişmez; bu özellik yalnızca o index'i tüketir.
- `Fund_Holdings_Live` (`/funds/{code}/holdings/live`) yanıtındaki sınırlı pozisyon alan listesi `_FUND_HOLDINGS_LIVE_POSITION_FIELDS` korunur; canlı endpoint sadece fiyat/getiri alanlarını döndürdüğü için `tefas_tradable` orada gerekmez.
- KAP holdings parser (KAP_Holdings_Parse_Version) ve KAP raw cache şeması değişmez.

## Architecture

```mermaid
flowchart LR
    subgraph Backend["Backend (FastAPI)"]
        A1["/funds<br/>fund_service.load_funds_snapshot"] --> SI[(Fund_Snapshot_Index<br/>funds_latest.json)]
        SI --> SM["api._fund_snapshot_row_map_with_meta()"]
        H1["/funds/{code}/holdings<br/>(_cached_response v=4)"] --> SP[("KAP holdings static payload<br/>parse_version=KAP_HOLDINGS_PARSE_VERSION")]
        SP --> EN["_enrich_fund_holdings_with_daily_market_data"]
        SM --> EN
        EN -->|"tefas_tradable derivation<br/>(only for asset_type='fund')"| RR[Holdings response JSON]
    end

    subgraph Frontend["Frontend (React + react-query)"]
        FE1[FundsPage / FundDetail] --> FH[FundHoldingsPanel]
        FH --> SR[SubFundRow<br/>(shared component)]
        SR -->|hover/focus 150 ms| HP[useHoverPrefetch]
        HP -->|prefetchQuery| QC[(react-query cache)]
        SR -->|click / Enter / Space| OF[onOpenFund]
        OF --> FE1
        FE1 --> RDN[RestrictedDataNotice]
    end

    RR -->|GET /funds/{code}/holdings| FH
    QC -->|GET /funds/{code} & /funds/{code}/performance?range=1Y| Backend
```

### Akış Özeti

1. Frontend `FundHoldingsPanel`'i çağırdığında `apiClient.fundHoldings(code)` üzerinden `/funds/{code}/holdings` çağrılır. Backend, cache'lenmiş `_fund_holdings_static_payload` payload'ını alır, `_enrich_fund_holdings_with_daily_market_data` ile zenginleştirir, ve bu adımda `tefas_tradable` türetimini Fund_Snapshot_Index üzerinden yapar.
2. Frontend, `FundPortfolioPosition.tefas_tradable` alanını okur ve paylaşılan `SubFundRow` bileşeninde tıklanabilir / devre dışı render kararını verir.
3. Kullanıcı satır üzerine hover ettiğinde veya klavye ile odaklandığında, `useHoverPrefetch` hook'u 150 ms debounce uygular, sonra `queryClient.prefetchQuery(["fund-detail", code])` ve `queryClient.prefetchQuery(["fund-performance", code, "1Y"])` çağırır.
4. Tıklama veya Enter/Space ile `onOpenFund(asset_code, "overview")` callback'i tetiklenir, üst seviye router (`App.tsx`) `toFundDetail(...)` ile rotaya geçer ve Fund_Detail_Page render edilir; react-query cache'i sıcak olduğu için fund detail ve 1Y performance neredeyse anlık görünür.
5. Fund_Detail_Page yüklendikten sonra kritik alanlardan biri eksikse `RestrictedDataNotice` gösterilir (Req 8).

### Hover → Prefetch → Click Sequence

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant Row as SubFundRow
    participant HP as useHoverPrefetch
    participant Q as queryClient
    participant API as Backend
    participant Page as FundDetailPage

    U->>Row: mouseenter / focus
    Row->>HP: schedule(asset_code)
    Note over HP: setTimeout 150 ms<br/>(Hover_Prefetch_Debounce_Ms)
    alt user leaves before 150 ms
        U->>Row: mouseleave / blur
        Row->>HP: cancel(asset_code)
        HP-->>HP: clearTimeout — no network
    else 150 ms elapsed
        HP->>HP: enqueue(asset_code)
        Note over HP: Max_Concurrent_Hover_Prefetch = 2;<br/>cancel oldest queued task if needed
        HP->>Q: prefetchQuery(["fund-detail", code])
        HP->>Q: prefetchQuery(["fund-performance", code, "1Y"])
        Q->>API: GET /funds/{code}
        Q->>API: GET /funds/{code}/performance
        API-->>Q: payloads cached
        Note over HP: per-key throttle:<br/>"completed" stamp < 60 s blocks re-prefetch
    end

    U->>Row: click / Enter / Space
    Row->>Page: onOpenFund(asset_code, "overview")
    Page->>Q: useQuery(["fund-detail", code])
    Q-->>Page: cache hit (warm)
    Page-->>U: render Fund_Detail_Page
    Page->>Page: detect missing critical fields
    alt critical fields missing
        Page-->>U: show RestrictedDataNotice
    end
```

## Components and Interfaces

### Backend

#### 1. `_FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION` (app/api.py)

Mevcut sabit `_FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION = 3`. Bu özellik yayına alınırken `4`'e çıkarılır. Cache anahtarı:

```python
key_fn=lambda *, normalized: f"api:fund-holdings:{normalized}:v{_FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION}"
```

Sürüm artışı, eski (`v3`) anahtarla yazılmış girdileri pratikte çöpe atar — yeni okumalar farklı anahtara denk gelir, eski girdiler doğal TTL'leri (`_FUND_HOLDINGS_RESPONSE_CACHE_TTL`, varsayılan 1 saat) ile yok olur. KAP_Holdings_Parse_Version (`KAP_HOLDINGS_PARSE_VERSION = 9`, `app/fund_service.py`) **dokunulmaz**, çünkü parser çıktısı değişmiyor; yalnızca response sarmalama sırasında yeni bir alan ekleniyor (Req 6.3, Req 9.3).

#### 2. `_derive_tefas_tradable` (app/api.py — yeni helper)

`_enrich_fund_holdings_with_daily_market_data` fonksiyonu zaten Fund_Snapshot_Index'i tek seferlik olarak yükler:

```python
fund_rows, fund_rows_meta = _fund_snapshot_row_map_with_meta()  # Dict[normalized_code -> snapshot_row]
```

`enriched_positions` döngüsünün içinde, `_holding_type(row) == "fund"` koşulunda yeni bir satır eklenir:

```python
def _derive_tefas_tradable(position: Dict[str, Any], snapshot_codes: set[str]) -> Optional[bool]:
    if _holding_type(position) != "fund":
        return None
    code = _holding_code(position)
    if not code:
        return False
    return code in snapshot_codes
```

`_enrich_fund_holdings_with_daily_market_data` döngüsü:

```python
snapshot_codes: set[str] = set(fund_rows.keys())
# Fund_Snapshot_Index erişilemezse (fund_rows boş + meta error) "fund" satırları False olarak işaretlenir.
snapshot_unavailable = (
    not snapshot_codes and (fund_rows_meta.get("error") == "snapshot_unavailable")
)
if snapshot_unavailable:
    logger.warning(
        "Fund_Snapshot_Index unavailable; marking all 'fund' positions as not TEFAS-tradable for %s",
        normalized_fund,
    )

for position in positions:
    row = dict(position)
    ...
    if _holding_type(row) == "fund":
        row["tefas_tradable"] = (
            False if snapshot_unavailable else _derive_tefas_tradable(row, snapshot_codes)
        )
    enriched_positions.append(row)
```

Sonuç:

- `asset_type == "fund"` AND `asset_code` Fund_Snapshot_Index'te → `tefas_tradable: true`
- `asset_type == "fund"` AND `asset_code` Fund_Snapshot_Index'te değil → `tefas_tradable: false`
- `asset_type == "fund"` AND Fund_Snapshot_Index erişilemez → `tefas_tradable: false` + backend log uyarısı
- `asset_type != "fund"` → `tefas_tradable` alanı eklenmez

Bu, Req 2.1–2.6'yı karşılar. Lookup `set` üzerinden O(1) olduğu için pozisyon başına harici sorgu yoktur (Req 2.5).

#### 3. `_fund_snapshot_row_map_with_meta` (app/api.py — değişiklik yok)

Fonksiyon zaten 5 dakikalık in-process + Redis cache ile snapshot satırlarını sağlar; bu özellik bu fonksiyonu olduğu gibi tüketir. `fund_rows_meta.get("error") == "snapshot_unavailable"` durumu yukarıdaki helper içinde yakalanır (Req 2.6).

### Frontend

#### 1. `SubFundRow` (yeni paylaşılan bileşen, `frontend/src/components/SubFundRow.tsx`)

Tüm alt fon satırı render noktaları (FundHoldingsPanel'in rank paneli, group table'ı, highlights bölümleri ve gelecekteki sub-fund listeleri) bu bileşeni kullanır.

```tsx
type SubFundRowProps = {
  position: FundPortfolioPosition;
  onOpenFund?: (fundCode: string, tab?: FundTab) => void;
  // Render slot'ları — satırın nasıl görüneceğini taşıyıcı bileşen kontrol eder
  children: ReactNode;
  className?: string;
  // Aksesibilite: satırın taşıdığı semantik etiket
  ariaLabel?: string;
};
```

Davranış:

- `position.asset_type !== "fund"` → bileşen yalnızca pasif bir `<div>` döner; tıklama / klavye davranışı yoktur (Req 1.5).
- `position.asset_type === "fund"` AND `position.tefas_tradable === true` → `<button type="button">` veya `tabindex="0"` taşıyan div; click + `onKeyDown` (Enter, Space → preventDefault + onOpenFund) (Req 1.1–1.4, Req 7.1, 7.3, 7.4).
- `position.asset_type === "fund"` AND (`tefas_tradable === false` OR `tefas_tradable === undefined`) → devre dışı render: `aria-disabled="true"`, `tabindex={-1}`, `cursor: not-allowed`, azaltılmış kontrast; click/Enter/Space `event.preventDefault()` ile yutulur ve `onOpenFund` çağrılmaz (Req 3.1–3.5, Req 6.1, Req 7.2, 7.5).
- `onOpenFund` prop verilmediyse hiçbir "fund" satırı tıklanabilir değildir; "fund" satırları da disabled görünümle render edilir, click/key etkinlikleri yutulur (Req 4.2).
- 400 ms hover/focus tooltip: `useDelayedTooltip(400)` ile mouseenter/focus zamanlayıcısı kurulur. Tooltip mesajı sabit dize: `"Bu fon TEFAS'ta açık olarak işlem görmediği için detay sayfası açılamıyor"`. Tooltip DOM'da `id="sub-fund-tooltip-${asset_code}"` ile render edilir ve satıra `aria-describedby` olarak bağlanır (Req 3.4, Req 7.5).

```tsx
function SubFundRow({ position, onOpenFund, children, className, ariaLabel }: SubFundRowProps) {
  const isFundCategory = (position.asset_type ?? "").toLowerCase() === "fund";
  const tradable = position.tefas_tradable === true;
  const disabled = isFundCategory && (!tradable || !onOpenFund);
  const interactive = isFundCategory && tradable && !!onOpenFund;
  const code = position.asset_code ?? "";

  const open = useCallback(() => {
    if (!interactive || !code) return;
    onOpenFund!(code, "overview");
  }, [interactive, code, onOpenFund]);

  const onKeyDown = useCallback((e: KeyboardEvent<HTMLDivElement>) => {
    if (!isFundCategory) return;
    if (e.key !== "Enter" && e.key !== " ") return;
    e.preventDefault();
    if (interactive) open();
    // disabled: yutulur, çağrı yapılmaz
  }, [isFundCategory, interactive, open]);

  const tooltip = useDelayedTooltip({ delayMs: 400, enabled: disabled && isFundCategory });

  if (!isFundCategory) {
    return <div className={className}>{children}</div>;
  }

  return (
    <div
      className={[className, "sub-fund-row", interactive ? "is-interactive" : "is-disabled"].filter(Boolean).join(" ")}
      role={interactive ? "link" : undefined}
      tabIndex={interactive ? 0 : -1}
      aria-disabled={disabled || undefined}
      aria-label={ariaLabel ?? (interactive ? `${code} fon detayını aç` : `${code} fonu TEFAS'ta işlem görmüyor`)}
      aria-describedby={tooltip.visible ? tooltip.id : undefined}
      onClick={interactive ? open : (e) => e.preventDefault()}
      onKeyDown={onKeyDown}
      onMouseEnter={tooltip.onEnter}
      onMouseLeave={tooltip.onLeave}
      onFocus={tooltip.onFocus}
      onBlur={tooltip.onBlur}
    >
      {children}
      {tooltip.visible && (
        <span role="tooltip" id={tooltip.id} className="sub-fund-row-tooltip">
          Bu fon TEFAS'ta açık olarak işlem görmediği için detay sayfası açılamıyor
        </span>
      )}
    </div>
  );
}
```

`FundHoldingsPanel` içindeki üç render noktası (rank panel, group table, highlights) bu bileşeni `children` olarak içeriklerini geçirerek kullanır. Mevcut hisse render mantığı (`onOpenTicker`) bozulmaz — yalnızca `asset_type === "fund"` dalı `SubFundRow`'a delege edilir (Req 1.5).

#### 2. `useHoverPrefetch` (yeni hook, `frontend/src/hooks/useHoverPrefetch.ts`)

Hover/focus üzerine fund detail ve fund performance prefetch'ini yöneten merkezi hook. Mantık tek bir module-scoped controller (`hoverPrefetchController`) üzerinden ilerler; böylece tüm satırlar aynı concurrency kuyruğunu paylaşır.

```ts
import { useQueryClient } from "@tanstack/react-query";

const HOVER_PREFETCH_DEBOUNCE_MS = 150;
const HOVER_PREFETCH_THROTTLE_MS = 60_000;
const HOVER_PREFETCH_MAX_CONCURRENT = 2;
const DEFAULT_PERFORMANCE_PERIOD = "1Y";

type PrefetchTask = { id: number; cacheKey: string; run: () => Promise<void> };

const completedAt = new Map<string, number>(); // cacheKey -> ms timestamp
const inflight = new Set<string>();             // cacheKey set
let queue: PrefetchTask[] = [];
let nextId = 0;

function cacheKeyOf(parts: unknown[]): string {
  return JSON.stringify(parts);
}

function isThrottled(cacheKey: string, now: number): boolean {
  const ts = completedAt.get(cacheKey);
  return ts != null && now - ts < HOVER_PREFETCH_THROTTLE_MS;
}

function pump() {
  while (inflight.size < HOVER_PREFETCH_MAX_CONCURRENT && queue.length > 0) {
    const task = queue.shift()!;
    if (inflight.has(task.cacheKey) || isThrottled(task.cacheKey, Date.now())) continue;
    inflight.add(task.cacheKey);
    task.run().finally(() => {
      inflight.delete(task.cacheKey);
      completedAt.set(task.cacheKey, Date.now());
      pump();
    });
  }
}

export function useHoverPrefetch() {
  const qc = useQueryClient();
  const timersByCode = useRef(new Map<string, number>());

  const cancel = useCallback((code: string) => {
    const handle = timersByCode.current.get(code);
    if (handle != null) {
      window.clearTimeout(handle);
      timersByCode.current.delete(code);
    }
  }, []);

  const schedule = useCallback((code: string, enabled: boolean) => {
    if (!enabled || !code) return;       // disabled → no scheduling at all (Req 5.9)
    cancel(code);
    const handle = window.setTimeout(() => {
      timersByCode.current.delete(code);
      enqueuePrefetch(qc, ["fund-detail", code]);
      enqueuePrefetch(qc, ["fund-performance", code, DEFAULT_PERFORMANCE_PERIOD]);
    }, HOVER_PREFETCH_DEBOUNCE_MS);
    timersByCode.current.set(code, handle);
  }, [qc, cancel]);

  useEffect(() => () => {
    // Component unmount: tüm bekleyen debounce'ları iptal et.
    for (const handle of timersByCode.current.values()) window.clearTimeout(handle);
    timersByCode.current.clear();
  }, []);

  return { schedule, cancel };
}

function enqueuePrefetch(qc: QueryClient, key: unknown[]) {
  const cacheKey = cacheKeyOf(key);
  const now = Date.now();
  if (inflight.has(cacheKey) || isThrottled(cacheKey, now)) return;

  // Kuyrukta zaten varsa duplicate eklemeyelim.
  if (queue.some((task) => task.cacheKey === cacheKey)) return;

  // Concurrency üst sınırına dayandıysak, henüz başlamamış (kuyrukta) en eski görevi iptal et.
  if (inflight.size >= HOVER_PREFETCH_MAX_CONCURRENT && queue.length > 0) {
    queue.shift(); // oldest queued task is cancelled (Req 5.7)
  }

  const task: PrefetchTask = {
    id: nextId++,
    cacheKey,
    run: async () => {
      await qc.prefetchQuery({
        queryKey: key,
        queryFn: () => prefetchFnFor(key),
        staleTime: HOVER_PREFETCH_THROTTLE_MS,
      });
    },
  };
  queue.push(task);
  pump();
}

function prefetchFnFor(key: unknown[]) {
  const [kind, code, period] = key as [string, string, string?];
  if (kind === "fund-detail") return apiClient.fundDetail(code);
  if (kind === "fund-performance") return apiClient.fundPerformance(code);
  throw new Error(`Unsupported prefetch key: ${kind}`);
}
```

Önemli sözleşmeler:

- **Debounce 150 ms (Req 5.1, 5.2):** `schedule(code, enabled)` her çağrıda satır bazlı timer'ı resetler. `cancel(code)` mouseleave/blur ile çağrıldığında timer süresi dolmadan iptal olur, ağ isteği başlatılmaz (Req 5.4).
- **Cache anahtarı (Req 5.1, 5.3, 5.6):** `["fund-detail", asset_code]` ve `["fund-performance", asset_code, "1Y"]`. `Default_Performance_Prefetch_Period = "1Y"`. Aynı `(code, "3M")` farklı bir cache anahtarıdır ve ayrı throttle'lanır.
- **60 sn per-key throttle (Req 5.5):** `completedAt` Map'i her başarılı prefetch tamamlandığında güncellenir; bir cache anahtarının son tamamlanma zamanı 60 sn'den daha yakınsa enqueue edilmez. Throttle penceresi anahtar başına bağımsız olduğu için Req 5.6 doğrudan sağlanır.
- **Max concurrent in-flight = 2 (Req 5.7, 5.8):** `inflight` Set'i ile sınır. Yeni bir görev geldiğinde sınır doluysa kuyruktaki en eski görev `queue.shift()` ile iptal edilir; **in-flight olan görev iptal edilmez** (Req 5.8). `pump()` her tamamlandığında bir sonraki görevi başlatır.
- **Disabled satır → no-op (Req 5.9):** `SubFundRow`, `interactive === false` durumunda `schedule(...)` çağrısını yapmaz; mouseenter/focus handler'ları yalnızca tooltip kontrol fonksiyonunu çağırır.
- **TanStack Query bağımlılığı:** Codebase şu anda `@tanstack/react-query` kullanmıyor (bkz. `frontend/package.json`). Bu özellik `@tanstack/react-query` ekler. `apiClient.fundDetail(code)` ve `apiClient.fundPerformance(code)` zaten mevcut olduğu için query fonksiyonları doğrudan bunları sarmalar; bileşenlerin `useQuery` benimsemesi aşamalı olur, bu özellik kapsamında en az `FundDetail` ve `FundPerformance` query'leri tanımlanır.

#### 3. `RestrictedDataNotice` (`frontend/src/components/RestrictedDataNotice.tsx`)

Fund_Detail_Page üzerinde, kritik alanlar eksik tespit edildiğinde tek satırlık dismissible bir banner gösterir.

Kritik alan listesi (`isFundDetailRestricted(detail)` helper):

- `detail.last_price` `null/undefined` veya `<= 0`
- `detail.last_price_date` `null/undefined`

Bu iki alandan en az biri eksikse banner gösterilir (Req 8.1, 8.2). Tüm alanlar doluysa gösterilmez (Req 8.3). Kullanıcı kapatma butonuna tıkladığında, banner mevcut görüntüleme oturumu için gizlenir; gizlilik durumu `sessionStorage` altında `restricted-notice:dismissed:{fund_code}` anahtarıyla tutulur (Req 8.4). Yeni bir tarayıcı oturumu (`sessionStorage` temiz) banner'ı yeniden gösterir.

#### 4. `FundsPage.tsx` Entegrasyonu

- Mevcut `FundHoldingsPanelProps`'a `onOpenFund?: (code: string, tab?: FundTab) => void` eklenir.
- `FundHoldingsPanel`'in çağrı noktasında (`activeTab === 'allocation'` bloğu, satır ~5850) prop olarak `onOpenFund` geçilir; bu zaten `FundsView` scope'unda tanımlıdır.
- `FundHoldingsPanel` içindeki üç render konumu (rank panel, group table, highlights) `SubFundRow` ile sarılır. Mevcut `onOpenTicker` davranışı `local_equity` dalında olduğu gibi bırakılır.

### Backward Compatibility (Req 6)

- **Backend:** Holdings_Response_Schema_Version 3 → 4 artırıldığı an, yeni anahtar (`api:fund-holdings:{code}:v4`) eski anahtarla yazılmış girdileri görmez; yeni okumalar tetiklendiğinde fresh response yazılır (Req 6.2, Req 9.1, 9.2).
- **Frontend:** `FundPortfolioPosition.tefas_tradable` opsiyonel olarak tanımlanır. Frontend, `position.asset_type === "fund"` AND `position.tefas_tradable !== true` durumunu tek bir devre dışı render kuralı altında birleştirir; yani `undefined` (eski yanıt) ve `false` (yeni yanıt) aynı görsel davranışa sahip olur (Req 6.1). Ancak Req 6.4 gereği, sayfa içinde herhangi bir `"fund"` satırı `tefas_tradable` taşıyorsa, **diğer "fund" satırları** için de yanıttaki değer aynen kullanılır — frontend tek bir varsayılana sabitlemez. Bu, `position.tefas_tradable !== true` koşulunun `position.tefas_tradable !== true` (yani `undefined → disabled`, `false → disabled`, `true → interactive`) şeklinde **per-position** uygulanması ile sağlanır.

## Data Models

### Backend Response Schema (Holdings)

`/funds/{fund_code}/holdings` cevabındaki `positions[]` öğesinin "fund" kategorisindeki şekli aşağıdaki gibi genişletilir:

```json
{
  "fund_code": "IIE",
  "status": "ok",
  "positions": [
    {
      "fund_code": "IIE",
      "asset_code": "IIP",
      "asset_name": "İŞ PORTFÖY PARA PİYASASI FONU",
      "asset_type": "fund",
      "weight": 12.34,
      "previous_weight": 11.20,
      "weight_change": 1.14,
      "amount": null,
      "market_value": null,
      "report_date": "2026-04-30",
      "previous_report_date": "2026-03-31",
      "source_report_url": "https://www.kap.org.tr/...",
      "source_type": "kap_portfolio_allocation_report",
      "parse_confidence": 0.97,
      "tefas_tradable": true,
      "sector_code": null,
      "sector_label": null,
      "estimated_exposure_value": 1234567.0,
      "estimated_pnl_value": 123.45,
      "estimated_fund_return_contribution_pct": 0.0123
    },
    {
      "asset_code": "ABCDE",
      "asset_type": "fund",
      "tefas_tradable": false,
      "...": "..."
    },
    {
      "asset_code": "GARAN",
      "asset_type": "local_equity",
      "...": "...",
      "comment": "tefas_tradable alanı yok"
    }
  ],
  "portfolio_effect": { "...": "..." },
  "source_metadata": {
    "schema_version": 4,
    "...": "..."
  }
}
```

Notlar:

- `tefas_tradable` yalnızca `asset_type == "fund"` satırlarında bulunur (Req 2.4).
- `null` değer yerine alan **olmaması** ile alan **`true`/`false`** olması arasında ayrım korunur. Frontend `tefas_tradable !== true` mantığını kullanır.
- `source_metadata.schema_version` opsiyonel olarak eklenebilir; ancak dış sözleşme (URL cache anahtarı) `_FUND_HOLDINGS_RESPONSE_SCHEMA_VERSION` üzerinden zaten taşınır.

### Frontend Type Update (`frontend/src/api/types.ts`)

```ts
export interface FundPortfolioPosition {
  // ... mevcut alanlar
  asset_type: string | null;
  tefas_tradable?: boolean; // YENİ — yalnızca asset_type === "fund" satırlarında dolu olabilir
}
```

`FundDetail` üzerinde `last_price`, `last_price_date` zaten mevcut; `RestrictedDataNotice` yeni alan eklemez.

### Frontend Internal State

- `useHoverPrefetch` module-state: `completedAt: Map<string, number>`, `inflight: Set<string>`, `queue: PrefetchTask[]`. Bunlar global'dir (modül scope) çünkü 2 concurrent sınırı tüm sayfa boyunca paylaşılır. Test edilebilirlik için `__resetHoverPrefetchStateForTests()` helper'ı eklenir.
- `RestrictedDataNotice` dismiss durumu: `sessionStorage["restricted-notice:dismissed:" + fund_code] = "1"`. Page reload aynı oturumda banner'ı tekrar göstermez; oturum kapanınca temizlenir.

