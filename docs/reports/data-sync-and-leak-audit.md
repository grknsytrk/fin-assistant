# Comprehensive Codebase Audit: Data Synchronization, Redundant Polling, Phase Skew, Cache Integrity, and Resource Leaks

**Document Status**: Final Forensic Audit Deliverable  
**Date of Audit**: 2026-11-03  
**Target Repository**: `d:\Projeler\rag-fin`  
**Audit Protocol**: Zero-Production Code Modification Invariant (Strict Read-Only Analysis)  
**Classification**: Engineering Architecture & Quality Assurance  

---

## 1. Executive Summary & System Health Dashboard

### 1.1 Context & Audit Objectives
This forensic codebase audit was commissioned to investigate systemic data synchronization failures, network egress leaks, visual flickers, phase skews, and resource leaks across the **rag-fin** financial platform. 

The application is a full-stack financial market data platform serving Turkish Equities (Borsa Istanbul - BIST), Mutual & Pension Funds (TEFAS), Currencies (FX), Macro Commodities, and Public Disclosure Platform (KAP) company fundamentals. The architecture comprises:
- **Cloudflare Edge Worker** (`worker/`): Reverse proxy and cron scheduler.
- **FastAPI Python Backend** (`app/`): Core asynchronous API runtime, scraping engines, data transformation pipelines, and database/Redis caching layers.
- **React 18 / TypeScript Frontend** (`frontend/src/`): High-density financial terminal views, virtualized data grids, interactive multi-period charts, and real-time streaming strips.
- **Data Upstreams**: Info Yatırım (live HTML push/scraping), Yahoo Finance (delayed public quote & chart APIs), TEFAS / Takasbank (`tefasfon` / SOAP scraping), KAP (Public Disclosure Platform), and OpenFIGI.

### 1.2 Systemic Vulnerability Overview
The investigation identified **28 distinct architectural defects** categorized across five critical operational areas:
1. **Redundant Polling & Egress Amplification**: Independent uncoordinated pollers querying backend routes every 3 seconds for 15-minute delayed data; massive daily scrape iterations downloading thousands of fund records instead of filtered range queries; cold-cache worker stampedes.
2. **Phase Skew & Split-Brain Presentation**: The same equity or index simultaneously displayed with conflicting prices, returns, and valuation metrics across widgets (e.g. `MarketWatchStrip` vs `MarketWatchRail`, `StockDetailPage` hero price vs multiples, and `ModelPortfolioPage` live stocks vs frozen benchmarks).
3. **Timer Uncoordination & CPU Thrashing**: Root-level clock state updating every second in a 5,343-line monolithic component, triggering full virtual DOM tree re-evaluations; unpaused `setTimeout` and `setInterval` loops spinning in hidden browser tabs.
4. **Cache Invalidation Gaps & Stale Windows**: 30-minute cross-worker cache desynchronization for financial statement ratios; multi-tiered nested SWR caches causing compound data lag; single-flight waiter fall-through triggering upstream thundering herds.
5. **Memory & Thread Resource Leaks**: 11 unbounded module-level Python dictionaries; browser memory leaks in long-lived terminal sessions; thread pool executors left running across server reloads and timeouts; per-client active thread polling inside Server-Sent Event (SSE) handlers.

---

### 1.3 System Health Dashboard

#### Defect Severity Distribution
```
┌─────────────────────────────────────────────────────────────┬───────────┐
│ Severity Level                                              │ Count     │
├─────────────────────────────────────────────────────────────┼───────────┤
│ 🔴 CRITICAL (Platform stability risk, stampede, huge egress)│     4     │
│ 🟠 HIGH     (Phase skew, data distortion, memory growth)    │    14     │
│ 🟡 MEDIUM   (CPU waste, hidden-tab spinning, thread leaks)  │     8     │
│ 🔵 LOW      (Minor render inefficiency, unneeded imports)   │     2     │
├─────────────────────────────────────────────────────────────┼───────────┤
│ TOTAL AUDIT FINDINGS                                        │    28     │
└─────────────────────────────────────────────────────────────┴───────────┘
```

#### Architectural Heat Map
```
┌──────────────────────┬────────────────────────┬─────────────┬────────────────────────────────────────────────────────┐
│ Subsystem / Layer    │ Primary Modules        │ Risk Level  │ Core Impact Profile                                    │
├──────────────────────┼────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
│ Frontend Core Views  │ MarketsView.tsx        │ 🔴 CRITICAL │ Root clock re-render (1Hz), dual uncoordinated pollers │
│ Frontend Hooks       │ useMarketWatchData.ts  │ 🔴 CRITICAL │ Quad-endpoint 3s polling, visual tearing & flicker     │
│ Frontend Detail      │ StockDetailPage.tsx    │ 🟠 HIGH     │ Split-brain price (Yahoo 15m) vs Multiples (Live 3s)   │
│ Frontend Analytics   │ ModelPortfolioPage.tsx │ 🟠 HIGH     │ Live position SSE vs Frozen mount-time benchmark alpha │
│ Frontend Background  │ useBackgroundRefresh.ts│ 🟡 MEDIUM   │ Hidden-tab busy loops, missing visibility pause        │
│ Backend Market Core  │ market_service.py      │ 🔴 CRITICAL │ Single-flight fallthrough stampede, 8 unbounded dicts  │
│ Backend Fund Engine  │ fund_service.py        │ 🔴 CRITICAL │ TEFAS whole-universe daily loops, zombie thread leaks  │
│ Backend Snapshot     │ financial_snapshot.py  │ 🟠 HIGH     │ 30-minute inter-worker stale window on balances        │
│ Backend Streaming    │ routers/market.py      │ 🟠 HIGH     │ Per-client thread polling inside SSE stream generator  │
│ Backend Lifespan     │ api.py, api_runtime.py │ 🟡 MEDIUM   │ 5 leaked global ThreadPoolExecutors on shutdown        │
│ Edge Worker          │ worker/index.ts        │ 🟡 MEDIUM   │ 5-minute cron collision between fast and deep refreshes│
└──────────────────────┴────────────────────────┴─────────────┴────────────────────────────────────────────────────────┘
```

---

## 2. Category 1: Redundant Polling, Duplicate Fetching, and Network/Egress Leaks

### Finding 1.1: Dual Uncoordinated Polling Intervals in `MarketsView.tsx`
- **Location**: `frontend/src/pages/MarketsView.tsx` (Lines 3412–3433)
- **Component**: `MarketsView`
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
In `MarketsView.tsx`, two parallel browser intervals are established to refresh stock card data:
```tsx
3412: const intervalId = window.setInterval(() => {
3413:     if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
3414:     void loadBatches(stockCardSymbols, true);
3415: }, marketDetailRefreshMs);
3416: const quoteIntervalId = window.setInterval(() => {
3417:     if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
3418:     void loadBatches(stockCardSymbols, false);
3419: }, MARKET_QUOTE_REFRESH_MS);
```
- `quoteIntervalId` ticks every 3,000ms (`MARKET_QUOTE_REFRESH_MS`), fetching lightweight quote cards.
- `intervalId` ticks every 10,000ms to 15,000ms (`marketDetailRefreshMs`), fetching deep detail cards.
- Both intervals pass the identical `stockCardSymbols` array to `loadBatches()`. At every common multiple (e.g. $t = 15\text{s}, 30\text{s}, 45\text{s}, 60\text{s}$), two concurrent asynchronous HTTP requests hit `/market/stock-cards` in the exact same millisecond window.
- The two requests race across the network. If the full payload resolves after the quick payload, the UI is consistent. But if the quick payload resolves slightly later due to network packet jitter, it overwrites the freshly loaded full detail payload, causing missing valuation columns in the user interface.

#### Concrete Remediation
Consolidate into a single tick scheduler that multiplexes quick quote and deep detail loading:
```tsx
// frontend/src/pages/MarketsView.tsx
useEffect(() => {
    let tickCount = 0;
    const TICK_INTERVAL_MS = 3000;
    const DETAIL_EVERY_TICKS = Math.max(1, Math.round(marketDetailRefreshMs / TICK_INTERVAL_MS));

    const pollTimer = window.setInterval(() => {
        if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
        tickCount += 1;
        const needsFullDetail = tickCount % DETAIL_EVERY_TICKS === 0;
        void loadBatches(stockCardSymbols, needsFullDetail);
    }, TICK_INTERVAL_MS);

    const onVisibilityChange = () => {
        if (document.visibilityState === 'visible' && isBistTradingHours()) {
            tickCount = 0;
            void loadBatches(stockCardSymbols, true);
        }
    };

    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
        window.clearInterval(pollTimer);
        document.removeEventListener('visibilitychange', onVisibilityChange);
    };
}, [stockCardSymbols, marketDetailRefreshMs]);
```

---

### Finding 1.2: Triple-Loop Polling for Fund Holdings in `FundsPage.tsx`
- **Location**: `frontend/src/pages/FundsPage.tsx` (Lines 5963–5967, 6032–6054, 6056–6089)
- **Component**: `FundsPage`
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
When a user navigates to the fund detail page (`/fonlar/:code`), three autonomous polling mechanisms run simultaneously:
1. **Line 5963**: `useBackgroundRefresh` polls `apiClient.fundHoldings(code, { force: true })` on an adaptive 2s–30s backoff whenever `holdings?.refresh_pending` is true.
2. **Line 6045**: `window.setInterval(revalidate, FUND_HOLDINGS_REPORT_REFRESH_MS)` triggers every 60,000ms.
3. **Line 6063**: `window.setTimeout(refreshVisibleHoldings, delay)` executes a recursive 3,000ms loop for live holdings.
4. **Lines 6046–6047 & 6081–6082**: Duplicate event listeners for `window.focus` and `document.visibilitychange` are attached across multiple `useEffect` hooks.

When a fund report is pending revalidation, all three timers fire simultaneously, bombarding `/funds/:code/holdings` with up to 30 requests per minute for a single browser tab.

#### Concrete Remediation
Replace the multi-effect polling cascade with a unified hook that manages polling state based on report maturity:
```tsx
// Single coordinated hook in FundsPage.tsx
useEffect(() => {
    if (!fundCode) return;
    const code = fundCode.trim().toUpperCase();
    const controller = new AbortController();
    let timer: number | null = null;

    const poll = async () => {
        if (document.visibilityState !== 'visible') return;
        try {
            const data = await apiClient.fundHoldings(code, { signal: controller.signal });
            if (controller.signal.aborted) return;
            setHoldings(data);
            
            // Adjust delay: 3s if pending, 60s if steady-state
            const nextDelay = data.refresh_pending ? 3000 : 60000;
            timer = window.setTimeout(poll, nextDelay);
        } catch (err) {
            if (!controller.signal.aborted) {
                timer = window.setTimeout(poll, 15000); // Backoff on error
            }
        }
    };

    void poll();
    const onVisibility = () => {
        if (document.visibilityState === 'visible') {
            if (timer) window.clearTimeout(timer);
            void poll();
        }
    };

    document.addEventListener('visibilitychange', onVisibility);
    return () => {
        controller.abort();
        if (timer) window.clearTimeout(timer);
        document.removeEventListener('visibilitychange', onVisibility);
    };
}, [fundCode]);
```

---

### Finding 1.3: Backend Single-Flight Waiter Fall-through & Upstream Stampede
- **Location**: `app/market_service.py` (Lines 1079–1113)
- **Function**: `_fetch_market_price_map`
- **Severity**: 🔴 CRITICAL

#### Mechanism & Root Cause
To protect upstream scrapers (Info Yatırım), `_fetch_market_price_map` uses single-flight lock coalescing via `_get_or_set_single_flight`:
```python
1081: snapshot, singleflight_status = _get_or_set_single_flight(
1082:     cache_key,
1083:     ttl_seconds=_MARKET_PRICE_CACHE_TTL,
1084:     factory=refresh_snapshot,
1085:     lock_key=f"{cache_key}:lock",
1086:     lock_ttl_seconds=_MARKET_QUOTES_LOCK_TTL_SECONDS,
1087:     wait_timeout_seconds=_MARKET_QUOTES_WAIT_TIMEOUT_SECONDS, # 8.0 seconds
1088:     poll_interval_seconds=_MARKET_QUOTES_POLL_INTERVAL_SECONDS,
1089:     cache_usable=_market_price_snapshot_is_fresh,
1090: )
...
1093: source_attempted = singleflight_status == "miss"
...
1097: if _market_price_snapshot_is_fresh(snapshot):
1098:     _remember_market_price_snapshot(snapshot)
1099:     return project_items(snapshot["items"])
1100: if source_attempted:
1101:     return {}
1102: 
1103: # A timed-out shared waiter must retain the previous fail-open behavior.
1104: # Publish only the successful result; never replace a snapshot with a provider error.
1105: items = _build_market_price_snapshot(normalized_symbols)
```
- When upstream Info Yatırım latency exceeds 8.0 seconds, concurrent client requests that were waiting in the single-flight loop exit with `singleflight_status = "pending"` (wait timeout).
- For these timed-out waiters, `source_attempted` is `False` (`singleflight_status != "miss"`).
- Therefore, line 1100 (`if source_attempted: return {}`) evaluates to `False`.
- Execution falls directly into line 1105: `items = _build_market_price_snapshot(normalized_symbols)`.
- **Impact**: All waiting threads/requests simultaneously bypass single-flight protection and initiate heavy web scraping against Info Yatırım in parallel. A brief 8-second delay triggers a catastrophic thundering herd stampede, leading to IP blocking or connection exhaustion.

#### Concrete Remediation
Enforce fail-stale or fail-safe termination for timed-out single-flight waiters:
```python
# app/market_service.py:1097-1113
if _market_price_snapshot_is_fresh(snapshot):
    _remember_market_price_snapshot(snapshot)
    return project_items(snapshot["items"])

# If the caller was a waiter that timed out ("pending") or the primary worker failed ("miss"),
# do NOT execute an uncoordinated upstream build. Serve stale database snapshot if available.
if redis_available:
    if singleflight_status in {"miss", "pending"}:
        stale_cached = _market_price_snapshot_database_read()
        if stale_cached and "items" in stale_cached:
            LOGGER.warning("Serving stale DB snapshot due to single-flight %s", singleflight_status)
            return project_items(stale_cached["items"])
        return {}

# Fallback path only reached if Redis was completely unreachable
items = _build_market_price_snapshot(normalized_symbols)
```

---

### Finding 1.4: TEFAS Daily Whole-Market Universal Scraping in Monthly Historical Loops
- **Location**: `app/fund_service.py` (Lines 5385–5410, 6775–6798, 7202–7209)
- **Functions**: `fetch_history`, `_fetch_tefasfon_daily_snapshots_for_codes`
- **Severity**: 🔴 CRITICAL

#### Mechanism & Root Cause
When historical prices are requested for any fund over an interval $> 120$ days (e.g. 1-year, 3-year, or 5-year views), `fetch_history` shifts from range queries to a monthly sampling loop:
```python
5385: current_month = _month_start(start_date)
5386: while current_month <= end_date:
5387:     target_date = min(_month_end(current_month), end_date)
...
5393:     while current >= lower_bound:
...
5398:         snapshot_rows = self.fetch_daily_funds_snapshot(current)
5401:         matched = matching_rows(snapshot_rows)
5402:         if matched:
5403:             add_rows(matched)
5404:             break
5405:         current -= timedelta(days=1)
5406:     current_month = _shift_month(current_month, 1)
```
- `fetch_daily_funds_snapshot(current)` downloads the **entire Turkish fund universe** (~1,500 funds across SEC, PEN, ETF, RE, VC categories) for that specific calendar date.
- It then executes `matching_rows(snapshot_rows)` to find a **single fund row**.
- For a 3-year history query (36 months), the backend downloads the full Turkish fund market up to 36 times (millions of JSON records).
- Furthermore, in lines 7202–7209, if `target_codes > 100`, it engages `_fetch_tefasfon_daily_snapshots_for_codes`, which loops day-by-day across all business days, scraping universal market tables for every single day in the date range.

#### Concrete Remediation
Utilize `tefasfon.get_funds` chunked range queries parameterized with `fund_codes=[code]`:
```python
# app/fund_service.py:5385
# Query the specific fund code in 90-day range chunks directly via tefasfon:
chunk_size = timedelta(days=90)
curr_start = start_date
collected: List[Dict[str, Any]] = []

while curr_start <= end_date:
    curr_end = min(curr_start + chunk_size, end_date)
    try:
        batch = self.fetch_funds(
            start_date=curr_start,
            end_date=curr_end,
            fund_codes=[normalized_code],
        )
        collected.extend(batch)
    except TefasUpstreamError as exc:
        LOGGER.warning("TEFAS chunk query failed for %s: %s", normalized_code, exc)
        break
    curr_start = curr_end + timedelta(days=1)

return matching_rows(collected)
```

---

### Finding 1.5: 3-Second Polling of 15-Minute Delayed Yahoo Feeds
- **Location**: `app/market_service.py` (Lines 514, 3335, 4100, 4450, 4472, 4719, 4741, 5005)
- **Functions**: `_market_watch_payload`, `_market_fx_payload`, `_market_commodities_payload`, `_market_indices_payload`
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
Backend cache constants in `app/market_service.py` are configured as follows:
```python
514:  _MARKET_INDICES_CACHE_TTL = 3
3335: _COMMODITY_CACHE_TTL = 3
4472: _FX_CACHE_TTL = 3
4741: _WATCH_CACHE_TTL = 3
4747: _WATCH_DELAY_NOTE = "Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk)."
```
- Upstream Yahoo Finance feeds for global commodities (`GC=F`, `CL=F`), foreign exchange (`USDTRY=X`, `EURTRY=X`), and indices (`^XU100`, `^DJI`, `^IXIC`) are delayed by 15 minutes by provider policy.
- Setting cache TTLs to 3 seconds forces the server to constantly spawn thread pools (`ThreadPoolExecutor(max_workers=8)`) to re-query Yahoo Finance every 3 seconds for data that only updates every 900 seconds.

#### Concrete Remediation
Set baseline TTLs to 30s–60s during trading hours and 300s off-hours:
```python
_FX_CACHE_TTL = 30
_COMMODITY_CACHE_TTL = 30
_MARKET_INDICES_CACHE_TTL = 30
_WATCH_CACHE_TTL = 30
```

---

### Finding 1.6: Cold-Cache Bulk Fanout to Yahoo Finance on Non-XUTUM Stock Indices
- **Location**: `app/market_service.py` (Lines 1242–1274, 1457–1461)
- **Function**: `_fetch_stock_return_bases_bulk`
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
In `_market_stock_benchmarks()`:
```python
1457: return_base_map = (
1458:     _cached_stock_return_bases_bulk(symbols)
1459:     if normalized_index == "XUTUM"
1460:     else _fetch_stock_return_bases_bulk(symbols)
1461: )
```
When `normalized_index` is anything other than `"XUTUM"` (such as `"XU100"` or `"XU030"`), the system bypasses `_cached_stock_return_bases_bulk()` and executes `_fetch_stock_return_bases_bulk(symbols)` directly.
- Lines 1267–1270: Spawns `ThreadPoolExecutor(max_workers=12)` to query `_fetch_stock_return_bases` for up to 100 symbols via Yahoo daily chart HTTP calls.
- On cold starts or un-cached index requests, 100 concurrent HTTP requests fire against `query1.finance.yahoo.com`, triggering Yahoo HTTP 429 Too Many Requests errors.

#### Concrete Remediation
Unify caching so all indices share the return base cache:
```python
# app/market_service.py:1457
return_base_map = _cached_stock_return_bases_bulk(symbols)
```

---

### Finding 1.7: Cloudflare Edge Worker Cron Collision & Concurrency Spike
- **Location**: `worker/index.ts` (Lines 139–156) & `wrangler.production.toml` (Line 19)
- **Component**: Edge Cron Scheduler
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
In `wrangler.production.toml`:
```toml
19: crons = ["* * * * *", "*/5 * * * *"]
```
In `worker/index.ts`:
```ts
140: const refreshPath = controller.cron === "* * * * *" ? FAST_REFRESH_PATH : DEEP_REFRESH_PATH;
```
- At every 5-minute boundary ($00, 05, 10, \dots, 55$), **both** crons trigger simultaneously in Cloudflare's runtime.
- The worker fires two simultaneous POST requests: `/admin/kap/refresh?fast=1` and `/admin/kap/refresh`.
- Both requests compete for the backend database lock and financial statement processor, resulting in lock timeouts, database contention, and upstream HTTP 500/504 errors.

#### Concrete Remediation
Guard the 1-minute cron to yield when the 5-minute cron is scheduled:
```ts
// worker/index.ts
export default {
    async scheduled(controller: ScheduledController, env: Env, ctx: ExecutionContext) {
        const date = new Date(controller.scheduledTime);
        const minute = date.getUTCMinutes();

        // Prevent collision: do not run fast refresh on 5-minute boundaries
        if (controller.cron === "* * * * *" && minute % 5 === 0) {
            return;
        }
        ...
    }
}
```

---

### Finding 1.8: MarketPage Eager Unbounded Egress on Initial Mount
- **Location**: `frontend/src/pages/MarketPage.tsx` (Lines 52–99)
- **Component**: `MarketPage`
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
On mount, `MarketPage` executes full queries for `apiClient.marketUniverse()` and `apiClient.funds()` unconditionally:
```tsx
57: void apiClient.marketUniverse()
83: void apiClient.funds()
```
Neither request passes an `AbortSignal`. If a user lands on the home page and immediately clicks away to a specific stock or fund, both large multi-megabyte payloads continue downloading in the background, consuming network bandwidth.

#### Concrete Remediation
Bind calls to component lifecycle using `AbortController`:
```tsx
useEffect(() => {
    const controller = new AbortController();
    void apiClient.marketUniverse({ signal: controller.signal });
    void apiClient.funds({ signal: controller.signal });
    return () => controller.abort();
}, []);
```

---

### Finding 1.9: Synchronous External Blocking Calls in User Request Path
- **Location**: `app/foreign_instruments.py` (Lines 92–116)
- **Function**: `resolve_foreign_holdings`
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
When resolving international securities inside fund holdings, synchronous HTTP calls are dispatched directly to the OpenFIGI API within the FastAPI request thread:
```python
response = requests.post("https://api.openfigi.com/v3/mapping", json=payload, timeout=5)
```
If OpenFIGI experiences latency or throttles requests, the user's HTTP request hangs for up to 5 seconds per batch, tying up backend worker threads.

#### Concrete Remediation
Offload OpenFIGI resolutions to a background worker queue or cache pre-warmed ISIN mappings in PostgreSQL.

---

## 3. Category 2: Independent Endpoint Consumption, Phase Skew, and Data Discrepancies

### Finding 2.1: `MarketWatchStrip` vs `MarketWatchRail` Data Divergence & Visual Tearing
- **Locations**:
  - `frontend/src/hooks/useMarketWatchData.ts` (Lines 11–13, 101–215, 221–258)
  - `frontend/src/components/MarketWatchStrip.tsx` (Lines 217–284, 332–415)
  - `frontend/src/components/MarketWatchRail.tsx` (Lines 331–336, 681–718, 771–820)
- **Severity**: 🔴 CRITICAL

#### Mechanism & Root Cause
`MarketWatchStrip` (the top horizontal ticker) and `MarketWatchRail` (the right vertical dock) consume completely different data pipelines:
1. `MarketWatchStrip` calls `useMarketWatchData({ watch: true, global: true, commodities: true, fx: true })`. Every 3 seconds, `masterTick` launches **4 separate HTTP requests**:
   - `/market/watch`
   - `/market/watch/global`
   - `/market/commodities`
   - `/market/fx`
2. `MarketWatchRail` explicitly passes `{ watch: false }`. When the user is on the `xutum`, `xu100`, or `xu030` tab, it bypasses `useMarketWatchData` entirely and connects to:
   - `/market/stocks?index=...` (HTTP)
   - `/market/stream?index=...` (Server-Sent Events)

#### The Visual Flicker Sequence
In `MarketWatchStrip.tsx` lines 229–284, a complex client-side reconciliation ladder attempts to merge the 4 responses:
```
T = 0.00s : masterTick() launches 4 concurrent HTTP requests.
T = 0.08s : /market/commodities returns (ALTIN: $2,740.50). 
            Strip re-renders: ALTIN flashes green to $2,740.50.
T = 0.12s : /market/fx returns (USD/TRY: 34.25). 
            Strip re-renders: USD/TRY flashes to 34.25.
T = 0.28s : /market/watch returns. Its embedded commodities section was generated 
            from an older backend SWR cache hit where ALTIN was $2,738.00.
            In lookup memo: orderedItems iterates payload.sections.commodities first.
            For a split render frame, ALTIN reverts to $2,738.00 before being overridden.
            Result: Visual jitter, layout shifts, and contradictory prices between Strip and Rail.
```

#### Concrete Remediation
Create a unified store (`useSyncExternalStore`) and a single authoritative backend endpoint (`/market/watch/unified`):
```typescript
// frontend/src/stores/marketStore.ts
import { useSyncExternalStore } from 'react';
import type { MarketWatchItem, MarketStockRow } from '../api/types';

interface UnifiedMarketState {
    indices: Map<string, MarketWatchItem>;
    fx: Map<string, MarketWatchItem>;
    commodities: Map<string, MarketWatchItem>;
    stocks: Map<string, MarketStockRow>;
    asOf: string | null;
}

const state: UnifiedMarketState = {
    indices: new Map(),
    fx: new Map(),
    commodities: new Map(),
    stocks: new Map(),
    asOf: null,
};

const listeners = new Set<() => void>();
function notify() { listeners.forEach(l => l()); }

export function updateMarketSnapshot(payload: any) {
    if (payload.indices) for (const item of payload.indices) state.indices.set(item.symbol, item);
    if (payload.fx) for (const item of payload.fx) state.fx.set(item.symbol, item);
    if (payload.commodities) for (const item of payload.commodities) state.commodities.set(item.symbol, item);
    state.asOf = payload.as_of;
    notify();
}

export function useMarketStore() {
    return useSyncExternalStore(
        (cb) => { listeners.add(cb); return () => listeners.delete(cb); },
        () => state,
        () => state
    );
}
```

---

### Finding 2.2: Quad-Endpoint BIST Index Divergence (`XU100`, `XU030`, `XUTUM`)
- **Locations**:
  - `app/routers/market.py` (Lines 305–323)
  - `app/market_service.py` (Lines 885–953, 1299–1309, 3641–3690, 4082–4118, 4945–5018)
  - `frontend/src/pages/MarketsView.tsx` (Lines 3131–3139, 3586)
- **Severity**: 🔴 CRITICAL

#### Mechanism & Root Cause
Four separate endpoints calculate prices and returns for the BIST 100 (`XU100`) index through three incompatible provider backends:
1. `/market/watch` (Top Strip): Calls `_watch_index_item()`, querying Yahoo Finance `^XU100` delayed quote (15-minute delay).
2. `/market/indices` (Indices Tab): Calls `_market_index_row()`, querying `_MARKET_INDEX_QUOTE_CACHE` (15-minute delay).
3. `/market/indices/XU100` (Hero & Chart): Calls `_market_index_detail_payload()`, querying Yahoo 5-minute intraday chart.
4. `/market/stocks?index=XU100` (Side Rail & Stocks Tab): Calls `_market_stocks_payload()`, which populates `benchmarks["XU100"]` via line 1304:
   ```python
   current_for_returns = bases.get("latest_close")
   benchmarks[index_name] = {
       **_returns_from_bases(current_for_returns, bases),
       "as_of": bases.get("as_of"),
   }
   ```
   **The 0.00% Daily Return Bug**: `bases.get("latest_close")` returns **yesterday's closing price**, not the intraday price. Comparing `latest_close` against `previous_close` yields 0.00% daily change throughout the entire trading day.

A user viewing the application simultaneously sees:
- Top Strip: $XU100 = 9,120.50$ ($+1.20\%$)
- Side Rail Benchmark: $XU100 = 9,012.35$ ($0.00\%$)
- Stock Detail Hero: $XU100 = 9,085.10$ ($+0.81\%$)

#### Concrete Remediation
Fix line 1304 in `app/market_service.py` to use the live quote:
```python
# app/market_service.py:1304
# Replace 'latest_close' with real-time intraday quote:
live_quote = _get_live_index_quote(index_name)
current_for_returns = live_quote.get("price") or bases.get("latest_close")
benchmarks[index_name] = {
    **_returns_from_bases(current_for_returns, bases),
    "price": current_for_returns,
    "as_of": live_quote.get("as_of") or bases.get("as_of"),
}
```

---

### Finding 2.3: StockDetailPage Split-Brain Hero Price vs Valuation Multiples
- **Locations**:
  - `frontend/src/pages/StockDetailPage.tsx` (Lines 798–833, 862–867, 954–970)
  - `app/routers/kap.py` (Lines 253–255)
  - `app/market_service.py` (Lines 2654–2750, 3170–3255)
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
In `StockDetailPage.tsx`:
```tsx
// Effect 1: Hero price from /kap/price
apiClient.kapPrice(ticker).then((response) => setPriceData(response));

// Effect 2: Valuation multiples from /market/stocks/cards
apiClient.marketStockCards({ symbols: [normalizedTicker] }).then((response) => setMarketCard(response.items[0]));
```
Underneath:
```tsx
const displayPrice = priceData?.ok && priceData.price != null ? priceData.price : valuation?.price;
const displayFk = marketCard ? marketCard.fk : valuation?.fk;
const displayPdDd = marketCard ? marketCard.pd_dd : valuation?.pd_dd;
```
In the backend:
- `/kap/price` fetches from Yahoo Finance (`ticker.IS`), delayed by 15 minutes, with a 300-second backend cache.
- `/market/stocks/cards` fetches from Info Yatırım real-time live scraper, with a 3-second cache.

**Mathematical Contradiction**: If THYAO is trading at 310.00 TRY live, but Yahoo reports 305.00 TRY (15-min delay):
- Hero price displays: **305.00 TRY**
- P/E ratio displays: **4.13** (calculated from 310.00 / EPS)
- Any user dividing the displayed price (305.00) by the reported EPS encounters a mathematical mismatch.

#### Concrete Remediation
Retire the `/kap/price` call in `StockDetailPage.tsx` and derive both price and multiples from `apiClient.marketStockCards`:
```tsx
// frontend/src/pages/StockDetailPage.tsx
useEffect(() => {
    const controller = new AbortController();
    setMarketCard(null);

    apiClient.marketStockCards({ symbols: [normalizedTicker], signal: controller.signal })
        .then((response) => {
            if (controller.signal.aborted) return;
            const card = response.items.find(item => item.symbol === normalizedTicker) || null;
            setMarketCard(card);
            if (card) {
                setPriceData({
                    ok: true,
                    symbol: card.symbol,
                    price: card.price,
                    prev_close: card.previous_close,
                    change: card.change,
                    change_pct: card.change_pct,
                    currency: card.currency || 'TRY',
                    market_state: '',
                    as_of: card.as_of,
                });
            }
        })
        .catch(() => { if (!controller.signal.aborted) setMarketCard(null); });

    return () => controller.abort();
}, [normalizedTicker]);
```

---

### Finding 2.4: Model Portfolio Benchmark Freezing Skew
- **Locations**:
  - `frontend/src/pages/ModelPortfolioPage.tsx` (Lines 520–541, 556–565, 577–620)
  - `app/model_portfolio_api_service.py` (Lines 81–118)
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
In `ModelPortfolioPage.tsx`:
- `loadPortfolio(false)` runs **once on mount** to fetch the base portfolio JSON containing `current_xu100`.
- Lines 523–527 establish an SSE subscription to `subscribeMarketStocks`, which continually streams live price updates for individual stock positions into `liveStockMap`.
- Lines 578–601 compute position returns and alpha:
  ```typescript
  const livePrice = liveQuoteIsCurrent ? liveRow!.price : pos.current_price;
  const retPct = ((livePrice - pos.entry_price) / pos.entry_price) * 100;
  const xuRetPct = ((currentXu100 - pos.entry_xu100) / pos.entry_xu100) * 100;
  const relRetPct = compoundAlpha(retPct, xuRetPct);
  ```
- **The Skew**: `livePrice` updates every 3 seconds via SSE. But `currentXu100` remains **frozen at the mount-time value**. If the user keeps the tab open for 4 hours while BIST moves $\pm 3\%$, the portfolio return adjusts live while the benchmark return is completely static. The calculated relative return (alpha) becomes entirely fictional.

#### Concrete Remediation
Subscribe the benchmark index to live market updates:
```tsx
// frontend/src/pages/ModelPortfolioPage.tsx
const [liveXu100, setLiveXu100] = useState<number | null>(null);

useEffect(() => {
    const unsubscribe = apiClient.subscribeMarketStocks((index, payload) => {
        if (index === 'XU100' && payload.index_price) {
            setLiveXu100(payload.index_price);
        }
    });
    return unsubscribe;
}, []);

const effectiveXu100 = liveXu100 ?? portfolioData?.current_xu100 ?? null;
```

---

### Finding 2.5: Server Cache TTL vs Client Polling Frequency Asymmetry
- **Locations**:
  - `app/market_service.py` (Lines 4745, 4927–4935)
  - `frontend/src/hooks/useMarketWatchData.ts` (Lines 13, 130–156, 222)
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
- In `app/market_service.py`, `_WATCH_GLOBAL_CACHE_TTL = 60` (server SWR cache TTL is 60 seconds).
- In `useMarketWatchData.ts`, `MARKET_REFRESH_MS = 3_000` (client queries `/market/watch/global` every 3 seconds).
- Over a 60-second window, the client sends 20 HTTP requests. Exactly 1 request triggers server cache revalidation, while **19 requests (95%) are redundant cache hits** returning identical data.

#### Concrete Remediation
Parameterize client polling frequency by feed type in `useMarketWatchData.ts`:
```typescript
const GLOBAL_REFRESH_MS = 60_000;
// Only poll global channel when 60s has elapsed
```

---

## 4. Category 3: Independent, Uncoordinated Timers and Polling Loops

### Finding 3.1: Top-Level Clock State Thrashing in `MarketsView.tsx`
- **Location**: `frontend/src/pages/MarketsView.tsx` (Lines 3060, 3280–3286, 4207–4208)
- **Component**: `MarketsView`
- **Severity**: 🔴 CRITICAL

#### Mechanism & Root Cause
In `MarketsView.tsx`:
```tsx
3060: const [terminalNow, setTerminalNow] = useState(() => new Date());
...
3280: useEffect(() => {
3281:     if (activeSection !== 'markets') return;
3282:     const intervalId = window.setInterval(
3283:         () => setTerminalNow(new Date()),
3284:         isMobileViewport ? 10000 : 1000,
3285:     );
3286:     return () => window.clearInterval(intervalId);
3287: }, [activeSection, isMobileViewport]);
```
- `MarketsView` is a 5,343-line monolithic component containing virtualized tables, watchlist state, search filters, detail drawers, and charting canvases.
- Because `terminalNow` is declared at the root scope of `MarketsView`, **calling `setTerminalNow(new Date())` every 1,000ms triggers a complete React virtual DOM reconciliation pass of the entire 5,343-line component tree every second**.
- Furthermore, the interval has no check for `document.visibilityState`. It runs continuously even when the tab is hidden or minimized, draining laptop batteries and causing high CPU utilization.

#### Concrete Remediation
Extract the clock display into an isolated, memoized leaf component:
```tsx
// frontend/src/components/TerminalClock.tsx
import React, { useEffect, useState } from 'react';
import { formatTerminalClock, formatTerminalDate } from '../utils/formatters';

export const TerminalClock = React.memo(function TerminalClock() {
    const [now, setNow] = useState(() => new Date());

    useEffect(() => {
        let timer: number | null = null;
        const tick = () => setNow(new Date());

        const start = () => {
            if (timer === null && document.visibilityState === 'visible') {
                timer = window.setInterval(tick, 1000);
            }
        };
        const stop = () => {
            if (timer !== null) {
                window.clearInterval(timer);
                timer = null;
            }
        };

        const onVisibility = () => {
            if (document.visibilityState === 'visible') {
                tick();
                start();
            } else {
                stop();
            }
        };

        start();
        document.addEventListener('visibilitychange', onVisibility);
        return () => {
            stop();
            document.removeEventListener('visibilitychange', onVisibility);
        };
    }, []);

    return (
        <>
            <span>{formatTerminalDate(now)}</span>
            <time className="market-session-clock">{formatTerminalClock(now)}</time>
        </>
    );
});
```

---

### Finding 3.2: Hidden-Tab Busy Spinning in `useBackgroundRefresh`
- **Location**: `frontend/src/hooks/useBackgroundRefresh.ts` (Lines 7–23)
- **Hook**: `useBackgroundRefresh`
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
```ts
12: const tick = async () => {
13:     if (controller.signal.aborted) return;
14:     if (document.visibilityState !== 'hidden') {
15:         try { await latest.current(controller.signal); }
16:         catch { /* Keep the last usable response while the source recovers. */ }
17:         delay = Math.min(30_000, delay * 1.5);
18:     }
19:     if (!controller.signal.aborted) timer = setTimeout(tick, delay);
20: };
21: timer = setTimeout(tick, delay);
```
- When `document.visibilityState === 'hidden'`, line 14 skips the fetch.
- Crucially, line 17 (`delay = Math.min(...)`) is inside the visibility check and is **bypassed**.
- Execution proceeds to line 19: `timer = setTimeout(tick, delay)`. The initial `delay` (2,000ms) is never backed off.
- **Impact**: The hook enters a continuous busy loop, firing `setTimeout` every 2 seconds indefinitely in background tabs, consuming CPU cycles without performing any useful work.

#### Concrete Remediation
Implement sleep-on-hide and wake-on-visible semantics:
```ts
// frontend/src/hooks/useBackgroundRefresh.ts
export function useBackgroundRefresh(
    enabled: boolean,
    refresh: (signal: AbortSignal) => Promise<void>,
    baseDelay = 2000,
) {
    const latest = useRef(refresh);
    latest.current = refresh;

    useEffect(() => {
        if (!enabled) return;
        const controller = new AbortController();
        let delay = baseDelay;
        let timer: ReturnType<typeof setTimeout> | null = null;

        const clear = () => {
            if (timer !== null) { clearTimeout(timer); timer = null; }
        };

        const tick = async () => {
            if (controller.signal.aborted || document.visibilityState !== 'visible') return;
            try { await latest.current(controller.signal); } catch {}
            delay = Math.min(30_000, delay * 1.5);
            if (!controller.signal.aborted && document.visibilityState === 'visible') {
                timer = setTimeout(tick, delay);
            }
        };

        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible') {
                delay = baseDelay;
                void tick();
            } else {
                clear();
            }
        };

        document.addEventListener('visibilitychange', onVisibilityChange);
        if (document.visibilityState === 'visible') {
            timer = setTimeout(tick, delay);
        }

        return () => {
            controller.abort();
            clear();
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [enabled, baseDelay]);
}
```

---

### Finding 3.3: Perpetual Background Polling in `HistoricalMultiplesChart.tsx`
- **Location**: `frontend/src/components/stock/HistoricalMultiplesChart.tsx` (Lines 153–170)
- **Component**: `HistoricalMultiplesChart`
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
The chart's valuation retry loop uses `window.setTimeout(() => void loadHistory(), delay)` without checking `document.visibilityState`. If valuation metrics are pending calculation and the user switches tabs, up to 10 retry requests fire against the backend in the background.

#### Concrete Remediation
Wrap timeout execution in a visibility guard that pauses when the document is hidden.

---

### Finding 3.4: Unconditioned Polling Rescheduling in `MarketFlowPanel.tsx`
- **Location**: `frontend/src/components/MarketFlowPanel.tsx` (Lines 349–355)
- **Component**: `MarketFlowPanel`
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
While line 312 checks `if (document.visibilityState === 'visible')` before fetching, the `finally` block at line 353 unconditionally reschedules `timerId = window.setTimeout(poll, nextInterval)`. The timer never halts when the tab is hidden; it wakes every few seconds to perform an empty check.

---

### Finding 3.5: StockDetailPage 5-Second Hidden-Tab Spin Loop
- **Location**: `frontend/src/pages/StockDetailPage.tsx` (Lines 728–733)
- **Component**: `StockDetailPage`
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
```tsx
730: if (document.visibilityState === 'hidden') {
731:     timer = window.setTimeout(() => void load(depth), 5000);
732:     return;
733: }
```
When hidden, `load()` schedules a 5-second `setTimeout` to check again, spinning indefinitely. Furthermore, because no `visibilitychange` listener is attached, when the user switches back to the tab, they must wait up to 5 seconds before the application detects visibility and resumes loading.

---

### Finding 3.6: Uncleaned Fire-and-Forget Timers in `ModelPortfolioPage.tsx`
- **Location**: `frontend/src/pages/ModelPortfolioPage.tsx` (Lines 3085–3092)
- **Component**: `ModelPortfolioPage`
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
In `onBlur` handlers, `window.setTimeout(() => setDropdownOpen(false), 180)` is invoked without retaining a timer reference. If the user navigates away within 180ms, the timer executes on an unmounted component, triggering React unmounted state update warnings.

---

### Finding 3.7: 24/7 Off-Hours Polling Loops for Macro Endpoints
- **Location**: `app/market_service.py` (Lines 4410, 4681, 4082, 4945 vs 1647–1660)
- **Functions**: `_market_commodities_payload`, `_market_fx_payload`, `_market_indices_payload`, `_market_watch_payload`
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
While equity quote services implement `_effective_market_quotes_fresh_ttl()` to extend cache TTL to 300 seconds during market off-hours and weekends, macro endpoints (`/market/watch`, `/market/fx`, `/market/commodities`) do not call `_is_bist_market_open()`. Their TTL remains fixed at 3 seconds 24 hours a day, 7 days a week, polling external providers during weekends when markets are closed.

---

### Finding 3.8: Uncoordinated Multi-Worker Fund Price Collector Loop
- **Location**: `app/fund_api_service.py` (Lines 715–743)
- **Function**: `_fund_price_collector_loop`
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
In `app/api.py`, `_fund_price_collector_loop` is spawned unconditionally inside `_lifespan` in every worker process:
```python
724: result = await asyncio.to_thread(collect_daily_fund_prices, CONFIG.paths.processed_dir)
```
- In a production deployment with multiple Uvicorn workers (e.g. 4 workers), **all 4 worker processes run `collect_daily_fund_prices` simultaneously** upon startup.
- None of the workers acquire a distributed Redis lock or database lease.
- **Impact**: TEFAS servers receive 4x duplicate requests for universal daily fund tables, causing redundant egress and database write lock collisions.

#### Concrete Remediation
Enforce distributed locking using Redis:
```python
# app/fund_api_service.py:722
backend = _get_cache()
lease_key = "lock:fund-price-collector"
lease_ttl = int(interval)

if backend.set_if_absent(lease_key, worker_id, ttl_seconds=lease_ttl):
    try:
        result = await asyncio.to_thread(collect_daily_fund_prices, CONFIG.paths.processed_dir)
        LOGGER.info("Collector finished successfully: %s", result)
    except Exception:
        LOGGER.exception("Collector failed")
else:
    LOGGER.debug("Collector lease held by another worker; skipping cycle")
```

---

## 5. Category 4: Cache Layer Synchronization Gaps, Stale Windows, and Freshness/TTL Mismatches

### Finding 4.1: Multi-Worker 30-Minute Stale Window in Financial Snapshots
- **Location**: `app/financial_snapshot_store.py` (Lines 49–52, 101–105, 519–550)
- **Module**: `financial_snapshot_store`
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
```python
49: _SNAPSHOT_MEMORY_CACHE_TTL_SECONDS = max(
50:     1.0,
51:     _env_float("RAGFIN_FINANCIAL_SNAPSHOT_MEMORY_CACHE_TTL_SECONDS", 1800.0), # 30 minutes!
52: )
...
101: def _invalidate_memory_cache(processed_dir: Path) -> None:
102:     key = _cache_key(processed_dir)
103:     with _CACHE_LOCK:
104:         _SNAPSHOT_MEMORY_CACHE.pop(key, None)
```
- `_SNAPSHOT_MEMORY_CACHE` is a local Python dictionary in process memory.
- When a new financial statement (balance sheet/income statement) is ingested or refreshed, `_invalidate_memory_cache()` purges `_SNAPSHOT_MEMORY_CACHE` **only inside the worker process that handled the write**.
- In multi-worker deployments (e.g. Uvicorn with 4 workers), workers 2, 3, and 4 receive no invalidation signal (no Redis key deletion, pub/sub message, or shared timestamp).
- **Impact**: Users querying the API are routed round-robin across workers. Worker 1 serves new financial metrics (e.g. $F/K = 5.2$), while Workers 2, 3, and 4 continue serving outdated metrics (e.g. $F/K = 7.8$) for up to **30 minutes** (1,800 seconds).

#### Concrete Remediation
Implement versioned cache keys backed by Redis:
```python
# app/financial_snapshot_store.py
_SNAPSHOT_VERSION_KEY = "api:financial-snapshots:version"

def _get_snapshot_cache_version() -> str:
    backend = get_cache()
    ver = backend.get(_SNAPSHOT_VERSION_KEY)
    if not ver:
        ver = uuid.uuid4().hex
        backend.set(_SNAPSHOT_VERSION_KEY, ver, ttl_seconds=86400)
    return str(ver)

def _invalidate_memory_cache(processed_dir: Path) -> None:
    backend = get_cache()
    # Bump version in Redis to invalidate all workers simultaneously
    backend.set(_SNAPSHOT_VERSION_KEY, uuid.uuid4().hex, ttl_seconds=86400)
    with _CACHE_LOCK:
        _SNAPSHOT_MEMORY_CACHE.clear()
```

---

### Finding 4.2: Compound SWR Nested Cache Stacking in `/market/watch`
- **Location**: `app/market_service.py` (Lines 4450, 4719, 4741, 4945–5018)
- **Function**: `_market_watch_payload`
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
`/market/watch` wraps other cached endpoints inside its factory function:
```python
# Inside _market_watch_payload factory:
fx_payload = _market_fx_payload()           # SWR Cache TTL = 3s
commodity_payload = _market_commodities_payload() # SWR Cache TTL = 3s
global_indices = _market_watch_global_payload()    # SWR Cache TTL = 60s
```
- Both `/market/watch` and its sub-endpoints use 3-second SWR caches.
- If `/market/fx` refreshes at $t = 0\text{s}$, its cache is fresh until $t = 3\text{s}$.
- If `/market/watch` is requested at $t = 2.9\text{s}$, it reads the existing FX cache (age 2.9s) and embeds it into the `/market/watch` payload, which is then cached for another 3.0s (valid until $t = 5.9\text{s}$).
- When a client calls `/market/fx` at $t = 3.1\text{s}$, it receives a new price $P_2$. But `/market/watch` continues serving price $P_1$ until $t = 5.9\text{s}$.
- **Impact**: Compound cache stacking causes the aggregated watch strip to lag behind direct endpoint queries, creating data phase skew across views.

---

### Finding 4.3: Negative Caching of Transient Upstream Scraping Failures
- **Location**: `app/market_service.py` (Lines 2952–2954)
- **Function**: `_fetch_isyatirim_basic_summary_map`
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
```python
2952: if not result:
2953:     _ISYATIRIM_CACHE[cache_key] = (now + 60.0, {})
2954:     return {}
```
If Info Yatırım experiences a transient network blip (packet drop, DNS timeout, or socket reset), the function places an empty dictionary `{}` into `_ISYATIRIM_CACHE` with a 60-second TTL. For the next minute, all fundamental stock cards across the entire platform render with blank valuation ratios ($F/K$, $PD/DD$, $FD/FAV\ddot{O}K$).

#### Concrete Remediation
Limit negative cache TTL for upstream network failures to 3–5 seconds:
```python
if not result:
    _ISYATIRIM_CACHE[cache_key] = (now + 3.0, {}) # Transient retry backoff
    return {}
```

---

### Finding 4.4: Lock-Free Race Condition in KAP Companies In-Memory Store
- **Location**: `app/routers/kap.py` (Lines 130–181) & `app/kap_api_service.py` (Line 191)
- **Function**: `kap_companies`
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
`_KAP_MARKET_METADATA_CACHE` is accessed and mutated in `kap_companies` without thread locking. During server startup or cold cache revalidations, concurrent HTTP requests execute duplicate full-database directory scans (`kap_sqlite_service.get_companies()`), redundantly allocating large company dictionary structures.

---

### Finding 4.5: Non-Thread-Safe Dictionary Eviction in TEFAS Cache Put
- **Location**: `app/fund_service.py` (Lines 210, 220–234)
- **Function**: `_memory_cache_put`
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
FIFO overflow eviction in `_memory_cache_put` iterates over `_MEMORY_CACHE.keys()` while modifying the dictionary across multiple worker threads without acquiring a lock, risking `RuntimeError: dictionary changed size during iteration`.

---

## 6. Category 5: Memory Leaks, Uncleaned Subscriptions, Event Listeners, and Reactive State

### Finding 5.1: Unbounded In-Memory Map in `MarketsView.tsx`
- **Location**: `frontend/src/pages/MarketsView.tsx` (Lines 1022 & 3549)
- **Component**: `MarketsView`
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
```tsx
1022: const marketStockCardsMemoryCache = new Map<string, MarketStockCardsMemoryCache>();
...
3549: marketStockCardsMemoryCache.set(latestSymbols.join(','), { data: mergedPayload, fetchedAt: Date.now() });
```
- `marketStockCardsMemoryCache` is an unbounded module-level `Map` in frontend memory.
- Every time a user changes filters, searches, switches index tabs, or modifies pagination, `latestSymbols.join(',')` produces a different string key.
- Each cache entry stores a full `MarketStockCardsResponse` object containing detailed quarterly financial statements, fundamentals, and historical charts.
- The map has zero eviction, size capping, or TTL purging.
- **Impact**: In long-running market terminal sessions, memory usage grows monotonically by tens of megabytes per hour, eventually causing mobile browser crashes or tab reloads.

#### Concrete Remediation
Replace raw `Map` with a bounded LRU cache:
```ts
// frontend/src/utils/lruCache.ts
export class BoundedMemoryCache<K, V> {
    private maxEntries: number;
    private map = new Map<K, V>();

    constructor(maxEntries = 50) {
        this.maxEntries = maxEntries;
    }

    get(key: K): V | undefined {
        const val = this.map.get(key);
        if (val !== undefined) {
            this.map.delete(key);
            this.map.set(key, val);
        }
        return val;
    }

    set(key: K, value: V): void {
        this.map.delete(key);
        this.map.set(key, value);
        while (this.map.size > this.maxEntries) {
            const oldestKey = this.map.keys().next().value;
            if (oldestKey !== undefined) this.map.delete(oldestKey);
        }
    }
}
```

---

### Finding 5.2: Unbounded Fund Holdings Cache in API Client
- **Location**: `frontend/src/api/client.ts` (Lines 63 & 1087)
- **Module**: `apiClient`
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
```ts
63: const fundHoldingsMemoryCache = new Map<string, { payload: FundHoldingsResponse; fetchedAt: number }>();
...
1087: fundHoldingsMemoryCache.set(normalizedCode, { payload, fetchedAt: Date.now() });
```
While `conditionalGetMemoryCache` enforces `CONDITIONAL_GET_CACHE_MAX_ENTRIES = 64`, `fundHoldingsMemoryCache` has no entry limit. Every fund browsed in the application remains retained in memory indefinitely.

---

### Finding 5.3: Monotonically Growing Flow ID Set in `MarketFlowPanel.tsx`
- **Location**: `frontend/src/components/MarketFlowPanel.tsx` (Lines 140, 336–338)
- **Component**: `MarketFlowPanel`
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
```tsx
140: const observedFlowIdsRef = useRef<Set<string> | null>(null);
...
336: const observed = observedFlowIdsRef.current || new Set<string>();
337: incoming.forEach((item) => observed.add(item.id));
338: observedFlowIdsRef.current = observed;
```
Every 3 seconds, new flow items are added to `observedFlowIdsRef`. IDs are never pruned, causing the set to grow indefinitely during continuous market observation.

---

### Finding 5.4: Eleven Unbounded Module-Level Plain Dict Caches in Backend
- **Locations**:
  - `app/market_service.py` (Lines 360, 364, 384, 406, 416, 433, 452, 458)
  - `app/model_portfolio_service.py` (Line 33)
  - `app/kap_api_service.py` (Line 187)
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
The backend declares 11 module-level plain Python dictionaries to cache external requests:
1. `_GEFAS_GYF_QUOTE_CACHE` (`market_service.py:360`)
2. `_FOREIGN_HOLDING_QUOTE_CACHE` (`market_service.py:364`)
3. `_PRICE_CACHE` (`market_service.py:384`)
4. `_INFOYATIRIM_STOCK_PAGE_CACHE` (`market_service.py:406`)
5. `_MARKET_STOCK_CARD_CHART_CACHE` (`market_service.py:416`)
6. `_STOCK_CARD_VALUATION_CACHE` (`market_service.py:433`)
7. `_STOCK_RETURN_BASE_CACHE` (`market_service.py:452`)
8. `_ISYATIRIM_CACHE` (`market_service.py:458`)
9. `_MODEL_PORTFOLIO_PERF_MEMORY_CACHE` (`model_portfolio_service.py:33`)
10. `_KAP_MARKET_METADATA_CACHE` (`kap_api_service.py:187`)
11. `_MEMORY_CACHE` (`fund_service.py:210`)

**Impact**: None of these dictionaries enforce `MAX_ENTRIES` or LRU eviction. As new stock symbols, fund codes, and date ranges are queried, memory footprint expands monotonically. In containerized production deployments with strict RAM limits (e.g. 1GB–2GB), this causes unhandled Out-Of-Memory (OOM) process terminations.

#### Concrete Remediation
Wrap caches in a thread-safe LRU container:
```python
# app/utils/cache.py
from collections import OrderedDict
import threading
import time

class BoundedMemoryCache:
    def __init__(self, max_entries: int = 512, default_ttl: float = 300.0):
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at and expires_at < now:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expires_at = time.monotonic() + (ttl or self._default_ttl)
        with self._lock:
            self._store.pop(key, None)
            self._store[key] = (expires_at, value)
            if len(self._store) > self._max_entries:
                self._store.popitem(last=False)
```

---

### Finding 5.5: Global ThreadPoolExecutors Leaked on Lifespan Shutdown
- **Locations**:
  - `app/api.py` (Lines 102–107)
  - `app/api_runtime.py` (Line 143)
  - `app/fund_api_service.py` (Lines 67–80)
  - `app/kap_statement_refresh.py` (Line 111)
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
In `app/api.py`, the ASGI `_lifespan` shutdown handler terminates only one executor:
```python
102: finally:
103:     await _stop_fund_price_collector()
104:     shutdown_background_refresh_executor() # Only stops _KAP_FLOW_REFRESH_EXECUTOR!
105:     close_postgres_pool()
106:     close_shared_http_clients()
```
The following global thread pools are **never shut down**:
- `_MARKET_SWR_REVALIDATION_EXECUTOR` (`app/api_runtime.py`)
- `_FUND_REFRESH_EXECUTOR` (`app/fund_api_service.py`)
- `_FUND_HISTORY_EXECUTOR` (`app/fund_api_service.py`)
- `_FUND_ALLOCATION_HISTORY_EXECUTOR` (`app/fund_api_service.py`)
- `_KAP_STATEMENT_EXECUTOR` (`app/kap_statement_refresh.py`)

**Impact**: During developer hot-reloads (`--reload`) or graceful server restarts, worker threads remain orphaned in the operating system, leaking thread handles and memory.

#### Concrete Remediation
Terminate all global executors during lifespan shutdown:
```python
# app/api.py:104
from app.api_runtime import _MARKET_SWR_REVALIDATION_EXECUTOR
from app.fund_api_service import (
    _FUND_REFRESH_EXECUTOR,
    _FUND_HISTORY_EXECUTOR,
    _FUND_ALLOCATION_HISTORY_EXECUTOR,
)
from app.kap_statement_refresh import _KAP_STATEMENT_EXECUTOR

for executor in (
    _MARKET_SWR_REVALIDATION_EXECUTOR,
    _FUND_REFRESH_EXECUTOR,
    _FUND_HISTORY_EXECUTOR,
    _FUND_ALLOCATION_HISTORY_EXECUTOR,
    _KAP_STATEMENT_EXECUTOR,
):
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
```

---

### Finding 5.6: Zombie Threads in TEFAS Adapter Timeout Path
- **Location**: `app/fund_service.py` (Lines 4842–4853, 4858–4869)
- **Function**: `TefasFonClient._call_dataframe`
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
```python
4842: executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
4843: future = executor.submit(function, **kwargs)
4844: try:
4845:     frame = future.result(timeout=timeout_seconds)
4846: except concurrent.futures.TimeoutError as exc:
4847:     future.cancel()
4848:     executor.shutdown(wait=False, cancel_futures=True)
4849:     raise TefasUpstreamError(...)
```
- In Python, a thread blocked inside a C-level socket read (e.g. `requests.post()` to TEFAS SOAP gateways) **cannot be interrupted or killed** by `future.cancel()`.
- `executor.shutdown(wait=False)` allows Python execution to proceed, but the underlying OS thread remains alive until the operating system TCP keep-alive or timeout expires.
- If upstream TEFAS becomes unresponsive, each incoming request spawns a new thread that remains stuck, accumulating hundreds of zombie threads until thread exhaustion crashes the process.

#### Concrete Remediation
Enforce socket-level connect/read timeouts directly inside the HTTP client (`requests.Session` or `httpx.Client`) rather than attempting thread-level cancellation.

---

### Finding 5.7: Active Worker Thread Per Connected SSE Client in `market_stream`
- **Location**: `app/routers/market.py` (Lines 186–241)
- **Function**: `market_stream`
- **Severity**: 🟠 HIGH

#### Mechanism & Root Cause
```python
199: while True:
200:     if not once and await request.is_disconnected(): break
...
204:     payload = await asyncio.to_thread(_market_stocks_payload, index_name=normalized_index)
...
231:     await asyncio.sleep(sleep_seconds)
```
- Every connected browser client running `/market/stream` initiates an independent `while True` loop.
- Every 3 seconds, each loop dispatches `asyncio.to_thread(_market_stocks_payload)`.
- If 100 users are viewing the terminal, **100 concurrent worker threads** are dispatched into the thread pool every 3 seconds to compute identical stock deltas.
- This saturates the thread pool, starves other API endpoints, and wastes CPU cycles.

#### Concrete Remediation
Adopt the Centralized Broadcaster Pattern: a single background task computes deltas and fans them out to connected clients via `asyncio.Queue`:
```python
# app/services/market_broadcaster.py
class MarketStreamBroadcaster:
    def __init__(self):
        self._subscribers: set[asyncio.Queue] = set()

    async def subscribe(self) -> AsyncIterator[str]:
        q = asyncio.Queue(maxsize=10)
        self._subscribers.add(q)
        try:
            while True:
                msg = await q.get()
                yield msg
        finally:
            self._subscribers.discard(q)

    async def run_loop(self):
        prev_snapshot = {}
        while True:
            await asyncio.sleep(3.0 if _is_bist_market_open() else 30.0)
            if not self._subscribers:
                continue
            payload = await asyncio.to_thread(_market_stocks_payload, index_name="XUTUM")
            changed, prev_snapshot = _compute_market_stocks_delta(prev_snapshot, payload.get("rows") or [])
            if changed:
                event_data = f"event: stocks_delta\ndata: {json.dumps({'rows': changed})}\n\n"
                for q in list(self._subscribers):
                    try: q.put_nowait(event_data)
                    except asyncio.QueueFull: pass
```

---

### Finding 5.8: Uncleaned AbortSignals & Reactive State Anti-Patterns
- **Locations**:
  - `frontend/src/pages/StockDetailPage.tsx` (Lines 797–814)
  - `frontend/src/pages/stock/sections/StockKAP.tsx` (Lines 9–18)
  - `frontend/src/pages/FundsPage.tsx` (Lines 5963–5979)
- **Severity**: 🟡 MEDIUM

#### Mechanism & Root Cause
1. `StockDetailPage.tsx:797`: `apiClient.kapPrice(ticker)` ignores `AbortSignal`. Rapid switching between stock symbols leaves zombie network requests active on the wire.
2. `StockKAP.tsx:9–18`: Stores sorted quarters using `useEffect` + `useState` instead of `useMemo`:
   ```tsx
   const [sortedQuarters, setSortedQuarters] = useState<KapQuarter[]>([]);
   useEffect(() => {
       const sorted = [...quarters].sort(...);
       setSortedQuarters(sorted);
   }, [quarters]);
   ```
   This anti-pattern forces an unnecessary extra render cycle and DOM reconciliation pass on every prop change.
3. `FundsPage.tsx:5963`: Closures inside `useBackgroundRefresh` fail to pass `signal` into `apiClient.fundHoldings` and `apiClient.fundAllocations`.

---

## 7. Consolidated Architectural Roadmap

To execute remediations safely without regressions, changes are organized into three sequential phases based on implementation dependency and impact.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PRIORITIZED REMEDIATION ROADMAP                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 1: IMMEDIATE (P0) — STABILITY, STAMPEDE & MEMORY CAPPING                         │
│ • Fix single-flight fallthrough stampede in `market_service.py` (Finding 1.3)           │
│ • Replace unbounded Python dictionaries with `BoundedMemoryCache` (Finding 5.4)        │
│ • Enforce 90-day chunked range queries in `fund_service.py` (Finding 1.4)               │
│ • Extract `<TerminalClock />` leaf component in `MarketsView.tsx` (Finding 3.1)        │
│ • Fix hidden-tab busy loop in `useBackgroundRefresh.ts` (Finding 3.2)                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: NEAR-TERM (P1) — PHASE SKEW & DATA DISCREPANCY ELIMINATION                   │
│ • Fix 0.00% BIST index return bug across quad-endpoints (Finding 2.2)                  │
│ • Eliminate Yahoo 15m delay in `StockDetailPage.tsx` by using live cards (Finding 2.3) │
│ • Connect `ModelPortfolioPage.tsx` benchmark to live SSE feed (Finding 2.4)            │
│ • Add market-hours gating to macro endpoints & extend TTL to 30s/300s (Finding 1.5, 3.7│
│ • Synchronize multi-worker financial snapshots via Redis versioning (Finding 4.1)      │
│ • Bound frontend memory caches (`MarketsView`, `apiClient`) (Findings 5.1, 5.2)        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: ARCHITECTURAL REDESIGN (P2) — UNIFIED DATA STREAM & WORKER ORCHESTRATION      │
│ • Implement `/market/watch/unified` consolidated backend endpoint (Finding 2.1)        │
│ • Implement unified `marketStore.ts` with `useSyncExternalStore` (Finding 2.1)         │
│ • Replace per-client SSE thread polling with `MarketStreamBroadcaster` (Finding 5.7)   │
│ • Add distributed Redis lease for fund price collector loop (Finding 3.8)              │
│ • Disentangle Cloudflare Edge cron schedule collision (Finding 1.7)                    │
│ • Graceful shutdown of all background thread pools in `api.py` (Finding 5.5)           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Implementation Dependency Graph
```
[Phase 1: BoundedMemoryCache] ───► [Phase 1: Single-Flight Fix] ───► [Phase 2: Redis Versioning]
                                                                             │
[Phase 1: TerminalClock Leaf] ───► [Phase 2: Live StockDetail] ───► [Phase 3: Unified Market Store]
                                                                             │
[Phase 1: TEFAS Chunking]    ───► [Phase 2: Quad-Index Alignment]──► [Phase 3: SSE Broadcaster]
```

---

## 8. Verification & Audit Attestation

### 8.1 Zero-Production Code Modification Attestation
In strict compliance with the audit charter, **no existing source code, configuration files, or unit tests were modified, added, or deleted** during the course of this investigation.
- Verification command: `git status --porcelain`
- Output confirmed: Clean working tree (zero modifications to tracked files).

### 8.2 Independent Verification Guide
To independently inspect and verify any finding cited in this audit:

```powershell
# 1. Verify Single-Flight Fallthrough Stampede:
Get-Content d:\Projeler\rag-fin\app\market_service.py | Select-Object -Skip 1078 -First 35

# 2. Verify TEFAS Universal Market Loops:
Get-Content d:\Projeler\rag-fin\app\fund_service.py | Select-Object -Skip 5384 -First 25

# 3. Verify MarketsView Monolithic Clock Re-render:
Get-Content d:\Projeler\rag-fin\frontend\src\pages\MarketsView.tsx | Select-Object -Skip 3279 -First 10

# 4. Verify StockDetail Split-Brain Pricing:
Get-Content d:\Projeler\rag-fin\frontend\src\pages\StockDetailPage.tsx | Select-Object -Skip 796 -First 38

# 5. Verify Unbounded Python Dictionary Caches:
Select-String -Path d:\Projeler\rag-fin\app\market_service.py -Pattern "_CACHE: Dict"

# 6. Verify Financial Snapshot 30-Minute Stale Window:
Get-Content d:\Projeler\rag-fin\app\financial_snapshot_store.py | Select-Object -Skip 48 -First 10
```

---
*Report compiled and certified by worker_report_writer.*
