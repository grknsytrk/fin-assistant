# Supabase Egress Audit — rag-fin

**Date:** 2026-11-10
**Scope:** every backend read path that can move bytes out of Supabase (`ragfin_json_cache`, `fund_snapshots`, `fund_prices`, `company_financial_snapshots`, `instruments`, leases/advisory locks)
**Live target:** `https://grknsytrk-fin-api.hf.space` (Space `grknsytrk/fin-api`, sha `f083db29f0a2…`, 1 replica, `zero-a10g`)
**Method:** static trace of all cache/DB call sites + measured payload sizes (local DB copies at production schema, live HTTP responses with and without gzip) + live `/health` counters.

> Supabase egress here means **Postgres → backend** transfer. The frontend never talks to Supabase directly, so every byte counted below is server-side.

---

## 1. Executive summary

Egress is not spread evenly. **Three mechanisms produce almost all of it:**

| Rank | Mechanism | Per-event size | Frequency | Share of leak |
|---|---|---|---|---|
| **1** | Market price snapshot blob (`_shared/market_price_snapshot_xutum.json`, 147,324 B) read on every quote rebuild, plus a 30 s DB poll loop | 147 KB × 2…122 | every 3 s while a Markets/Model Portfolio tab is open | **dominant** |
| **2** | Daily fund snapshot JSONs (`funds_cache/daily_snapshots/*.json`, 756 KB avg / 2.17 MB max) read whole for history/backfill | 0.75–2.17 MB | per cold date, per backfill sweep | large, bursty |
| **3** | Full fund catalogue + all-fund price window on every `/funds` rebuild | 1.58 MB + 0.85 MB | per cache invalidation | large, periodic |

Everything else (metadata reads, instrument batches, KAP companions, detail endpoints) is 1–2 orders of magnitude smaller but adds latency and death-by-a-thousand-cuts egress.

**Root cause is architectural, not a single bug:** the app is designed around Redis as the shared cache. Redis is currently **quota-locked and fully failing over to memory** (`/health`: `cache_redis_fallback=true`, `quota_limited=true`, 295 GETs / 234 SETs in 34 min, all memory). With Redis gone, Supabase becomes the cache — and the Supabase cache stores **large JSON blobs that are always read in full**, with freshness windows (3 s) far shorter than the time needed to produce them (65–81 s measured).

**Headline numbers (measured, not estimated):**

- Market snapshot: **147,324 B** per read, 633 items, `expires_at = created_at + 3 s`.
- `/market/stocks?index=XUTUM`: 621,926 B raw / 58,467 B gzip, **81 s** first call; XU100 **15 s**; second call 1.4 s (`cache=hit`).
- `/funds`: 1,313,741 B raw / 97,421 B gzip, 2,013 rows, 4.4–11.3 s.
- `/kap/snapshot?company=THYAO&max_quarters=5`: 611,326 B raw / 21,353 B gzip.
- `fund_snapshots`: 2,019 rows, **1,577,221 B** over the API read columns (781 B/row).
- `fund_prices` latest-point join: 4,071 rows, **~1,213,618 B**; 5-day window ~850 KB; 10-day window 19,939 rows / 1,696,816 B.
- `instruments` stock rows: 371 B/row → ~222 KB per full XUTUM batch.
- Daily snapshots: 134 files / 101,325,884 B, **avg 756,163 B, max 2,171,683 B**.

---

## 2. Measurement basis & assumptions

- Payload bytes were measured on the local copies of the production SQLite mirrors (`data/processed/fund_prices.sqlite3`, `reference_data.sqlite3`) using the exact column lists the API selects, so they track the Postgres row payload closely.
- Live endpoint sizes were measured with `Accept-Encoding: identity` and `--compressed`.
- Request-rate numbers are **not** measured (no traffic analytics, HF logs truncated to 53 lines). They are stated explicitly wherever used. The per-event costs in §3 are exact; the hourly totals in §6 are models.
- Supabase egress bills **response bytes**, so writes (147 KB market snapshot, 2 MB daily snapshot) are not egress, but they do cost WAL/IO and keep the blob large.

---

## 3. Per-event cost table

Use this to price any traffic assumption.

| Event | Bytes | Source |
|---|---:|---|
| Read market price snapshot JSON | **147,324** | `_shared/market_price_snapshot_xutum.json` |
| Write market price snapshot JSON | 147,324 | same |
| Read all `fund_snapshots` API columns (2,019 rows) | **1,577,221** | `fund_prices.sqlite3` |
| `fund_prices` latest-points join (4,071 rows) | **1,213,618** | measured |
| `fund_prices` 5-day all-fund window (≈12k rows) | ~**850,000** | interpolated from 10-day = 1,696,816 |
| `fund_prices` 10-day window (19,939 rows) | 1,696,816 | measured |
| `instruments` stock batch (≈600 rows × 371 B) | ~**222,000** | measured 371 B/row |
| `company_financial_snapshots` full `SELECT *` (567 rows) | ~**280,000** | estimated at ~500 B/row |
| Daily fund snapshot JSON (avg / max) | **756,163 / 2,171,683** | 134 files |
| KAP cache document (history) | **176,149** | live `/health` metrics |
| `fund_listing_state` metadata row | ~1,300 | `funds_latest.json` 1,343 B |
| `fund_listing_state` revision row | <100 | two ints |
| Startup hydration (332 rows) | **~1,120,000** | per restart |

---

## 4. Findings

### L1 — CRITICAL: market price snapshot blob is always cold, and the wait path multiplies it

**Evidence** — `app/market_service.py`

- `:404` `_MARKET_PRICE_CACHE_TTL = 3`
- `:825-831` payload written with `expires_at = created_at + 3`
- `:815-822` `_market_price_snapshot_is_fresh()` rejects anything past `expires_at`
- `:848-855` `_market_price_snapshot_database_read()` returns `None` for stale payloads — **after** transferring the 147 KB
- `:1073` read #1 in `_fetch_market_price_map`
- `:975` read #2 in `_market_price_snapshot_with_database_lease`
- `:1019-1035` 30 s poll loop: `deadline = monotonic() + 30`, interval 0.05 → 0.25 s, **one full 147 KB read per iteration**
- `:1132` Redis-down branch routes every miss into the lease path

**Mechanism**

1. Upstream rebuild measured at **65–81 s** (`/market/stocks?index=XUTUM` first call 81,008 ms).
2. The snapshot is declared fresh for **3 s**.
3. Therefore the snapshot is stale before the rebuild that produced it even finishes. Every request is a miss.
4. Every miss pays **2 × 147 KB = 294 KB** before doing anything useful.
5. A request that cannot win the lease enters the 30 s poll loop: `0.05 + 0.075 + 0.1125 + 0.169 + 0.25…` → **≈122 reads ≈ 17.97 MB** for one client request.

**Cost**

- Floor, owner path: 294 KB per origin request.
- Waiter path: ~18 MB per origin request.
- 1,200 requests/h (3 s cadence) at the floor = **345 MB/h**.
- With even 50 % of requests hitting the wait path = **~10.8 GB/h**. Upper bound (all waiters) ≈ 20.8 GB/h.

**Fix (P0)**

1. **Stop encoding a 3 s TTL into the stored document.** Store `created_at` only. Decide freshness at read time: intraday 60 s, off-hours 300 s.
2. **Serve stale instead of blocking.** `_market_price_snapshot_database_read()` should return *any* payload with `items` (add a `stale` flag), not only fresh ones. Quote staleness of 60 s is acceptable for a fund/market screener.
3. **Delete the poll loop.** The lease is per-process (`acquire_refresh_lease` on the same DB row), so a loop is pointless: one read, then either build or return the stale snapshot. Replace `:1019-1035` with a single read + `return snapshot_or_none`.
4. **Single-flight in-process** with a `threading.Event` keyed on the snapshot, so concurrent requests wait on the local event (0 bytes) instead of on Postgres.
5. **Shrink the blob.** Persist only the projected fields (`symbol, price, change_pct, volume, as_of`) or a `market_quotes` table. The 633-item map carries far more than `/market/stocks` reads.
6. **Raise `_MARKET_PRICE_CACHE_TTL`** to at least the rebuild duration (90 s) so a rebuild actually serves the next 90 s.

Expected: **~95–99 % reduction** on the market path.

---

### L2 — CRITICAL: Redis quota exhaustion turns every shared-cache hit into a Supabase read

**Evidence** — `app/market_service.py:867-872`, `:1066-1080`; live `/health`

```
cache_backend: redis
cache_redis_fallback: true
cache_last_error: "This database has reached current Fixed plan limits…"
cache_redis_circuit: { open: true, quota_limited: true, quota_retry_seconds: 3600 }
```

When `redis_available` is true, `:1067-1071` serves the snapshot from Redis and **never touches Postgres**. When false, `:1131` goes to the DB lease path. So the entire L1 cost appears *only* while Redis is unavailable — which is the current permanent state.

The same pattern repeats across the app: `_shared_cache_get_dict` always misses, KAP snapshot response cache misses, fund response cache misses, `json_cache` revalidates remotely instead of hitting Redis.

**Fix (P0)**

- Restore Upstash capacity (or move to a self-hosted/Redis-compatible instance with a real memory budget).
- Independently, make the DB fallback good (L1) so a Redis outage degrades gracefully instead of 100×.

---

### L3 — HIGH: daily fund snapshot JSONs are 0.75–2.17 MB blobs read whole

**Evidence** — `app/fund_service.py`

- `:1492-1503` `_cached_daily_funds_snapshot()` → `_read_json(path)` → `json_cache_store.read_json_cache_local_first` → on a cold container, full `SELECT payload`.
- `:1472-1489` historical dates are considered fresh forever, so they revalidate (cheap `not_modified`) — but a **cold** container has no local file, so the first read is the full document.
- Recent-detail fallback loops up to 35 days (`FUNDS_RECENT_DETAIL_LOOKBACK_DAYS`) ≈ 50 business days.
- Overview-metric backfill loops 6 months × 10-day lookback (`FUNDS_OVERVIEW_METRIC_MONTHS`, `FUNDS_OVERVIEW_METRIC_LOOKBACK_DAYS`) → up to ~60 distinct snapshots.

**Cost**

- One cold date = 756 KB (avg) / 2.17 MB (max).
- Recent-detail fallback = **~38 MB avg / ~105 MB max** per fund.
- Overview backfill = **~45 MB avg / ~126 MB max** per sweep.
- Full 134-date sweep = 101 MB.
- HF Space containers are ephemeral (`data/processed` absent from the image, `.gitignore:75-79`), so these costs repeat on every restart and every scale-to-zero wake.

**Fix (P1)**

1. Historical rows already exist in `fund_prices` (200,713 rows, 2,082 codes). Serve history/backfill from there and **keep daily snapshots only for the current date**.
2. If a snapshot is still needed for point-in-time catalogue reconstruction, store it as rows in a `fund_daily_rows(date, fund_code, …)` table, not one JSON.
3. Add `RAGFIN_FUNDS_DAILY_SNAPSHOT_LEGACY_READS=0` and delete/compact old `funds_cache/daily_snapshots/%` rows.
4. Cap the overview backfill to months already covered by `fund_prices` before falling back to snapshots.

Expected: **90 %+** of the fund-history egress.

---

### L4 — HIGH: every `/funds` rebuild reads the whole catalogue plus a 5-day all-fund price window

**Evidence**

- `app/fund_service.py:7697-7710` `_query_fund_snapshot_rows()` selects `_FUND_SNAPSHOT_READ_COLUMNS` for every match → 2,013 rows × 781 B = **1.58 MB**.
- `:7723` `_apply_daily_return_overrides(..., include_missing=True)` → `:3319` every row becomes a candidate → `:3328-3334` `fund_prices` query over `as_of − 5 days … as_of` for **all 2,013 codes** → **~0.85 MB**, uncached.
- `:6591` `_catalog_price_overlay_metadata` additionally reads all `fund_snapshots` `fund_code/as_of` and `read_latest_fund_price_points` (**1.21 MB**, 900 s TTL).

**Cost:** ~2.4 MB per rebuild, ~3.7 MB when the 15-min overlay expires. Bounded by the 600 s listing LRU (`app/fund_api_service.py:130`), but any `price_generation` bump invalidates the ETag and forces a rebuild.

**Fix (P1)**

1. `include_missing=True` is the bug: only funds whose `daily_return` is missing/zero need reconciliation. Pass the narrowed candidate set.
2. Precompute daily returns at write time (`collect_daily_fund_prices`) into a `fund_daily_returns(code, date, pct)` table; the list endpoint then joins a 2,013-row, ~40 KB result.
3. Select only the columns the listing actually serialises (drop `attributes_json`, `present_fields_json`, `search_text`, sort keys) — roughly halves the row payload.
4. Raise the listing LRU to 15 min and key it on `(catalog_generation, price_generation)` so it survives revision churn.

---

### L5 — HIGH: two `fund_listing_state` queries on *every* funds request, including 304s

**Evidence**

- `app/fund_service.py:6665` `load_funds_snapshot()` calls `ensure_fund_snapshot_store()` **before** any cache check.
- `:1104-1109` takes a global `_FUND_SNAPSHOT_MIGRATION_LOCK` and runs `SELECT metadata_json FROM fund_listing_state WHERE id = 1`.
- `:468-476` `fund_listing_revision()` runs a second query on the same row.
- `app/routers/funds.py:22` the preflight runs **before** ETag comparison, so a 304 costs both queries.
- `get_fund_detail_payload` (`:7964-7973`) calls `load_funds_snapshot` again → a third read.

**Cost:** ~1.5 KB × 2 per request. Small in bytes, but it is a synchronous round trip on the hot path and it serialises all request threads behind one global lock.

**Fix (P1):** memoise `(metadata, revision)` for 30–60 s in-process, keyed by `price_generation`; merge the two columns into one `SELECT catalog_generation, price_generation, metadata_json`; skip `ensure_fund_snapshot_store` once the store is known active.

---

### L6 — MEDIUM: `instruments` batch read on every market build

**Evidence** — `app/market_service.py:276`, `app/reference_data.py:506-527`. No cache. `seed_manual_instruments()` also runs per call.
**Cost:** ~222 KB per build (≈600 stock rows × 371 B).
**Fix:** cache per `(kind, frozenset(symbols))` for 10 min; better, join `instruments` into the quote query so one round trip returns names + logos.

---

### L7 — MEDIUM: `company_financial_snapshots` full `SELECT *`

**Evidence** — `app/financial_snapshot_store.py:628-635`. Runs whenever the version key changes or the 30-min TTL expires. Version is bumped by `_invalidate_memory_cache()` (`:144-157`) on every KAP upsert, so the fast cron (every minute) can trigger it.
Mitigation already present: `:601-626` compares `(MAX(updated_at), COUNT(*))` and reuses the stale map when unchanged. That keeps the common case cheap.
**Cost:** ~280 KB per real change.
**Fix:** `SELECT * FROM company_financial_snapshots WHERE updated_at > :watermark` and merge; or fetch only requested symbols for `/kap/companies` and `/market/stocks`.

---

### L8 — MEDIUM: no response cache on `/funds/{code}` (and the invalidation code is dead)

**Evidence** — `app/routers/funds.py:244-252` → `_fund_detail_payload` → `get_fund_detail_payload` with no cache. `app/fund_api_service.py:2455` invalidates the `api:fund-detail:` prefix that nothing ever populates.
**Cost:** 3–4 queries (metadata ×2, one snapshot row, reference metadata, category ranking) per view. Bytes are small (~5–10 KB) but it is the most-visited detail page.
**Fix:** 60–300 s LRU keyed by `(code, catalog_generation, price_generation)`; wire the existing invalidation to it.

---

### L9 — MEDIUM: `/kap/companies` rebuilds instruments + financial snapshots

**Evidence** — `app/routers/kap.py:129-150`: `get_instrument_names()` + `get_financial_snapshots_map(..., companies)` for 567 companies, then a 109 KB response.
**Cost:** ~222 KB + 280 KB per rebuild (300 s in-process, 3,600 s edge).
**Fix:** persist a precomputed companies payload (one row, ~110 KB) and serve it directly.

---

### L10 — LOW: latest-point join carries `metadata_json` for 4,071 rows

**Evidence** — `app/fund_service.py:3202-3221`. `metadata_json` is 31.6 MB across 200k rows even though the sample value is `{}`.
**Fix:** drop `metadata_json` from the projection (the public mapper doesn't use it, as the comment already notes for `raw_json`), or `NULLIF` it.

---

### L11 — LOW: startup hydration re-downloads 1.12 MB per restart

**Evidence** — observed `332 rows / ~1.12 MB`, `excluded_rows=1108` (`funds_cache/`, `kap_cache/`).
**Fix:** acceptable as-is once L1/L3/L4 are fixed; optionally hydrate only row-based stores.

---

### L12 — INFO: writes are not egress but they are not free

Every market rebuild writes a 147 KB JSON row; every fund refresh writes ~2 MB. On a 3 s cadence that is ~4 GB/day of WAL. Fixing L1 also removes this.

---

## 5. What is already good (don't touch)

- `read_json_cache_conditional` (`app/database.py:649`) avoids payload transfer on unchanged `(updated_at, version)`.
- `read_json_cache_history_metadata` (`app/database.py:719-737`) reads only `payload #>> '{…}'` fields — a model for how the other blob reads should work.
- `_funds_listing_preflight` + ETag: a 304 returns no body.
- `/kap/snapshot` response cache + disk fallback (`app/kap_api_service.py:1986-2016`) and the SWR stale window (30 min).
- Edge cache TTLs in `worker/api-cache.ts`: 3 s market (trading), 300 s funds, 3,600 s KAP companies, 10 s flow head.
- `fund_price_points` cache (60 s) and `read_latest_fund_price_points` bulk join instead of per-code queries.

---

## 6. Scenario model

Assumptions are explicit; substitute your real rate.

| Scenario | Origin requests | Cost model | Egress |
|---|---|---|---|
| **S1** 1 Markets tab, Redis **up**, 3 s TTL | 1,200/h | Redis hit ≈ 0 B; ~45 rebuild writes | **~7 MB/h** |
| **S2** 1 Markets tab, Redis **down**, owner path only | 1,200/h | 2 × 147 KB | **~345 MB/h** |
| **S3** S2 + waiters (50 % of requests) | 1,200/h | 294 KB + 0.5 × 18 MB | **~10.8 GB/h** |
| **S4** S2 + waiters (all) | 1,200/h | 294 KB + 18 MB | **~20.8 GB/h** |
| **S5** After L1 fix (60 s TTL, serve-stale, no poll) | 1,200/h | 1 read / 60 s × 147 KB | **~8.8 MB/h** |
| **S6** After L1 + compact 25 KB rows | 1,200/h | 1 read / 60 s × 25 KB | **~1.5 MB/h** |
| **F1** Cold start, first `/funds` | 1 | hydration 1.12 MB + 3.7 MB | **~4.8 MB** |
| **F2** One overview-metric backfill (cold) | 1 | 60 snapshots × 756 KB | **~45 MB** (max 126 MB) |
| **F3** One fund recent-detail fallback (cold) | 1 | 50 × 756 KB | **~38 MB** (max 105 MB) |
| **F4** One `/funds` rebuild | 1 | 1.58 + 0.85 (+1.21) MB | **~2.4–3.7 MB** |
| **K1** One `/kap/companies` rebuild | 1 | 222 + 280 KB | **~0.5 MB** |

**S3 is the number that explains a runaway bill.** Even S2 (a single tab doing nothing but sitting open) is ~345 MB/h — 2.5 GB over a 7-hour session, per tab.

---

## 7. Prioritised remediation plan

### P0 — today (kills ~90 % of egress)

1. **Market snapshot freshness**: remove the 3 s `expires_at`; freshness 60 s intraday / 300 s off-hours; serve stale payloads with a `stale: true` flag (`market_service.py:404, 815-831, 848-855`).
2. **Delete the 30 s DB poll loop** (`market_service.py:1019-1035`): one read, then build or return stale; single-flight via a process-local `threading.Event`.
3. **Raise `_MARKET_PRICE_CACHE_TTL` to ≥ 90 s** so a completed rebuild is actually reused.
4. **Restore Redis** (or accept memory-only and size the fallback accordingly).

### P1 — this week (kills most of the rest)

5. **Daily snapshots → `fund_prices`** for history and backfill; cap the 6-month/35-day sweeps; stop reading `daily_snapshots/*.json` on cold containers.
6. **Narrow `include_missing=True`** in `_apply_daily_return_overrides`; ideally precompute returns at write time.
7. **Cache `(metadata, revision)`** for 30–60 s and merge the two `fund_listing_state` queries; skip the store check when the row store is active.
8. **Trim the `/funds` projection** to serialised columns only.
9. **Incremental `company_financial_snapshots`** read.

### P2 — next sprint (structural)

10. **Row-based market quotes** (`market_quotes`) or a ≤25 KB compact snapshot.
11. **Response cache for fund detail / search / categories**, wired to the existing invalidation prefixes.
12. **Precomputed `/kap/companies` payload**.
13. **Per-table egress counters** (see §8) so the next audit is measurement, not inference.

---

## 8. How to verify (do this before and after)

Add a thin wrapper around `connect_postgres()` that counts `len(orjson.dumps(rows))` per `(table, operation)` and exposes it in `/health` next to `supabase_json_cache_metrics`. Suggested fields:

```
supabase_db_metrics: {
  "window_seconds": …,
  "tables": {
    "ragfin_json_cache":     {"reads": n, "bytes": n},
    "fund_snapshots":        {"reads": n, "bytes": n},
    "fund_prices":           {"reads": n, "bytes": n},
    "company_financial_snapshots": {…},
    "instruments":           {…}
  }
}
```

Then run a scripted 10-minute session (open Markets, open Funds, open one fund, wait) and compare `bytes` before/after each fix. That converts every estimate in §6 into a measurement.

Quick checks that need no code:

```bash
curl -s -o /dev/null -w '%{size_download}\n' -H 'Accept-Encoding: identity' \
  'https://grknsytrk-fin-api.hf.space/market/stocks?index=XU100'
curl -s https://grknsytrk-fin-api.hf.space/health | jq '.cache_redis_circuit, .supabase_json_cache_metrics'
```

---

## 9. Appendix — evidence index

| Claim | Location |
|---|---|
| Market snapshot TTL 3 s | `app/market_service.py:404`, `:825-831` |
| Freshness gate after transfer | `app/market_service.py:815-822`, `:848-855` |
| DB reads per request | `app/market_service.py:975`, `:1073`, `:1122` |
| 30 s poll loop | `app/market_service.py:1019-1035` |
| Redis-down branch | `app/market_service.py:1131-1143` |
| Instruments per build | `app/market_service.py:276` |
| Financial snapshot per stocks build | `app/market_service.py:1574` |
| SSE rebuild loop | `app/market_broadcaster.py:129-…` |
| Snapshot metadata query per call | `app/fund_service.py:6665`, `:1069-1101`, `:1104-1109` |
| Revision query | `app/fund_service.py:468-476` |
| Preflight before ETag | `app/routers/funds.py:22-69` |
| Daily snapshot JSON read | `app/fund_service.py:1492-1503`, `:1472-1489` |
| All-fund return window | `app/fund_service.py:3319-3334`, `:7723` |
| Catalogue row read | `app/fund_service.py:7534-7573`, `:7697-7710` |
| Latest-point overlay | `app/fund_service.py:3141`, `:6591` |
| Detail without cache | `app/fund_api_service.py:2411-2414` |
| Dead detail invalidation | `app/fund_api_service.py:2455` |
| Financial snapshot full read | `app/financial_snapshot_store.py:628-635` |
| Instruments batch | `app/reference_data.py:506-527` |
| KAP companies build | `app/routers/kap.py:129-150` |
| Edge TTLs | `worker/api-cache.ts` |
