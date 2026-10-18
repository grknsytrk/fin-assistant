import type {
    KapCompaniesResponse,
    KapSnapshotResponse,
    KapValuationHistoryRequest,
    KapValuationHistoryResponse,
    MarketUniverseResponse,
    MarketStocksResponse,
    MarketStockDeltaItem,
    MarketStocksDeltaEvent,
    MarketStockSearchResponse,
    MarketStockIndex,
    MarketIndexCode,
    MarketIndicesResponse,
    MarketIndexDetailResponse,
    MarketFlowResponse,
    MarketFlowHeadResponse,
    ModelPortfolioPerformanceResponse,
    ModelPortfolioPositionClosePayload,
    ModelPortfolioPositionCreatePayload,
    ModelPortfolioPositionUpdatePayload,
    ModelPortfolioPrefillResponse,
    ModelPortfolioResponse,
    ModelPortfolioSymbolsResponse,
    MarketStockCardsResponse,
    MarketStockCardChartRange,
    MarketStockCardChartResponse,
    MarketWatchResponse,
    MarketWatchGlobalResponse,
    MarketCommoditiesResponse,
    MarketIndexResponse,
    MarketFxResponse,
    MarketComparisonHistoryAssetRequest,
    MarketComparisonHistoryResponse,
    FundAllocationsResponse,
    FundAllocationsHistoryResponse,
    FundCategoriesResponse,
    FundDetail,
    FundHistoryJob,
    FundHoldingsLiveResponse,
    FundHoldingsResponse,
    FundPerformanceResponse,
    FundYieldSummaryResponse,
    FundsResponse,
    KapOverviewCommentaryRequest,
    KapOverviewCommentaryResponse,
} from './types';

const rawApiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();
const API_BASE = (rawApiBase && rawApiBase !== '/' ? rawApiBase : 'http://localhost:8000').replace(/\/+$/, '');

const RETRYABLE_HTTP_STATUSES = new Set([408, 429, 502, 503, 504]);
const STARTUP_HINT = 'Uygulama yeni baslatiliyor olabilir. Lutfen 5-10 saniye bekleyip tekrar deneyin.';
const NETWORK_ERROR_MESSAGE = 'Su an baglanti kurulamiyor. Lutfen kisa bir sure sonra tekrar deneyin.';
const SERVER_ERROR_MESSAGE = 'Islem su an tamamlanamiyor. Lutfen daha sonra tekrar deneyin.';
const REQUEST_TIMEOUT_MS = 15000;
const OVERVIEW_COMMENTARY_TIMEOUT_MS = Number(import.meta.env.VITE_KAP_OVERVIEW_COMMENTARY_TIMEOUT_MS || 300000);
const TIMEOUT_MESSAGE = 'Istek suresi asildi. Lutfen tekrar deneyin.';
const FUND_HOLDINGS_MEMORY_CACHE_TTL_MS = 15_000;
const fundHoldingsMemoryCache = new Map<string, { payload: FundHoldingsResponse; fetchedAt: number }>();
const fundHoldingsInFlight = new Map<string, Promise<FundHoldingsResponse>>();
const fundHoldingsLiveInFlight = new Map<string, Promise<FundHoldingsLiveResponse>>();
const MARKET_FLOW_MEMORY_CACHE_TTL_MS = 15_000;
const marketFlowMemoryCache = new Map<string, { payload: MarketFlowResponse; fetchedAt: number }>();
const marketFlowInFlight = new Map<string, Promise<MarketFlowResponse>>();
const MARKET_UNIVERSE_MEMORY_CACHE_TTL_MS = 60_000;
const marketUniverseMemoryCache = new Map<string, { payload: MarketUniverseResponse; fetchedAt: number }>();
const marketUniverseInFlight = new Map<string, Promise<MarketUniverseResponse>>();
import { isBistTradingHours } from '../utils/marketHours';

const MARKET_STOCKS_MEMORY_CACHE_TTL_MS = 3_000;
const MARKET_STOCKS_OFFHOURS_MEMORY_CACHE_TTL_MS = 60_000;
const marketStocksMemoryCache = new Map<string, { payload: MarketStocksResponse; fetchedAt: number }>();
const marketStocksInFlight = new Map<string, Promise<MarketStocksResponse>>();
const marketStocksListeners = new Set<(index: MarketStockIndex, payload: MarketStocksResponse) => void>();

interface MarketStocksStreamHandle {
    refCount: number;
    source: EventSource | null;
    connected: boolean;
    cleanupListeners: (() => void) | null;
}

const marketStocksStreams = new Map<MarketStockIndex, MarketStocksStreamHandle>();

function mergeDeltaIntoPayload(
    basePayload: MarketStocksResponse,
    deltaMap: Map<string, MarketStockDeltaItem>,
    asOf?: string,
): { nextPayload: MarketStocksResponse; changed: boolean } {
    let anyRowChanged = false;
    const nextRows = basePayload.rows.map((row) => {
        const sym = String(row.company || row.symbol || '').trim().toUpperCase();
        const delta = deltaMap.get(sym);
        if (!delta) return row;
        anyRowChanged = true;
        return {
            ...row,
            ...(delta.price !== undefined ? { price: delta.price } : {}),
            ...(delta.price_currency !== undefined ? { price_currency: delta.price_currency } : {}),
            ...(delta.change !== undefined ? { change: delta.change } : {}),
            ...(delta.change_pct !== undefined ? { change_pct: delta.change_pct } : {}),
            ...(delta.price_as_of !== undefined ? { price_as_of: delta.price_as_of } : {}),
            ...(delta.volume !== undefined ? { volume: delta.volume } : {}),
            ...(delta.market_cap !== undefined ? { market_cap: delta.market_cap } : {}),
            ...(delta.return_1w_pct !== undefined ? { return_1w_pct: delta.return_1w_pct } : {}),
            ...(delta.return_1m_pct !== undefined ? { return_1m_pct: delta.return_1m_pct } : {}),
            ...(delta.return_3m_pct !== undefined ? { return_3m_pct: delta.return_3m_pct } : {}),
            ...(delta.return_6m_pct !== undefined ? { return_6m_pct: delta.return_6m_pct } : {}),
            ...(delta.return_ytd_pct !== undefined ? { return_ytd_pct: delta.return_ytd_pct } : {}),
            ...(delta.return_1y_pct !== undefined ? { return_1y_pct: delta.return_1y_pct } : {}),
        };
    });
    if (!anyRowChanged) {
        return { nextPayload: basePayload, changed: false };
    }
    return {
        changed: true,
        nextPayload: {
            ...basePayload,
            rows: nextRows,
            as_of: asOf || basePayload.as_of,
            quote_status: 'fresh',
            stale: false,
        },
    };
}

function applyMarketStocksDeltaInternal(
    index: MarketStockIndex,
    deltaEvent: MarketStocksDeltaEvent,
): MarketStocksResponse | null {
    if (!deltaEvent || !Array.isArray(deltaEvent.rows) || deltaEvent.rows.length === 0) {
        return marketStocksMemoryCache.get(index)?.payload || null;
    }
    const deltaMap = new Map<string, MarketStockDeltaItem>();
    for (const item of deltaEvent.rows) {
        const sym = String(item?.company || '').trim().toUpperCase();
        if (sym) deltaMap.set(sym, item);
    }
    if (deltaMap.size === 0) {
        return marketStocksMemoryCache.get(index)?.payload || null;
    }

    const now = Date.now();
    const targetIndices: MarketStockIndex[] =
        index === 'XUTUM' ? ['XUTUM', 'XU100', 'XU030'] : [index];

    let primaryResult: MarketStocksResponse | null = null;
    for (const targetIndex of targetIndices) {
        const cached = marketStocksMemoryCache.get(targetIndex);
        if (!cached) continue;
        const { nextPayload, changed } = mergeDeltaIntoPayload(cached.payload, deltaMap, deltaEvent.as_of);
        if (changed) {
            marketStocksMemoryCache.set(targetIndex, { payload: nextPayload, fetchedAt: now });
            for (const listener of marketStocksListeners) {
                listener(targetIndex, nextPayload);
            }
        }
        if (targetIndex === index) {
            primaryResult = nextPayload;
        }
    }
    return primaryResult || marketStocksMemoryCache.get(index)?.payload || null;
}

function retainMarketStocksStreamInternal(index: MarketStockIndex = 'XUTUM'): () => void {
    if (typeof window === 'undefined' || typeof EventSource === 'undefined') {
        return () => {};
    }
    let handle = marketStocksStreams.get(index);
    if (!handle) {
        handle = { refCount: 0, source: null, connected: false, cleanupListeners: null };
        marketStocksStreams.set(index, handle);
    }
    handle.refCount += 1;

    const closeSource = () => {
        if (!handle) return;
        if (handle.source) {
            try {
                handle.source.close();
            } catch {
                // ignore
            }
            handle.source = null;
        }
        handle.connected = false;
    };

    const syncStreamState = () => {
        if (!handle || handle.refCount <= 0) {
            closeSource();
            return;
        }
        const tabVisible = typeof document === 'undefined' || document.visibilityState !== 'hidden';
        const marketOpen = isBistTradingHours();
        if (!tabVisible || !marketOpen) {
            closeSource();
            return;
        }
        if (handle.source) return;
        try {
            const url = `${API_BASE}/market/stream?index=${encodeURIComponent(index)}`;
            const es = new EventSource(url);
            handle.source = es;
            es.onopen = () => {
                if (handle) handle.connected = true;
            };
            es.onerror = () => {
                if (handle) handle.connected = false;
            };
            es.addEventListener('stocks_delta', (event: MessageEvent) => {
                try {
                    const parsed = JSON.parse(String(event.data || '{}')) as MarketStocksDeltaEvent;
                    applyMarketStocksDeltaInternal(index, parsed);
                } catch {
                    // ignore malformed event
                }
            });
        } catch {
            closeSource();
        }
    };

    if (!handle.cleanupListeners) {
        const onVisibilityChange = () => syncStreamState();
        document.addEventListener('visibilitychange', onVisibilityChange);
        const checkTimer = window.setInterval(syncStreamState, 30_000);
        handle.cleanupListeners = () => {
            document.removeEventListener('visibilitychange', onVisibilityChange);
            window.clearInterval(checkTimer);
        };
    }

    syncStreamState();

    let released = false;
    return () => {
        if (released) return;
        released = true;
        const current = marketStocksStreams.get(index);
        if (!current) return;
        current.refCount = Math.max(0, current.refCount - 1);
        if (current.refCount === 0) {
            if (current.cleanupListeners) {
                current.cleanupListeners();
                current.cleanupListeners = null;
            }
            if (current.source) {
                try {
                    current.source.close();
                } catch {
                    // ignore
                }
                current.source = null;
            }
            current.connected = false;
            marketStocksStreams.delete(index);
        }
    };
}

let latestFundsCatalog: { payload: FundsResponse; fetchedAt: number } | null = null;
let fundsCatalogInFlight: Promise<FundsResponse> | null = null;
const fundsCatalogListeners = new Set<(payload: FundsResponse) => void>();

const MODEL_PORTFOLIO_MEMORY_CACHE_TTL_MS = 15_000;
let modelPortfolioMemoryCache: { payload: ModelPortfolioResponse; fetchedAt: number } | null = null;
let modelPortfolioInFlight: Promise<ModelPortfolioResponse> | null = null;
let modelPortfolioPerformanceMemoryCache: { version: number; payload: ModelPortfolioPerformanceResponse; fetchedAt: number } | null = null;
let modelPortfolioPerformanceInFlight: Promise<ModelPortfolioPerformanceResponse> | null = null;
let modelPortfolioSymbolsMemoryCache: { symbols: string[]; version: number; fetchedAt: number } | null = null;
let modelPortfolioSymbolsInFlight: Promise<ModelPortfolioSymbolsResponse> | null = null;
const modelPortfolioSymbolsListeners = new Set<(symbols: string[]) => void>();

const kapSnapshotMemoryCache = new Map<string, { payload: KapSnapshotResponse; at: number }>();
export function cachedKapSnapshot(company: string): KapSnapshotResponse | null {
    const entry = kapSnapshotMemoryCache.get(company.trim().toUpperCase());
    return entry && Date.now() - entry.at < 5 * 60_000 ? entry.payload : null;
}

type FetchApiOptions = RequestInit & {
    timeoutMs?: number;
    debugLabel?: string;
    exposeErrorDetail?: boolean;
};

function isIdempotentMethod(method?: string): boolean {
    const normalized = (method || 'GET').toUpperCase();
    return normalized === 'GET' || normalized === 'HEAD';
}

function isRetryableNetworkError(error: unknown): boolean {
    if (!(error instanceof Error)) {
        return false;
    }
    const msg = String(error.message || '').toLowerCase();
    return (
        msg.includes('failed to fetch') ||
        msg.includes('networkerror') ||
        msg.includes('load failed') ||
        msg.includes('fetch')
    );
}

function retryDelayMs(attempt: number, retryAfterMs?: number | null): number {
    if (retryAfterMs != null && Number.isFinite(retryAfterMs)) return Math.max(0, retryAfterMs);
    const base = 250;
    const max = 1800;
    return Math.round(Math.min(max, base * Math.pow(1.6, Math.max(0, attempt - 1))) * (0.8 + Math.random() * 0.4));
}

function retryAfterMs(response: Response): number | null {
    const raw = response.headers.get('Retry-After')?.trim();
    if (!raw) return null;
    const seconds = Number(raw);
    if (Number.isFinite(seconds)) return Math.min(Math.max(0, seconds * 1000), 300_000);
    const when = Date.parse(raw);
    return Number.isFinite(when) ? Math.min(Math.max(0, when - Date.now()), 300_000) : null;
}

async function fetchApi<T>(endpoint: string, options: FetchApiOptions = {}): Promise<T> {
    const normalizedEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    const url = `${API_BASE}${normalizedEndpoint}`;
    const method = (options.method || 'GET').toUpperCase();
    const allowRetry = isIdempotentMethod(method);
    const maxAttempts = allowRetry ? 2 : 1;
    const { timeoutMs, debugLabel, exposeErrorDetail, ...requestOptions } = options;
    const effectiveTimeoutMs = timeoutMs ?? REQUEST_TIMEOUT_MS;

    // Keep the timeout per attempt so a timed-out idempotent request still has
    // enough time to use its retry instead of expiring at the overall deadline.
    const retryWindowMs = effectiveTimeoutMs * maxAttempts + 1800;
    const deadline = Date.now() + (allowRetry ? retryWindowMs : effectiveTimeoutMs);
    const waitForRetry = async (delay: number) => {
        if (Date.now() + delay >= deadline) throw new Error(TIMEOUT_MESSAGE);
        await new Promise<void>((resolve, reject) => {
            const signal = requestOptions.signal;
            const abort = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
            const timer = window.setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, delay);
            if (signal?.aborted) abort();
            else signal?.addEventListener('abort', abort, { once: true });
        });
    };
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        if (Date.now() >= deadline) throw new Error(TIMEOUT_MESSAGE);
        let response: Response;
        let body: any;
        const controller = new AbortController();
        const externalSignal = requestOptions.signal;
        const startedAt = window.performance?.now?.() ?? Date.now();
        const timeoutId = window.setTimeout(() => controller.abort(), Math.min(effectiveTimeoutMs, deadline - Date.now()));
        const abortFromExternal = () => controller.abort();
        if (externalSignal?.aborted) {
            controller.abort();
        } else {
            externalSignal?.addEventListener('abort', abortFromExternal, { once: true });
        }
        if (debugLabel) {
            console.debug(`[api:${debugLabel}] request started`, {
                url,
                method,
                attempt,
                timeoutMs: effectiveTimeoutMs,
            });
        }
        try {
            response = await fetch(url, {
                ...requestOptions,
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...requestOptions.headers,
                },
            });
            body = await response.json().catch((error: unknown) => {
                if (controller.signal.aborted) throw error;
                if (response.ok) throw error;
                return {};
            });
        } catch (error) {
            const elapsedMs = Math.round((window.performance?.now?.() ?? Date.now()) - startedAt);
            if ((error as Error)?.name === 'AbortError') {
                if (debugLabel) {
                    console.debug(`[api:${debugLabel}] request aborted`, {
                        url,
                        method,
                        attempt,
                        elapsedMs,
                    });
                }
                if (externalSignal?.aborted) {
                    throw error;
                }
                if (allowRetry && attempt < maxAttempts) {
                    await waitForRetry(retryDelayMs(attempt));
                    continue;
                }
                throw new Error(TIMEOUT_MESSAGE);
            }
            if (allowRetry && isRetryableNetworkError(error) && attempt < maxAttempts) {
                await waitForRetry(retryDelayMs(attempt));
                continue;
            }
            if (debugLabel) {
                console.error(`[api:${debugLabel}] network error`, {
                    url,
                    method,
                    attempt,
                    elapsedMs,
                    error,
                });
            }
            throw new Error(`${NETWORK_ERROR_MESSAGE} ${STARTUP_HINT}`);
        } finally {
            window.clearTimeout(timeoutId);
            externalSignal?.removeEventListener('abort', abortFromExternal);
        }

        if (!response.ok) {
            const elapsedMs = Math.round((window.performance?.now?.() ?? Date.now()) - startedAt);
            if (allowRetry && RETRYABLE_HTTP_STATUSES.has(response.status) && attempt < maxAttempts) {
                await waitForRetry(retryDelayMs(attempt, retryAfterMs(response)));
                continue;
            }

            const errorData = body || {};
            if (debugLabel) {
                console.error(`[api:${debugLabel}] http error`, {
                    url,
                    method,
                    attempt,
                    elapsedMs,
                    status: response.status,
                    errorData,
                });
            }
            const detail = typeof errorData?.detail === 'string' ? errorData.detail : null;
            if (detail && (response.status < 500 || exposeErrorDetail)) {
                throw new Error(detail);
            }
            if (RETRYABLE_HTTP_STATUSES.has(response.status)) {
                throw new Error(STARTUP_HINT);
            }
            if (response.status >= 500) {
                throw new Error(SERVER_ERROR_MESSAGE);
            }
            throw new Error(`Islem tamamlanamadi (${response.status}). Lutfen tekrar deneyin.`);
        }

        if (debugLabel) {
            const elapsedMs = Math.round((window.performance?.now?.() ?? Date.now()) - startedAt);
            console.debug(`[api:${debugLabel}] request completed`, {
                url,
                method,
                attempt,
                elapsedMs,
                status: response.status,
            });
        }
        return body as T;
    }

    throw new Error(STARTUP_HINT);
}

async function fetchFundResponse<T extends { status: string }>(endpoint: string, options: FetchApiOptions = {}): Promise<T> {
    const deadline = Date.now() + 120_000;
    let delay = 1000;
    while (true) {
        if (options.signal?.aborted) throw new DOMException('Aborted', 'AbortError');
        const payload = await fetchApi<T>(endpoint, options);
        if (payload.status !== 'pending') return payload;
        if (Date.now() >= deadline) throw new Error('Veri hazırlama işlemi beklenenden uzun sürüyor.');
        await new Promise<void>((resolve, reject) => {
            const signal = options.signal;
            const abort = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
            const timer = window.setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, delay);
            if (signal?.aborted) abort();
            else signal?.addEventListener('abort', abort, { once: true });
        });
        delay = Math.min(5000, delay * 1.5);
    }
}

export const apiClient = {
    health: () => fetchApi<{ status: string }>('/health'),

    marketUniverse: (options?: { index?: MarketStockIndex; refresh?: boolean }) => {
        const cacheKey = options?.index || 'XUTUM';
        const cached = marketUniverseMemoryCache.get(cacheKey);
        if (!options?.refresh && cached && Date.now() - cached.fetchedAt < MARKET_UNIVERSE_MEMORY_CACHE_TTL_MS) {
            return Promise.resolve(cached.payload);
        }

        const existingRequest = marketUniverseInFlight.get(cacheKey);
        if (existingRequest) return existingRequest;

        const params = new URLSearchParams();
        if (options?.index) params.append('index', options.index);
        if (options?.refresh) params.append('refresh', 'true');
        const query = params.toString();
        const request = fetchApi<MarketUniverseResponse>(query ? `/market/universe?${query}` : '/market/universe')
            .then((payload) => {
                marketUniverseMemoryCache.set(cacheKey, { payload, fetchedAt: Date.now() });
                return payload;
            })
            .finally(() => {
                if (marketUniverseInFlight.get(cacheKey) === request) marketUniverseInFlight.delete(cacheKey);
            });
        marketUniverseInFlight.set(cacheKey, request);
        return request;
    },
    getCachedMarketUniverse: (index: MarketStockIndex = 'XUTUM') => marketUniverseMemoryCache.get(index)?.payload || null,
    marketStocks: (options?: { index?: MarketStockIndex; refresh?: boolean }) => {
        const cacheKey = options?.index || 'XUTUM';
        const cached = marketStocksMemoryCache.get(cacheKey);
        const memoryTtlMs = isBistTradingHours()
            ? MARKET_STOCKS_MEMORY_CACHE_TTL_MS
            : MARKET_STOCKS_OFFHOURS_MEMORY_CACHE_TTL_MS;
        if (!options?.refresh && cached && Date.now() - cached.fetchedAt < memoryTtlMs) {
            return Promise.resolve(cached.payload);
        }

        const existingRequest = marketStocksInFlight.get(cacheKey);
        if (existingRequest) return existingRequest;

        const params = new URLSearchParams();
        if (options?.index) params.append('index', options.index);
        if (options?.refresh) params.append('refresh', 'true');
        const query = params.toString();
        const request = fetchApi<MarketStocksResponse>(query ? `/market/stocks?${query}` : '/market/stocks')
            .then((payload) => {
                marketStocksMemoryCache.set(cacheKey, { payload, fetchedAt: Date.now() });
                for (const listener of marketStocksListeners) listener(cacheKey, payload);
                return payload;
            })
            .finally(() => {
                if (marketStocksInFlight.get(cacheKey) === request) marketStocksInFlight.delete(cacheKey);
            });
        marketStocksInFlight.set(cacheKey, request);
        return request;
    },
    getCachedMarketStocks: (index: MarketStockIndex) => marketStocksMemoryCache.get(index)?.payload || null,
    getCachedMarketStocksEntry: (index: MarketStockIndex) => marketStocksMemoryCache.get(index) || null,
    subscribeMarketStocks: (listener: (index: MarketStockIndex, payload: MarketStocksResponse) => void) => {
        marketStocksListeners.add(listener);
        return () => {
            marketStocksListeners.delete(listener);
        };
    },
    applyMarketStocksDelta: (index: MarketStockIndex, delta: MarketStocksDeltaEvent) =>
        applyMarketStocksDeltaInternal(index, delta),
    retainMarketStocksStream: (index: MarketStockIndex = 'XUTUM') =>
        retainMarketStocksStreamInternal(index),
    isMarketStocksStreamActive: (index: MarketStockIndex = 'XUTUM') =>
        Boolean(marketStocksStreams.get(index)?.connected),
    marketStockSearch: (options: {
        q: string;
        index?: MarketStockIndex;
        limit?: number;
        signal?: AbortSignal;
    }) => {
        const params = new URLSearchParams({ q: options.q });
        if (options.index) params.append('index', options.index);
        if (options.limit) params.append('limit', String(options.limit));
        return fetchApi<MarketStockSearchResponse>(`/market/stocks/search?${params.toString()}`, {
            signal: options.signal,
        });
    },
    marketStockCards: (options: {
        symbols: string[];
        refresh?: boolean;
        tier?: 'quick' | 'full';
        signal?: AbortSignal;
    }) => {
        const params = new URLSearchParams();
        params.append('symbols', options.symbols.join(','));
        if (options.refresh) params.append('refresh', 'true');
        if (options.tier) params.append('tier', options.tier);
        return fetchApi<MarketStockCardsResponse>(`/market/stocks/cards?${params.toString()}`, {
            signal: options.signal,
        });
    },
    marketStockCardChart: (
        symbol: string,
        range: MarketStockCardChartRange,
        options?: { refresh?: boolean; signal?: AbortSignal },
    ) => {
        const params = new URLSearchParams({ symbol, range });
        if (options?.refresh) params.append('refresh', 'true');
        return fetchApi<MarketStockCardChartResponse>(`/market/stocks/cards/chart?${params.toString()}`, {
            signal: options?.signal,
        });
    },
    marketComparisonHistory: (
        request: {
            assets: MarketComparisonHistoryAssetRequest[];
            start_date: string;
            end_date: string;
        },
        options?: { signal?: AbortSignal },
    ) =>
        fetchApi<MarketComparisonHistoryResponse>('/market/comparison-history', {
            method: 'POST',
            body: JSON.stringify(request),
            signal: options?.signal,
            timeoutMs: 30000,
            exposeErrorDetail: true,
        }),
    marketIndices: (options?: { refresh?: boolean }) => {
        const params = new URLSearchParams();
        if (options?.refresh) params.append('refresh', 'true');
        const query = params.toString();
        return fetchApi<MarketIndicesResponse>(query ? `/market/indices?${query}` : '/market/indices');
    },
    marketIndexDetail: (index: MarketIndexCode, options?: { refresh?: boolean }) => {
        const params = new URLSearchParams();
        if (options?.refresh) params.append('refresh', 'true');
        const query = params.toString();
        return fetchApi<MarketIndexDetailResponse>(
            query ? `/market/indices/${index}?${query}` : `/market/indices/${index}`,
        );
    },
    marketFlow: (
        limit = 40,
        category?: string,
        options?: { refresh?: boolean; before?: string; after?: string },
    ) => {
        const params = new URLSearchParams({ limit: String(limit) });
        if (category) params.append('category', category);
        if (options?.refresh) params.append('refresh', 'true');
        if (options?.before) params.append('before', options.before);
        if (options?.after) params.append('after', options.after);
        const cacheKey = `${limit}:${category || ''}:${options?.before || ''}:${options?.after || ''}`;
        const cached = marketFlowMemoryCache.get(cacheKey);
        if (!options?.refresh && cached && Date.now() - cached.fetchedAt < MARKET_FLOW_MEMORY_CACHE_TTL_MS) {
            return Promise.resolve(cached.payload);
        }

        const existingRequest = marketFlowInFlight.get(cacheKey);
        if (existingRequest) return existingRequest;

        const request = fetchApi<MarketFlowResponse>(`/market/flow?${params.toString()}`)
            .then((payload) => {
                marketFlowMemoryCache.set(cacheKey, { payload, fetchedAt: Date.now() });
                return payload;
            })
            .finally(() => {
                if (marketFlowInFlight.get(cacheKey) === request) marketFlowInFlight.delete(cacheKey);
            });
        marketFlowInFlight.set(cacheKey, request);
        return request;
    },
    marketFlowHead: (category?: string) => {
        const params = new URLSearchParams();
        if (category) params.append('category', category);
        const query = params.toString();
        return fetchApi<MarketFlowHeadResponse>(query ? `/market/flow/head?${query}` : '/market/flow/head');
    },
    marketWatch: (options?: { refresh?: boolean }) => {
        const params = new URLSearchParams();
        if (options?.refresh) params.append('refresh', 'true');
        const query = params.toString();
        return fetchApi<MarketWatchResponse>(query ? `/market/watch?${query}` : '/market/watch');
    },
    marketWatchGlobal: (options?: { refresh?: boolean }) => {
        const params = new URLSearchParams();
        if (options?.refresh) params.append('refresh', 'true');
        const query = params.toString();
        return fetchApi<MarketWatchGlobalResponse>(query ? `/market/watch/global?${query}` : '/market/watch/global');
    },
    marketXu030: () => fetchApi<MarketIndexResponse>('/market/xu030'),
    marketCommodities: () => fetchApi<MarketCommoditiesResponse>('/market/commodities'),
    marketFx: () => fetchApi<MarketFxResponse>('/market/fx'),

    funds: (options?: {
        q?: string;
        fundType?: string;
        founder?: string;
        manager?: string;
        risk?: string;
        sort?: string;
        order?: 'asc' | 'desc';
        signal?: AbortSignal;
    }) => {
        const params = new URLSearchParams();
        if (options?.q) params.append('q', options.q);
        if (options?.fundType) params.append('fund_type', options.fundType);
        if (options?.founder) params.append('founder', options.founder);
        if (options?.manager) params.append('manager', options.manager);
        if (options?.risk) params.append('risk', options.risk);
        if (options?.sort) params.append('sort', options.sort);
        if (options?.order) params.append('order', options.order);
        const query = params.toString();
        const isUnfilteredCatalog = !options?.q
            && !options?.fundType
            && !options?.founder
            && !options?.manager
            && !options?.risk
            && !options?.sort
            && !options?.order;
        const canShareCatalogRequest = isUnfilteredCatalog && !options?.signal;
        if (canShareCatalogRequest && fundsCatalogInFlight) return fundsCatalogInFlight;

        const request = fetchFundResponse<FundsResponse>(query ? `/funds?${query}` : '/funds', { signal: options?.signal })
            .then((payload) => {
                if (isUnfilteredCatalog) {
                    latestFundsCatalog = { payload, fetchedAt: Date.now() };
                    for (const listener of fundsCatalogListeners) listener(payload);
                }
                return payload;
            })
            .finally(() => {
                if (canShareCatalogRequest && fundsCatalogInFlight === request) fundsCatalogInFlight = null;
            });
        if (canShareCatalogRequest) fundsCatalogInFlight = request;
        return request;
    },
    getCachedFundsCatalog: () => latestFundsCatalog,
    subscribeFundsCatalog: (listener: (payload: FundsResponse) => void) => {
        fundsCatalogListeners.add(listener);
        return () => {
            fundsCatalogListeners.delete(listener);
        };
    },
    modelPortfolio: (options?: { refresh?: boolean; signal?: AbortSignal }) => {
        if (
            !options?.refresh
            && modelPortfolioMemoryCache
            && Date.now() - modelPortfolioMemoryCache.fetchedAt < MODEL_PORTFOLIO_MEMORY_CACHE_TTL_MS
        ) {
            return Promise.resolve(modelPortfolioMemoryCache.payload);
        }
        if (!options?.refresh && modelPortfolioInFlight) {
            return modelPortfolioInFlight;
        }
        const request = fetchApi<ModelPortfolioResponse>('/model-portfolio', { signal: options?.signal })
            .then((payload) => {
                modelPortfolioMemoryCache = { payload, fetchedAt: Date.now() };
                const activeSyms = (payload.active_positions || []).map((p) => p.symbol);
                modelPortfolioSymbolsMemoryCache = {
                    symbols: activeSyms,
                    version: payload.version,
                    fetchedAt: Date.now(),
                };
                for (const listener of modelPortfolioSymbolsListeners) listener(activeSyms);
                return payload;
            })
            .finally(() => {
                if (modelPortfolioInFlight === request) modelPortfolioInFlight = null;
            });
        modelPortfolioInFlight = request;
        return request;
    },
    getCachedModelPortfolio: () => modelPortfolioMemoryCache?.payload || null,
    modelPortfolioPerformance: (options?: { version?: number; refresh?: boolean; signal?: AbortSignal }) => {
        if (
            !options?.refresh
            && modelPortfolioPerformanceMemoryCache
            && (options?.version == null || modelPortfolioPerformanceMemoryCache.version === options.version)
            && Date.now() - modelPortfolioPerformanceMemoryCache.fetchedAt < 5 * 60_000
        ) {
            return Promise.resolve(modelPortfolioPerformanceMemoryCache.payload);
        }
        if (!options?.refresh && modelPortfolioPerformanceInFlight) {
            return modelPortfolioPerformanceInFlight;
        }
        const request = fetchApi<ModelPortfolioPerformanceResponse>('/model-portfolio/performance', {
            signal: options?.signal,
            timeoutMs: 30000,
        })
            .then((payload) => {
                modelPortfolioPerformanceMemoryCache = {
                    version: payload.version,
                    payload,
                    fetchedAt: Date.now(),
                };
                return payload;
            })
            .finally(() => {
                if (modelPortfolioPerformanceInFlight === request) modelPortfolioPerformanceInFlight = null;
            });
        modelPortfolioPerformanceInFlight = request;
        return request;
    },
    modelPortfolioSymbols: (options?: { refresh?: boolean; signal?: AbortSignal }) => {
        if (
            !options?.refresh
            && modelPortfolioSymbolsMemoryCache
            && Date.now() - modelPortfolioSymbolsMemoryCache.fetchedAt < 60_000
        ) {
            return Promise.resolve({
                symbols: modelPortfolioSymbolsMemoryCache.symbols,
                count: modelPortfolioSymbolsMemoryCache.symbols.length,
                version: modelPortfolioSymbolsMemoryCache.version,
                updated_at: new Date(modelPortfolioSymbolsMemoryCache.fetchedAt).toISOString(),
            });
        }
        if (!options?.refresh && modelPortfolioSymbolsInFlight) {
            return modelPortfolioSymbolsInFlight;
        }
        const request = fetchApi<ModelPortfolioSymbolsResponse>('/model-portfolio/symbols', { signal: options?.signal })
            .then((payload) => {
                const normalized = Array.from(
                    new Set((payload.symbols || []).map((s) => String(s || '').trim().toUpperCase()).filter(Boolean)),
                );
                modelPortfolioSymbolsMemoryCache = {
                    symbols: normalized,
                    version: payload.version,
                    fetchedAt: Date.now(),
                };
                for (const listener of modelPortfolioSymbolsListeners) listener(normalized);
                return { ...payload, symbols: normalized };
            })
            .finally(() => {
                if (modelPortfolioSymbolsInFlight === request) modelPortfolioSymbolsInFlight = null;
            });
        modelPortfolioSymbolsInFlight = request;
        return request;
    },
    getCachedModelPortfolioSymbols: () => modelPortfolioSymbolsMemoryCache?.symbols || null,
    subscribeModelPortfolioSymbols: (listener: (symbols: string[]) => void) => {
        modelPortfolioSymbolsListeners.add(listener);
        return () => {
            modelPortfolioSymbolsListeners.delete(listener);
        };
    },
    modelPortfolioVerifyAdmin: (token: string) =>
        fetchApi<{ ok: boolean; session_token: string; expires_in_seconds: number }>('/model-portfolio/admin/verify', {
            method: 'POST',
            headers: { 'X-Admin-Token': encodeURIComponent(token.trim()) },
            exposeErrorDetail: true,
        }),
    modelPortfolioPrefill: (symbol: string, entryDate?: string, signal?: AbortSignal) => {
        const params = new URLSearchParams({ symbol: symbol.trim().toUpperCase() });
        if (entryDate) params.append('entry_date', entryDate);
        return fetchApi<ModelPortfolioPrefillResponse>(`/model-portfolio/prefill?${params.toString()}`, {
            signal,
            exposeErrorDetail: true,
        });
    },
    modelPortfolioCreatePosition: (payload: ModelPortfolioPositionCreatePayload, token: string) =>
        fetchApi<ModelPortfolioResponse>('/model-portfolio/positions', {
            method: 'POST',
            headers: { Authorization: `Bearer ${token.trim()}` },
            body: JSON.stringify(payload),
            exposeErrorDetail: true,
        }).then((res) => {
            modelPortfolioMemoryCache = { payload: res, fetchedAt: Date.now() };
            modelPortfolioPerformanceMemoryCache = null;
            const syms = (res.active_positions || []).map((p) => p.symbol);
            modelPortfolioSymbolsMemoryCache = { symbols: syms, version: res.version, fetchedAt: Date.now() };
            for (const listener of modelPortfolioSymbolsListeners) listener(syms);
            return res;
        }),
    modelPortfolioUpdatePosition: (positionId: string, payload: ModelPortfolioPositionUpdatePayload, token: string) =>
        fetchApi<ModelPortfolioResponse>(`/model-portfolio/positions/${encodeURIComponent(positionId)}`, {
            method: 'PUT',
            headers: { Authorization: `Bearer ${token.trim()}` },
            body: JSON.stringify(payload),
            exposeErrorDetail: true,
        }).then((res) => {
            modelPortfolioMemoryCache = { payload: res, fetchedAt: Date.now() };
            modelPortfolioPerformanceMemoryCache = null;
            const syms = (res.active_positions || []).map((p) => p.symbol);
            modelPortfolioSymbolsMemoryCache = { symbols: syms, version: res.version, fetchedAt: Date.now() };
            for (const listener of modelPortfolioSymbolsListeners) listener(syms);
            return res;
        }),
    modelPortfolioClosePosition: (positionId: string, payload: ModelPortfolioPositionClosePayload, token: string) =>
        fetchApi<ModelPortfolioResponse>(`/model-portfolio/positions/${encodeURIComponent(positionId)}/close`, {
            method: 'POST',
            headers: { Authorization: `Bearer ${token.trim()}` },
            body: JSON.stringify(payload),
            exposeErrorDetail: true,
        }).then((res) => {
            modelPortfolioMemoryCache = { payload: res, fetchedAt: Date.now() };
            modelPortfolioPerformanceMemoryCache = null;
            const syms = (res.active_positions || []).map((p) => p.symbol);
            modelPortfolioSymbolsMemoryCache = { symbols: syms, version: res.version, fetchedAt: Date.now() };
            for (const listener of modelPortfolioSymbolsListeners) listener(syms);
            return res;
        }),
    modelPortfolioDeletePosition: (positionId: string, token: string) =>
        fetchApi<ModelPortfolioResponse>(`/model-portfolio/positions/${encodeURIComponent(positionId)}`, {
            method: 'DELETE',
            headers: { Authorization: `Bearer ${token.trim()}` },
            exposeErrorDetail: true,
        }).then((res) => {
            modelPortfolioMemoryCache = { payload: res, fetchedAt: Date.now() };
            modelPortfolioPerformanceMemoryCache = null;
            const syms = (res.active_positions || []).map((p) => p.symbol);
            modelPortfolioSymbolsMemoryCache = { symbols: syms, version: res.version, fetchedAt: Date.now() };
            for (const listener of modelPortfolioSymbolsListeners) listener(syms);
            return res;
        }),
    fundSearch: (q: string, limit = 50, options?: { signal?: AbortSignal }) => {
        const params = new URLSearchParams({ q, limit: String(limit) });
        return fetchApi<FundsResponse>(`/funds/search?${params.toString()}`, {
            signal: options?.signal,
        });
    },
    fundCategories: () => fetchApi<FundCategoriesResponse>('/funds/categories'),
    fundDetail: (fundCode: string) => fetchApi<FundDetail>(`/funds/${encodeURIComponent(fundCode)}`),
    fundYieldSummary: (fundCode: string, options?: { signal?: AbortSignal }) =>
        fetchFundResponse<FundYieldSummaryResponse>(`/funds/${encodeURIComponent(fundCode)}/yield-summary`, {
            signal: options?.signal,
            timeoutMs: 30000,
            exposeErrorDetail: true,
        }),
    fundPerformance: (fundCode: string, options?: { startDate?: string; endDate?: string; refresh?: boolean; signal?: AbortSignal }) => {
        const params = new URLSearchParams();
        if (options?.startDate) params.append('start_date', options.startDate);
        if (options?.endDate) params.append('end_date', options.endDate);
        if (options?.refresh) params.append('refresh', 'true');
        const query = params.toString();
        return fetchApi<FundPerformanceResponse>(
            `/funds/${encodeURIComponent(fundCode)}/performance${query ? `?${query}` : ''}`,
            { signal: options?.signal },
        );
    },
    fundPerformanceStatus: (fundCode: string, options?: { startDate?: string; endDate?: string }) => {
        const params = new URLSearchParams();
        if (options?.startDate) params.append('start_date', options.startDate);
        if (options?.endDate) params.append('end_date', options.endDate);
        const query = params.toString();
        return fetchApi<{ fund_code: string; status: FundHistoryJob['status']; history_job: FundHistoryJob | null }>(
            `/funds/${encodeURIComponent(fundCode)}/performance/status${query ? `?${query}` : ''}`,
        );
    },
    fundAllocations: (fundCode: string) =>
        fetchApi<FundAllocationsResponse>(`/funds/${encodeURIComponent(fundCode)}/allocations`),
    fundAllocationsHistory: (fundCode: string, lookbackDays = 30) =>
        fetchApi<FundAllocationsHistoryResponse>(
            `/funds/${encodeURIComponent(fundCode)}/allocations/history?lookback_days=${lookbackDays}`,
            {
                timeoutMs: 60000,
                exposeErrorDetail: true,
            },
        ),
    fundHoldings: async (fundCode: string, options?: { force?: boolean }) => {
        const normalizedCode = fundCode.trim().toUpperCase();
        const now = Date.now();
        const cached = fundHoldingsMemoryCache.get(normalizedCode);
        if (!options?.force && cached && now - cached.fetchedAt < FUND_HOLDINGS_MEMORY_CACHE_TTL_MS) {
            return cached.payload;
        }
        const existingRequest = fundHoldingsInFlight.get(normalizedCode);
        if (existingRequest) return existingRequest;
        const request = fetchFundResponse<FundHoldingsResponse>(`/funds/${encodeURIComponent(normalizedCode)}/holdings`)
            .then((payload) => {
                fundHoldingsMemoryCache.set(normalizedCode, { payload, fetchedAt: Date.now() });
                return payload;
            })
            .finally(() => {
                if (fundHoldingsInFlight.get(normalizedCode) === request) {
                    fundHoldingsInFlight.delete(normalizedCode);
                }
            });
        fundHoldingsInFlight.set(normalizedCode, request);
        return request;
    },
    fundHoldingsLive: async (fundCode: string) => {
        const normalizedCode = fundCode.trim().toUpperCase();
        const existingRequest = fundHoldingsLiveInFlight.get(normalizedCode);
        if (existingRequest) return existingRequest;
        const request = fetchApi<FundHoldingsLiveResponse>(`/funds/${encodeURIComponent(normalizedCode)}/holdings/live`)
            .finally(() => {
                if (fundHoldingsLiveInFlight.get(normalizedCode) === request) {
                    fundHoldingsLiveInFlight.delete(normalizedCode);
                }
            });
        fundHoldingsLiveInFlight.set(normalizedCode, request);
        return request;
    },

    kapCompanies: () => fetchApi<KapCompaniesResponse>('/kap/companies'),

    kapSnapshot: async (company: string, refresh = false, maxQuarters = 10, signal?: AbortSignal) => {
        const params = new URLSearchParams({ company, max_quarters: String(maxQuarters) });
        if (refresh) params.append('refresh', 'true');
        const payload = await fetchApi<KapSnapshotResponse>(`/kap/snapshot?${params.toString()}`, { signal });
        if (payload.ok && payload.quarters.length) {
            const key = company.trim().toUpperCase();
            const cached = kapSnapshotMemoryCache.get(key);
            if (!cached || payload.quarters.length >= cached.payload.quarters.length) {
                if (!cached && kapSnapshotMemoryCache.size >= 30) {
                    kapSnapshotMemoryCache.delete(kapSnapshotMemoryCache.keys().next().value!);
                }
                kapSnapshotMemoryCache.set(key, { payload, at: Date.now() });
            }
        }
        return payload;
    },

    kapValuationHistory: (request: KapValuationHistoryRequest, signal?: AbortSignal) =>
        fetchApi<KapValuationHistoryResponse>('/kap/valuation-history', {
            method: 'POST',
            body: JSON.stringify(request),
            signal,
            timeoutMs: 30000,
        }),

    kapOverviewCommentary: (request: KapOverviewCommentaryRequest, options?: { signal?: AbortSignal }) =>
        fetchApi<KapOverviewCommentaryResponse>('/kap/overview-commentary', {
            method: 'POST',
            body: JSON.stringify(request),
            signal: options?.signal,
            timeoutMs: OVERVIEW_COMMENTARY_TIMEOUT_MS,
            debugLabel: 'kap-overview-commentary',
        }),

    kapPrice: (symbol: string) =>
        fetchApi<{
            ok: boolean;
            symbol: string;
            price: number | null;
            prev_close: number | null;
            change: number | null;
            change_pct: number | null;
            currency: string;
            market_state: string;
            as_of?: string | null;
            error?: string;
        }>(`/kap/price?symbol=${encodeURIComponent(symbol)}`),
};
