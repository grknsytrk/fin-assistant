import type {
    AskRequest,
    AskResponse,
    StatsResponse,
    FeedbackRequest,
    KapCompaniesResponse,
    KapSnapshotResponse,
    CompanyBreakdownResponse,
    MarketUniverseResponse,
    MarketStocksResponse,
    MarketStockIndex,
    MarketIndexCode,
    MarketIndicesResponse,
    MarketIndexDetailResponse,
    MarketFlowResponse,
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

const RETRYABLE_HTTP_STATUSES = new Set([502, 503, 504]);
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

type FetchApiOptions = RequestInit & {
    timeoutMs?: number;
    debugLabel?: string;
    exposeErrorDetail?: boolean;
};

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

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

function retryDelayMs(attempt: number): number {
    const base = 250;
    const max = 1800;
    return Math.min(max, base * Math.pow(1.6, Math.max(0, attempt - 1)));
}

async function fetchApi<T>(endpoint: string, options: FetchApiOptions = {}): Promise<T> {
    const normalizedEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    const url = `${API_BASE}${normalizedEndpoint}`;
    const method = (options.method || 'GET').toUpperCase();
    const allowRetry = isIdempotentMethod(method);
    const maxAttempts = allowRetry ? 8 : 1;
    const { timeoutMs, debugLabel, exposeErrorDetail, ...requestOptions } = options;
    const effectiveTimeoutMs = timeoutMs ?? REQUEST_TIMEOUT_MS;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        let response: Response;
        const controller = new AbortController();
        const externalSignal = requestOptions.signal;
        const startedAt = window.performance?.now?.() ?? Date.now();
        const timeoutId = window.setTimeout(() => controller.abort(), effectiveTimeoutMs);
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
                    await sleep(retryDelayMs(attempt));
                    continue;
                }
                throw new Error(TIMEOUT_MESSAGE);
            }
            if (allowRetry && isRetryableNetworkError(error) && attempt < maxAttempts) {
                await sleep(retryDelayMs(attempt));
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
                await sleep(retryDelayMs(attempt));
                continue;
            }

            const errorData = await response.json().catch(() => ({}));
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
        return response.json();
    }

    throw new Error(STARTUP_HINT);
}

export const apiClient = {
    health: () => fetchApi<{ status: string }>('/health'),

    stats: () => fetchApi<StatsResponse>('/stats'),
    companyBreakdown: () => fetchApi<CompanyBreakdownResponse>('/stats/company-breakdown'),
    marketUniverse: () => fetchApi<MarketUniverseResponse>('/market/universe'),
    marketStocks: (options?: { index?: MarketStockIndex; refresh?: boolean }) => {
        const params = new URLSearchParams();
        if (options?.index) params.append('index', options.index);
        if (options?.refresh) params.append('refresh', 'true');
        const query = params.toString();
        return fetchApi<MarketStocksResponse>(query ? `/market/stocks?${query}` : '/market/stocks');
    },
    marketStockCards: (options: { symbols: string[]; refresh?: boolean }) => {
        const params = new URLSearchParams();
        params.append('symbols', options.symbols.join(','));
        if (options.refresh) params.append('refresh', 'true');
        return fetchApi<MarketStockCardsResponse>(`/market/stocks/cards?${params.toString()}`);
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
    marketFlow: (limit = 40, category?: string, options?: { refresh?: boolean }) => {
        const params = new URLSearchParams({ limit: String(limit) });
        if (category) params.append('category', category);
        if (options?.refresh) params.append('refresh', 'true');
        return fetchApi<MarketFlowResponse>(`/market/flow?${params.toString()}`);
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
        return fetchApi<FundsResponse>(query ? `/funds?${query}` : '/funds');
    },
    fundSearch: (q: string, limit = 50, options?: { signal?: AbortSignal }) => {
        const params = new URLSearchParams({ q, limit: String(limit) });
        return fetchApi<FundsResponse>(`/funds/search?${params.toString()}`, {
            signal: options?.signal,
        });
    },
    fundCategories: () => fetchApi<FundCategoriesResponse>('/funds/categories'),
    refreshFundsSnapshot: (lookbackDays = 10) =>
        fetchApi<FundsResponse>(`/admin/funds/refresh-snapshot?lookback_days=${lookbackDays}`, {
            method: 'POST',
            timeoutMs: 120000,
            exposeErrorDetail: true,
        }),
    fundDetail: (fundCode: string) => fetchApi<FundDetail>(`/funds/${encodeURIComponent(fundCode)}`),
    fundYieldSummary: (fundCode: string) =>
        fetchApi<FundYieldSummaryResponse>(`/funds/${encodeURIComponent(fundCode)}/yield-summary`, {
            timeoutMs: 30000,
            exposeErrorDetail: true,
        }),
    fundPerformance: (fundCode: string, options?: { startDate?: string; endDate?: string }) => {
        const params = new URLSearchParams();
        if (options?.startDate) params.append('start_date', options.startDate);
        if (options?.endDate) params.append('end_date', options.endDate);
        const query = params.toString();
        return fetchApi<FundPerformanceResponse>(
            `/funds/${encodeURIComponent(fundCode)}/performance${query ? `?${query}` : ''}`,
        );
    },
    refreshFundPerformance: (fundCode: string, startDate: string, endDate?: string) => {
        const params = new URLSearchParams({ start_date: startDate });
        if (endDate) params.append('end_date', endDate);
        return fetchApi<FundPerformanceResponse>(
            `/admin/funds/${encodeURIComponent(fundCode)}/refresh-performance?${params.toString()}`,
            {
                method: 'POST',
                timeoutMs: 180000,
                exposeErrorDetail: true,
            },
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
    refreshFundAllocations: (fundCode: string, asOf?: string) => {
        const params = new URLSearchParams();
        if (asOf) params.append('as_of', asOf);
        const suffix = params.toString() ? `?${params.toString()}` : '';
        return fetchApi<FundAllocationsResponse>(
            `/admin/funds/${encodeURIComponent(fundCode)}/refresh-allocations${suffix}`,
            {
                method: 'POST',
                timeoutMs: 45000,
                exposeErrorDetail: true,
            },
        );
    },
    fundHoldings: async (fundCode: string, options?: { force?: boolean }) => {
        const normalizedCode = fundCode.trim().toUpperCase();
        const now = Date.now();
        const cached = fundHoldingsMemoryCache.get(normalizedCode);
        if (!options?.force && cached && now - cached.fetchedAt < FUND_HOLDINGS_MEMORY_CACHE_TTL_MS) {
            return cached.payload;
        }
        const existingRequest = fundHoldingsInFlight.get(normalizedCode);
        if (existingRequest) return existingRequest;
        const request = fetchApi<FundHoldingsResponse>(`/funds/${encodeURIComponent(normalizedCode)}/holdings`)
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

    ask: (request: AskRequest) =>
        fetchApi<AskResponse>('/ask', {
            method: 'POST',
            body: JSON.stringify(request),
    }),

    feedback: (request: FeedbackRequest) =>
        fetchApi<{ message: string; path: string; feedback: unknown }>('/feedback', {
            method: 'POST',
            body: JSON.stringify(request),
        }),

    commentary: (request: {
        question: string;
        answer_payload: Record<string, unknown>;
        company?: string;
        year?: string;
        quarter?: string;
        model?: string;
    }) =>
        fetchApi<{ commentary: string; model_used: string; error?: string }>('/commentary', {
            method: 'POST',
            body: JSON.stringify(request),
        }),

    exportUrl: (type: 'trend' | 'ratio', company?: string) => {
        const params = new URLSearchParams({ type });
        if (company) {
            params.append('company', company);
        }
        return `${API_BASE}/export?${params.toString()}`;
    },

    kapCompanies: () => fetchApi<KapCompaniesResponse>('/kap/companies'),

    kapSnapshot: (company: string, refresh = false, maxQuarters = 10) => {
        const params = new URLSearchParams({ company, max_quarters: String(maxQuarters) });
        if (refresh) params.append('refresh', 'true');
        return fetchApi<KapSnapshotResponse>(`/kap/snapshot?${params.toString()}`);
    },

    kapOverviewCommentary: (request: KapOverviewCommentaryRequest, options?: { signal?: AbortSignal }) =>
        fetchApi<KapOverviewCommentaryResponse>('/kap/overview-commentary', {
            method: 'POST',
            body: JSON.stringify(request),
            signal: options?.signal,
            timeoutMs: OVERVIEW_COMMENTARY_TIMEOUT_MS,
            debugLabel: 'kap-overview-commentary',
        }),

    ingest: () => fetchApi<{ message: string; pages_written: number; summary: unknown }>('/ingest', { method: 'POST' }),

    index: (version: 'v1' | 'v2' = 'v2') =>
        fetchApi<{ message: string; version: string; summary: unknown }>('/index', {
            method: 'POST',
            body: JSON.stringify({ version }),
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
