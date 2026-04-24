import type {
    AskRequest,
    AskResponse,
    StatsResponse,
    FeedbackRequest,
    KapSnapshotResponse,
    CompanyBreakdownResponse,
    MarketUniverseResponse,
    MarketStocksResponse,
    MarketIndexCode,
    MarketIndicesResponse,
    MarketIndexDetailResponse,
    MarketFlowResponse,
    MarketWatchResponse,
    MarketCommoditiesResponse,
    MarketIndexResponse,
    MarketFxResponse,
} from './types';

const rawApiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();
const API_BASE = (rawApiBase && rawApiBase !== '/' ? rawApiBase : 'http://localhost:8000').replace(/\/+$/, '');

const RETRYABLE_HTTP_STATUSES = new Set([502, 503, 504]);
const STARTUP_HINT = 'Uygulama yeni baslatiliyor olabilir. Lutfen 5-10 saniye bekleyip tekrar deneyin.';
const NETWORK_ERROR_MESSAGE = 'Su an baglanti kurulamiyor. Lutfen kisa bir sure sonra tekrar deneyin.';
const SERVER_ERROR_MESSAGE = 'Islem su an tamamlanamiyor. Lutfen daha sonra tekrar deneyin.';
const REQUEST_TIMEOUT_MS = 15000;
const TIMEOUT_MESSAGE = 'Istek suresi asildi. Lutfen tekrar deneyin.';

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

async function fetchApi<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const normalizedEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    const url = `${API_BASE}${normalizedEndpoint}`;
    const method = (options.method || 'GET').toUpperCase();
    const allowRetry = isIdempotentMethod(method);
    const maxAttempts = allowRetry ? 8 : 1;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        let response: Response;
        const controller = new AbortController();
        const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
        try {
            response = await fetch(url, {
                ...options,
                signal: options.signal ?? controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers,
                },
            });
        } catch (error) {
            if ((error as Error)?.name === 'AbortError') {
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
            throw new Error(`${NETWORK_ERROR_MESSAGE} ${STARTUP_HINT}`);
        } finally {
            window.clearTimeout(timeoutId);
        }

        if (!response.ok) {
            if (allowRetry && RETRYABLE_HTTP_STATUSES.has(response.status) && attempt < maxAttempts) {
                await sleep(retryDelayMs(attempt));
                continue;
            }

            const errorData = await response.json().catch(() => ({}));
            const detail = typeof errorData?.detail === 'string' ? errorData.detail : null;
            if (detail && response.status < 500) {
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

        return response.json();
    }

    throw new Error(STARTUP_HINT);
}

export const apiClient = {
    health: () => fetchApi<{ status: string }>('/health'),

    stats: () => fetchApi<StatsResponse>('/stats'),
    companyBreakdown: () => fetchApi<CompanyBreakdownResponse>('/stats/company-breakdown'),
    marketUniverse: () => fetchApi<MarketUniverseResponse>('/market/universe'),
    marketStocks: (options?: { index?: 'XU100' | 'XU030'; refresh?: boolean }) => {
        const params = new URLSearchParams();
        if (options?.index) params.append('index', options.index);
        if (options?.refresh) params.append('refresh', 'true');
        const query = params.toString();
        return fetchApi<MarketStocksResponse>(query ? `/market/stocks?${query}` : '/market/stocks');
    },
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
    marketXu030: () => fetchApi<MarketIndexResponse>('/market/xu030'),
    marketCommodities: () => fetchApi<MarketCommoditiesResponse>('/market/commodities'),
    marketFx: () => fetchApi<MarketFxResponse>('/market/fx'),

    ask: (request: AskRequest) =>
        fetchApi<AskResponse>('/ask', {
            method: 'POST',
            body: JSON.stringify(request),
        }),

    feedback: (request: FeedbackRequest) =>
        fetchApi<{ message: string; path: string; feedback: any }>('/feedback', {
            method: 'POST',
            body: JSON.stringify(request),
        }),

    commentary: (request: {
        question: string;
        answer_payload: Record<string, any>;
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

    kapCompanies: () => fetchApi<{ companies: string[] }>('/kap/companies'),

    kapSnapshot: (company: string, refresh = false, maxQuarters = 10) => {
        const params = new URLSearchParams({ company, max_quarters: String(maxQuarters) });
        if (refresh) params.append('refresh', 'true');
        return fetchApi<KapSnapshotResponse>(`/kap/snapshot?${params.toString()}`);
    },

    ingest: () => fetchApi<{ message: string; pages_written: number; summary: any }>('/ingest', { method: 'POST' }),

    index: (version: 'v1' | 'v2' = 'v2') =>
        fetchApi<{ message: string; version: string; summary: any }>('/index', {
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
