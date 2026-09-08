import type { FundSummary } from '../api/types';
import { apiClient } from '../api/client';

const WATCHLIST_FUND_CACHE_TTL_MS = 60_000;
const WATCHLIST_FUND_SEARCH_CACHE_TTL_MS = 5 * 60_000;

const watchlistFundCache = new Map<string, { data: FundSummary; fetchedAt: number }>();
const watchlistFundInFlight = new Map<string, Promise<FundSummary>>();
const watchlistFundSearchCache = new Map<string, { data: FundSummary[]; fetchedAt: number }>();
const watchlistFundSearchInFlight = new Map<string, Promise<FundSummary[]>>();

export function getCachedWatchlistFund(code: string): FundSummary | null {
    const cached = watchlistFundCache.get(code);
    if (!cached || Date.now() - cached.fetchedAt > WATCHLIST_FUND_CACHE_TTL_MS) return null;
    return cached.data;
}

export function cacheWatchlistFund(fund: FundSummary): void {
    watchlistFundCache.set(fund.fund_code, { data: fund, fetchedAt: Date.now() });
}

export function fetchWatchlistFund(code: string): Promise<FundSummary> {
    const cached = getCachedWatchlistFund(code);
    if (cached) return Promise.resolve(cached);

    const existing = watchlistFundInFlight.get(code);
    if (existing) return existing;

    const request = apiClient.fundDetail(code)
        .then((detail) => {
            cacheWatchlistFund(detail);
            return detail;
        })
        .finally(() => {
            if (watchlistFundInFlight.get(code) === request) watchlistFundInFlight.delete(code);
        });
    watchlistFundInFlight.set(code, request);
    return request;
}

export function searchWatchlistFunds(query: string): Promise<FundSummary[]> {
    const normalizedQuery = query.trim().toUpperCase();
    const cached = watchlistFundSearchCache.get(normalizedQuery);
    if (cached && Date.now() - cached.fetchedAt <= WATCHLIST_FUND_SEARCH_CACHE_TTL_MS) {
        return Promise.resolve(cached.data);
    }

    const existing = watchlistFundSearchInFlight.get(normalizedQuery);
    if (existing) return existing;

    const request = apiClient.fundSearch(normalizedQuery, 20)
        .then((payload) => {
            const rows = payload.rows || [];
            watchlistFundSearchCache.set(normalizedQuery, { data: rows, fetchedAt: Date.now() });
            rows.forEach(cacheWatchlistFund);
            return rows;
        })
        .finally(() => {
            if (watchlistFundSearchInFlight.get(normalizedQuery) === request) {
                watchlistFundSearchInFlight.delete(normalizedQuery);
            }
        });
    watchlistFundSearchInFlight.set(normalizedQuery, request);
    return request;
}
