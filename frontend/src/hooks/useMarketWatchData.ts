import { useCallback, useEffect, useSyncExternalStore } from 'react';
import { apiClient } from '../api/client';
import type {
    CommodityQuote,
    FxQuote,
    MarketWatchGlobalResponse,
    MarketWatchResponse,
} from '../api/types';
import { getAdaptiveRefreshMs } from '../utils/marketHours';

// The backend rebuilds /market/watch and the live quote endpoints on a 3
// second cache window, so every channel polls at the same cadence.
const MARKET_REFRESH_MS = 3_000;

export interface MarketWatchDataSnapshot {
    watch: MarketWatchResponse | null;
    global: MarketWatchGlobalResponse | null;
    commodities: CommodityQuote[] | null;
    fx: FxQuote[] | null;
    watchLoading: boolean;
    globalLoading: boolean;
    commoditiesLoading: boolean;
    fxLoading: boolean;
    watchError: string | null;
    globalError: string | null;
    commoditiesError: string | null;
    fxError: string | null;
}

const initialSnapshot: MarketWatchDataSnapshot = {
    watch: null,
    global: null,
    commodities: null,
    fx: null,
    watchLoading: false,
    globalLoading: false,
    commoditiesLoading: false,
    fxLoading: false,
    watchError: null,
    globalError: null,
    commoditiesError: null,
    fxError: null,
};

let snapshot = initialSnapshot;
const listeners = new Set<() => void>();

interface ChannelRuntime {
    inFlight: boolean;
    lastStartedAt: number;
}

const channelRuntime: Record<'watch' | 'global' | 'commodities' | 'fx', ChannelRuntime> = {
    watch: { inFlight: false, lastStartedAt: 0 },
    global: { inFlight: false, lastStartedAt: 0 },
    commodities: { inFlight: false, lastStartedAt: 0 },
    fx: { inFlight: false, lastStartedAt: 0 },
};

function emitChange(): void {
    for (const listener of listeners) listener();
}

function updateSnapshot(patch: Partial<MarketWatchDataSnapshot>): void {
    snapshot = { ...snapshot, ...patch };
    emitChange();
}

function subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

function getSnapshot(): MarketWatchDataSnapshot {
    return snapshot;
}

function toErrorMessage(error: unknown, fallback: string): string {
    return error instanceof Error ? error.message : fallback;
}

function canStart(lastStartedAt: number, minIntervalMs: number): boolean {
    return Date.now() - lastStartedAt >= minIntervalMs;
}

async function fetchWatch(options?: { force?: boolean }): Promise<void> {
    const runtime = channelRuntime.watch;
    if (runtime.inFlight) return;
    if (!options?.force && !canStart(runtime.lastStartedAt, MARKET_REFRESH_MS)) return;

    const showLoading = Boolean(options?.force) || snapshot.watch == null;
    runtime.inFlight = true;
    runtime.lastStartedAt = Date.now();
    if (showLoading) updateSnapshot({ watchLoading: true });

    try {
        const payload = await apiClient.marketWatch(options?.force ? { refresh: true } : undefined);
        updateSnapshot({ watch: payload, watchError: null });
    } catch (error) {
        updateSnapshot({ watchError: toErrorMessage(error, 'Borsa izleme verisi alınamadı.') });
    } finally {
        runtime.inFlight = false;
        if (showLoading) updateSnapshot({ watchLoading: false });
    }
}

async function fetchGlobal(options?: { force?: boolean }): Promise<void> {
    const runtime = channelRuntime.global;
    if (runtime.inFlight) return;
    if (!options?.force && !canStart(runtime.lastStartedAt, MARKET_REFRESH_MS)) return;

    const showLoading = Boolean(options?.force) || snapshot.global == null;
    runtime.inFlight = true;
    runtime.lastStartedAt = Date.now();
    if (showLoading) updateSnapshot({ globalLoading: true });

    try {
        const payload = await apiClient.marketWatchGlobal(options?.force ? { refresh: true } : undefined);
        updateSnapshot({ global: payload, globalError: null });
    } catch (error) {
        updateSnapshot({ globalError: toErrorMessage(error, 'Endeks verisi alınamadı.') });
    } finally {
        runtime.inFlight = false;
        if (showLoading) updateSnapshot({ globalLoading: false });
    }
}

async function fetchCommodities(options?: { force?: boolean }): Promise<void> {
    const runtime = channelRuntime.commodities;
    if (runtime.inFlight) return;
    if (!options?.force && !canStart(runtime.lastStartedAt, MARKET_REFRESH_MS)) return;

    const showLoading = Boolean(options?.force) || snapshot.commodities == null;
    runtime.inFlight = true;
    runtime.lastStartedAt = Date.now();
    if (showLoading) updateSnapshot({ commoditiesLoading: true });

    try {
        const payload = await apiClient.marketCommodities();
        updateSnapshot({ commodities: payload.items || [], commoditiesError: null });
    } catch (error) {
        updateSnapshot({ commoditiesError: toErrorMessage(error, 'Emtia verisi alınamadı.') });
    } finally {
        runtime.inFlight = false;
        if (showLoading) updateSnapshot({ commoditiesLoading: false });
    }
}

async function fetchFx(options?: { force?: boolean }): Promise<void> {
    const runtime = channelRuntime.fx;
    if (runtime.inFlight) return;
    if (!options?.force && !canStart(runtime.lastStartedAt, MARKET_REFRESH_MS)) return;

    const showLoading = Boolean(options?.force) || snapshot.fx == null;
    runtime.inFlight = true;
    runtime.lastStartedAt = Date.now();
    if (showLoading) updateSnapshot({ fxLoading: true });

    try {
        const payload = await apiClient.marketFx();
        updateSnapshot({ fx: payload.items || [], fxError: null });
    } catch (error) {
        updateSnapshot({ fxError: toErrorMessage(error, 'Döviz verisi alınamadı.') });
    } finally {
        runtime.inFlight = false;
        if (showLoading) updateSnapshot({ fxLoading: false });
    }
}

interface UseMarketWatchDataOptions {
    watch?: boolean;
    global?: boolean;
    commodities?: boolean;
    fx?: boolean;
    enabled?: boolean;
}

export interface MarketWatchDataResult extends MarketWatchDataSnapshot {
    refresh: () => void;
}

/**
 * Single shared data source for the market ticker and the Piyasalar rail.
 *
 * Both widgets previously polled /market/watch/global (and, for commodities and
 * FX, separate endpoints with their own timers), which doubled the request
 * volume and let the same instrument show two different prices at the same
 * time: /market/watch wraps its commodity/FX sections in an extra 3 second
 * cache, so the ticker lagged a few seconds behind the rail. Requests, caches
 * and polling now live here, so every consumer renders the exact same snapshot.
 */
export function useMarketWatchData(options: UseMarketWatchDataOptions): MarketWatchDataResult {
    const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
    const enabled = options.enabled !== false;
    const wantsWatch = enabled && Boolean(options.watch);
    const wantsGlobal = enabled && Boolean(options.global);
    const wantsCommodities = enabled && Boolean(options.commodities);
    const wantsFx = enabled && Boolean(options.fx);

    useEffect(() => {
        if (!wantsWatch && !wantsGlobal && !wantsCommodities && !wantsFx) return undefined;

        const tick = () => {
            if (document.visibilityState === 'hidden') return;
            if (getAdaptiveRefreshMs('global_fx') === null) return;
            if (wantsWatch) void fetchWatch();
            if (wantsGlobal) void fetchGlobal();
            if (wantsCommodities) void fetchCommodities();
            if (wantsFx) void fetchFx();
        };

        tick();

        const timer = window.setInterval(tick, MARKET_REFRESH_MS);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && getAdaptiveRefreshMs('global_fx') !== null) {
                tick();
            }
        };
        document.addEventListener('visibilitychange', onVisibilityChange);

        return () => {
            window.clearInterval(timer);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [wantsCommodities, wantsFx, wantsGlobal, wantsWatch]);

    const refresh = useCallback(() => {
        if (wantsWatch) void fetchWatch({ force: true });
        if (wantsGlobal) void fetchGlobal({ force: true });
        if (wantsCommodities) void fetchCommodities({ force: true });
        if (wantsFx) void fetchFx({ force: true });
    }, [wantsCommodities, wantsFx, wantsGlobal, wantsWatch]);

    return { ...state, refresh };
}
