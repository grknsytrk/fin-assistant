import { useCallback, useEffect, useSyncExternalStore } from 'react';
import { apiClient } from '../api/client';
import type {
    CommodityQuote,
    FxQuote,
    MarketWatchGlobalResponse,
    MarketWatchResponse,
} from '../api/types';
import { getAdaptiveRefreshMs, isGlobalFxTradingHours } from '../utils/marketHours';

// Rationalized polling cadence: global macro / delayed external feeds refresh every 30s
export const GLOBAL_MACRO_REFRESH_MS = 30_000;
export const MARKET_REFRESH_MS = GLOBAL_MACRO_REFRESH_MS;

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

type ChannelKey = 'watch' | 'global' | 'commodities' | 'fx';

interface ChannelRuntime {
    inFlight: boolean;
    lastStartedAt: number;
    pendingForce?: boolean;
}

const channelRuntime: Record<ChannelKey, ChannelRuntime> = {
    watch: { inFlight: false, lastStartedAt: 0 },
    global: { inFlight: false, lastStartedAt: 0 },
    commodities: { inFlight: false, lastStartedAt: 0 },
    fx: { inFlight: false, lastStartedAt: 0 },
};

const subscriberCounts: Record<ChannelKey, number> = {
    watch: 0,
    global: 0,
    commodities: 0,
    fx: 0,
};

let masterTimer: number | null = null;
let visibilityHandlerAttached = false;

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
    if (runtime.inFlight) {
        if (options?.force) runtime.pendingForce = true;
        return;
    }
    if (!options?.force && !canStart(runtime.lastStartedAt, GLOBAL_MACRO_REFRESH_MS)) return;

    const showLoading = Boolean(options?.force) || snapshot.watch == null;
    runtime.inFlight = true;
    runtime.lastStartedAt = Date.now();
    if (showLoading) updateSnapshot({ watchLoading: true });

    try {
        const payload = await apiClient.marketWatch(options?.force ? { refresh: true } : undefined);
        const watchPayload = payload?.sections ? payload : (snapshot.watch || payload);
        updateSnapshot({ watch: watchPayload, watchError: null });
    } catch (error) {
        updateSnapshot({ watchError: toErrorMessage(error, 'Borsa izleme verisi alınamadı.') });
    } finally {
        runtime.inFlight = false;
        if (showLoading) updateSnapshot({ watchLoading: false });
        if (runtime.pendingForce) {
            runtime.pendingForce = false;
            void fetchWatch({ force: true });
        }
    }
}

async function fetchGlobal(options?: { force?: boolean }): Promise<void> {
    const runtime = channelRuntime.global;
    if (runtime.inFlight) {
        if (options?.force) runtime.pendingForce = true;
        return;
    }
    if (!options?.force && !canStart(runtime.lastStartedAt, GLOBAL_MACRO_REFRESH_MS)) return;

    const showLoading = Boolean(options?.force) || snapshot.global == null;
    runtime.inFlight = true;
    runtime.lastStartedAt = Date.now();
    if (showLoading) updateSnapshot({ globalLoading: true });

    try {
        const payload = await apiClient.marketWatchGlobal(options?.force ? { refresh: true } : undefined);
        const globalPayload = payload?.items && payload.items.length > 0 ? payload : (snapshot.global || payload);
        updateSnapshot({ global: globalPayload, globalError: null });
    } catch (error) {
        updateSnapshot({ globalError: toErrorMessage(error, 'Endeks verisi alınamadı.') });
    } finally {
        runtime.inFlight = false;
        if (showLoading) updateSnapshot({ globalLoading: false });
        if (runtime.pendingForce) {
            runtime.pendingForce = false;
            void fetchGlobal({ force: true });
        }
    }
}

async function fetchCommodities(options?: { force?: boolean }): Promise<void> {
    const runtime = channelRuntime.commodities;
    if (runtime.inFlight) {
        if (options?.force) runtime.pendingForce = true;
        return;
    }
    if (!options?.force && !canStart(runtime.lastStartedAt, GLOBAL_MACRO_REFRESH_MS)) return;

    const showLoading = Boolean(options?.force) || snapshot.commodities == null;
    runtime.inFlight = true;
    runtime.lastStartedAt = Date.now();
    if (showLoading) updateSnapshot({ commoditiesLoading: true });

    try {
        const payload = await apiClient.marketCommodities(options?.force ? { refresh: true } : undefined);
        const items = payload?.items && payload.items.length > 0 ? payload.items : (snapshot.commodities || []);
        updateSnapshot({ commodities: items, commoditiesError: null });
    } catch (error) {
        updateSnapshot({ commoditiesError: toErrorMessage(error, 'Emtia verisi alınamadı.') });
    } finally {
        runtime.inFlight = false;
        if (showLoading) updateSnapshot({ commoditiesLoading: false });
        if (runtime.pendingForce) {
            runtime.pendingForce = false;
            void fetchCommodities({ force: true });
        }
    }
}

async function fetchFx(options?: { force?: boolean }): Promise<void> {
    const runtime = channelRuntime.fx;
    if (runtime.inFlight) {
        if (options?.force) runtime.pendingForce = true;
        return;
    }
    if (!options?.force && !canStart(runtime.lastStartedAt, GLOBAL_MACRO_REFRESH_MS)) return;

    const showLoading = Boolean(options?.force) || snapshot.fx == null;
    runtime.inFlight = true;
    runtime.lastStartedAt = Date.now();
    if (showLoading) updateSnapshot({ fxLoading: true });

    try {
        const payload = await apiClient.marketFx(options?.force ? { refresh: true } : undefined);
        const items = payload?.items && payload.items.length > 0 ? payload.items : (snapshot.fx || []);
        updateSnapshot({ fx: items, fxError: null });
    } catch (error) {
        updateSnapshot({ fxError: toErrorMessage(error, 'Döviz verisi alınamadı.') });
    } finally {
        runtime.inFlight = false;
        if (showLoading) updateSnapshot({ fxLoading: false });
        if (runtime.pendingForce) {
            runtime.pendingForce = false;
            void fetchFx({ force: true });
        }
    }
}

function masterTick(opts?: { isPolling?: boolean }): void {
    if (typeof document !== 'undefined' && document.visibilityState !== 'visible') return;
    if (opts?.isPolling && (!isGlobalFxTradingHours() || getAdaptiveRefreshMs('global_fx') === null)) return;

    if (subscriberCounts.watch > 0) void fetchWatch();
    if (subscriberCounts.global > 0) void fetchGlobal();
    if (subscriberCounts.commodities > 0) void fetchCommodities();
    if (subscriberCounts.fx > 0) void fetchFx();
}

function handleVisibilityChange(): void {
    if (typeof document !== 'undefined' && document.visibilityState === 'visible' && isGlobalFxTradingHours() && getAdaptiveRefreshMs('global_fx') !== null) {
        masterTick({ isPolling: true });
    }
}

function updateMasterTimer(): void {
    const total =
        subscriberCounts.watch +
        subscriberCounts.global +
        subscriberCounts.commodities +
        subscriberCounts.fx;

    if (total > 0) {
        if (masterTimer === null && typeof window !== 'undefined') {
            masterTimer = window.setInterval(() => masterTick({ isPolling: true }), GLOBAL_MACRO_REFRESH_MS);
        }
        if (!visibilityHandlerAttached && typeof document !== 'undefined') {
            document.addEventListener('visibilitychange', handleVisibilityChange);
            visibilityHandlerAttached = true;
        }
    } else {
        if (masterTimer !== null && typeof window !== 'undefined') {
            window.clearInterval(masterTimer);
            masterTimer = null;
        }
        if (visibilityHandlerAttached && typeof document !== 'undefined') {
            document.removeEventListener('visibilitychange', handleVisibilityChange);
            visibilityHandlerAttached = false;
        }
    }
}

function registerChannels(channels: ChannelKey[]): void {
    for (const ch of channels) {
        const wasZero = subscriberCounts[ch] === 0;
        subscriberCounts[ch] += 1;
        if (wasZero) {
            if (ch === 'watch') void fetchWatch();
            else if (ch === 'global') void fetchGlobal();
            else if (ch === 'commodities') void fetchCommodities();
            else if (ch === 'fx') void fetchFx();
        }
    }
    updateMasterTimer();
}

function unregisterChannels(channels: ChannelKey[]): void {
    for (const ch of channels) {
        subscriberCounts[ch] = Math.max(0, subscriberCounts[ch] - 1);
    }
    updateMasterTimer();
}

export function _resetMarketWatchStoreForTesting(): void {
    snapshot = initialSnapshot;
    listeners.clear();
    for (const key of Object.keys(channelRuntime) as ChannelKey[]) {
        channelRuntime[key].inFlight = false;
        channelRuntime[key].lastStartedAt = 0;
        channelRuntime[key].pendingForce = false;
        subscriberCounts[key] = 0;
    }
    if (masterTimer !== null && typeof window !== 'undefined') {
        window.clearInterval(masterTimer);
        masterTimer = null;
    }
    if (visibilityHandlerAttached && typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', handleVisibilityChange);
        visibilityHandlerAttached = false;
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
        const keys: ChannelKey[] = [];
        if (wantsWatch) keys.push('watch');
        if (wantsGlobal) keys.push('global');
        if (wantsCommodities) keys.push('commodities');
        if (wantsFx) keys.push('fx');

        if (keys.length === 0) return undefined;

        registerChannels(keys);
        return () => {
            unregisterChannels(keys);
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
