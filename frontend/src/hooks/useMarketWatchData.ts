import { useCallback, useEffect, useSyncExternalStore } from 'react';
import { apiClient } from '../api/client';
import type { MarketWatchGlobalResponse, MarketWatchResponse } from '../api/types';
import { getAdaptiveRefreshMs } from '../utils/marketHours';

// The backend rebuilds /market/watch on a 3 second cache window.
const WATCH_REFRESH_MS = 3_000;
// /market/watch/global is cached longer upstream, but the widget still polls
// at the same cadence so it picks up a refreshed snapshot immediately after
// the stale-while-revalidate cycle completes.
const GLOBAL_REFRESH_MS = 3_000;

export interface MarketWatchDataSnapshot {
    watch: MarketWatchResponse | null;
    global: MarketWatchGlobalResponse | null;
    watchLoading: boolean;
    globalLoading: boolean;
    watchError: string | null;
    globalError: string | null;
}

const initialSnapshot: MarketWatchDataSnapshot = {
    watch: null,
    global: null,
    watchLoading: false,
    globalLoading: false,
    watchError: null,
    globalError: null,
};

let snapshot = initialSnapshot;
const listeners = new Set<() => void>();

let watchInFlight = false;
let globalInFlight = false;
let watchLastStartedAt = 0;
let globalLastStartedAt = 0;

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
    if (watchInFlight) return;
    if (!options?.force && !canStart(watchLastStartedAt, WATCH_REFRESH_MS)) return;

    const showLoading = Boolean(options?.force) || snapshot.watch == null;
    watchInFlight = true;
    watchLastStartedAt = Date.now();
    if (showLoading) updateSnapshot({ watchLoading: true });

    try {
        const payload = await apiClient.marketWatch(options?.force ? { refresh: true } : undefined);
        updateSnapshot({ watch: payload, watchError: null });
    } catch (error) {
        updateSnapshot({ watchError: toErrorMessage(error, 'Borsa izleme verisi alınamadı.') });
    } finally {
        watchInFlight = false;
        if (showLoading) updateSnapshot({ watchLoading: false });
    }
}

async function fetchGlobal(options?: { force?: boolean }): Promise<void> {
    if (globalInFlight) return;
    if (!options?.force && !canStart(globalLastStartedAt, GLOBAL_REFRESH_MS)) return;

    const showLoading = Boolean(options?.force) || snapshot.global == null;
    globalInFlight = true;
    globalLastStartedAt = Date.now();
    if (showLoading) updateSnapshot({ globalLoading: true });

    try {
        const payload = await apiClient.marketWatchGlobal(options?.force ? { refresh: true } : undefined);
        updateSnapshot({ global: payload, globalError: null });
    } catch (error) {
        updateSnapshot({ globalError: toErrorMessage(error, 'Endeks verisi alınamadı.') });
    } finally {
        globalInFlight = false;
        if (showLoading) updateSnapshot({ globalLoading: false });
    }
}

interface UseMarketWatchDataOptions {
    watch?: boolean;
    global?: boolean;
    enabled?: boolean;
}

export interface MarketWatchDataResult extends MarketWatchDataSnapshot {
    refresh: () => void;
}

/**
 * Single shared data source for the market ticker and the Piyasalar rail.
 *
 * Both widgets previously polled /market/watch/global independently, which
 * doubled the request volume and let the same index show two different
 * prices at the same time. Requests, caches and polling now live here, so
 * every consumer renders the exact same snapshot.
 */
export function useMarketWatchData(options: UseMarketWatchDataOptions): MarketWatchDataResult {
    const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
    const enabled = options.enabled !== false;
    const wantsWatch = enabled && Boolean(options.watch);
    const wantsGlobal = enabled && Boolean(options.global);

    useEffect(() => {
        if (!wantsWatch) return undefined;

        void fetchWatch();

        const timer = window.setInterval(() => {
            if (document.visibilityState === 'hidden') return;
            if (getAdaptiveRefreshMs('global_fx') === null) return;
            void fetchWatch();
        }, WATCH_REFRESH_MS);

        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && getAdaptiveRefreshMs('global_fx') !== null) {
                void fetchWatch();
            }
        };
        document.addEventListener('visibilitychange', onVisibilityChange);

        return () => {
            window.clearInterval(timer);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [wantsWatch]);

    useEffect(() => {
        if (!wantsGlobal) return undefined;

        void fetchGlobal();

        const timer = window.setInterval(() => {
            if (document.visibilityState === 'hidden') return;
            if (getAdaptiveRefreshMs('global_fx') === null) return;
            void fetchGlobal();
        }, GLOBAL_REFRESH_MS);

        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && getAdaptiveRefreshMs('global_fx') !== null) {
                void fetchGlobal();
            }
        };
        document.addEventListener('visibilitychange', onVisibilityChange);

        return () => {
            window.clearInterval(timer);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [wantsGlobal]);

    const refresh = useCallback(() => {
        if (wantsWatch) void fetchWatch({ force: true });
        if (wantsGlobal) void fetchGlobal({ force: true });
    }, [wantsGlobal, wantsWatch]);

    return { ...state, refresh };
}
