import { useCallback, useSyncExternalStore } from 'react';
import { apiClient } from '../api/client';
import type { MarketWatchItem, UnifiedMarketWatchResponse } from '../api/types';
import { getAdaptiveRefreshMs, isGlobalFxTradingHours } from '../utils/marketHours';

export const UNIFIED_MARKET_REFRESH_MS = 30_000;
export const DEFAULT_DELAY_NOTE =
    'Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk). BIST verileri seans sırasında canlıdır.';

export interface UnifiedMarketSnapshot {
    indices: MarketWatchItem[];
    fx: MarketWatchItem[];
    commodities: MarketWatchItem[];
    // O(1) symbol lookup maps:
    indicesBySymbol: Map<string, MarketWatchItem>;
    fxBySymbol: Map<string, MarketWatchItem>;
    commoditiesBySymbol: Map<string, MarketWatchItem>;
    itemsBySymbol: Map<string, MarketWatchItem>;
    as_of: string | null;
    delay_note: string;
    source: string;
    loading: boolean;
    error: string | null;
    lastFetchedAt: number;
}

const initialSnapshot: UnifiedMarketSnapshot = {
    indices: [],
    fx: [],
    commodities: [],
    indicesBySymbol: new Map(),
    fxBySymbol: new Map(),
    commoditiesBySymbol: new Map(),
    itemsBySymbol: new Map(),
    as_of: null,
    delay_note: DEFAULT_DELAY_NOTE,
    source: '',
    loading: false,
    error: null,
    lastFetchedAt: 0,
};

let currentSnapshot: UnifiedMarketSnapshot = initialSnapshot;
const listeners = new Set<() => void>();

let masterTimer: number | null = null;
let visibilityHandlerAttached = false;
let inFlight = false;
let pendingForce = false;
let subscriberCount = 0;

function emitChange(): void {
    for (const listener of listeners) {
        listener();
    }
}

function updateSnapshot(patch: Partial<UnifiedMarketSnapshot>): void {
    currentSnapshot = { ...currentSnapshot, ...patch };
    emitChange();
}

export function normalizeSymbol(raw: string): string {
    return String(raw || '').trim().toUpperCase();
}

export function subscribe(listener: () => void): () => void {
    listeners.add(listener);
    subscriberCount += 1;
    if (subscriberCount === 1) {
        startCoordination();
    }
    return () => {
        listeners.delete(listener);
        subscriberCount = Math.max(0, subscriberCount - 1);
        if (subscriberCount === 0) {
            stopCoordination();
        }
    };
}

export function getSnapshot(): UnifiedMarketSnapshot {
    return currentSnapshot;
}

export async function fetchUnifiedMarketWatch(options?: { force?: boolean }): Promise<void> {
    if (inFlight) {
        if (options?.force) {
            pendingForce = true;
        }
        return;
    }

    const isStale = Date.now() - currentSnapshot.lastFetchedAt >= UNIFIED_MARKET_REFRESH_MS;
    if (!options?.force && !isStale && currentSnapshot.as_of !== null) {
        return;
    }

    const showLoading = Boolean(options?.force) || currentSnapshot.as_of === null;
    inFlight = true;
    if (showLoading) {
        updateSnapshot({ loading: true });
    }

    try {
        const payload: UnifiedMarketWatchResponse = await apiClient.marketWatchUnified(
            options?.force ? { refresh: true } : undefined,
        );

        const indicesBySymbol = new Map<string, MarketWatchItem>();
        const fxBySymbol = new Map<string, MarketWatchItem>();
        const commoditiesBySymbol = new Map<string, MarketWatchItem>();
        const itemsBySymbol = new Map<string, MarketWatchItem>();

        const rawIndices = payload.indices || payload.sections?.indices || [];
        const rawFx = payload.fx || payload.sections?.fx || [];
        const rawCommodities = payload.commodities || payload.sections?.commodities || [];

        for (const item of rawIndices) {
            const sym = normalizeSymbol(item.symbol);
            if (!sym) continue;
            indicesBySymbol.set(sym, item);
            itemsBySymbol.set(sym, item);
        }

        for (const item of rawFx) {
            const sym = normalizeSymbol(item.symbol);
            if (!sym) continue;
            fxBySymbol.set(sym, item);
            itemsBySymbol.set(sym, item);
            const noSlash = sym.replace('/', '');
            if (noSlash !== sym) {
                fxBySymbol.set(noSlash, item);
                itemsBySymbol.set(noSlash, item);
            }
        }

        for (const item of rawCommodities) {
            const sym = normalizeSymbol(item.symbol);
            if (!sym) continue;
            commoditiesBySymbol.set(sym, item);
            itemsBySymbol.set(sym, item);
        }

        currentSnapshot = {
            indices: rawIndices,
            fx: rawFx,
            commodities: rawCommodities,
            indicesBySymbol,
            fxBySymbol,
            commoditiesBySymbol,
            itemsBySymbol,
            as_of: payload.as_of || new Date().toISOString(),
            delay_note: payload.delay_note || currentSnapshot.delay_note,
            source: payload.source || 'unified',
            loading: false,
            error: null,
            lastFetchedAt: Date.now(),
        };
        emitChange();
    } catch (error) {
        // Fail-stale: preserve cached data, surface soft error
        const message = error instanceof Error ? error.message : 'Piyasa verisi alınamadı.';
        updateSnapshot({
            loading: false,
            error: currentSnapshot.as_of ? null : message,
        });
    } finally {
        inFlight = false;
        if (pendingForce) {
            pendingForce = false;
            void fetchUnifiedMarketWatch({ force: true });
        }
    }
}

function pollTick(): void {
    if (typeof document !== 'undefined' && document.visibilityState !== 'visible') {
        return;
    }
    if (!isGlobalFxTradingHours() || getAdaptiveRefreshMs('global_fx') === null) {
        return;
    }
    void fetchUnifiedMarketWatch();
}

function handleVisibility(): void {
    if (typeof document !== 'undefined' && document.visibilityState === 'visible') {
        const isStale = Date.now() - currentSnapshot.lastFetchedAt >= UNIFIED_MARKET_REFRESH_MS;
        if (isStale && isGlobalFxTradingHours() && getAdaptiveRefreshMs('global_fx') !== null) {
            void fetchUnifiedMarketWatch();
        }
    }
}

function startCoordination(): void {
    if (masterTimer === null && typeof window !== 'undefined') {
        masterTimer = window.setInterval(pollTick, UNIFIED_MARKET_REFRESH_MS);
    }
    if (!visibilityHandlerAttached && typeof document !== 'undefined') {
        document.addEventListener('visibilitychange', handleVisibility);
        visibilityHandlerAttached = true;
    }
    if (currentSnapshot.as_of === null) {
        void fetchUnifiedMarketWatch();
    }
}

function stopCoordination(): void {
    if (masterTimer !== null && typeof window !== 'undefined') {
        window.clearInterval(masterTimer);
        masterTimer = null;
    }
    if (visibilityHandlerAttached && typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', handleVisibility);
        visibilityHandlerAttached = false;
    }
}

export function applyMarketIndexDelta(symbol: string, patch: Partial<MarketWatchItem>): void {
    const norm = normalizeSymbol(symbol);
    const existing = currentSnapshot.itemsBySymbol.get(norm);
    if (!existing) return;

    const updated: MarketWatchItem = { ...existing, ...patch };
    const nextItemsBySymbol = new Map(currentSnapshot.itemsBySymbol);
    nextItemsBySymbol.set(norm, updated);

    const nextIndices = currentSnapshot.indices.map((item) =>
        normalizeSymbol(item.symbol) === norm ? updated : item,
    );
    const nextIndicesBySymbol = new Map(currentSnapshot.indicesBySymbol);
    nextIndicesBySymbol.set(norm, updated);

    currentSnapshot = {
        ...currentSnapshot,
        indices: nextIndices,
        indicesBySymbol: nextIndicesBySymbol,
        itemsBySymbol: nextItemsBySymbol,
    };
    emitChange();
}

export function useMarketStore(): UnifiedMarketSnapshot & { refresh: () => void } {
    const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
    const refresh = useCallback(() => {
        void fetchUnifiedMarketWatch({ force: true });
    }, []);
    return { ...state, refresh };
}

export function useMarketItem(symbol: string): MarketWatchItem | undefined {
    const store = useMarketStore();
    return store.itemsBySymbol.get(normalizeSymbol(symbol));
}

export function _resetMarketStoreForTesting(): void {
    currentSnapshot = {
        indices: [],
        fx: [],
        commodities: [],
        indicesBySymbol: new Map(),
        fxBySymbol: new Map(),
        commoditiesBySymbol: new Map(),
        itemsBySymbol: new Map(),
        as_of: null,
        delay_note: DEFAULT_DELAY_NOTE,
        source: '',
        loading: false,
        error: null,
        lastFetchedAt: 0,
    };
    listeners.clear();
    subscriberCount = 0;
    inFlight = false;
    pendingForce = false;
    stopCoordination();
}
