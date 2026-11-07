import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../api/client';
import type { MarketWatchItem, UnifiedMarketWatchResponse } from '../api/types';
import {
    _resetMarketStoreForTesting,
    applyMarketIndexDelta,
    fetchUnifiedMarketWatch,
    getSnapshot,
    normalizeSymbol,
    subscribe,
    useMarketItem,
    useMarketStore,
} from './marketStore';

const mockItem = (symbol: string, price: number, changePct: number): MarketWatchItem => ({
    symbol,
    label: symbol,
    yahoo_symbol: symbol,
    price,
    prev_close: price - 1,
    change: 1,
    change_pct: changePct,
    currency: 'TRY',
    market_state: 'REGULAR',
    as_of: '2026-11-07T18:00:00Z',
    error: null,
});

const mockPayload: UnifiedMarketWatchResponse = {
    indices: [mockItem('XU100', 9250.5, 1.25), mockItem('SP500', 5800.0, 0.5)],
    fx: [mockItem('USD/TRY', 38.45, 0.2), mockItem('EUR/TRY', 41.15, -0.1)],
    commodities: [mockItem('ALTIN', 3966.2, 0.4), mockItem('BRENT', 74.5, -0.8)],
    source: 'yahoo_finance_chart',
    delay_note: 'Test gecikme notu',
    as_of: '2026-11-07T18:00:00Z',
};

describe('marketStore', () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        _resetMarketStoreForTesting();
    });

    afterEach(() => {
        cleanup();
        vi.restoreAllMocks();
        _resetMarketStoreForTesting();
    });

    it('initializes with empty collections and null as_of', () => {
        const snapshot = getSnapshot();
        expect(snapshot.indices).toEqual([]);
        expect(snapshot.fx).toEqual([]);
        expect(snapshot.commodities).toEqual([]);
        expect(snapshot.itemsBySymbol.size).toBe(0);
        expect(snapshot.as_of).toBeNull();
        expect(snapshot.loading).toBe(false);
        expect(snapshot.error).toBeNull();
    });

    it('normalizes symbols correctly (trims, uppercases)', () => {
        expect(normalizeSymbol('  xu100  ')).toBe('XU100');
        expect(normalizeSymbol('usd/try')).toBe('USD/TRY');
        expect(normalizeSymbol('')).toBe('');
    });

    it('subscribes listeners and auto-fetches on first subscription', async () => {
        const spy = vi.spyOn(apiClient, 'marketWatchUnified').mockResolvedValue(mockPayload);
        const listener = vi.fn();

        const unsubscribe = subscribe(listener);
        expect(spy).toHaveBeenCalledTimes(1);

        await act(async () => {
            await Promise.resolve();
        });

        expect(listener).toHaveBeenCalled();
        const snap = getSnapshot();
        expect(snap.as_of).toBe('2026-11-07T18:00:00Z');
        expect(snap.indices.length).toBe(2);

        unsubscribe();
    });

    it('indexes items by symbol and handles both slashed and slashless FX lookups', async () => {
        vi.spyOn(apiClient, 'marketWatchUnified').mockResolvedValue(mockPayload);

        await act(async () => {
            await fetchUnifiedMarketWatch({ force: true });
        });

        const snap = getSnapshot();
        expect(snap.itemsBySymbol.get('XU100')?.price).toBe(9250.5);
        expect(snap.indicesBySymbol.get('SP500')?.price).toBe(5800.0);
        expect(snap.commoditiesBySymbol.get('ALTIN')?.price).toBe(3966.2);

        // FX: both USD/TRY and USDTRY work
        expect(snap.itemsBySymbol.get('USD/TRY')?.price).toBe(38.45);
        expect(snap.itemsBySymbol.get('USDTRY')?.price).toBe(38.45);
        expect(snap.fxBySymbol.get('EUR/TRY')?.price).toBe(41.15);
        expect(snap.fxBySymbol.get('EURTRY')?.price).toBe(41.15);
    });

    it('preserves snapshot immutability across state updates', async () => {
        vi.spyOn(apiClient, 'marketWatchUnified').mockResolvedValue(mockPayload);

        const initialSnap = getSnapshot();
        await act(async () => {
            await fetchUnifiedMarketWatch({ force: true });
        });
        const updatedSnap = getSnapshot();

        expect(updatedSnap).not.toBe(initialSnap);
        expect(updatedSnap.itemsBySymbol).not.toBe(initialSnap.itemsBySymbol);
    });

    it('implements fail-stale behavior: retains cached data on network error', async () => {
        vi.spyOn(apiClient, 'marketWatchUnified').mockResolvedValue(mockPayload);

        // Step 1: Successful initial load
        await act(async () => {
            await fetchUnifiedMarketWatch({ force: true });
        });
        const cachedSnap = getSnapshot();
        expect(cachedSnap.as_of).toBe('2026-11-07T18:00:00Z');

        // Step 2: Second fetch fails
        vi.spyOn(apiClient, 'marketWatchUnified').mockRejectedValue(new Error('503 Service Unavailable'));

        await act(async () => {
            await fetchUnifiedMarketWatch({ force: true });
        });

        const failStaleSnap = getSnapshot();
        // Preserves previous data and as_of
        expect(failStaleSnap.as_of).toBe('2026-11-07T18:00:00Z');
        expect(failStaleSnap.indices.length).toBe(2);
        expect(failStaleSnap.itemsBySymbol.get('XU100')?.price).toBe(9250.5);
        expect(failStaleSnap.loading).toBe(false);
        // Soft error: error remains null because data exists
        expect(failStaleSnap.error).toBeNull();
    });

    it('sets error string when initial fetch fails without any prior cached data', async () => {
        vi.spyOn(apiClient, 'marketWatchUnified').mockRejectedValue(new Error('Network Error'));

        await act(async () => {
            await fetchUnifiedMarketWatch({ force: true });
        });

        const snap = getSnapshot();
        expect(snap.as_of).toBeNull();
        expect(snap.error).toBe('Network Error');
        expect(snap.loading).toBe(false);
    });

    it('passes refresh query parameter on forced refresh', async () => {
        const spy = vi.spyOn(apiClient, 'marketWatchUnified').mockResolvedValue(mockPayload);

        await act(async () => {
            await fetchUnifiedMarketWatch();
        });
        expect(spy).toHaveBeenCalledWith(undefined);

        await act(async () => {
            await fetchUnifiedMarketWatch({ force: true });
        });
        expect(spy).toHaveBeenCalledWith({ refresh: true });
    });

    it('protects against duplicate in-flight requests and queues pending force refresh', async () => {
        let resolveFirst: (val: UnifiedMarketWatchResponse) => void = () => {};
        const firstPromise = new Promise<UnifiedMarketWatchResponse>((res) => {
            resolveFirst = res;
        });

        const spy = vi
            .spyOn(apiClient, 'marketWatchUnified')
            .mockImplementationOnce(() => firstPromise)
            .mockResolvedValue(mockPayload);

        // Start first request
        const p1 = fetchUnifiedMarketWatch();
        expect(spy).toHaveBeenCalledTimes(1);

        // Second non-force request while in-flight is dropped
        const p2 = fetchUnifiedMarketWatch();
        expect(spy).toHaveBeenCalledTimes(1);

        // Third forced request while in-flight queues pendingForce
        const p3 = fetchUnifiedMarketWatch({ force: true });
        expect(spy).toHaveBeenCalledTimes(1);

        // Resolve first request
        await act(async () => {
            resolveFirst(mockPayload);
            await Promise.all([p1, p2, p3]);
        });

        // The queued force refresh must execute with { refresh: true }
        expect(spy).toHaveBeenCalledTimes(2);
        expect(spy).toHaveBeenLastCalledWith({ refresh: true });
    });

    it('updates item state atomically via applyMarketIndexDelta', async () => {
        vi.spyOn(apiClient, 'marketWatchUnified').mockResolvedValue(mockPayload);

        await act(async () => {
            await fetchUnifiedMarketWatch({ force: true });
        });

        expect(getSnapshot().itemsBySymbol.get('XU100')?.price).toBe(9250.5);

        // Apply live SSE delta for XU100
        applyMarketIndexDelta('XU100', { price: 9320.0, change_pct: 1.85 });

        const updatedSnap = getSnapshot();
        expect(updatedSnap.itemsBySymbol.get('XU100')?.price).toBe(9320.0);
        expect(updatedSnap.itemsBySymbol.get('XU100')?.change_pct).toBe(1.85);
        expect(updatedSnap.indices.find((i) => i.symbol === 'XU100')?.price).toBe(9320.0);
    });

    it('works with useMarketStore and useMarketItem hooks', async () => {
        vi.spyOn(apiClient, 'marketWatchUnified').mockResolvedValue(mockPayload);

        const { result: storeResult } = renderHook(() => useMarketStore());
        const { result: itemResult } = renderHook(() => useMarketItem('XU100'));

        await act(async () => {
            await Promise.resolve();
        });

        expect(storeResult.current.as_of).toBe('2026-11-07T18:00:00Z');
        expect(itemResult.current?.price).toBe(9250.5);

        // Test hook refresh function
        const spy = vi.spyOn(apiClient, 'marketWatchUnified').mockResolvedValue(mockPayload);
        await act(async () => {
            storeResult.current.refresh();
        });
        expect(spy).toHaveBeenCalledWith({ refresh: true });
    });
});
