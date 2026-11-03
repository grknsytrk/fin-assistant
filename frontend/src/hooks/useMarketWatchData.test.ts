import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../api/client';
import { useMarketWatchData, _resetMarketWatchStoreForTesting } from './useMarketWatchData';
import { latestAsOf } from '../components/MarketWatchStrip';
import type { CommodityQuote, FxQuote } from '../api/types';

describe('useMarketWatchData', () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        _resetMarketWatchStoreForTesting();
    });

    afterEach(() => {
        cleanup();
        vi.restoreAllMocks();
        _resetMarketWatchStoreForTesting();
    });

    it('shares commodities snapshot across multiple hook instances without duplicate calls', async () => {
        const mockCommodities: CommodityQuote[] = [
            {
                symbol: 'ALTIN',
                label: 'Altın (Ons)',
                yahoo_symbol: 'GC=F',
                price: 3966.2,
                prev_close: 3950.0,
                change: 16.2,
                change_pct: 0.41,
                currency: 'USD',
                market_state: 'REGULAR',
                as_of: '2026-11-03T15:00:00Z',
                error: null,
            },
        ];

        const commoditiesSpy = vi.spyOn(apiClient, 'marketCommodities').mockResolvedValue({
            items: mockCommodities,
            source: 'yahoo_finance_chart',
            delay_note: '',
            as_of: '2026-11-03T15:00:00Z',
        });

        // Instance 1: Strip
        const { result: stripResult } = renderHook(() =>
            useMarketWatchData({ commodities: true, enabled: true }),
        );

        // Instance 2: Rail
        const { result: railResult } = renderHook(() =>
            useMarketWatchData({ commodities: true, enabled: true }),
        );

        // Wait for asynchronous fetch to finish
        await act(async () => {
            await Promise.resolve();
        });

        // Both hook instances must receive the exact same commodities array reference
        expect(stripResult.current.commodities).toBe(railResult.current.commodities);
        expect(stripResult.current.commodities?.[0]?.price).toBe(3966.2);
        expect(railResult.current.commodities?.[0]?.price).toBe(3966.2);

        // Only one network request was initiated
        expect(commoditiesSpy).toHaveBeenCalledTimes(1);
    });

    it('passes refresh query parameter when manual refresh is invoked', async () => {
        const mockFx: FxQuote[] = [
            {
                symbol: 'USD/TRY',
                label: 'Dolar / TL',
                yahoo_symbol: 'USDTRY=X',
                price: 38.25,
                prev_close: 38.2,
                change: 0.05,
                change_pct: 0.13,
                currency: 'TRY',
                market_state: 'REGULAR',
                as_of: '2026-11-03T15:00:00Z',
                error: null,
            },
        ];

        const fxSpy = vi.spyOn(apiClient, 'marketFx').mockResolvedValue({
            items: mockFx,
            source: 'yahoo_finance_chart',
            delay_note: '',
            as_of: '2026-11-03T15:00:00Z',
        });

        const { result } = renderHook(() =>
            useMarketWatchData({ fx: true, enabled: true }),
        );

        await act(async () => {
            await Promise.resolve();
        });

        expect(fxSpy).toHaveBeenCalledWith(undefined);

        // Trigger force refresh
        await act(async () => {
            result.current.refresh();
        });

        expect(fxSpy).toHaveBeenCalledWith({ refresh: true });
    });

    it('queues pending force refresh when refresh is triggered while a request is in flight', async () => {
        let resolveFirstCall: (value: { items: FxQuote[]; source: string; delay_note: string; as_of: string }) => void = () => {};
        const firstPromise = new Promise<{ items: FxQuote[]; source: string; delay_note: string; as_of: string }>((res) => {
            resolveFirstCall = res;
        });

        const fxSpy = vi.spyOn(apiClient, 'marketFx')
            .mockImplementationOnce(() => firstPromise)
            .mockResolvedValue({
                items: [],
                source: 'yahoo',
                delay_note: '',
                as_of: '2026-11-03T15:01:00Z',
            });

        const { result } = renderHook(() =>
            useMarketWatchData({ fx: true, enabled: true }),
        );

        // Initial fetch is now in flight
        expect(fxSpy).toHaveBeenCalledTimes(1);
        expect(fxSpy).toHaveBeenLastCalledWith(undefined);

        // While first is in flight, user triggers force refresh
        act(() => {
            result.current.refresh();
        });

        // Resolve first request
        await act(async () => {
            resolveFirstCall({
                items: [],
                source: 'yahoo',
                delay_note: '',
                as_of: '2026-11-03T15:00:00Z',
            });
            await Promise.resolve();
        });

        // The queued pendingForce refresh must execute with { refresh: true }
        expect(fxSpy).toHaveBeenCalledTimes(2);
        expect(fxSpy).toHaveBeenLastCalledWith({ refresh: true });
    });

    it('correctly picks latest timestamp with latestAsOf', () => {
        expect(latestAsOf(null, undefined)).toBeNull();
        expect(latestAsOf('2026-11-03T14:00:00Z', '2026-11-03T14:05:00Z')).toBe('2026-11-03T14:05:00Z');
        expect(latestAsOf('2026-11-03T15:00:00Z', '2026-11-03T14:59:00Z', '2026-11-03T14:30:00Z')).toBe('2026-11-03T15:00:00Z');
    });
});
