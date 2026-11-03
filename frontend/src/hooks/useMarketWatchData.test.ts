import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../api/client';
import { useMarketWatchData } from './useMarketWatchData';
import type { CommodityQuote, FxQuote } from '../api/types';

describe('useMarketWatchData', () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    afterEach(() => {
        cleanup();
        vi.restoreAllMocks();
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
});
