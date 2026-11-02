import { describe, expect, it } from 'vitest';
import type { MarketFlowItem } from '../api/types';
import {
    FLOW_HEAD_POLL_CLOSED_MS,
    FLOW_HEAD_POLL_OPEN_MS,
    getFlowHeadPollInterval,
    getMatchingFavoriteSymbols,
    getNewFlowItemIds,
    matchesFlowFilter,
} from './MarketFlowPanel';

const flowItem: MarketFlowItem = {
    id: 'kap-example',
    source: 'Özel Durum',
    symbol: 'DOH',
    stock_codes: ['SKP'],
    related_symbols: ['DOH', 'MANAS', 'THF', 'TLY'],
    title: 'Pay Alım Satım Bildirimi',
    published_at: '2026-09-08T18:10:00',
    category: 'ozel_durum',
};

describe('MarketFlowPanel favorites matching', () => {
    it('returns the favorited related instruments instead of the primary stock code', () => {
        expect(getMatchingFavoriteSymbols(flowItem, new Set(['BIMAS', 'TLY', 'THF']))).toEqual(['TLY', 'THF']);
    });

    it('does not match an unrelated flow item', () => {
        expect(getMatchingFavoriteSymbols(flowItem, new Set(['BIMAS']))).toEqual([]);
    });

    it('matches model_portfolio filter only when symbol is in modelPortfolioSymbols', () => {
        expect(matchesFlowFilter(flowItem, 'model_portfolio', new Set(), new Set(['DOH']))).toBe(true);
        expect(matchesFlowFilter(flowItem, 'model_portfolio', new Set(['DOH']), new Set(['THYAO']))).toBe(false);
        expect(matchesFlowFilter(flowItem, 'model_portfolio', new Set(['DOH']), new Set())).toBe(false);
    });

    it('only marks records that arrived after the previous refresh', () => {
        expect(getNewFlowItemIds(new Set(['old-1']), [
            { ...flowItem, id: 'new-1' },
            { ...flowItem, id: 'old-1' },
        ])).toEqual(['new-1']);
        expect(getNewFlowItemIds(null, [flowItem])).toEqual([]);
    });
});

describe('MarketFlowPanel polling intervals', () => {
    it('uses 5,000 ms during BIST trading hours', () => {
        // Wednesday 11:00 UTC = 14:00 Istanbul (during session 09:50 - 18:20)
        const openDate = new Date('2026-10-21T11:00:00.000Z');
        expect(getFlowHeadPollInterval(openDate)).toBe(FLOW_HEAD_POLL_OPEN_MS);
        expect(getFlowHeadPollInterval(openDate)).toBe(5_000);
    });

    it('relaxes to 60,000 ms when BIST market is closed (nights and weekends)', () => {
        // Sunday 11:00 UTC
        const weekendDate = new Date('2026-10-25T11:00:00.000Z');
        expect(getFlowHeadPollInterval(weekendDate)).toBe(FLOW_HEAD_POLL_CLOSED_MS);
        expect(getFlowHeadPollInterval(weekendDate)).toBe(60_000);

        // Wednesday 21:00 UTC = 00:00 Istanbul (night)
        const nightDate = new Date('2026-10-21T21:00:00.000Z');
        expect(getFlowHeadPollInterval(nightDate)).toBe(FLOW_HEAD_POLL_CLOSED_MS);
        expect(getFlowHeadPollInterval(nightDate)).toBe(60_000);
    });
});
