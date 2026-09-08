import { describe, expect, it } from 'vitest';
import type { MarketFlowItem } from '../api/types';
import { getMatchingFavoriteSymbols, getNewFlowItemIds } from './MarketFlowPanel';

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

    it('only marks records that arrived after the previous refresh', () => {
        expect(getNewFlowItemIds(new Set(['old-1']), [
            { ...flowItem, id: 'new-1' },
            { ...flowItem, id: 'old-1' },
        ])).toEqual(['new-1']);
        expect(getNewFlowItemIds(null, [flowItem])).toEqual([]);
    });
});
