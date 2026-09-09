import { describe, expect, it } from 'vitest';
import type { MarketStockCardItem, MarketStockCardsResponse } from '../api/types';
import {
    getMarketStockCardSymbolsNeedingLoad,
    retainMarketStockCardsForSymbols,
} from './stockCardData';

function card(symbol: string, cardStage: MarketStockCardItem['card_stage'] = 'full'): MarketStockCardItem {
    return {
        symbol,
        company: symbol,
        yahoo_symbol: `${symbol}.IS`,
        price: 100,
        currency: 'TRY',
        change: 1,
        change_pct: 1,
        volume: 10,
        volume_lot: 10,
        volume_tl: 1000,
        market_cap: 10000,
        high: 101,
        low: 99,
        previous_close: 99,
        fk: null,
        pd_dd: null,
        fd_favok: null,
        net_borc_favok: null,
        return_1w_pct: null,
        return_1m_pct: null,
        return_3m_pct: null,
        return_6m_pct: null,
        return_ytd_pct: null,
        return_1y_pct: null,
        market_state: 'open',
        as_of: '2026-09-09T12:00:00Z',
        line_points: [],
        error: null,
        logo_url: null,
        logo_source: null,
        card_stage: cardStage,
    };
}

function payload(...items: MarketStockCardItem[]): MarketStockCardsResponse {
    return {
        items,
        source: 'test',
        as_of: '2026-09-09T12:00:00Z',
    };
}

describe('stock card selection data', () => {
    it('keeps existing cards and only adds the newly selected symbol', () => {
        const current = payload(card('THYAO'), card('BIMAS'));

        expect(getMarketStockCardSymbolsNeedingLoad(['THYAO', 'BIMAS', 'KCHOL'], current)).toEqual(['KCHOL']);
        expect(retainMarketStockCardsForSymbols(current, ['KCHOL', 'THYAO', 'BIMAS'])?.items.map((item) => item.symbol))
            .toEqual(['THYAO', 'BIMAS']);
    });

    it('does not request a second load for a reorder or removal', () => {
        const current = payload(card('THYAO'), card('BIMAS'));

        expect(getMarketStockCardSymbolsNeedingLoad(['BIMAS', 'THYAO'], current)).toEqual([]);
        expect(retainMarketStockCardsForSymbols(current, ['BIMAS'])?.items.map((item) => item.symbol)).toEqual(['BIMAS']);
    });

    it('loads a quick card again to complete its full data', () => {
        const current = payload(card('THYAO', 'quick'), card('BIMAS'));

        expect(getMarketStockCardSymbolsNeedingLoad(['THYAO', 'BIMAS'], current)).toEqual(['THYAO']);
    });
});
