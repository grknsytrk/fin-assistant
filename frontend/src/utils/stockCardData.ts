import type { MarketStockCardsResponse } from '../api/types';

function normalizeSymbol(symbol: string): string {
    return String(symbol || '').trim().toUpperCase().replace(/\.IS$/, '');
}

/**
 * Keep already loaded card data when the user's selected-card order changes.
 * The returned item list follows the new order and excludes removed symbols.
 */
export function retainMarketStockCardsForSymbols(
    payload: MarketStockCardsResponse | null,
    orderedSymbols: readonly string[],
): MarketStockCardsResponse | null {
    if (!payload) return null;

    const itemsBySymbol = new Map<string, MarketStockCardsResponse['items'][number]>();
    for (const item of payload.items || []) {
        itemsBySymbol.set(normalizeSymbol(item.symbol), item);
    }

    return {
        ...payload,
        items: orderedSymbols
            .map((symbol) => itemsBySymbol.get(normalizeSymbol(symbol)))
            .filter((item): item is MarketStockCardsResponse['items'][number] => Boolean(item)),
    };
}

/**
 * Return only cards that need an initial request for the current selection.
 * A quick-stage item still needs its full-stage request; a completed card does
 * not need to be fetched again just because another card was added or moved.
 */
export function getMarketStockCardSymbolsNeedingLoad(
    selectedSymbols: readonly string[],
    payload: MarketStockCardsResponse | null,
): string[] {
    if (!payload) return [...selectedSymbols];

    const itemsBySymbol = new Map<string, MarketStockCardsResponse['items'][number]>();
    for (const item of payload.items || []) {
        itemsBySymbol.set(normalizeSymbol(item.symbol), item);
    }

    return selectedSymbols.filter((symbol) => {
        const item = itemsBySymbol.get(normalizeSymbol(symbol));
        return !item || item.card_stage === 'quick';
    });
}
