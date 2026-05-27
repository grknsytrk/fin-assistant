import type { MarketIndexCode, MarketStockIndex } from '../api/types';

export const ROUTE_PATHS = {
    landing: '/',
    markets: '/markets',
    marketsOverview: '/markets/overview',
    marketsStocks: '/markets/stocks',
    marketsStocksIndex: '/markets/stocks/:indexCode',
    marketsIndices: '/markets/indices',
    marketsIndicesDetail: '/markets/indices/:indexCode',
    funds: '/funds',
    fundComparison: '/funds/compare',
    fundDetail: '/funds/:fundCode/:tab',
    fundDetailNoTab: '/funds/:fundCode',
    stocks: '/stocks',
    stockDetail: '/stocks/:ticker/:tab',
    stockDetailNoTab: '/stocks/:ticker',
} as const;

export const MARKET_STOCK_INDEX_CODES: readonly MarketStockIndex[] = ['XUTUM', 'XU100', 'XU030'] as const;
export const MARKET_SECTOR_INDEX_CODES = [
    'XUSIN',
    'XUHIZ',
    'XUMAL',
    'XUTEK',
    'XBANK',
    'XAKUR',
    'XBLSM',
    'XELKT',
    'XFINK',
    'XGMYO',
    'XGIDA',
    'XHOLD',
    'XILTM',
    'XINSA',
    'XKAGT',
    'XKMYA',
    'XMADN',
    'XMANA',
    'XMESY',
    'XSGRT',
    'XSPOR',
    'XTAST',
    'XTCRT',
    'XTEKS',
    'XTRZM',
    'XULAS',
    'XYORT',
] as const;
export const MARKET_INDEX_CODES: readonly MarketIndexCode[] = [
    ...MARKET_STOCK_INDEX_CODES,
    ...MARKET_SECTOR_INDEX_CODES,
] as const;
export const STOCK_TABS = ['overview', 'financials', 'kap', 'ask'] as const;
export const FUND_TABS = ['overview', 'allocation', 'history'] as const;
export const STOCK_RETURN_MODES = ['absolute', 'relative_xu100', 'relative_xu030'] as const;

export type StockTab = (typeof STOCK_TABS)[number];
export type FundTab = (typeof FUND_TABS)[number];
export type StockReturnMode = (typeof STOCK_RETURN_MODES)[number];

export const DEFAULT_MARKET_INDEX: MarketStockIndex = 'XUTUM';
export const DEFAULT_STOCK_TAB: StockTab = 'overview';
export const DEFAULT_FUND_TAB: FundTab = 'overview';
export const DEFAULT_STOCK_RETURN_MODE: StockReturnMode = 'absolute';

export function normalizeTicker(raw: string | null | undefined): string {
    return String(raw ?? '')
        .trim()
        .toUpperCase()
        .replace(/\s+/g, '');
}

export function normalizeFundCode(raw: string | null | undefined): string {
    return String(raw ?? '')
        .trim()
        .toUpperCase()
        .replace(/\s+/g, '');
}

export function isValidMarketIndexCode(raw: string | null | undefined): raw is MarketIndexCode {
    const normalized = String(raw ?? '').trim().toUpperCase();
    return MARKET_INDEX_CODES.includes(normalized as MarketIndexCode);
}

export function isValidMarketStockIndex(raw: string | null | undefined): raw is MarketStockIndex {
    const normalized = String(raw ?? '').trim().toUpperCase();
    return MARKET_STOCK_INDEX_CODES.includes(normalized as MarketStockIndex);
}

export function normalizeMarketStockIndex(
    raw: string | null | undefined,
    fallback: MarketStockIndex = DEFAULT_MARKET_INDEX,
): MarketStockIndex {
    const normalized = String(raw ?? '').trim().toUpperCase();
    if (isValidMarketStockIndex(normalized)) {
        return normalized;
    }
    return fallback;
}

export function normalizeMarketIndexCode(
    raw: string | null | undefined,
    fallback: MarketIndexCode = DEFAULT_MARKET_INDEX,
): MarketIndexCode {
    const normalized = String(raw ?? '').trim().toUpperCase();
    if (isValidMarketIndexCode(normalized)) {
        return normalized;
    }
    return fallback;
}

export function isValidStockTab(raw: string | null | undefined): raw is StockTab {
    const normalized = String(raw ?? '').trim().toLowerCase();
    return STOCK_TABS.includes(normalized as StockTab);
}

export function normalizeStockTab(raw: string | null | undefined, fallback: StockTab = DEFAULT_STOCK_TAB): StockTab {
    const normalized = String(raw ?? '').trim().toLowerCase();
    if (normalized === 'charts') {
        return 'overview';
    }
    if (isValidStockTab(normalized)) {
        return normalized;
    }
    return fallback;
}

export function isValidFundTab(raw: string | null | undefined): raw is FundTab {
    const normalized = String(raw ?? '').trim().toLowerCase();
    return FUND_TABS.includes(normalized as FundTab);
}

export function normalizeFundTab(raw: string | null | undefined, fallback: FundTab = DEFAULT_FUND_TAB): FundTab {
    const normalized = String(raw ?? '').trim().toLowerCase();
    if (normalized === 'performance') {
        return 'history';
    }
    if (normalized === 'holdings') {
        return 'allocation';
    }
    if (isValidFundTab(normalized)) {
        return normalized;
    }
    return fallback;
}

export function isValidStockReturnMode(raw: string | null | undefined): raw is StockReturnMode {
    const normalized = String(raw ?? '').trim().toLowerCase();
    return STOCK_RETURN_MODES.includes(normalized as StockReturnMode);
}

export function normalizeStockReturnMode(
    raw: string | null | undefined,
    fallback: StockReturnMode = DEFAULT_STOCK_RETURN_MODE,
): StockReturnMode {
    const normalized = String(raw ?? '').trim().toLowerCase();
    if (isValidStockReturnMode(normalized)) {
        return normalized;
    }
    return fallback;
}

export function toStocksReturnModeSearch(mode: string | null | undefined): string {
    const normalizedMode = normalizeStockReturnMode(mode);
    if (normalizedMode === DEFAULT_STOCK_RETURN_MODE) {
        return '';
    }
    return `?return_mode=${normalizedMode}`;
}

export function canonicalizeStocksReturnModeSearch(search: string): {
    mode: StockReturnMode;
    canonicalSearch: string;
} {
    const params = new URLSearchParams(search || '');
    const mode = normalizeStockReturnMode(params.get('return_mode'));
    if (mode === DEFAULT_STOCK_RETURN_MODE) {
        params.delete('return_mode');
    } else {
        params.set('return_mode', mode);
    }
    return { mode, canonicalSearch: params.toString() };
}

export function toLanding(): string {
    return ROUTE_PATHS.landing;
}

export function toMarketsRoot(): string {
    return ROUTE_PATHS.markets;
}

export function toMarketsOverview(): string {
    return ROUTE_PATHS.marketsOverview;
}

export function toMarketsStocks(indexCode: string | null | undefined = DEFAULT_MARKET_INDEX): string {
    return `${ROUTE_PATHS.marketsStocks}/${normalizeMarketStockIndex(indexCode)}`;
}

export function toMarketsIndices(): string {
    return ROUTE_PATHS.marketsIndices;
}

export function toMarketsIndexDetail(indexCode: string | null | undefined): string {
    return `${ROUTE_PATHS.marketsIndices}/${normalizeMarketIndexCode(indexCode)}`;
}

export function toFunds(): string {
    return ROUTE_PATHS.funds;
}

export function toFundComparison(): string {
    return ROUTE_PATHS.fundComparison;
}

export function toFundDetail(
    fundCode: string | null | undefined,
    tab: string | null | undefined = DEFAULT_FUND_TAB,
): string {
    const normalizedFundCode = normalizeFundCode(fundCode);
    const safeFundCode = normalizedFundCode || 'UNKNOWN';
    return `${ROUTE_PATHS.funds}/${encodeURIComponent(safeFundCode)}/${normalizeFundTab(tab)}`;
}

export function toStockDetail(
    ticker: string | null | undefined,
    tab: string | null | undefined = DEFAULT_STOCK_TAB,
): string {
    const normalizedTicker = normalizeTicker(ticker);
    const safeTicker = normalizedTicker || 'UNKNOWN';
    return `${ROUTE_PATHS.stocks}/${encodeURIComponent(safeTicker)}/${normalizeStockTab(tab)}`;
}

export function legacySearchToCanonical(search: string): string | null {
    const params = new URLSearchParams(search || '');
    const ticker = normalizeTicker(params.get('ticker'));
    const section = normalizeStockTab(params.get('section'));
    if (ticker) {
        return toStockDetail(ticker, section);
    }

    const page = String(params.get('page') || '').trim().toLowerCase();
    if (page === 'markets') {
        return toMarketsStocks(DEFAULT_MARKET_INDEX);
    }
    return null;
}
