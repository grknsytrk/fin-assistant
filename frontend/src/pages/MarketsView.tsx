import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { CalendarDays, GripHorizontal, Plus, Search, Trash2, X } from 'lucide-react';
import { apiClient } from '../api/client';
import type {
    KapCompanySearchItem,
    MarketIndexCode,
    MarketIndexConstituent,
    MarketIndexDetailResponse,
    MarketIndexListRow,
    MarketIndicesResponse,
    MarketIndexLinePoint,
    MarketStockCardChartRange,
    MarketStockCardItem,
    MarketStockCardsResponse,
    MarketReturnBenchmark,
    MarketStockIndex,
    MarketStockRow,
    MarketStocksResponse,
    MarketUniverseResponse,
    MarketUniverseRow,
    FundDetail,
} from '../api/types';
import { DEFAULT_STOCK_RETURN_MODE } from '../routing/routes';
import type { StockReturnMode } from '../routing/routes';
import MarketWatchRail from '../components/MarketWatchRail';
import MarketSidebar from '../components/MarketSidebar';
import MarketWatchStrip from '../components/MarketWatchStrip';
import MarketsNavigation, { type MarketsNavigationFundSection, type MarketsNavigationSection } from '../components/MarketsNavigation';
import SymbolLogo from '../components/SymbolLogo';
import { buildDocumentTitle, formatTitleNumber, formatTitlePct, useDocumentTitle } from '../hooks/useDocumentTitle';
import { MAX_WATCHLIST_ITEMS, normalizeWatchlistSymbol, useWatchlist, type WatchlistItem } from '../hooks/useWatchlist';
import './MarketsView.css';

type MarketSection = MarketsNavigationSection;
type SortDirection = 'asc' | 'desc';
type IndexConstituentDataSortKey = 'symbol' | 'price' | 'change_pct' | 'volume' | 'weight_pct' | 'point_effect';
type IndexConstituentSortKey = IndexConstituentDataSortKey | 'impact_pct' | 'impact_abs';
type StockCardDropPlacement = 'before' | 'after';
type StockSortKey =
    | 'company'
    | 'price'
    | 'change_pct'
    | 'volume'
    | 'market_cap'
    | 'return_1w_pct'
    | 'return_1m_pct'
    | 'return_3m_pct'
    | 'return_6m_pct'
    | 'return_ytd_pct'
    | 'return_1y_pct';
type IndexSortKey = keyof MarketIndexListRow;

type StockReturnKey = Extract<
    StockSortKey,
    'return_1w_pct' | 'return_1m_pct' | 'return_3m_pct' | 'return_6m_pct' | 'return_ytd_pct' | 'return_1y_pct'
>;

const STOCK_COLUMNS: Array<{ key: StockSortKey; label: string; sublabel?: string; align?: 'left' | 'right' }> = [
    { key: 'company', label: 'Hisse', align: 'left' },
    { key: 'price', label: 'Fiyat', align: 'right' },
    { key: 'change_pct', label: 'Gün %', align: 'right' },
    { key: 'volume', label: 'Hacim', align: 'right' },
    { key: 'market_cap', label: 'Piyasa Değeri', align: 'right' },
    { key: 'return_1w_pct', label: 'Getiri %', sublabel: 'Son 1 hafta', align: 'right' },
    { key: 'return_1m_pct', label: 'Getiri %', sublabel: 'Son 1 ay', align: 'right' },
    { key: 'return_3m_pct', label: 'Getiri %', sublabel: 'Son 3 ay', align: 'right' },
    { key: 'return_6m_pct', label: 'Getiri %', sublabel: 'Son 6 ay', align: 'right' },
    { key: 'return_ytd_pct', label: 'Getiri %', sublabel: 'YTA', align: 'right' },
    { key: 'return_1y_pct', label: 'Getiri %', sublabel: 'Son 1 yıl', align: 'right' },
];
const STOCK_INDEX_OPTIONS: MarketStockIndex[] = ['XUTUM', 'XU100', 'XU030'];
const RETURN_MODE_OPTIONS: Array<{ id: StockReturnMode; label: string }> = [
    { id: 'absolute', label: 'Mutlak' },
    { id: 'relative_xu100', label: "XU100'a göre" },
    { id: 'relative_xu030', label: "XU030'a göre" },
];
const RETURN_KEYS: StockReturnKey[] = [
    'return_1w_pct',
    'return_1m_pct',
    'return_3m_pct',
    'return_6m_pct',
    'return_ytd_pct',
    'return_1y_pct',
];
const STOCK_CARD_STORAGE_KEY = 'ragfin_market_stock_cards';
const MAX_STOCK_CARDS = 12;
const LIVE_MARKET_REFRESH_MS = 3000;
const STOCK_CARD_CHART_RANGES: Array<{ id: MarketStockCardChartRange; label: string; title: string }> = [
    { id: '1d', label: 'G', title: 'Gün içi' },
    { id: '1w', label: '1H', title: '1 Hafta' },
    { id: '1m', label: '1A', title: '1 Ay' },
    { id: '1y', label: '1Y', title: '1 Yıl' },
];
const INDEX_COLUMNS: Array<{ key: keyof MarketIndexListRow; label: string; align?: 'left' | 'right' }> = [
    { key: 'symbol', label: 'Endeks', align: 'left' },
    { key: 'price', label: 'Son Fiyat', align: 'right' },
    { key: 'change_pct', label: 'Gün %', align: 'right' },
    { key: 'volume', label: 'Hacim', align: 'right' },
    { key: 'return_1w_pct', label: '1 Hafta %', align: 'right' },
    { key: 'return_1m_pct', label: '1 Ay %', align: 'right' },
    { key: 'return_3m_pct', label: '3 Ay %', align: 'right' },
    { key: 'return_6m_pct', label: '6 Ay %', align: 'right' },
    { key: 'return_ytd_pct', label: 'YTA %', align: 'right' },
    { key: 'return_1y_pct', label: '1 Yıl %', align: 'right' },
];
const INDEX_CONSTITUENT_COLUMNS: Array<{ key: Exclude<IndexConstituentSortKey, 'impact_abs'>; label: string; align?: 'left' | 'right' }> = [
    { key: 'symbol', label: 'Şirket', align: 'left' },
    { key: 'price', label: 'Son Fiyat', align: 'right' },
    { key: 'change_pct', label: '%', align: 'right' },
    { key: 'volume', label: 'Hacim', align: 'right' },
    { key: 'weight_pct', label: 'Endeks Ağırlığı', align: 'right' },
    { key: 'point_effect', label: 'Puan Etkisi', align: 'right' },
    { key: 'impact_pct', label: 'Etki %', align: 'right' },
];
const DETAIL_RETURN_KEYS: Array<{ key: keyof MarketIndexListRow; label: string }> = [
    { key: 'change_pct', label: 'Gün içi' },
    { key: 'return_1w_pct', label: '1 Hafta' },
    { key: 'return_1m_pct', label: '1 Ay' },
    { key: 'return_ytd_pct', label: 'YTA' },
    { key: 'return_6m_pct', label: '6 Ay' },
    { key: 'return_1y_pct', label: '1 Yıl' },
    { key: 'return_5y_pct', label: '5 Yıl' },
];

function formatStockPrice(row: MarketStockRow): string {
    if (row.price == null) return '-';
    const currencyPrefix = row.price_currency && row.price_currency !== 'TRY' ? `${row.price_currency} ` : '₺';
    return `${currencyPrefix}${row.price.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatIndexPrice(value: number | null): string {
    if (value == null) return '-';
    return value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

function formatMaybeCurrency(value: number | null, currency?: string | null): string {
    if (value == null) return '-';
    const prefix = currency && currency !== 'TRY' ? `${currency} ` : '₺';
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatTablePct(value: number | null): string {
    if (value == null) return 'N/A';
    const sign = value > 0 ? '+' : '';
    return `%${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatVolume(value: number | null): string {
    if (value == null) return '-';
    const abs = Math.abs(value);
    if (abs >= 1_000_000_000) {
        return `${(value / 1_000_000_000).toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })} mr`;
    }
    if (abs >= 1_000_000) {
        return `${(value / 1_000_000).toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })} mn`;
    }
    return value.toLocaleString('tr-TR', { maximumFractionDigits: 0 });
}

function formatMarketCap(value: number | null): string {
    if (value == null) return '-';
    return `₺${formatVolume(value)}`;
}

function formatCardCurrency(value: number | null | undefined, currency?: string | null): string {
    if (value == null) return '-';
    const prefix = currency && currency !== 'TRY' ? `${currency} ` : '₺';
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatCardPct(value: number | null | undefined): string {
    if (value == null) return '% -';
    const sign = value > 0 ? '+' : '';
    return `%${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatCardRatio(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

function formatCardPositiveRatio(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value) || value <= 0) return '-';
    return formatCardRatio(value);
}

function hasStockCardLoadedData(item: MarketStockCardItem | null | undefined): boolean {
    if (!item) return false;
    if (item.error) return true;
    const hasMarketField = [
        item.price,
        item.change_pct,
        item.high,
        item.low,
        item.previous_close,
        item.volume,
        item.volume_lot,
        item.volume_tl,
        item.market_cap,
    ].some((value) => Number.isFinite(value));
    const hasChart = (item.line_points ?? []).some((point) => Number.isFinite(point.close));
    return hasMarketField || hasChart;
}

function formatCardFullNumber(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return value.toLocaleString('tr-TR', { maximumFractionDigits: 0 });
}

function formatCardFullCurrency(value: number | null | undefined, currency?: string | null): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const prefix = currency && currency !== 'TRY' ? `${currency} ` : '₺';
    return `${prefix}${formatCardFullNumber(value)}`;
}

function formatStockCardChartDate(iso: string | null | undefined, range: MarketStockCardChartRange): string {
    if (!iso) return '-';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '-';
    const withTime = range === '1d' || range === '1w';
    return new Intl.DateTimeFormat('tr-TR', {
        timeZone: 'Europe/Istanbul',
        day: '2-digit',
        month: 'short',
        year: '2-digit',
        ...(withTime ? { hour: '2-digit', minute: '2-digit' } : {}),
    }).format(dt);
}

function formatStockCardAxisDate(iso: string | null | undefined, range: MarketStockCardChartRange): string {
    if (!iso) return '';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '';
    if (range === '1d') {
        return new Intl.DateTimeFormat('tr-TR', {
            timeZone: 'Europe/Istanbul',
            hour: '2-digit',
            minute: '2-digit',
        }).format(dt);
    }
    return new Intl.DateTimeFormat('tr-TR', {
        timeZone: 'Europe/Istanbul',
        day: '2-digit',
        month: 'short',
    }).format(dt);
}

function formatStockCardAxisTime(date: Date): string {
    if (Number.isNaN(date.getTime())) return '';
    return new Intl.DateTimeFormat('tr-TR', {
        timeZone: 'Europe/Istanbul',
        hour: '2-digit',
        minute: '2-digit',
    }).format(date);
}

function formatUpdateTime(iso: string | null | undefined): string {
    if (!iso) return '--:--';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '--:--';
    return dt.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatStockCardTradeTime(iso: string | null | undefined): string {
    if (!iso) return '--:--';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '--:--';
    const today = new Date();
    const isToday = dt.toLocaleDateString('tr-TR', { timeZone: 'Europe/Istanbul' })
        === today.toLocaleDateString('tr-TR', { timeZone: 'Europe/Istanbul' });
    return new Intl.DateTimeFormat('tr-TR', {
        timeZone: 'Europe/Istanbul',
        ...(isToday ? {} : { day: '2-digit', month: 'short' }),
        hour: '2-digit',
        minute: '2-digit',
    }).format(dt);
}

function latestStockCardPointTime(points: MarketIndexLinePoint[] | null | undefined): string | null {
    const valid = (points || [])
        .map((point) => point.time)
        .filter((time): time is string => Boolean(time))
        .sort();
    return valid[valid.length - 1] || null;
}

function isPreviousIstanbulDate(iso: string | null | undefined): boolean {
    if (!iso) return false;
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return false;
    return istanbulDateKey(dt) < istanbulDateKey(new Date());
}

function istanbulDateKey(date: Date): string {
    if (Number.isNaN(date.getTime())) return '';
    return new Intl.DateTimeFormat('en-CA', {
        timeZone: 'Europe/Istanbul',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
    }).format(date);
}

function formatDateTime(iso: string | null | undefined): string {
    if (!iso) return '-';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '-';
    return dt.toLocaleString('tr-TR', {
        day: '2-digit',
        month: 'long',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
}

function formatTerminalDate(date: Date): string {
    const dayMonth = date.toLocaleDateString('tr-TR', {
        day: 'numeric',
        month: 'long',
    });
    const weekday = date.toLocaleDateString('tr-TR', { weekday: 'long' });
    return `${dayMonth}, ${weekday.charAt(0).toUpperCase()}${weekday.slice(1)}`;
}

function formatTerminalClock(date: Date): string {
    return date.toLocaleTimeString('tr-TR', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
}

function normalizeStockCardSymbol(raw: string): string {
    return raw.trim().toUpperCase().replace(/\.IS$/, '');
}

function readStoredStockCards(): string[] {
    if (typeof window === 'undefined') return [];
    try {
        const raw = window.localStorage.getItem(STOCK_CARD_STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        if (!Array.isArray(parsed)) return [];
        const seen = new Set<string>();
        const symbols: string[] = [];
        for (const item of parsed) {
            if (typeof item !== 'string') continue;
            const symbol = normalizeStockCardSymbol(item);
            if (!/^[A-Z0-9]{2,12}$/.test(symbol) || seen.has(symbol)) continue;
            symbols.push(symbol);
            seen.add(symbol);
            if (symbols.length >= MAX_STOCK_CARDS) break;
        }
        return symbols;
    } catch {
        return [];
    }
}

function mobileSparklinePath(points: MarketIndexLinePoint[]): string {
    const valid = points.filter((point) => Number.isFinite(point.close));
    if (valid.length < 2) return '';
    const sampled = valid.length > 36
        ? valid.filter((_, index) => index % Math.ceil(valid.length / 36) === 0)
        : valid;
    const values = sampled.map((point) => point.close);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(max - min, Math.abs(max) * 0.001, 0.0001);
    return sampled
        .map((point, index) => {
            const x = (index / (sampled.length - 1)) * 220;
            const y = 46 - ((point.close - min) / span) * 38;
            return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(' ');
}

function MobileMarketOverview({
    index,
    rows,
    watchlistItems,
    fundRows,
    onSelectTicker,
    onSelectIndex,
    onOpenFund,
    onAddStock,
    onRemoveWatchlistItem,
}: {
    index: MarketIndexDetailResponse | null;
    rows: MarketUniverseRow[];
    watchlistItems: WatchlistItem[];
    fundRows: Record<string, FundDetail | null>;
    onSelectTicker: (ticker: string) => void;
    onSelectIndex: (index: MarketIndexCode) => void;
    onOpenFund?: (fundCode: string) => void;
    onAddStock: (symbol: string) => void;
    onRemoveWatchlistItem: (item: WatchlistItem) => void;
}) {
    const [searchOpen, setSearchOpen] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [companySearchItems, setCompanySearchItems] = useState<KapCompanySearchItem[]>([]);
    const [companySearchLoaded, setCompanySearchLoaded] = useState(false);
    const [companySearchLoading, setCompanySearchLoading] = useState(false);
    const [swipedWatchlistKey, setSwipedWatchlistKey] = useState<string | null>(null);
    const swipeStartRef = useRef<{ key: string; x: number; y: number } | null>(null);
    const suppressRowClickRef = useRef(false);
    const rowBySymbol = new Map(rows.map((row) => [normalizeWatchlistSymbol(row.company), row]));
    const savedStockSymbols = new Set(
        watchlistItems
            .filter((item) => item.kind === 'stock')
            .map((item) => normalizeWatchlistSymbol(item.symbol)),
    );
    const isWatchlistFull = watchlistItems.length >= MAX_WATCHLIST_ITEMS;
    const normalizedSearch = searchTerm.trim().toUpperCase();
    useEffect(() => {
        if (!searchOpen || companySearchLoaded || companySearchLoading) return;
        let active = true;
        setCompanySearchLoading(true);
        apiClient.kapCompanies()
            .then((payload) => {
                if (!active) return;
                if (payload.items?.length) {
                    setCompanySearchItems(payload.items);
                } else {
                    setCompanySearchItems((payload.companies || []).map((symbol) => ({
                        symbol,
                        title: null,
                    })));
                }
            })
            .catch(() => {
                if (active) setCompanySearchItems([]);
            })
            .finally(() => {
                if (active) {
                    setCompanySearchLoading(false);
                    setCompanySearchLoaded(true);
                }
            });
        return () => {
            active = false;
        };
    }, [companySearchLoaded, companySearchLoading, searchOpen]);

    const searchCandidates = new Map<string, { symbol: string; name: string; logoUrl?: string | null }>();
    for (const row of rows) {
        const symbol = normalizeWatchlistSymbol(row.company);
        searchCandidates.set(symbol, { symbol, name: row.company, logoUrl: row.logo_url });
    }
    for (const item of companySearchItems) {
        const symbol = normalizeWatchlistSymbol(item.symbol);
        if (!symbol || searchCandidates.has(symbol)) continue;
        searchCandidates.set(symbol, { symbol, name: item.title || symbol });
    }
    const searchResults = normalizedSearch
        ? [...searchCandidates.values()]
            .filter((item) => !savedStockSymbols.has(item.symbol))
            .filter((item) => (
                item.symbol.includes(normalizedSearch)
                || item.name.toLocaleUpperCase('tr-TR').includes(normalizedSearch)
            ))
            .slice(0, 8)
        : [];
    const visibleWatchlist = watchlistItems.slice(0, 5).map((item) => ({
        item,
        row: item.kind === 'stock' ? rowBySymbol.get(normalizeWatchlistSymbol(item.symbol)) || null : null,
    }));
    const sparkline = mobileSparklinePath(index?.line_points || []);

    const handleWatchlistPointerDown = (event: ReactPointerEvent<HTMLButtonElement>, key: string) => {
        if (event.pointerType === 'mouse' && event.button !== 0) return;
        event.currentTarget.setPointerCapture?.(event.pointerId);
        swipeStartRef.current = { key, x: event.clientX, y: event.clientY };
    };

    const handleWatchlistPointerUp = (event: ReactPointerEvent<HTMLButtonElement>, key: string) => {
        const start = swipeStartRef.current;
        swipeStartRef.current = null;
        if (!start || start.key !== key) return;

        const deltaX = event.clientX - start.x;
        const deltaY = event.clientY - start.y;
        if (Math.abs(deltaX) < 45 || Math.abs(deltaX) <= Math.abs(deltaY)) return;

        suppressRowClickRef.current = true;
        setSwipedWatchlistKey(deltaX < 0 ? key : null);
        window.setTimeout(() => {
            suppressRowClickRef.current = false;
        }, 0);
    };

    const handleWatchlistRowClick = (item: WatchlistItem, symbol: string) => {
        if (suppressRowClickRef.current) return;
        if (swipedWatchlistKey) {
            setSwipedWatchlistKey(null);
            return;
        }
        if (item.kind === 'fund') onOpenFund?.(symbol);
        else onSelectTicker(symbol);
    };

    return (
        <section className="mobile-market-overview" aria-label="Mobil piyasa özeti">
            <button
                type="button"
                className="mobile-market-index-card"
                onClick={() => onSelectIndex('XU100')}
                aria-label="XU100 endeks detayını aç"
            >
                <div className="mobile-market-index-head">
                    <div className="mobile-market-index-identity">
                        <SymbolLogo symbol="XU100" name="BIST 100" kind="index" size="md" />
                        <div>
                            <strong>XU100</strong>
                            <span>BIST 100</span>
                            <small>{formatUpdateTime(index?.as_of)}</small>
                        </div>
                    </div>
                    <div className="mobile-market-index-quote">
                        <span>G</span>
                        <strong>{formatIndexPrice(index?.price ?? null)}</strong>
                        <b className={getTableChangeClass(index?.change_pct ?? null)}>
                            {formatTablePct(index?.change_pct ?? null)}
                        </b>
                    </div>
                </div>
                <div className={`mobile-market-index-chart ${getTableChangeClass(index?.change_pct ?? null)}`} aria-hidden="true">
                    {sparkline ? (
                        <svg viewBox="0 0 220 52" preserveAspectRatio="none">
                            <path d={sparkline} />
                        </svg>
                    ) : (
                        <span />
                    )}
                </div>
            </button>

            <div className="mobile-market-watchlist">
                <div className="mobile-market-watchlist-head">
                    <div>
                        <h2>İzleme listesi</h2>
                        <span>Favori hisselerini hızlıca takip et</span>
                    </div>
                </div>

                {visibleWatchlist.length ? (
                    <div className="mobile-market-watchlist-list">
                        {visibleWatchlist.map(({ item, row }) => {
                            const symbol = normalizeWatchlistSymbol(item.symbol);
                            const itemKey = `${item.kind}:${symbol}`;
                            const fundRow = item.kind === 'fund' ? fundRows[symbol] : null;
                            const price = row?.price ?? fundRow?.price ?? null;
                            const changePct = row?.change_pct ?? fundRow?.daily_return ?? null;
                            const asOf = row?.price_as_of ?? fundRow?.as_of ?? null;
                            return (
                                <div
                                    key={itemKey}
                                    className={`mobile-market-watchlist-row-shell${swipedWatchlistKey === itemKey ? ' is-swiped' : ''}`}
                                >
                                    <button
                                        type="button"
                                        className="mobile-market-watchlist-row"
                                        onClick={() => handleWatchlistRowClick(item, symbol)}
                                        onPointerDown={(event) => handleWatchlistPointerDown(event, itemKey)}
                                        onPointerUp={(event) => handleWatchlistPointerUp(event, itemKey)}
                                        onPointerCancel={() => { swipeStartRef.current = null; }}
                                    >
                                        <SymbolLogo
                                            symbol={symbol}
                                            name={item.label || row?.company || fundRow?.name || symbol}
                                            kind={item.kind}
                                            logoUrl={row?.logo_url}
                                            size="sm"
                                        />
                                        <span className="mobile-market-watchlist-symbol">
                                            <strong>{symbol}</strong>
                                            <small>{formatUpdateTime(asOf)}</small>
                                        </span>
                                        <span className="mobile-market-watchlist-price">
                                            <small>G</small>
                                            {formatIndexPrice(price)}
                                        </span>
                                        <span className={`mobile-market-watchlist-change ${getTableChangeClass(changePct)}`}>
                                            {formatTablePct(changePct)}
                                        </span>
                                    </button>
                                    <button
                                        type="button"
                                        className="mobile-market-watchlist-delete"
                                        onClick={() => {
                                            onRemoveWatchlistItem(item);
                                            setSwipedWatchlistKey(null);
                                        }}
                                        aria-label={`${symbol} izleme listesinden sil`}
                                        title="İzleme listesinden sil"
                                    >
                                        <Trash2 size={19} aria-hidden="true" />
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                ) : (
                    <div className="mobile-market-watchlist-empty">
                        İzleme listene hisse veya fon eklemek için Sembol ekle butonuna dokun.
                    </div>
                )}

                {searchOpen ? (
                    <div className="mobile-market-watchlist-search-area">
                        <div className="mobile-market-watchlist-search">
                            <Search size={18} aria-hidden="true" />
                            <input
                                type="search"
                                autoFocus
                                value={searchTerm}
                                onChange={(event) => setSearchTerm(event.target.value)}
                                placeholder="Hisse kodu ara..."
                                aria-label="İzleme listesine eklenecek hisseyi ara"
                            />
                            <button
                                type="button"
                                onClick={() => {
                                    setSearchOpen(false);
                                    setSearchTerm('');
                                }}
                                aria-label="Aramayı kapat"
                                title="Kapat"
                            >
                                <X size={17} aria-hidden="true" />
                            </button>
                        </div>
                        {searchTerm.trim() && (
                            <div className="mobile-market-watchlist-search-results">
                                {companySearchLoading && searchResults.length === 0 ? (
                                    <span className="mobile-market-watchlist-search-empty">Hisseler yükleniyor...</span>
                                ) : searchResults.length > 0 ? searchResults.map((result) => (
                                    <button
                                        key={result.symbol}
                                        type="button"
                                        onClick={() => {
                                            onAddStock(result.symbol);
                                            setSearchTerm('');
                                            setSearchOpen(false);
                                        }}
                                    >
                                        <SymbolLogo
                                            symbol={result.symbol}
                                            name={result.name}
                                            kind="stock"
                                            logoUrl={result.logoUrl}
                                            size="sm"
                                        />
                                        <strong>{result.symbol}</strong>
                                        <span>Ekle</span>
                                    </button>
                                )) : (
                                    <span className="mobile-market-watchlist-search-empty">Sonuç bulunamadı.</span>
                                )}
                            </div>
                        )}
                    </div>
                ) : (
                    <button
                        type="button"
                        className="mobile-market-watchlist-add"
                        onClick={() => setSearchOpen(true)}
                        disabled={isWatchlistFull}
                    >
                        <Plus size={20} aria-hidden="true" />
                        {isWatchlistFull ? `İzleme listesi dolu (${MAX_WATCHLIST_ITEMS}/${MAX_WATCHLIST_ITEMS})` : 'Sembol ekle'}
                    </button>
                )}
            </div>
        </section>
    );
}

type MarketUniverseMemoryCache = { data: MarketUniverseResponse; fetchedAt: number };
type MarketStockCardsMemoryCache = { data: MarketStockCardsResponse; fetchedAt: number };

// Keep the last visible market data while the user changes sections.  The
// backend still refreshes in the background, so this is a stale-while-
// revalidate UX cache rather than a replacement for server-side freshness.
let marketUniverseMemoryCache: MarketUniverseMemoryCache | null = null;
const marketStockCardsMemoryCache = new Map<string, MarketStockCardsMemoryCache>();

function stockCardsMemoryKey(symbols: string[]): string {
    return symbols.join(',');
}

function getTableChangeClass(value: number | null): string {
    if (value == null || value === 0) return 'stocks-flat';
    return value > 0 ? 'stocks-up' : 'stocks-down';
}

function numericOrNull(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function indexCellValue(row: MarketIndexListRow, key: keyof MarketIndexListRow): string {
    if (key === 'symbol') return row.symbol;
    if (key === 'price') return formatIndexPrice(row.price);
    if (key === 'volume') return formatVolume(row.volume);
    if (String(key).includes('return') || key === 'change_pct') {
        return formatTablePct(numericOrNull(row[key]));
    }
    const value = row[key];
    return value == null ? '-' : String(value);
}

function constituentPrice(row: MarketIndexConstituent): string {
    return formatMaybeCurrency(row.price, row.price_currency);
}

function formatWeight(value: number | null): string {
    if (value == null) return '-';
    return `% ${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatPointEffect(value: number | null): string {
    if (value == null) return '-';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatPointEffectShort(value: number | null): string {
    const formatted = formatPointEffect(value);
    return formatted === '-' ? formatted : `${formatted}p`;
}

function getImpactPct(row: MarketIndexConstituent, indexLevel: number | null): number | null {
    if (row.point_effect == null || indexLevel == null || indexLevel <= 0) return null;
    return (row.point_effect / indexLevel) * 100;
}

function formatImpactPct(value: number | null): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}%`;
}

function formatHeatmapChangePct(value: number | null): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}%`;
}

function isReturnKey(key: StockSortKey): key is StockReturnKey {
    return RETURN_KEYS.includes(key as StockReturnKey);
}

function getBenchmarkIndex(returnMode: StockReturnMode): MarketStockIndex | null {
    if (returnMode === 'relative_xu100') return 'XU100';
    if (returnMode === 'relative_xu030') return 'XU030';
    return null;
}

function getReturnValue(
    row: MarketStockRow,
    key: StockReturnKey,
    benchmarks: Record<MarketStockIndex, MarketReturnBenchmark> | undefined,
    returnMode: StockReturnMode,
): number | null {
    const rawValue = row[key];
    const benchmarkIndex = getBenchmarkIndex(returnMode);
    if (!benchmarkIndex) return rawValue;

    const benchmarkValue = benchmarks?.[benchmarkIndex]?.[key];
    if (rawValue == null || benchmarkValue == null) return null;
    return Math.round((rawValue - benchmarkValue) * 100) / 100;
}

function getColumnLabel(key: StockSortKey, returnMode: StockReturnMode): string {
    if (!isReturnKey(key)) {
        return STOCK_COLUMNS.find((column) => column.key === key)?.label || '';
    }
    return returnMode === 'absolute' ? 'Getiri %' : 'Relatif %';
}

function stockSortValue(
    row: MarketStockRow,
    key: StockSortKey,
    benchmarks: Record<MarketStockIndex, MarketReturnBenchmark> | undefined,
    returnMode: StockReturnMode,
): string | number | null {
    if (key === 'company') return row.company;
    if (isReturnKey(key)) return getReturnValue(row, key, benchmarks, returnMode);
    return row[key];
}

function constituentSortValue(
    row: MarketIndexConstituent,
    key: IndexConstituentSortKey,
    indexLevel: number | null,
): string | number | null {
    if (key === 'symbol') return row.symbol;
    if (key === 'impact_abs') return row.point_effect == null ? null : Math.abs(row.point_effect);
    if (key === 'impact_pct') return getImpactPct(row, indexLevel);
    if (key === 'price') return row.price;
    if (key === 'change_pct') return row.change_pct;
    if (key === 'volume') return row.volume;
    if (key === 'weight_pct') return row.weight_pct;
    if (key === 'point_effect') return row.point_effect;
    return null;
}

function FlashStockRow({
    row,
    rank,
    children,
    onClick,
}: React.PropsWithChildren<{ row: MarketStockRow; rank: number; onClick: () => void }>) {
    const prevPriceRef = useRef(row.price);
    const [flashClass, setFlashClass] = useState('');

    useEffect(() => {
        if (row.price != null && prevPriceRef.current != null && row.price !== prevPriceRef.current) {
            setFlashClass(row.price > prevPriceRef.current ? 'stocks-flash-up' : 'stocks-flash-down');
            const timer = window.setTimeout(() => setFlashClass(''), 1100);
            prevPriceRef.current = row.price;
            return () => window.clearTimeout(timer);
        }
        prevPriceRef.current = row.price;
    }, [row.price]);

    return (
        <tr className={flashClass} onClick={onClick}>
            <td className="stocks-rank">{rank}</td>
            {children}
        </tr>
    );
}


const isBistSymbol = (sym?: string) => {
    if (!sym) return true;
    const globalSymbols = ['SP500', 'NASDAQ', 'DOW', 'DAX', 'FTSE', 'CAC40', 'NIKKEI', 'HANGSENG', 'VIX', 'DXY'];
    if (globalSymbols.includes(sym.toUpperCase())) return false;
    if (sym.includes('/')) return false;
    return true;
};

function IndexLineChart({
    symbol,
    points,
    prevClose,
    changePct,
}: {
    symbol?: string;
    points: MarketIndexDetailResponse['line_points'];
    prevClose: number | null;
    changePct: number | null;
}) {
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const width = 1120;
    const height = 400;
    const padding = { top: 30, right: 65, bottom: 40, left: 16 };
    const validPoints = points.filter((point) => Number.isFinite(point.close));

    if (validPoints.length < 2) {
        return <div className="indices-chart-empty">Grafik verisi bekleniyor.</div>;
    }

    const values = validPoints.map((point) => point.close);
    if (prevClose != null) values.push(prevClose);
    
    // Add small visual padding around min/max values
    let minValue = Math.min(...values);
    let maxValue = Math.max(...values);
    const spanRaw = Math.max(1, maxValue - minValue);
    minValue -= spanRaw * 0.05;
    maxValue += spanRaw * 0.05;
    const span = maxValue - minValue;

    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const isBist = isBistSymbol(symbol);
    const useTimeScale = isBist && validPoints.length > 0;
    let startTimeMs = 0;
    let endTimeMs = 0;
    if (useTimeScale) {
        const d = new Date(validPoints[0].time);
        const start = new Date(d); start.setHours(10, 0, 0, 0);
        const end = new Date(d); end.setHours(18, 0, 0, 0);
        startTimeMs = start.getTime();
        endTimeMs = end.getTime();
    }

    const xFor = (index: number) => {
        if (useTimeScale) {
            const pointTime = new Date(validPoints[index].time).getTime();
            let ratio = (pointTime - startTimeMs) / (endTimeMs - startTimeMs);
            ratio = Math.max(0, Math.min(1, ratio));
            return padding.left + ratio * plotWidth;
        }
        return padding.left + (validPoints.length === 1 ? 0 : (index / (validPoints.length - 1)) * plotWidth);
    };
    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;

    const last = validPoints[validPoints.length - 1].close;

    // Y ekseni çizgileri (5 adet)
    const tickCount = 6;
    const tickValues = Array.from({ length: tickCount }).map((_, i) => minValue + (span * i) / (tickCount - 1));

    // X ekseni (saat başları için tahmini çizim veya eşit dağılımlı)
    const timeTickLabels: Array<{x: number, label: string, key: string}> = [];
    if (useTimeScale) {
        for (let h = 10; h <= 18; h += 1) {
            const d = new Date(validPoints[0].time);
            d.setHours(h, 0, 0, 0);
            timeTickLabels.push({
                x: padding.left + ((d.getTime() - startTimeMs) / (endTimeMs - startTimeMs)) * plotWidth,
                label: `${h.toString().padStart(2, '0')}:00`,
                key: `fixed-${h}`
            });
        }
    } else {
        const timeTickCount = 8;
        const timeTicks = Array.from({ length: timeTickCount }).map((_, i) => Math.floor((validPoints.length - 1) * (i / (timeTickCount - 1))));
        timeTicks.forEach(index => {
            const dt = new Date(validPoints[index].time);
            const label = Number.isNaN(dt.getTime()) ? '' : dt.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
            timeTickLabels.push({ x: xFor(index), label, key: `dyn-${index}` });
        });
    }

    const chartColor = changePct != null && changePct < 0 ? '#ff4d5e' : '#22c55e';

    const pathData = validPoints
        .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(index)} ${yFor(point.close)}`)
        .join(' ');
        
    const areaData = `${pathData} L ${xFor(validPoints.length - 1)} ${height - padding.bottom} L ${padding.left} ${height - padding.bottom} Z`;

    const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        if (rect.width <= 0) return;
        const x = ((event.clientX - rect.left) / rect.width) * width;
        let closestIndex = 0;
        let minDiff = Infinity;
        for (let i = 0; i < validPoints.length; i++) {
            const diff = Math.abs(xFor(i) - x);
            if (diff < minDiff) {
                minDiff = diff;
                closestIndex = i;
            }
        }
        setHoverIndex(closestIndex);
    };

    const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);
    const activeHoverIndex = hoverIndex == null ? null : clamp(hoverIndex, 0, validPoints.length - 1);
    const hoverPoint = activeHoverIndex == null ? null : validPoints[activeHoverIndex];
    const hoverX = activeHoverIndex == null ? null : xFor(activeHoverIndex);
    const hoverY = hoverPoint ? yFor(hoverPoint.close) : null;
    const tooltipWidth = 145;
    const tooltipHeight = 60;
    const tooltipX = hoverX == null ? 0 : clamp(hoverX + 16, padding.left, width - tooltipWidth - padding.right);
    const tooltipY = hoverY == null ? 0 : clamp(hoverY - tooltipHeight / 2, padding.top, height - tooltipHeight - padding.bottom);

    return (
        <div style={{ backgroundColor: '#0f1214', borderRadius: '8px', border: '1px solid #1e2327', position: 'relative', overflow: 'hidden' }}>
        <svg 
            className="indices-line-chart" 
            viewBox={`0 0 ${width} ${height}`} 
            role="img" 
            aria-label="Endeks çizgi grafiği" 
            style={{ display: 'block', width: '100%', height: 'auto', borderBottom: 'none' }}
            onPointerMove={handlePointerMove}
            onPointerLeave={() => setHoverIndex(null)}
        >
            <defs>
                <linearGradient id="areaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor={chartColor} stopOpacity="0.24" />
                    <stop offset="100%" stopColor={chartColor} stopOpacity="0" />
                </linearGradient>
            </defs>
            {/* Koyu Arkaplan (Uygulamanın karanlık temasına uygun) */}
            <rect x="0" y="0" width={width} height={height} fill="#0d1113" rx="4" />
            
            {/* Yatay Grid ve Değerler */}
            {tickValues.map((value) => (
                <g key={value}>
                    <line
                        x1={padding.left}
                        x2={width - padding.right}
                        y1={yFor(value)}
                        y2={yFor(value)}
                        stroke="rgba(255,255,255,0.05)"
                        strokeWidth="1"
                    />
                    <text x={width - padding.right + 10} y={yFor(value) + 4} fill="rgba(255,255,255,0.4)" fontSize="11" fontFamily="monospace">
                        {formatIndexPrice(value)}
                    </text>
                </g>
            ))}

            {/* Önceki Kapanış Referans Çizgisi */}
            {prevClose != null && (
                <line
                    x1={padding.left}
                    x2={width - padding.right}
                    y1={yFor(prevClose)}
                    y2={yFor(prevClose)}
                    stroke="rgba(180, 180, 180, 0.3)"
                    strokeWidth="1"
                    strokeDasharray="4 4"
                />
            )}

            {/* Dikey Grid Zaman Etiketleri */}
            {timeTickLabels.map(({ x, label, key }) => (
                <g key={key}>
                    <line
                        x1={x}
                        x2={x}
                        y1={padding.top}
                        y2={height - padding.bottom}
                        stroke="rgba(255,255,255,0.03)"
                        strokeWidth="1"
                    />
                    <text x={x} y={height - 15} fill="rgba(255,255,255,0.5)" fontSize="11" fontFamily="monospace" textAnchor="middle">
                        {label}
                    </text>
                </g>
            ))}

            {/* Alan ve Çizgi (Ağ/Line) (Açılış, yüksek, düşük kullanılmıyor, SADECE CLOSE) */}
            <path d={areaData} fill="url(#areaGrad)" />
            <path d={pathData} fill="none" stroke={chartColor} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
            
            {/* Uç Noktası (Kapanış) Dot ve Pulse */}
            <circle cx={xFor(validPoints.length - 1)} cy={yFor(last)} r="4" fill={chartColor} opacity="0.6">
                <animate attributeName="r" values="4; 14; 14" keyTimes="0; 0.5; 1" dur="2s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.6; 0; 0" keyTimes="0; 0.5; 1" dur="2s" repeatCount="indefinite" />
            </circle>
            <circle cx={xFor(validPoints.length - 1)} cy={yFor(last)} r="4" fill={chartColor} />

            {/* Anlık Fiyat İşareti */}
            <g transform={`translate(${width - padding.right}, ${yFor(last)})`}>
                <rect x="0" y="-10" width="65" height="20" fill={chartColor} rx="2" />
                <path d="M 0 0 L 6 -6 L 6 6 Z" fill={chartColor} transform="translate(-5, 0)" />
                <text x="32" y="3" fill="#ffffff" fontSize="11" fontFamily="monospace" textAnchor="middle" fontWeight="bold">
                    {formatIndexPrice(last)}
                                </text>
            </g>

            {/* Hover Tooltip */}
            {hoverPoint && hoverX != null && hoverY != null && (
                <g className="indices-chart-hover">
                    <line
                        x1={hoverX}
                        x2={hoverX}
                        y1={padding.top}
                        y2={height - padding.bottom}
                        stroke="rgba(255,255,255,0.42)"
                        strokeDasharray="4 4"
                    />
                    <line
                        x1={padding.left}
                        x2={width - padding.right}
                        y1={hoverY}
                        y2={hoverY}
                        stroke="rgba(255,255,255,0.24)"
                        strokeDasharray="5 5"
                    />
                    <circle cx={hoverX} cy={hoverY} r="4" fill={chartColor} stroke="#0a0c0f" strokeWidth="2" />
                    <g transform={`translate(${tooltipX}, ${tooltipY})`}>
                        <rect width={tooltipWidth} height={tooltipHeight} rx="4" fill="#07090b" stroke="rgba(255,255,255,0.1)" />
                        <text x="12" y="22" fill="#d8dee9" fontSize="13" fontFamily="monospace">
                            {new Intl.DateTimeFormat('tr-TR', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }).format(new Date(hoverPoint.time))}
                        </text>
                        <text x="12" y="46" fill="#f8fafc" fontSize="14" fontFamily="monospace" fontWeight="700">
                            {symbol || 'Endeks'}
                        </text>
                        <text x={tooltipWidth - 12} y="46" fill="#f8fafc" fontSize="14" fontFamily="monospace" fontWeight="700" textAnchor="end">
                            {formatIndexPrice(hoverPoint.close)}
                        </text>
                    </g>
                </g>
            )}
        </svg>
        </div>
    );
}

function StockCardMiniChart({
    symbol,
    points,
    previousClose,
    changePct,
    currency,
    selectedRange,
    pendingRange,
    rangeLoading,
    rangeError,
    onRangeSelect,
    isLoading = false,
    isLive = false,
}: {
    symbol: string;
    points: MarketIndexLinePoint[] | undefined;
    previousClose?: number | null;
    changePct: number | null;
    currency: string | null;
    selectedRange: MarketStockCardChartRange;
    pendingRange: MarketStockCardChartRange | null;
    rangeLoading: Partial<Record<MarketStockCardChartRange, boolean>>;
    rangeError: Partial<Record<MarketStockCardChartRange, string | null>>;
    onRangeSelect: (range: MarketStockCardChartRange) => void;
    isLoading?: boolean;
    isLive?: boolean;
}) {
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const width = 360;
    const height = 150;
    const padding = { top: 8, right: 2, bottom: 18, left: 2 };
    const validPoints = (points ?? []).filter((point) => Number.isFinite(point.close));
    const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);
    const rangeControls = (
        <div className="stock-card-chart-ranges" onClick={(event) => event.stopPropagation()}>
            {STOCK_CARD_CHART_RANGES.map((range) => {
                const isActive = selectedRange === range.id;
                const isPending = pendingRange === range.id || Boolean(rangeLoading[range.id]);
                const error = rangeError[range.id];
                return (
                    <button
                        key={range.id}
                        type="button"
                        className={[
                            'stock-card-chart-range',
                            isActive ? 'is-active' : '',
                            isPending ? 'is-loading' : '',
                            error ? 'has-error' : '',
                        ]
                            .filter(Boolean)
                            .join(' ')}
                        title={error ? `${range.title}: ${error}` : range.title}
                        aria-pressed={isActive}
                        onClick={(event) => {
                            event.stopPropagation();
                            onRangeSelect(range.id);
                        }}
                    >
                        {range.label}
                    </button>
                );
            })}
        </div>
    );
    const skeletonTop = 6;
    const skeletonBottom = height - 4;
    const skeletonSpan = Math.max(1, skeletonBottom - skeletonTop);
    const skeletonPointCount = 44;
    let skeletonSeed = `${symbol}:${selectedRange}`
        .split('')
        .reduce((acc, char) => ((acc * 31 + char.charCodeAt(0)) >>> 0), 7);
    const seededRandom = () => {
        skeletonSeed = (1664525 * skeletonSeed + 1013904223) >>> 0;
        return skeletonSeed / 0xffffffff;
    };
    const trend = (seededRandom() - 0.5) * 0.2;
    let level = 0.56 + (seededRandom() - 0.5) * 0.16;
    const skeletonPoints: Array<[number, number]> = Array.from({ length: skeletonPointCount }, (_, index) => {
        if (index > 0) {
            const burst = index % 7 === 0 ? (seededRandom() - 0.5) * 0.22 : 0;
            const drift = trend / skeletonPointCount + (seededRandom() - 0.5) * 0.16 + burst;
            level = clamp(level + drift, 0.08, 0.92);
        }
        const x = (index / Math.max(1, skeletonPointCount - 1)) * width;
        const y = skeletonTop + level * skeletonSpan;
        return [x, y];
    });
    const skeletonLinePath = skeletonPoints
        .map(([x, y], index) => `${index === 0 ? 'M' : 'L'} ${x} ${y}`)
        .join(' ');
    const skeletonAreaPath = `${skeletonLinePath} L ${width} ${height - 2} L 0 ${height - 2} Z`;
    const skeletonIdBase = `stock-card-skeleton-${symbol.replace(/[^a-zA-Z0-9]/g, '') || 'sym'}-${selectedRange}`;
    const skeletonAreaGradientId = `${skeletonIdBase}-area`;

    if (isLoading) {
        return (
            <div className="stock-card-chart-shell">
                {rangeControls}
                <div className="stock-card-inline-chart-skeleton" aria-hidden="true">
                    <div className="stock-card-inline-chart-track stock-card-inline-skeleton-pulse">
                        <svg
                            className="stock-card-inline-chart-svg"
                            viewBox={`0 0 ${width} ${height}`}
                            preserveAspectRatio="none"
                        >
                            <defs>
                                <linearGradient id={skeletonAreaGradientId} x1="0%" y1="0%" x2="0%" y2="100%">
                                    <stop offset="0%" stopColor="rgba(148,163,184,0.22)" />
                                    <stop offset="100%" stopColor="rgba(148,163,184,0)" />
                                </linearGradient>
                            </defs>
                            <line className="stock-card-inline-chart-gridline" x1={0} x2={width} y1={height * 0.56} y2={height * 0.56} />
                            <path className="stock-card-inline-chart-area" d={skeletonAreaPath} fill={`url(#${skeletonAreaGradientId})`} />
                            <path
                                className="stock-card-inline-chart-line"
                                d={skeletonLinePath}
                                pathLength={100}
                            />
                        </svg>
                    </div>
                </div>
            </div>
        );
    }

    if (validPoints.length < 2) {
        return (
            <div className="stock-card-chart-shell">
                {rangeControls}
                <div className="stock-card-chart-empty">Veri yok</div>
            </div>
        );
    }

    const values = validPoints.map((point) => point.close);
    let minValue = Math.min(...values);
    let maxValue = Math.max(...values);
    const spanRaw = Math.max(0.01, maxValue - minValue);
    minValue -= spanRaw * 0.08;
    maxValue += spanRaw * 0.08;
    const span = Math.max(0.01, maxValue - minValue);
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const isBist = isBistSymbol(symbol);
    const useTimeScale = isBist && selectedRange === '1d' && validPoints.length > 0;
    let startTimeMs = 0;
    let endTimeMs = 0;
    if (useTimeScale) {
        const d = new Date(validPoints[0].time);
        const start = new Date(d); start.setHours(10, 0, 0, 0);
        const end = new Date(d);
        const lastPointDate = new Date(validPoints[validPoints.length - 1].time);
        const hasTodayPoint = istanbulDateKey(lastPointDate) === istanbulDateKey(new Date());
        if (isLive || hasTodayPoint) {
            end.setHours(18, 0, 0, 0);
        } else {
            const lastMinutes = lastPointDate.getHours() * 60 + lastPointDate.getMinutes();
            const roundedEndHour = Math.min(18, Math.max(13, Math.ceil(lastMinutes / 60)));
            end.setHours(roundedEndHour, 0, 0, 0);
            if (end.getTime() <= lastPointDate.getTime()) {
                end.setHours(Math.min(18, end.getHours() + 1), 0, 0, 0);
            }
        }
        startTimeMs = start.getTime();
        endTimeMs = Math.max(end.getTime(), startTimeMs + 60 * 60 * 1000);
    }

    const xFor = (index: number) => {
        if (useTimeScale) {
            const pointTime = new Date(validPoints[index].time).getTime();
            let ratio = (pointTime - startTimeMs) / (endTimeMs - startTimeMs);
            ratio = Math.max(0, Math.min(1, ratio));
            return padding.left + ratio * plotWidth;
        }
        return padding.left + (validPoints.length === 1 ? 0 : (index / (validPoints.length - 1)) * plotWidth);
    };

    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;
    const pathData = validPoints
        .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(index)} ${yFor(point.close)}`)
        .join(' ');
    const areaData = `${pathData} L ${xFor(validPoints.length - 1)} ${height - 2} L ${padding.left} ${height - 2} Z`;
    const color = changePct == null || changePct >= 0 ? '#22c55e' : '#ff4d5e';
    const gradientId = `stock-card-area-${symbol.replace(/[^a-zA-Z0-9]/g, '')}-${selectedRange}`;
    const baselineValue =
        selectedRange === '1d' && Number.isFinite(previousClose)
            ? Number(previousClose)
            : validPoints.length >= 2
                ? validPoints[validPoints.length - 2].close
                : validPoints[0].close;
    const baselineY = clamp(yFor(baselineValue), padding.top, height - padding.bottom);
    const timeTickIndexes = Array.from(
        new Set([0, Math.floor((validPoints.length - 1) / 2), validPoints.length - 1]),
    );
    const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        if (rect.width <= 0) return;
        const x = ((event.clientX - rect.left) / rect.width) * width;
        let closestIndex = 0;
        let minDiff = Infinity;
        for (let i = 0; i < validPoints.length; i++) {
            const diff = Math.abs(xFor(i) - x);
            if (diff < minDiff) {
                minDiff = diff;
                closestIndex = i;
            }
        }
        setHoverIndex(closestIndex);
    };
    const activeHoverIndex = hoverIndex == null ? null : clamp(hoverIndex, 0, validPoints.length - 1);
    const hoverPoint = activeHoverIndex == null ? null : validPoints[activeHoverIndex];
    const hoverX = activeHoverIndex == null ? null : xFor(activeHoverIndex);
    const hoverY = hoverPoint ? yFor(hoverPoint.close) : null;
    const tooltipWidth = 132;
    const tooltipHeight = 58;
    const tooltipX = hoverX == null ? 0 : clamp(hoverX + 12, 8, width - tooltipWidth - 8);
    const tooltipY = hoverY == null ? 0 : clamp(hoverY - tooltipHeight / 2, padding.top, height - tooltipHeight - 8);

    return (
        <div className="stock-card-chart-shell">
            {rangeControls}
            <svg
                className="stock-card-mini-chart"
                viewBox={`0 0 ${width} ${height}`}
                preserveAspectRatio="none"
                role="img"
                aria-label={`${symbol} ${selectedRange} çizgi grafiği`}
                onPointerMove={handlePointerMove}
                onPointerLeave={() => setHoverIndex(null)}
            >
                <defs>
                    <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor={color} stopOpacity="0.22" />
                        <stop offset="100%" stopColor={color} stopOpacity="0" />
                    </linearGradient>
                </defs>
                <line
                    x1={0}
                    x2={width}
                    y1={baselineY}
                    y2={baselineY}
                    stroke="rgba(255,255,255,0.16)"
                    strokeDasharray="3 4"
                />
                {useTimeScale ? (
                    [startTimeMs, startTimeMs + (endTimeMs - startTimeMs) / 2, endTimeMs].map((timeMs, index) => {
                        const x = index === 0 ? padding.left : index === 2 ? width - padding.right : padding.left + plotWidth / 2;
                        return (
                            <text
                                key={timeMs}
                                x={x}
                                y={height - 6}
                                fill="rgba(255,255,255,0.24)"
                                fontSize="10"
                                fontFamily="monospace"
                                textAnchor={index === 0 ? 'start' : index === 2 ? 'end' : 'middle'}
                            >
                                {formatStockCardAxisTime(new Date(timeMs))}
                            </text>
                        );
                    })
                ) : (
                    timeTickIndexes.map((index) => {
                        const x = xFor(index);
                        return (
                            <text
                                key={index}
                                x={x}
                                y={height - 6}
                                fill="rgba(255,255,255,0.24)"
                                fontSize="10"
                                fontFamily="monospace"
                                textAnchor={index === 0 ? 'start' : index === validPoints.length - 1 ? 'end' : 'middle'}
                            >
                                {formatStockCardAxisDate(validPoints[index].time, selectedRange)}
                            </text>
                        );
                    })
                )}
                <path d={areaData} fill={`url(#${gradientId})`} />
                <path d={pathData} fill="none" stroke={color} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
                
                {selectedRange === '1d' && (
                    <>
                        {isLive && (
                            <circle cx={xFor(validPoints.length - 1)} cy={yFor(validPoints[validPoints.length - 1].close)} r="4" fill={color} opacity="0.6">
                                <animate attributeName="r" values="4; 14; 14" keyTimes="0; 0.5; 1" dur="2s" repeatCount="indefinite" />
                                <animate attributeName="opacity" values="0.6; 0; 0" keyTimes="0; 0.5; 1" dur="2s" repeatCount="indefinite" />
                            </circle>
                        )}
                        <circle cx={xFor(validPoints.length - 1)} cy={yFor(validPoints[validPoints.length - 1].close)} r="4" fill={color} />
                    </>
                )}

                {hoverPoint && hoverX != null && hoverY != null && (
                    <g className="stock-card-chart-hover">
                        <line
                            x1={hoverX}
                            x2={hoverX}
                            y1={padding.top}
                            y2={height - padding.bottom}
                            stroke="rgba(255,255,255,0.42)"
                            strokeDasharray="4 4"
                        />
                        <line
                            x1={padding.left}
                            x2={width - padding.right}
                            y1={hoverY}
                            y2={hoverY}
                            stroke="rgba(255,255,255,0.24)"
                            strokeDasharray="5 5"
                        />
                        <circle cx={hoverX} cy={hoverY} r="3.5" fill={color} stroke="#0a0c0f" strokeWidth="1.5" />
                        <g transform={`translate(${tooltipX}, ${tooltipY})`}>
                            <rect width={tooltipWidth} height={tooltipHeight} rx="4" fill="#07090b" stroke="rgba(59,130,246,0.2)" />
                            <text x="10" y="19" fill="#d8dee9" fontSize="12" fontFamily="monospace">
                                {formatStockCardChartDate(hoverPoint.time, selectedRange)}
                            </text>
                            <text x="10" y="42" fill="#f8fafc" fontSize="12" fontFamily="monospace" fontWeight="700">
                                {symbol}
                            </text>
                            <text x={tooltipWidth - 10} y="42" fill="#f8fafc" fontSize="12" fontFamily="monospace" fontWeight="700" textAnchor="end">
                                {formatCardCurrency(hoverPoint.close, currency)}
                            </text>
                        </g>
                    </g>
                )}
            </svg>
        </div>
    );
}

function emptyStockCardItem(symbol: string): MarketStockCardItem {
    return {
        symbol,
        company: symbol,
        yahoo_symbol: null,
        price: null,
        currency: 'TRY',
        change: null,
        change_pct: null,
        volume: null,
        volume_lot: null,
        volume_tl: null,
        market_cap: null,
        high: null,
        low: null,
        previous_close: null,
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
        market_state: '',
        as_of: null,
        session_status: 'unknown',
        session_label: 'Veri bekleniyor',
        is_live: false,
        is_stale: false,
        last_trade_at: null,
        last_trade_date: null,
        line_points: [],
        error: 'data_unavailable',
        logo_url: null,
        logo_source: null,
    };
}

const STOCK_CARD_PERFORMANCE_KEYS: Array<{
    key: 'return_1w_pct' | 'return_1m_pct' | 'return_3m_pct' | 'return_6m_pct' | 'return_ytd_pct' | 'return_1y_pct';
    label: string;
}> = [
    { key: 'return_1w_pct', label: '1H' },
    { key: 'return_1m_pct', label: '1A' },
    { key: 'return_3m_pct', label: '3A' },
    { key: 'return_6m_pct', label: '6A' },
    { key: 'return_ytd_pct', label: 'YTD' },
    { key: 'return_1y_pct', label: '1Y' },
];

function MarketStockCard({
    item,
    onOpen,
    onRemove,
    onMoveStart,
    onMoveOver,
    onMoveEnd,
    isLoading = false,
    isDragging = false,
}: {
    item: MarketStockCardItem;
    onOpen: () => void;
    onRemove: () => void;
    onMoveStart: () => void;
    onMoveOver: (placement: StockCardDropPlacement) => void;
    onMoveEnd: () => void;
    isLoading?: boolean;
    isDragging?: boolean;
}) {
    const [selectedRange, setSelectedRange] = useState<MarketStockCardChartRange>('1d');
    const [pendingRange, setPendingRange] = useState<MarketStockCardChartRange | null>(null);
    const [chartDataByRange, setChartDataByRange] = useState<Partial<Record<MarketStockCardChartRange, MarketIndexLinePoint[]>>>(
        () => ({ '1d': item.line_points ?? [] }),
    );
    const [rangeLoading, setRangeLoading] = useState<Partial<Record<MarketStockCardChartRange, boolean>>>({});
    const [rangeError, setRangeError] = useState<Partial<Record<MarketStockCardChartRange, string | null>>>({});
    const chartAbortRef = useRef<AbortController | null>(null);
    const chartRequestIdRef = useRef(0);

    useEffect(() => {
        setChartDataByRange((previous) => ({ ...previous, '1d': item.line_points ?? [] }));
        setRangeError((previous) => ({ ...previous, '1d': null }));
    }, [item.line_points]);

    useEffect(() => {
        return () => {
            chartAbortRef.current?.abort();
        };
    }, []);

    const [hoveredData, setHoveredData] = useState<{
        base: number | null | undefined;
        high: number | null | undefined;
        low: number | null | undefined;
        anchorX: number;
    } | null>(null);
    const [tooltipActive, setTooltipActive] = useState(false);

    const performanceRef = useRef<HTMLDivElement>(null);
    const [tooltipStyles, setTooltipStyles] = useState<React.CSSProperties>({});
    const tooltipCloseTimerRef = useRef<number | null>(null);
    const tooltipOpenFrameRef = useRef<number | null>(null);

    useLayoutEffect(() => {
        if (!isLoading && hoveredData && performanceRef.current) {
            const performanceRect = performanceRef.current.getBoundingClientRect();
            const cardElement = performanceRef.current.closest('.stock-card');
            const cardRect = cardElement?.getBoundingClientRect() ?? performanceRect;
            const halfTooltipWidth = cardRect.width / 2;
            const viewportPadding = 12;
            const minCenterX = viewportPadding + halfTooltipWidth;
            const maxCenterX = window.innerWidth - viewportPadding - halfTooltipWidth;
            const tooltipCenterX = Math.min(Math.max(hoveredData.anchorX, minCenterX), maxCenterX);
            setTooltipStyles({
                top: `${performanceRect.top - 8}px`,
                left: `${tooltipCenterX}px`,
                width: `${cardRect.width}px`,
                maxWidth: `${cardRect.width}px`,
            });
        }
    }, [hoveredData, isLoading]);

    useEffect(() => {
        return () => {
            if (tooltipCloseTimerRef.current != null) {
                window.clearTimeout(tooltipCloseTimerRef.current);
            }
            if (tooltipOpenFrameRef.current != null) {
                window.cancelAnimationFrame(tooltipOpenFrameRef.current);
            }
        };
    }, []);

    useEffect(() => {
        if (isLoading) {
            if (tooltipCloseTimerRef.current != null) {
                window.clearTimeout(tooltipCloseTimerRef.current);
                tooltipCloseTimerRef.current = null;
            }
            if (tooltipOpenFrameRef.current != null) {
                window.cancelAnimationFrame(tooltipOpenFrameRef.current);
                tooltipOpenFrameRef.current = null;
            }
            setTooltipActive(false);
            setHoveredData(null);
        }
    }, [isLoading]);

    const openPerformanceTooltip = (
        data: {
            base: number | null | undefined;
            high: number | null | undefined;
            low: number | null | undefined;
            anchorX: number;
        },
    ) => {
        if (tooltipCloseTimerRef.current != null) {
            window.clearTimeout(tooltipCloseTimerRef.current);
            tooltipCloseTimerRef.current = null;
        }
        if (tooltipOpenFrameRef.current != null) {
            window.cancelAnimationFrame(tooltipOpenFrameRef.current);
            tooltipOpenFrameRef.current = null;
        }
        setTooltipActive(false);
        setHoveredData(data);
        tooltipOpenFrameRef.current = window.requestAnimationFrame(() => {
            tooltipOpenFrameRef.current = window.requestAnimationFrame(() => {
                setTooltipActive(true);
                tooltipOpenFrameRef.current = null;
            });
        });
    };

    const closePerformanceTooltip = () => {
        if (tooltipOpenFrameRef.current != null) {
            window.cancelAnimationFrame(tooltipOpenFrameRef.current);
            tooltipOpenFrameRef.current = null;
        }
        setTooltipActive(false);
        if (tooltipCloseTimerRef.current != null) {
            window.clearTimeout(tooltipCloseTimerRef.current);
        }
        tooltipCloseTimerRef.current = window.setTimeout(() => {
            setHoveredData(null);
            tooltipCloseTimerRef.current = null;
        }, 180);
    };

    const handleRangeSelect = (nextRange: MarketStockCardChartRange) => {
        if (isLoading) {
            return;
        }
        if (nextRange === selectedRange || rangeLoading[nextRange]) {
            return;
        }

        const cachedPoints = chartDataByRange[nextRange];
        if (cachedPoints !== undefined && !rangeError[nextRange]) {
            setSelectedRange(nextRange);
            setPendingRange(null);
            return;
        }

        chartAbortRef.current?.abort();
        const controller = new AbortController();
        chartAbortRef.current = controller;
        const requestId = chartRequestIdRef.current + 1;
        chartRequestIdRef.current = requestId;
        setPendingRange(nextRange);
        setRangeLoading((previous) => ({ ...previous, [nextRange]: true }));
        setRangeError((previous) => ({ ...previous, [nextRange]: null }));

        apiClient
            .marketStockCardChart(item.symbol, nextRange, { signal: controller.signal })
            .then((payload) => {
                if (controller.signal.aborted || requestId !== chartRequestIdRef.current) {
                    return;
                }
                const nextPoints = payload.line_points ?? [];
                setChartDataByRange((previous) => ({ ...previous, [nextRange]: nextPoints }));
                if (payload.error || nextPoints.length < 2) {
                    setRangeError((previous) => ({
                        ...previous,
                        [nextRange]: payload.error || 'Veri yok',
                    }));
                    return;
                }
                setSelectedRange(nextRange);
            })
            .catch((error) => {
                if ((error as Error)?.name === 'AbortError' || requestId !== chartRequestIdRef.current) {
                    return;
                }
                setRangeError((previous) => ({
                    ...previous,
                    [nextRange]: (error as Error)?.message || 'Grafik verisi alınamadı',
                }));
            })
            .finally(() => {
                if (requestId !== chartRequestIdRef.current) {
                    return;
                }
                setRangeLoading((previous) => ({ ...previous, [nextRange]: false }));
                setPendingRange(null);
                if (chartAbortRef.current === controller) {
                    chartAbortRef.current = null;
                }
            });
    };

    const formatTooltipMetricValue = (value: number | null | undefined): string => {
        if (value == null || !Number.isFinite(value)) return '-';
        return value.toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        });
    };
    const latestPointTime = latestStockCardPointTime(item.line_points);
    const hasPreviousSessionData = item.is_stale === true || item.session_status === 'previous_session' || isPreviousIstanbulDate(item.last_trade_at || latestPointTime);
    const isCardLive = !hasPreviousSessionData && (item.is_live === true || item.session_status === 'open');
    const sessionLabel = item.session_label || (isCardLive ? 'Canlı' : hasPreviousSessionData ? 'Piyasa kapalı' : 'Piyasa kapalı');
    const sessionStatusClass = isCardLive
        ? 'is-live'
        : hasPreviousSessionData
            ? 'is-previous'
            : 'is-closed';
    const sessionTime = formatStockCardTradeTime(item.last_trade_at || latestPointTime || item.as_of);
    const footerLabel = isCardLive ? 'Canlı güncelleme' : hasPreviousSessionData ? 'Son işlem' : sessionLabel;

    return (
        <article
            className={`stock-card${isDragging ? ' stock-card-dragging' : ''}`}
            data-stock-card-symbol={item.symbol}
            onClick={onOpen}
            onDragEnter={(event) => {
                event.preventDefault();
            }}
            onDragOver={(event) => {
                event.preventDefault();
                event.stopPropagation();
                event.dataTransfer.dropEffect = 'move';
                const rect = event.currentTarget.getBoundingClientRect();
                const placement: StockCardDropPlacement =
                    event.clientX >= rect.left + rect.width / 2 ? 'after' : 'before';
                onMoveOver(placement);
            }}
            onDrop={(event) => {
                event.preventDefault();
                event.stopPropagation();
                onMoveEnd();
            }}
        >
            <div
                className="stock-card-controls"
                onClick={(event) => event.stopPropagation()}
            >
                <button
                    type="button"
                    className="stock-card-control stock-card-drag-handle"
                    draggable
                    onDragStart={(event) => {
                        event.stopPropagation();
                        event.dataTransfer.effectAllowed = 'move';
                        event.dataTransfer.setData('text/plain', item.symbol);
                        const cardElement = event.currentTarget.closest('.stock-card') as HTMLElement | null;
                        if (cardElement) {
                            event.dataTransfer.setDragImage(cardElement, cardElement.offsetWidth / 2, 24);
                        }
                        onMoveStart();
                    }}
                    onDragEnd={(event) => {
                        event.stopPropagation();
                        onMoveEnd();
                    }}
                    aria-label={`${item.symbol} kartını taşı`}
                    title="Kartı taşı"
                >
                    <GripHorizontal size={14} aria-hidden="true" />
                </button>
                <button
                    type="button"
                    className="stock-card-control stock-card-control-remove"
                    onClick={(event) => {
                        event.stopPropagation();
                        onRemove();
                    }}
                    aria-label={`${item.symbol} kartını kaldır`}
                    title="Kartı kaldır"
                >
                    <X size={13} aria-hidden="true" />
                </button>
            </div>
            <div className="stock-card-head">
                <div className="stock-card-identity">
                    <SymbolLogo
                        symbol={item.symbol}
                        name={item.company}
                        kind="stock"
                        logoUrl={item.logo_url}
                        size="md"
                        className="stock-card-logo"
                    />
                    <div>
                        <h3>{item.symbol}</h3>
                        <span>{item.company}</span>
                    </div>
                </div>
                {!isLoading && (
                    <span 
                        className={`stock-card-session-dot ${sessionStatusClass}`}
                        title={sessionLabel}
                    />
                )}
            </div>

            <div className="stock-card-price-row">
                {isLoading ? (
                    <>
                        <span className="stock-card-inline-skeleton-price stock-card-inline-skeleton-pulse" aria-hidden="true" />
                        <span className="stock-card-inline-skeleton-change stock-card-inline-skeleton-pulse" aria-hidden="true" />
                    </>
                ) : (
                    <>
                        <span className="stock-card-price">{formatCardCurrency(item.price, item.currency)}</span>
                        <span className={`stock-card-change ${getTableChangeClass(item.change_pct)}`}>
                            {formatCardPct(item.change_pct)}
                        </span>
                    </>
                )}
            </div>

            <div className="stock-card-metrics">
                <div>
                    <span>Yüksek</span>
                    <strong>
                        {isLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardCurrency(item.high, item.currency)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Düşük</span>
                    <strong>
                        {isLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardCurrency(item.low, item.currency)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Önc.Kap.</span>
                    <strong>
                        {isLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardCurrency(item.previous_close, item.currency)
                        )}
                    </strong>
                </div>
            </div>

            <StockCardMiniChart
                symbol={item.symbol}
                points={chartDataByRange[selectedRange]}
                previousClose={item.previous_close}
                changePct={(() => {
                    if (selectedRange === '1d') return item.change_pct;
                    if (selectedRange === '1w') return item.return_1w_pct;
                    if (selectedRange === '1m') return item.return_1m_pct;
                    if (selectedRange === '1y') return item.return_1y_pct;
                    return item.change_pct;
                })()}
                currency={item.currency}
                selectedRange={selectedRange}
                pendingRange={pendingRange}
                rangeLoading={rangeLoading}
                rangeError={rangeError}
                onRangeSelect={handleRangeSelect}
                isLoading={isLoading}
                isLive={isCardLive}
            />

            <div className="stock-card-detail-metrics">
                <div>
                    <span>F/K</span>
                    <strong>
                        {isLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardPositiveRatio(item.fk)
                        )}
                    </strong>
                </div>
                <div>
                    <span>FD/FAVÖK</span>
                    <strong>
                        {isLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardPositiveRatio(item.fd_favok)
                        )}
                    </strong>
                </div>
                <div>
                    <span>PD/DD</span>
                    <strong>
                        {isLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardPositiveRatio(item.pd_dd)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Net Borç/FAVÖK</span>
                    <strong>
                        {isLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardRatio(item.net_borc_favok)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Hacim Lot</span>
                    <strong>
                        {isLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardFullNumber(item.volume_lot)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Hacim TL</span>
                    <strong>
                        {isLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardFullCurrency(item.volume_tl, item.currency)
                        )}
                    </strong>
                </div>
                <div className="stock-card-detail-wide">
                    <span>Piyasa Değeri</span>
                    <strong>
                        {isLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric-wide" aria-hidden="true" />
                        ) : (
                            formatCardFullCurrency(item.market_cap, item.currency)
                        )}
                    </strong>
                </div>
            </div>


            <div className="stock-card-performance" ref={performanceRef}>
                {!isLoading && hoveredData && createPortal(
                    <div
                        className={`stock-card-performance-tooltip${tooltipActive ? ' active' : ''}`}
                        style={tooltipStyles}
                    >
                        <div className="tooltip-row">
                            <div className="tooltip-metric">
                                <span className="tooltip-metric-label">Önc.Kap.:</span>
                                <strong className="tooltip-metric-value">{formatTooltipMetricValue(hoveredData.base)}</strong>
                            </div>
                            <div className="tooltip-metric">
                                <span className="tooltip-metric-label">Düşük:</span>
                                <strong className="tooltip-metric-value">{formatTooltipMetricValue(hoveredData.low)}</strong>
                            </div>
                            <div className="tooltip-metric">
                                <span className="tooltip-metric-label">Yüksek:</span>
                                <strong className="tooltip-metric-value">{formatTooltipMetricValue(hoveredData.high)}</strong>
                            </div>
                        </div>
                    </div>,
                    document.body
                )}
                {isLoading
                    ? STOCK_CARD_PERFORMANCE_KEYS.map(({ key }) => (
                          <span key={key} className="stock-card-performance-chip stock-card-performance-chip-skeleton" aria-hidden="true">
                              <span className="stock-card-inline-skeleton-chip-line stock-card-inline-skeleton-pulse" />
                          </span>
                      ))
                    : STOCK_CARD_PERFORMANCE_KEYS.map(({ key, label }) => {
                          const value = item[key];
                          const period = key.replace('return_', '').replace('_pct', '');
                          const base = (item as any)[`base_${period}`];
                          const high = (item as any)[`high_${period}`];
                          const low = (item as any)[`low_${period}`];

                          return (
                              <span
                                  key={key}
                                  className={`stock-card-performance-chip ${getTableChangeClass(value)}`}
                                  onMouseEnter={(event) => {
                                      const chipRect = event.currentTarget.getBoundingClientRect();
                                      openPerformanceTooltip({
                                          base,
                                          high,
                                          low,
                                          anchorX: chipRect.left + chipRect.width / 2,
                                      });
                                  }}
                                  onMouseLeave={closePerformanceTooltip}
                              >
                                  <strong>{label}</strong>
                                  {formatCardPct(value)}
                              </span>
                          );
                      })}
            </div>

            <div className="stock-card-foot">
                {isLoading ? (
                    <span className="stock-card-inline-skeleton-time stock-card-inline-skeleton-pulse" aria-hidden="true" />
                ) : (
                    <span>{footerLabel} {sessionTime}</span>
                )}
            </div>
        </article>
    );
}

const TREEMAP_LAYOUT_WIDTH = 100;
const TREEMAP_LAYOUT_HEIGHT = 62;

type HeatmapTile = {
    row: MarketIndexConstituent;
    impactPct: number | null;
    changePct: number | null;
    pointEffect: number | null;
    left: number;
    top: number;
    width: number;
    height: number;
    areaPct: number;
};

type HeatmapLayoutItem = {
    row: MarketIndexConstituent;
    area: number;
    impactPct: number | null;
    changePct: number | null;
};

type HeatmapLayoutRect = {
    x: number;
    y: number;
    width: number;
    height: number;
};

function treemapValue(row: MarketIndexConstituent): number {
    if (row.weight_pct != null && row.weight_pct > 0) return row.weight_pct;
    const impact = Math.abs(row.point_effect ?? 0);
    return impact > 0 ? impact : 0.01;
}

function worstTreemapAspect(row: HeatmapLayoutItem[], shortSide: number): number {
    if (!row.length || shortSide <= 0) return Number.POSITIVE_INFINITY;
    const areas = row.map((item) => item.area);
    const sum = areas.reduce((total, area) => total + area, 0);
    const min = Math.min(...areas);
    const max = Math.max(...areas);
    if (sum <= 0 || min <= 0) return Number.POSITIVE_INFINITY;
    const sideSquared = shortSide * shortSide;
    return Math.max((sideSquared * max) / (sum * sum), (sum * sum) / (sideSquared * min));
}

function heatmapTileFromRect(item: HeatmapLayoutItem, rect: HeatmapLayoutRect): HeatmapTile {
    const width = Math.max(0, rect.width);
    const height = Math.max(0, rect.height);
    return {
        row: item.row,
        impactPct: item.impactPct,
        changePct: item.changePct,
        pointEffect: item.row.point_effect,
        left: (rect.x / TREEMAP_LAYOUT_WIDTH) * 100,
        top: (rect.y / TREEMAP_LAYOUT_HEIGHT) * 100,
        width: (width / TREEMAP_LAYOUT_WIDTH) * 100,
        height: (height / TREEMAP_LAYOUT_HEIGHT) * 100,
        areaPct: (width * height) / (TREEMAP_LAYOUT_WIDTH * TREEMAP_LAYOUT_HEIGHT) * 100,
    };
}

function layoutTreemapRow(
    row: HeatmapLayoutItem[],
    remaining: HeatmapLayoutRect,
    tiles: HeatmapTile[],
): void {
    if (!row.length || remaining.width <= 0 || remaining.height <= 0) return;
    const rowArea = row.reduce((total, item) => total + item.area, 0);
    if (rowArea <= 0) return;

    if (remaining.width >= remaining.height) {
        const columnWidth = Math.min(remaining.width, rowArea / remaining.height);
        let y = remaining.y;
        row.forEach((item, index) => {
            const height = index === row.length - 1 ? remaining.y + remaining.height - y : item.area / columnWidth;
            tiles.push(heatmapTileFromRect(item, {
                x: remaining.x,
                y,
                width: columnWidth,
                height: Math.max(0, height),
            }));
            y += height;
        });
        remaining.x += columnWidth;
        remaining.width = Math.max(0, remaining.width - columnWidth);
        return;
    }

    const rowHeight = Math.min(remaining.height, rowArea / remaining.width);
    let x = remaining.x;
    row.forEach((item, index) => {
        const width = index === row.length - 1 ? remaining.x + remaining.width - x : item.area / rowHeight;
        tiles.push(heatmapTileFromRect(item, {
            x,
            y: remaining.y,
            width: Math.max(0, width),
            height: rowHeight,
        }));
        x += width;
    });
    remaining.y += rowHeight;
    remaining.height = Math.max(0, remaining.height - rowHeight);
}

function buildHeatmapTiles(items: MarketIndexConstituent[], indexLevel: number | null): HeatmapTile[] {
    const sorted = [...items].sort((a, b) => {
        const result = treemapValue(b) - treemapValue(a);
        return result === 0 ? a.symbol.localeCompare(b.symbol, 'tr') : result;
    });
    const totalValue = sorted.reduce((total, row) => total + treemapValue(row), 0);
    if (totalValue <= 0) return [];

    const totalArea = TREEMAP_LAYOUT_WIDTH * TREEMAP_LAYOUT_HEIGHT;
    const layoutItems: HeatmapLayoutItem[] = sorted.map((row) => {
        const value = treemapValue(row);
        return {
            row,
            area: (value / totalValue) * totalArea,
            impactPct: getImpactPct(row, indexLevel),
            changePct: numericOrNull(row.change_pct),
        };
    });
    const remaining: HeatmapLayoutRect = {
        x: 0,
        y: 0,
        width: TREEMAP_LAYOUT_WIDTH,
        height: TREEMAP_LAYOUT_HEIGHT,
    };
    const tiles: HeatmapTile[] = [];
    let row: HeatmapLayoutItem[] = [];

    layoutItems.forEach((item) => {
        const nextRow = [...row, item];
        const shortSide = Math.min(remaining.width, remaining.height);
        if (!row.length || worstTreemapAspect(nextRow, shortSide) <= worstTreemapAspect(row, shortSide)) {
            row = nextRow;
            return;
        }
        layoutTreemapRow(row, remaining, tiles);
        row = [item];
    });
    layoutTreemapRow(row, remaining, tiles);
    return tiles;
}

function formatTreemapScalePct(value: number): string {
    const rounded = Math.abs(value) >= 1 ? Math.round(value) : Number(value.toFixed(2));
    if (rounded === 0) return '0%';
    return `${rounded.toLocaleString('tr-TR', {
        minimumFractionDigits: Math.abs(rounded) >= 1 ? 0 : 2,
        maximumFractionDigits: Math.abs(rounded) >= 1 ? 0 : 2,
    })}%`;
}

function getTreemapTileColor(changePct: number | null, scaledMove: number): string {
    if (changePct == null || changePct === 0) return 'hsl(146 10% 28%)';
    if (changePct > 0) {
        return `hsl(123 92% ${23 + scaledMove * 24}%)`;
    }
    return `hsl(0 40% ${25 + scaledMove * 18}%)`;
}

interface MarketsViewProps {
    routeSection?: MarketSection;
    routeStockIndex?: MarketStockIndex;
    routeSelectedIndex?: MarketIndexCode | null;
    routeReturnMode?: StockReturnMode;
    onNavigateSection?: (section: MarketSection) => void;
    onNavigateFundSection?: (section: MarketsNavigationFundSection) => void;
    onNavigateStockIndex?: (index: MarketStockIndex) => void;
    onNavigateReturnMode?: (mode: StockReturnMode) => void;
    onNavigateIndexDetail?: (index: MarketIndexCode | null) => void;
    onOpenTicker?: (ticker: string) => void;
    onOpenFund?: (fundCode: string) => void;
}

export default function MarketsView({
    routeSection = 'stocks',
    routeStockIndex = 'XUTUM',
    routeSelectedIndex = null,
    routeReturnMode = DEFAULT_STOCK_RETURN_MODE,
    onNavigateSection,
    onNavigateFundSection,
    onNavigateStockIndex,
    onNavigateReturnMode,
    onNavigateIndexDetail,
    onOpenTicker,
    onOpenFund,
}: MarketsViewProps) {
    const [market, setMarket] = useState<MarketUniverseResponse | null>(() => marketUniverseMemoryCache?.data || null);
    const [stocks, setStocks] = useState<MarketStocksResponse | null>(null);
    const [indices, setIndices] = useState<MarketIndicesResponse | null>(null);
    const [indexDetail, setIndexDetail] = useState<MarketIndexDetailResponse | null>(null);
    const [loading, setLoading] = useState(() => !marketUniverseMemoryCache);
    const [error, setError] = useState<string | null>(null);
    const [stocksLoading, setStocksLoading] = useState(false);
    const [stocksError, setStocksError] = useState<string | null>(null);
    const [indicesLoading, setIndicesLoading] = useState(false);
    const [indicesError, setIndicesError] = useState<string | null>(null);
    const [indexDetailLoading, setIndexDetailLoading] = useState(false);
    const [indexDetailError, setIndexDetailError] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [stockCardSearchTerm, setStockCardSearchTerm] = useState('');
    const [navCollapsed, setNavCollapsed] = useState(false);
    const [activeSection, setActiveSection] = useState<MarketSection>(routeSection);
    const [isMobileViewport, setIsMobileViewport] = useState(() => (
        typeof window !== 'undefined' && window.matchMedia('(max-width: 767px)').matches
    ));
    const [mobileMarketPanelOpen, setMobileMarketPanelOpen] = useState(false);
    const [mobileIndexDetail, setMobileIndexDetail] = useState<MarketIndexDetailResponse | null>(null);
    const [mobileWatchlistFundRows, setMobileWatchlistFundRows] = useState<Record<string, FundDetail | null>>({});
    const [selectedIndex, setSelectedIndex] = useState<MarketIndexCode | null>(routeSelectedIndex);
    const [stockIndex, setStockIndex] = useState<MarketStockIndex>(routeStockIndex);
    const [returnMode, setReturnMode] = useState<StockReturnMode>(routeReturnMode);
    const [terminalNow, setTerminalNow] = useState(() => new Date());
    const [stockCardSymbols, setStockCardSymbols] = useState<string[]>(readStoredStockCards);
    const [stockCards, setStockCards] = useState<MarketStockCardsResponse | null>(() => {
        const symbols = readStoredStockCards();
        return marketStockCardsMemoryCache.get(stockCardsMemoryKey(symbols))?.data || null;
    });
    const [stockCardsLoading, setStockCardsLoading] = useState(false);
    const [stockCardsError, setStockCardsError] = useState<string | null>(null);
    const [stockCardPendingSymbols, setStockCardPendingSymbols] = useState<string[]>(() => [...readStoredStockCards()]);
    const [stockCardPickerOpen, setStockCardPickerOpen] = useState(false);
    const [stockSort, setStockSort] = useState<{ key: StockSortKey; direction: SortDirection }>({
        key: 'company',
        direction: 'asc',
    });
    const [indexSort, setIndexSort] = useState<{ key: IndexSortKey; direction: SortDirection }>({
        key: 'symbol',
        direction: 'asc',
    });
    const [indexConstituentSort, setIndexConstituentSort] = useState<{
        key: IndexConstituentSortKey;
        direction: SortDirection;
    }>({
        key: 'impact_abs',
        direction: 'desc',
    });
    const stocksInFlightRef = useRef(false);
    const indicesInFlightRef = useRef(false);
    const indexDetailInFlightRef = useRef(false);
    const stockCardsInFlightRef = useRef(false);
    const marketPageRef = useRef<HTMLDivElement | null>(null);
    const pendingMarketScrollResetRef = useRef(true);
    const marketScrollResetFrameRef = useRef<number | null>(null);
    const latestStockIndexRef = useRef<MarketStockIndex>(stockIndex);
    const latestSelectedIndexRef = useRef<MarketIndexCode | null>(selectedIndex);
    const latestStockCardSymbolsRef = useRef(stockCardSymbols.join(','));
    const previousStockCardSymbolsRef = useRef<string[]>(stockCardSymbols);
    const draggingStockCardSymbolRef = useRef<string | null>(null);
    const [draggingStockCardSymbol, setDraggingStockCardSymbol] = useState<string | null>(null);
    const stockCardSymbolsKey = stockCardSymbols.join(',');
    const watchlist = useWatchlist();
    const mobileWatchlistFundCodes = useMemo(
        () => watchlist.items
            .filter((item) => item.kind === 'fund')
            .map((item) => normalizeWatchlistSymbol(item.symbol)),
        [watchlist.items],
    );

    useEffect(() => {
        const mediaQuery = window.matchMedia('(max-width: 767px)');
        const updateViewport = () => setIsMobileViewport(mediaQuery.matches);
        updateViewport();
        mediaQuery.addEventListener('change', updateViewport);
        return () => mediaQuery.removeEventListener('change', updateViewport);
    }, []);

    useEffect(() => {
        if (activeSection !== 'markets' || !isMobileViewport) return;
        let alive = true;
        const loadMobileIndex = () => {
            apiClient
                .marketIndexDetail('XU100')
                .then((payload) => {
                    if (alive) setMobileIndexDetail(payload);
                })
                .catch(() => {
                    if (alive) setMobileIndexDetail(null);
                });
        };
        loadMobileIndex();
        const intervalId = window.setInterval(loadMobileIndex, 10000);
        return () => {
            alive = false;
            window.clearInterval(intervalId);
        };
    }, [activeSection, isMobileViewport]);

    useEffect(() => {
        if (activeSection !== 'markets' || !isMobileViewport) return;
        const missingCodes = mobileWatchlistFundCodes.filter(
            (code) => !(code in mobileWatchlistFundRows),
        );
        if (missingCodes.length === 0) return;

        let cancelled = false;
        Promise.all(
            missingCodes.map((code) => apiClient
                .fundDetail(code)
                .then((detail) => [code, detail] as const)
                .catch(() => [code, null] as const)),
        ).then((entries) => {
            if (cancelled) return;
            setMobileWatchlistFundRows((current) => {
                const next = { ...current };
                for (const [code, detail] of entries) next[code] = detail;
                return next;
            });
        });

        return () => {
            cancelled = true;
        };
    }, [
        activeSection,
        isMobileViewport,
        mobileWatchlistFundCodes,
        mobileWatchlistFundRows,
    ]);

    useEffect(() => {
        setActiveSection(routeSection);
        if (routeSection !== 'markets') {
            setMobileMarketPanelOpen(false);
        }
    }, [routeSection]);

    useEffect(() => {
        setStockIndex(routeStockIndex);
    }, [routeStockIndex]);

    useEffect(() => {
        setSelectedIndex(routeSelectedIndex);
    }, [routeSelectedIndex]);

    useEffect(() => {
        setReturnMode(routeReturnMode);
    }, [routeReturnMode]);

    useEffect(() => {
        if (activeSection === 'markets') {
            pendingMarketScrollResetRef.current = true;
        }
    }, [activeSection]);

    useLayoutEffect(() => {
        if (activeSection !== 'markets' || loading || error || !market) return;
        if (!pendingMarketScrollResetRef.current) return;
        if (stockCardSymbols.length > 0 && !stockCards && !stockCardsError) return;

        const resetScroll = () => {
            marketPageRef.current?.scrollTo({ top: 0, left: 0, behavior: 'auto' });
            window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
            document.documentElement.scrollTop = 0;
            document.body.scrollTop = 0;
        };

        resetScroll();
        marketScrollResetFrameRef.current = window.requestAnimationFrame(() => {
            resetScroll();
            marketScrollResetFrameRef.current = window.requestAnimationFrame(resetScroll);
        });
        pendingMarketScrollResetRef.current = false;

        return () => {
            if (marketScrollResetFrameRef.current != null) {
                window.cancelAnimationFrame(marketScrollResetFrameRef.current);
                marketScrollResetFrameRef.current = null;
            }
        };
    }, [activeSection, loading, error, market, stockCardSymbols.length, stockCards, stockCardsError]);

    useEffect(() => {
        latestStockIndexRef.current = stockIndex;
    }, [stockIndex]);

    useEffect(() => {
        latestSelectedIndexRef.current = selectedIndex;
    }, [selectedIndex]);

    useEffect(() => {
        latestStockCardSymbolsRef.current = stockCardSymbolsKey;
        try {
            window.localStorage.setItem(STOCK_CARD_STORAGE_KEY, JSON.stringify(stockCardSymbols));
        } catch {
            // localStorage may be unavailable in private or restricted contexts.
        }
    }, [stockCardSymbols, stockCardSymbolsKey]);

    useEffect(() => {
        const previousSymbols = previousStockCardSymbolsRef.current;
        const previousSet = new Set(previousSymbols);
        const selectedSet = new Set(stockCardSymbols);
        const addedSymbols = stockCardSymbols.filter((symbol) => !previousSet.has(symbol));

        setStockCardPendingSymbols((previous) => {
            if (stockCardSymbols.length === 0) return [];
            const next = previous.filter((symbol) => selectedSet.has(symbol));
            const existing = new Set(next);
            for (const symbol of addedSymbols) {
                if (!existing.has(symbol)) {
                    next.push(symbol);
                }
            }
            return next;
        });

        previousStockCardSymbolsRef.current = stockCardSymbols;
    }, [stockCardSymbols, stockCardSymbolsKey]);

    useEffect(() => {
        if (activeSection !== 'markets') return;
        const intervalId = window.setInterval(
            () => setTerminalNow(new Date()),
            isMobileViewport ? 10000 : 1000,
        );
        return () => window.clearInterval(intervalId);
    }, [activeSection, isMobileViewport]);

    useEffect(() => {
        if (activeSection !== 'markets') return;
        loadStats();
        const intervalId = window.setInterval(() => {
            if (document.visibilityState === 'visible') {
                loadStats(true);
            }
        }, 10000);
        return () => window.clearInterval(intervalId);
    }, [activeSection]);

    useEffect(() => {
        if (activeSection !== 'stocks') return;
        setStocks(null);
        setStocksError(null);
        loadStocks(false, false, stockIndex);
        const intervalId = window.setInterval(() => {
            loadStocks(true, false, stockIndex);
        }, LIVE_MARKET_REFRESH_MS);
        return () => window.clearInterval(intervalId);
    }, [activeSection, stockIndex]);

    useEffect(() => {
        if (activeSection !== 'markets' || isMobileViewport) return;
        if (stockCardSymbols.length === 0) {
            setStockCards(null);
            setStockCardsError(null);
            setStockCardsLoading(false);
            setStockCardPendingSymbols([]);
            return;
        }
        const cacheKey = stockCardsMemoryKey(stockCardSymbols);
        const cached = marketStockCardsMemoryCache.get(cacheKey);
        if (cached) {
            setStockCards(cached.data);
            setStockCardPendingSymbols((previous) => previous.filter(
                (symbol) => !(cached.data.items || []).some((item) => item.symbol === symbol),
            ));
        } else {
            setStockCards(null);
        }
        loadStockCards(Boolean(cached), false, stockCardSymbols);
        const intervalId = window.setInterval(() => {
            loadStockCards(true, false, stockCardSymbols);
        }, LIVE_MARKET_REFRESH_MS);
        return () => window.clearInterval(intervalId);
    }, [activeSection, isMobileViewport, stockCardSymbols, stockCardSymbolsKey]);

    useEffect(() => {
        if (activeSection !== 'indices') return;
        loadIndices(false, false);
        const intervalId = window.setInterval(() => {
            loadIndices(true, false);
        }, LIVE_MARKET_REFRESH_MS);
        return () => window.clearInterval(intervalId);
    }, [activeSection]);

    useEffect(() => {
        if (activeSection !== 'indices' || !selectedIndex) return;
        setIndexDetail(null);
        setIndexDetailError(null);
        loadIndexDetail(false, false, selectedIndex);
        const intervalId = window.setInterval(() => {
            loadIndexDetail(true, false, selectedIndex);
        }, LIVE_MARKET_REFRESH_MS);
        return () => window.clearInterval(intervalId);
    }, [activeSection, selectedIndex]);

    async function loadStats(silent = false) {
        const cached = marketUniverseMemoryCache?.data;
        if (cached) setMarket(cached);
        if (!silent) setLoading(!cached);
        if (!silent) setError(null);
        try {
            const marketPayload = await apiClient.marketUniverse();
            marketUniverseMemoryCache = { data: marketPayload, fetchedAt: Date.now() };
            setMarket(marketPayload);
        } catch (err: any) {
            if (!silent) setError(err.message || 'Veriler yüklenemedi.');
        } finally {
            if (!silent) setLoading(false);
        }
    }

    async function loadStocks(silent = false, refresh = false, requestedIndex: MarketStockIndex = stockIndex) {
        if (stocksInFlightRef.current) return;
        stocksInFlightRef.current = true;
        if (!silent) setStocksLoading(true);
        if (!silent) setStocksError(null);
        try {
            const stocksPayload = await apiClient.marketStocks({ index: requestedIndex, refresh });
            if (latestStockIndexRef.current !== requestedIndex) return;
            setStocks(stocksPayload);
            setStocksError(null);
        } catch (err: any) {
            if (!silent || !stocks) {
                setStocksError(err.message || 'Hisse verileri yüklenemedi.');
            }
        } finally {
            stocksInFlightRef.current = false;
            if (!silent) setStocksLoading(false);
        }
    }

    async function loadStockCards(
        silent = false,
        refresh = false,
        requestedSymbols: string[] = stockCardSymbols,
    ) {
        if (stockCardsInFlightRef.current) return;
        if (requestedSymbols.length === 0) return;
        const requestedKey = requestedSymbols.join(',');
        stockCardsInFlightRef.current = true;
        if (!silent) setStockCardsLoading(true);
        if (!silent) setStockCardsError(null);
        try {
            const payload = await apiClient.marketStockCards({ symbols: requestedSymbols, refresh });
            if (latestStockCardSymbolsRef.current !== requestedKey) return;
            marketStockCardsMemoryCache.set(requestedKey, { data: payload, fetchedAt: Date.now() });
            setStockCards(payload);
            setStockCardsError(null);
            const readySymbols = new Set(
                (payload.items ?? [])
                    .filter((item) => hasStockCardLoadedData(item))
                    .map((item) => item.symbol),
            );
            if (readySymbols.size > 0) {
                setStockCardPendingSymbols((previous) => previous.filter((symbol) => !readySymbols.has(symbol)));
            }
        } catch (err: any) {
            if (!silent || !stockCards) {
                setStockCardsError(err.message || 'Hisse kartları yüklenemedi.');
            }
        } finally {
            stockCardsInFlightRef.current = false;
            if (!silent) setStockCardsLoading(false);
        }
    }

    async function loadIndices(silent = false, refresh = false) {
        if (indicesInFlightRef.current) return;
        indicesInFlightRef.current = true;
        if (!silent) setIndicesLoading(true);
        if (!silent) setIndicesError(null);
        try {
            const payload = await apiClient.marketIndices({ refresh });
            setIndices(payload);
            setIndicesError(null);
        } catch (err: any) {
            if (!silent || !indices) {
                setIndicesError(err.message || 'Endeks verileri yüklenemedi.');
            }
        } finally {
            indicesInFlightRef.current = false;
            if (!silent) setIndicesLoading(false);
        }
    }

    async function loadIndexDetail(
        silent = false,
        refresh = false,
        requestedIndex: MarketIndexCode = selectedIndex || 'XUTUM',
    ) {
        if (indexDetailInFlightRef.current) return;
        indexDetailInFlightRef.current = true;
        if (!silent) setIndexDetailLoading(true);
        if (!silent) setIndexDetailError(null);
        try {
            const payload = await apiClient.marketIndexDetail(requestedIndex, { refresh });
            if (latestSelectedIndexRef.current !== requestedIndex) return;
            setIndexDetail(payload);
            setIndexDetailError(null);
        } catch (err: any) {
            if (!silent || !indexDetail) {
                setIndexDetailError(err.message || 'Endeks detayı yüklenemedi.');
            }
        } finally {
            indexDetailInFlightRef.current = false;
            if (!silent) setIndexDetailLoading(false);
        }
    }

    const normalizedSearch = searchTerm.trim().toLowerCase();
    const normalizedStockCardSearch = stockCardSearchTerm.trim().toLowerCase();
    const stockCardCandidates = useMemo(
        () =>
            (market?.rows || [])
                .filter((row) => !stockCardSymbols.includes(row.company))
                .filter((row) => {
                    if (!normalizedStockCardSearch) return true;
                    return row.company.toLowerCase().includes(normalizedStockCardSearch);
                })
                .slice(0, 18),
        [market?.rows, normalizedStockCardSearch, stockCardSymbols],
    );

    const filteredStocks = useMemo(
        () =>
            (stocks?.rows || []).filter((row) =>
                row.company.toLowerCase().includes(normalizedSearch),
            ),
        [stocks?.rows, normalizedSearch],
    );

    const filteredIndices = useMemo(
        () =>
            (indices?.rows || []).filter((row) => {
                const query = normalizedSearch;
                if (!query) return true;
                return (
                    row.symbol.toLowerCase().includes(query) ||
                    row.label.toLowerCase().includes(query)
                );
            }),
        [indices?.rows, normalizedSearch],
    );

    const sortedIndices = useMemo(() => {
        const arr = [...filteredIndices];
        arr.sort((a, b) => {
            const av = a[indexSort.key];
            const bv = b[indexSort.key];
            const aMissing = av == null || av === '';
            const bMissing = bv == null || bv === '';

            if (aMissing && bMissing) return String(a.symbol).localeCompare(String(b.symbol), 'tr');
            if (aMissing) return 1;
            if (bMissing) return -1;

            let result = 0;
            if (typeof av === 'string' || typeof bv === 'string') {
                result = String(av).localeCompare(String(bv), 'tr');
            } else {
                result = Number(av) - Number(bv);
            }
            if (result === 0) result = String(a.symbol).localeCompare(String(b.symbol), 'tr');
            return indexSort.direction === 'asc' ? result : -result;
        });
        return arr;
    }, [filteredIndices, indexSort]);

    const sortedStocks = useMemo(() => {
        const arr = [...filteredStocks];
        arr.sort((a, b) => {
            const av = stockSortValue(a, stockSort.key, stocks?.benchmarks, returnMode);
            const bv = stockSortValue(b, stockSort.key, stocks?.benchmarks, returnMode);
            const aMissing = av == null || av === '';
            const bMissing = bv == null || bv === '';
            if (aMissing && bMissing) return a.company.localeCompare(b.company, 'tr');
            if (aMissing) return 1;
            if (bMissing) return -1;

            let result = 0;
            if (typeof av === 'string' || typeof bv === 'string') {
                result = String(av).localeCompare(String(bv), 'tr');
            } else {
                result = Number(av) - Number(bv);
            }
            if (result === 0) result = a.company.localeCompare(b.company, 'tr');
            return stockSort.direction === 'asc' ? result : -result;
        });
        return arr;
    }, [filteredStocks, stockSort, stocks?.benchmarks, returnMode]);

    const stockCardsBySymbol = useMemo(() => {
        const map = new Map<string, MarketStockCardItem>();
        for (const item of stockCards?.items || []) {
            map.set(item.symbol, item);
        }
        return map;
    }, [stockCards?.items]);
    const stockCardPendingSet = useMemo(() => new Set(stockCardPendingSymbols), [stockCardPendingSymbols]);
    const canAddStockCards = stockCardSymbols.length < MAX_STOCK_CARDS;
    const activeBenchmarkIndex = getBenchmarkIndex(returnMode);
    const activeReturnModeLabel = RETURN_MODE_OPTIONS.find((option) => option.id === returnMode)?.label || 'Mutlak';
    const indexConstituents = useMemo(() => indexDetail?.constituents ?? [], [indexDetail?.constituents]);
    const indexImpactLevel = numericOrNull(indexDetail?.price) ?? numericOrNull(indexDetail?.prev_close);
    const positiveConstituents = indexConstituents.filter((row) => (row.point_effect || 0) > 0).length;
    const negativeConstituents = indexConstituents.filter((row) => (row.point_effect || 0) < 0).length;
    const neutralConstituents = indexConstituents.length - positiveConstituents - negativeConstituents;
    const sortedIndexConstituents = useMemo(() => {
        const arr = [...indexConstituents];
        arr.sort((a, b) => {
            const av = constituentSortValue(a, indexConstituentSort.key, indexImpactLevel);
            const bv = constituentSortValue(b, indexConstituentSort.key, indexImpactLevel);
            const aMissing = av == null || av === '';
            const bMissing = bv == null || bv === '';
            if (aMissing && bMissing) return a.symbol.localeCompare(b.symbol, 'tr');
            if (aMissing) return 1;
            if (bMissing) return -1;

            let result = 0;
            if (typeof av === 'string' || typeof bv === 'string') {
                result = String(av).localeCompare(String(bv), 'tr');
            } else {
                result = Number(av) - Number(bv);
            }
            if (result === 0) result = a.symbol.localeCompare(b.symbol, 'tr');
            return indexConstituentSort.direction === 'asc' ? result : -result;
        });
        return arr;
    }, [indexConstituents, indexConstituentSort, indexImpactLevel]);
    const heatmapConstituents = useMemo(
        () =>
            indexConstituents
                .filter((row) => row.weight_pct != null && row.weight_pct > 0)
                .sort((a, b) => treemapValue(b) - treemapValue(a)),
        [indexConstituents],
    );
    const maxHeatmapMovePct = Math.max(
        0.01,
        ...heatmapConstituents.map((row) => Math.abs(row.change_pct ?? 0)),
    );
    const heatmapTiles = useMemo(
        () => buildHeatmapTiles(heatmapConstituents, indexImpactLevel),
        [heatmapConstituents, indexImpactLevel],
    );
    const hasPointEffects = useMemo(
        () => indexConstituents.some((row) => row.point_effect != null),
        [indexConstituents],
    );
    const netPointEffect = useMemo<number | null>(
        () => {
            if (!hasPointEffects) return null;
            return indexConstituents.reduce((total, row) => {
                if (row.point_effect == null || !Number.isFinite(row.point_effect)) return total;
                return total + row.point_effect;
            }, 0);
        },
        [indexConstituents, hasPointEffects],
    );
    const netImpactPct = netPointEffect != null && indexImpactLevel && indexImpactLevel > 0 ? (netPointEffect / indexImpactLevel) * 100 : null;
    const pageTitle =
        activeSection === 'indices'
            ? 'Borsa İstanbul Endeksleri'
            : activeSection === 'stocks'
              ? 'BIST Hisseleri'
              : 'Piyasa Görünümü';
    const pageDescription =
        activeSection === 'indices'
            ? 'BIST ana ve sektör endekslerini, getirileri ve endeks içi şirket hareketlerini takip edin.'
            : activeSection === 'stocks'
              ? 'XUTUM, XU100 ve XU030 hisselerini fiyat, hacim ve piyasa değeriyle karşılaştırın.'
              : 'Güncel fiyatlar, finansal görünüm ve analiz erişimi tek ekranda.';
    const documentTitle = useMemo(() => {
        if (activeSection === 'indices' && selectedIndex) {
            const indexQuoteTitle = [
                formatTitleNumber(indexDetail?.price),
                formatTitlePct(indexDetail?.change_pct),
            ].filter(Boolean).join(' ');
            return buildDocumentTitle(
                indexDetail?.symbol || selectedIndex,
                indexQuoteTitle,
                indexDetail?.label || 'Endeks Detayı',
            );
        }
        if (activeSection === 'indices') {
            const indexCount = sortedIndices.length || indices?.rows.length;
            return buildDocumentTitle('Endeksler', indexCount ? `${indexCount} endeks` : null);
        }
        if (activeSection === 'stocks') {
            const stockCount = sortedStocks.length || stocks?.rows.length;
            return buildDocumentTitle('BIST Hisseleri', stocks?.index || stockIndex, stockCount ? `${stockCount} hisse` : null);
        }
        return buildDocumentTitle('Piyasa Özeti', market?.rows.length ? `${market.rows.length} hisse` : null);
    }, [
        activeSection,
        indexDetail?.change_pct,
        indexDetail?.label,
        indexDetail?.price,
        indexDetail?.symbol,
        indices?.rows.length,
        market?.rows.length,
        selectedIndex,
        sortedIndices.length,
        sortedStocks.length,
        stockIndex,
        stocks?.index,
        stocks?.rows.length,
    ]);
    useDocumentTitle(documentTitle);

    const onCompanyClick = (ticker: string) => {
        const normalizedTicker = String(ticker || '').trim().toUpperCase();
        if (!normalizedTicker) return;
        if (onOpenTicker) {
            onOpenTicker(normalizedTicker);
            return;
        }
        window.location.href = `/?ticker=${normalizedTicker}`;
    };

    const handleStockIndexChange = (index: MarketStockIndex) => {
        setStockIndex(index);
        onNavigateStockIndex?.(index);
    };

    const handleReturnModeChange = (mode: StockReturnMode) => {
        if (mode === returnMode) return;
        setReturnMode(mode);
        onNavigateReturnMode?.(mode);
    };

    const handleSelectIndex = (index: MarketIndexCode | null) => {
        setSelectedIndex(index);
        onNavigateIndexDetail?.(index);
    };

    const handleAddStockCard = (symbol: string) => {
        const normalized = normalizeStockCardSymbol(symbol);
        setStockCardSymbols((prev) => {
            if (prev.includes(normalized) || prev.length >= MAX_STOCK_CARDS) return prev;
            return [...prev, normalized];
        });
        setStockCardSearchTerm('');
        setStockCardPickerOpen(false);
    };

    const handleStockCardMoveStart = (symbol: string) => {
        draggingStockCardSymbolRef.current = symbol;
        setDraggingStockCardSymbol(symbol);
    };

    const moveStockCard = (
        sourceSymbol: string,
        targetSymbol: string,
        placement: StockCardDropPlacement,
    ) => {
        if (sourceSymbol === targetSymbol) return;

        setStockCardSymbols((prev) => {
            if (!prev.includes(sourceSymbol) || !prev.includes(targetSymbol)) return prev;
            const next = prev.filter((symbol) => symbol !== sourceSymbol);
            const targetIndex = next.indexOf(targetSymbol);
            if (targetIndex < 0) return prev;
            next.splice(targetIndex + (placement === 'after' ? 1 : 0), 0, sourceSymbol);
            const didChange = next.some((symbol, index) => symbol !== prev[index]);
            return didChange ? next : prev;
        });
    };

    const handleStockCardMoveOver = (targetSymbol: string, placement: StockCardDropPlacement) => {
        const sourceSymbol = draggingStockCardSymbolRef.current;
        if (!sourceSymbol || sourceSymbol === targetSymbol) return;
        moveStockCard(sourceSymbol, targetSymbol, placement);
    };

    const handleStockCardMoveToEnd = () => {
        const sourceSymbol = draggingStockCardSymbolRef.current;
        if (!sourceSymbol) return;
        setStockCardSymbols((prev) => {
            if (!prev.includes(sourceSymbol) || prev[prev.length - 1] === sourceSymbol) return prev;
            const next = prev.filter((symbol) => symbol !== sourceSymbol);
            next.push(sourceSymbol);
            return next;
        });
    };

    const handleStockCardGridDragOver = (event: React.DragEvent<HTMLDivElement>) => {
        const sourceSymbol = draggingStockCardSymbolRef.current;
        if (!sourceSymbol) return;

        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';

        const cardElements = Array.from(
            event.currentTarget.querySelectorAll<HTMLElement>('.stock-card:not(.stock-card-dragging)'),
        );
        let closestDrop: {
            symbol: string;
            placement: StockCardDropPlacement;
            distance: number;
        } | null = null;

        for (const cardElement of cardElements) {
            const targetSymbol = cardElement.dataset.stockCardSymbol;
            if (!targetSymbol || targetSymbol === sourceSymbol) continue;

            const rect = cardElement.getBoundingClientRect();
            const outsideX = event.clientX < rect.left
                ? rect.left - event.clientX
                : event.clientX > rect.right
                  ? event.clientX - rect.right
                  : 0;
            const outsideY = event.clientY < rect.top
                ? rect.top - event.clientY
                : event.clientY > rect.bottom
                  ? event.clientY - rect.bottom
                  : 0;
            const placement: StockCardDropPlacement =
                event.clientX >= rect.left + rect.width / 2 ? 'after' : 'before';
            const distance = outsideY * 1000 + outsideX;

            if (!closestDrop || distance < closestDrop.distance) {
                closestDrop = { symbol: targetSymbol, placement, distance };
            }
        }

        if (closestDrop) {
            moveStockCard(sourceSymbol, closestDrop.symbol, closestDrop.placement);
        } else {
            handleStockCardMoveToEnd();
        }
    };

    const handleStockCardMoveEnd = () => {
        draggingStockCardSymbolRef.current = null;
        setDraggingStockCardSymbol(null);
    };

    const handleRemoveStockCard = (symbol: string) => {
        if (draggingStockCardSymbolRef.current === symbol) {
            handleStockCardMoveEnd();
        }
        setStockCardSymbols((prev) => prev.filter((item) => item !== symbol));
    };

    const handleSectionChange = (section: MarketSection) => {
        if (onNavigateSection) {
            onNavigateSection(section);
            return;
        }
        setActiveSection(section);
        setSelectedIndex(null);
    };

    const handleStockSort = (key: StockSortKey) => {
        setStockSort((prev) => {
            if (prev.key === key) {
                return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
            }
            return { key, direction: key === 'company' ? 'asc' : 'desc' };
        });
    };

    const handleIndexSort = (key: IndexSortKey) => {
        setIndexSort((prev) => {
            if (prev.key === key) {
                return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
            }
            return { key, direction: key === 'symbol' ? 'asc' : 'desc' };
        });
    };

    const handleIndexConstituentSort = (key: IndexConstituentSortKey) => {
        setIndexConstituentSort((prev) => {
            if (prev.key === key) {
                return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
            }
            return { key, direction: key === 'symbol' ? 'asc' : 'desc' };
        });
    };

    const getTreemapTileScale = (row: MarketIndexConstituent): number => {
        const movePct = Math.abs(row.change_pct ?? 0);
        return Math.min(1, Math.sqrt(movePct / maxHeatmapMovePct));
    };

    const getHeatmapTileFontSize = (tile: HeatmapTile): string => {
        if (tile.areaPct >= 16 && tile.width >= 22 && tile.height >= 22) return '2rem';
        if (tile.areaPct >= 8) return '1.45rem';
        if (tile.areaPct >= 3.5) return '1rem';
        if (tile.areaPct >= 1.2) return '0.78rem';
        return '0.64rem';
    };

    const renderReturnCell = (row: MarketStockRow, key: StockReturnKey) => {
        const value = getReturnValue(row, key, stocks?.benchmarks, returnMode);
        return (
            <td className={`stocks-cell-right ${getTableChangeClass(value)}`}>
                {formatTablePct(value)}
            </td>
        );
    };

    return (
        <div className={`mn-layout${navCollapsed ? ' mn-nav-collapsed' : ''}`}>
            <MarketsNavigation
                collapsed={navCollapsed}
                activeSection={activeSection}
                onCollapsedChange={setNavCollapsed}
                onSectionChange={handleSectionChange}
                onFundSectionChange={onNavigateFundSection}
                onSelectTicker={onCompanyClick}
                onSelectFund={onOpenFund}
            />
            <div className={`markets-workspace${activeSection === 'markets' ? ' markets-workspace-with-rail' : ''}`}>
                <div className="market-page" ref={marketPageRef}>
                {market && activeSection !== 'markets' && (
                    <MarketSidebar rows={market.rows} onSelectTicker={onCompanyClick} />
                )}
            {activeSection !== 'markets' && (
                <header className="market-header">
                    <div className="market-title">
                        <h1>{pageTitle}</h1>
                        <p>{pageDescription}</p>
                    </div>
                </header>
            )}

            {activeSection === 'markets' && (
                <div className="market-terminal-shell">
                    <div className="market-overview-head">
                        <div className="market-title">
                            <h1>Piyasa Görünümü</h1>
                            <p>Güncel fiyatlar, finansal görünüm ve seçtiğiniz hisse kartları tek ekranda.</p>
                        </div>
                    </div>
                    <div className="market-session-bar">
                        <div className="market-session-date">
                            <CalendarDays size={18} aria-hidden="true" />
                            <strong>Bugün</strong>
                            <span>{formatTerminalDate(terminalNow)}</span>
                            <time>{formatTerminalClock(terminalNow)}</time>
                            <span className="market-session-dot" aria-label="Piyasa izleme aktif" />
                        </div>
                    </div>
                    <MarketWatchStrip variant="compact" />
                </div>
            )}

            {activeSection === 'markets' && loading && <div className="loading-state"><div className="spinner"/> Yükleniyor...</div>}
            
            {activeSection === 'markets' && error && (
                <div className="error-message">
                    <strong>Hata:</strong> {error}
                    <button onClick={() => loadStats()} className="btn-secondary" style={{ marginLeft: '1rem' }}>Tekrar Dene</button>
                </div>
            )}

            {activeSection === 'markets' && !loading && !error && market && (
                <div className="market-content market-content-terminal">
                    <div className="market-main-column">
                        <MobileMarketOverview
                            index={mobileIndexDetail}
                            rows={market.rows}
                            watchlistItems={watchlist.items}
                            fundRows={mobileWatchlistFundRows}
                            onSelectTicker={onCompanyClick}
                            onSelectIndex={handleSelectIndex}
                            onOpenFund={onOpenFund}
                            onAddStock={(symbol) => watchlist.addItem({ kind: 'stock', symbol })}
                            onRemoveWatchlistItem={(item) => watchlist.removeItem(item.kind, item.symbol)}
                        />
                        {!isMobileViewport && (
                        <section className="panel stock-cards-panel">
                            <div className="stock-cards-toolbar">
                                <div className="stock-cards-title-row">
                                    <h2>Hisse Kartları</h2>
                                    <span className="panel-kicker">
                                        {stockCardSymbols.length}/{MAX_STOCK_CARDS} kart
                                    </span>
                                    <button
                                        type="button"
                                        className="stock-card-add-link"
                                        onClick={() => setStockCardPickerOpen((open) => !open)}
                                        disabled={!canAddStockCards}
                                    >
                                        <span>Ekle</span>
                                        <Plus size={16} aria-hidden="true" />
                                    </button>
                                </div>
                            </div>

                            {stockCardPickerOpen && (
                                <div className="stock-card-picker">
                                    <div className="stock-card-picker-search">
                                        <Search size={16} aria-hidden="true" />
                                        <input
                                            type="text"
                                            placeholder="Hisse ara..."
                                            value={stockCardSearchTerm}
                                            onChange={(event) => setStockCardSearchTerm(event.target.value)}
                                            autoFocus
                                        />
                                        <button
                                            type="button"
                                            className="stock-card-picker-close"
                                            onClick={() => {
                                                setStockCardPickerOpen(false);
                                                setStockCardSearchTerm('');
                                            }}
                                            aria-label="Hisse seçiciyi kapat"
                                            title="Kapat"
                                        >
                                            <X size={16} aria-hidden="true" />
                                        </button>
                                    </div>
                                    <div className="stock-card-picker-heading">Hisse Senetleri</div>
                                    <div className="stock-card-picker-list">
                                        {stockCardCandidates.length > 0 ? (
                                            stockCardCandidates.map((row) => (
                                                <button
                                                    key={row.company}
                                                    type="button"
                                                    onClick={() => handleAddStockCard(row.company)}
                                                >
                                                    <SymbolLogo
                                                        symbol={row.company}
                                                        name={row.company}
                                                        kind="stock"
                                                        logoUrl={row.logo_url}
                                                        size="sm"
                                                        className="stock-card-picker-logo"
                                                    />
                                                    <span className="stock-card-picker-copy">
                                                        <strong>{row.company}</strong>
                                                    </span>
                                                </button>
                                            ))
                                        ) : (
                                            <div className="stock-card-picker-empty">
                                                Eklenebilecek hisse bulunamadı.
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {stockCardsError && (
                                <div className="stock-card-error">
                                    <span>{stockCardsError}</span>
                                    <button
                                        type="button"
                                        onClick={() => loadStockCards(false, true, stockCardSymbols)}
                                    >
                                        Tekrar dene
                                    </button>
                                </div>
                            )}

                            {stockCardSymbols.length === 0 ? (
                                <div className="stock-card-empty-state">
                                    <h3>Henüz hisse kartı yok</h3>
                                    <p>Takip etmek istediğiniz hisseleri ekleyin; fiyat, gün içi grafik ve temel piyasa verileri burada görünsün.</p>
                                    <button
                                        type="button"
                                        className="stock-card-empty-action"
                                        onClick={() => setStockCardPickerOpen(true)}
                                    >
                                        <Plus size={16} aria-hidden="true" />
                                        Hisse kartı ekle
                                    </button>
                                </div>
                            ) : !stockCards && !stockCardsError ? (
                                <div className="stock-card-skeleton-grid">
                                    {stockCardSymbols.map((symbol) => (
                                        <div key={symbol} className="stock-card-skeleton">
                                            <div className="stock-card-skeleton-head">
                                                <div className="stock-card-skeleton-logo stock-card-skeleton-pulse" />
                                                <div className="stock-card-skeleton-id">
                                                    <div className="stock-card-skeleton-symbol stock-card-skeleton-pulse" />
                                                    <div className="stock-card-skeleton-company stock-card-skeleton-pulse" />
                                                </div>
                                            </div>
                                            <div className="stock-card-skeleton-price-row">
                                                <div className="stock-card-skeleton-price stock-card-skeleton-pulse" />
                                                <div className="stock-card-skeleton-change stock-card-skeleton-pulse" />
                                            </div>
                                            <div className="stock-card-skeleton-metrics">
                                                <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                                <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                                <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                            </div>
                                            <div className="stock-card-skeleton-chart stock-card-skeleton-pulse" />
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div
                                    className="stock-card-grid"
                                    onDragOver={handleStockCardGridDragOver}
                                    onDrop={(event) => {
                                        event.preventDefault();
                                        handleStockCardMoveEnd();
                                    }}
                                >
                                    {stockCardSymbols.map((symbol) => {
                                        const item = stockCardsBySymbol.get(symbol) || emptyStockCardItem(symbol);
                                        const hasLoadedItem = stockCardsBySymbol.has(symbol);
                                        const isCardLoading =
                                            !hasLoadedItem
                                            && (stockCardPendingSet.has(symbol) || stockCardsLoading);
                                        return (
                                            <MarketStockCard
                                                key={symbol}
                                                item={item}
                                                onOpen={() => onCompanyClick(symbol)}
                                                onRemove={() => handleRemoveStockCard(symbol)}
                                                onMoveStart={() => handleStockCardMoveStart(symbol)}
                                                onMoveOver={(placement) => handleStockCardMoveOver(symbol, placement)}
                                                onMoveEnd={handleStockCardMoveEnd}
                                                isLoading={isCardLoading}
                                                isDragging={draggingStockCardSymbol === symbol}
                                            />
                                        );
                                    })}
                                </div>
                            )}
                        </section>
                        )}
                    </div>
                </div>
            )}

            {activeSection === 'stocks' && (
                <div className="stocks-view">
                    <div className="panel stocks-table-panel">
                        <div className="panel-header stocks-panel-header">
                            <div>
                                <h2>Hisseler</h2>
                                <span className="panel-kicker">
                                    {stocks?.index || stockIndex} ·{' '}
                                    {sortedStocks.length || stocks?.rows.length || 0} hisse
                                </span>
                            </div>
                            <div className="stocks-panel-actions">
                                <div
                                    className="stocks-segment"
                                    aria-label="Endeks seçimi"
                                    style={{
                                        '--item-count': STOCK_INDEX_OPTIONS.length,
                                        '--active-index': STOCK_INDEX_OPTIONS.indexOf(stockIndex)
                                    } as React.CSSProperties}
                                >
                                    {STOCK_INDEX_OPTIONS.map((option) => (
                                        <button
                                            key={option}
                                            type="button"
                                            className={stockIndex === option ? 'active' : ''}
                                            aria-pressed={stockIndex === option}
                                            onClick={() => handleStockIndexChange(option)}
                                        >
                                            {option}
                                        </button>
                                    ))}
                                </div>
                                <div
                                    className="stocks-segment stocks-return-segment"
                                    aria-label="Getiri modu"
                                    style={{
                                        '--item-count': RETURN_MODE_OPTIONS.length,
                                        '--active-index': RETURN_MODE_OPTIONS.findIndex(o => o.id === returnMode)
                                    } as React.CSSProperties}
                                >
                                    {RETURN_MODE_OPTIONS.map((option) => (
                                        <button
                                            key={option.id}
                                            type="button"
                                            className={returnMode === option.id ? 'active' : ''}
                                            aria-pressed={returnMode === option.id}
                                            onClick={() => handleReturnModeChange(option.id)}
                                        >
                                            {option.label}
                                        </button>
                                    ))}
                                </div>
                                <div className="search-box">
                                    <input
                                        type="text"
                                        placeholder="Hisse kodu ile ara..."
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                        className="input-field"
                                    />
                                </div>
                            </div>
                        </div>

                        {stocksLoading && !stocks && (
                            <div className="loading-state"><div className="spinner" /> Hisseler yükleniyor...</div>
                        )}

                        {stocksError && !stocks && (
                            <div className="error-message">
                                <strong>Hata:</strong> {stocksError}
                                <button
                                    onClick={() => loadStocks(false, true, stockIndex)}
                                    className="btn-secondary"
                                    style={{ marginLeft: '1rem' }}
                                >
                                    Tekrar Dene
                                </button>
                            </div>
                        )}

                        {stocks && (
                            <>
                                <div className="stocks-table-meta">
                                    <span>Son güncelleme: {formatUpdateTime(stocks.as_of)}</span>
                                    <span>Getiri: {activeReturnModeLabel}</span>
                                    {activeBenchmarkIndex && (
                                        <span>
                                            Benchmark: {activeBenchmarkIndex} ·{' '}
                                            {formatUpdateTime(stocks.benchmarks?.[activeBenchmarkIndex]?.as_of)}
                                        </span>
                                    )}
                                    {stocksError && <span className="stocks-soft-error">Son yenileme başarısız oldu.</span>}
                                </div>

                                {sortedStocks.length === 0 ? (
                                    <div className="no-results">Aranan kritere uyan hisse bulunamadı.</div>
                                ) : (
                                    <div className="stocks-table-wrap">
                                        <table className="stocks-table">
                                            <thead>
                                                <tr>
                                                    <th className="stocks-rank">#</th>
                                                    {STOCK_COLUMNS.map((column) => (
                                                        <th
                                                            key={column.key}
                                                            className={column.align === 'right' ? 'stocks-cell-right' : undefined}
                                                        >
                                                            <button
                                                                type="button"
                                                                className="stocks-sort-button"
                                                                onClick={() => handleStockSort(column.key)}
                                                                aria-sort={
                                                                    stockSort.key === column.key
                                                                        ? stockSort.direction === 'asc'
                                                                            ? 'ascending'
                                                                            : 'descending'
                                                                        : 'none'
                                                                }
                                                            >
                                                                <span>{getColumnLabel(column.key, returnMode)}</span>
                                                                {column.sublabel && <small>{column.sublabel}</small>}
                                                                {stockSort.key === column.key && (
                                                                    <span className="stocks-sort-indicator">
                                                                        {stockSort.direction === 'asc' ? '↑' : '↓'}
                                                                    </span>
                                                                )}
                                                            </button>
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {sortedStocks.map((row, index) => (
                                                    <FlashStockRow
                                                        key={row.company}
                                                        row={row}
                                                        rank={index + 1}
                                                        onClick={() => onCompanyClick(row.company)}
                                                    >
                                                        <td className="stocks-symbol-cell">
                                                            <span className="stocks-symbol-main">
                                                                <SymbolLogo
                                                                    symbol={row.company}
                                                                    name={row.company}
                                                                    kind="stock"
                                                                    logoUrl={row.logo_url}
                                                                    size="xs"
                                                                    className="stocks-symbol-logo"
                                                                />
                                                                <span>{row.company}</span>
                                                            </span>
                                                        </td>
                                                        <td className="stocks-cell-right stocks-price-cell">{formatStockPrice(row)}</td>
                                                        <td className={`stocks-cell-right ${getTableChangeClass(row.change_pct)}`}>
                                                            {formatTablePct(row.change_pct)}
                                                        </td>
                                                        <td className="stocks-cell-right stocks-volume-cell">{formatVolume(row.volume)}</td>
                                                        <td className="stocks-cell-right stocks-market-cap-cell">
                                                            {formatMarketCap(row.market_cap)}
                                                        </td>
                                                        {renderReturnCell(row, 'return_1w_pct')}
                                                        {renderReturnCell(row, 'return_1m_pct')}
                                                        {renderReturnCell(row, 'return_3m_pct')}
                                                        {renderReturnCell(row, 'return_6m_pct')}
                                                        {renderReturnCell(row, 'return_ytd_pct')}
                                                        {renderReturnCell(row, 'return_1y_pct')}
                                                    </FlashStockRow>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>
            )}

            {activeSection === 'indices' && !selectedIndex && (
                <div className="indices-view">
                    <div className="panel stocks-table-panel">
                        <div className="panel-header stocks-panel-header">
                            <div>
                                <h2>Endeksler</h2>
                                <span className="panel-kicker">
                                    {filteredIndices.length || indices?.rows.length || 0} endeks
                                </span>
                            </div>
                            <div className="stocks-panel-actions">
                                <div className="search-box">
                                    <input
                                        type="text"
                                        placeholder="Endeks kodu ile ara..."
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                        className="input-field"
                                    />
                                </div>
                            </div>
                        </div>

                        {indicesLoading && !indices && (
                            <div className="loading-state"><div className="spinner" /> Endeksler yükleniyor...</div>
                        )}

                        {indicesError && !indices && (
                            <div className="error-message">
                                <strong>Hata:</strong> {indicesError}
                                <button
                                    onClick={() => loadIndices(false, true)}
                                    className="btn-secondary"
                                    style={{ marginLeft: '1rem' }}
                                >
                                    Tekrar Dene
                                </button>
                            </div>
                        )}

                        {indices && (
                            <>
                                <div className="stocks-table-meta">
                                    <span>Son güncelleme: {formatUpdateTime(indices.as_of)}</span>
                                    {indicesError && <span className="stocks-soft-error">Son yenileme başarısız oldu.</span>}
                                </div>
                                {filteredIndices.length === 0 ? (
                                    <div className="no-results">Aranan kritere uyan endeks bulunamadı.</div>
                                ) : (
                                    <div className="stocks-table-wrap indices-table-wrap">
                                        <table className="stocks-table indices-table">
                                            <thead>
                                                <tr>
                                                    <th className="stocks-rank">#</th>
                                                    {INDEX_COLUMNS.map((column) => (
                                                        <th
                                                            key={column.key}
                                                            className={column.align === 'right' ? 'stocks-cell-right' : undefined}
                                                        >
                                                            <button
                                                                type="button"
                                                                className="stocks-sort-button"
                                                                onClick={() => handleIndexSort(column.key)}
                                                                aria-sort={
                                                                    indexSort.key === column.key
                                                                        ? indexSort.direction === 'asc'
                                                                            ? 'ascending'
                                                                            : 'descending'
                                                                        : 'none'
                                                                }
                                                            >
                                                                <span>{column.label}</span>
                                                                {indexSort.key === column.key && (
                                                                    <span className="stocks-sort-indicator">
                                                                        {indexSort.direction === 'asc' ? '↑' : '↓'}
                                                                    </span>
                                                                )}
                                                            </button>
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {sortedIndices.map((row, index) => (
                                                    <tr
                                                        key={row.symbol}
                                                        onClick={() => handleSelectIndex(row.symbol)}
                                                    >
                                                        <td className="stocks-rank">{index + 1}</td>
                                                        {INDEX_COLUMNS.map((column) => {
                                                            const value = numericOrNull(row[column.key]);
                                                            const isPct =
                                                                column.key === 'change_pct' ||
                                                                String(column.key).startsWith('return_');
                                                            return (
                                                                <td
                                                                    key={column.key}
                                                                    className={[
                                                                        column.align === 'right' ? 'stocks-cell-right' : '',
                                                                        isPct ? getTableChangeClass(value) : '',
                                                                        column.key === 'symbol' ? 'stocks-symbol-cell' : '',
                                                                        column.key === 'price' ? 'stocks-price-cell' : '',
                                                                    ].join(' ')}
                                                                >
                                                                    {column.key === 'symbol' ? (
                                                                        <span className="indices-symbol-main">
                                                                            <span className="indices-symbol-head">
                                                                                <SymbolLogo
                                                                                    symbol={row.symbol}
                                                                                    name={row.label}
                                                                                    kind="index"
                                                                                    size="xs"
                                                                                    className="indices-symbol-logo"
                                                                                />
                                                                                <span>{row.symbol}</span>
                                                                            </span>
                                                                            <small>{row.label}</small>
                                                                        </span>
                                                                    ) : (
                                                                        indexCellValue(row, column.key)
                                                                    )}
                                                                </td>
                                                            );
                                                        })}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>
            )}

            {activeSection === 'indices' && selectedIndex && (
                <div className="indices-detail-view">
                    <div className="indices-breadcrumb">
                        <button type="button" onClick={() => handleSelectIndex(null)}>Endeksler</button>
                        <span>/</span>
                        <strong>{selectedIndex}</strong>
                    </div>

                    {indexDetailLoading && !indexDetail && (
                        <div className="loading-state"><div className="spinner" /> Endeks detayı yükleniyor...</div>
                    )}

                    {indexDetailError && !indexDetail && (
                        <div className="error-message">
                            <strong>Hata:</strong> {indexDetailError}
                            <button
                                onClick={() => loadIndexDetail(false, true, selectedIndex)}
                                className="btn-secondary"
                                style={{ marginLeft: '1rem' }}
                            >
                                Tekrar Dene
                            </button>
                        </div>
                    )}

                    {indexDetail && (
                        <>
                            <section className="indices-hero">
                                <div className="indices-hero-main">
                                    <SymbolLogo
                                        symbol={indexDetail.symbol}
                                        name={indexDetail.label}
                                        kind="index"
                                        size="lg"
                                        className="indices-logo"
                                    />
                                    <div>
                                        <h2>{indexDetail.symbol}</h2>
                                        <p>{indexDetail.label}</p>
                                    </div>
                                </div>
                                <div className="indices-hero-price">
                                    <strong>{formatIndexPrice(indexDetail.price)}</strong>
                                    <span className={getTableChangeClass(indexDetail.change_pct)}>
                                        {formatTablePct(indexDetail.change_pct)}
                                    </span>
                                    <small>{formatDateTime(indexDetail.as_of)}</small>
                                </div>
                            </section>

                            <div className="indices-stat-row">
                                <span>Yüksek: <strong>{formatIndexPrice(indexDetail.high)}</strong></span>
                                <span>Düşük: <strong>{formatIndexPrice(indexDetail.low)}</strong></span>
                                <span>Hacim: <strong>{formatVolume(indexDetail.volume)}</strong></span>
                                <span>Önc.Kap.: <strong>{formatIndexPrice(indexDetail.prev_close)}</strong></span>
                            </div>

                            <div className="indices-return-strip">
                                {DETAIL_RETURN_KEYS.map((item) => {
                                    const value = numericOrNull(indexDetail[item.key]);
                                    return (
                                        <div key={item.key} className="indices-return-card">
                                            <span>{item.label}</span>
                                            <strong className={getTableChangeClass(value)}>{formatTablePct(value)}</strong>
                                        </div>
                                    );
                                })}
                            </div>

                            <section className="indices-chart-panel">
                                <IndexLineChart
                                    symbol={indexDetail.symbol}
                                    points={indexDetail.line_points}
                                    prevClose={indexDetail.prev_close}
                                    changePct={indexDetail.change_pct}
                                />
                            </section>

                            <section className="indices-impact-panel">
                                <div className="indices-impact-head">
                                    <div>
                                        <h3>{indexDetail.symbol} Endeksindeki Şirketler ve Etkileri</h3>
                                        <span className="panel-kicker">
                                            {indexDetail.constituents.length} şirket · {indexDetail.weight_status === 'available' ? 'ağırlık hesaplandı' : 'ağırlık bekleniyor'}
                                        </span>
                                    </div>
                                    {indexDetailError && <span className="stocks-soft-error">Son yenileme başarısız oldu.</span>}
                                </div>

                                {indexDetail.weight_note && (
                                    <div className="indices-weight-note">{indexDetail.weight_note}</div>
                                )}

                                {indexDetail.weight_status === 'available' && (
                                    <>
                                        <div className="indices-treemap-scale" aria-hidden="true">
                                            <div className="indices-treemap-scale-labels">
                                                <span>{formatTreemapScalePct(-maxHeatmapMovePct)}</span>
                                                <span>0%</span>
                                                <span>{formatTreemapScalePct(maxHeatmapMovePct)}</span>
                                            </div>
                                            <div className="indices-treemap-scale-bar" />
                                        </div>
                                        <div className="indices-treemap">
                                            {heatmapTiles.map((tile) => {
                                                const row = tile.row;
                                                const impactPct = tile.impactPct;
                                                const scaledMove = getTreemapTileScale(row);
                                                const showChange = tile.areaPct >= 1.05 && tile.width >= 5 && tile.height >= 4.5;
                                                const showSymbol = tile.areaPct >= 0.45 && tile.width >= 3.5 && tile.height >= 3;
                                                const tileDetail = `${row.symbol} · Değişim ${formatHeatmapChangePct(tile.changePct)} · Ağırlık ${formatWeight(row.weight_pct)} · Endeks etkisi ${formatImpactPct(impactPct)} · ${formatPointEffectShort(tile.pointEffect)}`;
                                                const tileDensityClass = tile.areaPct >= 14
                                                    ? 'is-hero'
                                                    : showChange
                                                    ? 'is-full'
                                                    : showSymbol
                                                        ? 'is-label-only'
                                                        : 'is-tiny';
                                                return (
                                                    <div
                                                        key={row.symbol}
                                                        className={`indices-tree-tile ${tileDensityClass} ${getTableChangeClass(tile.changePct)}`}
                                                        style={{
                                                            left: `${tile.left}%`,
                                                            top: `${tile.top}%`,
                                                            width: `${tile.width}%`,
                                                            height: `${tile.height}%`,
                                                            backgroundColor: getTreemapTileColor(tile.changePct, scaledMove),
                                                            fontSize: getHeatmapTileFontSize(tile),
                                                        }}
                                                        title={tileDetail}
                                                        aria-label={tileDetail}
                                                    >
                                                        {showSymbol && <strong>{row.symbol}</strong>}
                                                        {showChange && (
                                                            <span className="indices-tree-impact-pct">{formatHeatmapChangePct(tile.changePct)}</span>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </>
                                )}

                                <div className="indices-impact-summary">
                                    <div className="indices-impact-counts">
                                        <span className="indices-impact-count stocks-up">
                                            <span className="indices-impact-dot" />
                                            {positiveConstituents} pozitif
                                        </span>
                                        <span className="indices-impact-count stocks-flat">
                                            <span className="indices-impact-dot" />
                                            {neutralConstituents} nötr
                                        </span>
                                        <span className="indices-impact-count stocks-down">
                                            <span className="indices-impact-dot" />
                                            {negativeConstituents} negatif
                                        </span>
                                    </div>
                                    <div className="indices-net-impact">
                                        <span>{indexDetail.symbol} tahmini katkı:</span>
                                        <strong className={getTableChangeClass(netPointEffect)}>
                                            {formatPointEffectShort(netPointEffect)}
                                        </strong>
                                        <span>·</span>
                                        <strong className={getTableChangeClass(netImpactPct)}>
                                            {formatImpactPct(netImpactPct)}
                                        </strong>
                                    </div>
                                </div>

                                <div className="stocks-table-wrap indices-constituent-wrap">
                                    <table className="stocks-table indices-constituent-table">
                                        <thead>
                                            <tr>
                                                {INDEX_CONSTITUENT_COLUMNS.map((column) => {
                                                    const isActive = indexConstituentSort.key === column.key;
                                                    return (
                                                        <th
                                                            key={column.key}
                                                            className={column.align === 'right' ? 'stocks-cell-right' : undefined}
                                                            scope="col"
                                                            aria-sort={
                                                                isActive
                                                                    ? indexConstituentSort.direction === 'asc'
                                                                        ? 'ascending'
                                                                        : 'descending'
                                                                    : 'none'
                                                            }
                                                        >
                                                            <button
                                                                type="button"
                                                                className="stocks-sort-button"
                                                                onClick={() => handleIndexConstituentSort(column.key)}
                                                            >
                                                                <span>{column.label}</span>
                                                                {isActive && (
                                                                    <span className="stocks-sort-indicator">
                                                                        {indexConstituentSort.direction === 'asc' ? '↑' : '↓'}
                                                                    </span>
                                                                )}
                                                            </button>
                                                        </th>
                                                    );
                                                })}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {sortedIndexConstituents.map((row) => (
                                                <tr key={row.symbol} onClick={() => onCompanyClick(row.symbol)}>
                                                    <td className="stocks-symbol-cell">
                                                        <span className="stocks-symbol-main">
                                                            <SymbolLogo
                                                                symbol={row.symbol}
                                                                name={row.symbol}
                                                                kind="stock"
                                                                logoUrl={row.logo_url}
                                                                size="xs"
                                                                className="stocks-symbol-logo"
                                                            />
                                                            <span>{row.symbol}</span>
                                                        </span>
                                                    </td>
                                                    <td data-label="Fiyat" className="stocks-cell-right stocks-price-cell">{constituentPrice(row)}</td>
                                                    <td data-label="Değişim" className={`stocks-cell-right ${getTableChangeClass(row.change_pct)}`}>
                                                        {formatTablePct(row.change_pct)}
                                                    </td>
                                                    <td data-label="Hacim" className="stocks-cell-right stocks-volume-cell">{formatVolume(row.volume)}</td>
                                                    <td data-label="Ağırlık" className="stocks-cell-right">{formatWeight(row.weight_pct)}</td>
                                                    <td data-label="Puan" className={`stocks-cell-right ${getTableChangeClass(row.point_effect)}`}>
                                                        {formatPointEffect(row.point_effect)}
                                                    </td>
                                                    <td data-label="Endeks etkisi" className={`stocks-cell-right ${getTableChangeClass(getImpactPct(row, indexImpactLevel))}`}>
                                                        {formatImpactPct(getImpactPct(row, indexImpactLevel))}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </section>
                        </>
                    )}
                </div>
            )}
                </div>
                {activeSection === 'markets' && market && (
                    <MarketWatchRail
                        xu100Rows={market.rows}
                        onSelectTicker={onCompanyClick}
                        onSelectFund={(fundCode) => onOpenFund?.(fundCode)}
                        mobilePanelOpen={isMobileViewport ? mobileMarketPanelOpen : undefined}
                        onMobilePanelOpenChange={isMobileViewport ? setMobileMarketPanelOpen : undefined}
                    />
                )}
            </div>
        </div>
    );
}
