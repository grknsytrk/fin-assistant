import { useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { Plus, Search, Trash2, X } from 'lucide-react';
import { apiClient } from '../../../api/client';
import type {
    FundSummary,
    MarketIndexCode,
    MarketIndexDetailResponse,
    MarketIndexLinePoint,
    MarketStockCardItem,
    MarketStockRow,
    MarketUniverseRow,
} from '../../../api/types';
import SymbolLogo from '../../../components/SymbolLogo';
import { MAX_WATCHLIST_ITEMS, normalizeWatchlistSymbol, type WatchlistItem } from '../../../hooks/useWatchlist';
import { searchWatchlistFunds } from '../../../utils/watchlistData';
import {
    formatIndexPrice,
    formatTablePct,
    formatUpdateTime,
    getTableChangeClass,
    mobileSparklinePath,
} from '../stockScreenerConfig';
import './MobileMarketOverview.css';

export interface MobileMarketOverviewProps {
    index: MarketIndexDetailResponse | null;
    rows: MarketUniverseRow[];
    stockQuoteRows: MarketStockRow[];
    stockCardItems: MarketStockCardItem[];
    watchlistItems: WatchlistItem[];
    fundRows: Record<string, FundSummary | null>;
    isMobileViewport: boolean;
    onSelectTicker: (ticker: string) => void;
    onSelectIndex: (index: MarketIndexCode) => void;
    onOpenFund?: (fundCode: string) => void;
    onAddStock: (symbol: string) => void;
    onAddFund: (fund: FundSummary) => void;
    onRemoveWatchlistItem: (item: WatchlistItem) => void;
    onSearchOpenChange: (open: boolean) => void;
}

export function MobileMarketOverview({
    index,
    rows,
    stockQuoteRows,
    stockCardItems,
    watchlistItems,
    fundRows,
    isMobileViewport,
    onSelectTicker,
    onSelectIndex,
    onOpenFund,
    onAddStock,
    onAddFund,
    onRemoveWatchlistItem,
    onSearchOpenChange,
}: MobileMarketOverviewProps) {
    const [searchOpen, setSearchOpen] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [companySearchItems, setCompanySearchItems] = useState<MarketUniverseRow[]>([]);
    const [companySearchLoading, setCompanySearchLoading] = useState(false);
    const [fundSearchItems, setFundSearchItems] = useState<FundSummary[]>([]);
    const [fundSearchLoading, setFundSearchLoading] = useState(false);
    const [watchlistChartPoints, setWatchlistChartPoints] = useState<Record<string, MarketIndexLinePoint[] | null>>({});
    const [swipedWatchlistKey, setSwipedWatchlistKey] = useState<string | null>(null);
    const swipeStartRef = useRef<{ key: string; x: number; y: number } | null>(null);
    const suppressRowClickRef = useRef(false);
    const rowBySymbol = new Map(rows.map((row) => [normalizeWatchlistSymbol(row.company), row]));
    const quoteRowBySymbol = new Map(stockQuoteRows.map((row) => [normalizeWatchlistSymbol(row.symbol || row.company), row]));
    const cardBySymbol = useMemo(
        () => new Map(stockCardItems.map((row) => [normalizeWatchlistSymbol(row.symbol), row])),
        [stockCardItems],
    );
    const indexRowBySymbol = new Map((index?.constituents || []).map((row) => [normalizeWatchlistSymbol(row.symbol), row]));
    const visibleWatchlistChartSymbols = useMemo(
        () => watchlistItems
            .slice(0, 5)
            .filter((item) => item.kind === 'stock')
            .map((item) => normalizeWatchlistSymbol(item.symbol))
            .filter(Boolean),
        [watchlistItems],
    );
    const watchlistChartSymbolsToLoad = useMemo(
        () => visibleWatchlistChartSymbols.filter((symbol) => (
            !mobileSparklinePath(cardBySymbol.get(symbol)?.line_points ?? [])
            && !(symbol in watchlistChartPoints)
        )),
        [cardBySymbol, visibleWatchlistChartSymbols, watchlistChartPoints],
    );

    useEffect(() => {
        if (!isMobileViewport || watchlistChartSymbolsToLoad.length === 0) return undefined;
        const controller = new AbortController();
        let active = true;
        void Promise.all(watchlistChartSymbolsToLoad.map(async (symbol) => {
            try {
                const payload = await apiClient.marketStockCardChart(symbol, '1d', { signal: controller.signal });
                return [symbol, payload.line_points ?? []] as const;
            } catch {
                return [symbol, null] as const;
            }
        })).then((entries) => {
            if (!active || controller.signal.aborted) return;
            setWatchlistChartPoints((current) => ({
                ...current,
                ...Object.fromEntries(entries),
            }));
        });
        return () => {
            active = false;
            controller.abort();
        };
    }, [isMobileViewport, watchlistChartSymbolsToLoad]);

    useEffect(() => {
        onSearchOpenChange(searchOpen);
        return () => onSearchOpenChange(false);
    }, [onSearchOpenChange, searchOpen]);
    const savedStockSymbols = new Set(
        watchlistItems
            .filter((item) => item.kind === 'stock')
            .map((item) => normalizeWatchlistSymbol(item.symbol)),
    );
    const isWatchlistFull = watchlistItems.length >= MAX_WATCHLIST_ITEMS;
    const normalizedSearch = searchTerm.trim().toUpperCase();
    useEffect(() => {
        const query = searchTerm.trim();
        if (!searchOpen || !query) {
            setCompanySearchItems([]);
            setCompanySearchLoading(false);
            return;
        }
        let active = true;
        const controller = new AbortController();
        setCompanySearchLoading(true);
        const timer = window.setTimeout(() => {
            apiClient.marketStockSearch({ q: query, index: 'XUTUM', limit: 20, signal: controller.signal })
                .then((payload) => {
                    if (active) setCompanySearchItems(payload.rows || []);
                })
                .catch((error) => {
                    if ((error as Error)?.name === 'AbortError') return;
                    if (active) setCompanySearchItems([]);
                })
                .finally(() => {
                    if (active) setCompanySearchLoading(false);
                });
        }, 160);
        return () => {
            active = false;
            window.clearTimeout(timer);
            controller.abort();
        };
    }, [searchOpen, searchTerm]);

    useEffect(() => {
        const query = searchTerm.trim();
        if (!searchOpen || !query) {
            setFundSearchItems([]);
            setFundSearchLoading(false);
            return undefined;
        }

        let active = true;
        setFundSearchLoading(true);
        const timer = window.setTimeout(() => {
            searchWatchlistFunds(query)
                .then((payload) => {
                    if (active) setFundSearchItems(payload);
                })
                .catch(() => {
                    if (active) setFundSearchItems([]);
                })
                .finally(() => {
                    if (active) setFundSearchLoading(false);
                });
        }, 180);
        return () => {
            active = false;
            window.clearTimeout(timer);
        };
    }, [searchOpen, searchTerm]);

    const searchCandidates = new Map<string, { symbol: string; name: string; logoUrl?: string | null }>();
    for (const row of rows) {
        const symbol = normalizeWatchlistSymbol(row.company);
        searchCandidates.set(symbol, { symbol, name: row.name || row.company, logoUrl: row.logo_url });
    }
    for (const item of companySearchItems) {
        const symbol = normalizeWatchlistSymbol(item.symbol || item.company);
        if (!symbol || searchCandidates.has(symbol)) continue;
        searchCandidates.set(symbol, { symbol, name: item.name || symbol, logoUrl: item.logo_url });
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
    const savedFundCodes = new Set(
        watchlistItems
            .filter((item) => item.kind === 'fund')
            .map((item) => normalizeWatchlistSymbol(item.symbol)),
    );
    const fundSearchResults = fundSearchItems
        .filter((fund) => !savedFundCodes.has(normalizeWatchlistSymbol(fund.fund_code)))
        .slice(0, 8);
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
        <section className={`mobile-market-overview${searchOpen ? ' is-watchlist-search-open' : ''}`} aria-label="Mobil piyasa özeti">
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
                            <small>{index ? formatUpdateTime(index.as_of) : 'Güncelleniyor…'}</small>
                        </div>
                    </div>
                    <div className="mobile-market-index-quote">
                        <span title="Günlük">Günlük</span>
                        <strong>{formatIndexPrice(index?.price ?? null)}</strong>
                        <b className={getTableChangeClass(index?.change_pct ?? null)}>
                            {formatTablePct(index?.change_pct ?? null)}
                        </b>
                    </div>
                </div>
                <div className={`mobile-market-index-chart ${getTableChangeClass(index?.change_pct ?? null)}${!sparkline && !index ? ' is-loading' : ''}`} aria-hidden="true">
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
                            const quoteRow = item.kind === 'stock' ? quoteRowBySymbol.get(symbol) : null;
                            const cardRow = item.kind === 'stock' ? cardBySymbol.get(symbol) : null;
                            const indexRow = item.kind === 'stock' ? indexRowBySymbol.get(symbol) : null;
                            const price = quoteRow?.price ?? cardRow?.price ?? indexRow?.price ?? row?.price ?? fundRow?.price ?? null;
                            const changePct = quoteRow?.change_pct ?? cardRow?.change_pct ?? indexRow?.change_pct ?? row?.change_pct ?? fundRow?.daily_return ?? null;
                            const asOf = quoteRow?.price_as_of ?? cardRow?.as_of ?? row?.price_as_of ?? fundRow?.as_of ?? (indexRow ? index?.as_of ?? null : null);
                            const chartPoints = cardRow?.line_points?.length
                                ? cardRow.line_points
                                : (watchlistChartPoints[symbol] || []);
                            const chartPath = item.kind === 'stock' ? mobileSparklinePath(chartPoints) : '';
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
                                            name={item.label || quoteRow?.company || cardRow?.company || indexRow?.symbol || row?.company || fundRow?.name || symbol}
                                            kind={item.kind}
                                            logoUrl={quoteRow?.logo_url ?? cardRow?.logo_url ?? indexRow?.logo_url ?? row?.logo_url}
                                            size="sm"
                                        />
                                        <span className="mobile-market-watchlist-symbol">
                                            <strong>{symbol}</strong>
                                            <small>{formatUpdateTime(asOf)}</small>
                                        </span>
                                        <span className="mobile-market-watchlist-price">
                                            <small title="Günlük">Günlük</small>
                                            {formatIndexPrice(price)}
                                        </span>
                                        <span className={`mobile-market-watchlist-chart ${getTableChangeClass(changePct)}`} aria-hidden="true">
                                            {chartPath ? (
                                                <svg viewBox="0 0 220 52" preserveAspectRatio="none">
                                                    <path d={chartPath} />
                                                </svg>
                                            ) : null}
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
                                placeholder="Hisse veya fon ara..."
                                aria-label="İzleme listesine eklenecek hisse veya fonu ara"
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
                                {companySearchLoading && fundSearchLoading && searchResults.length === 0 && fundSearchResults.length === 0 ? (
                                    <span className="mobile-market-watchlist-search-empty">Varlıklar aranıyor...</span>
                                ) : searchResults.length > 0 || fundSearchResults.length > 0 ? (
                                    <>
                                        {searchResults.map((result) => (
                                            <button
                                                key={`stock-${result.symbol}`}
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
                                                <span>Hisse · Ekle</span>
                                            </button>
                                        ))}
                                        {fundSearchResults.map((fund) => (
                                            <button
                                                key={`fund-${fund.fund_code}`}
                                                type="button"
                                                onClick={() => {
                                                    onAddFund(fund);
                                                    setSearchTerm('');
                                                    setSearchOpen(false);
                                                }}
                                            >
                                                <SymbolLogo
                                                    symbol={fund.fund_code}
                                                    name={fund.name}
                                                    kind="fund"
                                                    size="sm"
                                                />
                                                <strong>{fund.fund_code}</strong>
                                                <span>Fon · Ekle</span>
                                            </button>
                                        ))}
                                    </>
                                ) : (
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
