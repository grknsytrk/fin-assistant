import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { DragEvent as ReactDragEvent } from 'react';
import { apiClient } from '../api/client';
import type {
    FundSummary,
    MarketIndexCode,
    MarketIndexConstituent,
    MarketIndexDetailResponse,
    MarketIndicesResponse,
    MarketStockCardItem,
    MarketStockCardsResponse,
    MarketStockIndex,
    MarketStocksResponse,
    MarketUniverseResponse,
} from '../api/types';
import { DEFAULT_STOCK_RETURN_MODE } from '../routing/routes';
import type { StockReturnMode } from '../routing/routes';
import MarketWatchRail from '../components/MarketWatchRail';
import MarketsNavigation from '../components/MarketsNavigation';
import { buildDocumentTitle, formatTitleNumber, formatTitlePct, useDocumentTitle } from '../hooks/useDocumentTitle';
import { normalizeWatchlistSymbol, useWatchlist } from '../hooks/useWatchlist';
import { useTableVirtualizer } from '../hooks/useTableVirtualizer';
import {
    getMarketStockCardSymbolsNeedingLoad,
    retainMarketStockCardsForSymbols,
} from '../utils/stockCardData';
import { isBistTradingHours } from '../utils/marketHours';
import { cacheWatchlistFund, getCachedWatchlistFund } from '../utils/watchlistData';
import { BoundedMemoryCache } from '../utils/lruCache';
import '../styles/market-shared.css';
import './MarketsView.css';
import { MarketOverviewSection } from './markets/sections/MarketOverviewSection';
import { StocksSection } from './markets/sections/StocksSection';
import { IndicesSection } from './markets/sections/IndicesSection';
import {
    EMPTY_STOCK_SCREENER_FILTERS,
    MAIN_MARKET_INDICES,
    MARKET_DETAIL_REFRESH_DESKTOP_MS,
    MARKET_DETAIL_REFRESH_MOBILE_MS,
    MARKET_LIST_REFRESH_DESKTOP_MS,
    MARKET_LIST_REFRESH_MOBILE_MS,
    MARKET_QUOTE_REFRESH_MS,
    MAX_STOCK_CARDS,
    PRIMARY_MARKET_INDEX_RANK,
    SCREENER_RANGE_SPECS,
    SCREENER_TOGGLE_SPECS,
    STOCK_CARD_BACKGROUND_BATCH_SIZE,
    STOCK_CARD_BACKGROUND_DELAY_MS,
    STOCK_CARD_INITIAL_BATCH_SIZE,
    STOCK_CARD_STORAGE_KEY,
    STOCK_COLUMNS,
    STOCK_FUNDAMENTAL_COLUMNS,
    STOCK_SCREENER_PRESETS,
    buildHeatmapTiles,
    computeScreenerData,
    constituentSortValue,
    getBenchmarkIndex,
    hasStockCardLoadedData,
    mergeStockCardPayloads,
    normalizeStockCardSymbol,
    numericOrNull,
    quickStocksFromUniverse,
    readStoredStockCards,
    splitStockCardBatches,
    stockCardsMemoryKey,
    stockSortValue,
    treemapValue,
} from './markets/stockScreenerConfig';
import type {
    HeatmapTile,
    IndexConstituentSortKey,
    IndexGroupFilter,
    IndexSortKey,
    MarketSection,
    MarketStockCardsMemoryCache,
    MarketUniverseMemoryCache,
    MarketsViewProps,
    ScreenerActiveChip,
    SortDirection,
    StockCardDropPlacement,
    StockScreenerCategory,
    StockScreenerFilterState,
    StockScreenerPresetId,
    StockScreenerRangeKey,
    StockSortKey,
    StockTableMode,
} from './markets/types';

// Keep the last visible market data while the user changes sections. The
// backend still refreshes in the background, so this is a stale-while-
// revalidate UX cache rather than a replacement for server-side freshness.
let marketUniverseMemoryCache: MarketUniverseMemoryCache | null = null;
const marketStockCardsMemoryCache = new BoundedMemoryCache<string, MarketStockCardsMemoryCache>(50);

export default function MarketsView({
    routeSection = 'stocks',
    routeStockIndex = 'XUTUM',
    routeSelectedIndex = null,
    routeReturnMode = DEFAULT_STOCK_RETURN_MODE,
    onNavigateSection,
    onNavigateStockSection,
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
    const [indexGroup, setIndexGroup] = useState<IndexGroupFilter>('all');
    const [stockCardSearchTerm, setStockCardSearchTerm] = useState('');
    const [navCollapsed, setNavCollapsed] = useState(false);
    const [railCollapsed, setRailCollapsed] = useState(false);
    const collapsedMenuCount = (navCollapsed ? 1 : 0) + (railCollapsed ? 1 : 0);
    const [activeSection, setActiveSection] = useState<MarketSection>(routeSection);
    const [isMobileViewport, setIsMobileViewport] = useState(() => (
        typeof window !== 'undefined' && window.matchMedia('(max-width: 760px)').matches
    ));
    const [mobileMarketPanelOpen, setMobileMarketPanelOpen] = useState(false);
    const [mobileWatchlistSearchOpen, setMobileWatchlistSearchOpen] = useState(false);
    const [mobileIndexDetail, setMobileIndexDetail] = useState<MarketIndexDetailResponse | null>(null);
    const [mobileWatchlistFundRows, setMobileWatchlistFundRows] = useState<Record<string, FundSummary | null>>({});
    const [selectedIndex, setSelectedIndex] = useState<MarketIndexCode | null>(routeSelectedIndex);
    const [stockIndex, setStockIndex] = useState<MarketStockIndex>(routeStockIndex);
    const [returnMode, setReturnMode] = useState<StockReturnMode>(routeReturnMode);
    const [stockTableMode, setStockTableMode] = useState<StockTableMode>('performance');
    const [screenerFilters, setScreenerFilters] = useState<StockScreenerFilterState>({ ...EMPTY_STOCK_SCREENER_FILTERS });
    const [activeScreenerPreset, setActiveScreenerPreset] = useState<StockScreenerPresetId | null>(null);
    const [screenerModalOpen, setScreenerModalOpen] = useState(false);
    const [screenerCategory, setScreenerCategory] = useState<StockScreenerCategory>('all');

    const { activeFilterCount, screenerCategoryCounts, screenerActiveChips } = useMemo(() => {
        const counts: Record<StockScreenerCategory, number> = {
            all: 0,
            profile: 0,
            valuation: 0,
            financials: 0,
            market: 0,
            historical: 0,
        };
        const chips: ScreenerActiveChip[] = [];

        if (screenerFilters.sector) {
            counts.profile += 1;
            chips.push({
                id: 'sector',
                label: `Sektör: ${screenerFilters.sector}`,
                onClear: () => {
                    setScreenerFilters((c) => ({ ...c, sector: '' }));
                    setActiveScreenerPreset(null);
                },
            });
        }

        for (const toggle of SCREENER_TOGGLE_SPECS) {
            if (screenerFilters[toggle.key]) {
                counts[toggle.category] += 1;
                chips.push({
                    id: toggle.key,
                    label: toggle.label,
                    onClear: () => {
                        setScreenerFilters((c) => ({ ...c, [toggle.key]: false }));
                        setActiveScreenerPreset(null);
                    },
                });
            }
        }

        for (const spec of SCREENER_RANGE_SPECS) {
            const minRaw = screenerFilters[spec.minKey].trim();
            const maxRaw = screenerFilters[spec.maxKey].trim();
            const hasMin = minRaw !== '';
            const hasMax = maxRaw !== '';
            if (hasMin) counts[spec.category] += 1;
            if (hasMax) counts[spec.category] += 1;
            if (hasMin || hasMax) {
                const boundText = hasMin && hasMax
                    ? `${minRaw} – ${maxRaw} ${spec.unit}`
                    : hasMin
                        ? `≥ ${minRaw} ${spec.unit}`
                        : `≤ ${maxRaw} ${spec.unit}`;
                chips.push({
                    id: spec.field,
                    label: `${spec.label}: ${boundText}`,
                    onClear: () => {
                        setScreenerFilters((c) => ({ ...c, [spec.minKey]: '', [spec.maxKey]: '' }));
                        setActiveScreenerPreset(null);
                    },
                });
            }
        }

        counts.all = counts.profile + counts.valuation + counts.financials + counts.market + counts.historical;
        return {
            activeFilterCount: counts.all,
            screenerCategoryCounts: counts,
            screenerActiveChips: chips,
        };
    }, [screenerFilters]);

    const isScreenerActive = activeFilterCount > 0;

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
    const [indexConstituentFilter, setIndexConstituentFilter] = useState<'all' | 'positive' | 'neutral' | 'negative'>('all');
    const stocksRequestIdRef = useRef(0);
    const stocksInFlightIndexRef = useRef<MarketStockIndex | null>(null);
    const indicesInFlightRef = useRef(false);
    const indexDetailRequestIdRef = useRef(0);
    const indexDetailInFlightCodeRef = useRef<MarketIndexCode | null>(null);
    const mobileWatchlistFundFailedRef = useRef<Record<string, number>>({});
    const stockCardQuickInFlightRef = useRef<Promise<void> | null>(null);
    const stockCardFullInFlightRef = useRef<Promise<void> | null>(null);
    const stockCardsRef = useRef<MarketStockCardsResponse | null>(stockCards);
    const marketPageRef = useRef<HTMLDivElement | null>(null);
    const pendingMarketScrollResetRef = useRef(true);
    const marketScrollResetFrameRef = useRef<number | null>(null);
    const latestActiveSectionRef = useRef<MarketSection>(activeSection);
    const latestStockIndexRef = useRef<MarketStockIndex>(stockIndex);
    const latestSelectedIndexRef = useRef<MarketIndexCode | null>(selectedIndex);
    const latestStockCardSymbolsRef = useRef(stockCardSymbols.join(','));
    const previousStockCardSymbolsRef = useRef<string[]>(stockCardSymbols);
    const draggingStockCardSymbolRef = useRef<string | null>(null);
    const [draggingStockCardSymbol, setDraggingStockCardSymbol] = useState<string | null>(null);
    const stockCardSymbolsKey = stockCardSymbols.join(',');
    const marketListRefreshMs = isMobileViewport
        ? MARKET_LIST_REFRESH_MOBILE_MS
        : MARKET_LIST_REFRESH_DESKTOP_MS;
    const marketDetailRefreshMs = isMobileViewport
        ? MARKET_DETAIL_REFRESH_MOBILE_MS
        : MARKET_DETAIL_REFRESH_DESKTOP_MS;
    const watchlist = useWatchlist();
    const mobileWatchlistFundCodes = useMemo(
        () => watchlist.items
            .filter((item) => item.kind === 'fund')
            .map((item) => normalizeWatchlistSymbol(item.symbol)),
        [watchlist.items],
    );

    useEffect(() => {
        stockCardsRef.current = stockCards;
    }, [stockCards]);

    useEffect(() => {
        const mediaQuery = window.matchMedia('(max-width: 760px)');
        const updateViewport = () => setIsMobileViewport(mediaQuery.matches);
        updateViewport();
        mediaQuery.addEventListener('change', updateViewport);
        return () => mediaQuery.removeEventListener('change', updateViewport);
    }, []);

    useEffect(() => {
        if (activeSection !== 'markets' || !isMobileViewport) return;
        let alive = true;
        let hasLoaded = false;
        const loadMobileIndex = () => {
            apiClient
                .marketIndexDetail('XU100')
                .then((payload) => {
                    if (!alive) return;
                    hasLoaded = true;
                    setMobileIndexDetail(payload);
                })
                .catch(() => {
                    // Keep previously loaded XU100 data instead of wiping it to null on transient errors.
                });
        };
        loadMobileIndex();
        const intervalId = window.setInterval(() => {
            if (document.visibilityState === 'visible' && (!hasLoaded || isBistTradingHours())) {
                loadMobileIndex();
            }
        }, marketDetailRefreshMs);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && (!hasLoaded || isBistTradingHours())) {
                loadMobileIndex();
            }
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            alive = false;
            window.clearInterval(intervalId);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [activeSection, isMobileViewport, marketDetailRefreshMs]);

    useEffect(() => {
        if (activeSection !== 'markets' || !isMobileViewport) return;
        const cachedCatalog = apiClient.getCachedFundsCatalog();
        const freshCatalogRows = (
            cachedCatalog
            && Date.now() - cachedCatalog.fetchedAt < 5 * 60_000
            && Array.isArray(cachedCatalog.payload?.rows)
        )
            ? cachedCatalog.payload.rows
            : null;
        const catalogByCode = freshCatalogRows
            ? new Map(freshCatalogRows.map((row) => [normalizeWatchlistSymbol(row.fund_code), row]))
            : null;

        const cachedUpdates: Record<string, FundSummary> = {};
        const missingCodes: string[] = [];
        for (const code of mobileWatchlistFundCodes) {
            const localCached = getCachedWatchlistFund(code);
            if (localCached) {
                if (mobileWatchlistFundRows[code] !== localCached) {
                    cachedUpdates[code] = localCached;
                }
                continue;
            }
            const catalogMatch = catalogByCode?.get(code);
            if (catalogMatch) {
                cacheWatchlistFund(catalogMatch);
                if (mobileWatchlistFundRows[code] !== catalogMatch) {
                    cachedUpdates[code] = catalogMatch;
                }
                continue;
            }
            if (!mobileWatchlistFundRows[code]) {
                const failedAt = mobileWatchlistFundFailedRef.current[code];
                if (!failedAt || Date.now() - failedAt > 60_000) {
                    missingCodes.push(code);
                }
            }
        }

        if (Object.keys(cachedUpdates).length > 0) {
            setMobileWatchlistFundRows((current) => ({ ...current, ...cachedUpdates }));
        }
        if (missingCodes.length === 0) return;

        let cancelled = false;
        Promise.all(
            missingCodes.map((code) => apiClient
                .fundDetail(code)
                .then((detail) => {
                    delete mobileWatchlistFundFailedRef.current[code];
                    cacheWatchlistFund(detail);
                    return [code, detail] as const;
                })
                .catch(() => {
                    mobileWatchlistFundFailedRef.current[code] = Date.now();
                    return [code, null] as const;
                })),
        ).then((entries) => {
            if (cancelled) return;
            const resolved = entries.filter((entry): entry is readonly [string, NonNullable<typeof entry[1]>] => entry[1] != null);
            if (resolved.length === 0) return;
            setMobileWatchlistFundRows((current) => {
                const next = { ...current };
                for (const [code, detail] of resolved) next[code] = detail;
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
        if (!isMobileViewport && stockCardSymbols.length > 0 && !stockCards && !stockCardsError) return;

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
        latestActiveSectionRef.current = activeSection;
    }, [activeSection]);

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
        loadStats();
        const intervalId = window.setInterval(() => {
            if (document.visibilityState === 'visible' && isBistTradingHours()) {
                loadStats(true);
            }
        }, 5 * 60 * 1000);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && isBistTradingHours()) loadStats(true);
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            window.clearInterval(intervalId);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [activeSection]);

    useEffect(() => {
        return apiClient.subscribeMarketStocks((index, payload) => {
            if (
                (activeSection === 'stocks' && index === latestStockIndexRef.current)
                || (activeSection === 'markets' && isMobileViewport && index === 'XUTUM')
            ) {
                setStocks(payload);
                setStocksError(null);
            }
        });
    }, [activeSection, isMobileViewport]);

    useEffect(() => {
        const needsMobileOverviewQuotes = activeSection === 'markets' && isMobileViewport;
        if (activeSection !== 'stocks' && !needsMobileOverviewQuotes) return;
        const targetIndex: MarketStockIndex = needsMobileOverviewQuotes ? 'XUTUM' : stockIndex;
        const cached = apiClient.getCachedMarketStocks(targetIndex);
        if (cached) {
            setStocks(cached);
        } else if (activeSection === 'stocks') {
            setStocks(null);
        }
        setStocksError(null);
        loadStocks(Boolean(cached), false, targetIndex);
        const releaseStream = apiClient.retainMarketStocksStream(targetIndex);
        const intervalId = window.setInterval(() => {
            if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
            if (apiClient.isMarketStocksStreamActive(targetIndex)) return;
            loadStocks(true, false, targetIndex);
        }, marketListRefreshMs);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && isBistTradingHours()) {
                loadStocks(true, false, targetIndex);
            }
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            releaseStream();
            window.clearInterval(intervalId);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [activeSection, isMobileViewport, marketListRefreshMs, stockIndex]);

    useEffect(() => {
        if (activeSection !== 'markets' || isMobileViewport) return;
        if (stockCardSymbols.length === 0) {
            stockCardsRef.current = null;
            setStockCards(null);
            setStockCardsError(null);
            setStockCardsLoading(false);
            setStockCardPendingSymbols([]);
            return;
        }
        const cacheKey = stockCardsMemoryKey(stockCardSymbols);
        const cached = marketStockCardsMemoryCache.get(cacheKey);
        if (cached) {
            stockCardsRef.current = cached.data;
            setStockCards(cached.data);
            setStockCardPendingSymbols((previous) => previous.filter(
                (symbol) => !(cached.data.items || []).some(
                    (item) => item.symbol === symbol && item.card_stage !== 'quick',
                ),
            ));
        } else if (stockCardsRef.current) {
            // Adding or reordering a card must not blank cards that are
            // already visible. Keep the current payload and request only the
            // symbols that are not present in it.
            const retained = retainMarketStockCardsForSymbols(stockCardsRef.current, stockCardSymbols);
            stockCardsRef.current = retained;
            setStockCards(retained);
        } else {
            setStockCards(null);
        }

        const controller = new AbortController();
        let delayTimer: number | null = null;
        let detailBatchesLoading = false;
        let quoteBatchesLoading = false;
        const initialSymbols = getMarketStockCardSymbolsNeedingLoad(stockCardSymbols, stockCardsRef.current);
        const buildBatches = (symbols: string[]) => {
            const initialBatch = symbols.slice(0, STOCK_CARD_INITIAL_BATCH_SIZE);
            return initialBatch.length > 0
                ? [initialBatch, ...splitStockCardBatches(symbols.slice(STOCK_CARD_INITIAL_BATCH_SIZE), STOCK_CARD_BACKGROUND_BATCH_SIZE)]
                : [];
        };
        const loadBatches = async (symbols: string[], includeFull = true, includeQuick = true) => {
            if (includeFull ? detailBatchesLoading : quoteBatchesLoading) return;
            const batches = includeFull
                ? buildBatches(symbols)
                : symbols.length > 0 ? [symbols] : [];
            if (batches.length === 0) return;
            if (includeFull) detailBatchesLoading = true;
            else quoteBatchesLoading = true;
            try {
                for (const [index, batch] of batches.entries()) {
                    if (controller.signal.aborted) return;
                    if (includeFull && index > 0) {
                        await new Promise<void>((resolve) => {
                            delayTimer = window.setTimeout(resolve, STOCK_CARD_BACKGROUND_DELAY_MS);
                        });
                        if (controller.signal.aborted) return;
                    }
                    if (includeQuick) {
                        await loadStockCards(includeFull && (index > 0 || Boolean(cached)), false, batch, controller.signal, 'quick');
                    }
                    if (includeFull) {
                        await loadStockCards(true, false, batch, controller.signal, 'full');
                    }
                }
            } finally {
                if (includeFull) detailBatchesLoading = false;
                else quoteBatchesLoading = false;
            }
        };

        // When the selection changes, load only new or incomplete cards. The
        // periodic refresh below still revalidates the complete selection.
        void loadBatches(initialSymbols);
        const intervalId = window.setInterval(() => {
            if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
            void loadBatches(stockCardSymbols, true, false);
        }, marketDetailRefreshMs);
        const quoteIntervalId = window.setInterval(() => {
            if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
            void loadBatches(stockCardSymbols, false, true);
        }, MARKET_QUOTE_REFRESH_MS);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && isBistTradingHours()) {
                void loadBatches(stockCardSymbols, true);
            }
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            controller.abort();
            if (delayTimer !== null) window.clearTimeout(delayTimer);
            window.clearInterval(intervalId);
            window.clearInterval(quoteIntervalId);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [activeSection, isMobileViewport, marketDetailRefreshMs, stockCardSymbols, stockCardSymbolsKey]);

    useEffect(() => {
        if (activeSection !== 'indices') return;
        loadIndices(false, false);
        const intervalId = window.setInterval(() => {
            if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
            loadIndices(true, false);
        }, marketListRefreshMs);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && isBistTradingHours()) loadIndices(true, false);
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            window.clearInterval(intervalId);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [activeSection, marketListRefreshMs]);

    useEffect(() => {
        if (activeSection !== 'indices' || !selectedIndex) return;
        setIndexDetail(null);
        setIndexDetailError(null);
        loadIndexDetail(false, false, selectedIndex);
        const intervalId = window.setInterval(() => {
            if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
            loadIndexDetail(true, false, selectedIndex);
        }, marketDetailRefreshMs);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && isBistTradingHours()) loadIndexDetail(true, false, selectedIndex);
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            window.clearInterval(intervalId);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [activeSection, marketDetailRefreshMs, selectedIndex]);

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
        if (silent && stocksInFlightIndexRef.current === requestedIndex) return;
        const requestId = stocksRequestIdRef.current + 1;
        stocksRequestIdRef.current = requestId;
        stocksInFlightIndexRef.current = requestedIndex;
        const cachedStocks = apiClient.getCachedMarketStocks(requestedIndex);
        if (cachedStocks) {
            setStocks(cachedStocks);
        } else if (!silent) {
            void apiClient
                .marketUniverse({ index: requestedIndex })
                .then((universePayload) => {
                    if (requestId !== stocksRequestIdRef.current) return;
                    if (latestActiveSectionRef.current === 'stocks' && latestStockIndexRef.current !== requestedIndex) return;
                    if (latestActiveSectionRef.current === 'markets' && requestedIndex !== 'XUTUM') return;
                    setStocks((current) => (
                        current?.index === requestedIndex && !current.partial
                            ? current
                            : quickStocksFromUniverse(requestedIndex, universePayload)
                    ));
                })
                .catch(() => {
                    // The detailed stock request remains the source of truth.
                });
        }
        if (!silent) setStocksLoading(true);
        if (!silent) setStocksError(null);
        try {
            const stocksPayload = await apiClient.marketStocks({ index: requestedIndex, refresh });
            if (requestId !== stocksRequestIdRef.current) return;
            if (latestActiveSectionRef.current === 'stocks' && latestStockIndexRef.current !== requestedIndex) return;
            if (latestActiveSectionRef.current === 'markets' && requestedIndex !== 'XUTUM') return;
            setStocks(stocksPayload);
            setStocksError(null);
        } catch (err: any) {
            if (requestId !== stocksRequestIdRef.current) return;
            if (!silent || !stocks) {
                setStocksError(err.message || 'Hisse verileri yüklenemedi.');
            }
        } finally {
            if (requestId === stocksRequestIdRef.current) {
                stocksInFlightIndexRef.current = null;
                if (!silent) setStocksLoading(false);
            }
        }
    }

    async function loadStockCards(
        silent = false,
        refresh = false,
        requestedSymbols: string[] = stockCardSymbols,
        signal?: AbortSignal,
        tier: 'quick' | 'full' = 'full',
    ) {
        if (requestedSymbols.length === 0) return;
        const inFlightRef = tier === 'quick' ? stockCardQuickInFlightRef : stockCardFullInFlightRef;
        const previousRequest = inFlightRef.current;
        const request = (async () => {
            if (previousRequest) await previousRequest;
            if (signal?.aborted) return;

            if (!silent) setStockCardsLoading(true);
            if (!silent) setStockCardsError(null);
            try {
                const payload = await apiClient.marketStockCards({ symbols: requestedSymbols, refresh, tier, signal });
                if (signal?.aborted) return;
                const latestSymbols = latestStockCardSymbolsRef.current.split(',').filter(Boolean);
                const latestSymbolSet = new Set(latestSymbols);
                if (!requestedSymbols.every((symbol) => latestSymbolSet.has(symbol))) return;
                const mergedPayload = mergeStockCardPayloads(stockCardsRef.current, payload, latestSymbols);
                stockCardsRef.current = mergedPayload;
                marketStockCardsMemoryCache.set(latestSymbols.join(','), { data: mergedPayload, fetchedAt: Date.now() });
                setStockCards(mergedPayload);
                setStockCardsError(null);
                const readySymbols = new Set(
                    (mergedPayload.items ?? [])
                        .filter((item) => item.card_stage !== 'quick' && hasStockCardLoadedData(item))
                        .map((item) => item.symbol),
                );
                if (readySymbols.size > 0) {
                    setStockCardPendingSymbols((previous) => previous.filter((symbol) => !readySymbols.has(symbol)));
                }
            } catch (err: any) {
                if (err?.name === 'AbortError' || signal?.aborted) return;
                if (!silent || !stockCardsRef.current) {
                    setStockCardsError(err.message || 'Hisse kartları şu an hazırlanıyor.');
                }
            } finally {
                if (!silent) setStockCardsLoading(false);
            }
        })();

        inFlightRef.current = request;
        try {
            await request;
        } finally {
            if (inFlightRef.current === request) {
                inFlightRef.current = null;
            }
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
        if (silent && indexDetailInFlightCodeRef.current === requestedIndex) return;
        const requestId = indexDetailRequestIdRef.current + 1;
        indexDetailRequestIdRef.current = requestId;
        indexDetailInFlightCodeRef.current = requestedIndex;
        if (!silent) setIndexDetailLoading(true);
        if (!silent) setIndexDetailError(null);
        try {
            const payload = await apiClient.marketIndexDetail(requestedIndex, { refresh });
            if (requestId !== indexDetailRequestIdRef.current) return;
            if (latestSelectedIndexRef.current !== requestedIndex) return;
            setIndexDetail(payload);
            setIndexDetailError(null);
        } catch (err: any) {
            if (requestId !== indexDetailRequestIdRef.current) return;
            if (!silent || !indexDetail) {
                setIndexDetailError(err.message || 'Endeks detayı yüklenemedi.');
            }
        } finally {
            if (requestId === indexDetailRequestIdRef.current) {
                indexDetailInFlightCodeRef.current = null;
                if (!silent) setIndexDetailLoading(false);
            }
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
                    return row.company.toLowerCase().includes(normalizedStockCardSearch)
                        || String(row.name || '').toLowerCase().includes(normalizedStockCardSearch);
                })
                .slice(0, 18),
        [market?.rows, normalizedStockCardSearch, stockCardSymbols],
    );

    const screenerData = useMemo(
        () => computeScreenerData(stocks?.rows || [], screenerFilters),
        [stocks?.rows, screenerFilters],
    );

    const stockRowsForMode = useMemo(() => {
        if (isScreenerActive) return screenerData.rows;
        return stocks?.rows || [];
    }, [isScreenerActive, screenerData.rows, stocks?.rows]);

    const filteredStocks = useMemo(
        () =>
            stockRowsForMode.filter((row) =>
                row.company.toLowerCase().includes(normalizedSearch),
            ),
        [stockRowsForMode, normalizedSearch],
    );

    const filteredIndices = useMemo(
        () =>
            (indices?.rows || []).filter((row) => {
                if (indexGroup === 'main' && !MAIN_MARKET_INDICES.has(row.symbol)) return false;
                if (indexGroup === 'sector' && MAIN_MARKET_INDICES.has(row.symbol)) return false;
                const query = normalizedSearch;
                if (!query) return true;
                return (
                    row.symbol.toLowerCase().includes(query) ||
                    row.label.toLowerCase().includes(query)
                );
            }),
        [indices?.rows, indexGroup, normalizedSearch],
    );
    const isIndicesFiltered = indexGroup !== 'all' || Boolean(normalizedSearch);

    const sortedIndices = useMemo(() => {
        const arr = [...filteredIndices];
        arr.sort((a, b) => {
            const aPrimaryRank = PRIMARY_MARKET_INDEX_RANK.get(a.symbol);
            const bPrimaryRank = PRIMARY_MARKET_INDEX_RANK.get(b.symbol);
            if (aPrimaryRank !== undefined || bPrimaryRank !== undefined) {
                if (aPrimaryRank === undefined) return 1;
                if (bPrimaryRank === undefined) return -1;
                return aPrimaryRank - bPrimaryRank;
            }

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
            const av = stockSortValue(
                a,
                stockSort.key,
                stocks?.benchmarks,
                returnMode,
                screenerData.sectorDiffByCompany.get(a.company),
            );
            const bv = stockSortValue(
                b,
                stockSort.key,
                stocks?.benchmarks,
                returnMode,
                screenerData.sectorDiffByCompany.get(b.company),
            );
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
    }, [filteredStocks, stockSort, stocks?.benchmarks, returnMode, screenerData.sectorDiffByCompany]);

    const stockTableWrapRef = useRef<HTMLDivElement>(null);
    const {
        startIndex: stockStartIndex,
        endIndex: stockEndIndex,
        topSpacerHeight: stockTopSpacerHeight,
        bottomSpacerHeight: stockBottomSpacerHeight,
        containerRef: stockTableWrapCallbackRef,
    } = useTableVirtualizer({
        totalRows: sortedStocks.length,
        rowHeight: 38,
        overscan: 12,
        containerRef: stockTableWrapRef,
        resetDeps: [
            searchTerm,
            stockIndex,
            stockTableMode,
            returnMode,
            stockSort.key,
            stockSort.direction,
            activeFilterCount,
            screenerFilters,
        ],
    });

    const visibleStocks = useMemo(() => {
        return sortedStocks.slice(stockStartIndex, stockEndIndex);
    }, [sortedStocks, stockStartIndex, stockEndIndex]);

    const activeStockColumns = stockTableMode === 'fundamentals' ? STOCK_FUNDAMENTAL_COLUMNS : STOCK_COLUMNS;
    const stockTableTotalWidth = useMemo(
        () => 48 + activeStockColumns.reduce((sum, column) => sum + (column.width || 100), 0),
        [activeStockColumns],
    );

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
    const totalStocksCount = stocks?.rows?.length ?? sortedStocks.length;
    const isStocksFiltered = sortedStocks.length !== totalStocksCount;
    const indexConstituents = useMemo(() => indexDetail?.constituents ?? [], [indexDetail?.constituents]);
    const indexImpactLevel = numericOrNull(indexDetail?.price) ?? numericOrNull(indexDetail?.prev_close);
    const positiveConstituents = indexConstituents.filter((row) => (row.point_effect || 0) > 0).length;
    const negativeConstituents = indexConstituents.filter((row) => (row.point_effect || 0) < 0).length;
    const neutralConstituents = indexConstituents.length - positiveConstituents - negativeConstituents;
    const sortedIndexConstituents = useMemo(() => {
        const arr = indexConstituents.filter((row) => {
            const effect = row.point_effect ?? 0;
            if (indexConstituentFilter === 'positive') return effect > 0;
            if (indexConstituentFilter === 'negative') return effect < 0;
            if (indexConstituentFilter === 'neutral') return effect === 0;
            return true;
        });
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
    }, [indexConstituents, indexConstituentFilter, indexConstituentSort, indexImpactLevel]);

    const handleIndexConstituentPreset = useCallback((preset: 'all' | 'weight' | 'positive' | 'negative' | 'neutral') => {
        if (preset === 'all') {
            setIndexConstituentFilter('all');
            setIndexConstituentSort({ key: 'impact_abs', direction: 'desc' });
            return;
        }
        if (preset === 'weight') {
            setIndexConstituentFilter('all');
            setIndexConstituentSort({ key: 'weight_pct', direction: 'desc' });
            return;
        }
        if (preset === 'positive') {
            setIndexConstituentFilter((current) => {
                if (current === 'positive') {
                    setIndexConstituentSort({ key: 'impact_abs', direction: 'desc' });
                    return 'all';
                }
                setIndexConstituentSort({ key: 'point_effect', direction: 'desc' });
                return 'positive';
            });
            return;
        }
        if (preset === 'negative') {
            setIndexConstituentFilter((current) => {
                if (current === 'negative') {
                    setIndexConstituentSort({ key: 'impact_abs', direction: 'desc' });
                    return 'all';
                }
                setIndexConstituentSort({ key: 'point_effect', direction: 'asc' });
                return 'negative';
            });
            return;
        }
        setIndexConstituentFilter((current) => {
            if (current === 'neutral') {
                setIndexConstituentSort({ key: 'impact_abs', direction: 'desc' });
                return 'all';
            }
            setIndexConstituentSort({ key: 'weight_pct', direction: 'desc' });
            return 'neutral';
        });
    }, []);

    useEffect(() => {
        setIndexConstituentFilter('all');
        setIndexConstituentSort({ key: 'impact_abs', direction: 'desc' });
    }, [selectedIndex]);

    const indexConstituentWrapRef = useRef<HTMLDivElement>(null);
    const {
        startIndex: indexConstituentStartIndex,
        endIndex: indexConstituentEndIndex,
        topSpacerHeight: indexConstituentTopSpacerHeight,
        bottomSpacerHeight: indexConstituentBottomSpacerHeight,
        containerRef: indexConstituentTableWrapCallbackRef,
    } = useTableVirtualizer({
        totalRows: sortedIndexConstituents.length,
        rowHeight: isMobileViewport ? 95 : 38,
        overscan: 12,
        containerRef: indexConstituentWrapRef,
        resetDeps: [
            activeSection,
            selectedIndex,
            indexConstituentFilter,
            indexConstituentSort.key,
            indexConstituentSort.direction,
            isMobileViewport,
        ],
    });

    const visibleIndexConstituents = useMemo(() => {
        return sortedIndexConstituents.slice(indexConstituentStartIndex, indexConstituentEndIndex);
    }, [sortedIndexConstituents, indexConstituentStartIndex, indexConstituentEndIndex]);
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
            ? selectedIndex
                ? indexDetail
                    ? `${indexDetail.symbol} — ${indexDetail.label}`
                    : `${selectedIndex} Endeks Detayı`
                : 'Borsa İstanbul Endeksleri'
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

    const onCompanyClick = useCallback(
        (ticker: string) => {
            const normalizedTicker = String(ticker || '').trim().toUpperCase();
            if (!normalizedTicker) return;
            if (onOpenTicker) {
                onOpenTicker(normalizedTicker);
                return;
            }
            window.location.href = `/?ticker=${normalizedTicker}`;
        },
        [onOpenTicker]
    );

    const handleStockIndexChange = (index: MarketStockIndex) => {
        setStockIndex(index);
        onNavigateStockIndex?.(index);
    };

    const handleReturnModeChange = (mode: StockReturnMode) => {
        if (mode === returnMode) return;
        setReturnMode(mode);
        onNavigateReturnMode?.(mode);
    };

    const handleStockTableModeChange = (mode: StockTableMode) => {
        if (mode === stockTableMode) return;
        setStockTableMode(mode);
        setStockSort({ key: mode === 'fundamentals' ? 'fk' : 'company', direction: 'asc' });
    };

    const handleScreenerFilterChange = <K extends keyof StockScreenerFilterState,>(
        key: K,
        value: StockScreenerFilterState[K],
    ) => {
        setScreenerFilters((current) => ({ ...current, [key]: value }));
        setActiveScreenerPreset(null);
    };

    const handleClearScreenerRangeSpec = (
        minKey: StockScreenerRangeKey,
        maxKey: StockScreenerRangeKey,
    ) => {
        setScreenerFilters((c) => ({ ...c, [minKey]: '', [maxKey]: '' }));
        setActiveScreenerPreset(null);
    };

    const applyScreenerPreset = (presetId: StockScreenerPresetId) => {
        if (activeScreenerPreset === presetId) {
            setScreenerFilters({ ...EMPTY_STOCK_SCREENER_FILTERS });
            setActiveScreenerPreset(null);
            return;
        }
        const preset = STOCK_SCREENER_PRESETS.find((option) => option.id === presetId);
        if (!preset) return;
        setScreenerFilters({ ...EMPTY_STOCK_SCREENER_FILTERS, ...preset.filters });
        setActiveScreenerPreset(presetId);
    };

    const clearScreenerFilters = () => {
        setScreenerFilters({ ...EMPTY_STOCK_SCREENER_FILTERS });
        setActiveScreenerPreset(null);
    };

    const handleSelectIndex = (index: MarketIndexCode | null) => {
        setSelectedIndex(index);
        if (index) {
            setActiveSection('indices');
        }
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

    const handleStockCardGridDragOver = (event: ReactDragEvent<HTMLDivElement>) => {
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
        if (isMobileViewport) {
            const minDim = Math.min(tile.width, tile.height);
            if (tile.areaPct >= 14 && minDim >= 18) return '0.92rem';
            if (tile.areaPct >= 7 && minDim >= 12) return '0.74rem';
            if (tile.areaPct >= 3 && minDim >= 8.5) return '0.62rem';
            if (tile.areaPct >= 1.2 && minDim >= 5.5) return '0.52rem';
            return '0.44rem';
        }
        if (tile.areaPct >= 16 && tile.width >= 22 && tile.height >= 22) return '2rem';
        if (tile.areaPct >= 8) return '1.45rem';
        if (tile.areaPct >= 3.5) return '1rem';
        if (tile.areaPct >= 1.2) return '0.78rem';
        return '0.64rem';
    };

    return (
        <div className={`mn-layout${navCollapsed ? ' mn-nav-collapsed' : ''}`}>
            <MarketsNavigation
                collapsed={navCollapsed}
                activeSection={activeSection}
                activeStockSection="list"
                onCollapsedChange={setNavCollapsed}
                onSectionChange={handleSectionChange}
                onStockSectionChange={onNavigateStockSection}
                onFundSectionChange={onNavigateFundSection}
                onSelectTicker={onCompanyClick}
                onSelectFund={onOpenFund}
            />
            <div
                className={`markets-workspace${activeSection === 'markets' ? ' markets-workspace-with-rail' : ''}${navCollapsed ? ' is-nav-collapsed' : ''}${railCollapsed ? ' is-rail-collapsed' : ''}`}
                data-collapsed-menus={collapsedMenuCount}
            >
                <div className="market-page" ref={marketPageRef}>
                    {activeSection !== 'markets' && (
                        <header className="market-header">
                            <div className="market-title">
                                <h1>{pageTitle}</h1>
                                <p>{pageDescription}</p>
                            </div>
                        </header>
                    )}

                    {activeSection === 'markets' && (
                        <MarketOverviewSection
                            loading={loading}
                            error={error}
                            market={market}
                            isMobileViewport={isMobileViewport}
                            mobileIndexDetail={mobileIndexDetail}
                            mobileWatchlistSearchOpen={mobileWatchlistSearchOpen}
                            mobileWatchlistFundRows={mobileWatchlistFundRows}
                            stockQuoteRows={stocks?.rows ?? []}
                            watchlistItems={watchlist.items}
                            stockCardSymbols={stockCardSymbols}
                            stockCards={stockCards}
                            stockCardsLoading={stockCardsLoading}
                            stockCardsError={stockCardsError}
                            stockCardsBySymbol={stockCardsBySymbol}
                            stockCardPendingSet={stockCardPendingSet}
                            canAddStockCards={canAddStockCards}
                            stockCardPickerOpen={stockCardPickerOpen}
                            stockCardSearchTerm={stockCardSearchTerm}
                            stockCardCandidates={stockCardCandidates}
                            draggingStockCardSymbol={draggingStockCardSymbol}
                            onRetryLoadStats={() => loadStats()}
                            onSelectTicker={onCompanyClick}
                            onSelectIndex={handleSelectIndex}
                            onOpenFund={onOpenFund}
                            onAddWatchlistStock={(symbol) => watchlist.addItem({ kind: 'stock', symbol })}
                            onAddWatchlistFund={(fund) => {
                                const normalizedCode = normalizeWatchlistSymbol(fund.fund_code);
                                setMobileWatchlistFundRows((current) => ({ ...current, [normalizedCode]: fund }));
                                watchlist.addItem({ kind: 'fund', symbol: normalizedCode, label: fund.name });
                            }}
                            onRemoveWatchlistItem={(item) => watchlist.removeItem(item.kind, item.symbol)}
                            onMobileWatchlistSearchOpenChange={setMobileWatchlistSearchOpen}
                            onToggleStockCardPicker={() => setStockCardPickerOpen((open) => !open)}
                            onCloseStockCardPicker={() => setStockCardPickerOpen(false)}
                            onStockCardSearchTermChange={setStockCardSearchTerm}
                            onAddStockCard={handleAddStockCard}
                            onRetryStockCards={() => loadStockCards()}
                            onOpenStockCardPicker={() => setStockCardPickerOpen(true)}
                            onStockCardGridDragOver={handleStockCardGridDragOver}
                            onStockCardMoveStart={handleStockCardMoveStart}
                            onStockCardMoveOver={handleStockCardMoveOver}
                            onStockCardMoveEnd={handleStockCardMoveEnd}
                            onRemoveStockCard={handleRemoveStockCard}
                        />
                    )}

                    {activeSection === 'stocks' && (
                        <>
                            {onNavigateStockSection && (
                                <div className="mobile-subsection-switcher" role="navigation" aria-label="Hisse görünümleri">
                                    <button
                                        type="button"
                                        className="mobile-subsection-btn active"
                                        aria-pressed={true}
                                        onClick={() => onNavigateStockSection('list')}
                                    >
                                        Hisse Listesi
                                    </button>
                                    <button
                                        type="button"
                                        className="mobile-subsection-btn"
                                        aria-pressed={false}
                                        onClick={() => onNavigateStockSection('compare')}
                                    >
                                        Hisse Karşılaştırma
                                    </button>
                                </div>
                            )}
                            <StocksSection
                            stocks={stocks}
                            stocksLoading={stocksLoading}
                            stocksError={stocksError}
                            searchTerm={searchTerm}
                            stockIndex={stockIndex}
                            stockTableMode={stockTableMode}
                            returnMode={returnMode}
                            activeBenchmarkIndex={activeBenchmarkIndex}
                            stockSort={stockSort}
                            sortedStocks={sortedStocks}
                            visibleStocks={visibleStocks}
                            totalStocksCount={totalStocksCount}
                            isStocksFiltered={isStocksFiltered}
                            activeStockColumns={activeStockColumns}
                            stockTableTotalWidth={stockTableTotalWidth}
                            stockStartIndex={stockStartIndex}
                            stockTopSpacerHeight={stockTopSpacerHeight}
                            stockBottomSpacerHeight={stockBottomSpacerHeight}
                            stockTableWrapCallbackRef={stockTableWrapCallbackRef}
                            screenerFilters={screenerFilters}
                            screenerCategory={screenerCategory}
                            activeScreenerPreset={activeScreenerPreset}
                            screenerModalOpen={screenerModalOpen}
                            activeFilterCount={activeFilterCount}
                            isScreenerActive={isScreenerActive}
                            screenerCategoryCounts={screenerCategoryCounts}
                            screenerActiveChips={screenerActiveChips}
                            screenerData={screenerData}
                            onSearchTermChange={setSearchTerm}
                            onStockIndexChange={handleStockIndexChange}
                            onStockTableModeChange={handleStockTableModeChange}
                            onReturnModeChange={handleReturnModeChange}
                            onRefreshStocks={() => loadStocks(false, true, stockIndex)}
                            onStockSort={handleStockSort}
                            onCompanyClick={onCompanyClick}
                            onOpenScreenerModal={() => setScreenerModalOpen(true)}
                            onCloseScreenerModal={() => setScreenerModalOpen(false)}
                            onScreenerCategoryChange={setScreenerCategory}
                            onScreenerFilterChange={handleScreenerFilterChange}
                            onClearScreenerRangeSpec={handleClearScreenerRangeSpec}
                            onApplyScreenerPreset={applyScreenerPreset}
                            onClearScreenerFilters={clearScreenerFilters}
                        />
                        </>
                    )}

                    {activeSection === 'indices' && (
                        <IndicesSection
                            selectedIndex={selectedIndex}
                            indices={indices}
                            indicesLoading={indicesLoading}
                            indicesError={indicesError}
                            searchTerm={searchTerm}
                            indexGroup={indexGroup}
                            isIndicesFiltered={isIndicesFiltered}
                            indexSort={indexSort}
                            filteredIndices={filteredIndices}
                            sortedIndices={sortedIndices}
                            indexDetail={indexDetail}
                            indexDetailLoading={indexDetailLoading}
                            indexDetailError={indexDetailError}
                            indexImpactLevel={indexImpactLevel}
                            positiveConstituents={positiveConstituents}
                            neutralConstituents={neutralConstituents}
                            negativeConstituents={negativeConstituents}
                            sortedIndexConstituents={sortedIndexConstituents}
                            visibleIndexConstituents={visibleIndexConstituents}
                            indexConstituentSort={indexConstituentSort}
                            indexConstituentFilter={indexConstituentFilter}
                            indexConstituentStartIndex={indexConstituentStartIndex}
                            indexConstituentTopSpacerHeight={indexConstituentTopSpacerHeight}
                            indexConstituentBottomSpacerHeight={indexConstituentBottomSpacerHeight}
                            indexConstituentTableWrapCallbackRef={indexConstituentTableWrapCallbackRef}
                            maxHeatmapMovePct={maxHeatmapMovePct}
                            heatmapTiles={heatmapTiles}
                            netPointEffect={netPointEffect}
                            netImpactPct={netImpactPct}
                            onSearchTermChange={setSearchTerm}
                            onIndexGroupChange={setIndexGroup}
                            onRefreshIndices={() => loadIndices(false, true)}
                            onIndexSort={handleIndexSort}
                            onSelectIndex={handleSelectIndex}
                            onRetryIndexDetail={(index) => loadIndexDetail(false, false, index)}
                            onIndexConstituentSort={handleIndexConstituentSort}
                            onIndexConstituentPreset={handleIndexConstituentPreset}
                            onCompanyClick={onCompanyClick}
                            getTreemapTileScale={getTreemapTileScale}
                            getHeatmapTileFontSize={getHeatmapTileFontSize}
                        />
                    )}
                </div>
                {activeSection === 'markets' && (
                    <MarketWatchRail
                        xu100Rows={market?.rows ?? []}
                        stockQuoteRows={stocks?.rows ?? []}
                        stockCardItems={stockCards?.items ?? []}
                        onSelectTicker={onCompanyClick}
                        onSelectFund={(fundCode) => onOpenFund?.(fundCode)}
                        mobilePanelOpen={isMobileViewport ? mobileMarketPanelOpen : undefined}
                        onMobilePanelOpenChange={isMobileViewport ? setMobileMarketPanelOpen : undefined}
                        onCollapseChange={setRailCollapsed}
                    />
                )}
            </div>
        </div>
    );
}
