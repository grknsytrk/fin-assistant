import {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
    type CSSProperties,
    type PointerEvent as ReactPointerEvent,
} from 'react';
import { ChevronDown, Search, X } from 'lucide-react';
import { apiClient } from '../../../api/client';
import type {
    FundPricePoint,
    FundSummary,
    FxQuote,
    MarketComparisonHistoryResponse,
    MarketIndexListRow,
    MarketUniverseRow,
} from '../../../api/types';
import SymbolLogo from '../../../components/SymbolLogo';
import {
    compactCompareSearch,
    comparisonAssetId,
    comparisonAssetKindLabel,
    comparisonLogoKind,
    formatPct,
    fundToComparisonAsset,
    fxToComparisonAsset,
    indexToComparisonAsset,
    normalizeCompareSearch,
    normalizeCompareSymbol,
    periodReturnFromAsset,
    returnFromDatedValues,
    returnFromFundPoints,
    searchCachedFundCatalog,
    stockToComparisonAsset,
} from '../formatters';
import {
    DEFAULT_COMPARISON_IDS,
    DETAIL_RETURN_PERIODS,
    GLOBAL_COMPARISON_INDICES,
    STOCK_SEARCH_LABELS,
    type ComparisonAsset,
    type ComparisonPeriod,
    type FundComparisonControlSurface,
    type FundComparisonState,
} from '../types';
import './FundReturnComparison.css';

const GLOBAL_COMPARISON_INDEX_ASSETS: ComparisonAsset[] = GLOBAL_COMPARISON_INDICES.map((spec) => ({
    id: comparisonAssetId('index', spec.symbol),
    kind: 'index',
    symbol: spec.symbol,
    displaySymbol: spec.displaySymbol,
    label: spec.label,
    logoName: spec.label,
    returns: {},
}));

const GLOBAL_INDEX_SEARCH_ALIASES: Record<string, string[]> = Object.fromEntries(
    GLOBAL_COMPARISON_INDICES.map((spec) => [spec.symbol, spec.aliases]),
);

export function useFundComparisonState({
    selectedFund,
    baseReturns,
    baseReturnsReady,
    funds,
    openSearchSignal,
}: {
    selectedFund: FundSummary | null;
    baseReturns: FundSummary['period_returns'] | null;
    baseReturnsReady: boolean;
    funds: FundSummary[];
    openSearchSignal: number;
}): FundComparisonState {
    const [period, setPeriod] = useState<ComparisonPeriod>('1m');
    const [selectedIds, setSelectedIds] = useState<string[]>(() => [...DEFAULT_COMPARISON_IDS]);
    const [query, setQuery] = useState('');
    const [searchOpen, setSearchOpen] = useState(false);
    const [searchSurface, setSearchSurface] = useState<FundComparisonControlSurface | null>(null);
    const [fundSearchRows, setFundSearchRows] = useState<FundSummary[]>([]);
    const [stockRows, setStockRows] = useState<MarketUniverseRow[]>([]);
    const [indexRows, setIndexRows] = useState<MarketIndexListRow[]>([]);
    const [fxRows, setFxRows] = useState<FxQuote[]>([]);
    const [fundSearchLoading, setFundSearchLoading] = useState(false);
    const [stocksLoading, setStocksLoading] = useState(false);
    const [indicesLoading, setIndicesLoading] = useState(false);
    const [fxLoading, setFxLoading] = useState(false);
    const inputRef = useRef<HTMLInputElement | null>(null);

    const selectedFundCode = normalizeCompareSymbol(selectedFund?.fund_code);
    const baseAsset = useMemo<ComparisonAsset | null>(() => {
        if (!selectedFund || !selectedFundCode) return null;
        return {
            id: comparisonAssetId('fund', selectedFundCode),
            kind: 'fund',
            symbol: selectedFundCode,
            label: selectedFund.name,
            logoName: selectedFund.founder_company || selectedFund.manager_company || selectedFund.name,
            returns: {
                '1w': baseReturnsReady ? (baseReturns?.['1w'] ?? selectedFund.period_returns?.['1w'] ?? null) : null,
                '1m': baseReturnsReady ? (baseReturns?.['1m'] ?? selectedFund.period_returns?.['1m'] ?? null) : null,
                '3m': baseReturnsReady ? (baseReturns?.['3m'] ?? selectedFund.period_returns?.['3m'] ?? null) : null,
                '6m': baseReturnsReady ? (baseReturns?.['6m'] ?? selectedFund.period_returns?.['6m'] ?? null) : null,
                ytd: baseReturnsReady ? (baseReturns?.ytd ?? selectedFund.period_returns?.ytd ?? null) : null,
                '1y': baseReturnsReady ? (baseReturns?.['1y'] ?? selectedFund.period_returns?.['1y'] ?? null) : null,
            },
        };
    }, [baseReturns, baseReturnsReady, selectedFund, selectedFundCode]);

    useEffect(() => {
        setSelectedIds([...DEFAULT_COMPARISON_IDS]);
        setQuery('');
        setSearchOpen(false);
        setSearchSurface(null);
    }, [selectedFundCode]);

    useEffect(() => {
        if (!selectedFundCode || !searchOpen) return;
        let alive = true;
        setIndicesLoading(true);
        setFxLoading(true);
        Promise.allSettled([apiClient.marketIndices(), apiClient.marketFx()])
            .then(([indicesResult, fxResult]) => {
                if (!alive) return;
                setIndexRows(indicesResult.status === 'fulfilled' ? indicesResult.value.rows || [] : []);
                setFxRows(fxResult.status === 'fulfilled' ? fxResult.value.items || [] : []);
            })
            .finally(() => {
                if (alive) {
                    setIndicesLoading(false);
                    setFxLoading(false);
                }
            });
        return () => {
            alive = false;
        };
    }, [searchOpen, selectedFundCode]);

    const openSearch = useCallback((surface: FundComparisonControlSurface = 'returns') => {
        setSearchSurface(surface);
        setSearchOpen(true);
        window.setTimeout(() => inputRef.current?.focus(), 0);
    }, []);

    useEffect(() => {
        if (openSearchSignal <= 0 || !selectedFundCode) return;
        openSearch('returns');
    }, [openSearch, openSearchSignal, selectedFundCode]);

    useEffect(() => {
        const trimmed = query.trim();
        if (!searchOpen || !trimmed) {
            setFundSearchRows([]);
            setFundSearchLoading(false);
            return;
        }
        const cachedMatches = searchCachedFundCatalog(trimmed, 10);
        if (cachedMatches !== null) {
            setFundSearchRows(cachedMatches);
            setFundSearchLoading(false);
            return;
        }
        let alive = true;
        const controller = new AbortController();
        setFundSearchLoading(true);
        const timer = window.setTimeout(() => {
            apiClient
                .fundSearch(trimmed, 10, { signal: controller.signal })
                .then((payload) => {
                    if (alive) setFundSearchRows(payload.rows || []);
                })
                .catch((error) => {
                    if ((error as Error)?.name === 'AbortError') return;
                    if (alive) setFundSearchRows([]);
                })
                .finally(() => {
                    if (alive) setFundSearchLoading(false);
                });
        }, 160);
        return () => {
            alive = false;
            window.clearTimeout(timer);
            controller.abort();
        };
    }, [query, searchOpen]);

    useEffect(() => {
        const trimmed = query.trim();
        if (!searchOpen || !trimmed) {
            setStockRows([]);
            setStocksLoading(false);
            return;
        }
        let alive = true;
        const controller = new AbortController();
        setStocksLoading(true);
        const timer = window.setTimeout(() => {
            apiClient
                .marketStockSearch({ q: trimmed, index: 'XUTUM', limit: 20, signal: controller.signal })
                .then((payload) => {
                    if (alive) setStockRows(payload.rows || []);
                })
                .catch((error) => {
                    if ((error as Error)?.name === 'AbortError') return;
                    if (alive) setStockRows([]);
                })
                .finally(() => {
                    if (alive) setStocksLoading(false);
                });
        }, 160);
        return () => {
            alive = false;
            window.clearTimeout(timer);
            controller.abort();
        };
    }, [query, searchOpen]);

    const allAssets = useMemo(() => {
        if (!selectedFundCode) return [];
        const currentFundId = comparisonAssetId('fund', selectedFundCode);
        const mergedFunds = new Map<string, FundSummary>();
        for (const row of funds) {
            mergedFunds.set(normalizeCompareSymbol(row.fund_code), row);
        }
        for (const row of fundSearchRows) {
            mergedFunds.set(normalizeCompareSymbol(row.fund_code), row);
        }
        const fundAssets = [...mergedFunds.values()]
            .filter((row) => comparisonAssetId('fund', row.fund_code) !== currentFundId)
            .map(fundToComparisonAsset);
        const dynamicIndexAssets = indexRows.map(indexToComparisonAsset);
        const existingIndexIds = new Set(dynamicIndexAssets.map((asset) => asset.id));
        const globalIndexAssets = GLOBAL_COMPARISON_INDEX_ASSETS.filter((asset) => !existingIndexIds.has(asset.id));
        return [
            ...fundAssets,
            ...stockRows.map(stockToComparisonAsset),
            ...dynamicIndexAssets,
            ...globalIndexAssets,
            ...fxRows.map(fxToComparisonAsset).filter((asset): asset is ComparisonAsset => Boolean(asset)),
        ];
    }, [fundSearchRows, funds, fxRows, indexRows, selectedFundCode, stockRows]);

    const assetById = useMemo(() => new Map(allAssets.map((asset) => [asset.id, asset])), [allAssets]);
    const selectedAssets = useMemo(
        () => selectedIds.map((id) => assetById.get(id)).filter((asset): asset is ComparisonAsset => Boolean(asset)),
        [assetById, selectedIds],
    );

    const searchResults = useMemo(() => {
        const normalizedQuery = normalizeCompareSearch(query);
        const compactQuery = compactCompareSearch(query);
        if (!normalizedQuery) return [];
        const selectedSet = new Set(selectedIds);
        return allAssets
            .filter((asset) => !selectedSet.has(asset.id))
            .map((asset) => {
                const searchParts = [
                    asset.symbol,
                    asset.displaySymbol || '',
                    asset.label,
                    comparisonAssetKindLabel(asset.kind),
                    ...(asset.kind === 'stock' ? STOCK_SEARCH_LABELS[asset.symbol] || [] : []),
                    ...(asset.kind === 'index' ? GLOBAL_INDEX_SEARCH_ALIASES[asset.symbol] || [] : []),
                ];
                const normalizedParts = searchParts.map(normalizeCompareSearch);
                const compactParts = searchParts.map(compactCompareSearch);
                const starts = normalizedParts.some((item) => item.startsWith(normalizedQuery))
                    || compactParts.some((item) => item.startsWith(compactQuery));
                const includes = normalizedParts.some((item) => item.includes(normalizedQuery))
                    || compactParts.some((item) => item.includes(compactQuery));
                if (!starts && !includes) return null;
                return { asset, score: starts ? 0 : 1 };
            })
            .filter((item): item is { asset: ComparisonAsset; score: number } => Boolean(item))
            .sort((a, b) => a.score - b.score || a.asset.symbol.localeCompare(b.asset.symbol, 'tr'))
            .slice(0, 8)
            .map((item) => item.asset);
    }, [allAssets, query, selectedIds]);

    const addAsset = useCallback((asset: ComparisonAsset) => {
        setSelectedIds((current) => current.includes(asset.id) ? current : [...current, asset.id]);
        setQuery('');
        setSearchOpen(false);
    }, []);

    const removeAsset = useCallback((assetId: string) => {
        setSelectedIds((current) => current.filter((id) => id !== assetId));
    }, []);

    return {
        baseAsset,
        selectedAssets,
        selectedIds,
        period,
        setPeriod,
        query,
        setQuery,
        searchOpen,
        setSearchOpen,
        searchSurface,
        searchResults,
        addAsset,
        removeAsset,
        openSearch,
        fundSearchLoading,
        stocksLoading,
        indicesLoading,
        fxLoading,
        inputRef,
    };
}

export function FundComparisonControls({
    comparison,
    surface,
    className,
}: {
    comparison: FundComparisonState;
    surface: FundComparisonControlSurface;
    className?: string;
}) {
    const {
        selectedAssets,
        query,
        setQuery,
        searchOpen,
        setSearchOpen,
        searchSurface,
        searchResults,
        addAsset,
        removeAsset,
        openSearch,
        fundSearchLoading,
        stocksLoading,
        indicesLoading,
        fxLoading,
        inputRef,
    } = comparison;
    const surfaceOpen = searchOpen && searchSurface === surface;

    return (
        <div className={`fund-comparison-controls${className ? ` ${className}` : ''}`}>
            <div className={`fund-comparison-search-wrap${surfaceOpen ? ' is-open' : ''}`}>
                {surfaceOpen ? (
                    <div className="fund-comparison-search-active">
                        <Search size={15} aria-hidden="true" />
                        <input
                            ref={inputRef}
                            value={query}
                            onChange={(event) => setQuery(event.target.value)}
                            onKeyDown={(event) => {
                                if (event.key === 'Escape') {
                                    setSearchOpen(false);
                                    setQuery('');
                                }
                                if (event.key === 'Enter' && searchResults[0]) {
                                    event.preventDefault();
                                    addAsset(searchResults[0]);
                                }
                            }}
                            placeholder="Karşılaştır"
                            aria-label="Fon, hisse, endeks veya döviz ara"
                        />
                        <button
                            type="button"
                            onClick={() => {
                                setSearchOpen(false);
                                setQuery('');
                            }}
                            aria-label="Aramayı kapat"
                        >
                            <X size={15} aria-hidden="true" />
                        </button>
                    </div>
                ) : (
                    <button type="button" className="fund-comparison-search-button" onClick={() => openSearch(surface)}>
                        <Search size={15} aria-hidden="true" />
                        Karşılaştırma
                    </button>
                )}
                {surfaceOpen && (
                    <div className="fund-comparison-results">
                        {query.trim() ? (
                            searchResults.length ? (
                                searchResults.map((asset) => (
                                    <button key={asset.id} type="button" onClick={() => addAsset(asset)}>
                                        <SymbolLogo
                                            symbol={asset.symbol === 'NASDAQ100' ? 'NASDAQ' : asset.symbol}
                                            name={asset.logoName || asset.label}
                                            kind={comparisonLogoKind(asset.kind)}
                                            logoUrl={asset.logoUrl}
                                            size="sm"
                                        />
                                        <strong>{asset.displaySymbol || asset.symbol}</strong>
                                        <span>{asset.label}</span>
                                    </button>
                                ))
                            ) : (
                                <div className="fund-comparison-result-empty">
                                    {fundSearchLoading || stocksLoading || indicesLoading || fxLoading ? 'Aranıyor...' : 'Sonuç bulunamadı.'}
                                </div>
                            )
                        ) : (
                            <div className="fund-comparison-result-empty">Fon, hisse, endeks veya döviz yazın.</div>
                        )}
                    </div>
                )}
            </div>

            <div className="fund-comparison-chips" aria-label="Seçili karşılaştırmalar">
                {selectedAssets.map((asset) => (
                    <button
                        key={asset.id}
                        type="button"
                        onClick={() => removeAsset(asset.id)}
                        title={`${asset.displaySymbol || asset.symbol} karşılaştırmadan çıkar`}
                    >
                        <i />
                        {asset.displaySymbol || asset.symbol}
                    </button>
                ))}
            </div>
        </div>
    );
}

export function FundReturnComparison({
    comparison,
    containerRef,
    comparisonHistory,
    basePerformancePoints,
}: {
    comparison: FundComparisonState;
    containerRef: { current: HTMLDivElement | null };
    comparisonHistory: MarketComparisonHistoryResponse | null;
    basePerformancePoints: FundPricePoint[];
}) {
    const [hover, setHover] = useState<{ assetId: string; x: number; y: number; width: number; height: number } | null>(null);
    const {
        baseAsset,
        selectedAssets,
        selectedIds,
        period,
        setPeriod,
    } = comparison;

    const chartItems = useMemo(() => {
        if (!baseAsset) return [];
        const comparisonHistoryById = new Map((comparisonHistory?.assets || []).map((asset) => [asset.id, asset]));
        const items = [baseAsset, ...selectedAssets]
            .map((asset) => ({
                asset,
                value: asset.id === baseAsset.id
                    ? returnFromFundPoints(basePerformancePoints, period) ?? periodReturnFromAsset(asset, period)
                    : returnFromDatedValues(comparisonHistoryById.get(asset.id)?.points, period)
                        ?? periodReturnFromAsset(asset, period),
            }))
            .filter((item) => item.value != null);
        return items.sort((a, b) => {
            const valueDiff = Number(b.value) - Number(a.value);
            if (valueDiff !== 0) return valueDiff;
            return (a.asset.displaySymbol || a.asset.symbol).localeCompare(b.asset.displaySymbol || b.asset.symbol, 'tr');
        });
    }, [baseAsset, basePerformancePoints, comparisonHistory, period, selectedAssets]);

    useEffect(() => {
        setHover(null);
    }, [period, selectedIds]);

    if (!baseAsset) {
        return null;
    }

    const chartWidth = 1040;
    const chartHeight = 340;
    const padding = { top: 28, right: 22, bottom: 58, left: 64 };
    const plotWidth = chartWidth - padding.left - padding.right;
    const plotHeight = chartHeight - padding.top - padding.bottom;
    const values = chartItems.map((item) => Number(item.value));
    const rawMax = Math.max(0, ...values);
    const rawMin = Math.min(0, ...values);
    const maxValue = Math.max(1, Math.ceil(rawMax + Math.max(1, rawMax - rawMin) * 0.08));
    const minValue = rawMin < 0 ? Math.floor(rawMin - Math.max(1, rawMax - rawMin) * 0.08) : 0;
    const valueSpan = Math.max(1, maxValue - minValue);
    const yFor = (value: number) => padding.top + ((maxValue - value) / valueSpan) * plotHeight;
    const zeroY = yFor(0);
    const barGap = chartItems.length > 1 ? 46 : 0;
    const rawBarWidth = chartItems.length ? (plotWidth - barGap * (chartItems.length - 1)) / chartItems.length : 0;
    const barWidth = Math.max(46, Math.min(132, rawBarWidth));
    const totalBarsWidth = chartItems.length * barWidth + Math.max(0, chartItems.length - 1) * barGap;
    const startX = padding.left + Math.max(0, (plotWidth - totalBarsWidth) / 2);
    const ticks = Array.from({ length: 6 }, (_, index) => minValue + (valueSpan * index) / 5);
    const hoverItem = hover ? chartItems.find((item) => item.asset.id === hover.assetId) || null : null;
    const tooltipAlignX = hover && hover.x > hover.width - 180 ? 'left' : 'right';
    const tooltipAlignY = hover && hover.y > hover.height - 110 ? 'up' : 'down';

    const handleBarPointerMove = (event: ReactPointerEvent<SVGRectElement>, assetId: string) => {
        const rect = event.currentTarget.ownerSVGElement?.getBoundingClientRect();
        if (!rect) return;
        setHover({
            assetId,
            x: event.clientX - rect.left,
            y: event.clientY - rect.top,
            width: rect.width,
            height: rect.height,
        });
    };

    const handleSvgPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const target = event.target as Element | null;
        if (!target || !(target instanceof SVGElement)) return;
        // Only keep hover state while pointer is genuinely over a bar.
        if (!target.classList.contains('fund-comparison-bar')) {
            setHover(null);
        }
    };

    return (
        <section className="fund-comparison-panel" ref={containerRef} aria-label="Getiri karşılaştırma">
            <div className="fund-comparison-head">
                <h2>Getiri Karşılaştırma</h2>
                <label className="fund-comparison-period">
                    <select value={period} onChange={(event) => setPeriod(event.target.value as ComparisonPeriod)}>
                        {DETAIL_RETURN_PERIODS.map((item) => (
                            <option key={item.key} value={item.key}>{item.label}</option>
                        ))}
                    </select>
                    <ChevronDown size={15} aria-hidden="true" />
                </label>
            </div>

            <div className="fund-comparison-chart-wrap">
                {chartItems.length ? (
                    <svg
                        className="fund-comparison-chart"
                        viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                        role="img"
                        aria-label="Seçili varlıkların dönem getirileri"
                        onPointerMove={handleSvgPointerMove}
                        onPointerLeave={() => setHover(null)}
                    >
                        {ticks.map((tick) => {
                            const y = yFor(tick);
                            return (
                                <g key={tick}>
                                    <line x1={padding.left} x2={chartWidth - padding.right} y1={y} y2={y} className="fund-comparison-gridline" />
                                    <text x={padding.left - 12} y={y + 4} className="fund-chart-axis" textAnchor="end">
                                        {formatPct(tick).replace('+', '')}
                                    </text>
                                </g>
                            );
                        })}
                        <line x1={padding.left} x2={chartWidth - padding.right} y1={zeroY} y2={zeroY} className="fund-comparison-zero" />
                        {chartItems.map(({ asset, value }, index) => {
                            const numericValue = Number(value);
                            const x = startX + index * (barWidth + barGap);
                            const y = numericValue >= 0 ? yFor(numericValue) : zeroY;
                            const height = Math.max(2, Math.abs(zeroY - yFor(numericValue)));
                            const isBase = asset.id === baseAsset.id;
                            const isActive = hover?.assetId === asset.id;
                            const displaySymbol = asset.displaySymbol || asset.symbol;
                            return (
                                <g
                                    key={asset.id}
                                    className="fund-comparison-bar-group"
                                    style={{ transform: `translateX(${x}px)` } as CSSProperties}
                                >
                                    <rect
                                        x={0}
                                        y={y}
                                        width={barWidth}
                                        height={height}
                                        className={[
                                            'fund-comparison-bar',
                                            isBase ? 'is-base' : '',
                                            isActive ? 'is-active' : '',
                                            hover && !isActive ? 'is-muted' : '',
                                            numericValue < 0 ? 'is-negative' : '',
                                        ].filter(Boolean).join(' ')}
                                        style={{ y: `${y}px`, height: `${height}px` } as CSSProperties}
                                        onPointerMove={(event) => handleBarPointerMove(event, asset.id)}
                                        onPointerEnter={(event) => handleBarPointerMove(event, asset.id)}
                                    />
                                    <text
                                        x={barWidth / 2}
                                        y={numericValue >= 0 ? y - 8 : y + height + 18}
                                        className="fund-comparison-value-label"
                                        textAnchor="middle"
                                        style={{ y: `${numericValue >= 0 ? y - 8 : y + height + 18}px` } as CSSProperties}
                                    >
                                        {formatPct(numericValue).replace('+', '')}
                                    </text>
                                    <text
                                        x={barWidth / 2}
                                        y={chartHeight - 18}
                                        className="fund-comparison-x-label"
                                        textAnchor="middle"
                                        transform={`rotate(-32 ${barWidth / 2} ${chartHeight - 18})`}
                                    >
                                        {displaySymbol}
                                    </text>
                                </g>
                            );
                        })}
                    </svg>
                ) : (
                    <div className="fund-comparison-empty">Bu dönem için karşılaştırılabilir getiri yok.</div>
                )}
                {hover && hoverItem ? (
                    <div
                        className={[
                            'fund-comparison-tooltip',
                            tooltipAlignX === 'left' ? 'align-left' : '',
                            tooltipAlignY === 'up' ? 'align-up' : '',
                        ].filter(Boolean).join(' ')}
                        style={{ left: `${hover.x}px`, top: `${hover.y}px` }}
                    >
                        <span>{hoverItem.asset.displaySymbol || hoverItem.asset.symbol}</span>
                        <strong>{formatPct(hoverItem.value)}</strong>
                    </div>
                ) : null}
            </div>

            <FundComparisonControls comparison={comparison} surface="returns" />
        </section>
    );
}
