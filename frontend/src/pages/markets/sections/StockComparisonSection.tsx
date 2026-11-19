import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { ChevronDown, ChevronRight, Layers, Plus, Search, X } from 'lucide-react';
import { Link, useSearchParams } from 'react-router-dom';
import { apiClient } from '../../../api/client';
import type {
    MarketComparisonHistoryPoint,
    MarketStockRow,
    MarketStocksResponse,
    MarketUniverseRow,
} from '../../../api/types';
import SymbolLogo from '../../../components/SymbolLogo';
import { buildDocumentTitle, useDocumentTitle } from '../../../hooks/useDocumentTitle';
import { toStockDetail } from '../../../routing/routes';
import { hasFundRangeStartCoverage } from '../../../utils/fundPresentation';
import {
    chartDateMs,
    compactCompareSearch,
    finiteNumber,
    formatChartDate,
    formatCompactCurrency,
    formatDate,
    formatPct,
    formatPlainPct,
    nearestChartPoint,
    normalizeCompareSearch,
    pctClass,
    rangeStartDate,
    returnBetween,
} from '../../funds/formatters';
import {
    FUND_COMPARE_COLORS,
    FUND_COMPARE_RANGES,
    STOCK_SEARCH_LABELS,
    type FundCompareCell,
    type FundCompareRange,
    type FundCompareTableRow,
} from '../../funds/types';
import './StockComparisonSection.css';

const STOCK_COMPARE_MAX = 4;
const STOCK_COMPARE_ALL_START_ISO = '2000-01-01';

function stockSummaryShell(ticker: string): MarketStockRow {
    const normalized = ticker.trim().toUpperCase();
    return {
        company: normalized,
        symbol: normalized,
        name: STOCK_SEARCH_LABELS[normalized]?.[0] || normalized,
        latest_quarter: null,
        has_kap_cache: false,
        price: null,
        price_currency: 'TRY',
        change: null,
        change_pct: null,
        price_as_of: null,
        market_cap: null,
        volume: null,
        return_1w_pct: null,
        return_1m_pct: null,
        return_3m_pct: null,
        return_6m_pct: null,
        return_ytd_pct: null,
        return_1y_pct: null,
    };
}

function universeRowToStockShell(row: MarketUniverseRow): MarketStockRow {
    const ticker = String(row.symbol || row.company || '').trim().toUpperCase();
    return {
        ...stockSummaryShell(ticker),
        ...row,
        company: row.company || ticker,
        symbol: row.symbol || ticker,
    };
}

function stockCompareRangeStart(range: FundCompareRange, endIso: string): string {
    if (range === 'all') return STOCK_COMPARE_ALL_START_ISO;
    return rangeStartDate(range, endIso, '');
}

function comparisonCell(
    primary: string | number | null | undefined,
    secondary?: string | null,
    className?: string,
): FundCompareCell {
    return {
        primary: primary == null || primary === '' ? '-' : String(primary),
        secondary: secondary || undefined,
        className,
    };
}

function formatStockMoney(value: number | null | undefined, currency?: string | null): string {
    const num = finiteNumber(value);
    if (num == null) return '-';
    const normalizedCurrency = String(currency || 'TRY').trim().toUpperCase();
    const prefix = !normalizedCurrency || normalizedCurrency === 'TRY' ? '₺' : `${normalizedCurrency} `;
    return `${prefix}${num.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatMultiple(value: number | null | undefined): string {
    const num = finiteNumber(value);
    if (num == null) return '-';
    return `${num.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}x`;
}

function sortHistoryPoints(points: MarketComparisonHistoryPoint[] | undefined): MarketComparisonHistoryPoint[] {
    if (!points?.length) return [];
    const byDate = new Map<string, MarketComparisonHistoryPoint>();
    for (const point of points) {
        const value = Number(point.value);
        if (!point.date || !Number.isFinite(value) || value <= 0) continue;
        byDate.set(point.date, { date: point.date, value });
    }
    return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
}

function nearestPointOnOrAfter(
    points: MarketComparisonHistoryPoint[],
    startIso: string,
): MarketComparisonHistoryPoint | null {
    return points.find((point) => point.date >= startIso) || null;
}

function returnForStockRange(
    points: MarketComparisonHistoryPoint[],
    range: FundCompareRange,
    endIso: string,
): number | null {
    const ordered = sortHistoryPoints(points);
    if (ordered.length < 2) return null;
    const latest = ordered[ordered.length - 1];
    const startIso = stockCompareRangeStart(range, endIso);
    const base = range === 'all' ? ordered[0] : nearestPointOnOrAfter(ordered, startIso);
    if (range !== 'all' && base) {
        if (!hasFundRangeStartCoverage(startIso, base.date)) return null;
    }
    return returnBetween(finiteNumber(latest.value), finiteNumber(base?.value));
}

function summaryReturnForStockRange(stock: MarketStockRow, range: FundCompareRange): number | null {
    if (range === '1w') return finiteNumber(stock.return_1w_pct);
    if (range === '1m') return finiteNumber(stock.return_1m_pct);
    if (range === '3m') return finiteNumber(stock.return_3m_pct);
    if (range === '6m') return finiteNumber(stock.return_6m_pct);
    if (range === 'ytd') return finiteNumber(stock.return_ytd_pct);
    if (range === '1y') return finiteNumber(stock.return_1y_pct);
    return null;
}

function cagrForStockRange(
    points: MarketComparisonHistoryPoint[],
    years: number,
    endIso: string,
): number | null {
    const ordered = sortHistoryPoints(points);
    if (ordered.length < 2) return null;
    const latest = ordered[ordered.length - 1];
    const base = nearestPointOnOrAfter(ordered, rangeStartDate(years === 3 ? '3y' : '5y', endIso, ''));
    const latestPrice = finiteNumber(latest.value);
    const basePrice = finiteNumber(base?.value);
    if (latestPrice == null || basePrice == null || basePrice <= 0 || latestPrice <= 0) return null;
    const baseTs = new Date(base?.date || '').getTime();
    const latestTs = new Date(latest.date).getTime();
    const actualYears = (latestTs - baseTs) / (365.25 * 24 * 60 * 60 * 1000);
    if (!Number.isFinite(actualYears) || actualYears < years * 0.78) return null;
    return (Math.pow(latestPrice / basePrice, 1 / actualYears) - 1) * 100;
}

function historicalMultipleSecondary(
    avg: number | null | undefined,
    diffPct: number | null | undefined,
): string | undefined {
    const formattedAvg = formatMultiple(avg);
    if (formattedAvg === '-') return undefined;
    const diff = finiteNumber(diffPct);
    return diff == null ? `Ort: ${formattedAvg}` : `Ort: ${formattedAvg} (${formatPct(diff)})`;
}

function StockCompareLineChart({
    selectedStocks,
    historyByTicker,
    range,
}: {
    selectedStocks: MarketStockRow[];
    historyByTicker: Record<string, MarketComparisonHistoryPoint[]>;
    range: FundCompareRange;
}) {
    const width = 1180;
    const height = 420;
    const padding = { top: 30, right: 78, bottom: 52, left: 22 };
    const chartEndIso = selectedStocks.find((stock) => stock.price_as_of)?.price_as_of?.slice(0, 10)
        || new Date().toISOString().slice(0, 10);
    const chartStartIso = stockCompareRangeStart(range, chartEndIso);
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const rawSeries = selectedStocks.map((stock, index) => {
        const points = sortHistoryPoints(historyByTicker[stock.company])
            .filter((point) => range === 'all' || point.date >= chartStartIso);
        const base = points.find((point) => Number(point.value) > 0);
        return {
            stock,
            color: FUND_COMPARE_COLORS[index % FUND_COMPARE_COLORS.length],
            points: base?.value
                ? points.map((point) => ({
                    date: point.date,
                    value: ((Number(point.value) / Number(base.value)) - 1) * 100,
                }))
                : [],
        };
    }).filter((series) => series.points.length >= 2);
    const allValues = rawSeries.flatMap((series) => series.points.map((point) => point.value));
    const seriesKey = `${range}-${rawSeries.map((series) => series.stock.company).join('-')}`;
    const [hoverIso, setHoverIso] = useState<string | null>(null);
    const [hoverPointer, setHoverPointer] = useState<{ x: number; y: number } | null>(null);

    useEffect(() => {
        setHoverIso(null);
        setHoverPointer(null);
    }, [seriesKey]);

    if (!rawSeries.length || !allValues.length) {
        return <div className="fund-compare-chart-empty">Grafik için yeterli performans verisi yok.</div>;
    }

    const dates = Array.from(new Set(rawSeries.flatMap((series) => series.points.map((point) => point.date)))).sort();
    const startTs = chartDateMs(dates[0]);
    const endTs = Math.max(startTs + 1, chartDateMs(dates[dates.length - 1]));
    const rawMin = Math.min(0, ...allValues);
    const rawMax = Math.max(0, ...allValues);
    const rawSpan = Math.max(1, rawMax - rawMin);
    const minValue = rawMin - rawSpan * 0.08;
    const maxValue = rawMax + rawSpan * 0.08;
    const span = Math.max(1, maxValue - minValue);
    const xForDate = (date: string) => padding.left + ((chartDateMs(date) - startTs) / (endTs - startTs)) * plotWidth;
    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;
    const yTicks = Array.from({ length: 5 }, (_, index) => maxValue - (span * index) / 4);
    const tickDates = Array.from(new Set([
        dates[0],
        dates[Math.floor((dates.length - 1) * 0.25)],
        dates[Math.floor((dates.length - 1) * 0.5)],
        dates[Math.floor((dates.length - 1) * 0.75)],
        dates[dates.length - 1],
    ].filter(Boolean)));

    const clipId = `stock-compare-clip-${seriesKey}`;

    const pointerFromEvent = (event: ReactPointerEvent<SVGSVGElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return null;
        const x = ((event.clientX - rect.left) / rect.width) * width;
        const y = ((event.clientY - rect.top) / rect.height) * height;
        return {
            x: Math.min(Math.max(x, 0), width),
            y: Math.min(Math.max(y, 0), height),
        };
    };

    const closestDateForX = (x: number): string | null => {
        if (!dates.length) return null;
        const ratio = endTs === startTs ? 0 : (x - padding.left) / Math.max(1, plotWidth);
        const targetTs = startTs + Math.min(Math.max(ratio, 0), 1) * (endTs - startTs);
        let best = dates[0];
        let bestDiff = Math.abs(chartDateMs(dates[0]) - targetTs);
        for (let i = 1; i < dates.length; i += 1) {
            const diff = Math.abs(chartDateMs(dates[i]) - targetTs);
            if (diff < bestDiff) {
                best = dates[i];
                bestDiff = diff;
            }
        }
        return best;
    };

    const hoverPoint = hoverIso
        ? rawSeries
            .map((series) => nearestChartPoint(series.points, hoverIso))
            .find((point) => point != null) ?? null
        : null;
    const tooltipRows = hoverPoint
        ? rawSeries
            .map((series) => {
                const point = nearestChartPoint(series.points, hoverPoint.date);
                return point ? { series, point } : null;
            })
            .filter((item): item is { series: typeof rawSeries[number]; point: { date: string; value: number } } => Boolean(item))
        : [];

    const tooltipDate = tooltipRows.length ? tooltipRows[0].point.date : null;
    const tooltipX0 = tooltipDate ? xForDate(tooltipDate) : null;
    const tooltipWidth = 196;
    const tooltipHeight = Math.max(54, 28 + tooltipRows.length * 18);
    const tooltipGap = 14;
    const tooltipAnchorX = hoverPointer?.x ?? tooltipX0;
    const tooltipAnchorY = hoverPointer?.y ?? padding.top;
    const tooltipShouldFlip = tooltipAnchorX != null && tooltipAnchorX + tooltipGap + tooltipWidth > width - 8;
    const tooltipX = tooltipAnchorX == null
        ? 0
        : tooltipShouldFlip
            ? Math.max(8, tooltipAnchorX - tooltipGap - tooltipWidth)
            : Math.min(tooltipAnchorX + tooltipGap, width - tooltipWidth - 8);
    const tooltipY = tooltipAnchorY - tooltipGap - tooltipHeight >= padding.top
        ? tooltipAnchorY - tooltipGap - tooltipHeight
        : Math.min(tooltipAnchorY + tooltipGap, height - tooltipHeight - 8);

    const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const pointer = pointerFromEvent(event);
        if (!pointer) return;
        setHoverPointer(pointer);
        setHoverIso(closestDateForX(pointer.x));
    };
    const handlePointerLeave = () => {
        setHoverIso(null);
        setHoverPointer(null);
    };

    return (
        <div className="fund-compare-chart-shell">
            <svg
                key={seriesKey}
                className="fund-compare-line-chart"
                viewBox={`0 0 ${width} ${height}`}
                role="img"
                aria-label="Hisse karşılaştırma getiri grafiği"
                onPointerMove={handlePointerMove}
                onPointerLeave={handlePointerLeave}
            >
                <defs>
                    <clipPath id={clipId}>
                        <rect x={padding.left} y={0} width={plotWidth} height={height} className="fund-chart-reveal" />
                    </clipPath>
                </defs>
                {yTicks.map((tick) => {
                    const y = yFor(tick);
                    return (
                        <g key={tick}>
                            <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="fund-chart-gridline" />
                            <text x={width - padding.right + 10} y={y + 4} className="fund-chart-axis">
                                {formatPct(tick).replace('+', '')}
                            </text>
                        </g>
                    );
                })}
                <line x1={padding.left} x2={width - padding.right} y1={yFor(0)} y2={yFor(0)} className="fund-comparison-zero" />
                {tickDates.map((date, index) => (
                    <text
                        key={`${date}-${index}`}
                        x={xForDate(date)}
                        y={height - 12}
                        className="fund-chart-axis"
                        textAnchor={index === 0 ? 'start' : index === tickDates.length - 1 ? 'end' : 'middle'}
                    >
                        {formatChartDate(date, range)}
                    </text>
                ))}
                <g clipPath={`url(#${clipId})`}>
                    {rawSeries.map((series) => {
                        const pathData = series.points
                            .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xForDate(point.date)} ${yFor(point.value)}`)
                            .join(' ');
                        return (
                            <path
                                key={series.stock.company}
                                d={pathData}
                                fill="none"
                                stroke={series.color}
                                strokeWidth="2.8"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                className="fund-chart-line"
                            />
                        );
                    })}
                </g>
                {rawSeries.map((series) => {
                    const last = series.points[series.points.length - 1];
                    return (
                        <g key={`${series.stock.company}-endpoint`} className="fund-chart-endpoint">
                            <circle cx={xForDate(last.date)} cy={yFor(last.value)} r="3.8" fill={series.color} />
                            <g transform={`translate(${Math.min(width - padding.right + 8, xForDate(last.date) + 8)}, ${yFor(last.value) - 11})`}>
                                <rect width="58" height="22" rx="4" fill={series.color} />
                                <text x="29" y="15" textAnchor="middle" className="fund-compare-end-label">
                                    {formatPct(last.value).replace('% ', '')}
                                </text>
                            </g>
                        </g>
                    );
                })}
                {tooltipRows.length > 0 && tooltipX0 != null && (
                    <g>
                        <line
                            x1={tooltipX0}
                            x2={tooltipX0}
                            y1={padding.top}
                            y2={height - padding.bottom}
                            className="fund-chart-hoverline"
                        />
                        {tooltipRows.map(({ series, point }) => (
                            <circle
                                key={`${series.stock.company}-hover`}
                                className="fund-chart-hover-point"
                                cx={xForDate(point.date)}
                                cy={yFor(point.value)}
                                r="4.2"
                                fill={series.color}
                                stroke="#0a0c0f"
                                strokeWidth="1.6"
                            />
                        ))}
                        <g
                            className="fund-chart-tooltip"
                            style={{ transform: `translate(${tooltipX}px, ${tooltipY}px)` }}
                        >
                            <rect width={tooltipWidth} height={tooltipHeight} rx="6" className="fund-chart-tooltip-bg" />
                            <text x="10" y="18" className="fund-chart-tooltip-muted">{formatDate(tooltipDate || '')}</text>
                            {tooltipRows.map(({ series, point }, idx) => (
                                <g key={series.stock.company} transform={`translate(10, ${36 + idx * 18})`}>
                                    <circle cx="4" cy="-4" r="4" fill={series.color} />
                                    <text x="14" y="0" className="fund-chart-tooltip-value">
                                        {series.stock.company}
                                    </text>
                                    <text x={tooltipWidth - 18} y="0" textAnchor="end" className="fund-chart-tooltip-value">
                                        {formatPct(point.value)}
                                    </text>
                                </g>
                            ))}
                        </g>
                    </g>
                )}
            </svg>
        </div>
    );
}

function StockCompareTable({
    title,
    selectedStocks,
    rows,
}: {
    title: string;
    selectedStocks: MarketStockRow[];
    rows: FundCompareTableRow[];
}) {
    return (
        <section className="fund-compare-section">
            <h2>{title}</h2>
            <div className="fund-compare-table-wrap">
                <table className="fund-compare-table">
                    <thead>
                        <tr>
                            <th />
                            {selectedStocks.map((stock) => (
                                <th key={stock.company}>
                                    <span>{stock.name || STOCK_SEARCH_LABELS[stock.company]?.[0] || stock.company}</span>
                                    <strong>{stock.company}</strong>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row) => (
                            <tr key={row.label}>
                                <td>{row.label}</td>
                                {row.values.map((cell, index) => (
                                    <td key={`${row.label}-${selectedStocks[index]?.company || index}`} className={cell.className}>
                                        <strong>{cell.primary}</strong>
                                        {cell.secondary && <span>{cell.secondary}</span>}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </section>
    );
}

export function StockComparisonSection({
    stocks,
    loading,
    error,
}: {
    stocks: MarketStocksResponse | null;
    loading: boolean;
    error: string | null;
}) {
    const [searchParams] = useSearchParams();
    const rows = stocks?.rows || [];

    const [selectedTickers, setSelectedTickers] = useState<string[]>(() => {
        const raw = searchParams.get('symbols');
        if (!raw) return [];
        const unique: string[] = [];
        for (const item of raw.split(',')) {
            const ticker = item.trim().toUpperCase().replace(/\s+/g, '');
            if (ticker && !unique.includes(ticker)) {
                unique.push(ticker);
                if (unique.length >= STOCK_COMPARE_MAX) break;
            }
        }
        return unique;
    });
    const [selectedStockCache, setSelectedStockCache] = useState<Record<string, MarketStockRow>>({});
    const [range, setRange] = useState<FundCompareRange>('6m');
    const [history5y, setHistory5y] = useState<Record<string, MarketComparisonHistoryPoint[]>>({});
    const [historyAll, setHistoryAll] = useState<Record<string, MarketComparisonHistoryPoint[]>>({});
    const history5yRef = useRef<Record<string, MarketComparisonHistoryPoint[]>>({});
    history5yRef.current = history5y;
    const historyAllRef = useRef<Record<string, MarketComparisonHistoryPoint[]>>({});
    historyAllRef.current = historyAll;
    const [historyLoading, setHistoryLoading] = useState(false);
    const [historyError, setHistoryError] = useState<string | null>(null);
    const [picker, setPicker] = useState<{ mode: 'add' } | { mode: 'swap'; index: number } | null>(null);
    const [query, setQuery] = useState('');
    const [searchRows, setSearchRows] = useState<MarketStockRow[]>([]);
    const [searchLoading, setSearchLoading] = useState(false);
    const inputRef = useRef<HTMLInputElement | null>(null);
    const swapPopoverRef = useRef<HTMLDivElement | null>(null);

    const searchOpen = picker !== null;
    const isSwapping = (index: number) => picker?.mode === 'swap' && picker.index === index;
    const isAllRange = range === 'all';

    useEffect(() => {
        if (!selectedTickers.length) {
            setHistoryLoading(false);
            setHistoryError(null);
            return;
        }
        const currentStore = isAllRange ? historyAllRef.current : history5yRef.current;
        const missingTickers = selectedTickers.filter((ticker) => currentStore[ticker] === undefined);
        if (!missingTickers.length) {
            setHistoryLoading(false);
            setHistoryError(null);
            return;
        }

        let alive = true;
        const controller = new AbortController();
        const todayIso = new Date().toISOString().slice(0, 10);
        const startIso = isAllRange ? STOCK_COMPARE_ALL_START_ISO : rangeStartDate('5y', todayIso, '');

        setHistoryLoading(true);
        setHistoryError(null);

        apiClient
            .marketComparisonHistory(
                {
                    assets: missingTickers.map((ticker) => ({
                        id: `stock:${ticker}`,
                        kind: 'stock',
                        symbol: ticker,
                        label: ticker,
                    })),
                    start_date: startIso,
                    end_date: todayIso,
                },
                { signal: controller.signal },
            )
            .then((response) => {
                if (!alive) return;
                const patch: Record<string, MarketComparisonHistoryPoint[]> = {};
                let firstError: string | null = null;
                for (const asset of response.assets || []) {
                    const symbol = String(asset.symbol || '').trim().toUpperCase();
                    if (!symbol) continue;
                    patch[symbol] = asset.points || [];
                    if (asset.error && !firstError) {
                        firstError = `${symbol}: ${asset.error}`;
                    }
                }
                for (const ticker of missingTickers) {
                    if (patch[ticker] === undefined) {
                        patch[ticker] = [];
                    }
                }
                if (isAllRange) {
                    setHistoryAll((current) => ({ ...current, ...patch }));
                } else {
                    setHistory5y((current) => ({ ...current, ...patch }));
                }
                setHistoryError(firstError);
            })
            .catch((err) => {
                if (!alive || (err as Error)?.name === 'AbortError') return;
                setHistoryError(err instanceof Error ? err.message : 'Geçmiş fiyat verisi alınamadı.');
            })
            .finally(() => {
                if (alive) setHistoryLoading(false);
            });

        return () => {
            alive = false;
            controller.abort();
        };
    }, [isAllRange, selectedTickers]);

    useEffect(() => {
        const trimmed = query.trim();
        if (!searchOpen || !trimmed || rows.length > 0) {
            setSearchRows([]);
            setSearchLoading(false);
            return;
        }
        let alive = true;
        const controller = new AbortController();
        setSearchLoading(true);
        const timer = window.setTimeout(() => {
            apiClient
                .marketStockSearch({ q: trimmed, index: 'XUTUM', limit: 12, signal: controller.signal })
                .then((payload) => {
                    if (alive) {
                        setSearchRows((payload.rows || []).map(universeRowToStockShell));
                    }
                })
                .catch((err) => {
                    if ((err as Error)?.name === 'AbortError') return;
                    if (alive) setSearchRows([]);
                })
                .finally(() => {
                    if (alive) setSearchLoading(false);
                });
        }, 160);
        return () => {
            alive = false;
            window.clearTimeout(timer);
            controller.abort();
        };
    }, [query, rows.length, searchOpen]);

    const selectedStocks = useMemo(() => selectedTickers
        .map((ticker) => rows.find((row) => row.company === ticker)
            || selectedStockCache[ticker]
            || searchRows.find((row) => row.company === ticker)
            || stockSummaryShell(ticker)), [rows, searchRows, selectedStockCache, selectedTickers]);

    const documentTitle = useMemo(
        () => buildDocumentTitle(
            'Hisse Karşılaştırma',
            selectedStocks.length ? selectedStocks.map((stock) => stock.company).join(' / ') : 'Hisse seçimi',
        ),
        [selectedStocks],
    );
    useDocumentTitle(documentTitle);

    const searchResults = useMemo(() => {
        const selectedSet = new Set(selectedTickers);
        const merged = new Map<string, MarketStockRow>();
        for (const row of rows) merged.set(row.company, row);
        for (const row of searchRows) {
            if (!merged.has(row.company)) merged.set(row.company, row);
        }
        const normalizedQuery = normalizeCompareSearch(query);
        const compactQuery = compactCompareSearch(query);
        if (!normalizedQuery) return [];
        const exactTicker: MarketStockRow[] = [];
        const prefixTicker: MarketStockRow[] = [];
        const otherMatches: MarketStockRow[] = [];
        for (const row of merged.values()) {
            if (selectedSet.has(row.company)) continue;
            const aliases = STOCK_SEARCH_LABELS[row.company] || [];
            const parts = [row.company, row.name || '', row.sector || '', ...aliases];
            const normalizedParts = parts.map(normalizeCompareSearch);
            const compactParts = parts.map(compactCompareSearch);
            const matched = normalizedParts.some((part) => part.includes(normalizedQuery))
                || compactParts.some((part) => part.includes(compactQuery));
            if (!matched) continue;
            const normalizedTicker = normalizeCompareSearch(row.company);
            if (normalizedTicker === normalizedQuery) exactTicker.push(row);
            else if (normalizedTicker.startsWith(normalizedQuery)) prefixTicker.push(row);
            else otherMatches.push(row);
        }
        return [...exactTicker, ...prefixTicker, ...otherMatches].slice(0, 8);
    }, [query, rows, searchRows, selectedTickers]);

    const addStock = useCallback((stock: MarketStockRow) => {
        setSelectedStockCache((current) => ({ ...current, [stock.company]: stock }));
        setSelectedTickers((current) => {
            if (current.length >= STOCK_COMPARE_MAX) return current;
            if (current.includes(stock.company)) return current;
            return [...current, stock.company];
        });
        setPicker(null);
        setQuery('');
    }, []);

    const swapStockAt = useCallback((index: number, stock: MarketStockRow) => {
        setSelectedStockCache((current) => ({ ...current, [stock.company]: stock }));
        setSelectedTickers((current) => {
            if (index < 0 || index >= current.length) return current;
            if (current.some((code, position) => code === stock.company && position !== index)) {
                return current;
            }
            if (current[index] === stock.company) return current;
            const next = current.slice();
            next[index] = stock.company;
            return next;
        });
        setPicker(null);
        setQuery('');
    }, []);

    const removeStock = useCallback((tickerToRemove: string) => {
        setSelectedTickers((current) => current.filter((ticker) => ticker !== tickerToRemove));
    }, []);

    const openAddPicker = useCallback(() => {
        if (selectedTickers.length >= STOCK_COMPARE_MAX) return;
        setPicker({ mode: 'add' });
        setQuery('');
        window.setTimeout(() => {
            inputRef.current?.focus();
            inputRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 0);
    }, [selectedTickers.length]);

    const openSwapPicker = useCallback((index: number) => {
        setPicker({ mode: 'swap', index });
        setQuery('');
        window.setTimeout(() => {
            inputRef.current?.focus();
            inputRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 0);
    }, []);

    const closePicker = useCallback(() => {
        setPicker(null);
        setQuery('');
    }, []);

    useEffect(() => {
        if (picker?.mode !== 'swap') return;
        const handleDocumentClick = (event: MouseEvent) => {
            const target = event.target as Node | null;
            if (target && swapPopoverRef.current && swapPopoverRef.current.contains(target)) return;
            setPicker(null);
            setQuery('');
        };
        document.addEventListener('mousedown', handleDocumentClick);
        return () => document.removeEventListener('mousedown', handleDocumentClick);
    }, [picker]);

    const activeHistory = isAllRange ? historyAll : history5y;
    const tableEndIso = selectedStocks.find((stock) => stock.price_as_of)?.price_as_of?.slice(0, 10)
        || new Date().toISOString().slice(0, 10);

    const basicRows: FundCompareTableRow[] = [
        {
            label: 'Fiyat',
            values: selectedStocks.map((stock) => comparisonCell(
                formatStockMoney(stock.price, stock.price_currency),
                formatDate(stock.price_as_of),
            )),
        },
        {
            label: 'Günlük Değişim',
            values: selectedStocks.map((stock) => {
                const value = finiteNumber(stock.change_pct);
                return comparisonCell(formatPct(value), undefined, pctClass(value));
            }),
        },
        {
            label: 'Sektör',
            values: selectedStocks.map((stock) => comparisonCell(stock.sector)),
        },
        {
            label: 'Piyasa Değeri',
            values: selectedStocks.map((stock) => comparisonCell(formatCompactCurrency(stock.market_cap))),
        },
        {
            label: 'İşlem Hacmi',
            values: selectedStocks.map((stock) => comparisonCell(formatCompactCurrency(stock.volume))),
        },
        {
            label: 'Fiili Dolaşım',
            values: selectedStocks.map((stock) => comparisonCell(formatPlainPct(stock.free_float_pct))),
        },
        {
            label: 'Son Bilanço',
            values: selectedStocks.map((stock) => comparisonCell(stock.fundamental_period || stock.latest_quarter)),
        },
    ];

    const valuationRows: FundCompareTableRow[] = [
        {
            label: 'F/K',
            values: selectedStocks.map((stock) => comparisonCell(
                formatMultiple(stock.fk),
                historicalMultipleSecondary(stock.hist_fk_avg, stock.fk_hist_diff_pct),
            )),
        },
        {
            label: 'PD/DD',
            values: selectedStocks.map((stock) => comparisonCell(
                formatMultiple(stock.pd_dd),
                historicalMultipleSecondary(stock.hist_pd_dd_avg, stock.pd_dd_hist_diff_pct),
            )),
        },
        {
            label: 'FD/FAVÖK',
            values: selectedStocks.map((stock) => comparisonCell(
                formatMultiple(stock.fd_favok),
                historicalMultipleSecondary(stock.hist_fd_favok_avg, stock.fd_favok_hist_diff_pct),
            )),
        },
        {
            label: 'FD/Satış',
            values: selectedStocks.map((stock) => comparisonCell(formatMultiple(stock.fd_satis))),
        },
        {
            label: 'Net Borç / FAVÖK',
            values: selectedStocks.map((stock) => comparisonCell(formatMultiple(stock.net_borc_favok))),
        },
    ];

    const profitabilityAndFinancialsRows: FundCompareTableRow[] = [
        {
            label: 'Özsermaye Karlılığı (ROE)',
            values: selectedStocks.map((stock) => {
                const value = finiteNumber(stock.roe_pct);
                return comparisonCell(formatPlainPct(value), undefined, pctClass(value));
            }),
        },
        {
            label: 'FAVÖK Marjı',
            values: selectedStocks.map((stock) => {
                const value = finiteNumber(stock.favok_margin_pct);
                return comparisonCell(formatPlainPct(value), undefined, pctClass(value));
            }),
        },
        {
            label: 'Net Kar Marjı',
            values: selectedStocks.map((stock) => {
                const value = finiteNumber(stock.net_margin_pct);
                return comparisonCell(formatPlainPct(value), undefined, pctClass(value));
            }),
        },
        {
            label: 'Satışlar (TTM)',
            values: selectedStocks.map((stock) => comparisonCell(formatCompactCurrency(stock.ttm_satis_gelirleri))),
        },
        {
            label: 'FAVÖK (TTM)',
            values: selectedStocks.map((stock) => comparisonCell(formatCompactCurrency(stock.ttm_favok))),
        },
        {
            label: 'Net Kar (TTM)',
            values: selectedStocks.map((stock) => comparisonCell(formatCompactCurrency(stock.ttm_net_kar))),
        },
        {
            label: 'Serbest Nakit Akışı (TTM)',
            values: selectedStocks.map((stock) => comparisonCell(formatCompactCurrency(stock.ttm_fcf))),
        },
        {
            label: 'Özkaynaklar',
            values: selectedStocks.map((stock) => comparisonCell(formatCompactCurrency(stock.ozkaynaklar))),
        },
        {
            label: 'Net Borç',
            values: selectedStocks.map((stock) => comparisonCell(formatCompactCurrency(stock.net_borc))),
        },
    ];

    const returnRows: FundCompareTableRow[] = [
        ['1 Hafta', '1w'],
        ['1 Ay', '1m'],
        ['3 Ay', '3m'],
        ['6 Ay', '6m'],
        ['Yılbaşından Bugüne', 'ytd'],
        ['1 Yıl', '1y'],
        ['3 Yıl', '3y'],
        ['5 Yıl', '5y'],
    ].map(([label, key]) => ({
        label,
        values: selectedStocks.map((stock) => {
            const summaryValue = summaryReturnForStockRange(stock, key as FundCompareRange);
            const points = history5y[stock.company] || historyAll[stock.company] || [];
            const calculated = returnForStockRange(points, key as FundCompareRange, tableEndIso);
            const value = summaryValue ?? calculated;
            return comparisonCell(formatPct(value), undefined, pctClass(value));
        }),
    }));
    returnRows.push(
        {
            label: 'CAGR (3 Yıl)',
            values: selectedStocks.map((stock) => {
                const points = history5y[stock.company] || historyAll[stock.company] || [];
                const value = cagrForStockRange(points, 3, tableEndIso);
                return comparisonCell(formatPct(value), undefined, pctClass(value));
            }),
        },
        {
            label: 'CAGR (5 Yıl)',
            values: selectedStocks.map((stock) => {
                const points = history5y[stock.company] || historyAll[stock.company] || [];
                const value = cagrForStockRange(points, 5, tableEndIso);
                return comparisonCell(formatPct(value), undefined, pctClass(value));
            }),
        },
    );

    const technicalRows: FundCompareTableRow[] = [
        {
            label: '52H Zirve',
            values: selectedStocks.map((stock) => comparisonCell(
                formatStockMoney(stock.high_1y, stock.price_currency),
                stock.high_52w_dist_pct != null ? formatPct(stock.high_52w_dist_pct) : undefined,
            )),
        },
        {
            label: '52H Dip',
            values: selectedStocks.map((stock) => comparisonCell(
                formatStockMoney(stock.low_1y, stock.price_currency),
                stock.low_52w_dist_pct != null ? formatPct(stock.low_52w_dist_pct) : undefined,
            )),
        },
        {
            label: 'SMA 50',
            values: selectedStocks.map((stock) => comparisonCell(
                formatStockMoney(stock.sma_50, stock.price_currency),
                stock.sma_50_diff_pct != null ? formatPct(stock.sma_50_diff_pct) : undefined,
                pctClass(stock.sma_50_diff_pct),
            )),
        },
        {
            label: 'SMA 200',
            values: selectedStocks.map((stock) => comparisonCell(
                formatStockMoney(stock.sma_200, stock.price_currency),
                stock.sma_200_diff_pct != null ? formatPct(stock.sma_200_diff_pct) : undefined,
                pctClass(stock.sma_200_diff_pct),
            )),
        },
    ];

    const hasChartPerformance = selectedStocks.some((stock) => (activeHistory[stock.company] || []).length >= 2);

    return (
        <div className={`fund-compare-page ${searchOpen ? 'is-picker-open' : ''}`}>
            <header className="fund-compare-hero">
                <div className="fund-market-breadcrumb" aria-label="Hisse karşılaştırma konumu">
                    <span className="fund-breadcrumb-group">
                        <Layers size={17} aria-hidden="true" />
                        Hisseler
                    </span>
                    <ChevronRight size={14} aria-hidden="true" />
                    <span className="fund-breadcrumb-code">Hisse Karşılaştırma</span>
                </div>
                <div className="fund-compare-title">
                    <h1>Hisse Karşılaştırma</h1>
                    <p>BIST hisselerini getiri performansı, değerleme çarpanları, karlılık ve teknik göstergelere göre yan yana karşılaştırabilirsiniz.</p>
                </div>
            </header>

            <section className="fund-compare-chart-panel">
                <div className="fund-compare-range-tabs" aria-label="Karşılaştırma dönemi">
                    {FUND_COMPARE_RANGES.map((item) => (
                        <button
                            key={item.key}
                            type="button"
                            className={range === item.key ? 'active' : undefined}
                            onClick={() => setRange(item.key)}
                            aria-pressed={range === item.key}
                        >
                            {item.label}
                        </button>
                    ))}
                </div>
                {loading && !selectedStocks.length ? (
                    <div className="fund-compare-chart-empty">Hisse listesi yükleniyor...</div>
                ) : error && !rows.length && !selectedStocks.length ? (
                    <div className="fund-compare-chart-empty funds-state-error">{error}</div>
                ) : selectedStocks.length === 0 ? (
                    <div className="fund-compare-chart-empty">
                        Karşılaştırmak için en az bir hisse ekleyin.
                    </div>
                ) : historyLoading && !hasChartPerformance ? (
                    <div className="fund-compare-chart-empty">Grafik verileri yükleniyor...</div>
                ) : (
                    <StockCompareLineChart selectedStocks={selectedStocks} historyByTicker={activeHistory} range={range} />
                )}
                {historyError && <div className="fund-chart-error">{historyError}</div>}
                <div className="fund-compare-selector-row">
                    {selectedStocks.map((stock, index) => {
                        const swapping = isSwapping(index);
                        const swapResultsToShow = searchResults.filter((row) => row.company !== stock.company);
                        const displayName = stock.name || STOCK_SEARCH_LABELS[stock.company]?.[0] || stock.company;
                        return (
                            <div
                                key={stock.company}
                                className={`fund-compare-card ${swapping ? 'is-swapping' : ''}`}
                                ref={swapping ? swapPopoverRef : undefined}
                            >
                                <span><i style={{ background: FUND_COMPARE_COLORS[index % FUND_COMPARE_COLORS.length] }} /> {index + 1}. Hisse</span>
                                <button
                                    type="button"
                                    className="fund-compare-remove"
                                    onClick={() => removeStock(stock.company)}
                                    aria-label={`${stock.company} hissesini çıkar`}
                                >
                                    <X size={14} aria-hidden="true" />
                                </button>
                                <div className="fund-compare-card-main">
                                    <Link
                                        to={toStockDetail(stock.company, 'overview')}
                                        className="fund-compare-card-fund"
                                        aria-label={`${stock.company} hissesini aç`}
                                    >
                                        <SymbolLogo
                                            symbol={stock.company}
                                            name={displayName}
                                            logoUrl={stock.logo_url}
                                            kind="stock"
                                            size="sm"
                                        />
                                        <strong>{stock.company}</strong>
                                    </Link>
                                    <button
                                        type="button"
                                        className={`fund-compare-card-swap ${swapping ? 'is-active' : ''}`}
                                        onClick={() => (swapping ? closePicker() : openSwapPicker(index))}
                                        aria-haspopup="dialog"
                                        aria-expanded={swapping}
                                        aria-label={`${stock.company} hissesini değiştir`}
                                    >
                                        <ChevronDown size={16} aria-hidden="true" />
                                    </button>
                                </div>
                                {swapping && (
                                    <div className="fund-compare-card-popover" role="dialog" aria-label={`${index + 1}. hisseyi değiştir`}>
                                        <div className="fund-compare-search">
                                            <Search size={16} aria-hidden="true" />
                                            <input
                                                ref={inputRef}
                                                value={query}
                                                onChange={(event) => setQuery(event.target.value)}
                                                onKeyDown={(event) => {
                                                    if (event.key === 'Escape') {
                                                        closePicker();
                                                    }
                                                    if (event.key === 'Enter' && swapResultsToShow[0]) {
                                                        swapStockAt(index, swapResultsToShow[0]);
                                                    }
                                                }}
                                                placeholder="Hisse ara"
                                                aria-label={`${index + 1}. hisseyi değiştir`}
                                            />
                                            <button type="button" onClick={closePicker} aria-label="Aramayı kapat">
                                                <X size={15} aria-hidden="true" />
                                            </button>
                                        </div>
                                        <div className="fund-compare-search-results fund-compare-search-results-inline">
                                            {query.trim() ? (
                                                swapResultsToShow.length ? swapResultsToShow.map((row) => {
                                                    const rowName = row.name || STOCK_SEARCH_LABELS[row.company]?.[0] || row.company;
                                                    return (
                                                        <button key={row.company} type="button" onClick={() => swapStockAt(index, row)}>
                                                            <SymbolLogo
                                                                symbol={row.company}
                                                                name={rowName}
                                                                logoUrl={row.logo_url}
                                                                kind="stock"
                                                                size="sm"
                                                            />
                                                            <strong>{row.company}</strong>
                                                            <span>{rowName}</span>
                                                        </button>
                                                    );
                                                }) : (
                                                    <div>{searchLoading ? 'Aranıyor...' : 'Hisse bulunamadı.'}</div>
                                                )
                                            ) : (
                                                <div>Hisse kodu veya adı yazın.</div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                    <div className={`fund-compare-add ${picker?.mode === 'add' ? 'is-open' : ''}`}>
                        {picker?.mode === 'add' ? (
                            <div className="fund-compare-search">
                                <Search size={16} aria-hidden="true" />
                                <input
                                    ref={inputRef}
                                    value={query}
                                    onChange={(event) => setQuery(event.target.value)}
                                    onKeyDown={(event) => {
                                        if (event.key === 'Escape') {
                                            closePicker();
                                        }
                                        if (event.key === 'Enter' && searchResults[0]) {
                                            addStock(searchResults[0]);
                                        }
                                    }}
                                    placeholder="Hisse ara"
                                    aria-label="Karşılaştırmaya hisse ekle"
                                />
                                <button type="button" onClick={closePicker} aria-label="Aramayı kapat">
                                    <X size={15} aria-hidden="true" />
                                </button>
                                <div className="fund-compare-search-results">
                                    {query.trim() ? (
                                        searchResults.length ? searchResults.map((stock) => {
                                            const stockName = stock.name || STOCK_SEARCH_LABELS[stock.company]?.[0] || stock.company;
                                            return (
                                                <button key={stock.company} type="button" onClick={() => addStock(stock)}>
                                                    <SymbolLogo
                                                        symbol={stock.company}
                                                        name={stockName}
                                                        logoUrl={stock.logo_url}
                                                        kind="stock"
                                                        size="sm"
                                                    />
                                                    <strong>{stock.company}</strong>
                                                    <span>{stockName}</span>
                                                </button>
                                            );
                                        }) : (
                                            <div>{searchLoading ? 'Aranıyor...' : 'Hisse bulunamadı.'}</div>
                                        )
                                    ) : (
                                        <div>Hisse kodu veya adı yazın.</div>
                                    )}
                                </div>
                            </div>
                        ) : (
                            <button type="button" onClick={openAddPicker} disabled={selectedTickers.length >= STOCK_COMPARE_MAX}>
                                <Plus size={18} aria-hidden="true" />
                                Yeni Hisse Ekle
                            </button>
                        )}
                    </div>
                </div>
            </section>

            {selectedStocks.length > 0 && (
                <div className="fund-compare-sections">
                    <StockCompareTable title="Temel Bilgiler" selectedStocks={selectedStocks} rows={basicRows} />
                    <StockCompareTable title="Değerleme Çarpanları" selectedStocks={selectedStocks} rows={valuationRows} />
                    <StockCompareTable title="Karlılık ve Finansallar (TTM)" selectedStocks={selectedStocks} rows={profitabilityAndFinancialsRows} />
                    <StockCompareTable title="Getiri Analizi" selectedStocks={selectedStocks} rows={returnRows} />
                    <StockCompareTable title="Teknik ve 52 Haftalık Seviyeler" selectedStocks={selectedStocks} rows={technicalRows} />
                </div>
            )}
        </div>
    );
}

export default StockComparisonSection;
