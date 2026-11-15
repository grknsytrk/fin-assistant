import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { ChevronDown, ChevronRight, Database, Plus, Search, X } from 'lucide-react';
import { Link } from 'react-router-dom';
import { apiClient } from '../../../api/client';
import type {
    FundAllocation,
    FundAllocationsResponse,
    FundDetail,
    FundHoldingsResponse,
    FundPortfolioPosition,
    FundPricePoint,
    FundSummary,
    FundsResponse,
} from '../../../api/types';
import SymbolLogo from '../../../components/SymbolLogo';
import { buildDocumentTitle, useDocumentTitle } from '../../../hooks/useDocumentTitle';
import { toFundDetail } from '../../../routing/routes';
import { hasFundRangeStartCoverage } from '../../../utils/fundPresentation';
import {
    chartDateMs,
    compactCompareSearch,
    finiteNumber,
    formatChartDate,
    formatCompactCurrency,
    formatCompactOverviewNumber,
    formatDate,
    formatPct,
    formatPlainPct,
    formatQuotePrice,
    formatSignedCompactCurrency,
    formatWholeNumber,
    holdingWeightValue,
    nearestChartPoint,
    normalizeCompareSearch,
    pctClass,
    rangeStartDate,
    returnBetween,
    searchCachedFundCatalog,
    sortFundPoints,
} from '../formatters';
import {
    FUND_COMPARE_COLORS,
    FUND_COMPARE_MAX_FUNDS,
    FUND_COMPARE_RANGES,
    type FundCompareCell,
    type FundCompareDataset,
    type FundCompareRange,
    type FundCompareTableRow,
} from '../types';
import './FundComparisonSection.css';

export function emptyFundCompareDataset(): FundCompareDataset {
    return {
        detail: null,
        performance: null,
        allocations: null,
        holdings: null,
        detailLoading: false,
        performanceLoading: false,
        allocationsLoading: false,
        holdingsLoading: false,
        detailError: null,
        performanceError: null,
        allocationsError: null,
        holdingsError: null,
        loading: false,
        error: null,
    };
}

export function mergeFundCompareDataset(
    dataset: FundCompareDataset | undefined,
    patch: Partial<FundCompareDataset>,
): FundCompareDataset {
    const next = { ...(dataset || emptyFundCompareDataset()), ...patch };
    return {
        ...next,
        loading: next.detailLoading || next.performanceLoading || next.allocationsLoading || next.holdingsLoading,
        error: next.detailError || next.performanceError || next.allocationsError || next.holdingsError,
    };
}

export function fundSummaryShell(fundCode: string): FundSummary {
    return {
        fund_code: fundCode,
        name: fundCode,
        fund_type: null,
        founder_company: null,
        manager_company: null,
        price: null,
        daily_return: null,
        period_returns: {},
        risk_value: null,
        currency: 'TRY',
        as_of: null,
        source: 'selected',
    };
}

export function fundCompareRangeStart(range: FundCompareRange, endIso: string): string {
    if (range === 'all') return '1900-01-01';
    return rangeStartDate(range, endIso, '');
}

export function comparisonCell(primary: string | number | null | undefined, secondary?: string | null, className?: string): FundCompareCell {
    return {
        primary: primary == null || primary === '' ? '-' : String(primary),
        secondary: secondary || undefined,
        className,
    };
}

export function fundUmbrellaLabel(value: string | null | undefined): string {
    const text = String(value || '').trim();
    if (!text) return '-';
    return text
        .replace(/\s*Şemsiye\s+Fonu\s*$/i, '')
        .replace(/\s*Fonu\s*$/i, '')
        .trim() || text;
}

export function nearestPointOnOrAfter(points: FundPricePoint[], startIso: string): FundPricePoint | null {
    return points.find((point) => point.date >= startIso) || null;
}

export function returnForRange(points: FundPricePoint[], range: FundCompareRange, endIso: string): number | null {
    const ordered = sortFundPoints(points);
    if (ordered.length < 2) return null;
    const latest = ordered[ordered.length - 1];
    const startIso = fundCompareRangeStart(range, endIso);
    const base = range === 'all' ? ordered[0] : nearestPointOnOrAfter(ordered, startIso);
    if (range !== 'all' && base) {
        // A weekend/short holiday gap is expected. Anything beyond a trading
        // week is incomplete history and must not masquerade as the selected
        // one-year/three-year range.
        if (!hasFundRangeStartCoverage(startIso, base.date)) return null;
    }
    return returnBetween(finiteNumber(latest.price), finiteNumber(base?.price));
}

export function summaryReturnForCompareRange(
    returns: FundSummary['period_returns'] | null | undefined,
    range: FundCompareRange,
): number | null {
    if (!returns) return null;
    if (range === '1w') return finiteNumber(returns['1w']);
    if (range === '1m') return finiteNumber(returns['1m']);
    if (range === '3m') return finiteNumber(returns['3m']);
    if (range === '6m') return finiteNumber(returns['6m']);
    if (range === 'ytd') return finiteNumber(returns.ytd);
    if (range === '1y') return finiteNumber(returns['1y']);
    return null;
}

export function cagrForRange(points: FundPricePoint[], years: number, endIso: string): number | null {
    const ordered = sortFundPoints(points);
    if (ordered.length < 2) return null;
    const latest = ordered[ordered.length - 1];
    const base = nearestPointOnOrAfter(ordered, rangeStartDate(years === 3 ? '3y' : '5y', endIso, ''));
    const latestPrice = finiteNumber(latest.price);
    const basePrice = finiteNumber(base?.price);
    if (latestPrice == null || basePrice == null || basePrice <= 0 || latestPrice <= 0) return null;
    const baseTs = new Date(base?.date || '').getTime();
    const latestTs = new Date(latest.date).getTime();
    const actualYears = (latestTs - baseTs) / (365.25 * 24 * 60 * 60 * 1000);
    if (!Number.isFinite(actualYears) || actualYears < years * 0.78) return null;
    return (Math.pow(latestPrice / basePrice, 1 / actualYears) - 1) * 100;
}

export function cashFlowForRange(points: FundPricePoint[], range: FundCompareRange, endIso: string): number | null {
    const ordered = sortFundPoints(points).filter((point) => finiteNumber(point.aum) != null && finiteNumber(point.price) != null);
    if (ordered.length < 2) return null;
    const latest = ordered[ordered.length - 1];
    const base = nearestPointOnOrAfter(ordered, fundCompareRangeStart(range, endIso));
    const latestAum = finiteNumber(latest.aum);
    const latestPrice = finiteNumber(latest.price);
    const baseAum = finiteNumber(base?.aum);
    const basePrice = finiteNumber(base?.price);
    if (latestAum == null || latestPrice == null || baseAum == null || basePrice == null || basePrice <= 0) return null;
    return latestAum - baseAum * (latestPrice / basePrice);
}

export function topAllocations(payload: FundAllocationsResponse | null): FundAllocation[] {
    return [...(payload?.allocations || [])]
        .filter((item) => finiteNumber(item.weight) != null)
        .sort((a, b) => Math.abs(Number(b.weight || 0)) - Math.abs(Number(a.weight || 0)))
        .slice(0, 5);
}

export function topHoldingPositions(payload: FundHoldingsResponse | null): FundPortfolioPosition[] {
    return [...(payload?.positions || [])]
        .filter((position) => {
            const weight = holdingWeightValue(position);
            return weight != null && weight > 0;
        })
        .sort((a, b) => Number(holdingWeightValue(b) || 0) - Number(holdingWeightValue(a) || 0))
        .slice(0, 5);
}

export function topHoldingSectors(payload: FundHoldingsResponse | null): Array<{ label: string; weight: number }> {
    const weights = new Map<string, number>();
    for (const position of payload?.positions || []) {
        if (position.asset_type !== 'local_equity' || !position.sector_label) continue;
        const weight = holdingWeightValue(position);
        if (weight == null || weight <= 0) continue;
        weights.set(position.sector_label, (weights.get(position.sector_label) || 0) + weight);
    }
    return [...weights.entries()]
        .map(([label, weight]) => ({ label, weight }))
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 5);
}

export function fundMarketShare(row: FundDetail | FundSummary | null, allFunds: FundSummary[]): number | null {
    const aum = finiteNumber(row?.aum);
    if (aum == null || aum <= 0 || !row?.fund_type) return null;
    const total = allFunds.reduce((sum, item) => {
        if (item.fund_type !== row.fund_type) return sum;
        const value = finiteNumber(item.aum);
        return value == null || value <= 0 ? sum : sum + value;
    }, 0);
    return total > 0 ? (aum / total) * 100 : null;
}

export function FundCompareLineChart({
    selectedFunds,
    datasets,
    range,
}: {
    selectedFunds: FundSummary[];
    datasets: Record<string, FundCompareDataset>;
    range: FundCompareRange;
}) {
    const width = 1180;
    const height = 420;
    const padding = { top: 30, right: 78, bottom: 52, left: 22 };
    const chartEndIso = selectedFunds.find((fund) => fund.as_of)?.as_of || new Date().toISOString().slice(0, 10);
    const chartStartIso = fundCompareRangeStart(range, chartEndIso);
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const rawSeries = selectedFunds.map((fund, index) => {
        const points = sortFundPoints(datasets[fund.fund_code]?.performance?.points)
            .filter((point) => range === 'all' || point.date >= chartStartIso);
        const base = points.find((point) => Number(point.price) > 0);
        return {
            fund,
            color: FUND_COMPARE_COLORS[index % FUND_COMPARE_COLORS.length],
            points: base?.price
                ? points.map((point) => ({
                    date: point.date,
                    value: ((Number(point.price) / Number(base.price)) - 1) * 100,
                }))
                : [],
        };
    }).filter((series) => series.points.length >= 2);
    const allValues = rawSeries.flatMap((series) => series.points.map((point) => point.value));
    const seriesKey = `${range}-${rawSeries.map((series) => series.fund.fund_code).join('-')}`;
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

    const clipId = `fund-compare-clip-${seriesKey}`;

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
                aria-label="Fon karşılaştırma getiri grafiği"
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
                                key={series.fund.fund_code}
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
                        <g key={`${series.fund.fund_code}-endpoint`} className="fund-chart-endpoint">
                            <circle cx={xForDate(last.date)} cy={yFor(last.value)} r="3.8" fill={series.color} />
                            <g transform={`translate(${Math.min(width - padding.right + 8, xForDate(last.date) + 8)}, ${yFor(last.value) - 11})`}>
                                <rect width="58" height="22" rx="4" fill={series.color} />
                                <text x="29" y="15" textAnchor="middle" className="fund-compare-end-label">{formatPct(last.value).replace('% ', '')}</text>
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
                                key={`${series.fund.fund_code}-hover`}
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
                                <g key={series.fund.fund_code} transform={`translate(10, ${36 + idx * 18})`}>
                                    <circle cx="4" cy="-4" r="4" fill={series.color} />
                                    <text x="14" y="0" className="fund-chart-tooltip-value">
                                        {series.fund.fund_code}
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

export function FundCompareTable({
    title,
    selectedFunds,
    rows,
}: {
    title: string;
    selectedFunds: FundSummary[];
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
                            {selectedFunds.map((fund) => (
                                <th key={fund.fund_code}>
                                    <span>{fund.name}</span>
                                    <strong>{fund.fund_code}</strong>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row) => (
                            <tr key={row.label}>
                                <td>{row.label}</td>
                                {row.values.map((cell, index) => (
                                    <td key={`${row.label}-${selectedFunds[index]?.fund_code || index}`} className={cell.className}>
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

export function FundComparisonPage({
    funds,
    loading,
    error,
}: {
    funds: FundsResponse | null;
    loading: boolean;
    error: string | null;
}) {
    const rows = funds?.rows || [];
    const [selectedCodes, setSelectedCodes] = useState<string[]>([]);
    const [selectedFundCache, setSelectedFundCache] = useState<Record<string, FundSummary>>({});
    const [range, setRange] = useState<FundCompareRange>('6m');
    const [datasets, setDatasets] = useState<Record<string, FundCompareDataset>>({});
    const [picker, setPicker] = useState<{ mode: 'add' } | { mode: 'swap'; index: number } | null>(null);
    const [query, setQuery] = useState('');
    const [searchRows, setSearchRows] = useState<FundSummary[]>([]);
    const [searchLoading, setSearchLoading] = useState(false);
    const inputRef = useRef<HTMLInputElement | null>(null);
    const swapPopoverRef = useRef<HTMLDivElement | null>(null);
    const staticRequestRef = useRef(0);
    const performanceRequestRef = useRef(0);

    const searchOpen = picker !== null;
    const isSwapping = (index: number) => picker?.mode === 'swap' && picker.index === index;

    useEffect(() => {
        if (!selectedCodes.length) {
            setDatasets({});
            return;
        }
        const generation = staticRequestRef.current + 1;
        staticRequestRef.current = generation;
        const selectedSet = new Set(selectedCodes);
        setDatasets((current) => {
            const next: Record<string, FundCompareDataset> = {};
            for (const code of selectedCodes) {
                const currentDataset = current[code];
                next[code] = mergeFundCompareDataset(currentDataset, {
                    detailLoading: !currentDataset?.detail,
                    allocationsLoading: !currentDataset?.allocations,
                    holdingsLoading: !currentDataset?.holdings,
                    detailError: null,
                    allocationsError: null,
                    holdingsError: null,
                });
            }
            return next;
        });

        const updateSlice = (code: string, patch: Partial<FundCompareDataset>) => {
            if (staticRequestRef.current !== generation || !selectedSet.has(code)) return;
            setDatasets((current) => {
                if (!selectedSet.has(code)) return current;
                return {
                    ...current,
                    [code]: mergeFundCompareDataset(current[code], patch),
                };
            });
        };

        for (const code of selectedCodes) {
            apiClient
                .fundDetail(code)
                .then((detail) => {
                    setSelectedFundCache((current) => ({ ...current, [code]: detail }));
                    updateSlice(code, { detail, detailLoading: false, detailError: null });
                })
                .catch((err) => updateSlice(code, {
                    detailLoading: false,
                    detailError: err instanceof Error ? err.message : 'Fon detayı alınamadı.',
                }));
            apiClient
                .fundAllocations(code)
                .then((allocations) => updateSlice(code, { allocations, allocationsLoading: false, allocationsError: null }))
                .catch((err) => updateSlice(code, {
                    allocationsLoading: false,
                    allocationsError: err instanceof Error ? err.message : 'Varlık dağılımı alınamadı.',
                }));
            apiClient
                .fundHoldings(code)
                .then((holdings) => updateSlice(code, { holdings, holdingsLoading: false, holdingsError: null }))
                .catch((err) => updateSlice(code, {
                    holdingsLoading: false,
                    holdingsError: err instanceof Error ? err.message : 'KAP fon içeriği alınamadı.',
                }));
        }
    }, [selectedCodes]);

    useEffect(() => {
        if (!selectedCodes.length) return;
        const generation = performanceRequestRef.current + 1;
        performanceRequestRef.current = generation;
        const selectedSet = new Set(selectedCodes);
        const todayIso = new Date().toISOString().slice(0, 10);
        const performanceStartDate = range === 'all'
            ? undefined
            : rangeStartDate(range, todayIso, '');
        setDatasets((current) => {
            const next: Record<string, FundCompareDataset> = {};
            for (const code of selectedCodes) {
                next[code] = mergeFundCompareDataset(current[code], {
                    performanceLoading: true,
                    performanceError: null,
                });
            }
            return next;
        });
        for (const code of selectedCodes) {
            apiClient
                .fundPerformance(code, performanceStartDate ? { startDate: performanceStartDate } : undefined)
                .then((performance) => {
                    if (performanceRequestRef.current !== generation || !selectedSet.has(code)) return;
                    setDatasets((current) => ({
                        ...current,
                        [code]: mergeFundCompareDataset(current[code], { performance, performanceLoading: false, performanceError: null }),
                    }));
                })
                .catch((err) => {
                    if (performanceRequestRef.current !== generation || !selectedSet.has(code)) return;
                    setDatasets((current) => ({
                        ...current,
                        [code]: mergeFundCompareDataset(current[code], {
                            performanceLoading: false,
                            performanceError: err instanceof Error ? err.message : 'Performans verisi alınamadı.',
                        }),
                    }));
                });
        }
    }, [selectedCodes, range]);

    useEffect(() => {
        const trimmed = query.trim();
        if (!searchOpen || !trimmed) {
            setSearchRows([]);
            setSearchLoading(false);
            return;
        }
        const cachedMatches = searchCachedFundCatalog(trimmed, 12);
        if (cachedMatches !== null) {
            setSearchRows(cachedMatches);
            setSearchLoading(false);
            return;
        }
        let alive = true;
        const controller = new AbortController();
        setSearchLoading(true);
        const timer = window.setTimeout(() => {
            apiClient
                .fundSearch(trimmed, 12, { signal: controller.signal })
                .then((payload) => {
                    if (alive) setSearchRows(payload.rows || []);
                })
                .catch((error) => {
                    if ((error as Error)?.name === 'AbortError') return;
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
    }, [query, searchOpen]);

    const selectedFunds = useMemo(() => selectedCodes
        .map((code) => datasets[code]?.detail
            || selectedFundCache[code]
            || rows.find((row) => row.fund_code === code)
            || searchRows.find((row) => row.fund_code === code)
            || fundSummaryShell(code)), [datasets, rows, searchRows, selectedCodes, selectedFundCache]);
    const documentTitle = useMemo(
        () => buildDocumentTitle(
            'Fon Karşılaştırma',
            selectedFunds.length ? selectedFunds.map((fund) => fund.fund_code).join(' / ') : 'Fon seçimi',
        ),
        [selectedFunds],
    );
    useDocumentTitle(documentTitle);

    const searchResults = useMemo(() => {
        const selectedSet = new Set(selectedCodes);
        const merged = new Map<string, FundSummary>();
        for (const row of rows) merged.set(row.fund_code, row);
        for (const row of searchRows) merged.set(row.fund_code, row);
        const normalizedQuery = normalizeCompareSearch(query);
        const compactQuery = compactCompareSearch(query);
        if (!normalizedQuery) return [];
        return [...merged.values()]
            .filter((row) => !selectedSet.has(row.fund_code))
            .filter((row) => {
                const parts = [row.fund_code, row.name, row.fund_type || '', row.founder_company || ''];
                const normalizedParts = parts.map(normalizeCompareSearch);
                const compactParts = parts.map(compactCompareSearch);
                return normalizedParts.some((part) => part.includes(normalizedQuery))
                    || compactParts.some((part) => part.includes(compactQuery));
            })
            .slice(0, 8);
    }, [query, rows, searchRows, selectedCodes]);

    const addFund = useCallback((fund: FundSummary) => {
        setSelectedFundCache((current) => ({ ...current, [fund.fund_code]: fund }));
        setSelectedCodes((current) => {
            if (current.length >= FUND_COMPARE_MAX_FUNDS) return current;
            if (current.includes(fund.fund_code)) return current;
            return [...current, fund.fund_code];
        });
        setPicker(null);
        setQuery('');
    }, []);

    const swapFundAt = useCallback((index: number, fund: FundSummary) => {
        setSelectedFundCache((current) => ({ ...current, [fund.fund_code]: fund }));
        setSelectedCodes((current) => {
            if (index < 0 || index >= current.length) return current;
            // If the picked fund is already in another slot, ignore the swap silently.
            if (current.some((code, position) => code === fund.fund_code && position !== index)) {
                return current;
            }
            if (current[index] === fund.fund_code) return current;
            const next = current.slice();
            next[index] = fund.fund_code;
            return next;
        });
        setPicker(null);
        setQuery('');
    }, []);

    const removeFund = useCallback((fundCodeToRemove: string) => {
        setSelectedCodes((current) => current.filter((code) => code !== fundCodeToRemove));
    }, []);

    const openAddPicker = useCallback(() => {
        if (selectedCodes.length >= FUND_COMPARE_MAX_FUNDS) return;
        setPicker({ mode: 'add' });
        setQuery('');
        window.setTimeout(() => {
            inputRef.current?.focus();
            inputRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 0);
    }, [selectedCodes.length]);

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

    // Close swap popover when clicking outside.
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

    const tableEndIso = selectedFunds.find((fund) => fund.as_of)?.as_of || new Date().toISOString().slice(0, 10);
    const valueForFund = (fund: FundSummary, picker: (dataset: FundCompareDataset | undefined, fund: FundSummary) => FundCompareCell) =>
        picker(datasets[fund.fund_code], fund);

    const basicRows: FundCompareTableRow[] = [
        {
            label: 'Fon Fiyatı',
            values: selectedFunds.map((fund) => valueForFund(fund, (dataset, item) => comparisonCell(
                formatQuotePrice(dataset?.detail?.price ?? item.price, dataset?.detail?.currency || item.currency),
                formatDate(dataset?.detail?.as_of || item.as_of),
            ))),
        },
        { label: 'Kurucu', values: selectedFunds.map((fund) => comparisonCell((datasets[fund.fund_code]?.detail || fund).founder_company)) },
        { label: 'Şemsiye', values: selectedFunds.map((fund) => comparisonCell(fundUmbrellaLabel((datasets[fund.fund_code]?.detail || fund).fund_type))) },
        { label: 'TEFAS', values: selectedFunds.map(() => comparisonCell('Açık')) },
    ];
    const metricRows: FundCompareTableRow[] = [
        { label: 'Fon Toplam Değer', values: selectedFunds.map((fund) => comparisonCell(formatCompactCurrency((datasets[fund.fund_code]?.detail || fund).aum))) },
        { label: 'Yatırımcı Sayısı', values: selectedFunds.map((fund) => comparisonCell(formatWholeNumber((datasets[fund.fund_code]?.detail || fund).investor_count))) },
        { label: 'Risk Değeri', values: selectedFunds.map((fund) => {
            const value = (datasets[fund.fund_code]?.detail || fund).risk_value;
            return comparisonCell(value == null ? '-' : `${value} / 7`);
        }) },
        { label: 'Yönetim Ücreti', values: selectedFunds.map((fund) => comparisonCell(formatPlainPct(datasets[fund.fund_code]?.detail?.management_fee))) },
        { label: 'Stopaj Oranı', values: selectedFunds.map((fund) => comparisonCell(datasets[fund.fund_code]?.detail?.tax_info || '-')) },
        { label: 'Pazar Payı', values: selectedFunds.map((fund) => comparisonCell(formatPlainPct(fundMarketShare(datasets[fund.fund_code]?.detail || fund, rows)))) },
    ];
    const allocationRowsForTable: FundCompareTableRow[] = Array.from({ length: 5 }, (_, index) => ({
        label: `${index + 1}. Varlık`,
        values: selectedFunds.map((fund) => {
            const dataset = datasets[fund.fund_code];
            if (dataset?.allocationsLoading && !dataset.allocations) return comparisonCell('Yükleniyor');
            if (dataset?.allocationsError && !dataset.allocations) return comparisonCell('Alınamadı');
            const allocation = topAllocations(dataset?.allocations || null)[index];
            return comparisonCell(allocation?.label, allocation ? `(${formatPlainPct(allocation.weight)})` : undefined);
        }),
    }));
    const positionRows: FundCompareTableRow[] = Array.from({ length: 5 }, (_, index) => ({
        label: `${index + 1}. Hisse`,
        values: selectedFunds.map((fund) => {
            const dataset = datasets[fund.fund_code];
            if (dataset?.holdingsLoading && !dataset.holdings) return comparisonCell('Yükleniyor');
            if (dataset?.holdingsError && !dataset.holdings) return comparisonCell('Alınamadı');
            const position = topHoldingPositions(dataset?.holdings || null)[index];
            return comparisonCell(position?.asset_code || position?.asset_name, position ? formatPlainPct(holdingWeightValue(position)) : undefined);
        }),
    }));
    const sectorRows: FundCompareTableRow[] = Array.from({ length: 5 }, (_, index) => ({
        label: `${index + 1}. Sektör`,
        values: selectedFunds.map((fund) => {
            const dataset = datasets[fund.fund_code];
            if (dataset?.holdingsLoading && !dataset.holdings) return comparisonCell('Yükleniyor');
            if (dataset?.holdingsError && !dataset.holdings) return comparisonCell('Alınamadı');
            const sector = topHoldingSectors(dataset?.holdings || null)[index];
            return comparisonCell(sector?.label, sector ? formatPlainPct(sector.weight) : undefined);
        }),
    }));
    const volumeRows: FundCompareTableRow[] = [
        { label: 'Toplam Adet', values: selectedFunds.map(() => comparisonCell('-')) },
        { label: 'Aktif Adet', values: selectedFunds.map((fund) => {
            const dataset = datasets[fund.fund_code];
            const sortedPoints = sortFundPoints(dataset?.performance?.points);
            const latestPoint = sortedPoints[sortedPoints.length - 1];
            return comparisonCell(formatCompactOverviewNumber(dataset?.detail?.share_count ?? fund.share_count ?? latestPoint?.share_count));
        }) },
        { label: 'Doluluk Oranı', values: selectedFunds.map(() => comparisonCell('-')) },
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
        values: selectedFunds.map((fund) => {
            const detail = datasets[fund.fund_code]?.detail || fund;
            const summaryValue = summaryReturnForCompareRange(detail.period_returns, key as FundCompareRange);
            const calculated = returnForRange(datasets[fund.fund_code]?.performance?.points || [], key as FundCompareRange, tableEndIso);
            const value = finiteNumber(summaryValue) ?? calculated;
            return comparisonCell(formatPct(value), undefined, pctClass(value));
        }),
    }));
    returnRows.push(
        {
            label: 'CAGR (3 Yıl)',
            values: selectedFunds.map((fund) => {
                const value = cagrForRange(datasets[fund.fund_code]?.performance?.points || [], 3, tableEndIso);
                return comparisonCell(formatPct(value), undefined, pctClass(value));
            }),
        },
        {
            label: 'CAGR (5 Yıl)',
            values: selectedFunds.map((fund) => {
                const value = cagrForRange(datasets[fund.fund_code]?.performance?.points || [], 5, tableEndIso);
                return comparisonCell(formatPct(value), undefined, pctClass(value));
            }),
        },
    );
    const cashFlowRows: FundCompareTableRow[] = [
        ['1 Hafta', '1w'],
        ['1 Ay', '1m'],
        ['3 Ay', '3m'],
        ['6 Ay', '6m'],
        ['Yılbaşından Bugüne', 'ytd'],
        ['1 Yıl', '1y'],
    ].map(([label, key]) => ({
        label,
        values: selectedFunds.map((fund) => {
            const value = cashFlowForRange(datasets[fund.fund_code]?.performance?.points || [], key as FundCompareRange, tableEndIso);
            return comparisonCell(formatSignedCompactCurrency(value), undefined, pctClass(value));
        }),
    }));
    const datasetErrors = selectedFunds
        .flatMap((fund) => {
            const dataset = datasets[fund.fund_code];
            return [
                dataset?.performanceError,
                dataset?.detailError,
                dataset?.allocationsError,
                dataset?.holdingsError,
            ];
        })
        .filter((item): item is string => Boolean(item));
    const anyPerformanceLoading = selectedFunds.some((fund) => {
        const dataset = datasets[fund.fund_code];
        return Boolean(dataset?.performanceLoading && !(dataset.performance?.points || []).length);
    });
    const hasChartPerformance = selectedFunds.some((fund) => (datasets[fund.fund_code]?.performance?.points || []).length >= 2);

    return (
        <div className={`fund-compare-page ${searchOpen ? 'is-picker-open' : ''}`}>
            <header className="fund-compare-hero">
                <div className="fund-market-breadcrumb" aria-label="Fon karşılaştırma konumu">
                    <span className="fund-breadcrumb-group">
                        <Database size={17} aria-hidden="true" />
                        Fonlar
                    </span>
                    <ChevronRight size={14} aria-hidden="true" />
                    <span className="fund-breadcrumb-code">Fon Karşılaştırma</span>
                </div>
                <div className="fund-compare-title">
                    <h1>Fon Karşılaştırma</h1>
                    <p>Tüm yatırım fonlarını performans, risk, portföy ve varlık dağılımı gibi kriterlere göre karşılaştırabilirsiniz.</p>
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
                {loading && !selectedFunds.length ? (
                    <div className="fund-compare-chart-empty">Fon listesi yükleniyor...</div>
                ) : error ? (
                    <div className="fund-compare-chart-empty funds-state-error">{error}</div>
                ) : selectedFunds.length === 0 ? (
                    <div className="fund-compare-chart-empty">
                        Karşılaştırmak için en az bir fon ekleyin.
                    </div>
                ) : anyPerformanceLoading && !hasChartPerformance ? (
                    <div className="fund-compare-chart-empty">Grafik verileri yükleniyor...</div>
                ) : (
                    <FundCompareLineChart selectedFunds={selectedFunds} datasets={datasets} range={range} />
                )}
                {datasetErrors[0] && <div className="fund-chart-error">{datasetErrors[0]}</div>}
                <div className="fund-compare-selector-row">
                    {selectedFunds.map((fund, index) => {
                        const swapping = isSwapping(index);
                        const swapResultsToShow = searchResults.filter((row) => row.fund_code !== fund.fund_code);
                        return (
                            <div
                                key={fund.fund_code}
                                className={`fund-compare-card ${swapping ? 'is-swapping' : ''}`}
                                ref={swapping ? swapPopoverRef : undefined}
                            >
                                <span><i style={{ background: FUND_COMPARE_COLORS[index % FUND_COMPARE_COLORS.length] }} /> {index + 1}. Fon</span>
                                <button type="button" className="fund-compare-remove" onClick={() => removeFund(fund.fund_code)} aria-label={`${fund.fund_code} fonunu çıkar`}>
                                    <X size={14} aria-hidden="true" />
                                </button>
                                <div className="fund-compare-card-main">
                                    <Link
                                        to={toFundDetail(fund.fund_code, 'overview')}
                                        className="fund-compare-card-fund"
                                        aria-label={`${fund.fund_code} fonunu aç`}
                                    >
                                        <SymbolLogo symbol={fund.fund_code} name={fund.founder_company || fund.manager_company || fund.name} kind="fund" size="sm" />
                                        <strong>{fund.fund_code}</strong>
                                    </Link>
                                    <button
                                        type="button"
                                        className={`fund-compare-card-swap ${swapping ? 'is-active' : ''}`}
                                        onClick={() => (swapping ? closePicker() : openSwapPicker(index))}
                                        aria-haspopup="dialog"
                                        aria-expanded={swapping}
                                        aria-label={`${fund.fund_code} fonunu değiştir`}
                                    >
                                        <ChevronDown size={16} aria-hidden="true" />
                                    </button>
                                </div>
                                {swapping && (
                                    <div className="fund-compare-card-popover" role="dialog" aria-label={`${index + 1}. fonu değiştir`}>
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
                                                        swapFundAt(index, swapResultsToShow[0]);
                                                    }
                                                }}
                                                placeholder="Fon ara"
                                                aria-label={`${index + 1}. fonu değiştir`}
                                            />
                                            <button type="button" onClick={closePicker} aria-label="Aramayı kapat">
                                                <X size={15} aria-hidden="true" />
                                            </button>
                                        </div>
                                        <div className="fund-compare-search-results fund-compare-search-results-inline">
                                            {query.trim() ? (
                                                swapResultsToShow.length ? swapResultsToShow.map((row) => (
                                                    <button key={row.fund_code} type="button" onClick={() => swapFundAt(index, row)}>
                                                        <SymbolLogo symbol={row.fund_code} name={row.founder_company || row.manager_company || row.name} kind="fund" size="sm" />
                                                        <strong>{row.fund_code}</strong>
                                                        <span>{row.name}</span>
                                                    </button>
                                                )) : (
                                                    <div>{searchLoading ? 'Aranıyor...' : 'Fon bulunamadı.'}</div>
                                                )
                                            ) : (
                                                <div>Fon kodu veya adı yazın.</div>
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
                                            addFund(searchResults[0]);
                                        }
                                    }}
                                    placeholder="Fon ara"
                                    aria-label="Karşılaştırmaya fon ekle"
                                />
                                <button type="button" onClick={closePicker} aria-label="Aramayı kapat">
                                    <X size={15} aria-hidden="true" />
                                </button>
                                <div className="fund-compare-search-results">
                                    {query.trim() ? (
                                        searchResults.length ? searchResults.map((fund) => (
                                            <button key={fund.fund_code} type="button" onClick={() => addFund(fund)}>
                                                <SymbolLogo symbol={fund.fund_code} name={fund.founder_company || fund.manager_company || fund.name} kind="fund" size="sm" />
                                                <strong>{fund.fund_code}</strong>
                                                <span>{fund.name}</span>
                                            </button>
                                        )) : (
                                            <div>{searchLoading ? 'Aranıyor...' : 'Fon bulunamadı.'}</div>
                                        )
                                    ) : (
                                        <div>Fon kodu veya adı yazın.</div>
                                    )}
                                </div>
                            </div>
                        ) : (
                            <button type="button" onClick={openAddPicker} disabled={selectedCodes.length >= FUND_COMPARE_MAX_FUNDS}>
                                <Plus size={18} aria-hidden="true" />
                                Yeni Fon Ekle
                            </button>
                        )}
                    </div>
                </div>
            </section>

            {selectedFunds.length > 0 && (
                <div className="fund-compare-sections">
                    <FundCompareTable title="Temel Bilgiler" selectedFunds={selectedFunds} rows={basicRows} />
                    <FundCompareTable title="Metrikler" selectedFunds={selectedFunds} rows={metricRows} />
                    <FundCompareTable title="Varlık Dağılımı" selectedFunds={selectedFunds} rows={allocationRowsForTable} />
                    <FundCompareTable title="En Büyük Pozisyonlar" selectedFunds={selectedFunds} rows={positionRows} />
                    <FundCompareTable title="Sektör Bilgileri" selectedFunds={selectedFunds} rows={sectorRows} />
                    <FundCompareTable title="Hacim Bilgileri" selectedFunds={selectedFunds} rows={volumeRows} />
                    <FundCompareTable title="Getiri Analizi" selectedFunds={selectedFunds} rows={returnRows} />
                    <FundCompareTable title="Nakit Giriş Çıkış" selectedFunds={selectedFunds} rows={cashFlowRows} />
                </div>
            )}
        </div>
    );
}

export default FundComparisonPage;
