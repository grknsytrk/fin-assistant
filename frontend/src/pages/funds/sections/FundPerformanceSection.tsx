import { useEffect, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react';
import { Calendar, Trophy, X } from 'lucide-react';
import type {
    FundDetail,
    FundPerformanceResponse,
    FundPricePoint,
    FundSummary,
    FundYieldSummaryResponse,
    MarketComparisonHistoryResponse,
} from '../../../api/types';
import { useMobileChartLayout } from '../../../hooks/useMobileChartLayout';
import { FundDatePicker } from '../components/FundDatePicker';
import { FundComparisonControls } from '../components/FundReturnComparison';
import {
    chartDateMs,
    comparisonAssetId,
    downsamplePointsWeekly,
    estimateCashFlow,
    formatChartDate,
    formatCompactCurrency,
    formatCompactOverviewNumber,
    formatCurrency,
    formatDate,
    formatOverviewMetricFullValue,
    formatPct,
    formatSignedCompactCurrency,
    formatSignedInteger,
    formatWholeNumber,
    formatYieldBounds,
    latestFiniteValue,
    nearestChartPoint,
    pctClass,
    returnBetween,
    shiftIsoDateDays,
} from '../formatters';
import {
    DETAIL_RETURN_PERIODS,
    FUND_CHART_COMPARISON_COLORS,
    type ComparisonAsset,
    type FundChartComparisonSeries,
    type FundChartRange,
    type FundComparisonState,
    type FundDetailReturnPeriod,
    type FundOverviewMetricPoint,
    type FundOverviewMetricSeries,
    type MonthlyReturn,
} from '../types';
import './FundPerformanceSection.css';

export function FundChartReturnStrip({
    visiblePeriods = DETAIL_RETURN_PERIODS,
    periodReturns,
    visibleRangeReturn,
    activeRangeCovered = true,
    yieldPeriods,
    currency,
    selectedRange,
    onRangeSelect,
}: {
    visiblePeriods?: readonly FundDetailReturnPeriod[];
    periodReturns: FundSummary['period_returns'] | undefined;
    visibleRangeReturn?: number | null;
    activeRangeCovered?: boolean;
    yieldPeriods?: FundYieldSummaryResponse['periods'];
    currency?: string | null;
    selectedRange: FundChartRange;
    onRangeSelect: (range: FundChartRange) => void;
}) {
    return (
        <div className="fund-chart-return-strip" aria-label="Dönem getirileri">
            {visiblePeriods.map((period) => {
                const isSelected = selectedRange === period.key;
                const isExtended = period.key === '3y' || period.key === '5y';
                const summaryVal = periodReturns?.[period.key];
                const value = isSelected
                    ? (isExtended && !activeRangeCovered ? summaryVal : (visibleRangeReturn ?? summaryVal))
                    : summaryVal;
                const bounds = formatYieldBounds(yieldPeriods?.[period.key], currency || 'TRY');
                return (
                    <button
                        type="button"
                        key={period.key}
                        className={isSelected ? 'is-active' : undefined}
                        aria-pressed={isSelected}
                        onClick={() => onRangeSelect(period.key)}
                        title={bounds || period.label}
                    >
                        <span>{period.label}</span>
                        <strong className={pctClass(value)}>{formatPct(value)}</strong>
                    </button>
                );
            })}
        </div>
    );
}

export function FundPerformanceChart({
    fundCode,
    points,
    comparisonAssets,
    comparisonHistory,
    comparisonLoading,
    comparisonError,
    selectedRange,
    pendingRange,
    rangeError,
    customStartDate,
    customEndDate,
    maxDate,
    visiblePeriods = DETAIL_RETURN_PERIODS,
    activeRangeCovered = true,
    periodReturns,
    historyPending,
    yieldPeriods,
    currency,
    comparison,
    onRangeSelect,
    onCustomStartDateChange,
    onCustomEndDateChange,
}: {
    fundCode: string;
    points: FundPricePoint[];
    comparisonAssets: ComparisonAsset[];
    comparisonHistory: MarketComparisonHistoryResponse | null;
    comparisonLoading: boolean;
    comparisonError: string | null;
    selectedRange: FundChartRange;
    pendingRange: FundChartRange | null;
    rangeError: string | null;
    customStartDate: string;
    customEndDate: string;
    maxDate?: string;
    visiblePeriods?: readonly FundDetailReturnPeriod[];
    activeRangeCovered?: boolean;
    periodReturns: FundSummary['period_returns'] | undefined;
    historyPending: boolean;
    yieldPeriods?: FundYieldSummaryResponse['periods'];
    currency?: string | null;
    comparison: FundComparisonState;
    onRangeSelect: (range: FundChartRange) => void;
    onCustomStartDateChange: (value: string) => void;
    onCustomEndDateChange: (value: string) => void;
}) {
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const [hoverPointer, setHoverPointer] = useState<{ x: number; y: number } | null>(null);
    const [measureAnchorIndex, setMeasureAnchorIndex] = useState<number | null>(null);
    const [selectedMeasureRange, setSelectedMeasureRange] = useState<{ start: number; end: number } | null>(null);
    const isMobileChart = useMobileChartLayout();
    const width = 860;
    const height = isMobileChart ? 760 : 430;
    const padding = isMobileChart
        ? { top: 44, right: 24, bottom: 90, left: 92 }
        : { top: 34, right: 24, bottom: 50, left: 74 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const validPoints = points.filter((point) => Number.isFinite(Number(point.price)) && Number(point.price) > 0);
    const loading = historyPending || pendingRange === selectedRange;
    const pointWindowKey = `${fundCode}:${selectedRange}:${validPoints.length}:${validPoints[0]?.date || ''}:${validPoints[validPoints.length - 1]?.date || ''}`;
    useEffect(() => {
        setHoverIndex(null);
        setHoverPointer(null);
        setMeasureAnchorIndex(null);
        setSelectedMeasureRange(null);
    }, [pointWindowKey]);
    const chartPointerFromEvent = (event: ReactPointerEvent<SVGSVGElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return null;
        const x = ((event.clientX - rect.left) / rect.width) * width;
        const y = ((event.clientY - rect.top) / rect.height) * height;
        return {
            x: Math.min(Math.max(x, 0), width),
            y: Math.min(Math.max(y, 0), height),
        };
    };
    const rangeReturn = validPoints.length >= 2
        ? returnBetween(Number(validPoints[validPoints.length - 1].price), Number(validPoints[0].price))
        : null;
    const visibleRangeReturn = loading ? null : rangeReturn;
    const color = rangeReturn == null || rangeReturn >= 0 ? '#22c55e' : '#ff4d5e';

    const returnStrip = (
        <FundChartReturnStrip
            visiblePeriods={visiblePeriods}
            periodReturns={historyPending ? undefined : periodReturns}
            visibleRangeReturn={visibleRangeReturn}
            activeRangeCovered={activeRangeCovered}
            yieldPeriods={yieldPeriods}
            currency={currency}
            selectedRange={selectedRange}
            onRangeSelect={onRangeSelect}
        />
    );
    const controls = (
        <div className="fund-chart-toolbar">
            <div className="fund-chart-range-row">
                {returnStrip}
            </div>
            <div className="fund-chart-custom-control">
                <button
                    type="button"
                    className={[
                        'fund-chart-range',
                        selectedRange === 'custom' ? 'is-active' : '',
                        pendingRange === 'custom' ? 'is-loading' : '',
                    ].filter(Boolean).join(' ')}
                    title="Özel tarih aralığı seç"
                    aria-pressed={selectedRange === 'custom'}
                    onClick={() => onRangeSelect('custom')}
                >
                    <Calendar size={13} aria-hidden="true" />
                    <span>Özel</span>
                </button>
            </div>
        </div>
    );
    const customRangeInputs = selectedRange === 'custom' ? (
        <div className="fund-custom-range">
            <div className="fund-custom-range-pickers">
                <FundDatePicker
                    label="Başlangıç"
                    value={customStartDate}
                    onChange={onCustomStartDateChange}
                    max={maxDate}
                    rangeStart={customStartDate}
                    rangeEnd={customEndDate}
                />
                <span className="fund-custom-range-sep" aria-hidden="true">–</span>
                <FundDatePicker
                    label="Bitiş"
                    value={customEndDate}
                    onChange={onCustomEndDateChange}
                    max={maxDate}
                    rangeStart={customStartDate}
                    rangeEnd={customEndDate}
                />
            </div>
            <div className="fund-custom-range-summary">
                <span>Dönem Getirisi</span>
                <strong className={pctClass(visibleRangeReturn)}>{formatPct(visibleRangeReturn)}</strong>
            </div>
        </div>
    ) : null;

    if (loading) {
        const loadingTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => padding.top + ratio * plotHeight);
        return (
            <section className="fund-chart-panel" aria-busy="true" aria-label={`${fundCode} fiyat grafiği`}>
                {controls}
                {customRangeInputs}
                <svg
                    className={`fund-detail-chart${isMobileChart ? ' is-mobile-chart' : ''} fund-chart-loading`}
                    viewBox={`0 0 ${width} ${height}`}
                    preserveAspectRatio="xMidYMid meet"
                    role="img"
                    aria-label="Grafik yükleniyor"
                >
                    {loadingTicks.map((y) => (
                        <line key={y} x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="fund-chart-gridline" />
                    ))}
                    <text x={width / 2} y={padding.top + plotHeight / 2} className="fund-chart-loading-label" textAnchor="middle">
                        {historyPending ? 'Getiri geçmişi hazırlanıyor…' : 'Grafik verisi yükleniyor…'}
                    </text>
                </svg>
                {rangeError && <div className="fund-chart-error">{rangeError}</div>}
            </section>
        );
    }

    if (validPoints.length < 2) {
        return (
            <section className="fund-chart-panel">
                {controls}
                {customRangeInputs}
                <div className="fund-chart-empty">{pendingRange ? 'Grafik verisi yükleniyor...' : 'Bu aralık için grafik verisi yok.'}</div>
                {rangeError && <div className="fund-chart-error">{rangeError}</div>}
                {comparisonError && <div className="fund-chart-error">{comparisonError}</div>}
            </section>
        );
    }

    const gradientId = `fund-chart-area-${fundCode.replace(/[^a-zA-Z0-9]/g, '')}-${selectedRange}`;
    const tickIndexes = Array.from(new Set([0, Math.floor((validPoints.length - 1) / 2), validPoints.length - 1]));
    const firstChartDate = validPoints[0]?.date || '';
    const lastChartDate = validPoints[validPoints.length - 1]?.date || '';
    const minHolidayAnchorDate = firstChartDate ? shiftIsoDateDays(firstChartDate, -10) : '';
    const selectedComparisonIds = comparisonAssets.map((asset) => asset.id);
    const comparisonHistoryAssets = comparisonHistory?.assets || [];
    const comparisonHistoryById = new Map(comparisonHistoryAssets.map((asset) => [asset.id, asset]));
    const comparisonPointsById = new Map(
        comparisonHistoryAssets.map((asset) => {
            const orderedValid = (asset.points || [])
                .filter((point) => (
                    Boolean(point.date)
                    && Number.isFinite(Number(point.value))
                    && Number(point.value) > 0
                ))
                .sort((a, b) => a.date.localeCompare(b.date));
            const inWindow = orderedValid.filter((point) => (
                point.date >= firstChartDate && point.date <= lastChartDate
            ));
            if (inWindow.length > 0 && inWindow[0].date > firstChartDate && minHolidayAnchorDate) {
                let preAnchor: (typeof orderedValid)[number] | null = null;
                for (const point of orderedValid) {
                    if (point.date >= firstChartDate) break;
                    preAnchor = point;
                }
                if (preAnchor && preAnchor.date >= minHolidayAnchorDate) {
                    inWindow.unshift({ date: firstChartDate, value: preAnchor.value });
                }
            }
            const sampledWindow = (selectedRange === '3y' || selectedRange === '5y')
                ? downsamplePointsWeekly(inWindow)
                : inWindow;
            return [asset.id, sampledWindow];
        }),
    );
    const comparisonPendingSymbols = comparisonAssets
        .filter((asset) => {
            const historyItem = comparisonHistoryById.get(asset.id);
            return Boolean(historyItem && (historyItem.refresh_pending || historyItem.error === 'refresh_pending'));
        })
        .map((asset) => asset.displaySymbol || asset.symbol);
    const comparisonAssetErrors = comparisonAssets
        .map((asset) => {
            const historyItem = comparisonHistoryById.get(asset.id);
            const err = historyItem?.error;
            if (!err || err === 'refresh_pending' || historyItem?.refresh_pending) return null;
            return `${asset.displaySymbol || asset.symbol}: ${err}`;
        })
        .filter((msg): msg is string => Boolean(msg));
    const comparisonHistoryMatches = comparisonAssets.length > 0
        && Boolean(comparisonHistory)
        && selectedComparisonIds.every((id) => comparisonHistoryById.has(id));
    const hasUsableComparisonSeries = comparisonHistoryMatches
        && selectedComparisonIds.some((id) => (
            (comparisonPointsById.get(id) || []).length >= 2
        ));
    const comparisonMode = comparisonHistoryMatches && hasUsableComparisonSeries;
    const comparisonSeriesDisplayLabel = (
        series: Pick<FundChartComparisonSeries, 'asset' | 'fxConverted' | 'nativeCurrency'>,
    ) => {
        const baseLabel = series.asset.displaySymbol || series.asset.symbol;
        if (series.fxConverted) return `${baseLabel} (TL)`;
        if (series.nativeCurrency && series.nativeCurrency !== 'TRY') return `${baseLabel} (${series.nativeCurrency})`;
        return baseLabel;
    };
    const defaultComparisonLegendItems = [
        {
            id: comparisonAssetId('fund', fundCode),
            color: FUND_CHART_COMPARISON_COLORS[0],
            label: fundCode,
            removable: false,
        },
        ...comparisonAssets.map((asset, index) => {
            const history = comparisonHistoryById.get(asset.id);
            const nativeCurrency = history?.native_currency || asset.nativeCurrency || 'TRY';
            const isForeign = Boolean(nativeCurrency && nativeCurrency !== 'TRY');
            return {
                id: asset.id,
                color: FUND_CHART_COMPARISON_COLORS[(index + 1) % FUND_CHART_COMPARISON_COLORS.length],
                label: comparisonSeriesDisplayLabel({
                    asset,
                    fxConverted: Boolean(history?.fx_converted ?? (isForeign && comparison.fxAdjusted)),
                    nativeCurrency,
                }),
                removable: true,
            };
        }),
    ];
    const renderComparisonLegend = (
        items: Array<{ id: string; color: string; label: string; removable: boolean }>,
    ) => (
        <div className="fund-chart-legend" aria-label="Grafik karşılaştırmaları">
            {items.map((item) => (
                item.removable ? (
                    <button
                        key={item.id}
                        type="button"
                        className="fund-chart-legend-item is-removable"
                        onClick={() => comparison.removeAsset(item.id)}
                        title={`${item.label} karşılaştırmadan çıkar`}
                        aria-label={`${item.label} karşılaştırmadan çıkar`}
                    >
                        <i style={{ background: item.color }} />
                        <span>{item.label}</span>
                        <X size={11} aria-hidden="true" className="fund-chart-legend-remove" />
                    </button>
                ) : (
                    <span key={item.id} className="fund-chart-legend-item">
                        <i style={{ background: item.color }} />
                        <span>{item.label}</span>
                    </span>
                )
            ))}
        </div>
    );
    const waitingForComparison = comparisonAssets.length > 0
        && !comparisonMode
        && (comparisonLoading || !comparisonHistoryMatches)
        && !comparisonError;

    if (waitingForComparison) {
        const yTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => padding.top + ratio * plotHeight);
        return (
            <section className="fund-chart-panel">
                {controls}
                {customRangeInputs}
                <svg
                        className={`fund-detail-chart${isMobileChart ? ' is-mobile-chart' : ''}`}
                    viewBox={`0 0 ${width} ${height}`}
                    preserveAspectRatio="xMidYMid meet"
                    role="img"
                    aria-label={`${fundCode} karşılaştırmalı getiri grafiği yükleniyor`}
                >
                    {yTicks.map((y) => (
                        <line key={y} x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="fund-chart-gridline" />
                    ))}
                    {tickIndexes.map((index) => (
                        <text
                            key={index}
                            x={padding.left + (index / Math.max(1, validPoints.length - 1)) * plotWidth}
                            y={height - (isMobileChart ? 30 : 10)}
                            className="fund-chart-axis"
                            textAnchor={index === 0 ? 'start' : index === validPoints.length - 1 ? 'end' : 'middle'}
                        >
                            {formatChartDate(validPoints[index].date, selectedRange)}
                        </text>
                    ))}
                    <text x={width / 2} y={height / 2} className="fund-chart-loading-label" textAnchor="middle">
                        Karşılaştırma grafiği hazırlanıyor...
                    </text>
                </svg>
                {renderComparisonLegend(defaultComparisonLegendItems)}
                <FundComparisonControls comparison={comparison} surface="chart" className="fund-chart-compare-controls" hideChips />
                {rangeError && <div className="fund-chart-error">{rangeError}</div>}
            </section>
        );
    }

    if (comparisonMode) {
        const historyById = comparisonHistoryById;
        const baseAsset: ComparisonAsset = {
            id: comparisonAssetId('fund', fundCode),
            kind: 'fund',
            symbol: fundCode,
            label: fundCode,
            returns: {},
        };
        const baseSeries: FundChartComparisonSeries = {
            asset: baseAsset,
            points: validPoints.map((point) => ({ date: point.date, value: Number(point.price) })),
            source: 'fund_performance',
            nativeCurrency: (currency || 'TRY').toUpperCase(),
            currency: (currency || 'TRY').toUpperCase(),
            fxConverted: false,
            fxPair: null,
        };
        const externalSeries: FundChartComparisonSeries[] = comparisonAssets.map((asset) => {
            const history = historyById.get(asset.id);
            return {
                asset,
                points: (comparisonPointsById.get(asset.id) || [])
                    .map((point) => ({ date: point.date, value: Number(point.value) })),
                source: history?.source,
                error: history?.error,
                nativeCurrency: history?.native_currency || asset.nativeCurrency || 'TRY',
                currency: history?.currency || asset.nativeCurrency || 'TRY',
                fxConverted: Boolean(history?.fx_converted),
                fxPair: history?.fx_pair || null,
            };
        });
        const rawSeries = [baseSeries, ...externalSeries];
        const normalizedSeries = rawSeries
            .map((series, index) => {
                const ordered = [...series.points].sort((a, b) => a.date.localeCompare(b.date));
                const baseValue = ordered.find((point) => point.value > 0)?.value ?? null;
                return {
                    ...series,
                    color: FUND_CHART_COMPARISON_COLORS[index % FUND_CHART_COMPARISON_COLORS.length],
                    rawPoints: ordered,
                    points: baseValue
                        ? ordered.map((point) => ({
                            date: point.date,
                            value: ((point.value / baseValue) - 1) * 100,
                        }))
                        : [],
                };
            })
            .filter((series) => series.points.length >= 2);
        const returnValues = normalizedSeries.flatMap((series) => series.points.map((point) => point.value));
        const rawMin = Math.min(0, ...returnValues);
        const rawMax = Math.max(0, ...returnValues);
        const rawSpan = Math.max(1, rawMax - rawMin);
        const minValue = rawMin - rawSpan * 0.08;
        const maxValue = rawMax + rawSpan * 0.08;
        const span = Math.max(1, maxValue - minValue);
        const startTs = chartDateMs(validPoints[0].date);
        const endTs = Math.max(startTs + 1, chartDateMs(validPoints[validPoints.length - 1].date));
        const xForDate = (date: string) => {
            const ratio = (chartDateMs(date) - startTs) / (endTs - startTs);
            return padding.left + Math.min(1, Math.max(0, ratio)) * plotWidth;
        };
        const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;
        const yTicks = Array.from({ length: 5 }, (_, index) => maxValue - (span * index) / 4);
        const zeroY = yFor(0);
        const closestIndexForX = (x: number) => {
            let closestIndex = 0;
            let minDiff = Infinity;
            for (let index = 0; index < validPoints.length; index += 1) {
                const diff = Math.abs(xForDate(validPoints[index].date) - x);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestIndex = index;
                }
            }
            return closestIndex;
        };
        const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
            const pointer = chartPointerFromEvent(event);
            if (!pointer) return;
            setHoverPointer(pointer);
            setHoverIndex(closestIndexForX(pointer.x));
        };
        const handlePointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
            if (event.pointerType === 'mouse' && event.button !== 0) return;
            event.preventDefault();
            const pointer = chartPointerFromEvent(event);
            if (!pointer) return;
            const closestIndex = closestIndexForX(pointer.x);
            const rect = event.currentTarget.getBoundingClientRect();
            const handleHitRadius = rect.width > 0 ? (20 * width) / rect.width : 32;
            let anchorIndex = closestIndex;
            let hoverTargetIndex = closestIndex;
            if (selectedMeasureRange) {
                const startDistance = Math.abs(xForDate(validPoints[selectedMeasureRange.start].date) - pointer.x);
                const endDistance = Math.abs(xForDate(validPoints[selectedMeasureRange.end].date) - pointer.x);
                if (Math.min(startDistance, endDistance) <= handleHitRadius) {
                    if (startDistance <= endDistance) {
                        anchorIndex = selectedMeasureRange.end;
                        hoverTargetIndex = selectedMeasureRange.start;
                    } else {
                        anchorIndex = selectedMeasureRange.start;
                        hoverTargetIndex = selectedMeasureRange.end;
                    }
                } else {
                    setSelectedMeasureRange(null);
                }
            }
            setHoverPointer(pointer);
            setHoverIndex(hoverTargetIndex);
            setMeasureAnchorIndex(anchorIndex);
            event.currentTarget.setPointerCapture?.(event.pointerId);
        };
        const handlePointerUp = (event: ReactPointerEvent<SVGSVGElement>) => {
            const pointer = chartPointerFromEvent(event);
            if (pointer && measureAnchorIndex != null) {
                const endIndex = closestIndexForX(pointer.x);
                setHoverPointer(pointer);
                setHoverIndex(endIndex);
                if (measureAnchorIndex !== endIndex) {
                    setSelectedMeasureRange({
                        start: Math.min(measureAnchorIndex, endIndex),
                        end: Math.max(measureAnchorIndex, endIndex),
                    });
                }
            }
            setMeasureAnchorIndex(null);
            try {
                if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
                    event.currentTarget.releasePointerCapture(event.pointerId);
                }
            } catch {
                // Ignore release errors when the browser has already cancelled capture.
            }
        };
        const hoverPoint = hoverIndex == null ? null : validPoints[Math.max(0, Math.min(validPoints.length - 1, hoverIndex))];
        const dragSelection = measureAnchorIndex != null && hoverIndex != null && measureAnchorIndex !== hoverIndex
            ? { start: Math.min(measureAnchorIndex, hoverIndex), end: Math.max(measureAnchorIndex, hoverIndex) }
            : null;
        const visibleMeasureRange = dragSelection || selectedMeasureRange;
        const measureStartPoint = visibleMeasureRange ? validPoints[visibleMeasureRange.start] : null;
        const measureEndPoint = visibleMeasureRange ? validPoints[visibleMeasureRange.end] : null;
        const measureStartX = measureStartPoint ? xForDate(measureStartPoint.date) : null;
        const measureEndX = measureEndPoint ? xForDate(measureEndPoint.date) : null;
        const measureLeftX = measureStartX != null && measureEndX != null ? Math.min(measureStartX, measureEndX) : null;
        const measureRightX = measureStartX != null && measureEndX != null ? Math.max(measureStartX, measureEndX) : null;
        const hasVisibleSelection = Boolean(
            visibleMeasureRange && measureLeftX != null && measureRightX != null && measureRightX > measureLeftX,
        );
        const measureRows = hasVisibleSelection && measureStartPoint && measureEndPoint
            ? normalizedSeries
                .map((series) => {
                    const startRaw = nearestChartPoint(series.rawPoints, measureStartPoint.date);
                    const endRaw = nearestChartPoint(series.rawPoints, measureEndPoint.date);
                    const startNorm = nearestChartPoint(series.points, measureStartPoint.date);
                    const endNorm = nearestChartPoint(series.points, measureEndPoint.date);
                    if (!startRaw || !endRaw || !startNorm || !endNorm) return null;
                    const periodReturn = returnBetween(endRaw.value, startRaw.value);
                    if (periodReturn == null) return null;
                    return {
                        series,
                        startNorm,
                        endNorm,
                        periodReturn,
                    };
                })
                .filter((item): item is {
                    series: typeof normalizedSeries[number];
                    startNorm: { date: string; value: number };
                    endNorm: { date: string; value: number };
                    periodReturn: number;
                } => Boolean(item))
                .slice(0, 6)
            : [];
        const measureTooltipWidth = isMobileChart ? 350 : 224;
        const measureTooltipHeight = isMobileChart
            ? Math.max(92, 48 + measureRows.length * 32)
            : Math.max(54, 30 + measureRows.length * 18);
        const measureTooltipGap = isMobileChart ? 20 : 14;
        const measureTooltipX = measureLeftX == null || measureRightX == null
            ? 0
            : measureRightX + measureTooltipGap + measureTooltipWidth <= width - 8
                ? measureRightX + measureTooltipGap
                : measureLeftX - measureTooltipGap - measureTooltipWidth >= padding.left + 8
                    ? measureLeftX - measureTooltipGap - measureTooltipWidth
                    : Math.min(Math.max(padding.left + 8, measureRightX - measureTooltipWidth - 8), width - measureTooltipWidth - 8);
        const measureAnchorYCenter = measureRows.length > 0
            ? measureRows.reduce((sum, row) => sum + yFor(row.endNorm.value), 0) / measureRows.length
            : padding.top + plotHeight * 0.25;
        const rawMeasureTooltipY = (hoverPointer?.y ?? measureAnchorYCenter) - measureTooltipHeight * 0.5;
        const maxMeasureTooltipY = Math.max(padding.top + 8, height - padding.bottom - measureTooltipHeight - 36);
        const measureTooltipY = Math.min(Math.max(padding.top + 8, rawMeasureTooltipY), maxMeasureTooltipY);
        const selectionClipId = `${gradientId}-compare-selection`;
        const hoverX = hoverPoint ? xForDate(hoverPoint.date) : null;
        const tooltipRows = hoverPoint
            ? normalizedSeries
                .map((series) => {
                    const point = nearestChartPoint(series.points, hoverPoint.date);
                    return point ? { series, point } : null;
                })
                .filter((item): item is { series: typeof normalizedSeries[number]; point: { date: string; value: number } } => Boolean(item))
                .slice(0, 4)
            : [];
        const tooltipWidth = isMobileChart ? 340 : 212;
        const tooltipHeight = isMobileChart
            ? Math.max(86, 42 + tooltipRows.length * 30)
            : Math.max(48, 24 + tooltipRows.length * 15);
        const tooltipGap = isMobileChart ? 18 : 12;
        const tooltipAnchorX = hoverPointer?.x ?? hoverX;
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

        return (
            <section className="fund-chart-panel">
                {controls}
                {customRangeInputs}
                <svg
                    key={`${fundCode}-${selectedRange}-${validPoints.length}-compare-${comparison.fxAdjusted ? 'try' : 'raw'}-${comparisonAssets.map((asset) => asset.id).join('-')}`}
                    className={`fund-detail-chart${isMobileChart ? ' is-mobile-chart' : ''}`}
                    viewBox={`0 0 ${width} ${height}`}
                    preserveAspectRatio="xMidYMid meet"
                    role="img"
                    aria-label={`${fundCode} karşılaştırmalı getiri grafiği`}
                    onPointerMove={handlePointerMove}
                    onPointerDown={handlePointerDown}
                    onPointerUp={handlePointerUp}
                    onPointerCancel={handlePointerUp}
                    onPointerLeave={() => {
                        setHoverIndex(null);
                        setHoverPointer(null);
                        setMeasureAnchorIndex(null);
                    }}
                >
                    <defs>
                        <clipPath id={`${gradientId}-compare-clip`}>
                            <rect x={padding.left} y={0} width={plotWidth} height={height} className="fund-chart-reveal" />
                        </clipPath>
                        {hasVisibleSelection && measureLeftX != null && measureRightX != null && (
                            <clipPath id={selectionClipId}>
                                <rect x={measureLeftX} y={padding.top} width={measureRightX - measureLeftX} height={plotHeight} />
                            </clipPath>
                        )}
                    </defs>
                    {yTicks.map((tick) => {
                        const y = yFor(tick);
                        return (
                            <g key={tick}>
                                <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="fund-chart-gridline" />
                                <text x={padding.left - 10} y={y + 4} className="fund-chart-axis" textAnchor="end">
                                    {formatPct(tick).replace('+', '')}
                                </text>
                            </g>
                        );
                    })}
                    <line x1={padding.left} x2={width - padding.right} y1={zeroY} y2={zeroY} className="fund-comparison-zero" />
                    {tickIndexes.map((index) => (
                        <text
                            key={index}
                            x={xForDate(validPoints[index].date)}
                            y={height - (isMobileChart ? 30 : 10)}
                            className="fund-chart-axis"
                            textAnchor={index === 0 ? 'start' : index === validPoints.length - 1 ? 'end' : 'middle'}
                        >
                            {formatChartDate(validPoints[index].date, selectedRange)}
                        </text>
                    ))}
                    <g clipPath={`url(#${gradientId}-compare-clip)`}>
                        {normalizedSeries.map((series) => {
                            const pathData = series.points
                                .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xForDate(point.date)} ${yFor(point.value)}`)
                                .join(' ');
                            return (
                                <path
                                    key={series.asset.id}
                                    d={pathData}
                                    fill="none"
                                    stroke={hasVisibleSelection ? '#89939e' : series.color}
                                    strokeWidth={series.asset.id === baseAsset.id ? 3 : 2.2}
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    className="fund-chart-line"
                                />
                            );
                        })}
                        {normalizedSeries.flatMap((series) => (
                            series.points.map((point, index) => (
                                <circle
                                    key={`${series.asset.id}-node-${index}`}
                                    className="fund-chart-node"
                                    cx={xForDate(point.date)}
                                    cy={yFor(point.value)}
                                    r={series.asset.id === baseAsset.id ? 1.9 : 1.6}
                                    fill={hasVisibleSelection ? '#89939e' : series.color}
                                />
                            ))
                        ))}
                    </g>
                    {normalizedSeries.map((series) => {
                        const lastPoint = series.points[series.points.length - 1];
                        return lastPoint ? (
                            <circle
                                key={`${series.asset.id}-endpoint`}
                                className="fund-chart-endpoint"
                                cx={xForDate(lastPoint.date)}
                                cy={yFor(lastPoint.value)}
                                r={series.asset.id === baseAsset.id ? 4 : 3.4}
                                fill={hasVisibleSelection ? '#89939e' : series.color}
                            />
                        ) : null;
                    })}
                    {hasVisibleSelection && measureLeftX != null && measureRightX != null && (
                        <g className="fund-chart-selection" aria-hidden="true">
                            <rect x={padding.left} y={padding.top} width={Math.max(0, measureLeftX - padding.left)} height={plotHeight} className="fund-chart-selection-outside" />
                            <rect x={measureRightX} y={padding.top} width={Math.max(0, width - padding.right - measureRightX)} height={plotHeight} className="fund-chart-selection-outside" />
                            <rect x={measureLeftX} y={padding.top} width={measureRightX - measureLeftX} height={plotHeight} className="fund-chart-selection-band" style={{ color: normalizedSeries[0]?.color, fill: normalizedSeries[0]?.color }} />
                            <g clipPath={`url(#${selectionClipId})`}>
                                {normalizedSeries.map((series) => {
                                    const pathData = series.points
                                        .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xForDate(point.date)} ${yFor(point.value)}`)
                                        .join(' ');
                                    return (
                                        <g key={`${series.asset.id}-selected`}>
                                            <path d={pathData} fill="none" stroke={series.color} strokeWidth={series.asset.id === baseAsset.id ? 3 : 2.2} strokeLinecap="round" strokeLinejoin="round" />
                                            {series.points.map((point, index) => (
                                                <circle key={`${series.asset.id}-selected-node-${index}`} cx={xForDate(point.date)} cy={yFor(point.value)} r={series.asset.id === baseAsset.id ? 1.9 : 1.6} fill={series.color} />
                                            ))}
                                        </g>
                                    );
                                })}
                            </g>
                            {[measureLeftX, measureRightX].map((x, index) => (
                                <g key={`${x}-${index}`}>
                                    <line x1={x} x2={x} y1={padding.top} y2={height - padding.bottom} className="fund-chart-selection-edge" />
                                    <rect x={x - 18} y={height - padding.bottom - 24} width="36" height="30" rx="7" className="fund-chart-selection-handle" />
                                    <line x1={x - 6} x2={x - 6} y1={height - padding.bottom - 17} y2={height - padding.bottom - 1} className="fund-chart-selection-grip" />
                                    <line x1={x + 6} x2={x + 6} y1={height - padding.bottom - 17} y2={height - padding.bottom - 1} className="fund-chart-selection-grip" />
                                </g>
                            ))}
                        </g>
                    )}
                    {hasVisibleSelection && measureStartPoint && measureEndPoint && measureRows.length > 0 && (
                        <g className="fund-chart-measure">
                            {measureRows.map(({ series, startNorm, endNorm }) => (
                                <g key={`${series.asset.id}-measure-dots`}>
                                    <circle
                                        cx={xForDate(startNorm.date)}
                                        cy={yFor(startNorm.value)}
                                        r={3.8}
                                        className="fund-chart-measure-point"
                                        fill={series.color}
                                    />
                                    <circle
                                        cx={xForDate(endNorm.date)}
                                        cy={yFor(endNorm.value)}
                                        r={3.8}
                                        className="fund-chart-measure-point"
                                        fill={series.color}
                                    />
                                </g>
                            ))}
                            <g className="fund-chart-measure-tooltip" style={{ transform: `translate(${measureTooltipX}px, ${measureTooltipY}px)` }}>
                                <rect width={measureTooltipWidth} height={measureTooltipHeight} rx="7" className="fund-chart-measure-tooltip-bg" />
                                <text x="10" y={isMobileChart ? 28 : 18} className="fund-chart-measure-muted">
                                    {formatChartDate(measureStartPoint.date, selectedRange)} - {formatChartDate(measureEndPoint.date, selectedRange)}
                                </text>
                                {measureRows.map(({ series, periodReturn }, index) => (
                                    <g key={`${series.asset.id}-measure-row`} transform={`translate(10, ${isMobileChart ? 60 + index * 32 : 38 + index * 18})`}>
                                        <circle cx="3.5" cy="-4" r="2.8" fill={series.color} />
                                        <text x="13" y="0" className="fund-chart-measure-muted">
                                            {comparisonSeriesDisplayLabel(series).slice(0, 14)}
                                        </text>
                                        <text
                                            x={measureTooltipWidth - 20}
                                            y="0"
                                            textAnchor="end"
                                            className={`fund-chart-measure-row-value ${periodReturn >= 0 ? 'is-positive' : 'is-negative'}`}
                                        >
                                            {formatPct(periodReturn)}
                                        </text>
                                    </g>
                                ))}
                            </g>
                        </g>
                    )}
                    {measureAnchorIndex == null && selectedMeasureRange == null && hoverPoint && hoverX != null && tooltipRows.length > 0 && (
                        <g>
                            <line x1={hoverX} x2={hoverX} y1={padding.top} y2={height - padding.bottom} className="fund-chart-hoverline" />
                            {tooltipRows.map(({ series, point }) => (
                                <circle
                                    key={`${series.asset.id}-hover`}
                                    className="fund-chart-hover-point"
                                    cx={xForDate(point.date)}
                                    cy={yFor(point.value)}
                                    r={series.asset.id === baseAsset.id ? 4 : 3.5}
                                    fill={series.color}
                                    stroke="#0a0c0f"
                                    strokeWidth="1.6"
                                />
                            ))}
                            <g className="fund-chart-tooltip" style={{ transform: `translate(${tooltipX}px, ${tooltipY}px)` }}>
                                <rect width={tooltipWidth} height={tooltipHeight} rx="6" className="fund-chart-tooltip-bg" />
                                <text x="8" y={isMobileChart ? 27 : 17} className="fund-chart-tooltip-muted">{formatDate(hoverPoint.date)}</text>
                                {tooltipRows.map(({ series, point }, index) => (
                                    <g key={series.asset.id} transform={`translate(8, ${isMobileChart ? 52 + index * 29 : 35 + index * 15})`}>
                                        <circle cx="3.5" cy="-4" r="2.6" fill={series.color} />
                                        <text x="12" y="0" className="fund-chart-tooltip-muted">
                                            {comparisonSeriesDisplayLabel(series).slice(0, 14)}
                                        </text>
                                        <text x={tooltipWidth - 10} y="0" className="fund-chart-tooltip-value" textAnchor="end">
                                            {formatPct(point.value)}
                                        </text>
                                    </g>
                                ))}
                            </g>
                        </g>
                    )}
                </svg>
                {renderComparisonLegend(
                    normalizedSeries.map((series) => ({
                        id: series.asset.id,
                        color: series.color,
                        label: comparisonSeriesDisplayLabel(series),
                        removable: series.asset.id !== baseAsset.id,
                    })),
                )}
                <FundComparisonControls comparison={comparison} surface="chart" className="fund-chart-compare-controls" hideChips />
                {normalizedSeries.some((series) => series.fxConverted) && (
                    <div className="fund-chart-note">
                        Yabancı endeksler günlük kapanış kurlarıyla (USD/TRY, EUR/TRY vb.) TL bazlı getiriye endekslenmiştir.
                    </div>
                )}
                {comparisonPendingSymbols.length > 0 && (
                    <div className="fund-chart-note">
                        Karşılaştırma geçmişi hazırlanıyor ({comparisonPendingSymbols.join(', ')})...
                    </div>
                )}
                {rangeError && <div className="fund-chart-error">{rangeError}</div>}
                {comparisonError && <div className="fund-chart-error">{comparisonError}</div>}
                {comparisonAssetErrors.length > 0 && (
                    <div className="fund-chart-error">{comparisonAssetErrors.join(' · ')}</div>
                )}
            </section>
        );
    }

    const prices = validPoints.map((point) => Number(point.price));
    let minValue = Math.min(...prices);
    let maxValue = Math.max(...prices);
    const rawSpan = Math.max(0.01, maxValue - minValue);
    minValue -= rawSpan * 0.08;
    maxValue += rawSpan * 0.08;
    const span = Math.max(0.01, maxValue - minValue);
    const xFor = (index: number) => padding.left + (validPoints.length === 1 ? 0 : (index / (validPoints.length - 1)) * plotWidth);
    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;
    const pathData = validPoints.map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(index)} ${yFor(Number(point.price))}`).join(' ');
    const areaData = `${pathData} L ${xFor(validPoints.length - 1)} ${height - padding.bottom} L ${padding.left} ${height - padding.bottom} Z`;
    const yTicks = [0, 0.5, 1].map((ratio) => maxValue - ratio * span);

    const closestPriceIndexForX = (x: number) => {
        let closestIndex = 0;
        let minDiff = Infinity;
        for (let index = 0; index < validPoints.length; index += 1) {
            const diff = Math.abs(xFor(index) - x);
            if (diff < minDiff) {
                minDiff = diff;
                closestIndex = index;
            }
        }
        return closestIndex;
    };
    const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const pointer = chartPointerFromEvent(event);
        if (!pointer) return;
        setHoverPointer(pointer);
        setHoverIndex(closestPriceIndexForX(pointer.x));
        };
        const handlePointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
            if (event.pointerType === 'mouse' && event.button !== 0) return;
            event.preventDefault();
        const pointer = chartPointerFromEvent(event);
        if (!pointer) return;
        const closestIndex = closestPriceIndexForX(pointer.x);
        const rect = event.currentTarget.getBoundingClientRect();
        const handleHitRadius = rect.width > 0 ? (20 * width) / rect.width : 32;
        let anchorIndex = closestIndex;
        let hoverTargetIndex = closestIndex;
        if (selectedMeasureRange) {
            const startDistance = Math.abs(xFor(selectedMeasureRange.start) - pointer.x);
            const endDistance = Math.abs(xFor(selectedMeasureRange.end) - pointer.x);
            if (Math.min(startDistance, endDistance) <= handleHitRadius) {
                if (startDistance <= endDistance) {
                    anchorIndex = selectedMeasureRange.end;
                    hoverTargetIndex = selectedMeasureRange.start;
                } else {
                    anchorIndex = selectedMeasureRange.start;
                    hoverTargetIndex = selectedMeasureRange.end;
                }
            } else {
                setSelectedMeasureRange(null);
            }
        }
        setHoverPointer(pointer);
        setHoverIndex(hoverTargetIndex);
        setMeasureAnchorIndex(anchorIndex);
        event.currentTarget.setPointerCapture?.(event.pointerId);
    };
    const handlePointerUp = (event: ReactPointerEvent<SVGSVGElement>) => {
        const pointer = chartPointerFromEvent(event);
        if (pointer && measureAnchorIndex != null) {
            const endIndex = closestPriceIndexForX(pointer.x);
            setHoverPointer(pointer);
            setHoverIndex(endIndex);
            if (measureAnchorIndex !== endIndex) {
                setSelectedMeasureRange({
                    start: Math.min(measureAnchorIndex, endIndex),
                    end: Math.max(measureAnchorIndex, endIndex),
                });
            }
        }
        setMeasureAnchorIndex(null);
        try {
            if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
            }
        } catch {
            // Ignore release errors when the browser has already cancelled capture.
        }
    };
    const hoverPoint = hoverIndex == null ? null : validPoints[Math.max(0, Math.min(validPoints.length - 1, hoverIndex))];
    const hoverPointIndex = hoverPoint ? validPoints.indexOf(hoverPoint) : null;
    const hoverX = hoverPointIndex == null ? null : xFor(hoverPointIndex);
    const hoverY = hoverPoint ? yFor(Number(hoverPoint.price)) : null;
    const dragSelection = measureAnchorIndex != null && hoverPointIndex != null && measureAnchorIndex !== hoverPointIndex
        ? { start: Math.min(measureAnchorIndex, hoverPointIndex), end: Math.max(measureAnchorIndex, hoverPointIndex) }
        : null;
    const visibleMeasureRange = dragSelection || selectedMeasureRange;
    const measureAnchorPoint = visibleMeasureRange ? validPoints[visibleMeasureRange.start] : null;
    const measureEndPoint = visibleMeasureRange ? validPoints[visibleMeasureRange.end] : null;
    const measureAnchorX = visibleMeasureRange ? xFor(visibleMeasureRange.start) : null;
    const measureAnchorY = measureAnchorPoint ? yFor(Number(measureAnchorPoint.price)) : null;
    const measureEndX = visibleMeasureRange ? xFor(visibleMeasureRange.end) : null;
    const measureEndY = measureEndPoint ? yFor(Number(measureEndPoint.price)) : null;
    const measureStartPrice = measureAnchorPoint ? Number(measureAnchorPoint.price) : null;
    const measureEndPrice = measureEndPoint ? Number(measureEndPoint.price) : null;
    const measureDelta = measureStartPrice != null && measureEndPrice != null ? measureEndPrice - measureStartPrice : null;
    const measureReturn = returnBetween(measureEndPrice, measureStartPrice);
    const measureLineY = measureAnchorY != null && measureEndY != null
        ? Math.max(padding.top + (isMobileChart ? 28 : 12), Math.min(measureAnchorY, measureEndY) - (isMobileChart ? 34 : 18))
        : null;
    const measureTooltipWidth = isMobileChart ? 300 : 180;
    const measureTooltipHeight = isMobileChart ? 112 : 70;
    const measureTooltipGap = isMobileChart ? 22 : 14;
    const measureAnchorXForTooltip = measureEndX;
    const measureAnchorYForTooltip = measureEndY;
    const measureTooltipShouldFlip = measureAnchorXForTooltip != null
        && measureAnchorXForTooltip + measureTooltipGap + measureTooltipWidth > width - 8;
    const measureTooltipX = measureAnchorXForTooltip == null
        ? 0
        : measureTooltipShouldFlip
            ? Math.max(8, measureAnchorXForTooltip - measureTooltipGap - measureTooltipWidth)
            : Math.min(measureAnchorXForTooltip + measureTooltipGap, width - measureTooltipWidth - 8);
    const measureTooltipY = measureAnchorYForTooltip == null
        ? 0
        : measureAnchorYForTooltip - measureTooltipGap - measureTooltipHeight >= padding.top
            ? measureAnchorYForTooltip - measureTooltipGap - measureTooltipHeight
            : Math.min(measureAnchorYForTooltip + measureTooltipGap, height - measureTooltipHeight - 8);
    const measureDeltaText = measureDelta == null
        ? '-'
        : `${measureDelta > 0 ? '+' : measureDelta < 0 ? '-' : ''}${formatCurrency(Math.abs(measureDelta), 'TRY')}`;
    const measureClass = measureReturn == null ? '' : measureReturn >= 0 ? 'is-positive' : 'is-negative';
    const measureLeftX = measureAnchorX != null && measureEndX != null ? Math.min(measureAnchorX, measureEndX) : null;
    const measureRightX = measureAnchorX != null && measureEndX != null ? Math.max(measureAnchorX, measureEndX) : null;
    const hasVisibleSelection = Boolean(
        visibleMeasureRange && measureLeftX != null && measureRightX != null && measureRightX > measureLeftX,
    );
    const selectionClipId = `${gradientId}-selection`;
    const tooltipWidth = isMobileChart ? 250 : 152;
    const tooltipHeight = isMobileChart ? 90 : 58;
    const tooltipGap = 12;
    const tooltipAnchorX = hoverPointer?.x ?? hoverX;
    const tooltipAnchorY = hoverPointer?.y ?? hoverY;
    const tooltipShouldFlip = tooltipAnchorX != null && tooltipAnchorX + tooltipGap + tooltipWidth > width - 8;
    const tooltipX = tooltipAnchorX == null
        ? 0
        : tooltipShouldFlip
            ? Math.max(8, tooltipAnchorX - tooltipGap - tooltipWidth)
            : Math.min(tooltipAnchorX + tooltipGap, width - tooltipWidth - 8);
    const tooltipY = tooltipAnchorY == null
        ? 0
        : tooltipAnchorY - tooltipGap - tooltipHeight >= padding.top
            ? tooltipAnchorY - tooltipGap - tooltipHeight
            : Math.min(tooltipAnchorY + tooltipGap, height - tooltipHeight - 8);

    return (
        <section className="fund-chart-panel">
            {controls}
            {customRangeInputs}
            <svg
                key={`${fundCode}-${selectedRange}-${validPoints.length}`}
                className={`fund-detail-chart${isMobileChart ? ' is-mobile-chart' : ''}`}
                viewBox={`0 0 ${width} ${height}`}
                preserveAspectRatio="xMidYMid meet"
                role="img"
                aria-label={`${fundCode} fiyat grafiği`}
                onPointerMove={handlePointerMove}
                onPointerDown={handlePointerDown}
                onPointerUp={handlePointerUp}
                onPointerCancel={handlePointerUp}
                onPointerLeave={() => {
                    setHoverIndex(null);
                    setHoverPointer(null);
                    setMeasureAnchorIndex(null);
                }}
            >
                <defs>
                    <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor={hasVisibleSelection ? '#9aa3ad' : color} stopOpacity={hasVisibleSelection ? '0.14' : '0.24'} />
                        <stop offset="100%" stopColor={hasVisibleSelection ? '#9aa3ad' : color} stopOpacity="0" />
                    </linearGradient>
                    <linearGradient id={`${gradientId}-selected-area`} x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor={color} stopOpacity="0.28" />
                        <stop offset="100%" stopColor={color} stopOpacity="0.02" />
                    </linearGradient>
                    <clipPath id={`${gradientId}-clip`}>
                        <rect x={padding.left} y={0} width={plotWidth} height={height} className="fund-chart-reveal" />
                    </clipPath>
                    {hasVisibleSelection && measureLeftX != null && measureRightX != null && (
                        <clipPath id={selectionClipId}>
                            <rect x={measureLeftX} y={padding.top} width={measureRightX - measureLeftX} height={plotHeight} />
                        </clipPath>
                    )}
                </defs>
                {yTicks.map((tick) => {
                    const y = yFor(tick);
                    return (
                        <g key={tick}>
                            <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="fund-chart-gridline" />
                            <text x={padding.left - 10} y={y + 4} className="fund-chart-axis" textAnchor="end">
                                {tick.toLocaleString('tr-TR', { maximumFractionDigits: 1 })}
                            </text>
                        </g>
                    );
                })}
                {tickIndexes.map((index) => (
                    <text
                        key={index}
                        x={xFor(index)}
                        y={height - (isMobileChart ? 30 : 10)}
                        className="fund-chart-axis"
                        textAnchor={index === 0 ? 'start' : index === validPoints.length - 1 ? 'end' : 'middle'}
                    >
                        {formatChartDate(validPoints[index].date, selectedRange)}
                    </text>
                ))}
                <g clipPath={`url(#${gradientId}-clip)`}>
                    <path d={areaData} fill={`url(#${gradientId})`} />
                    <path d={pathData} fill="none" stroke={hasVisibleSelection ? '#89939e' : color} strokeWidth="2.7" strokeLinecap="round" strokeLinejoin="round" />
                </g>
                <circle
                    className="fund-chart-endpoint"
                    cx={xFor(validPoints.length - 1)}
                    cy={yFor(Number(validPoints[validPoints.length - 1].price))}
                    r="4"
                    fill={hasVisibleSelection ? '#89939e' : color}
                />
                {hasVisibleSelection && measureLeftX != null && measureRightX != null && (
                    <g className="fund-chart-selection" style={{ color }} aria-hidden="true">
                        <rect x={padding.left} y={padding.top} width={Math.max(0, measureLeftX - padding.left)} height={plotHeight} className="fund-chart-selection-outside" />
                        <rect x={measureRightX} y={padding.top} width={Math.max(0, width - padding.right - measureRightX)} height={plotHeight} className="fund-chart-selection-outside" />
                        <rect x={measureLeftX} y={padding.top} width={measureRightX - measureLeftX} height={plotHeight} className="fund-chart-selection-band" />
                        <path d={areaData} fill={`url(#${gradientId}-selected-area)`} clipPath={`url(#${selectionClipId})`} />
                        <path d={pathData} fill="none" stroke={color} strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" clipPath={`url(#${selectionClipId})`} />
                        {[measureLeftX, measureRightX].map((x, index) => (
                            <g key={`${x}-${index}`}>
                                <line x1={x} x2={x} y1={padding.top} y2={height - padding.bottom} className="fund-chart-selection-edge" />
                                <rect x={x - 18} y={height - padding.bottom - 24} width="36" height="30" rx="7" className="fund-chart-selection-handle" />
                                <line x1={x - 6} x2={x - 6} y1={height - padding.bottom - 17} y2={height - padding.bottom - 1} className="fund-chart-selection-grip" />
                                <line x1={x + 6} x2={x + 6} y1={height - padding.bottom - 17} y2={height - padding.bottom - 1} className="fund-chart-selection-grip" />
                            </g>
                        ))}
                    </g>
                )}
                {measureAnchorPoint
                    && measureEndPoint
                    && measureAnchorX != null
                    && measureAnchorY != null
                    && measureEndX != null
                    && measureEndY != null
                    && measureLeftX != null
                    && measureRightX != null
                    && measureLineY != null
                    && measureReturn != null
                    && measureDelta != null && (
                    <g className="fund-chart-measure">
                        <line
                            x1={measureAnchorX}
                            y1={measureAnchorY}
                            x2={measureEndX}
                            y2={measureEndY}
                            className="fund-chart-measure-path"
                        />
                        <line
                            x1={measureLeftX}
                            y1={measureLineY}
                            x2={measureRightX}
                            y2={measureLineY}
                            className="fund-chart-measure-line"
                        />
                        <line
                            x1={measureAnchorX}
                            y1={measureLineY - 5}
                            x2={measureAnchorX}
                            y2={measureLineY + 5}
                            className="fund-chart-measure-line"
                        />
                        <line
                            x1={measureEndX}
                            y1={measureLineY - 5}
                            x2={measureEndX}
                            y2={measureLineY + 5}
                            className="fund-chart-measure-line"
                        />
                        <circle cx={measureAnchorX} cy={measureAnchorY} r="4.2" className="fund-chart-measure-point" fill={color} />
                        <circle cx={measureEndX} cy={measureEndY} r="4.2" className="fund-chart-measure-point" fill={color} />
                        <g className="fund-chart-measure-tooltip" style={{ transform: `translate(${measureTooltipX}px, ${measureTooltipY}px)` }}>
                            <rect width={measureTooltipWidth} height={measureTooltipHeight} rx="7" className="fund-chart-measure-tooltip-bg" />
                            <text x="10" y={isMobileChart ? 30 : 18} className="fund-chart-measure-muted">
                                {formatChartDate(measureAnchorPoint.date, selectedRange)} - {formatChartDate(measureEndPoint.date, selectedRange)}
                            </text>
                            <text x="10" y={isMobileChart ? 65 : 42} className={`fund-chart-measure-value ${measureClass}`}>
                                {formatPct(measureReturn)}
                            </text>
                            <text x="10" y={isMobileChart ? 96 : 60} className="fund-chart-measure-muted">
                                {measureDeltaText}
                            </text>
                        </g>
                    </g>
                )}
                {measureAnchorIndex == null && selectedMeasureRange == null && hoverPoint && hoverX != null && hoverY != null && (
                    <g>
                        <line x1={hoverX} x2={hoverX} y1={padding.top} y2={height - padding.bottom} className="fund-chart-hoverline" />
                        <circle className="fund-chart-hover-point" cx={hoverX} cy={hoverY} r="3.6" fill={color} stroke="#0a0c0f" strokeWidth="1.6" />
                        <g className="fund-chart-tooltip" style={{ transform: `translate(${tooltipX}px, ${tooltipY}px)` }}>
                            <rect width={tooltipWidth} height={tooltipHeight} rx="6" className="fund-chart-tooltip-bg" />
                            <text x="10" y={isMobileChart ? 31 : 20} className="fund-chart-tooltip-muted">{formatDate(hoverPoint.date)}</text>
                            <text x="10" y={isMobileChart ? 65 : 43} className="fund-chart-tooltip-value">{formatCurrency(hoverPoint.price, currency || 'TRY')}</text>
                        </g>
                    </g>
                )}
            </svg>
            {comparisonAssets.length > 0 && renderComparisonLegend(defaultComparisonLegendItems)}
            <FundComparisonControls comparison={comparison} surface="chart" className="fund-chart-compare-controls" hideChips />
            {(comparisonPendingSymbols.length > 0 || (comparisonLoading && comparisonAssets.length > 0)) && (
                <div className="fund-chart-note">
                    {comparisonPendingSymbols.length > 0
                        ? `Karşılaştırma geçmişi hazırlanıyor (${comparisonPendingSymbols.join(', ')})...`
                        : 'Karşılaştırma geçmişi yükleniyor...'}
                </div>
            )}
            {comparisonError && <div className="fund-chart-error">{comparisonError}</div>}
            {comparisonAssetErrors.length > 0 && (
                <div className="fund-chart-error">{comparisonAssetErrors.join(' · ')}</div>
            )}
            {rangeError && <div className="fund-chart-error">{rangeError}</div>}
        </section>
    );
}

export type FundOverviewMetricCardProps = {
    title: string;
    points: FundOverviewMetricPoint[];
    suffix?: string;
};

export function FundOverviewMetricCard({ title, points, suffix }: FundOverviewMetricCardProps) {
    const latest = latestFiniteValue(points);
    const finiteValues = points.map((point) => point.value).filter((value): value is number => value != null && Number.isFinite(value));
    const maxAbs = Math.max(1, ...finiteValues.map((value) => Math.abs(value)));

    return (
        <section className="fund-overview-metric-card" aria-label={title}>
            <div className="fund-overview-metric-copy">
                <span>{title}</span>
                <strong className={latest != null && latest < 0 ? 'is-negative' : undefined}>
                    {formatWholeNumber(latest)}
                    {suffix ? <small>{suffix}</small> : null}
                </strong>
            </div>
            <div className="fund-overview-metric-bars">
                {points.map((point) => {
                    const value = point.value;
                    const hasValue = value != null && Number.isFinite(value);
                    const height = hasValue ? Math.max(4, (Math.abs(value) / maxAbs) * 100) : 0;
                    const fullValue = formatOverviewMetricFullValue(value, suffix);
                    return (
                        <span
                            className="fund-overview-metric-bar"
                            key={point.month}
                            title={`${point.label}: ${fullValue}`}
                            tabIndex={0}
                            aria-label={`${point.label}: ${fullValue}`}
                        >
                            <span className="fund-overview-metric-tooltip">
                                <b>{point.label}</b>
                                <strong className={hasValue && value < 0 ? 'is-negative' : undefined}>
                                    {hasValue ? formatCompactOverviewNumber(value) : 'Veri yok'}
                                </strong>
                            </span>
                            <i
                                className={!hasValue ? 'is-missing' : value < 0 ? 'is-negative' : undefined}
                                style={{ '--metric-height': `${height}%` } as CSSProperties}
                            />
                            <em>{point.label}</em>
                        </span>
                    );
                })}
            </div>
        </section>
    );
}

export function FundOverviewMetrics({ series }: { series: FundOverviewMetricSeries }) {
    const hasAnyPoint = [...series.aum, ...series.investors, ...series.cashFlow].some((point) => point.value != null);
    if (!hasAnyPoint) return null;

    return (
        <div className="fund-overview-metrics-row" aria-label="Fon aylık özet göstergeleri">
            <FundOverviewMetricCard title="Fon Toplam Değer" points={series.aum} suffix="TL" />
            <FundOverviewMetricCard title="Yatırımcı Sayısı" points={series.investors} />
            <FundOverviewMetricCard title="Nakit Giriş Çıkışı" points={series.cashFlow} />
        </div>
    );
}

export function FundCategoryRankingsPanel({ rankings }: { rankings: FundDetail['category_rankings'] | undefined }) {
    const items = (rankings?.items || []).filter((item) => item.rank > 0 && item.total > 0).slice(0, 4);
    if (!items.length) return null;

    return (
        <section className="fund-category-ranking-panel" aria-label="Kategori sıralaması">
            <div className="fund-category-ranking-head">
                <div>
                    <span>Kategori Sıralaması</span>
                    <strong>{rankings?.category || 'Fon türü'}</strong>
                </div>
                <small>
                    <Trophy size={15} aria-hidden="true" />
                    {(rankings?.category_total || items[0]?.total || 0).toLocaleString('tr-TR')} fon
                </small>
            </div>
            <div className="fund-category-ranking-grid">
                {items.map((item) => (
                    <div className="fund-category-ranking-card" key={item.key} title={`${item.label}: ${formatPct(item.value)}`}>
                        <span>{item.label}</span>
                        <strong>{item.rank.toLocaleString('tr-TR')}/{item.total.toLocaleString('tr-TR')}</strong>
                        <small>{item.top_percentile != null ? `Üst %${item.top_percentile.toLocaleString('tr-TR')}` : formatPct(item.value)}</small>
                    </div>
                ))}
            </div>
        </section>
    );
}

export function FundMonthlyHeatmap({ monthlyReturns }: { monthlyReturns: MonthlyReturn[] }) {
    const maxAbs = Math.max(1, ...monthlyReturns.map((item) => Math.abs(item.returnPct)));
    const byMonth = new Map(monthlyReturns.map((item) => [item.month, item]));
    const years = Array.from(new Set(monthlyReturns.map((item) => item.year))).sort((a, b) => a - b);
    const best = monthlyReturns.reduce<MonthlyReturn | null>((current, item) => (!current || item.returnPct > current.returnPct ? item : current), null);
    const worst = monthlyReturns.reduce<MonthlyReturn | null>((current, item) => (!current || item.returnPct < current.returnPct ? item : current), null);
    const positiveCount = monthlyReturns.filter((item) => item.returnPct > 0).length;
    const avg = monthlyReturns.length
        ? monthlyReturns.reduce((sum, item) => sum + item.returnPct, 0) / monthlyReturns.length
        : null;
    const monthNames = ['Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz', 'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara'];

    return (
        <section className="fund-heatmap-panel">
            <h2>Aylık Getiriler</h2>
            {monthlyReturns.length ? (
                <>
                    <div className="fund-heatmap">
                        <div className="fund-heatmap-months">
                            <span />
                            {monthNames.map((month) => <span key={month}>{month}</span>)}
                        </div>
                        {years.map((year) => (
                            <div className="fund-heatmap-row" key={year}>
                                <strong>{year}</strong>
                                {monthNames.map((_, index) => {
                                    const key = `${year}-${String(index + 1).padStart(2, '0')}`;
                                    const item = byMonth.get(key);
                                    const intensity = item ? Math.min(1, Math.abs(item.returnPct) / maxAbs) : 0;
                                    return (
                                        <span
                                            key={key}
                                            className={[
                                                'fund-heatmap-cell',
                                                item ? (item.returnPct >= 0 ? 'is-positive' : 'is-negative') : 'is-empty',
                                            ].join(' ')}
                                            style={{ '--heat-intensity': intensity.toFixed(2) } as CSSProperties}
                                            title={item ? `${item.label}: ${formatPct(item.returnPct)}` : `${monthNames[index]} ${year}: veri yok`}
                                        >
                                            {item ? item.returnPct.toLocaleString('tr-TR', { maximumFractionDigits: 1 }) : ''}
                                        </span>
                                    );
                                })}
                            </div>
                        ))}
                    </div>
                    <div className="fund-heatmap-stats">
                        <div><span>En iyi ay</span><strong>{best ? `${best.label} ${formatPct(best.returnPct)}` : '-'}</strong></div>
                        <div><span>En düşük ay</span><strong>{worst ? `${worst.label} ${formatPct(worst.returnPct)}` : '-'}</strong></div>
                        <div><span>Ay sayısı</span><strong>{monthlyReturns.length}</strong></div>
                        <div><span>Pozitif ay</span><strong>{positiveCount}</strong></div>
                        <div><span>Aylık ortalama</span><strong>{formatPct(avg)}</strong></div>
                    </div>
                </>
            ) : (
                <div className="funds-state">Aylık getiri için yeterli geçmiş veri yok.</div>
            )}
        </section>
    );
}

export function FundHistoryTable({ points }: { points: FundPricePoint[] }) {
    const orderedPoints = [...points].sort((a, b) => a.date.localeCompare(b.date));
    const latestPoint = orderedPoints[orderedPoints.length - 1] || null;
    const latestDate = latestPoint ? new Date(`${latestPoint.date}T00:00:00`) : null;
    const startDate = latestDate && !Number.isNaN(latestDate.getTime()) ? new Date(latestDate) : null;
    if (startDate) {
        startDate.setDate(startDate.getDate() - 29);
    }
    const rows = orderedPoints
        .filter((point) => {
            if (!startDate || !latestDate) return true;
            const pointDate = new Date(`${point.date}T00:00:00`);
            return !Number.isNaN(pointDate.getTime()) && pointDate >= startDate && pointDate <= latestDate;
        })
        .sort((a, b) => b.date.localeCompare(a.date));
    const investorDeltas = new Map<string, number>();
    const cashFlowDeltas = new Map<string, number | null>();
    let previousInvestorCount: number | null = null;
    let previousPoint: FundPricePoint | null = null;

    orderedPoints.forEach((point) => {
        const currentInvestorCount = point.investor_count;
        if (typeof currentInvestorCount === 'number' && Number.isFinite(currentInvestorCount)) {
            if (previousInvestorCount != null) {
                investorDeltas.set(point.date, currentInvestorCount - previousInvestorCount);
            }
            previousInvestorCount = currentInvestorCount;
        }
        if (previousPoint) {
            cashFlowDeltas.set(point.date, estimateCashFlow(point, previousPoint));
        }
        previousPoint = point;
    });

    return (
        <div className="fund-history-table-wrap">
            <table className="fund-history-table">
                <thead>
                    <tr>
                        <th>Tarih</th>
                        <th>Fiyat</th>
                        <th>Günlük getiri</th>
                        <th>Portföy</th>
                        <th>Net nakit giriş/çıkış</th>
                        <th>Yatırımcı</th>
                    </tr>
                </thead>
                <tbody>
                    {rows.map((point) => {
                        const investorDelta = investorDeltas.get(point.date);
                        const cashFlowDelta = cashFlowDeltas.get(point.date);
                        return (
                            <tr key={point.date}>
                                <td>{formatDate(point.date)}</td>
                                <td>{formatCurrency(point.price, 'TRY')}</td>
                                <td className={pctClass(point.daily_return)}>{formatPct(point.daily_return)}</td>
                                <td>{formatCompactCurrency(point.aum)}</td>
                                <td className={pctClass(cashFlowDelta)}>{formatSignedCompactCurrency(cashFlowDelta)}</td>
                                <td>
                                    <span className="fund-investor-cell">
                                        <span>{point.investor_count?.toLocaleString('tr-TR') || '-'}</span>
                                        {investorDelta != null ? (
                                            <span className={`fund-investor-delta ${pctClass(investorDelta)}`}>
                                                ({formatSignedInteger(investorDelta)})
                                            </span>
                                        ) : null}
                                    </span>
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export function fundPeriodStatTitle(key: string, label: string): string {
    if (key === 'current_month') return 'Bu ay';
    if (key === 'last_30_days') return 'Son 30 gün';
    if (key === 'previous_month') return 'Önceki ay';
    return label;
}

export function FundPeriodStatsStrip({ stats }: { stats: FundPerformanceResponse['period_stats'] }) {
    const periods = (stats?.periods || [])
        .filter((period) => period.trading_days > 0)
        .filter((period) => ['current_month', 'last_30_days', 'previous_month'].includes(period.key));
    if (!periods.length) return null;

    return (
        <div className="fund-period-stats" aria-label="Dönem performans özeti">
            {periods.map((period) => (
                <div className="fund-period-stat" key={period.key}>
                    <div className="fund-period-stat-head">
                        <span>{fundPeriodStatTitle(period.key, period.label)}</span>
                        <small>{formatDate(period.start_date)} – {formatDate(period.end_date)}</small>
                    </div>
                    <div className="fund-period-stat-metrics">
                        <div>
                            <span>Toplam</span>
                            <strong className={pctClass(period.cumulative_return)}>{formatPct(period.cumulative_return)}</strong>
                        </div>
                        <div>
                            <span>Günlük Ort.</span>
                            <strong className={pctClass(period.average_daily_return)}>{formatPct(period.average_daily_return)}</strong>
                        </div>
                        <div>
                            <span>İş Günü</span>
                            <strong>{period.trading_days.toLocaleString('tr-TR')}</strong>
                        </div>
                        <div>
                            <span>Poz / Neg</span>
                            <strong>
                                {period.positive_days.toLocaleString('tr-TR')} / {period.negative_days.toLocaleString('tr-TR')}
                            </strong>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
}
