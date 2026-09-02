import { useCallback, useEffect, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { ArrowLeft, BarChart3, BookOpen, ChevronRight, FileText, Info, Star } from 'lucide-react';
import './StockDetailPage.css';
import { apiClient } from '../api/client';
import type {
    KapSnapshotResponse,
    KapQuarter,
    MarketIndexLinePoint,
    MarketStockCardChartRange,
    MarketStockCardChartResponse,
} from '../api/types';
import { prepareOrderedQuarters } from '../utils/chartBuilders';
import SymbolLogo from '../components/SymbolLogo';
import MarketsNavigation, { type MarketsNavigationFundSection, type MarketsNavigationSection } from '../components/MarketsNavigation';
import { buildDocumentTitle, formatTitleCurrency, formatTitlePct, useDocumentTitle } from '../hooks/useDocumentTitle';
import { useWatchlist } from '../hooks/useWatchlist';
import type { StockTab } from '../routing/routes';

import StockOverview from './stock/sections/StockOverview';
import StockFinancials from './stock/sections/StockFinancials';
import StockKAP from './stock/sections/StockKAP';

interface StockDetailPageProps {
    ticker: string;
    activeTab?: StockTab;
    onTabChange?: (tab: StockTab) => void;
    onBack: () => void;
    onNavigateSection: (section: MarketsNavigationSection) => void;
    onNavigateFundSection?: (section: MarketsNavigationFundSection) => void;
    onOpenTicker: (ticker: string) => void;
    onOpenFund: (fundCode: string) => void;
}

type TabType = StockTab;

type StockPriceData = {
    ok: boolean;
    symbol: string;
    price: number | null;
    change: number | null;
    change_pct: number | null;
    currency: string;
    market_state: string;
    as_of?: string | null;
    error?: string;
};

const STOCK_DETAIL_TABS: Array<{ key: StockTab; label: string; icon: typeof Info }> = [
    { key: 'overview', label: 'Genel Bakış', icon: Info },
    { key: 'financials', label: 'Finansal Tablolar', icon: FileText },
    { key: 'kap', label: 'KAP Bildirimleri', icon: BookOpen },
];
const FULL_KAP_QUARTER_COUNT = 20;

function formatAsOf(value?: string | null): string {
    if (!value) return 'Veri güncelleniyor';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return 'Veri güncelleniyor';

    return new Intl.DateTimeFormat('tr-TR', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
    }).format(parsed);
}

function formatQuotePrice(value: number | null | undefined, currency?: string | null): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const currencyLabel = !currency || currency === 'TRY' || currency === 'TL' ? '₺' : currency;
    return `${currencyLabel}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatCompactCurrency(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return new Intl.NumberFormat('tr-TR', {
        style: 'currency',
        currency: 'TRY',
        notation: 'compact',
        maximumFractionDigits: 2,
    }).format(value);
}

function formatRatio(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return `${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}x`;
}

function formatPct(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '';
    const sign = value > 0 ? '+' : '';
    return `% ${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function pctClass(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '';
    if (value > 0) return 'positive';
    if (value < 0) return 'negative';
    return 'neutral';
}

const STOCK_DETAIL_CHART_RANGES: Array<{ id: MarketStockCardChartRange; label: string; title: string }> = [
    { id: '1d', label: 'G', title: 'Gün içi' },
    { id: '1w', label: '1H', title: '1 Hafta' },
    { id: '1m', label: '1A', title: '1 Ay' },
    { id: '1y', label: '1Y', title: '1 Yıl' },
];

function stockDetailChartDateKey(value: Date): string {
    if (Number.isNaN(value.getTime())) return '';
    return new Intl.DateTimeFormat('en-CA', {
        timeZone: 'Europe/Istanbul',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
    }).format(value);
}

function formatStockDetailChartTime(value: string, range: MarketStockCardChartRange): string {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '-';
    if (range === '1d') {
        return date.toLocaleTimeString('tr-TR', {
            hour: '2-digit',
            minute: '2-digit',
        });
    }
    if (range === '1m') {
        return date.toLocaleString('tr-TR', {
            day: '2-digit',
            month: 'short',
            hour: '2-digit',
            minute: '2-digit',
        });
    }
    return date.toLocaleDateString('tr-TR', {
        day: '2-digit',
        month: 'short',
        year: range === '1y' ? 'numeric' : undefined,
    });
}

function stockDetailChartReturn(latest: number | null, base: number | null): number | null {
    if (latest == null || base == null || !Number.isFinite(latest) || !Number.isFinite(base) || base === 0) {
        return null;
    }
    return ((latest - base) / base) * 100;
}

function StockDetailPriceChart({
    ticker,
    currency,
    dailyChangePct,
    currentPrice,
}: {
    ticker: string;
    currency?: string | null;
    dailyChangePct?: number | null;
    currentPrice?: number | null;
}) {
    const [selectedRange, setSelectedRange] = useState<MarketStockCardChartRange>('1d');
    const [chartDataByRange, setChartDataByRange] = useState<
        Partial<Record<MarketStockCardChartRange, MarketStockCardChartResponse>>
    >({});
    const [rangeErrors, setRangeErrors] = useState<Partial<Record<MarketStockCardChartRange, string | null>>>({});
    const chartAbortRef = useRef<AbortController | null>(null);
    const chartRequestIdRef = useRef(0);
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const [measureAnchorIndex, setMeasureAnchorIndex] = useState<number | null>(null);

    useEffect(() => {
        const cached = chartDataByRange[selectedRange];
        if (cached || rangeErrors[selectedRange]) return undefined;

        chartAbortRef.current?.abort();
        const controller = new AbortController();
        chartAbortRef.current = controller;
        const requestId = chartRequestIdRef.current + 1;
        chartRequestIdRef.current = requestId;

        apiClient
            .marketStockCardChart(ticker, selectedRange, { signal: controller.signal })
            .then((payload) => {
                if (controller.signal.aborted || requestId !== chartRequestIdRef.current) return;
                setChartDataByRange((previous) => ({ ...previous, [selectedRange]: payload }));
                const nextPoints = payload.line_points ?? [];
                setRangeErrors((previous) => ({
                    ...previous,
                    [selectedRange]: payload.error || nextPoints.length < 2 ? payload.error || 'Grafik verisi yok' : null,
                }));
            })
            .catch((error) => {
                if ((error as Error)?.name === 'AbortError' || requestId !== chartRequestIdRef.current) return;
                setRangeErrors((previous) => ({
                    ...previous,
                    [selectedRange]: (error as Error)?.message || 'Grafik verisi alınamadı',
                }));
            })
            .finally(() => {
                if (controller.signal.aborted || requestId !== chartRequestIdRef.current) return;
                if (chartAbortRef.current === controller) chartAbortRef.current = null;
            });

        return () => {
            controller.abort();
            if (chartAbortRef.current === controller) chartAbortRef.current = null;
        };
    }, [chartDataByRange, rangeErrors, selectedRange, ticker]);

    useEffect(() => {
        return () => chartAbortRef.current?.abort();
    }, []);

    const handleRangeSelect = (nextRange: MarketStockCardChartRange) => {
        if (nextRange === selectedRange && !rangeErrors[nextRange]) return;
        setHoverIndex(null);
        setMeasureAnchorIndex(null);
        if (rangeErrors[nextRange]) {
            setRangeErrors((previous) => ({ ...previous, [nextRange]: null }));
        }
        setSelectedRange(nextRange);
    };

    const chart = chartDataByRange[selectedRange];
    const validPoints: MarketIndexLinePoint[] = [...(chart?.line_points ?? [])]
        .filter((point) => {
            const time = new Date(point.time).getTime();
            return Number.isFinite(Number(point.close)) && Number.isFinite(time);
        })
        .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime())
        .map((point) => ({ ...point, close: Number(point.close) }));
    const rangeError = rangeErrors[selectedRange] || (validPoints.length < 2 ? chart?.error : null);
    const isLoading = !chart && !rangeError;

    const controls = (
        <div className="sd-price-chart-toolbar">
            <div>
                <span className="sd-price-chart-kicker">Fiyat grafiği</span>
                <strong>{STOCK_DETAIL_CHART_RANGES.find((range) => range.id === selectedRange)?.title}</strong>
            </div>
            <div className="sd-price-chart-ranges" role="group" aria-label="Grafik zaman aralığı">
                {STOCK_DETAIL_CHART_RANGES.map((range) => (
                    <button
                        key={range.id}
                        type="button"
                        className={selectedRange === range.id ? 'active' : ''}
                        aria-pressed={selectedRange === range.id}
                        title={range.title}
                        onClick={() => handleRangeSelect(range.id)}
                    >
                        {range.label}
                    </button>
                ))}
            </div>
        </div>
    );

    if (isLoading) {
        return (
            <section className="sd-price-chart-panel" aria-label={`${ticker} fiyat grafiği`}>
                {controls}
                <div className="sd-price-chart-state">Grafik verisi yükleniyor...</div>
            </section>
        );
    }

    if (validPoints.length < 2) {
        return (
            <section className="sd-price-chart-panel" aria-label={`${ticker} fiyat grafiği`}>
                {controls}
                <div className="sd-price-chart-state">{rangeError || 'Bu aralık için grafik verisi yok.'}</div>
            </section>
        );
    }

    const width = 1120;
    const height = 400;
    const padding = { top: 30, right: 76, bottom: 44, left: 16 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const values = validPoints.map((point) => point.close);
    const markerPrice = currentPrice != null && Number.isFinite(currentPrice)
        ? currentPrice
        : validPoints[validPoints.length - 1].close;
    let minValue = Math.min(...values, markerPrice);
    let maxValue = Math.max(...values, markerPrice);
    const rawSpan = Math.max(0.01, maxValue - minValue);
    minValue -= rawSpan * 0.05;
    maxValue += rawSpan * 0.05;
    const span = Math.max(0.01, maxValue - minValue);
    const useIntradayScale = selectedRange === '1d';
    let startTimeMs = 0;
    let endTimeMs = 0;
    if (useIntradayScale) {
        const firstDate = new Date(validPoints[0].time);
        const lastDate = new Date(validPoints[validPoints.length - 1].time);
        const start = new Date(firstDate);
        start.setHours(10, 0, 0, 0);
        const end = new Date(firstDate);
        const isLive = chart?.is_live === true || chart?.session_status === 'open';
        const hasTodayPoint = stockDetailChartDateKey(lastDate) === stockDetailChartDateKey(new Date());
        if (isLive || hasTodayPoint) {
            end.setHours(18, 0, 0, 0);
        } else {
            const lastMinutes = lastDate.getHours() * 60 + lastDate.getMinutes();
            const roundedEndHour = Math.min(18, Math.max(13, Math.ceil(lastMinutes / 60)));
            end.setHours(roundedEndHour, 0, 0, 0);
            if (end.getTime() <= lastDate.getTime()) {
                end.setHours(Math.min(18, end.getHours() + 1), 0, 0, 0);
            }
        }
        startTimeMs = start.getTime();
        endTimeMs = Math.max(end.getTime(), startTimeMs + 60 * 60 * 1000);
    }

    const xFor = (index: number) => {
        if (useIntradayScale) {
            const pointTime = new Date(validPoints[index].time).getTime();
            const ratio = Math.min(1, Math.max(0, (pointTime - startTimeMs) / (endTimeMs - startTimeMs)));
            return padding.left + ratio * plotWidth;
        }
        return padding.left + (index / Math.max(1, validPoints.length - 1)) * plotWidth;
    };
    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;
    const pathData = validPoints
        .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(index)} ${yFor(point.close)}`)
        .join(' ');
    const areaData = `${pathData} L ${xFor(validPoints.length - 1)} ${height - padding.bottom} L ${padding.left} ${height - padding.bottom} Z`;
    const rangeReturn = stockDetailChartReturn(validPoints[validPoints.length - 1].close, validPoints[0].close);
    const colorReturn = selectedRange === '1d' && dailyChangePct != null ? dailyChangePct : rangeReturn;
    const chartColor = colorReturn == null || colorReturn >= 0 ? '#22c55e' : '#ff4d5e';
    const tickValues = Array.from({ length: 6 }, (_, index) => minValue + (span * index) / 5);
    const tickIndexes = Array.from(new Set([0, Math.floor((validPoints.length - 1) / 2), validPoints.length - 1]));
    const timeTickLabels = useIntradayScale
        ? [startTimeMs, startTimeMs + (endTimeMs - startTimeMs) / 2, endTimeMs].map((time, index) => ({
              x: index === 0 ? padding.left : index === 2 ? width - padding.right : padding.left + plotWidth / 2,
              label: new Date(time).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
              key: `time-${time}`,
          }))
        : tickIndexes.map((index) => ({
              x: xFor(index),
              label: formatStockDetailChartTime(validPoints[index].time, selectedRange),
              key: `point-${index}`,
          }));

    const closestIndexForX = (x: number) => {
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
    const chartXFromEvent = (event: ReactPointerEvent<SVGSVGElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return null;
        return Math.min(Math.max(((event.clientX - rect.left) / rect.width) * width, 0), width);
    };
    const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const x = chartXFromEvent(event);
        if (x == null) return;
        setHoverIndex(closestIndexForX(x));
    };
    const handlePointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
        if (event.pointerType === 'mouse' && event.button !== 0) return;
        event.preventDefault();
        const x = chartXFromEvent(event);
        if (x == null) return;
        const closestIndex = closestIndexForX(x);
        setHoverIndex(closestIndex);
        setMeasureAnchorIndex(closestIndex);
        event.currentTarget.setPointerCapture?.(event.pointerId);
    };
    const handlePointerUp = (event: ReactPointerEvent<SVGSVGElement>) => {
        setMeasureAnchorIndex(null);
        try {
            if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
            }
        } catch {
            // Ignore release errors when the browser has already cancelled capture.
        }
    };

    const activeHoverIndex = hoverIndex == null ? null : Math.min(Math.max(hoverIndex, 0), validPoints.length - 1);
    const hoverPoint = activeHoverIndex == null ? null : validPoints[activeHoverIndex];
    const hoverX = activeHoverIndex == null ? null : xFor(activeHoverIndex);
    const hoverY = hoverPoint ? yFor(hoverPoint.close) : null;
    const measureAnchorPoint = measureAnchorIndex == null
        ? null
        : validPoints[Math.min(Math.max(measureAnchorIndex, 0), validPoints.length - 1)];
    const measureEndPoint = measureAnchorPoint && hoverPoint && activeHoverIndex != null && activeHoverIndex !== measureAnchorIndex
        ? hoverPoint
        : null;
    const measureAnchorX = measureAnchorIndex == null
        ? null
        : xFor(Math.min(Math.max(measureAnchorIndex, 0), validPoints.length - 1));
    const measureAnchorY = measureAnchorPoint ? yFor(measureAnchorPoint.close) : null;
    const measureEndX = measureEndPoint && activeHoverIndex != null ? xFor(activeHoverIndex) : null;
    const measureEndY = measureEndPoint ? yFor(measureEndPoint.close) : null;
    const measureDelta = measureAnchorPoint && measureEndPoint ? measureEndPoint.close - measureAnchorPoint.close : null;
    const measureReturn = measureAnchorPoint && measureEndPoint
        ? stockDetailChartReturn(measureEndPoint.close, measureAnchorPoint.close)
        : null;
    const measureLineY = measureAnchorY != null && measureEndY != null
        ? Math.max(padding.top + 12, Math.min(measureAnchorY, measureEndY) - 18)
        : null;
    const measureTooltipWidth = 190;
    const measureTooltipHeight = 70;
    const measureTooltipGap = 14;
    const measureTooltipAnchorX = hoverX ?? measureEndX;
    const measureTooltipAnchorY = hoverY ?? measureEndY;
    const measureTooltipShouldFlip = measureTooltipAnchorX != null
        && measureTooltipAnchorX + measureTooltipGap + measureTooltipWidth > width - 8;
    const measureTooltipX = measureTooltipAnchorX == null
        ? 0
        : measureTooltipShouldFlip
            ? Math.max(8, measureTooltipAnchorX - measureTooltipGap - measureTooltipWidth)
            : Math.min(measureTooltipAnchorX + measureTooltipGap, width - measureTooltipWidth - 8);
    const measureTooltipY = measureTooltipAnchorY == null
        ? 0
        : measureTooltipAnchorY - measureTooltipGap - measureTooltipHeight >= padding.top
            ? measureTooltipAnchorY - measureTooltipGap - measureTooltipHeight
            : Math.min(measureTooltipAnchorY + measureTooltipGap, height - measureTooltipHeight - 8);
    const measureDeltaText = measureDelta == null
        ? '-'
        : `${measureDelta > 0 ? '+' : measureDelta < 0 ? '-' : ''}${formatQuotePrice(Math.abs(measureDelta), currency)}`;
    const measureClass = measureReturn == null ? '' : pctClass(measureReturn);
    const measureLeftX = measureAnchorX != null && measureEndX != null ? Math.min(measureAnchorX, measureEndX) : null;
    const measureRightX = measureAnchorX != null && measureEndX != null ? Math.max(measureAnchorX, measureEndX) : null;
    const tooltipWidth = 166;
    const tooltipHeight = 60;
    const tooltipX = hoverX == null ? 0 : Math.min(Math.max(hoverX + 16, padding.left), width - tooltipWidth - 8);
    const tooltipY = hoverY == null
        ? 0
        : hoverY - tooltipHeight / 2 < padding.top
            ? padding.top
            : Math.min(hoverY - tooltipHeight / 2, height - tooltipHeight - padding.bottom);
    const markerLabel = formatQuotePrice(markerPrice, currency);
    const markerWidth = Math.max(70, markerLabel.length * 8 + 18);
    const markerY = yFor(markerPrice);
    const markerX = width - padding.right + 4;

    return (
        <section className="sd-price-chart-panel" aria-label={`${ticker} fiyat grafiği`}>
            {controls}
            <svg
                className="sd-price-chart-svg"
                viewBox={`0 0 ${width} ${height}`}
                preserveAspectRatio="xMidYMid meet"
                role="img"
                aria-label={`${ticker} fiyat grafiği`}
                onPointerMove={handlePointerMove}
                onPointerDown={handlePointerDown}
                onPointerUp={handlePointerUp}
                onPointerCancel={handlePointerUp}
                onPointerLeave={() => {
                    setHoverIndex(null);
                    setMeasureAnchorIndex(null);
                }}
            >
                <defs>
                    <linearGradient id={`sd-price-chart-area-${ticker}-${selectedRange}`} x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor={chartColor} stopOpacity="0.24" />
                        <stop offset="100%" stopColor={chartColor} stopOpacity="0" />
                    </linearGradient>
                </defs>
                <rect x="0" y="0" width={width} height={height} fill="#0d1113" />
                {tickValues.map((value) => (
                    <g key={value}>
                        <line
                            x1={padding.left}
                            x2={width - padding.right}
                            y1={yFor(value)}
                            y2={yFor(value)}
                            className="sd-price-chart-gridline"
                        />
                        <text x={width - padding.right + 10} y={yFor(value) + 4} className="sd-price-chart-axis">
                            {formatQuotePrice(value, currency)}
                        </text>
                    </g>
                ))}
                {timeTickLabels.map(({ x, label, key }) => (
                    <g key={key}>
                        <line
                            x1={x}
                            x2={x}
                            y1={padding.top}
                            y2={height - padding.bottom}
                            className="sd-price-chart-time-gridline"
                        />
                        <text
                            x={x}
                            y={height - 15}
                            className="sd-price-chart-axis"
                            textAnchor={x === padding.left ? 'start' : x === width - padding.right ? 'end' : 'middle'}
                        >
                            {label}
                        </text>
                    </g>
                ))}
                <path d={areaData} fill={`url(#sd-price-chart-area-${ticker}-${selectedRange})`} />
                <path d={pathData} fill="none" stroke={chartColor} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
                <circle cx={xFor(validPoints.length - 1)} cy={yFor(validPoints[validPoints.length - 1].close)} r="4" fill={chartColor} />
                <g transform={`translate(${markerX}, ${markerY})`}>
                    <rect width={markerWidth} height="20" y="-10" rx="2" fill={chartColor} />
                    <path d="M 0 0 L -6 -6 L -6 6 Z" fill={chartColor} />
                    <text x={markerWidth / 2} y="3" fill="#ffffff" fontSize="11" fontFamily="monospace" textAnchor="middle" fontWeight="bold">
                        {markerLabel}
                    </text>
                </g>

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
                    <g className="sd-price-chart-measure">
                        <line x1={measureAnchorX} y1={measureAnchorY} x2={measureEndX} y2={measureEndY} className="sd-price-chart-measure-path" />
                        <line x1={measureLeftX} y1={measureLineY} x2={measureRightX} y2={measureLineY} className="sd-price-chart-measure-line" />
                        <line x1={measureAnchorX} y1={measureLineY - 5} x2={measureAnchorX} y2={measureLineY + 5} className="sd-price-chart-measure-line" />
                        <line x1={measureEndX} y1={measureLineY - 5} x2={measureEndX} y2={measureLineY + 5} className="sd-price-chart-measure-line" />
                        <circle cx={measureAnchorX} cy={measureAnchorY} r="4.2" className="sd-price-chart-measure-point" fill={chartColor} />
                        <circle cx={measureEndX} cy={measureEndY} r="4.2" className="sd-price-chart-measure-point" fill={chartColor} />
                        <g
                            className="sd-price-chart-measure-tooltip"
                            style={{ transform: `translate(${measureTooltipX}px, ${measureTooltipY}px)` }}
                        >
                            <rect width={measureTooltipWidth} height={measureTooltipHeight} rx="7" className="sd-price-chart-measure-tooltip-bg" />
                            <text x="10" y="18" className="sd-price-chart-measure-muted">
                                {formatStockDetailChartTime(measureAnchorPoint.time, selectedRange)} - {formatStockDetailChartTime(measureEndPoint.time, selectedRange)}
                            </text>
                            <text x="10" y="42" className={`sd-price-chart-measure-value ${measureClass}`}>
                                {formatPct(measureReturn)}
                            </text>
                            <text x="10" y="60" className="sd-price-chart-measure-muted">
                                {measureDeltaText}
                            </text>
                        </g>
                    </g>
                )}

                {measureAnchorIndex == null && hoverPoint && hoverX != null && hoverY != null && (
                    <g className="sd-price-chart-hover">
                        <line x1={hoverX} x2={hoverX} y1={padding.top} y2={height - padding.bottom} className="sd-price-chart-hoverline" />
                        <line x1={padding.left} x2={width - padding.right} y1={hoverY} y2={hoverY} className="sd-price-chart-hoverline horizontal" />
                        <circle cx={hoverX} cy={hoverY} r="4" fill={chartColor} stroke="#0a0c0f" strokeWidth="2" />
                        <g transform={`translate(${tooltipX}, ${tooltipY})`}>
                            <rect width={tooltipWidth} height={tooltipHeight} rx="4" className="sd-price-chart-tooltip-bg" />
                            <text x="12" y="21" className="sd-price-chart-tooltip-muted">
                                {formatStockDetailChartTime(hoverPoint.time, selectedRange)}
                            </text>
                            <text x="12" y="45" className="sd-price-chart-tooltip-value">
                                {formatQuotePrice(hoverPoint.close, currency)}
                            </text>
                        </g>
                    </g>
                )}
            </svg>
            {rangeError && chart && validPoints.length >= 2 && <div className="sd-price-chart-note">{rangeError}</div>}
        </section>
    );
}

export default function StockDetailPage({
    ticker,
    activeTab = 'overview',
    onTabChange,
    onBack,
    onNavigateSection,
    onNavigateFundSection,
    onOpenTicker,
    onOpenFund,
}: StockDetailPageProps) {
    const [selectedTab, setSelectedTab] = useState<TabType>(activeTab);
    const [snapshot, setSnapshot] = useState<KapSnapshotResponse | null>(null);
    const [quarters, setQuarters] = useState<KapQuarter[]>([]);
    const [loading, setLoading] = useState(false);
    const [snapshotQuarterDepth, setSnapshotQuarterDepth] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [navCollapsed, setNavCollapsed] = useState(false);
    const [priceData, setPriceData] = useState<StockPriceData | null>(null);
    const watchlist = useWatchlist();
    const normalizedTicker = ticker.trim().toUpperCase();

    useEffect(() => {
        setSelectedTab(activeTab);
    }, [activeTab]);

    const handleTabChange = (tab: TabType) => {
        setSelectedTab(tab);
        onTabChange?.(tab);
    };

    useEffect(() => {
        let mounted = true;
        setSnapshot(null);
        setQuarters([]);
        setSnapshotQuarterDepth(0);
        setError(null);
        setLoading(true);
        // Genel Bakış tab'ında 5/10/15/20 çeyrek seçenekleri var; ilk istekte
        // de tam 20 çeyrek çekiyoruz ki 15c/20c butonları boş görünmesin. KAP
        // cache'i taze ise tek istekte gelir, soğuk ise tek seferde uzun ama
        // sonraki tab geçişlerinde yeniden istek atılmaz.
        const initialQuarterCount = FULL_KAP_QUARTER_COUNT;
        apiClient.kapSnapshot(ticker, false, initialQuarterCount)
            .then(data => {
                if (mounted) {
                    setSnapshot(data);
                    setQuarters(prepareOrderedQuarters(data));
                    setSnapshotQuarterDepth(initialQuarterCount);
                    if (!data.ok && data.error) setError(data.error);
                }
            })
            .catch(err => {
                if (mounted) setError(err.message || 'Veri alınamadı');
            })
            .finally(() => {
                if (mounted) setLoading(false);
            });
        
        return () => { mounted = false; };
    }, [ticker]);

    useEffect(() => {
        if (selectedTab !== 'financials' && selectedTab !== 'kap') return;
        if (!snapshot) return;
        if (snapshotQuarterDepth >= FULL_KAP_QUARTER_COUNT) return;
        let mounted = true;
        apiClient.kapSnapshot(ticker, false, FULL_KAP_QUARTER_COUNT)
            .then(data => {
                if (!mounted) return;
                setSnapshot(data);
                setQuarters(prepareOrderedQuarters(data));
                setSnapshotQuarterDepth(FULL_KAP_QUARTER_COUNT);
                if (!data.ok && data.error) setError(data.error);
            })
            .catch(() => {
                // İlk özet snapshot ekranda kalır; derin veri yüklenemese de fiyat akışı bloklanmaz.
            });
        return () => { mounted = false; };
    }, [selectedTab, snapshot, snapshotQuarterDepth, ticker]);

    useEffect(() => {
        let cancelled = false;
        setPriceData(null);
        apiClient.kapPrice(ticker)
            .then((response) => {
                if (!cancelled) {
                    setPriceData(response as StockPriceData);
                }
            })
            .catch(() => {
                if (!cancelled) {
                    setPriceData(null);
                }
            });

        return () => {
            cancelled = true;
        };
    }, [ticker]);

    const renderContent = () => {
        if (loading && !snapshot) {
            return <div className="sd-loading"><div className="spinner" /> Veriler yükleniyor...</div>;
        }
        
        if (!snapshot) return null;

        switch (selectedTab) {
            case 'overview':
                return <StockOverview snapshot={snapshot} quarters={quarters} />;
            case 'financials':
                return <StockFinancials quarters={quarters} analysisNote={snapshot.analysis_note} />;
            case 'kap':
                return <StockKAP ticker={ticker} quarters={quarters} />;
            default:
                return null;
        }
    };

    const valuation = snapshot?.valuation;
    const displayPrice = priceData?.ok && priceData.price != null ? priceData.price : valuation?.price;
    const displayCurrency = priceData?.currency || valuation?.price_currency;
    const displayAsOf = priceData?.as_of || valuation?.price_as_of || snapshot?.fetched_at;
    const displayChangePct = priceData?.ok ? priceData.change_pct : null;
    const selectedTabLabel = STOCK_DETAIL_TABS.find((tab) => tab.key === selectedTab)?.label;
    const quoteTitle = [
        formatTitleCurrency(displayPrice, displayCurrency),
        formatTitlePct(displayChangePct),
    ].filter(Boolean).join(' ');
    useDocumentTitle(buildDocumentTitle(normalizedTicker, quoteTitle, selectedTabLabel));
    const isStarred = watchlist.hasItem('stock', normalizedTicker);
    const toggleStarredStock = useCallback(() => {
        if (!normalizedTicker) return;
        watchlist.toggleItem({
            kind: 'stock',
            symbol: normalizedTicker,
            label: snapshot?.company_title || normalizedTicker,
        });
    }, [normalizedTicker, snapshot?.company_title, watchlist]);

    return (
        <div className={`mn-layout stock-detail-shell${navCollapsed ? ' mn-nav-collapsed' : ''}`}>
            <MarketsNavigation
                collapsed={navCollapsed}
                activeSection="stocks"
                onCollapsedChange={setNavCollapsed}
                onSectionChange={onNavigateSection}
                onFundSectionChange={onNavigateFundSection}
                onSelectTicker={onOpenTicker}
                onSelectFund={onOpenFund}
            />
            <div className="stock-workspace">
                <div className="stock-detail-page">
                    <header className="stock-market-shell">
                        <div className="stock-market-breadcrumb" aria-label="Hisse konumu">
                            <button type="button" className="stock-breadcrumb-back" onClick={onBack}>
                                <ArrowLeft size={15} aria-hidden="true" />
                                Hisseler
                            </button>
                            <ChevronRight size={14} aria-hidden="true" />
                            <span className="stock-breadcrumb-group">
                                <BarChart3 size={17} aria-hidden="true" />
                                BIST Hisseleri
                            </span>
                            <ChevronRight size={14} aria-hidden="true" />
                            <span className="stock-breadcrumb-code">
                                <SymbolLogo
                                    symbol={ticker}
                                    name={snapshot?.company_title || ticker}
                                    kind="stock"
                                    size="xs"
                                />
                                {normalizedTicker}
                            </span>
                        </div>

                        <div className="stock-market-hero">
                            <div className="stock-market-title">
                                <SymbolLogo
                                    symbol={ticker}
                                    name={snapshot?.company_title || ticker}
                                    kind="stock"
                                    size="lg"
                                    className="stock-market-logo"
                                />
                                <div className="stock-market-copy">
                                    <div className="stock-market-code-row">
                                        <h1>{normalizedTicker}</h1>
                                        <button
                                            type="button"
                                            className={`stock-icon-action${isStarred ? ' active' : ''}`}
                                            onClick={toggleStarredStock}
                                            aria-pressed={isStarred}
                                            aria-label={isStarred ? `${normalizedTicker} izleme listesinden çıkar` : `${normalizedTicker} izleme listesine ekle`}
                                            title={isStarred ? 'İzleme listesinden çıkar' : 'İzleme listesine ekle'}
                                        >
                                            <Star size={19} aria-hidden="true" />
                                        </button>
                                    </div>
                                    <p>{snapshot?.company_title || (snapshot?.error ? ticker : 'Şirket bilgisi yükleniyor...')}</p>
                                    <small>
                                        {snapshot?.error
                                            ? snapshot.error
                                            : snapshot?.latest_quarter
                                                ? `Son dönem · ${snapshot.latest_quarter}`
                                                : 'KAP finansal verileri yükleniyor'}
                                    </small>
                                </div>
                            </div>

                            <div className="stock-market-quote">
                                <div>
                                    <strong>{formatQuotePrice(displayPrice, displayCurrency)}</strong>
                                    {displayChangePct != null && (
                                        <span className={pctClass(displayChangePct)}>{formatPct(displayChangePct)}</span>
                                    )}
                                </div>
                                <small>{formatAsOf(displayAsOf)}</small>
                            </div>
                        </div>

                        <div className="stock-market-stats">
                            <div><span>Piyasa Değeri</span><strong>{formatCompactCurrency(valuation?.market_cap)}</strong></div>
                            <div><span>F/K</span><strong>{formatRatio(valuation?.fk)}</strong></div>
                            <div><span>PD/DD</span><strong>{formatRatio(valuation?.pd_dd)}</strong></div>
                            <div><span>Son Dönem</span><strong>{snapshot?.latest_quarter || '-'}</strong></div>
                        </div>

                        <nav className="stock-tabs" aria-label="Hisse detay sekmeleri">
                            {STOCK_DETAIL_TABS.map((tab) => {
                                const Icon = tab.icon;
                                return (
                                    <button
                                        key={tab.key}
                                        type="button"
                                        className={selectedTab === tab.key ? 'active' : ''}
                                        onClick={() => handleTabChange(tab.key)}
                                    >
                                        <Icon size={16} aria-hidden="true" />
                                        {tab.label}
                                    </button>
                                );
                            })}
                        </nav>
                    </header>

                    <div className="sd-content-area">
                        {error && <div className="alert-error">{error}</div>}
                        {selectedTab === 'overview' && (
                            <StockDetailPriceChart
                                key={normalizedTicker}
                                ticker={normalizedTicker}
                                currency={displayCurrency}
                                dailyChangePct={displayChangePct}
                                currentPrice={displayPrice}
                            />
                        )}
                        {renderContent()}
                    </div>
                </div>
            </div>
        </div>
    );
}
