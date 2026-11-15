import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { CSSProperties, PointerEvent as ReactPointerEvent } from 'react';
import { createPortal } from 'react-dom';
import { GripHorizontal, Maximize2, X } from 'lucide-react';
import { apiClient } from '../../../api/client';
import type {
    MarketIndexLinePoint,
    MarketStockCardChartRange,
    MarketStockCardItem,
} from '../../../api/types';
import SymbolLogo from '../../../components/SymbolLogo';
import {
    STOCK_CARD_CHART_RANGES,
    STOCK_CARD_PERFORMANCE_KEYS,
    emptyStockCardItem,
    formatCardCurrency,
    formatCardFullCurrency,
    formatCardFullNumber,
    formatCardPct,
    formatCardPositiveRatio,
    formatCardRatio,
    formatStockCardAxisDate,
    formatStockCardAxisTime,
    formatStockCardChartDate,
    formatStockCardTradeTime,
    getTableChangeClass,
    isBistSymbol,
    isPreviousIstanbulDate,
    istanbulDateKey,
    latestStockCardPointTime,
} from '../stockScreenerConfig';
import type { StockCardDropPlacement } from '../types';
import './MarketStockCard.css';

export { emptyStockCardItem };

export interface StockCardMiniChartProps {
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
    onOpenDetail?: () => void;
    isLoading?: boolean;
    isLive?: boolean;
}

export function StockCardMiniChart({
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
    onOpenDetail,
    isLoading = false,
    isLive = false,
}: StockCardMiniChartProps) {
    const canvasWrapRef = useRef<HTMLDivElement>(null);
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const [dimensions, setDimensions] = useState<{ width: number; height: number }>({
        width: 340,
        height: 125,
    });

    useLayoutEffect(() => {
        const el = canvasWrapRef.current;
        if (!el) return;

        let frameId: number | null = null;
        const updateFromRect = (rect: DOMRectReadOnly | DOMRect) => {
            const w = Math.round(rect.width);
            const h = Math.round(rect.height);
            if (w <= 0 || h <= 0) return;
            if (frameId !== null) cancelAnimationFrame(frameId);
            frameId = requestAnimationFrame(() => {
                setDimensions((prev) => (prev.width === w && prev.height === h ? prev : { width: w, height: h }));
            });
        };

        updateFromRect(el.getBoundingClientRect());

        if (typeof ResizeObserver !== 'undefined') {
            const observer = new ResizeObserver((entries) => {
                const entry = entries[0];
                if (entry) updateFromRect(entry.contentRect);
            });
            observer.observe(el);
            return () => {
                if (frameId !== null) cancelAnimationFrame(frameId);
                observer.disconnect();
            };
        }

        const onResize = () => updateFromRect(el.getBoundingClientRect());
        window.addEventListener('resize', onResize);
        return () => {
            if (frameId !== null) cancelAnimationFrame(frameId);
            window.removeEventListener('resize', onResize);
        };
    }, []);

    const { width, height } = dimensions;
    const padding = { top: 12, right: 6, bottom: 20, left: 6 };
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
    const chartToolbar = (
        <div className="stock-card-chart-toolbar">
            {rangeControls}
            {onOpenDetail && (
                <button
                    type="button"
                    className="stock-card-chart-action-btn"
                    title={`${symbol} detaylı grafik ve analiz`}
                    aria-label={`${symbol} detaylı analiz`}
                    onClick={(event) => {
                        event.stopPropagation();
                        onOpenDetail();
                    }}
                >
                    <Maximize2 size={13} aria-hidden="true" />
                </button>
            )}
        </div>
    );
    if (isLoading) {
        return (
            <div className="stock-card-chart-shell">
                {chartToolbar}
                <div ref={canvasWrapRef} className="stock-card-chart-canvas-wrap">
                    <div className="stock-card-inline-chart-skeleton" role="status" aria-label="Grafik yükleniyor">
                        <svg
                            className="stock-card-inline-chart-svg"
                            viewBox={`0 0 ${width} ${height}`}
                            preserveAspectRatio="none"
                            aria-hidden="true"
                        >
                            {[0.28, 0.52, 0.76].map((ratio) => (
                                <line
                                    key={ratio}
                                    className="stock-card-inline-chart-gridline"
                                    x1={0}
                                    x2={width}
                                    y1={height * ratio}
                                    y2={height * ratio}
                                />
                            ))}
                        </svg>
                        <div className="stock-card-inline-chart-baseline stock-card-inline-skeleton-pulse" aria-hidden="true" />
                    </div>
                </div>
            </div>
        );
    }

    if (validPoints.length < 2) {
        return (
            <div className="stock-card-chart-shell">
                {chartToolbar}
                <div ref={canvasWrapRef} className="stock-card-chart-canvas-wrap">
                    <div className="stock-card-chart-empty">Veri yok</div>
                </div>
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
    const tooltipWidth = 100;
    const tooltipHeight = 36;
    const isLeftHalf = hoverX != null && hoverX < width / 2;
    const horizontalOffset = 12;
    let tooltipX = 0;
    let tooltipY = 0;
    if (hoverX != null && hoverY != null) {
        if (isLeftHalf) {
            tooltipX = Math.min(width - tooltipWidth - 6, hoverX + horizontalOffset);
        } else {
            tooltipX = Math.max(6, hoverX - tooltipWidth - horizontalOffset);
        }
        const minTop = 4;
        const maxTop = Math.max(minTop, height - tooltipHeight - 4);
        tooltipY = clamp(hoverY - tooltipHeight / 2, minTop, maxTop);
    }

    return (
        <div className="stock-card-chart-shell">
            {chartToolbar}
            <div ref={canvasWrapRef} className="stock-card-chart-canvas-wrap">
                <svg
                    className="stock-card-mini-chart"
                    viewBox={`0 0 ${width} ${height}`}
                    preserveAspectRatio="none"
                    role="img"
                    aria-label={`${symbol} ${selectedRange} çizgi grafiği`}
                    onPointerMove={handlePointerMove}
                    onPointerLeave={() => setHoverIndex(null)}
                    shapeRendering="geometricPrecision"
                    textRendering="geometricPrecision"
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
                        stroke="rgba(255,255,255,0.12)"
                        strokeDasharray="3 3"
                    />
                    {useTimeScale ? (
                        [startTimeMs, startTimeMs + (endTimeMs - startTimeMs) / 2, endTimeMs].map((timeMs, index) => {
                            const x = index === 0 ? padding.left : index === 2 ? width - padding.right : padding.left + plotWidth / 2;
                            return (
                                <text
                                    key={timeMs}
                                    x={x}
                                    y={height - 6}
                                    fill="#71717a"
                                    fontSize="10"
                                    fontFamily="var(--font-mono)"
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
                                    fill="#71717a"
                                    fontSize="10"
                                    fontFamily="var(--font-mono)"
                                    textAnchor={index === 0 ? 'start' : index === validPoints.length - 1 ? 'end' : 'middle'}
                                >
                                    {formatStockCardAxisDate(validPoints[index].time, selectedRange)}
                                </text>
                            );
                        })
                    )}
                    <path d={areaData} fill={`url(#${gradientId})`} />
                    <path d={pathData} fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />

                    {selectedRange === '1d' && validPoints.length > 0 && (
                        <circle
                            cx={xFor(validPoints.length - 1)}
                            cy={yFor(validPoints[validPoints.length - 1].close)}
                            r="3"
                            fill={color}
                        />
                    )}

                    {hoverPoint && hoverX != null && hoverY != null && (
                        <g className="stock-card-chart-hover">
                            <line
                                x1={hoverX}
                                x2={hoverX}
                                y1={padding.top}
                                y2={height - padding.bottom}
                                stroke="rgba(255,255,255,0.28)"
                                strokeDasharray="3 3"
                            />
                            <line
                                x1={padding.left}
                                x2={width - padding.right}
                                y1={hoverY}
                                y2={hoverY}
                                stroke="rgba(255,255,255,0.18)"
                                strokeDasharray="4 4"
                            />
                            <circle cx={hoverX} cy={hoverY} r="5" fill={color} opacity="0.25" />
                            <circle cx={hoverX} cy={hoverY} r="3" fill={color} stroke="#141413" strokeWidth="1.5" />
                            <g transform={`translate(${tooltipX}, ${tooltipY})`} style={{ pointerEvents: 'none' }}>
                                <rect
                                    width={tooltipWidth}
                                    height={tooltipHeight}
                                    rx="5"
                                    fill="#18181b"
                                    stroke="rgba(255,255,255,0.12)"
                                    strokeWidth="1"
                                    style={{ filter: 'drop-shadow(0 4px 10px rgba(0,0,0,0.5))' }}
                                />
                                <text x="8" y="14" fill="#a1a1aa" fontSize="9.5" fontFamily="var(--font-mono)">
                                    {formatStockCardChartDate(hoverPoint.time, selectedRange)}
                                </text>
                                <text x="8" y="28" fill="#f8fafc" fontSize="12" fontFamily="var(--font-mono)" fontWeight="700">
                                    {formatCardCurrency(hoverPoint.close, currency)}
                                </text>
                            </g>
                        </g>
                    )}
                </svg>
            </div>
        </div>
    );
}

export interface MarketStockCardProps {
    item: MarketStockCardItem;
    onOpen: () => void;
    onRemove: () => void;
    onMoveStart: () => void;
    onMoveOver: (placement: StockCardDropPlacement) => void;
    onMoveEnd: () => void;
    isLoading?: boolean;
    isDragging?: boolean;
}

export function MarketStockCard({
    item,
    onOpen,
    onRemove,
    onMoveStart,
    onMoveOver,
    onMoveEnd,
    isLoading = false,
    isDragging = false,
}: MarketStockCardProps) {
    const isDetailLoading = !isLoading && item.card_stage === 'quick';
    const isCardDataLoading = isLoading || isDetailLoading;
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
    const [tooltipStyles, setTooltipStyles] = useState<CSSProperties>({});
    const tooltipCloseTimerRef = useRef<number | null>(null);
    const tooltipOpenFrameRef = useRef<number | null>(null);

    useLayoutEffect(() => {
        if (!isCardDataLoading && hoveredData && performanceRef.current) {
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
    }, [hoveredData, isCardDataLoading]);

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
        if (isCardDataLoading) {
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
    }, [isCardDataLoading]);

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
        if (isCardDataLoading) {
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
    const sessionTime = formatStockCardTradeTime(
        isCardLive
            ? item.as_of || item.last_trade_at || latestPointTime
            : item.last_trade_at || latestPointTime || item.as_of,
    );
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
                {!isCardDataLoading && (
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
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardCurrency(item.high, item.currency)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Düşük</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardCurrency(item.low, item.currency)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Önc.Kap.</span>
                    <strong>
                        {isCardDataLoading ? (
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
                    if (selectedRange === 'ytd') return item.return_ytd_pct;
                    if (selectedRange === '1y') return item.return_1y_pct;
                    if (selectedRange === '5y') {
                        const points = chartDataByRange['5y'];
                        if (points && points.length >= 2 && points[0].close > 0) {
                            return ((points[points.length - 1].close - points[0].close) / points[0].close) * 100;
                        }
                        return null;
                    }
                    return item.change_pct;
                })()}
                currency={item.currency}
                selectedRange={selectedRange}
                pendingRange={pendingRange}
                rangeLoading={rangeLoading}
                rangeError={rangeError}
                onRangeSelect={handleRangeSelect}
                onOpenDetail={onOpen}
                isLoading={isCardDataLoading}
                isLive={isCardLive}
            />

            <div className="stock-card-detail-metrics">
                <div>
                    <span>F/K</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardPositiveRatio(item.fk)
                        )}
                    </strong>
                </div>
                <div>
                    <span>FD/FAVÖK</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardPositiveRatio(item.fd_favok)
                        )}
                    </strong>
                </div>
                <div>
                    <span>PD/DD</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardPositiveRatio(item.pd_dd)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Net Borç/FAVÖK</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardRatio(item.net_borc_favok)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Hacim Lot</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardFullNumber(item.volume_lot)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Hacim TL</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardFullCurrency(item.volume_tl, item.currency)
                        )}
                    </strong>
                </div>
                <div className="stock-card-detail-wide">
                    <span>Piyasa Değeri</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric-wide" aria-hidden="true" />
                        ) : (
                            formatCardFullCurrency(item.market_cap, item.currency)
                        )}
                    </strong>
                </div>
            </div>

            <div className="stock-card-performance" ref={performanceRef}>
                {!isCardDataLoading && hoveredData && createPortal(
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
                {isCardDataLoading
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
                {isCardDataLoading ? (
                    <span className="stock-card-inline-skeleton-time stock-card-inline-skeleton-pulse" aria-hidden="true" />
                ) : (
                    <span>{footerLabel} {sessionTime}</span>
                )}
            </div>
        </article>
    );
}
