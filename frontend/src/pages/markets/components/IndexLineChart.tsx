import { useEffect, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import type { MarketIndexDetailResponse } from '../../../api/types';
import AnimatedChartSkeleton from '../../../components/charts/AnimatedChartSkeleton';
import { useMobileChartLayout } from '../../../hooks/useMobileChartLayout';
import { formatIndexPrice, formatTablePct, isBistSymbol } from '../stockScreenerConfig';
import './IndexLineChart.css';

export interface IndexLineChartProps {
    symbol?: string;
    points: MarketIndexDetailResponse['line_points'];
    prevClose: number | null;
    changePct: number | null;
    isLoading?: boolean;
}

export function IndexLineChart({
    symbol,
    points,
    prevClose,
    changePct,
    isLoading = false,
}: IndexLineChartProps) {
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const [measureAnchorIndex, setMeasureAnchorIndex] = useState<number | null>(null);
    const [selectedMeasureRange, setSelectedMeasureRange] = useState<{ start: number; end: number } | null>(null);
    const touchPointersRef = useRef<Map<number, number>>(new Map());
    const isMobileChart = useMobileChartLayout();
    const width = 1120;
    const height = isMobileChart ? 540 : 400;
    const padding = isMobileChart
        ? { top: 30, right: 92, bottom: 52, left: 16 }
        : { top: 30, right: 65, bottom: 40, left: 16 };
    const validPoints = points.filter((point) => Number.isFinite(point.close));
    const pointWindowKey = `${symbol || ''}:${validPoints.length}:${validPoints[0]?.time || ''}:${validPoints[validPoints.length - 1]?.time || ''}`;

    useEffect(() => {
        setHoverIndex(null);
        setMeasureAnchorIndex(null);
        setSelectedMeasureRange(null);
        touchPointersRef.current.clear();
    }, [pointWindowKey]);

    if (validPoints.length < 2) {
        if (isLoading) {
            return (
                <AnimatedChartSkeleton
                    seed={`index-${symbol || 'IDX'}`}
                    width={width}
                    height={height}
                    padding={padding}
                    className="indices-line-chart"
                    ariaLabel={`${symbol || 'Endeks'} grafiği yükleniyor`}
                />
            );
        }
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
    const timeTickLabels: Array<{ x: number; label: string; key: string }> = [];
    if (useTimeScale) {
        const hourStep = isMobileChart ? 2 : 1;
        for (let h = 10; h <= 18; h += hourStep) {
            const d = new Date(validPoints[0].time);
            d.setHours(h, 0, 0, 0);
            timeTickLabels.push({
                x: padding.left + ((d.getTime() - startTimeMs) / (endTimeMs - startTimeMs)) * plotWidth,
                label: `${h.toString().padStart(2, '0')}:00`,
                key: `fixed-${h}`,
            });
        }
    } else {
        const timeTickCount = isMobileChart ? 5 : 8;
        const timeTicks = Array.from({ length: timeTickCount }).map((_, i) => Math.floor((validPoints.length - 1) * (i / (timeTickCount - 1))));
        timeTicks.forEach((index) => {
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

    const closestIndexForX = (x: number) => {
        let closestIndex = 0;
        let minDiff = Infinity;
        for (let i = 0; i < validPoints.length; i++) {
            const diff = Math.abs(xFor(i) - x);
            if (diff < minDiff) {
                minDiff = diff;
                closestIndex = i;
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
        const closestIndex = closestIndexForX(x);
        if (event.pointerType === 'touch') {
            if (touchPointersRef.current.has(event.pointerId)) {
                touchPointersRef.current.set(event.pointerId, closestIndex);
            }
            if (touchPointersRef.current.size >= 2) {
                event.preventDefault();
                const [firstIdx, secondIdx] = Array.from(touchPointersRef.current.values());
                setSelectedMeasureRange(null);
                setMeasureAnchorIndex(firstIdx);
                setHoverIndex(secondIdx);
                return;
            }
        }
        setHoverIndex(closestIndex);
    };

    const handlePointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
        if (event.pointerType === 'mouse' && event.button !== 0) return;
        const x = chartXFromEvent(event);
        if (x == null) return;
        const closestIndex = closestIndexForX(x);
        if (event.pointerType === 'touch') {
            touchPointersRef.current.set(event.pointerId, closestIndex);
            if (touchPointersRef.current.size >= 2) {
                event.preventDefault();
                const [firstIdx, secondIdx] = Array.from(touchPointersRef.current.values());
                setSelectedMeasureRange(null);
                setMeasureAnchorIndex(firstIdx);
                setHoverIndex(secondIdx);
                return;
            }
            setSelectedMeasureRange(null);
            setMeasureAnchorIndex(null);
            setHoverIndex(closestIndex);
            return;
        }
        event.preventDefault();
        setSelectedMeasureRange(null);
        setHoverIndex(closestIndex);
        setMeasureAnchorIndex(closestIndex);
        event.currentTarget.setPointerCapture?.(event.pointerId);
    };

    const handlePointerUp = (event: ReactPointerEvent<SVGSVGElement>) => {
        if (event.pointerType === 'touch') {
            const wasTwoFinger = touchPointersRef.current.size >= 2;
            const prevIndices = wasTwoFinger ? Array.from(touchPointersRef.current.values()) : [];
            touchPointersRef.current.delete(event.pointerId);
            if (touchPointersRef.current.size >= 2) {
                const [firstIdx, secondIdx] = Array.from(touchPointersRef.current.values());
                setMeasureAnchorIndex(firstIdx);
                setHoverIndex(secondIdx);
                return;
            }
            if (wasTwoFinger && prevIndices.length >= 2 && prevIndices[0] !== prevIndices[1]) {
                setSelectedMeasureRange({
                    start: Math.min(prevIndices[0], prevIndices[1]),
                    end: Math.max(prevIndices[0], prevIndices[1]),
                });
            }
            setMeasureAnchorIndex(null);
            return;
        }
        const x = chartXFromEvent(event);
        if (x != null && measureAnchorIndex != null) {
            const endIndex = closestIndexForX(x);
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

    const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);
    const activeHoverIndex = hoverIndex == null ? null : clamp(hoverIndex, 0, validPoints.length - 1);
    const hoverPoint = activeHoverIndex == null ? null : validPoints[activeHoverIndex];
    const hoverX = activeHoverIndex == null ? null : xFor(activeHoverIndex);
    const hoverY = hoverPoint ? yFor(hoverPoint.close) : null;
    const activeSelection = measureAnchorIndex != null && activeHoverIndex != null && activeHoverIndex !== measureAnchorIndex
        ? {
            start: Math.min(clamp(measureAnchorIndex, 0, validPoints.length - 1), activeHoverIndex),
            end: Math.max(clamp(measureAnchorIndex, 0, validPoints.length - 1), activeHoverIndex),
        }
        : selectedMeasureRange && selectedMeasureRange.start !== selectedMeasureRange.end
            ? {
                start: clamp(selectedMeasureRange.start, 0, validPoints.length - 1),
                end: clamp(selectedMeasureRange.end, 0, validPoints.length - 1),
            }
            : null;
    const measureAnchorPoint = activeSelection ? validPoints[activeSelection.start] : null;
    const measureEndPoint = activeSelection ? validPoints[activeSelection.end] : null;
    const measureAnchorX = activeSelection ? xFor(activeSelection.start) : null;
    const measureAnchorY = measureAnchorPoint ? yFor(measureAnchorPoint.close) : null;
    const measureEndX = activeSelection ? xFor(activeSelection.end) : null;
    const measureEndY = measureEndPoint ? yFor(measureEndPoint.close) : null;
    const measureStartValue = measureAnchorPoint?.close ?? null;
    const measureEndValue = measureEndPoint?.close ?? null;
    const measureDelta = measureStartValue != null && measureEndValue != null
        ? measureEndValue - measureStartValue
        : null;
    const measureReturn = measureStartValue != null && measureEndValue != null && measureStartValue !== 0
        ? ((measureEndValue - measureStartValue) / measureStartValue) * 100
        : null;
    const selectionColor = measureReturn == null ? chartColor : measureReturn >= 0 ? '#00c805' : '#ef4444';
    const selectedSegmentPath = activeSelection
        ? validPoints
            .slice(activeSelection.start, activeSelection.end + 1)
            .map((point, offset) => `${offset === 0 ? 'M' : 'L'} ${xFor(activeSelection.start + offset)} ${yFor(point.close)}`)
            .join(' ')
        : '';
    const selectedSegmentArea = activeSelection && selectedSegmentPath
        ? `${selectedSegmentPath} L ${xFor(activeSelection.end)} ${height - padding.bottom} L ${xFor(activeSelection.start)} ${height - padding.bottom} Z`
        : '';
    const measureLineY = measureAnchorY != null && measureEndY != null
        ? Math.max(padding.top + 12, Math.min(measureAnchorY, measureEndY) - 18)
        : null;
    const measureTooltipWidth = isMobileChart ? 260 : 196;
    const measureTooltipHeight = isMobileChart ? 92 : 70;
    const plotRight = width - padding.right;
    const plotMidX = padding.left + (plotRight - padding.left) / 2;
    const measureTooltipGap = 14;
    const measureTooltipAnchorX = activeSelection ? (xFor(activeSelection.start) + xFor(activeSelection.end)) / 2 : (hoverX ?? measureEndX);
    const measureTooltipAnchorY = measureLineY ?? (hoverY ?? measureEndY);
    const measureTooltipX = measureTooltipAnchorX == null
        ? 0
        : clamp(measureTooltipAnchorX - measureTooltipWidth / 2, padding.left + 4, plotRight - measureTooltipWidth - 4);
    const measureTooltipY = measureTooltipAnchorY == null
        ? 0
        : measureTooltipAnchorY - measureTooltipGap - measureTooltipHeight >= padding.top
            ? measureTooltipAnchorY - measureTooltipGap - measureTooltipHeight
            : Math.min(measureTooltipAnchorY + measureTooltipGap, height - measureTooltipHeight - padding.bottom);
    const measureDeltaText = measureDelta == null
        ? '-'
        : `${measureDelta > 0 ? '+' : measureDelta < 0 ? '-' : ''}${formatIndexPrice(Math.abs(measureDelta))} puan`;
    const measureClass = measureReturn == null ? '' : measureReturn >= 0 ? 'is-positive' : 'is-negative';
    const measureLeftX = measureAnchorX != null && measureEndX != null ? Math.min(measureAnchorX, measureEndX) : null;
    const measureRightX = measureAnchorX != null && measureEndX != null ? Math.max(measureAnchorX, measureEndX) : null;
    const tooltipWidth = 145;
    const tooltipHeight = 60;
    const tooltipGap = 16;
    const tooltipX = hoverX == null
        ? 0
        : hoverX < plotMidX && hoverX + tooltipGap + tooltipWidth <= plotRight
            ? clamp(hoverX + tooltipGap, padding.left, plotRight - tooltipWidth)
            : Math.max(padding.left, hoverX - tooltipWidth - tooltipGap);
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
                onPointerDown={handlePointerDown}
                onPointerUp={handlePointerUp}
                onPointerCancel={handlePointerUp}
                onPointerLeave={() => {
                    touchPointersRef.current.clear();
                    setHoverIndex(null);
                    setMeasureAnchorIndex(null);
                }}
            >
                <defs>
                    <linearGradient id="areaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor={activeSelection ? '#9aa3ad' : chartColor} stopOpacity={activeSelection ? '0.12' : '0.24'} />
                        <stop offset="100%" stopColor={activeSelection ? '#9aa3ad' : chartColor} stopOpacity="0" />
                    </linearGradient>
                    <linearGradient id="indexSelectedAreaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor={selectionColor} stopOpacity="0.28" />
                        <stop offset="100%" stopColor={selectionColor} stopOpacity="0.02" />
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
                        <text x={width - padding.right + 10} y={yFor(value) + 4} fill="rgba(255,255,255,0.4)" fontSize={isMobileChart ? '15' : '11'} fontFamily="monospace">
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
                        <text x={x} y={height - 15} fill="rgba(255,255,255,0.5)" fontSize={isMobileChart ? '15' : '11'} fontFamily="monospace" textAnchor="middle">
                            {label}
                        </text>
                    </g>
                ))}

                {/* Alan ve Çizgi (Ağ/Line) (Açılış, yüksek, düşük kullanılmıyor, SADECE CLOSE) */}
                <path d={areaData} fill="url(#areaGrad)" />
                <path
                    d={pathData}
                    fill="none"
                    stroke={activeSelection ? 'rgba(154, 163, 173, 0.48)' : chartColor}
                    strokeWidth="2.5"
                    strokeLinejoin="round"
                    strokeLinecap="round"
                />
                {activeSelection && selectedSegmentArea && (
                    <>
                        <path d={selectedSegmentArea} fill="url(#indexSelectedAreaGrad)" />
                        <path
                            d={selectedSegmentPath}
                            fill="none"
                            stroke={selectionColor}
                            strokeWidth="2.8"
                            strokeLinejoin="round"
                            strokeLinecap="round"
                        />
                    </>
                )}

                {/* Uç Noktası (Kapanış) Dot ve Pulse */}
                <circle cx={xFor(validPoints.length - 1)} cy={yFor(last)} r="4" fill={chartColor} opacity="0.6">
                    <animate attributeName="r" values="4; 14; 14" keyTimes="0; 0.5; 1" dur="2s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.6; 0; 0" keyTimes="0; 0.5; 1" dur="2s" repeatCount="indefinite" />
                </circle>
                <circle cx={xFor(validPoints.length - 1)} cy={yFor(last)} r="4" fill={chartColor} />

                {/* Anlık Fiyat İşareti */}
                <g transform={`translate(${width - padding.right}, ${yFor(last)})`}>
                    <rect x="0" y={isMobileChart ? '-12' : '-10'} width={isMobileChart ? '86' : '65'} height={isMobileChart ? '24' : '20'} fill={chartColor} rx="2" />
                    <path d="M 0 0 L 6 -6 L 6 6 Z" fill={chartColor} transform="translate(-5, 0)" />
                    <text x={isMobileChart ? '43' : '32'} y={isMobileChart ? '5' : '3'} fill="#ffffff" fontSize={isMobileChart ? '15' : '11'} fontFamily="monospace" textAnchor="middle" fontWeight="bold">
                        {formatIndexPrice(last)}
                    </text>
                </g>

                {/* Drag-to-measure */}
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
                    <g className="indices-chart-measure">
                        <line
                            x1={measureAnchorX}
                            y1={measureAnchorY}
                            x2={measureEndX}
                            y2={measureEndY}
                            className="indices-chart-measure-path"
                        />
                        <line
                            x1={measureLeftX}
                            y1={measureLineY}
                            x2={measureRightX}
                            y2={measureLineY}
                            className="indices-chart-measure-line"
                        />
                        <line
                            x1={measureAnchorX}
                            y1={measureLineY - 5}
                            x2={measureAnchorX}
                            y2={measureLineY + 5}
                            className="indices-chart-measure-line"
                        />
                        <line
                            x1={measureEndX}
                            y1={measureLineY - 5}
                            x2={measureEndX}
                            y2={measureLineY + 5}
                            className="indices-chart-measure-line"
                        />
                        <circle cx={measureAnchorX} cy={measureAnchorY} r={isMobileChart ? '5.2' : '4.2'} className="indices-chart-measure-point" fill={selectionColor} />
                        <circle cx={measureEndX} cy={measureEndY} r={isMobileChart ? '5.2' : '4.2'} className="indices-chart-measure-point" fill={selectionColor} />
                        <g
                            className="indices-chart-measure-tooltip"
                            style={{ transform: `translate(${measureTooltipX}px, ${measureTooltipY}px)` }}
                        >
                            <rect width={measureTooltipWidth} height={measureTooltipHeight} rx="7" className="indices-chart-measure-tooltip-bg" />
                            <text x="12" y={isMobileChart ? '24' : '18'} className="indices-chart-measure-muted">
                                {new Intl.DateTimeFormat('tr-TR', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }).format(new Date(measureAnchorPoint.time))}
                                {' - '}
                                {new Intl.DateTimeFormat('tr-TR', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }).format(new Date(measureEndPoint.time))}
                            </text>
                            <text x="12" y={isMobileChart ? '56' : '42'} className={`indices-chart-measure-value ${measureClass}`}>
                                {formatTablePct(measureReturn)}
                            </text>
                            <text x="12" y={isMobileChart ? '80' : '60'} className="indices-chart-measure-muted">
                                {measureDeltaText}
                            </text>
                        </g>
                    </g>
                )}

                {/* Hover Tooltip */}
                {measureAnchorIndex == null && !activeSelection && hoverPoint && hoverX != null && hoverY != null && (
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
