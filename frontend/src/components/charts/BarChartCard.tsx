import { useRef, useState, type KeyboardEvent, type MouseEvent } from 'react';
import type { SeriesPoint } from '../../utils/chartBuilders';
import './Charts.css';

type TooltipState = {
    x: number;
    y: number;
    label: string;
    value: string;
    average?: string;
};

const MAX_VISIBLE_X_LABELS = 8;

function formatYAxisTick(value: number, series: SeriesPoint[]): string {
    const sample = series.find((point) => point.display)?.display.trim() || '';
    const unit = sample.endsWith('%') ? '%' : /x$/i.test(sample) ? 'x' : '';
    const absolute = Math.abs(value);
    let divisor = 1;
    let scale = '';
    if (absolute >= 1_000_000_000) {
        divisor = 1_000_000_000;
        scale = ' mr.';
    } else if (absolute >= 1_000_000) {
        divisor = 1_000_000;
        scale = ' mio.';
    } else if (absolute >= 1_000) {
        divisor = 1_000;
        scale = ' bin';
    }
    const scaled = value / divisor;
    const formatted = scaled.toLocaleString('tr-TR', {
        maximumFractionDigits: Math.abs(scaled) >= 100 ? 0 : 1,
    });
    return `${formatted}${scale}${unit}`;
}

function shouldShowXAxisLabel(index: number, total: number): boolean {
    if (total <= 10) {
        return true;
    }
    if (index === 0 || index === total - 1) {
        return true;
    }
    const visibleLabels = total > 16 ? 6 : MAX_VISIBLE_X_LABELS;
    const interval = Math.max(2, Math.ceil((total - 1) / (visibleLabels - 1)));
    return index % interval === 0;
}

export function BarChartCard({
    title,
    series,
    highlightedIndex,
    highlightedQuarterKey,
    onHighlight,
    onHighlightQuarterKey,
    onOpen,
    className,
}: {
    title: string;
    series: SeriesPoint[];
    highlightedIndex?: number | null;
    highlightedQuarterKey?: string | null;
    onHighlight?: (idx: number | null) => void;
    onHighlightQuarterKey?: (key: string | null) => void;
    onOpen?: () => void;
    className?: string;
}) {
    const cardRef = useRef<HTMLDivElement | null>(null);
    const [tooltip, setTooltip] = useState<TooltipState | null>(null);

    if (!series.length) {
        return null;
    }

    const width = 420;
    const height = 250;
    const padLeft = 62;
    const padRight = 20;
    const padTop = 14;
    const padBottom = 44;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const values = series.map((s) => s.value);
    let minVal = Math.min(...values, 0);
    let maxVal = Math.max(...values, 0);
    if (maxVal === minVal) {
        maxVal = minVal + 1;
    }
    const range = maxVal - minVal;
    const y = (v: number) => padTop + ((maxVal - v) / range) * plotHeight;
    const zeroY = y(0);

    const slot = plotWidth / series.length;
    const barWidth = Math.min(52, slot * 0.68);

    const onHover = (event: MouseEvent<SVGElement>, point: SeriesPoint, idx: number) => {
        const cardRect = cardRef.current?.getBoundingClientRect();
        if (!cardRect) {
            return;
        }
        setTooltip({
            x: event.clientX - cardRect.left,
            y: event.clientY - cardRect.top,
            label: point.label,
            value: point.display,
        });
        if (onHighlight) onHighlight(idx);
        if (onHighlightQuarterKey) onHighlightQuarterKey(point.quarterKey ?? null);
    };

    const clearHighlight = () => {
        if (onHighlight) onHighlight(null);
        if (onHighlightQuarterKey) onHighlightQuarterKey(null);
    };

    const onChartMove = (event: MouseEvent<SVGSVGElement>) => {
        const target = event.target;
        const isDataMark = target instanceof Element && Boolean(target.closest('.kap-bar-positive, .kap-bar-negative'));
        if (isDataMark) {
            return;
        }
        setTooltip(null);
        clearHighlight();
    };

    const onChartLeave = () => {
        setTooltip(null);
        clearHighlight();
    };

    const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
        if (!onOpen) {
            return;
        }
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onOpen();
        }
    };

    return (
        <div
            className={`kap-chart-card${onOpen ? ' kap-chart-card-clickable' : ''}${className ? ` ${className}` : ''}`}
            ref={cardRef}
            role={onOpen ? 'button' : undefined}
            tabIndex={onOpen ? 0 : undefined}
            aria-label={onOpen ? `${title} grafiğini büyüt` : undefined}
            onClick={onOpen}
            onKeyDown={onKeyDown}
            onMouseLeave={onChartLeave}
        >
            <h4>{title}</h4>
            <svg
                className="kap-chart-svg"
                viewBox={`0 0 ${width} ${height}`}
                role="img"
                aria-label={title}
                onMouseMove={onChartMove}
                onMouseLeave={onChartLeave}
            >
                {[0, 1, 2, 3, 4].map((i) => {
                    const yy = padTop + (plotHeight / 4) * i;
                    const tickValue = maxVal - (range / 4) * i;
                    return (
                        <g key={`h-${i}`}>
                            <line x1={padLeft} y1={yy} x2={width - padRight} y2={yy} className="kap-grid-line" />
                            <text x={padLeft - 7} y={yy + 3} textAnchor="end" className="kap-y-label">
                                {formatYAxisTick(tickValue, series)}
                            </text>
                        </g>
                    );
                })}

                <line x1={padLeft} y1={padTop} x2={padLeft} y2={padTop + plotHeight} className="kap-axis-line" />

                {series.map((point, idx) => {
                    const xCenter = padLeft + slot * idx + slot / 2;
                    return (
                        <line
                            key={`${point.key}-v`}
                            x1={xCenter}
                            y1={padTop}
                            x2={xCenter}
                            y2={padTop + plotHeight}
                            className="kap-grid-line kap-grid-line-v"
                        />
                    );
                })}

                <line x1={padLeft} y1={zeroY} x2={width - padRight} y2={zeroY} className="kap-axis-line" />

                {series.map((point, idx) => {
                    const x = padLeft + slot * idx + (slot - barWidth) / 2;
                    const yVal = y(point.value);
                    const top = point.value >= 0 ? yVal : zeroY;
                    const h = Math.max(2, Math.abs(yVal - zeroY));
                    const hasHighlight = highlightedQuarterKey != null || (highlightedIndex !== null && highlightedIndex !== undefined);
                    const isFocus = highlightedQuarterKey != null
                        ? point.quarterKey === highlightedQuarterKey
                        : highlightedIndex === idx;
                    const isFaded = hasHighlight && !isFocus;
                    return (
                        <g key={point.key}>
                            <rect
                                x={x}
                                y={top}
                                width={barWidth}
                                height={h}
                                rx={1}
                                className={`${point.value < 0 ? 'kap-bar-negative' : 'kap-bar-positive'} ${isFocus ? 'kap-bar-highlighted' : ''} ${isFaded ? 'kap-bar-faded' : ''}`}
                                onMouseEnter={(event) => onHover(event, point, idx)}
                            />
                            {shouldShowXAxisLabel(idx, series.length) && (
                                <text x={x + barWidth / 2} y={height - 16} textAnchor="middle" className="kap-x-label">
                                    {point.label}
                                </text>
                            )}
                        </g>
                    );
                })}

            </svg>
            {tooltip && (
                <div className="kap-chart-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
                    <div className="kap-chart-tooltip-label">{tooltip.label}</div>
                    <div className="kap-chart-tooltip-value">{tooltip.value}</div>
                </div>
            )}
        </div>
    );
}

export function LineChartCard({
    title,
    series,
    averageSeries,
    averageLabel,
    highlightedIndex,
    highlightedKey,
    highlightedQuarterKey,
    onHighlight,
    onHighlightKey,
    onHighlightQuarterKey,
    onOpen,
    className,
}: {
    title: string;
    series: SeriesPoint[];
    averageSeries?: SeriesPoint[];
    averageLabel?: string;
    highlightedIndex?: number | null;
    highlightedKey?: string | null;
    highlightedQuarterKey?: string | null;
    onHighlight?: (idx: number | null) => void;
    onHighlightKey?: (key: string | null) => void;
    onHighlightQuarterKey?: (key: string | null) => void;
    onOpen?: () => void;
    className?: string;
}) {
    const cardRef = useRef<HTMLDivElement | null>(null);
    const [tooltip, setTooltip] = useState<TooltipState | null>(null);

    if (!series.length) {
        return null;
    }

    const width = 420;
    const height = 250;
    const padLeft = 62;
    const padRight = 20;
    const padTop = 14;
    const padBottom = 44;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const values = series.map((point) => point.value);
    if (averageSeries?.length) values.push(...averageSeries.map((point) => point.value));
    const minRaw = Math.min(...values);
    const maxRaw = Math.max(...values);
    const span = Math.max(maxRaw - minRaw, Math.abs(maxRaw) * 0.1, 1);
    const minVal = minRaw - span * 0.12;
    const maxVal = maxRaw + span * 0.12;
    const range = maxVal - minVal;
    const y = (v: number) => padTop + ((maxVal - v) / range) * plotHeight;
    const stepX = series.length > 1 ? plotWidth / (series.length - 1) : 0;

    const points = series.map((point, idx) => ({
        x: padLeft + idx * stepX,
        y: y(point.value),
        ...point,
    }));
    const averagePoints = (averageSeries || []).map((point, idx) => ({
        x: padLeft + idx * stepX,
        y: y(point.value),
        ...point,
    }));
    const path = points
        .map((point, idx) => `${idx === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
        .join(' ');
    const averagePath = averagePoints
        .map((point, idx) => `${idx === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
        .join(' ');
    const highlightedPoint = highlightedQuarterKey != null
        ? points.find((point) => point.quarterKey === highlightedQuarterKey)
        : highlightedKey != null
        ? points.find((point) => point.key === highlightedKey)
        : highlightedIndex != null
            ? points[highlightedIndex]
            : null;
    const highlightedAveragePoint = highlightedQuarterKey != null
        ? averagePoints.find((point) => point.quarterKey === highlightedQuarterKey)
        : highlightedKey != null
        ? averagePoints.find((point) => point.key === highlightedKey)
        : highlightedIndex != null
            ? averagePoints[highlightedIndex]
            : null;

    const onHover = (event: MouseEvent<SVGElement>, point: SeriesPoint, idx: number) => {
        const cardRect = cardRef.current?.getBoundingClientRect();
        if (!cardRect) {
            return;
        }
        setTooltip({
            x: event.clientX - cardRect.left,
            y: event.clientY - cardRect.top,
            label: point.label,
            value: point.display,
            average: averageSeries?.find((averagePoint) => averagePoint.key === point.key)?.display,
        });
        if (onHighlight) onHighlight(idx);
        if (onHighlightKey) onHighlightKey(point.key);
        if (onHighlightQuarterKey) onHighlightQuarterKey(point.quarterKey ?? null);
    };

    const clearHighlight = () => {
        if (onHighlight) onHighlight(null);
        if (onHighlightKey) onHighlightKey(null);
        if (onHighlightQuarterKey) onHighlightQuarterKey(null);
    };

    const onChartMove = (event: MouseEvent<SVGSVGElement>) => {
        const target = event.target;
        const isDataMark = target instanceof Element && Boolean(target.closest('.kap-line-point'));
        if (isDataMark) {
            return;
        }
        setTooltip(null);
        clearHighlight();
    };

    const onChartLeave = () => {
        setTooltip(null);
        clearHighlight();
    };

    const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
        if (!onOpen) {
            return;
        }
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onOpen();
        }
    };

    return (
        <div
            className={`kap-chart-card${onOpen ? ' kap-chart-card-clickable' : ''}${className ? ` ${className}` : ''}`}
            ref={cardRef}
            role={onOpen ? 'button' : undefined}
            tabIndex={onOpen ? 0 : undefined}
            aria-label={onOpen ? `${title} grafiğini büyüt` : undefined}
            onClick={onOpen}
            onKeyDown={onKeyDown}
            onMouseLeave={onChartLeave}
        >
            <h4>{title}</h4>
            <svg
                className="kap-chart-svg"
                viewBox={`0 0 ${width} ${height}`}
                role="img"
                aria-label={title}
                onMouseMove={onChartMove}
                onMouseLeave={onChartLeave}
            >
                {[0, 1, 2, 3, 4].map((i) => {
                    const yy = padTop + (plotHeight / 4) * i;
                    const tickValue = maxVal - (range / 4) * i;
                    return (
                        <g key={`h-${i}`}>
                            <line x1={padLeft} y1={yy} x2={width - padRight} y2={yy} className="kap-grid-line" />
                            <text x={padLeft - 7} y={yy + 3} textAnchor="end" className="kap-y-label">
                                {formatYAxisTick(tickValue, series)}
                            </text>
                        </g>
                    );
                })}

                <line x1={padLeft} y1={padTop} x2={padLeft} y2={padTop + plotHeight} className="kap-axis-line" />
                <line x1={padLeft} y1={padTop + plotHeight} x2={width - padRight} y2={padTop + plotHeight} className="kap-axis-line" />

                {points.map((point) => (
                    <line
                        key={`${point.key}-v`}
                        x1={point.x}
                        y1={padTop}
                        x2={point.x}
                        y2={padTop + plotHeight}
                        className="kap-grid-line kap-grid-line-v"
                    />
                ))}

                {highlightedPoint ? (
                    <line
                        x1={highlightedPoint.x}
                        y1={padTop}
                        x2={highlightedPoint.x}
                        y2={padTop + plotHeight}
                        className="kap-chart-highlight-line"
                    />
                ) : null}

                <path d={path} className="kap-line-path" />

                {averagePoints.length ? (
                    <>
                        <path d={averagePath} className="kap-average-line" />
                        {averageLabel ? (
                            <text
                                x={averagePoints[averagePoints.length - 1].x}
                                y={Math.max(padTop + 11, averagePoints[averagePoints.length - 1].y - 5)}
                                textAnchor="end"
                                className="kap-average-label"
                            >
                                {averageLabel}
                            </text>
                        ) : null}
                    </>
                ) : null}

                {points.map((point, idx) => {
                    const isFocus = highlightedQuarterKey != null
                        ? point.quarterKey === highlightedQuarterKey
                        : highlightedKey != null
                            ? point.key === highlightedKey
                            : highlightedIndex === idx;
                    const hasHighlight = highlightedQuarterKey != null || highlightedKey != null || (highlightedIndex !== null && highlightedIndex !== undefined);
                    const isFaded = hasHighlight && !isFocus;
                    return (
                        <g key={point.key}>
                            <circle
                                cx={point.x}
                                cy={point.y}
                                r={isFocus ? 6 : 4}
                                className={`kap-line-point ${isFocus ? 'kap-point-highlighted' : ''} ${isFaded ? 'kap-point-faded' : ''}`}
                                onMouseEnter={(event) => onHover(event, point, idx)}
                            />
                            {shouldShowXAxisLabel(idx, series.length) && (
                                <text x={point.x} y={height - 16} textAnchor="middle" className="kap-x-label">
                                    {point.label}
                                </text>
                            )}
                        </g>
                    );
                })}
                {highlightedAveragePoint ? (
                    <circle
                        cx={highlightedAveragePoint.x}
                        cy={highlightedAveragePoint.y}
                        r="4"
                        className="kap-average-point-highlighted"
                    />
                ) : null}
            </svg>
            {averagePoints.length ? (
                <div className="kap-chart-legend" aria-label="Grafik göstergesi">
                    <span className="kap-chart-legend-item">
                        <span className="kap-chart-legend-swatch" />
                        Hisse çarpanı
                    </span>
                    <span className="kap-chart-legend-item">
                        <span className="kap-chart-legend-swatch kap-chart-legend-swatch-average" />
                        Döneme kadar ortalama
                    </span>
                </div>
            ) : null}
            {tooltip && (
                <div className="kap-chart-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
                    <div className="kap-chart-tooltip-label">{tooltip.label}</div>
                    <div className="kap-chart-tooltip-value">{tooltip.value}</div>
                    {tooltip.average ? <div className="kap-chart-tooltip-average">Ort. {tooltip.average}</div> : null}
                </div>
            )}
        </div>
    );
}
