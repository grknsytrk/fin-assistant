import { useRef, useState, type MouseEvent } from 'react';
import type { SeriesPoint } from '../../utils/chartBuilders';
import './Charts.css';

type TooltipState = {
    x: number;
    y: number;
    label: string;
    value: string;
};

export function BarChartCard({
    title,
    series,
    highlightedIndex,
    onHighlight,
}: {
    title: string;
    series: SeriesPoint[];
    highlightedIndex?: number | null;
    onHighlight?: (idx: number | null) => void;
}) {
    const cardRef = useRef<HTMLDivElement | null>(null);
    const [tooltip, setTooltip] = useState<TooltipState | null>(null);

    if (!series.length) {
        return null;
    }

    const width = 420;
    const height = 250;
    const padLeft = 34;
    const padRight = 28;
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
    };

    const onLeave = () => {
        setTooltip(null);
        if (onHighlight) onHighlight(null);
    };

    return (
        <div className="kap-chart-card" ref={cardRef}>
            <h4>{title}</h4>
            <svg
                className="kap-chart-svg"
                viewBox={`0 0 ${width} ${height}`}
                role="img"
                aria-label={title}
                onMouseLeave={onLeave}
            >
                {[0, 1, 2, 3, 4].map((i) => {
                    const yy = padTop + (plotHeight / 4) * i;
                    return <line key={`h-${i}`} x1={padLeft} y1={yy} x2={width - padRight} y2={yy} className="kap-grid-line" />;
                })}

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
                    const isFocus = highlightedIndex === idx;
                    const isFaded = highlightedIndex !== null && highlightedIndex !== undefined && !isFocus;
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
                                onMouseLeave={onLeave}
                            />
                            <text x={x + barWidth / 2} y={height - 16} textAnchor="middle" className="kap-x-label">
                                {point.label}
                            </text>
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
    highlightedIndex,
    onHighlight,
}: {
    title: string;
    series: SeriesPoint[];
    highlightedIndex?: number | null;
    onHighlight?: (idx: number | null) => void;
}) {
    const cardRef = useRef<HTMLDivElement | null>(null);
    const [tooltip, setTooltip] = useState<TooltipState | null>(null);

    if (!series.length) {
        return null;
    }

    const width = 420;
    const height = 250;
    const padLeft = 34;
    const padRight = 28;
    const padTop = 14;
    const padBottom = 44;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const values = series.map((s) => s.value);
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

    const path = points
        .map((point, idx) => `${idx === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
        .join(' ');

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
    };

    const onLeave = () => {
        setTooltip(null);
        if (onHighlight) onHighlight(null);
    };

    return (
        <div className="kap-chart-card" ref={cardRef}>
            <h4>{title}</h4>
            <svg
                className="kap-chart-svg"
                viewBox={`0 0 ${width} ${height}`}
                role="img"
                aria-label={title}
                onMouseLeave={onLeave}
            >
                {[0, 1, 2, 3, 4].map((i) => {
                    const yy = padTop + (plotHeight / 4) * i;
                    return <line key={`h-${i}`} x1={padLeft} y1={yy} x2={width - padRight} y2={yy} className="kap-grid-line" />;
                })}

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

                <path d={path} className="kap-line-path" />

                {points.map((point, idx) => {
                    const isFocus = highlightedIndex === idx;
                    const isFaded = highlightedIndex !== null && highlightedIndex !== undefined && !isFocus;
                    return (
                        <g key={point.key}>
                            <circle
                                cx={point.x}
                                cy={point.y}
                                r={isFocus ? 6 : 4}
                                className={`kap-line-point ${isFocus ? 'kap-point-highlighted' : ''} ${isFaded ? 'kap-point-faded' : ''}`}
                                onMouseEnter={(event) => onHover(event, point, idx)}
                                onMouseLeave={onLeave}
                            />
                            <text x={point.x} y={height - 16} textAnchor="middle" className="kap-x-label">
                                {point.label}
                            </text>
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
