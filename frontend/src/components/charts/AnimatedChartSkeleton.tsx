import { useId } from 'react';
import './AnimatedChartSkeleton.css';

export interface AnimatedChartSkeletonProps {
    seed: string;
    values?: number[];
    width?: number;
    height?: number;
    padding?: {
        top: number;
        right: number;
        bottom: number;
        left: number;
    };
    gridRatios?: number[];
    pointCount?: number;
    preserveAspectRatio?: string;
    className?: string;
    ariaLabel?: string;
}

const DEFAULT_GRID_RATIOS = [0.24, 0.5, 0.76];

function buildSeededWaveValues(seed: string, pointCount: number): number[] {
    let state = seed
        .split('')
        .reduce((acc, char) => ((acc * 31 + char.charCodeAt(0)) >>> 0), 7);
    const nextRandom = () => {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 0xffffffff;
    };

    const trend = (nextRandom() - 0.5) * 0.24;
    let level = 0.54 + (nextRandom() - 0.5) * 0.18;

    return Array.from({ length: pointCount }, (_, index) => {
        if (index > 0) {
            const burst = index % 7 === 0 ? (nextRandom() - 0.5) * 0.22 : 0;
            const drift = trend / pointCount + (nextRandom() - 0.5) * 0.16 + burst;
            level = Math.min(0.92, Math.max(0.08, level + drift));
        }
        return 100 - level * 80;
    });
}

export default function AnimatedChartSkeleton({
    seed,
    values,
    width = 340,
    height = 125,
    padding = { top: 10, right: 6, bottom: 16, left: 6 },
    gridRatios = DEFAULT_GRID_RATIOS,
    pointCount = 44,
    preserveAspectRatio = 'xMidYMid meet',
    className = '',
    ariaLabel = 'Grafik verisi yükleniyor',
}: AnimatedChartSkeletonProps) {
    const gradientId = useId().replace(/:/g, '');
    const series = values && values.length >= 2 ? values : buildSeededWaveValues(seed, pointCount);
    const plotLeft = padding.left;
    const plotRight = Math.max(plotLeft + 1, width - padding.right);
    const plotTop = padding.top;
    const plotBottom = Math.max(plotTop + 1, height - padding.bottom);
    const plotWidth = plotRight - plotLeft;
    const plotHeight = plotBottom - plotTop;

    const min = Math.min(...series);
    const span = Math.max(0.01, Math.max(...series) - min);

    const points = series.map((value, index) => {
        const x = plotLeft + (index / Math.max(1, series.length - 1)) * plotWidth;
        const y = plotBottom - ((value - min) / span) * plotHeight * 0.86 - plotHeight * 0.07;
        return [x, y] as const;
    });

    const linePath = points
        .map(([x, y], index) => `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`)
        .join(' ');
    const areaPath = `${linePath} L ${points[points.length - 1][0].toFixed(2)} ${plotBottom.toFixed(2)} L ${points[0][0].toFixed(2)} ${plotBottom.toFixed(2)} Z`;

    return (
        <div className={`animated-chart-skeleton ${className}`.trim()} role="status" aria-live="polite" aria-label={ariaLabel}>
            <span className="animated-chart-skeleton-sr-only">{ariaLabel}…</span>
            <svg
                className="animated-chart-skeleton-svg"
                viewBox={`0 0 ${width} ${height}`}
                preserveAspectRatio={preserveAspectRatio}
                aria-hidden="true"
            >
                <defs>
                    <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor="rgba(255, 255, 255, 0.07)" />
                        <stop offset="100%" stopColor="rgba(255, 255, 255, 0)" />
                    </linearGradient>
                </defs>
                {gridRatios.map((ratio) => {
                    const y = plotTop + ratio * plotHeight;
                    return (
                        <line
                            key={ratio}
                            className="animated-chart-skeleton-gridline"
                            x1={plotLeft}
                            x2={plotRight}
                            y1={y}
                            y2={y}
                        />
                    );
                })}
                <path className="animated-chart-skeleton-area" d={areaPath} fill={`url(#${gradientId})`} />
                <path className="animated-chart-skeleton-track" d={linePath} />
                <path className="animated-chart-skeleton-line" d={linePath} pathLength={100} />
            </svg>
        </div>
    );
}
