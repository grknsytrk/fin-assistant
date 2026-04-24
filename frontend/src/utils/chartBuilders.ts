import type { KapQuarter } from '../api/types';
import { _resolveMetricValue, _resolveMetricDisplay, _periodLabel, intSafe } from './formatters';

export type SeriesPoint = {
    key: string;
    label: string;
    value: number;
    display: string;
};

export const CHART_WINDOW_QUARTERS = 10;

export function _buildMetricSeries(quarters: KapQuarter[], metricKey: string, asQuarterlyFlow: boolean): SeriesPoint[] {
    return quarters
        .map((q, idx) => {
            const numeric = _resolveMetricValue(quarters, idx, metricKey, asQuarterlyFlow);
            if (numeric === null) {
                return null;
            }
            return {
                key: `${q.quarter}-${metricKey}`,
                label: _periodLabel(q.year, q.period, q.quarter),
                value: numeric,
                display: _resolveMetricDisplay(quarters, idx, metricKey, asQuarterlyFlow),
            };
        })
        .filter((row): row is SeriesPoint => row !== null);
}

export function _buildCustomSeries(
    quarters: KapQuarter[],
    keySuffix: string,
    resolver: (rows: KapQuarter[], idx: number) => number | null,
    displayFormatter: (val: number, currency: string) => string
): SeriesPoint[] {
    return quarters
        .map((q, idx) => {
            const numeric = resolver(quarters, idx);
            if (numeric === null) {
                return null;
            }
            return {
                key: `${q.quarter}-${keySuffix}`,
                label: _periodLabel(q.year, q.period, q.quarter),
                value: numeric,
                display: displayFormatter(numeric, q.currency || 'TL'),
            };
        })
        .filter((row): row is SeriesPoint => row !== null);
}

export function _takeLastSeries(series: SeriesPoint[], count = 5): SeriesPoint[] {
    if (series.length <= count) {
        return series;
    }
    return series.slice(-count);
}

export function _buildRatioSeries(
    quarters: KapQuarter[],
    numeratorKey: string,
    denominatorKey: string,
    numeratorAsQuarterlyFlow: boolean,
    denominatorAsQuarterlyFlow: boolean,
    scale = 100,
    suffixFormatter: (val: number) => string
): SeriesPoint[] {
    return quarters
        .map((q, idx) => {
            const numerator = _resolveMetricValue(quarters, idx, numeratorKey, numeratorAsQuarterlyFlow);
            const denominator = _resolveMetricValue(quarters, idx, denominatorKey, denominatorAsQuarterlyFlow);
            if (numerator === null || denominator === null || denominator === 0) {
                return null;
            }
            const value = (numerator / denominator) * scale;
            return {
                key: `${q.quarter}-${numeratorKey}-${denominatorKey}`,
                label: _periodLabel(q.year, q.period, q.quarter),
                value,
                display: suffixFormatter(value),
            };
        })
        .filter((row): row is SeriesPoint => row !== null);
}

export function prepareOrderedQuarters(snapshot: any): KapQuarter[] {
    if (!snapshot || !snapshot.quarters) return [];
    
    const byPeriod = new Map<string, KapQuarter>();
    for (const q of snapshot.quarters) {
        const key = `${intSafe(q.year)}-${intSafe(q.period)}`;
        if (!byPeriod.has(key)) {
            byPeriod.set(key, q);
        }
    }
    return [...byPeriod.values()].sort((a, b) => {
        const aVal = intSafe(a.year) * 100 + (intSafe(a.period) > 0 ? (intSafe(a.period) <= 4 ? intSafe(a.period) * 3 : intSafe(a.period)) : 0);
        const bVal = intSafe(b.year) * 100 + (intSafe(b.period) > 0 ? (intSafe(b.period) <= 4 ? intSafe(b.period) * 3 : intSafe(b.period)) : 0);
        return aVal - bVal;
    });
}
