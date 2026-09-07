import type { KapQuarter } from '../api/types';
import { _resolveMetricValueByPriority } from './formatters';

export function hasConsecutiveQuarters(quarters: KapQuarter[]): boolean {
    if (quarters.length !== 4) return false;
    const ids = quarters.map(q => {
        const year = Number(q.year), period = Number(q.period);
        return Number.isInteger(year) && year > 0 && Number.isInteger(period) && period >= 1 && period <= 4
            ? year * 4 + period : NaN;
    });
    return ids.every((id, i) => Number.isFinite(id) && (i === 0 || id === ids[i - 1] + 1));
}

export function ttmSum(quarters: KapQuarter[], metric: string): number | null {
    const tail = quarters.slice(-4);
    if (!hasConsecutiveQuarters(tail)) return null;
    const values = tail.map(q => _resolveMetricValueByPriority(q, metric, ['metrics_quarterly']));
    if (values.some(v => v === null || !Number.isFinite(v))) return null;
    return values.reduce<number>((sum, v) => sum + v!, 0);
}
