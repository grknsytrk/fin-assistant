import { describe, it, expect } from 'vitest';
import type { KapQuarter } from '../api/types';
import { ttmSum } from './financialMetrics';
const quarter = (year: number, period: number, value: number | null) => ({
    year, period, metrics_quarterly: { net_kar: { value } }, metrics: { net_kar: { value: 999 } },
}) as unknown as KapQuarter;
const full = () => [quarter(2025, 2, 1), quarter(2025, 3, 2), quarter(2025, 4, 3), quarter(2026, 1, 4)];
describe('strict TTM', () => {
    it('sums consecutive quarters across the year boundary', () => expect(ttmSum(full(), 'net_kar')).toBe(10));
    it('rejects short histories', () => expect(ttmSum(full().slice(1), 'net_kar')).toBeNull());
    it('rejects gaps and duplicates', () => {
        const rows = full(); rows[1] = quarter(2025, 2, 2);
        expect(ttmSum(rows, 'net_kar')).toBeNull();
    });
    it('does not use a YTD fallback for missing quarterly data', () => {
        const rows = full(); rows[0] = quarter(2025, 2, null);
        expect(ttmSum(rows, 'net_kar')).toBeNull();
    });
    it('preserves zero', () => expect(ttmSum(full().map(q => quarter(q.year, q.period, 0)), 'net_kar')).toBe(0));
    it('rejects nonfinite values', () => {
        const rows = full(); rows[0] = quarter(2025, 2, Infinity);
        expect(ttmSum(rows, 'net_kar')).toBeNull();
    });
});
