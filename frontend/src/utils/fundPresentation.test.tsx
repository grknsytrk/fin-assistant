import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { FundPerformanceResponse, FundPricePoint } from '../api/types';
import {
    canonicalFundPrice,
    formatFundQuotePrice,
    formatFundReportDate,
    hasFundRangeStartCoverage,
    mergeFundPerformancePayloads,
} from './fundPresentation';

function performancePoint(date: string, price: number, source = 'tefasfon_funds'): FundPricePoint {
    return {
        fund_code: 'BOH',
        date,
        price,
        daily_return: null,
        aum: null,
        investor_count: null,
        share_count: null,
        source,
    };
}

function performancePayload(
    points: FundPricePoint[],
    metadata: Partial<FundPerformanceResponse['source_metadata']> = {},
    payload: Partial<FundPerformanceResponse> = {},
): FundPerformanceResponse {
    return {
        fund_code: 'BOH',
        status: 'ok',
        points,
        source: 'tefasfon_funds',
        as_of: points[points.length - 1]?.date || null,
        fetched_at: '2026-09-10T15:00:00Z',
        stale: false,
        period_stats: { as_of: points[points.length - 1]?.date || null, periods: [] },
        source_metadata: {
            source: 'tefasfon_funds',
            coverage_state: 'range_incomplete',
            resolution: 'daily',
            available_start_date: points[0]?.date || null,
            available_end_date: points[points.length - 1]?.date || null,
            ...metadata,
        },
        ...payload,
    };
}

describe('fund presentation helpers', () => {
    it('uses the canonical snapshot price and its currency', () => {
        expect(canonicalFundPrice(12.345678)).toBe(12.345678);
        expect(canonicalFundPrice(0)).toBeNull();
        expect(formatFundQuotePrice(12.345678, 'USD')).toMatch(/^USD /);
    });

    it('renders date-only values as reporting dates without UTC rollover', () => {
        render(<output>{formatFundReportDate('2026-09-06')}</output>);

        expect(screen.getByText(/6 Eyl 2026/i)).toBeTruthy();
    });

    it('rejects comparison ranges with a material history gap', () => {
        expect(hasFundRangeStartCoverage('2025-09-06', '2025-09-10')).toBe(true);
        expect(hasFundRangeStartCoverage('2025-09-06', '2025-09-20')).toBe(false);
    });

    it('keeps a wider history when a later backfill response is narrower', () => {
        const current = performancePayload(
            [
                performancePoint('2026-01-02', 1.0),
                performancePoint('2026-01-05', 1.1),
                performancePoint('2026-03-31', 1.2),
            ],
            {
                history_job: { job_id: 'history-1', fund_code: 'BOH', status: 'running' },
            },
        );
        const next = performancePayload(
            [
                performancePoint('2026-03-31', 1.2),
                performancePoint('2026-04-01', 1.25),
            ],
            {
                coverage_state: 'complete',
                history_job: { job_id: 'history-1', fund_code: 'BOH', status: 'succeeded' },
            },
        );

        const merged = mergeFundPerformancePayloads(current, next);

        expect(merged?.points.map((point) => point.date)).toEqual([
            '2026-01-02',
            '2026-01-05',
            '2026-03-31',
            '2026-04-01',
        ]);
        expect(merged?.source_metadata.history_job?.status).toBe('succeeded');
        expect(merged?.source_metadata.available_start_date).toBe('2026-01-02');
        expect(merged?.source_metadata.available_end_date).toBe('2026-04-01');
    });

    it('preserves existing points when a refresh response has no usable data', () => {
        const current = performancePayload([performancePoint('2026-01-02', 1.0)]);
        const unavailable = performancePayload([], { coverage_state: 'unavailable' }, { status: 'unavailable' });

        const merged = mergeFundPerformancePayloads(current, unavailable);

        expect(merged?.points).toEqual(current.points);
        expect(merged?.status).toBe('ok');
    });
});
