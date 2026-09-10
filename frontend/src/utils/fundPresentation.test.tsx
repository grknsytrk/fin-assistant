import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { FundPerformanceResponse, FundPricePoint } from '../api/types';
import {
    canonicalFundPrice,
    buildFundHistoryDiagnosticLines,
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

    it('keeps Fintables as the source when async responses overlap', () => {
        const current = performancePayload([
            performancePoint('2026-04-01', 100.0, 'tefasfon_funds'),
        ]);
        const next = performancePayload([
            performancePoint('2026-04-01', 99.0, 'fintables_udf_history'),
        ], {
            history_source_used: 'fintables_udf_history',
            primary_source: 'fintables',
        });

        const merged = mergeFundPerformancePayloads(current, next);

        expect(merged?.points[0].price).toBe(99.0);
        expect(merged?.points[0].source).toBe('fintables_udf_history');
        expect(merged?.source_metadata.history_source_used).toBe('fintables_udf_history');
    });

    it('preserves existing points when a refresh response has no usable data', () => {
        const current = performancePayload([performancePoint('2026-01-02', 1.0)]);
        const unavailable = performancePayload([], { coverage_state: 'unavailable' }, { status: 'unavailable' });

        const merged = mergeFundPerformancePayloads(current, unavailable);

        expect(merged?.points).toEqual(current.points);
        expect(merged?.status).toBe('ok');
    });

    it('explains partial young-fund history and the background job state', () => {
        const performance = performancePayload(
            [
                performancePoint('2026-05-05', 1.0, 'fintables_udf_history'),
                performancePoint('2026-09-10', 1.1, 'fintables_udf_history'),
            ],
            {
                source: 'sqlite',
                history_source_used: 'fintables_udf_history',
                primary_source: 'fintables',
                coverage_state: 'range_incomplete',
                requested_start_date: '2025-09-10',
                requested_end_date: '2026-09-10',
                available_start_date: '2026-05-05',
                available_end_date: '2026-09-10',
                coverage_gap_business_days: 0,
                source_policy: 'fintables_primary_tefas_fallback',
                history_job: {
                    job_id: 'history-puk',
                    fund_code: 'PUK',
                    status: 'succeeded',
                    requested_start: '2025-09-10',
                    requested_end: '2026-09-10',
                    effective_start: '2025-09-09',
                    effective_end: '2026-09-10',
                    fintables_point_count: 2,
                    phase: 1,
                },
            },
        );

        const lines = buildFundHistoryDiagnosticLines({
            fundCode: 'PUK',
            performance,
            points: performance.points,
            periodReturns: { '1w': -1, '1m': -2, '3m': 3, '6m': null, ytd: null, '1y': null },
        });

        expect(lines.join('\n')).toContain('İstenen aralık:');
        expect(lines.join('\n')).toContain('Mevcut aralık:');
        expect(lines.join('\n')).toContain('Fintables UDF geçmişi');
        expect(lines.join('\n')).toContain('güncel uç mevcut');
        expect(lines.join('\n')).toContain('6A, YBB, 1Y');
        expect(lines.join('\n')).toContain('history-puk');
    });
});
