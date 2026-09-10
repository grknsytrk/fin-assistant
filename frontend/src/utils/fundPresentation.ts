/** Pure fund quote helpers shared by the detail view and its tests. */

import type { FundPerformanceResponse, FundPricePoint } from '../api/types';

export function formatFundReportDate(value: string | null | undefined): string {
    if (!value) return '-';
    const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
    if (dateOnly) {
        const [, year, month, day] = dateOnly;
        return new Date(Number(year), Number(month) - 1, Number(day)).toLocaleDateString('tr-TR', {
            day: '2-digit', month: 'short', year: 'numeric',
        });
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleDateString('tr-TR', { day: '2-digit', month: 'short', year: 'numeric' });
}

export function formatFundQuotePrice(value: number | null | undefined, currency = 'TRY'): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const prefix = currency === 'TRY' ? '₺' : `${currency} `;
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 4,
        maximumFractionDigits: 6,
    })}`;
}

export function canonicalFundPrice(value: number | null | undefined): number | null {
    return value != null && Number.isFinite(value) && value > 0 ? value : null;
}

export function hasFundRangeStartCoverage(startIso: string, actualStartIso: string, maximumGapDays = 7): boolean {
    const requested = Date.parse(`${startIso}T00:00:00Z`);
    const actual = Date.parse(`${actualStartIso}T00:00:00Z`);
    return Number.isFinite(requested)
        && Number.isFinite(actual)
        && actual - requested <= maximumGapDays * 24 * 60 * 60 * 1000;
}

const FUND_PERFORMANCE_SOURCE_PRIORITY: Record<string, number> = {
    tefasfon_funds: 90,
    tefas_direct_funds: 85,
    fintables_udf_history: 70,
    legacy_json: 10,
};

const FUND_PERFORMANCE_COVERAGE_PRIORITY: Record<string, number> = {
    complete: 4,
    upgrading: 3,
    range_incomplete: 2,
    unavailable: 1,
};

function validPerformancePoints(points: FundPricePoint[] | undefined): FundPricePoint[] {
    return [...(points || [])]
        .filter((point) => point.date && Number.isFinite(Number(point.price)) && Number(point.price) > 0)
        .sort((a, b) => a.date.localeCompare(b.date));
}

function performanceSourcePriority(source: string | null | undefined): number {
    return FUND_PERFORMANCE_SOURCE_PRIORITY[String(source || '').trim().toLowerCase()] || 0;
}

function performancePointCompleteness(point: FundPricePoint): number {
    return [point.price, point.daily_return, point.aum, point.investor_count, point.share_count]
        .filter((value) => value != null && Number.isFinite(Number(value)))
        .length;
}

function performanceFetchedAt(payload: FundPerformanceResponse): number {
    const value = payload.fetched_at || payload.source_metadata?.fetched_at || null;
    const timestamp = value ? Date.parse(value) : NaN;
    return Number.isFinite(timestamp) ? timestamp : 0;
}

function comparePerformancePoints(
    candidate: FundPricePoint,
    candidatePayload: FundPerformanceResponse,
    existing: FundPricePoint,
    existingPayload: FundPerformanceResponse,
): number {
    const sourceDifference = performanceSourcePriority(candidate.source) - performanceSourcePriority(existing.source);
    if (sourceDifference !== 0) return sourceDifference;

    const completenessDifference = performancePointCompleteness(candidate) - performancePointCompleteness(existing);
    if (completenessDifference !== 0) return completenessDifference;
    return performanceFetchedAt(candidatePayload) - performanceFetchedAt(existingPayload);
}

function comparePerformancePayloadRange(a: FundPerformanceResponse, b: FundPerformanceResponse): number {
    const aPoints = validPerformancePoints(a.points);
    const bPoints = validPerformancePoints(b.points);
    if (!aPoints.length && !bPoints.length) return 0;
    if (!aPoints.length) return -1;
    if (!bPoints.length) return 1;

    if (aPoints[0].date !== bPoints[0].date) return aPoints[0].date < bPoints[0].date ? 1 : -1;
    if (aPoints[aPoints.length - 1].date !== bPoints[bPoints.length - 1].date) {
        return aPoints[aPoints.length - 1].date > bPoints[bPoints.length - 1].date ? 1 : -1;
    }

    const aCoverage = FUND_PERFORMANCE_COVERAGE_PRIORITY[a.source_metadata?.coverage_state || ''] || 0;
    const bCoverage = FUND_PERFORMANCE_COVERAGE_PRIORITY[b.source_metadata?.coverage_state || ''] || 0;
    if (aCoverage !== bCoverage) return aCoverage - bCoverage;
    if (aPoints.length !== bPoints.length) return aPoints.length - bPoints.length;
    return performanceFetchedAt(a) - performanceFetchedAt(b);
}

function comparePerformancePayloadRecency(a: FundPerformanceResponse, b: FundPerformanceResponse): number {
    const aPoints = validPerformancePoints(a.points);
    const bPoints = validPerformancePoints(b.points);
    const aDate = aPoints[aPoints.length - 1]?.date || '';
    const bDate = bPoints[bPoints.length - 1]?.date || '';
    if (aDate !== bDate) return aDate > bDate ? 1 : -1;
    return performanceFetchedAt(a) - performanceFetchedAt(b);
}

/**
 * Keep the best available fund history when an async backfill response arrives.
 * A narrower response must not replace an already wider series, while newer
 * points and higher-priority sources still win for duplicate dates.
 */
export function mergeFundPerformancePayloads(
    current: FundPerformanceResponse | null | undefined,
    next: FundPerformanceResponse | null | undefined,
): FundPerformanceResponse | null {
    if (!current) return next || null;
    if (!next || current.fund_code !== next.fund_code) return next || current;

    const currentPoints = validPerformancePoints(current.points);
    const nextPoints = validPerformancePoints(next.points);
    if (!nextPoints.length && currentPoints.length) {
        return {
            ...current,
            source_metadata: {
                ...current.source_metadata,
                history_job: next.source_metadata?.history_job || current.source_metadata?.history_job || null,
            },
        };
    }
    if (!currentPoints.length) return next;

    const byDate = new Map<string, { point: FundPricePoint; payload: FundPerformanceResponse }>();
    for (const [payload, points] of [[current, currentPoints], [next, nextPoints]] as const) {
        for (const point of points) {
            const existing = byDate.get(point.date);
            if (!existing || comparePerformancePoints(point, payload, existing.point, existing.payload) >= 0) {
                byDate.set(point.date, { point, payload });
            }
        }
    }

    const mergedPoints = [...byDate.values()]
        .sort((a, b) => a.point.date.localeCompare(b.point.date))
        .map(({ point }) => point);
    const rangePayload = comparePerformancePayloadRange(current, next) >= 0 ? current : next;
    const recencyPayload = comparePerformancePayloadRecency(current, next) >= 0 ? current : next;
    const rangeMetadata = rangePayload.source_metadata || next.source_metadata;
    const nextJob = next.source_metadata?.history_job;
    const currentJob = current.source_metadata?.history_job;
    const firstDate = mergedPoints[0]?.date || rangePayload.source_metadata?.date_min || null;
    const lastDate = mergedPoints[mergedPoints.length - 1]?.date || rangePayload.as_of || null;

    return {
        ...rangePayload,
        status: recencyPayload.status || rangePayload.status,
        as_of: lastDate,
        fetched_at: recencyPayload.fetched_at || rangePayload.fetched_at,
        stale: recencyPayload.stale,
        period_stats: recencyPayload.period_stats || rangePayload.period_stats,
        points: mergedPoints,
        source_metadata: {
            ...rangeMetadata,
            history_job: nextJob || currentJob || null,
            full_history_requested: Boolean(
                current.source_metadata?.full_history_requested || next.source_metadata?.full_history_requested,
            ),
            final_points_count: mergedPoints.length,
            date_min: firstDate,
            date_max: lastDate,
            available_start_date: firstDate,
            available_end_date: lastDate,
            as_of: lastDate,
        },
    };
}
