/** Pure fund quote helpers shared by the detail view and its tests. */

import type {
    FundHistoryJob,
    FundPeriodReturns,
    FundPerformanceResponse,
    FundPricePoint,
    FundYieldSummaryResponse,
} from '../api/types';

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
    fintables_udf_history: 100,
    tefasfon_funds: 90,
    tefas_direct_funds: 85,
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

function dominantPerformanceSource(points: FundPricePoint[]): string | null {
    const counts = new Map<string, number>();
    for (const point of points) {
        const source = String(point.source || '').trim().toLowerCase();
        if (source) counts.set(source, (counts.get(source) || 0) + 1);
    }
    let selected: string | null = null;
    for (const [source, count] of counts) {
        const selectedCount = selected ? (counts.get(selected) || 0) : 0;
        if (
            !selected
            || count > selectedCount
            || (count === selectedCount && performanceSourcePriority(source) > performanceSourcePriority(selected))
        ) {
            selected = source;
        }
    }
    return selected;
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
    const mergedHistorySource = dominantPerformanceSource(mergedPoints);
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
            ...(mergedHistorySource ? { history_source_used: mergedHistorySource } : {}),
            ...(mergedHistorySource === 'fintables_udf_history' ? { primary_source: 'fintables' } : {}),
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

const FUND_HISTORY_DIAGNOSTIC_PERIODS: Array<{ key: keyof FundPeriodReturns; label: string }> = [
    { key: '1w', label: '1H' },
    { key: '1m', label: '1A' },
    { key: '3m', label: '3A' },
    { key: '6m', label: '6A' },
    { key: 'ytd', label: 'YBB' },
    { key: '1y', label: '1Y' },
];

function diagnosticDateValue(value: string | null | undefined): number {
    if (!value) return NaN;
    const normalized = /^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00Z` : value;
    return Date.parse(normalized);
}

function diagnosticDateTime(value: string | null | undefined): string {
    if (!value) return '-';
    const timestamp = new Date(value);
    if (Number.isNaN(timestamp.getTime())) return value;
    return new Intl.DateTimeFormat('tr-TR', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZone: 'Europe/Istanbul',
    }).format(timestamp);
}

function diagnosticSourceLabel(source: string | null | undefined): string {
    const normalized = String(source || '').trim().toLowerCase();
    const labels: Record<string, string> = {
        fintables_udf_history: 'Fintables UDF geçmişi',
        fintables_yield_summary: 'Fintables getiri özeti',
        tefasfon_funds: 'TEFASFon',
        tefas_direct_funds: 'TEFAS direkt',
        tefasfon_returns: 'TEFASFon getirileri',
        legacy_json: 'eski yerel cache',
        sqlite: 'yerel SQLite cache',
    };
    return labels[normalized] || (source ? String(source) : 'bilinmiyor');
}

function diagnosticPointSourceCounts(points: FundPricePoint[]): string {
    const counts = new Map<string, number>();
    for (const point of points) {
        const source = String(point.source || '').trim().toLowerCase() || 'unknown';
        counts.set(source, (counts.get(source) || 0) + 1);
    }
    return [...counts.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([source, count]) => `${diagnosticSourceLabel(source)}: ${count}`)
        .join(' · ');
}

function diagnosticRange(start: string | null | undefined, end: string | null | undefined): string {
    if (!start && !end) return '-';
    return `${formatFundReportDate(start)} – ${formatFundReportDate(end)}`;
}

export type FundHistoryDiagnosticInput = {
    fundCode: string;
    performance: FundPerformanceResponse | null;
    points: FundPricePoint[];
    periodReturns?: FundPeriodReturns;
    historyJob?: FundHistoryJob | null;
    performanceLoading?: boolean;
    performanceError?: string | null;
    historyBackfillError?: string | null;
    yieldSummary?: FundYieldSummaryResponse | null;
    yieldLoading?: boolean;
    yieldError?: string | null;
};

/**
 * Build the detail-view diagnostic lines shown beside the fund history.
 * Keep this intentionally explicit: when a series is partial, the user needs
 * enough information to tell a young fund, a stale tail, a source fallback,
 * and a failed background job apart.
 */
export function buildFundHistoryDiagnosticLines(input: FundHistoryDiagnosticInput): string[] {
    const metadata = input.performance?.source_metadata;
    const points = validPerformancePoints(input.points);
    const job = input.historyJob || metadata?.history_job || null;
    const requestedStart = metadata?.requested_start_date || job?.requested_start || null;
    const requestedEnd = metadata?.requested_end_date || job?.requested_end || null;
    const availableStart = metadata?.available_start_date || metadata?.date_min || points[0]?.date || null;
    const availableEnd = metadata?.available_end_date || metadata?.date_max || points[points.length - 1]?.date || null;
    const coverage = metadata?.coverage_state || 'unknown';
    const source = metadata?.history_source_used || metadata?.primary_source || metadata?.source || input.performance?.source;
    const pointCount = metadata?.final_points_count ?? points.length;
    const missingPeriods = FUND_HISTORY_DIAGNOSTIC_PERIODS
        .filter((period) => input.periodReturns && input.periodReturns[period.key] == null)
        .map((period) => period.label);
    const hasDiagnosticSignal = Boolean(
        input.performanceLoading
        || input.performanceError
        || input.historyBackfillError
        || input.yieldLoading
        || input.yieldError
        || (metadata?.coverage_state && metadata.coverage_state !== 'complete')
        || metadata?.warnings?.length
        || metadata?.warning
        || metadata?.fallback_used
        || metadata?.fallback_reason
        || (job && !['succeeded', 'idle'].includes(job.status))
        || missingPeriods.length,
    );
    if (!hasDiagnosticSignal) return [];

    const lines: string[] = [
        `Fon geçmişi: ${input.fundCode} · API durumu: ${input.performance?.status || 'bilinmiyor'} · kapsama: ${coverage}`,
    ];

    if (requestedStart || requestedEnd) {
        lines.push(`İstenen aralık: ${diagnosticRange(requestedStart, requestedEnd)}`);
    }
    if (availableStart || availableEnd) {
        lines.push(`Mevcut aralık: ${diagnosticRange(availableStart, availableEnd)}`);
    }

    const sourceCounts = diagnosticPointSourceCounts(points);
    const sourceSuffix = sourceCounts ? ` · nokta kaynakları: ${sourceCounts}` : '';
    lines.push(
        `Veri: ${pointCount} nokta · çözünürlük: ${metadata?.resolution || metadata?.requested_resolution || 'bilinmiyor'} · kaynak: ${diagnosticSourceLabel(source)}${sourceSuffix}`,
    );

    if (availableEnd) {
        const requestedEndValue = diagnosticDateValue(requestedEnd);
        const availableEndValue = diagnosticDateValue(availableEnd);
        const gapBusinessDays = metadata?.coverage_gap_business_days;
        if (Number.isFinite(requestedEndValue) && Number.isFinite(availableEndValue)) {
            if (availableEndValue >= requestedEndValue) {
                lines.push(`Son kayıt: ${formatFundReportDate(availableEnd)} · hedef son: ${formatFundReportDate(requestedEnd)} · güncel uç mevcut.`);
            } else if (gapBusinessDays != null) {
                lines.push(`Son kayıt: ${formatFundReportDate(availableEnd)} · hedef son: ${formatFundReportDate(requestedEnd)} · gecikme: ${gapBusinessDays} iş günü.`);
            } else {
                lines.push(`Son kayıt: ${formatFundReportDate(availableEnd)} · hedef son: ${formatFundReportDate(requestedEnd)} · son uç geride.`);
            }
        } else {
            lines.push(`Son kayıt: ${formatFundReportDate(availableEnd)}.`);
        }
    }

    if (requestedStart && availableStart && diagnosticDateValue(availableStart) > diagnosticDateValue(requestedStart)) {
        lines.push(
            `Başlangıç durumu: Kaynakta istenen başlangıçtan önce kayıt yok; seri mevcut ilk kayıt olan ${formatFundReportDate(availableStart)} tarihinden başlıyor.`,
        );
    }
    if (coverage === 'complete') {
        lines.push('Kapsama durumu: İstenen aralığın başlangıç ve bitiş uçları mevcut.');
    } else if (coverage === 'range_incomplete') {
        lines.push('Kapsama durumu: Aralığın en az bir ucu eksik; grafikte mevcut kayıtlar gösteriliyor.');
    } else if (coverage === 'upgrading') {
        lines.push('Kapsama durumu: Günlük ayrıntılar arka planda hazırlanıyor; mevcut seri geçici olabilir.');
    }

    if (missingPeriods.length) {
        lines.push(`Getiri kartları: ${missingPeriods.join(', ')} için yeterli baz nokta bulunamadı; bu dönemler '-' gösterilir.`);
    }

    if (metadata?.source_policy) {
        lines.push(`Kaynak politikası: ${metadata.source_policy}.`);
    }
    if (metadata?.fallback_used || metadata?.fallback_reason) {
        lines.push(`Fallback: ${metadata.fallback_used ? 'kullanıldı' : 'kullanılmadı'}${metadata.fallback_reason ? ` · neden: ${metadata.fallback_reason}` : ''}.`);
    }

    if (job) {
        const jobRange = job.requested_start || job.requested_end
            ? ` · istek: ${diagnosticRange(job.requested_start, job.requested_end)}`
            : '';
        const effectiveRange = job.effective_start || job.effective_end
            ? ` · etkin: ${diagnosticRange(job.effective_start, job.effective_end)}`
            : '';
        const phase = job.phase != null ? ` · faz: ${job.phase}` : '';
        const finished = job.finished_at ? ` · tamamlandı: ${diagnosticDateTime(job.finished_at)}` : '';
        const pointCountDetail = job.fintables_point_count != null ? ` · Fintables nokta: ${job.fintables_point_count}` : '';
        lines.push(`Arka plan işi: ${job.status} · iş kimliği: ${job.job_id}${phase}${jobRange}${effectiveRange}${pointCountDetail}${finished}.`);
        if (job.error) lines.push(`Arka plan işi hatası: ${job.error}`);
    } else if (input.performanceLoading) {
        lines.push('Arka plan işi: performans geçmişi yükleniyor; henüz job bilgisi dönmedi.');
    } else {
        lines.push('Arka plan işi: bu yanıtta job kaydı yok.');
    }

    if (input.yieldSummary) {
        const periodCount = Object.keys(input.yieldSummary.periods || {}).length;
        lines.push(`Getiri özeti: ${diagnosticSourceLabel(input.yieldSummary.source || input.yieldSummary.source_metadata?.source)} · durum: ${input.yieldSummary.status} · dönem kaydı: ${periodCount}.`);
    } else if (input.yieldLoading) {
        lines.push('Getiri özeti: yükleniyor; performans grafiği bundan bağımsız gösteriliyor.');
    }

    const errors = [input.performanceError, input.historyBackfillError, input.yieldError].filter(Boolean);
    for (const error of errors) lines.push(`Hata: ${error}`);
    for (const warning of [...(metadata?.warnings || []), metadata?.warning].filter(Boolean)) {
        lines.push(`Kaynak uyarısı: ${warning}`);
    }

    return Array.from(new Set(lines));
}
