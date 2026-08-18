import type { KapSnapshotResponse, KapQuarter } from '../api/types';
import type { SeriesPoint } from './chartBuilders';
import {
    _buildMetricSeries,
    _buildCustomSeries,
    _takeLastSeries,
    _buildRatioSeries,
    _buildAnnualizedRoeSeries,
    CHART_WINDOW_QUARTERS,
} from './chartBuilders';
import {
    _resolveMetricValue,
    _resolveMetricDisplay,
    _resolveMetricYtdValue,
    _resolveMetricYtdDisplay,
    _calcPctChange,
    _pctText,
    intSafe,
    _periodLabel,
    FLOW_METRICS,
    classifyKapCompanyKind,
    type CompanyKind,
} from './formatters';

export type OverviewSummaryConfigRow = {
    key: string;
    label: string;
};

export type OverviewAiSummaryRow = {
    key: string;
    label: string;
    current_period: string;
    current_value: number | null;
    current_display: string;
    base_period: string;
    base_value: number | null;
    base_display: string;
    pct_change: number | null;
    pct_display: string;
};

export type OverviewAiChart = {
    title: string;
    kind: 'bar' | 'line';
    series: Array<{
        label: string;
        value: number;
        display: string;
    }>;
};

export type OverviewAiPayload = {
    income_summary: OverviewAiSummaryRow[];
    balance_summary: OverviewAiSummaryRow[];
    charts: OverviewAiChart[];
};

export type OverviewAiHistoryQuarter = {
    label: string;
    year: number;
    period: number;
    metrics: Record<string, number | null>;
    ratios: Record<string, number | null>;
};

export type OverviewAiHistoryContext = {
    company_kind: 'generic' | 'bank' | 'insurance';
    quarters: OverviewAiHistoryQuarter[];
};

export type OverviewChartGroup = {
    title: string;
    kind: 'bar' | 'line';
    series: SeriesPoint[];
};

export const DEFAULT_INCOME_SUMMARY_ROWS: OverviewSummaryConfigRow[] = [
    { key: 'satis_gelirleri', label: 'Satışlar' },
    { key: 'brut_kar', label: 'Brüt Kar' },
    { key: 'esas_faaliyet_kari', label: 'Esas Faaliyet Karı' },
    { key: 'favok', label: 'FAVÖK' },
    { key: 'net_faaliyet_kari', label: 'Net Faaliyet Karı' },
    { key: 'net_kar', label: 'Net Dönem Karı' },
];

export const DEFAULT_BALANCE_SUMMARY_ROWS: OverviewSummaryConfigRow[] = [
    { key: 'donen_varliklar', label: 'Dönen Varlıklar' },
    { key: 'duran_varliklar', label: 'Duran Varlıklar' },
    { key: 'toplam_varliklar', label: 'Toplam Varlıklar' },
    { key: 'nakit_ve_nakit_benzerleri', label: 'Nakit ve Nakit Benzerleri' },
    { key: 'finansal_borclar', label: 'Finansal Borçlar' },
    { key: 'net_borc', label: 'Net Borç' },
    { key: 'ozkaynaklar', label: 'Özkaynaklar' },
];

export const BANK_INCOME_SUMMARY_ROWS: OverviewSummaryConfigRow[] = [
    { key: 'faiz_gelirleri', label: 'Faiz Gelirleri' },
    { key: 'faiz_giderleri', label: 'Faiz Giderleri (-)' },
    { key: 'net_ucret_komisyon_gelirleri', label: 'Net Ücret Komisyon Gelirleri' },
    { key: 'net_faaliyet_kari', label: 'Net Faaliyet Karı (Zararı)' },
    { key: 'net_kar', label: 'Net Dönem Karı' },
];

export const BANK_BALANCE_SUMMARY_ROWS: OverviewSummaryConfigRow[] = [
    { key: 'finansal_varliklar_net', label: 'Finansal Varlıklar (Net)' },
    { key: 'krediler', label: 'Krediler' },
    { key: 'mevduatlar', label: 'Mevduatlar' },
    { key: 'beklenen_zarar_karsiliklari', label: 'Beklenen Zarar Karşılıkları' },
    { key: 'ozkaynaklar', label: 'Özkaynaklar' },
];

export const INSURANCE_INCOME_SUMMARY_ROWS: OverviewSummaryConfigRow[] = [
    { key: 'prim_uretimi', label: 'Prim Üretimi' },
    { key: 'alinan_net_primler', label: 'Alınan Net Primler' },
    { key: 'teknik_gelirler', label: 'Teknik Gelirler' },
    { key: 'teknik_denge', label: 'Teknik Denge' },
    { key: 'net_kar', label: 'Net Dönem Karı' },
];

export const INSURANCE_BALANCE_SUMMARY_ROWS: OverviewSummaryConfigRow[] = [
    { key: 'nakit_benzeri_finansal_varliklar', label: 'Nakit Benzeri Finansal Varlıklar' },
    { key: 'esas_faaliyetlerden_alacaklar', label: 'Esas Faaliyetlerden Alacaklar' },
    { key: 'teknik_karsiliklar', label: 'Teknik Karşılıklar' },
    { key: 'esas_faaliyetlerden_borclar', label: 'Esas Faaliyetlerden Borçlar' },
    { key: 'ozkaynaklar', label: 'Özkaynaklar' },
];

function companyKind(
    snapshot: KapSnapshotResponse,
): { isBankLike: boolean; isInsuranceLike: boolean; kind: CompanyKind } {
    const kind = classifyKapCompanyKind(snapshot);
    const isBankLike = kind === 'bank';
    const isInsuranceLike = kind === 'insurance';

    return {
        isBankLike,
        isInsuranceLike,
        kind,
    };
}

export function getOverviewSummaryRows(snapshot: KapSnapshotResponse) {
    const { isBankLike, isInsuranceLike } = companyKind(snapshot);
    return {
        incomeSummaryRows: isBankLike
            ? BANK_INCOME_SUMMARY_ROWS
            : isInsuranceLike
                ? INSURANCE_INCOME_SUMMARY_ROWS
                : DEFAULT_INCOME_SUMMARY_ROWS,
        balanceSummaryRows: isBankLike
            ? BANK_BALANCE_SUMMARY_ROWS
            : isInsuranceLike
                ? INSURANCE_BALANCE_SUMMARY_ROWS
                : DEFAULT_BALANCE_SUMMARY_ROWS,
    };
}

export function buildOverviewChartGroups(snapshot: KapSnapshotResponse, quarters: KapQuarter[], windowQuarters?: number): OverviewChartGroup[] {
    if (!quarters.length) return [];
    const window = windowQuarters ?? CHART_WINDOW_QUARTERS;

    const { isBankLike, isInsuranceLike } = companyKind(snapshot);
    let barCharts: OverviewChartGroup[];

    if (isBankLike) {
        barCharts = [
            {
                title: 'Çeyreklik Net Faiz Geliri veya Gideri',
                kind: 'bar' as const,
                series: _takeLastSeries(_buildCustomSeries(quarters, 'net_faiz_geliri_gideri', (rows, idx) => {
                    const gelir = _resolveMetricValue(rows, idx, 'faiz_gelirleri', true);
                    const gider = _resolveMetricValue(rows, idx, 'faiz_giderleri', true);
                    if (gelir === null && gider === null) return null;
                    return (gelir ?? 0) + (gider ?? 0);
                }, (val, curr) => `${val.toLocaleString('tr-TR')} ${curr}`), window),
            },
            {
                title: 'Çeyreklik Net Kar',
                kind: 'bar' as const,
                series: _takeLastSeries(_buildMetricSeries(quarters, 'net_kar', true), window),
            },
            {
                title: 'Krediler',
                kind: 'bar' as const,
                series: _takeLastSeries(_buildMetricSeries(quarters, 'krediler', false), window),
            },
        ].filter((item) => item.series.length > 0);
    } else if (isInsuranceLike) {
        barCharts = [
            {
                title: 'Çeyreklik Prim Üretimi',
                kind: 'bar' as const,
                series: _takeLastSeries(_buildMetricSeries(quarters, 'prim_uretimi', true), window),
            },
            {
                title: 'Çeyreklik Teknik Denge',
                kind: 'bar' as const,
                series: _takeLastSeries(_buildMetricSeries(quarters, 'teknik_denge', true), window),
            },
            {
                title: 'Çeyreklik Net Kar',
                kind: 'bar' as const,
                series: _takeLastSeries(_buildMetricSeries(quarters, 'net_kar', true), window),
            },
        ].filter((item) => item.series.length > 0);
    } else {
        barCharts = [
            {
                title: 'Çeyreklik Satışlar',
                kind: 'bar' as const,
                series: _takeLastSeries(_buildMetricSeries(quarters, 'satis_gelirleri', true), window),
            },
            {
                title: 'Çeyreklik FAVÖK',
                kind: 'bar' as const,
                series: _takeLastSeries(_buildMetricSeries(quarters, 'favok', true), window),
            },
            {
                title: 'Çeyreklik Net Kâr',
                kind: 'bar' as const,
                series: _takeLastSeries(_buildMetricSeries(quarters, 'net_kar', true), window),
            },
            {
                title: 'Çeyreklik Serbest Nakit Akışı',
                kind: 'bar' as const,
                series: _takeLastSeries(_buildMetricSeries(quarters, 'serbest_nakit_akisi', true), window),
            },
        ].filter((item) => item.series.length > 0);
    }

    const ratioSeries = {
        brutKarMarji: _buildRatioSeries(quarters, 'brut_kar', 'satis_gelirleri', true, true, 100, v => `${v.toFixed(2)}%`),
        favokMarji: _buildRatioSeries(quarters, 'favok', 'satis_gelirleri', true, true, 100, v => `${v.toFixed(2)}%`),
        netKarMarji: _buildRatioSeries(quarters, 'net_kar', 'satis_gelirleri', true, true, 100, v => `${v.toFixed(2)}%`),
        cariOran: _buildRatioSeries(quarters, 'donen_varliklar', 'kisa_vadeli_yukumlulukler', false, false, 1, v => `${v.toFixed(2)}x`),
        roe: _buildAnnualizedRoeSeries(quarters, 'net_kar', 'ozkaynaklar', v => `${v.toFixed(2)}%`),
    };

    const lineCharts: OverviewChartGroup[] = [
        { title: 'Brüt Kâr Marjı', kind: 'line' as const, series: _takeLastSeries(ratioSeries.brutKarMarji, window) },
        { title: 'FAVÖK Marjı', kind: 'line' as const, series: _takeLastSeries(ratioSeries.favokMarji, window) },
        { title: 'Net Kâr Marjı', kind: 'line' as const, series: _takeLastSeries(ratioSeries.netKarMarji, window) },
        { title: 'Cari Oran', kind: 'line' as const, series: _takeLastSeries(ratioSeries.cariOran, window) },
        { title: 'Özkaynak Karlılığı (ROE)', kind: 'line' as const, series: _takeLastSeries(ratioSeries.roe, window) },
    ].filter((item) => item.series.length > 0);

    return [...barCharts, ...lineCharts];
}

export function latestOverviewPeriod(quarters: KapQuarter[]): string {
    const latest = quarters.length ? quarters[quarters.length - 1] : null;
    if (!latest) return '';
    return _periodLabel(intSafe(latest.year), intSafe(latest.period), latest.quarter);
}

function safeDivide(numerator: number | null, denominator: number | null, scale = 1): number | null {
    if (numerator === null || denominator === null || denominator === 0) return null;
    return (numerator / denominator) * scale;
}

function metricValueAt(quarters: KapQuarter[], idx: number, metricKey: string, asQuarterlyFlow = FLOW_METRICS.has(metricKey)) {
    return _resolveMetricValue(quarters, idx, metricKey, asQuarterlyFlow);
}

function ratioMapForQuarter(
    companyKindValue: OverviewAiHistoryContext['company_kind'],
    quarters: KapQuarter[],
    idx: number,
): Record<string, number | null> {
    const netKar = metricValueAt(quarters, idx, 'net_kar', true);
    const ozkaynak = metricValueAt(quarters, idx, 'ozkaynaklar', false);

    if (companyKindValue === 'bank') {
        const krediler = metricValueAt(quarters, idx, 'krediler', false);
        const mevduatlar = metricValueAt(quarters, idx, 'mevduatlar', false);
        const karsiliklar = metricValueAt(quarters, idx, 'beklenen_zarar_karsiliklari', false);
        return {
            roe: safeDivide(netKar, ozkaynak, 100),
            kredi_mevduat_orani: safeDivide(krediler, mevduatlar),
            karsilik_kredi_orani: safeDivide(karsiliklar, krediler),
        };
    }

    if (companyKindValue === 'insurance') {
        const teknikDenge = metricValueAt(quarters, idx, 'teknik_denge', true);
        const alinanNetPrimler = metricValueAt(quarters, idx, 'alinan_net_primler', true);
        const primUretimi = metricValueAt(quarters, idx, 'prim_uretimi', true);
        const nakitBenzeri = metricValueAt(quarters, idx, 'nakit_benzeri_finansal_varliklar', false);
        const teknikKarsiliklar = metricValueAt(quarters, idx, 'teknik_karsiliklar', false);
        const esasFaaliyetBorclar = metricValueAt(quarters, idx, 'esas_faaliyetlerden_borclar', false);
        return {
            roe: safeDivide(netKar, ozkaynak, 100),
            teknik_denge_marji: safeDivide(teknikDenge, alinanNetPrimler ?? primUretimi, 100),
            nakit_karsilik_orani: safeDivide(nakitBenzeri, teknikKarsiliklar),
            ozkaynak_karsilik_orani: safeDivide(ozkaynak, teknikKarsiliklar),
            borc_ozkaynak_orani: safeDivide(esasFaaliyetBorclar, ozkaynak),
        };
    }

    const satislar = metricValueAt(quarters, idx, 'satis_gelirleri', true);
    const favok = metricValueAt(quarters, idx, 'favok', true);
    const donenVarliklar = metricValueAt(quarters, idx, 'donen_varliklar', false);
    const kisaVadeliYukumlulukler = metricValueAt(quarters, idx, 'kisa_vadeli_yukumlulukler', false);
    const netBorc = metricValueAt(quarters, idx, 'net_borc', false);
    const serbestNakitAkisi = metricValueAt(quarters, idx, 'serbest_nakit_akisi', true);
    const faaliyetNakitAkisi = metricValueAt(quarters, idx, 'faaliyet_nakit_akisi', true);
    return {
        favok_marji: safeDivide(favok, satislar, 100),
        net_kar_marji: safeDivide(netKar, satislar, 100),
        roe: safeDivide(netKar, ozkaynak, 100),
        cari_oran: safeDivide(donenVarliklar, kisaVadeliYukumlulukler),
        net_borc_ozkaynak: safeDivide(netBorc, ozkaynak),
        nakit_donusum: safeDivide(serbestNakitAkisi ?? faaliyetNakitAkisi, netKar),
    };
}

export function buildOverviewHistoryContext(
    snapshot: KapSnapshotResponse,
    quarters: KapQuarter[],
): OverviewAiHistoryContext {
    const { kind } = companyKind(snapshot);
    const historyRows: OverviewAiHistoryQuarter[] = [];
    const startIdx = Math.max(0, quarters.length - 12);

    const metricsByKind: Record<OverviewAiHistoryContext['company_kind'], string[]> = {
        generic: [
            'satis_gelirleri',
            'favok',
            'net_kar',
            'faaliyet_nakit_akisi',
            'serbest_nakit_akisi',
            'ozkaynaklar',
        ],
        bank: [
            'net_ucret_komisyon_gelirleri',
            'net_faaliyet_kari',
            'net_kar',
            'krediler',
            'mevduatlar',
            'beklenen_zarar_karsiliklari',
            'ozkaynaklar',
            'finansal_varliklar_net',
        ],
        insurance: [
            'prim_uretimi',
            'alinan_net_primler',
            'teknik_gelirler',
            'teknik_denge',
            'net_kar',
            'ozkaynaklar',
            'nakit_benzeri_finansal_varliklar',
            'teknik_karsiliklar',
            'esas_faaliyetlerden_borclar',
        ],
    };

    for (let idx = startIdx; idx < quarters.length; idx += 1) {
        const row = quarters[idx];
        const metrics = Object.fromEntries(
            metricsByKind[kind].map((metricKey) => [metricKey, metricValueAt(quarters, idx, metricKey)]),
        ) as Record<string, number | null>;
        historyRows.push({
            label: _periodLabel(intSafe(row.year), intSafe(row.period), row.quarter),
            year: intSafe(row.year),
            period: intSafe(row.period),
            metrics,
            ratios: ratioMapForQuarter(kind, quarters, idx),
        });
    }

    return {
        company_kind: kind,
        quarters: historyRows,
    };
}

export function buildOverviewAiPayload(snapshot: KapSnapshotResponse, quarters: KapQuarter[]): OverviewAiPayload {
    const latestQuarterIdx = quarters.length ? quarters.length - 1 : -1;
    const latestQuarter = latestQuarterIdx >= 0 ? quarters[latestQuarterIdx] : null;
    const prevQuarterIdx = latestQuarterIdx > 0 ? latestQuarterIdx - 1 : -1;
    const prevQuarter = prevQuarterIdx >= 0 ? quarters[prevQuarterIdx] : null;
    const prevYearSameQuarterIdx = latestQuarter
        ? quarters.findIndex(
            (q) =>
                intSafe(q.year) === intSafe(latestQuarter.year) - 1 &&
                intSafe(q.period) === intSafe(latestQuarter.period),
        )
        : -1;
    const prevYearSameQuarter = prevYearSameQuarterIdx >= 0 ? quarters[prevYearSameQuarterIdx] : null;
    const { incomeSummaryRows, balanceSummaryRows } = getOverviewSummaryRows(snapshot);
    const currentPeriod = latestQuarter ? _periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter) : '';
    const incomeBasePeriod = prevYearSameQuarter
        ? _periodLabel(intSafe(prevYearSameQuarter.year), intSafe(prevYearSameQuarter.period), prevYearSameQuarter.quarter)
        : '';
    const balanceBasePeriod = prevQuarter
        ? _periodLabel(intSafe(prevQuarter.year), intSafe(prevQuarter.period), prevQuarter.quarter)
        : '';

    const incomeSummary = latestQuarter
        ? incomeSummaryRows.map((row): OverviewAiSummaryRow | null => {
            const currentValue = _resolveMetricYtdValue(latestQuarter, row.key);
            const baseValue = prevYearSameQuarterIdx >= 0
                ? _resolveMetricYtdValue(prevYearSameQuarter, row.key)
                : null;
            if (currentValue === null && baseValue === null) return null;
            const pct = _calcPctChange(currentValue, baseValue);
            return {
                key: row.key,
                label: row.label,
                current_period: currentPeriod,
                current_value: currentValue,
                current_display: _resolveMetricYtdDisplay(latestQuarter, row.key),
                base_period: incomeBasePeriod,
                base_value: baseValue,
                base_display: prevYearSameQuarterIdx >= 0
                    ? _resolveMetricYtdDisplay(prevYearSameQuarter, row.key)
                    : '',
                pct_change: pct,
                pct_display: _pctText(pct),
            };
        }).filter((row): row is OverviewAiSummaryRow => row !== null)
        : [];

    const balanceSummary = latestQuarter
        ? balanceSummaryRows.map((row): OverviewAiSummaryRow | null => {
            const currentValue = _resolveMetricValue(quarters, latestQuarterIdx, row.key, false);
            const baseValue = prevQuarterIdx >= 0 ? _resolveMetricValue(quarters, prevQuarterIdx, row.key, false) : null;
            if (currentValue === null && baseValue === null) return null;
            const pct = _calcPctChange(currentValue, baseValue);
            return {
                key: row.key,
                label: row.label,
                current_period: currentPeriod,
                current_value: currentValue,
                current_display: _resolveMetricDisplay(quarters, latestQuarterIdx, row.key, false),
                base_period: balanceBasePeriod,
                base_value: baseValue,
                base_display: prevQuarterIdx >= 0 ? _resolveMetricDisplay(quarters, prevQuarterIdx, row.key, false) : '',
                pct_change: pct,
                pct_display: _pctText(pct),
            };
        }).filter((row): row is OverviewAiSummaryRow => row !== null)
        : [];

    return {
        income_summary: incomeSummary.slice(0, 8),
        balance_summary: balanceSummary.slice(0, 8),
        charts: buildOverviewChartGroups(snapshot, quarters).slice(0, 9).map((chart) => ({
            title: chart.title,
            kind: chart.kind,
            series: chart.series.slice(-10).map((point) => ({
                label: point.label,
                value: point.value,
                display: point.display,
            })),
        })),
    };
}

export function stableStringify(value: unknown): string {
    if (value === null || typeof value !== 'object') {
        return JSON.stringify(value);
    }
    if (Array.isArray(value)) {
        return `[${value.map((item) => stableStringify(item)).join(',')}]`;
    }
    const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
    return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableStringify(item)}`).join(',')}}`;
}

export function overviewPayloadHash(payload: unknown): string {
    const serialized = stableStringify(payload);
    let hash = 2166136261;
    for (let idx = 0; idx < serialized.length; idx += 1) {
        hash ^= serialized.charCodeAt(idx);
        hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(16).padStart(8, '0');
}
