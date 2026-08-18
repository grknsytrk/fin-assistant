import type { KapQuarter, KapSnapshotResponse } from '../api/types';

export const FLOW_METRICS = new Set([
    'satis_gelirleri', 'brut_kar', 'favok', 'net_kar', 'faaliyet_nakit_akisi',
    'capex', 'serbest_nakit_akisi', 'faiz_gelirleri', 'faiz_giderleri',
    'net_ucret_komisyon_gelirleri', 'net_faaliyet_kari', 'esas_faaliyet_kari',
    'amortisman_itfa_gideri', 'prim_uretimi', 'alinan_net_primler', 'teknik_gelirler', 'teknik_denge',
]);

export type CompanyKind = 'generic' | 'bank' | 'insurance';

export const BANK_TICKERS = new Set([
    'AKBNK', 'ALBRK', 'GARAN', 'HALKB', 'ICBCT', 'ISCTR', 'QNBFB',
    'SKBNK', 'TSKB', 'VAKBN', 'YKBNK',
]);

export const INSURANCE_TICKERS = new Set(['AGESA', 'ANHYT', 'AKGRT', 'ANSGR', 'RAYSG', 'TURSG']);

export const INVERSE_METRICS = new Set([
    'net_borc', 'finansal_borclar', 'faiz_giderleri', 'kisa_vadeli_yukumlulukler',
    'beklenen_zarar_karsiliklari', 'teknik_karsiliklar', 'esas_faaliyetlerden_borclar',
]);

// Keep the established core order, then append additional non-empty metrics.
export const KAP_FINANCIAL_TABLE_KEYS = [
    'net_kar',
    'satis_gelirleri',
    'brut_kar',
    'favok',
    'ozkaynaklar',
    'donen_varliklar',
    'kisa_vadeli_yukumlulukler',
    'toplam_varliklar',
    'finansal_borclar',
    'net_borc',
    'faaliyet_nakit_akisi',
    'serbest_nakit_akisi',
    'esas_faaliyet_kari',
    'net_faaliyet_kari',
    'duran_varliklar',
    'nakit_ve_nakit_benzerleri',
    'finansal_varliklar_net',
    'faiz_gelirleri',
    'faiz_giderleri',
    'net_ucret_komisyon_gelirleri',
    'krediler',
    'mevduatlar',
    'beklenen_zarar_karsiliklari',
    'prim_uretimi',
    'alinan_net_primler',
    'teknik_gelirler',
    'teknik_denge',
    'teknik_karsiliklar',
    'esas_faaliyetlerden_alacaklar',
    'esas_faaliyetlerden_borclar',
    'finansal_varliklar_sigortacilik',
    'amortisman_itfa_gideri',
    'capex',
    'odenmis_sermaye',
    'cikarilmis_sermaye',
];

export function intSafe(value: unknown): number {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
}

export function _asNumber(value: number | null | undefined): number | null {
    if (typeof value !== 'number' || Number.isNaN(value) || !Number.isFinite(value)) {
        return null;
    }
    return value;
}

export function _periodToMonth(period: number): number {
    if (!Number.isFinite(period)) return 0;
    if (period >= 1 && period <= 4) return period * 3;
    return period;
}

export function _periodLabel(year: number, period: number, fallbackQuarter: string): string {
    if (Number.isFinite(year) && Number.isFinite(period) && period > 0) {
        return `${year}/${_periodToMonth(period)}`;
    }
    return fallbackQuarter || '-';
}

export function _quarterSortValue(row: KapQuarter): number {
    return intSafe(row.year) * 100 + _periodToMonth(intSafe(row.period));
}

export function _normText(value: unknown): string {
    return String(value || '')
        .toUpperCase()
        .replaceAll('İ', 'I').replaceAll('Ş', 'S').replaceAll('Ğ', 'G')
        .replaceAll('Ü', 'U').replaceAll('Ö', 'O').replaceAll('Ç', 'C')
        .replaceAll('ı', 'I').replaceAll('ş', 'S').replaceAll('ğ', 'G')
        .replaceAll('ü', 'U').replaceAll('ö', 'O').replaceAll('ç', 'C');
}

export function classifyKapCompanyKind(
    snapshot: Partial<Pick<KapSnapshotResponse, 'company_kind' | 'company' | 'stock_code' | 'company_title'>>,
): CompanyKind {
    if (snapshot.company_kind === 'bank' || snapshot.company_kind === 'insurance' || snapshot.company_kind === 'generic') {
        return snapshot.company_kind;
    }

    const stockCodeNorm = _normText(snapshot.stock_code || snapshot.company);
    const companyTitleNorm = _normText(snapshot.company_title);
    const titleTokens = new Set(companyTitleNorm.split(/\s+/).filter(Boolean));
    if (BANK_TICKERS.has(stockCodeNorm) || titleTokens.has('BANK') || titleTokens.has('BANKASI')) {
        return 'bank';
    }
    if (INSURANCE_TICKERS.has(stockCodeNorm) || companyTitleNorm.includes('SIGORTA') || companyTitleNorm.includes('EMEKLILIK')) {
        return 'insurance';
    }
    return 'generic';
}

export function _formatRatio(value: number, suffix = '%'): string {
    return `${value.toFixed(2)}${suffix}`;
}

export function _formatMetric(value: number, currency: string): string {
    const absValue = Math.abs(value);
    const sign = value < 0 ? '-' : '';
    if (absValue >= 1_000_000_000) {
        return `${sign}${(absValue / 1_000_000_000).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} Milyar ${currency}`;
    }
    if (absValue >= 1_000_000) {
        return `${sign}${(absValue / 1_000_000).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} Milyon ${currency}`;
    }
    if (absValue >= 1_000) {
        return `${sign}${(absValue / 1_000).toLocaleString('tr-TR', { minimumFractionDigits: 1, maximumFractionDigits: 1 })} Bin ${currency}`;
    }
    return `${sign}${absValue.toLocaleString('tr-TR', { maximumFractionDigits: 0 })} ${currency}`;
}

export function _rawMetricValue(row: KapQuarter | null, metricKey: string, bucket: 'metrics' | 'metrics_quarterly' | 'metrics_ytd'): number | null {
    if (!row) return null;
    return _asNumber(row[bucket]?.[metricKey]?.value);
}

/** YTD farkıyla çeyrek akışı: ölçeklenmiş `metrics_ytd` karışmasın diye önce ham KAP değeri. */
export function _rawYtdForQuarterFlow(row: KapQuarter | null, metricKey: string): number | null {
    if (!row) return null;
    const fromOriginal = _asNumber(row.metrics_ytd_original?.[metricKey]?.value);
    if (fromOriginal !== null) return fromOriginal;
    return _asNumber(row.metrics_ytd?.[metricKey]?.value);
}

/** Çeyrek akışı: normalize_snapshot çarpanı eski çeyrekleri şişirir; grafik ham KAP kullanmalı. */
function _rawPointForFlow(row: KapQuarter | null, metricKey: string): number | null {
    if (!row) return null;
    const fromOriginal = _asNumber(row.metrics_original?.[metricKey]?.value);
    if (fromOriginal !== null) return fromOriginal;
    return _asNumber(row.metrics?.[metricKey]?.value);
}

function _rawQuarterlyForFlow(row: KapQuarter | null, metricKey: string): number | null {
    if (!row) return null;
    const fromOriginal = _asNumber(row.metrics_quarterly_original?.[metricKey]?.value);
    if (fromOriginal !== null) return fromOriginal;
    return _asNumber(row.metrics_quarterly?.[metricKey]?.value);
}

export function _resolveMetricValue(rows: KapQuarter[], idx: number, metricKey: string, asQuarterlyFlow: boolean): number | null {
    const row = rows[idx];
    if (!row) return null;

    const pointValue = _rawMetricValue(row, metricKey, 'metrics');
    const quarterlyValue = _rawMetricValue(row, metricKey, 'metrics_quarterly');
    const ytdValue = _rawMetricValue(row, metricKey, 'metrics_ytd');

    if (asQuarterlyFlow && FLOW_METRICS.has(metricKey)) {
        const pointForFlow = _rawPointForFlow(row, metricKey);
        const quarterlyForFlow = _rawQuarterlyForFlow(row, metricKey);
        const ytdForFlow = _rawYtdForQuarterFlow(row, metricKey);
        if (ytdForFlow !== null) {
            let prevYtd: number | null = null;
            const year = intSafe(row.year);
            const period = intSafe(row.period);
            for (let prevIdx = idx - 1; prevIdx >= 0; prevIdx -= 1) {
                const prevRow = rows[prevIdx];
                if (intSafe(prevRow.year) !== year) break;
                const prevCandidate = _rawYtdForQuarterFlow(prevRow, metricKey);
                if (prevCandidate !== null) {
                    prevYtd = prevCandidate;
                    break;
                }
            }
            if (period > 1) {
                if (prevYtd !== null) return ytdForFlow - prevYtd;
                // Aynı yılda önceki çeyrek yok (ör. ilk veri Q3): YTD farkı hesaplanamaz.
                // metrics ile metrics_ytd aynı çıktıysa genelde tabloda tek sütun okunmuş (9 aylık vb.) — bunu çeyrek sanma.
                if (quarterlyForFlow !== null && ytdForFlow !== null && quarterlyForFlow === ytdForFlow) {
                    return null;
                }
                if (quarterlyForFlow !== null) return quarterlyForFlow;
                if (pointForFlow !== null) return pointForFlow;
                return null;
            }
            return ytdForFlow;
        }
        if (quarterlyForFlow !== null) return quarterlyForFlow;
        if (pointForFlow !== null) return pointForFlow;
        return null;
    }

    if (pointValue !== null) return pointValue;
    if (quarterlyValue !== null) return quarterlyValue;
    if (ytdValue !== null) return ytdValue;
    return null;
}

export function _resolveMetricValueByPriority(row: KapQuarter | null, metricKey: string, priority: Array<'metrics' | 'metrics_quarterly' | 'metrics_ytd'>): number | null {
    if (!row) return null;
    for (const bucket of priority) {
        const value = _rawMetricValue(row, metricKey, bucket);
        if (value !== null) return value;
    }
    return null;
}

export function _resolveMetricDisplayByPriority(row: KapQuarter | null, metricKey: string, priority: Array<'metrics' | 'metrics_quarterly' | 'metrics_ytd'>): string {
    if (!row) return '-';
    const value = _resolveMetricValueByPriority(row, metricKey, priority);
    if (value === null) return '-';
    return _formatMetric(value, row.currency || 'TL');
}

export function _resolveMetricDisplay(rows: KapQuarter[], idx: number, metricKey: string, asQuarterlyFlow: boolean): string {
    const row = rows[idx];
    if (!row) return '-';
    const value = _resolveMetricValue(rows, idx, metricKey, asQuarterlyFlow);
    if (value === null) return '-';
    return _formatMetric(value, row.currency || 'TL');
}

export function _calcPctChange(current: number | null, base: number | null): number | null {
    if (current === null || base === null || base === 0) return null;
    return ((current - base) / Math.abs(base)) * 100;
}

export function _pctClass(value: number | null, metricKey?: string): string {
    if (value === null) return 'pct-neutral';
    const invert = metricKey ? INVERSE_METRICS.has(metricKey) : false;
    if (value > 0) return invert ? 'pct-negative' : 'pct-positive';
    if (value < 0) return invert ? 'pct-positive' : 'pct-negative';
    return 'pct-neutral';
}

export function _pctText(value: number | null): string {
    if (value === null || Number.isNaN(value)) return '-';
    return `% ${Math.round(value)}`;
}
