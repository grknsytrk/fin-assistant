import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { RefreshCw, ChevronUp, ChevronDown } from 'lucide-react';
import { apiClient } from '../api/client';
import type { KapQuarter, KapSnapshotResponse } from '../api/types';
import {
    classifyKapCompanyKind,
    KAP_FINANCIAL_TABLE_KEYS,
    _resolveMetricYtdValue,
    _resolveMetricYtdDisplay,
} from '../utils/formatters';
import './KapPage.css';

/* ── Yahoo Finance Price Ticker ── */
type PriceData = {
    ok: boolean;
    symbol: string;
    price: number | null;
    change: number | null;
    change_pct: number | null;
    currency: string;
    market_state: string;
    error?: string;
};

function PriceTicker({ symbol }: { symbol: string }) {
    const [data, setData] = useState<PriceData | null>(null);

    useEffect(() => {
        if (!symbol) return;
        let cancelled = false;
        apiClient.kapPrice(symbol).then((d) => {
            if (!cancelled) setData(d as PriceData);
        }).catch(() => {
            if (!cancelled) setData(null);
        });
        return () => { cancelled = true; };
    }, [symbol]);

    if (!data || !data.ok || data.price == null) return null;

    const isUp = (data.change ?? 0) >= 0;
    const colorClass = isUp ? 'price-up' : 'price-down';
    const marketLabel =
        data.market_state === 'REGULAR' ? 'Açık' :
            data.market_state === 'PRE' ? 'Açılış Öncesi' :
                data.market_state === 'POST' ? 'Kapanış Sonrası' :
                    'Kapalı';

    return (
        <div className="kap-price-ticker">
            <div className="kap-price-symbol">{data.symbol}</div>
            <div className="kap-price-value">
                ₺{data.price.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <div className={`kap-price-change ${colorClass}`}>
                <span className="kap-price-arrow">
                    {isUp ? <ChevronUp size={16} strokeWidth={3} /> : <ChevronDown size={16} strokeWidth={3} />}
                </span>
                {data.change != null && (
                    <span>{Math.abs(data.change).toFixed(2)}</span>
                )}
                {data.change_pct != null && (
                    <span>({Math.abs(data.change_pct).toFixed(2)}%)</span>
                )}
            </div>
            <div className="kap-price-market">{marketLabel}</div>
        </div>
    );
}

/* ── Valuation Multiples ── */
function MultiplesRow({ snapshot, quarters }: { snapshot: KapSnapshotResponse; quarters: KapQuarter[] }) {
    const valuation = snapshot.valuation;

    const groupedMultiples = useMemo(() => {
        if (!quarters.length) return null;

        const latest = quarters[quarters.length - 1];
        const last4 = quarters.slice(-4);
        const ttmNetKarFallback = last4.reduce((sum, q) => {
            const v = _resolveMetricValueByPriority(q, 'net_kar', ['metrics_quarterly', 'metrics']);
            return v !== null ? sum + v : sum;
        }, 0);
        const ttmFavokFallback = last4.reduce((sum, q) => {
            const v = _resolveMetricValueByPriority(q, 'favok', ['metrics_quarterly', 'metrics']);
            return v !== null ? sum + v : sum;
        }, 0);
        const ttmSatis = last4.reduce((sum, q) => {
            const v = _resolveMetricValueByPriority(q, 'satis_gelirleri', ['metrics_quarterly', 'metrics']);
            return v !== null ? sum + v : sum;
        }, 0);

        const ozkaynaklar = _resolveMetricValueByPriority(latest, 'ozkaynaklar', ['metrics', 'metrics_ytd']);
        const netBorc = _resolveMetricValueByPriority(latest, 'net_borc', ['metrics', 'metrics_ytd']);
        const donenVarliklar = _resolveMetricValueByPriority(latest, 'donen_varliklar', ['metrics', 'metrics_ytd']);
        const kisaVadeli = _resolveMetricValueByPriority(latest, 'kisa_vadeli_yukumlulukler', ['metrics', 'metrics_ytd']);

        const ttmNetKar = valuation?.ttm_net_kar ?? (ttmNetKarFallback !== 0 ? ttmNetKarFallback : null);
        const ttmFavok = valuation?.ttm_favok ?? (ttmFavokFallback !== 0 ? ttmFavokFallback : null);

        const marketItems: { label: string; value: string; isNeg?: boolean }[] = [];
        const multipleItems: { label: string; value: string; isNeg?: boolean }[] = [];
        const profitabilityItems: { label: string; value: string; isNeg?: boolean }[] = [];
        const balanceItems: { label: string; value: string; isNeg?: boolean }[] = [];

        if (valuation?.price != null) {
            marketItems.push({
                label: 'Fiyat',
                value: `₺${valuation.price.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
            });
        }
        if (valuation?.market_cap != null) {
            marketItems.push({ label: 'Piyasa Değeri', value: _formatMetric(valuation.market_cap, 'TL') });
        }
        multipleItems.push({
            label: 'F/K',
            value: valuation?.fk != null ? _formatRatio(valuation.fk, 'x') : '-',
            isNeg: valuation?.fk != null ? valuation.fk < 0 : false,
        });
        multipleItems.push({
            label: 'PD/DD',
            value: valuation?.pd_dd != null ? _formatRatio(valuation.pd_dd, 'x') : '-',
            isNeg: valuation?.pd_dd != null ? valuation.pd_dd < 0 : false,
        });
        multipleItems.push({
            label: 'FD/FAVÖK',
            value: valuation?.fd_favok != null ? _formatRatio(valuation.fd_favok, 'x') : '-',
            isNeg: valuation?.fd_favok != null ? valuation.fd_favok < 0 : false,
        });

        if (ttmSatis > 0 && ttmNetKar !== null) {
            const margin = (ttmNetKar / ttmSatis) * 100;
            profitabilityItems.push({ label: 'Net Kâr Marjı', value: _formatRatio(margin), isNeg: margin < 0 });
        }
        if (ttmSatis > 0 && ttmFavok !== null) {
            const margin = (ttmFavok / ttmSatis) * 100;
            profitabilityItems.push({ label: 'FAVÖK Marjı', value: _formatRatio(margin), isNeg: margin < 0 });
        }
        if (ozkaynaklar && ozkaynaklar > 0 && ttmNetKar !== null) {
            const roe = (ttmNetKar / ozkaynaklar) * 100;
            profitabilityItems.push({ label: 'ROE', value: _formatRatio(roe), isNeg: roe < 0 });
        }
        if (netBorc !== null && ozkaynaklar && ozkaynaklar > 0) {
            const ratio = netBorc / ozkaynaklar;
            balanceItems.push({ label: 'Borç/Özkaynak', value: _formatRatio(ratio, 'x'), isNeg: ratio > 1 });
        }
        if (donenVarliklar && kisaVadeli && kisaVadeli !== 0) {
            const cari = donenVarliklar / kisaVadeli;
            balanceItems.push({ label: 'Cari Oran', value: _formatRatio(cari, 'x'), isNeg: cari < 1 });
        }

        const groups = [
            { title: 'Piyasa', items: marketItems },
            { title: 'Çarpanlar', items: multipleItems },
            { title: 'Kârlılık', items: profitabilityItems },
            { title: 'Finansal Sağlık', items: balanceItems },
        ].filter((group) => group.items.length > 0);

        return groups.length ? groups : null;
    }, [quarters, valuation]);

    if (!groupedMultiples) return null;

    return (
        <div className="kap-multiples-row">
            {groupedMultiples.map((group, groupIdx) => (
                <section
                    key={group.title}
                    className="kap-multiple-group"
                    style={{ animationDelay: `${groupIdx * 90}ms` }}
                >
                    <h5 className="kap-multiple-group-title">{group.title}</h5>
                    <div className="kap-multiple-grid">
                        {group.items.map((m) => (
                            <div key={m.label} className="kap-multiple-item">
                                <span className="kap-multiple-label">{m.label}</span>
                                <span className={`kap-multiple-value${m.isNeg ? ' negative-ratio' : ''}`}>{m.value}</span>
                            </div>
                        ))}
                    </div>
                </section>
            ))}
        </div>
    );
}

const DEFAULT_INCOME_SUMMARY_ROWS = [
    { key: 'satis_gelirleri', label: 'Satışlar' },
    { key: 'brut_kar', label: 'Brüt Kar' },
    { key: 'esas_faaliyet_kari', label: 'Esas Faaliyet Karı' },
    { key: 'favok', label: 'FAVÖK' },
    { key: 'net_faaliyet_kari', label: 'Net Faaliyet Karı' },
    { key: 'net_kar', label: 'Net Dönem Karı' },
];

const DEFAULT_BALANCE_SUMMARY_ROWS = [
    { key: 'donen_varliklar', label: 'Dönen Varlıklar' },
    { key: 'duran_varliklar', label: 'Duran Varlıklar' },
    { key: 'toplam_varliklar', label: 'Toplam Varlıklar' },
    { key: 'nakit_ve_nakit_benzerleri', label: 'Nakit ve Nakit Benzerleri' },
    { key: 'finansal_borclar', label: 'Finansal Borçlar' },
    { key: 'net_borc', label: 'Net Borç' },
    { key: 'ozkaynaklar', label: 'Özkaynaklar' },
];

const BANK_INCOME_SUMMARY_ROWS = [
    { key: 'faiz_gelirleri', label: 'Faiz Gelirleri' },
    { key: 'faiz_giderleri', label: 'Faiz Giderleri (-)' },
    { key: 'net_ucret_komisyon_gelirleri', label: 'Net Ücret Komisyon Gelirleri' },
    { key: 'net_faaliyet_kari', label: 'Net Faaliyet Karı (Zararı)' },
    { key: 'net_kar', label: 'Net Dönem Karı' },
];

const BANK_BALANCE_SUMMARY_ROWS = [
    { key: 'finansal_varliklar_net', label: 'Finansal Varlıklar (Net)' },
    { key: 'krediler', label: 'Krediler' },
    { key: 'mevduatlar', label: 'Mevduatlar' },
    { key: 'beklenen_zarar_karsiliklari', label: 'Beklenen Zarar Karşılıkları' },
    { key: 'ozkaynaklar', label: 'Özkaynaklar' },
];

const INSURANCE_INCOME_SUMMARY_ROWS = [
    { key: 'prim_uretimi', label: 'Prim Üretimi' },
    { key: 'alinan_net_primler', label: 'Alınan Net Primler' },
    { key: 'teknik_gelirler', label: 'Teknik Gelirler' },
    { key: 'teknik_denge', label: 'Teknik Denge' },
    { key: 'net_kar', label: 'Net Dönem Karı' },
];

const INSURANCE_BALANCE_SUMMARY_ROWS = [
    { key: 'nakit_benzeri_finansal_varliklar', label: 'Nakit Benzeri Finansal Varlıklar' },
    { key: 'esas_faaliyetlerden_alacaklar', label: 'Esas Faaliyetlerden Alacaklar' },
    { key: 'teknik_karsiliklar', label: 'Teknik Karşılıklar' },
    { key: 'esas_faaliyetlerden_borclar', label: 'Esas Faaliyetlerden Borçlar' },
    { key: 'ozkaynaklar', label: 'Özkaynaklar' },
];

const FLOW_METRICS = new Set([
    'satis_gelirleri',
    'brut_kar',
    'favok',
    'net_kar',
    'faaliyet_nakit_akisi',
    'capex',
    'serbest_nakit_akisi',
    'faiz_gelirleri',
    'faiz_giderleri',
    'net_ucret_komisyon_gelirleri',
    'net_faaliyet_kari',
    'esas_faaliyet_kari',
    'amortisman_itfa_gideri',
    'prim_uretimi',
    'alinan_net_primler',
    'teknik_gelirler',
    'teknik_denge',
]);
const CHART_WINDOW_QUARTERS = 10;

type SeriesPoint = {
    key: string;
    label: string;
    value: number;
    display: string;
};

type TooltipState = {
    x: number;
    y: number;
    label: string;
    value: string;
};

function _asNumber(value: number | null | undefined): number | null {
    if (typeof value !== 'number' || Number.isNaN(value) || !Number.isFinite(value)) {
        return null;
    }
    return value;
}

function _periodToMonth(period: number): number {
    if (!Number.isFinite(period)) {
        return 0;
    }
    if (period >= 1 && period <= 4) {
        return period * 3;
    }
    return period;
}

function _periodLabel(year: number, period: number, fallbackQuarter: string): string {
    if (Number.isFinite(year) && Number.isFinite(period) && period > 0) {
        return `${year}/${_periodToMonth(period)}`;
    }
    return fallbackQuarter || '-';
}

function _quarterSortValue(row: KapQuarter): number {
    return intSafe(row.year) * 100 + _periodToMonth(intSafe(row.period));
}

function intSafe(value: unknown): number {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
}

function _normText(value: unknown): string {
    return String(value || '')
        .toUpperCase()
        .replaceAll('İ', 'I')
        .replaceAll('Ş', 'S')
        .replaceAll('Ğ', 'G')
        .replaceAll('Ü', 'U')
        .replaceAll('Ö', 'O')
        .replaceAll('Ç', 'C')
        .replaceAll('ı', 'I')
        .replaceAll('ş', 'S')
        .replaceAll('ğ', 'G')
        .replaceAll('ü', 'U')
        .replaceAll('ö', 'O')
        .replaceAll('ç', 'C');
}

function _formatRatio(value: number, suffix = '%'): string {
    return `${value.toFixed(2)}${suffix}`;
}

function _formatMetric(value: number, currency: string): string {
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

function _metricLabel(row: KapQuarter | null, metricKey: string): string {
    if (!row) {
        return metricKey;
    }
    return (
        row.metrics?.[metricKey]?.label ||
        row.metrics_quarterly?.[metricKey]?.label ||
        row.metrics_ytd?.[metricKey]?.label ||
        metricKey
    );
}

function _rawMetricValue(row: KapQuarter | null, metricKey: string, bucket: 'metrics' | 'metrics_quarterly' | 'metrics_ytd'): number | null {
    if (!row) {
        return null;
    }
    return _asNumber(row[bucket]?.[metricKey]?.value);
}

function _rawYtdForQuarterFlow(row: KapQuarter | null, metricKey: string): number | null {
    if (!row) {
        return null;
    }
    const fromOriginal = _asNumber(row.metrics_ytd_original?.[metricKey]?.value);
    if (fromOriginal !== null) {
        return fromOriginal;
    }
    return _asNumber(row.metrics_ytd?.[metricKey]?.value);
}

function _rawPointForFlow(row: KapQuarter | null, metricKey: string): number | null {
    if (!row) {
        return null;
    }
    const fromOriginal = _asNumber(row.metrics_original?.[metricKey]?.value);
    if (fromOriginal !== null) {
        return fromOriginal;
    }
    return _asNumber(row.metrics?.[metricKey]?.value);
}

function _rawQuarterlyForFlow(row: KapQuarter | null, metricKey: string): number | null {
    if (!row) {
        return null;
    }
    const fromOriginal = _asNumber(row.metrics_quarterly_original?.[metricKey]?.value);
    if (fromOriginal !== null) {
        return fromOriginal;
    }
    return _asNumber(row.metrics_quarterly?.[metricKey]?.value);
}

function _resolveMetricValue(
    rows: KapQuarter[],
    idx: number,
    metricKey: string,
    asQuarterlyFlow: boolean,
): number | null {
    const row = rows[idx];
    if (!row) {
        return null;
    }

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
                if (intSafe(prevRow.year) !== year) {
                    break;
                }
                const prevCandidate = _rawYtdForQuarterFlow(prevRow, metricKey);
                if (prevCandidate !== null) {
                    prevYtd = prevCandidate;
                    break;
                }
            }
            if (period > 1) {
                if (prevYtd !== null) {
                    return ytdForFlow - prevYtd;
                }
                if (quarterlyForFlow !== null && ytdForFlow !== null && quarterlyForFlow === ytdForFlow) {
                    return null;
                }
                if (quarterlyForFlow !== null) {
                    return quarterlyForFlow;
                }
                if (pointForFlow !== null) {
                    return pointForFlow;
                }
                return null;
            }
            return ytdForFlow;
        }
        if (quarterlyForFlow !== null) {
            return quarterlyForFlow;
        }
        if (pointForFlow !== null) {
            return pointForFlow;
        }
        return null;
    }

    if (pointValue !== null) {
        return pointValue;
    }
    if (quarterlyValue !== null) {
        return quarterlyValue;
    }
    if (ytdValue !== null) {
        return ytdValue;
    }
    return null;
}

function _resolveMetricValueByPriority(
    row: KapQuarter | null,
    metricKey: string,
    priority: Array<'metrics' | 'metrics_quarterly' | 'metrics_ytd'>,
): number | null {
    if (!row) {
        return null;
    }
    for (const bucket of priority) {
        const value = _rawMetricValue(row, metricKey, bucket);
        if (value !== null) {
            return value;
        }
    }
    return null;
}

function _resolveMetricDisplay(
    rows: KapQuarter[],
    idx: number,
    metricKey: string,
    asQuarterlyFlow: boolean,
): string {
    const row = rows[idx];
    if (!row) {
        return '-';
    }
    const value = _resolveMetricValue(rows, idx, metricKey, asQuarterlyFlow);
    if (value === null) {
        return '-';
    }
    return _formatMetric(value, row.currency || 'TL');
}

function _buildMetricSeries(quarters: KapQuarter[], metricKey: string, asQuarterlyFlow: boolean): SeriesPoint[] {
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

function _buildCustomSeries(
    quarters: KapQuarter[],
    keySuffix: string,
    resolver: (rows: KapQuarter[], idx: number) => number | null,
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
                display: _formatMetric(numeric, q.currency || 'TL'),
            };
        })
        .filter((row): row is SeriesPoint => row !== null);
}

function _takeLastSeries(series: SeriesPoint[], count = 5): SeriesPoint[] {
    if (series.length <= count) {
        return series;
    }
    return series.slice(-count);
}

function _buildRatioSeries(
    quarters: KapQuarter[],
    numeratorKey: string,
    denominatorKey: string,
    numeratorAsQuarterlyFlow: boolean,
    denominatorAsQuarterlyFlow: boolean,
    scale = 100,
    suffix = '%',
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
                display: _formatRatio(value, suffix),
            };
        })
        .filter((row): row is SeriesPoint => row !== null);
}


function _buildAnnualizedRoeSeries(
    quarters: KapQuarter[],
    numeratorKey: string,
    denominatorKey: string,
    suffix = '%',
): SeriesPoint[] {
    const TTM_LOOKBACK = 4;
    return quarters
        .map((q, idx) => {
            let ttmFlow = 0;
            for (let lookback = 0; lookback < TTM_LOOKBACK; lookback += 1) {
                const cursor = idx - lookback;
                if (cursor < 0) return null;
                const flow = _resolveMetricValue(quarters, cursor, numeratorKey, true);
                if (flow === null) return null;
                ttmFlow += flow;
            }

            const denominatorEnd = _resolveMetricValue(quarters, idx, denominatorKey, false);
            if (denominatorEnd === null || denominatorEnd === 0) return null;
            const denominatorPrev = idx > 0
                ? _resolveMetricValue(quarters, idx - 1, denominatorKey, false)
                : null;
            const averageDenominator = denominatorPrev !== null && denominatorPrev !== 0
                ? (denominatorEnd + denominatorPrev) / 2
                : denominatorEnd;
            if (!averageDenominator) return null;

            const value = (ttmFlow / averageDenominator) * 100;
            return {
                key: `${q.quarter}-${numeratorKey}-${denominatorKey}-ttm`,
                label: _periodLabel(q.year, q.period, q.quarter),
                value,
                display: _formatRatio(value, suffix),
            };
        })
        .filter((row): row is SeriesPoint => row !== null);
}

function _calcPctChange(current: number | null, base: number | null): number | null {
    if (current === null || base === null || base === 0) {
        return null;
    }
    return ((current - base) / Math.abs(base)) * 100;
}

// Metrics where a decrease (negative %) is actually good (debt, expenses)
const INVERSE_METRICS = new Set([
    'net_borc',
    'finansal_borclar',
    'faiz_giderleri',
    'kisa_vadeli_yukumlulukler',
    'beklenen_zarar_karsiliklari',
    'teknik_karsiliklar',
    'esas_faaliyetlerden_borclar',
]);

function _pctClass(value: number | null, metricKey?: string): string {
    if (value === null) {
        return 'pct-neutral';
    }
    const invert = metricKey ? INVERSE_METRICS.has(metricKey) : false;
    if (value > 0) {
        return invert ? 'pct-negative' : 'pct-positive';
    }
    if (value < 0) {
        return invert ? 'pct-positive' : 'pct-negative';
    }
    return 'pct-neutral';
}

function _pctText(value: number | null): string {
    if (value === null || Number.isNaN(value)) {
        return '-';
    }
    return `% ${Math.round(value)}`;
}

function BarChartCard({
    title,
    series,
    highlightedIndex,
    onHighlight,
}: {
    title: string;
    series: SeriesPoint[];
    highlightedIndex?: number | null;
    onHighlight?: (idx: number | null) => void;
}) {
    const cardRef = useRef<HTMLDivElement | null>(null);
    const [tooltip, setTooltip] = useState<TooltipState | null>(null);

    if (!series.length) {
        return null;
    }

    const width = 420;
    const height = 250;
    const padLeft = 34;
    const padRight = 28;
    const padTop = 14;
    const padBottom = 44;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const values = series.map((s) => s.value);
    let minVal = Math.min(...values, 0);
    let maxVal = Math.max(...values, 0);
    if (maxVal === minVal) {
        maxVal = minVal + 1;
    }
    const range = maxVal - minVal;
    const y = (v: number) => padTop + ((maxVal - v) / range) * plotHeight;
    const zeroY = y(0);

    const slot = plotWidth / series.length;
    const barWidth = Math.min(52, slot * 0.68);

    const onHover = (event: React.MouseEvent<SVGElement>, point: SeriesPoint, idx: number) => {
        const cardRect = cardRef.current?.getBoundingClientRect();
        if (!cardRect) {
            return;
        }
        setTooltip({
            x: event.clientX - cardRect.left,
            y: event.clientY - cardRect.top,
            label: point.label,
            value: point.display,
        });
        if (onHighlight) onHighlight(idx);
    };

    const onLeave = () => {
        setTooltip(null);
        if (onHighlight) onHighlight(null);
    };

    return (
        <div className="kap-chart-card" ref={cardRef}>
            <h4>{title}</h4>
            <svg
                className="kap-chart-svg"
                viewBox={`0 0 ${width} ${height}`}
                role="img"
                aria-label={title}
                onMouseLeave={onLeave}
            >
                {[0, 1, 2, 3, 4].map((i) => {
                    const yy = padTop + (plotHeight / 4) * i;
                    return <line key={`h-${i}`} x1={padLeft} y1={yy} x2={width - padRight} y2={yy} className="kap-grid-line" />;
                })}

                {series.map((point, idx) => {
                    const xCenter = padLeft + slot * idx + slot / 2;
                    return (
                        <line
                            key={`${point.key}-v`}
                            x1={xCenter}
                            y1={padTop}
                            x2={xCenter}
                            y2={padTop + plotHeight}
                            className="kap-grid-line kap-grid-line-v"
                        />
                    );
                })}

                <line x1={padLeft} y1={zeroY} x2={width - padRight} y2={zeroY} className="kap-axis-line" />

                {series.map((point, idx) => {
                    const x = padLeft + slot * idx + (slot - barWidth) / 2;
                    const yVal = y(point.value);
                    const top = point.value >= 0 ? yVal : zeroY;
                    const h = Math.max(2, Math.abs(yVal - zeroY));
                    const isFocus = highlightedIndex === idx;
                    const isFaded = highlightedIndex !== null && highlightedIndex !== undefined && !isFocus;
                    return (
                        <g key={point.key}>
                            <rect
                                x={x}
                                y={top}
                                width={barWidth}
                                height={h}
                                rx={4}
                                className={`${point.value < 0 ? 'kap-bar-negative' : 'kap-bar-positive'} ${isFocus ? 'kap-bar-highlighted' : ''} ${isFaded ? 'kap-bar-faded' : ''}`}
                                onMouseEnter={(event) => onHover(event, point, idx)}
                            />
                            <text x={x + barWidth / 2} y={height - 16} textAnchor="middle" className="kap-x-label">
                                {point.label}
                            </text>
                        </g>
                    );
                })}
            </svg>
            {tooltip && (
                <div className="kap-chart-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
                    <div className="kap-chart-tooltip-label">{tooltip.label}</div>
                    <div className="kap-chart-tooltip-value">{tooltip.value}</div>
                </div>
            )}
        </div>
    );
}

function LineChartCard({
    title,
    series,
    highlightedIndex,
    onHighlight,
}: {
    title: string;
    series: SeriesPoint[];
    highlightedIndex?: number | null;
    onHighlight?: (idx: number | null) => void;
}) {
    const cardRef = useRef<HTMLDivElement | null>(null);
    const [tooltip, setTooltip] = useState<TooltipState | null>(null);

    if (!series.length) {
        return null;
    }

    const width = 420;
    const height = 250;
    const padLeft = 34;
    const padRight = 28;
    const padTop = 14;
    const padBottom = 44;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const values = series.map((s) => s.value);
    const minRaw = Math.min(...values);
    const maxRaw = Math.max(...values);
    const span = Math.max(maxRaw - minRaw, Math.abs(maxRaw) * 0.1, 1);
    const minVal = minRaw - span * 0.12;
    const maxVal = maxRaw + span * 0.12;
    const range = maxVal - minVal;
    const y = (v: number) => padTop + ((maxVal - v) / range) * plotHeight;
    const stepX = series.length > 1 ? plotWidth / (series.length - 1) : 0;

    const points = series.map((point, idx) => ({
        x: padLeft + idx * stepX,
        y: y(point.value),
        ...point,
    }));

    const path = points
        .map((point, idx) => `${idx === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
        .join(' ');

    const onHover = (event: React.MouseEvent<SVGElement>, point: SeriesPoint, idx: number) => {
        const cardRect = cardRef.current?.getBoundingClientRect();
        if (!cardRect) {
            return;
        }
        setTooltip({
            x: event.clientX - cardRect.left,
            y: event.clientY - cardRect.top,
            label: point.label,
            value: point.display,
        });
        if (onHighlight) onHighlight(idx);
    };

    const onLeave = () => {
        setTooltip(null);
        if (onHighlight) onHighlight(null);
    };

    return (
        <div className="kap-chart-card" ref={cardRef}>
            <h4>{title}</h4>
            <svg
                className="kap-chart-svg"
                viewBox={`0 0 ${width} ${height}`}
                role="img"
                aria-label={title}
                onMouseLeave={onLeave}
            >
                {[0, 1, 2, 3, 4].map((i) => {
                    const yy = padTop + (plotHeight / 4) * i;
                    return <line key={`h-${i}`} x1={padLeft} y1={yy} x2={width - padRight} y2={yy} className="kap-grid-line" />;
                })}

                {points.map((point) => (
                    <line
                        key={`${point.key}-v`}
                        x1={point.x}
                        y1={padTop}
                        x2={point.x}
                        y2={padTop + plotHeight}
                        className="kap-grid-line kap-grid-line-v"
                    />
                ))}

                <path d={path} className="kap-line-path" />

                {points.map((point, idx) => {
                    const isFocus = highlightedIndex === idx;
                    const isFaded = highlightedIndex !== null && highlightedIndex !== undefined && !isFocus;
                    return (
                        <g key={point.key}>
                            <circle
                                cx={point.x}
                                cy={point.y}
                                r={isFocus ? 6 : 4}
                                className={`kap-line-point ${isFocus ? 'kap-point-highlighted' : ''} ${isFaded ? 'kap-point-faded' : ''}`}
                                onMouseEnter={(event) => onHover(event, point, idx)}
                            />
                            <text x={point.x} y={height - 16} textAnchor="middle" className="kap-x-label">
                                {point.label}
                            </text>
                        </g>
                    );
                })}
            </svg>
            {tooltip && (
                <div className="kap-chart-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
                    <div className="kap-chart-tooltip-label">{tooltip.label}</div>
                    <div className="kap-chart-tooltip-value">{tooltip.value}</div>
                </div>
            )}
        </div>
    );
}

export default function KapPage() {
    const [companies, setCompanies] = useState<string[]>([]);
    const [companiesLoading, setCompaniesLoading] = useState(true);
    const [selectedCompany, setSelectedCompany] = useState('');
    const [searchTerm, setSearchTerm] = useState('');
    const [snapshot, setSnapshot] = useState<KapSnapshotResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [highlightedIndex, setHighlightedIndex] = useState<number | null>(null);
    const initialTickerRef = useRef<string | null>(null);
    const autoFetchedForRef = useRef<string | null>(null);
    const handleHighlight = useCallback((idx: number | null) => {
        setHighlightedIndex((prev) => (prev === idx ? prev : idx));
    }, []);

    // Read ?ticker= from URL on first render
    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const ticker = params.get('ticker');
        if (ticker) {
            initialTickerRef.current = ticker.toUpperCase();
        }
    }, []);

    useEffect(() => {
        setCompaniesLoading(true);
        apiClient
            .kapCompanies()
            .then((d) => {
                const items = d.companies || [];
                setCompanies(items);
                const initial = initialTickerRef.current;
                if (initial && items.map(c => c.toUpperCase()).includes(initial)) {
                    setSelectedCompany(initial);
                } else if (!selectedCompany && items.length > 0) {
                    setSelectedCompany(items[0]);
                }
            })
            .catch((err: any) => setError(err?.message || 'Şirket listesi yüklenemedi.'))
            .finally(() => setCompaniesLoading(false));
    }, []);

    // Update URL when company changes
    useEffect(() => {
        if (!selectedCompany) return;
        const url = new URL(window.location.href);
        url.searchParams.set('ticker', selectedCompany);
        window.history.replaceState({}, '', url.toString());
    }, [selectedCompany]);

    const fetchSnapshot = useCallback(async (refresh = false) => {
        if (!selectedCompany) return;
        setLoading(true);
        setError(null);
        setSnapshot(null);
        try {
            const data = await apiClient.kapSnapshot(selectedCompany, refresh, 10);
            if (!data.ok && data.error) {
                setError(data.error);
            }
            setSnapshot(data);
        } catch (err: any) {
            setError(err.message || 'Snapshot alınamadı.');
        } finally {
            setLoading(false);
        }
    }, [selectedCompany]);

    // Auto-fetch snapshot when a company is selected.
    useEffect(() => {
        if (!selectedCompany || loading) {
            return;
        }
        const loadedSymbol = _normText(snapshot?.stock_code || snapshot?.company);
        const selectedSymbol = _normText(selectedCompany);
        if (loadedSymbol === selectedSymbol) {
            autoFetchedForRef.current = selectedSymbol;
            return;
        }
        if (autoFetchedForRef.current === selectedSymbol) {
            return;
        }
        autoFetchedForRef.current = selectedSymbol;
        fetchSnapshot(false);
    }, [selectedCompany, snapshot, loading, fetchSnapshot]);

    const filteredCompanies = useMemo(
        () =>
            searchTerm
                ? companies.filter((c) => c.toLowerCase().includes(searchTerm.toLowerCase()))
                : companies,
        [companies, searchTerm],
    );

    // Auto-select when search term exactly matches a company code
    useEffect(() => {
        if (!searchTerm) return;
        const upper = searchTerm.trim().toUpperCase();
        const exactMatch = companies.find((c) => c.toUpperCase() === upper);
        if (exactMatch) {
            setSelectedCompany(exactMatch);
        }
    }, [searchTerm, companies]);

    // Warn when search term doesn't match anything
    const searchNoMatch = searchTerm.length >= 2 && filteredCompanies.length === 0;

    const orderedQuarters = useMemo(
        () => {
            if (!snapshot) {
                return [];
            }
            const byPeriod = new Map<string, KapQuarter>();
            for (const q of snapshot.quarters) {
                const key = `${intSafe(q.year)}-${intSafe(q.period)}`;
                if (!byPeriod.has(key)) {
                    byPeriod.set(key, q);
                }
            }
            return [...byPeriod.values()].sort((a, b) => _quarterSortValue(a) - _quarterSortValue(b));
        },
        [snapshot],
    );

    const latestQuarterIdx = orderedQuarters.length ? orderedQuarters.length - 1 : -1;
    const latestQuarter = latestQuarterIdx >= 0 ? orderedQuarters[latestQuarterIdx] : null;
    const prevQuarterIdx = latestQuarterIdx > 0 ? latestQuarterIdx - 1 : -1;
    const prevQuarter = prevQuarterIdx >= 0 ? orderedQuarters[prevQuarterIdx] : null;
    const prevYearSameQuarterIdx = latestQuarter
        ? orderedQuarters.findIndex(
            (q) =>
                intSafe(q.year) === intSafe(latestQuarter.year) - 1 &&
                _periodToMonth(intSafe(q.period)) === _periodToMonth(intSafe(latestQuarter.period)),
        )
        : -1;
    const prevYearSameQuarter = prevYearSameQuarterIdx >= 0 ? orderedQuarters[prevYearSameQuarterIdx] : null;

    const companyKind = classifyKapCompanyKind(snapshot || { company: selectedCompany });
    const isBankLike = companyKind === 'bank';
    const isInsuranceLike = companyKind === 'insurance';

    const incomeSummaryRows = isBankLike
        ? BANK_INCOME_SUMMARY_ROWS
        : isInsuranceLike
            ? INSURANCE_INCOME_SUMMARY_ROWS
            : DEFAULT_INCOME_SUMMARY_ROWS;
    const balanceSummaryRows = isBankLike
        ? BANK_BALANCE_SUMMARY_ROWS
        : isInsuranceLike
            ? INSURANCE_BALANCE_SUMMARY_ROWS
            : DEFAULT_BALANCE_SUMMARY_ROWS;

    const barCharts = useMemo(() => {
        if (!orderedQuarters.length) {
            return [] as Array<{ title: string; series: SeriesPoint[] }>;
        }

        if (isBankLike) {
            return [
                {
                    title: 'Çeyreklik Net Faiz Geliri veya Gideri',
                    series: _takeLastSeries(_buildCustomSeries(orderedQuarters, 'net_faiz_geliri_gideri', (rows, idx) => {
                        const gelir = _resolveMetricValue(rows, idx, 'faiz_gelirleri', true);
                        const gider = _resolveMetricValue(rows, idx, 'faiz_giderleri', true);
                        if (gelir === null && gider === null) {
                            return null;
                        }
                        return (gelir ?? 0) + (gider ?? 0);
                    }), CHART_WINDOW_QUARTERS),
                },
                {
                    title: 'Çeyreklik Net Kar',
                    series: _takeLastSeries(_buildMetricSeries(orderedQuarters, 'net_kar', true), CHART_WINDOW_QUARTERS),
                },
                {
                    title: 'Krediler',
                    series: _takeLastSeries(_buildMetricSeries(orderedQuarters, 'krediler', false), CHART_WINDOW_QUARTERS),
                },
            ].filter((item) => item.series.length > 0);
        }

        if (isInsuranceLike) {
            return [
                {
                    title: 'Çeyreklik Prim Üretimi',
                    series: _takeLastSeries(_buildMetricSeries(orderedQuarters, 'prim_uretimi', true), CHART_WINDOW_QUARTERS),
                },
                {
                    title: 'Çeyreklik Teknik Denge',
                    series: _takeLastSeries(_buildMetricSeries(orderedQuarters, 'teknik_denge', true), CHART_WINDOW_QUARTERS),
                },
                {
                    title: 'Çeyreklik Net Kar',
                    series: _takeLastSeries(_buildMetricSeries(orderedQuarters, 'net_kar', true), CHART_WINDOW_QUARTERS),
                },
            ].filter((item) => item.series.length > 0);
        }

        return [
            {
                title: 'Çeyreklik Satışlar',
                series: _takeLastSeries(_buildMetricSeries(orderedQuarters, 'satis_gelirleri', true), CHART_WINDOW_QUARTERS),
            },
            {
                title: 'Çeyreklik FAVÖK',
                series: _takeLastSeries(_buildMetricSeries(orderedQuarters, 'favok', true), CHART_WINDOW_QUARTERS),
            },
            {
                title: 'Çeyreklik Net Kâr',
                series: _takeLastSeries(_buildMetricSeries(orderedQuarters, 'net_kar', true), CHART_WINDOW_QUARTERS),
            },
            {
                title: 'Çeyreklik Serbest Nakit Akışı',
                series: _takeLastSeries(_buildMetricSeries(orderedQuarters, 'serbest_nakit_akisi', true), CHART_WINDOW_QUARTERS),
            },
        ].filter((item) => item.series.length > 0);
    }, [orderedQuarters, isBankLike, isInsuranceLike]);

    const ratioSeries = useMemo(
        () =>
            orderedQuarters.length
                ? {
                    brutKarMarji: _buildRatioSeries(orderedQuarters, 'brut_kar', 'satis_gelirleri', true, true, 100, '%'),
                    favokMarji: _buildRatioSeries(orderedQuarters, 'favok', 'satis_gelirleri', true, true, 100, '%'),
                    netKarMarji: _buildRatioSeries(orderedQuarters, 'net_kar', 'satis_gelirleri', true, true, 100, '%'),
                    cariOran: _buildRatioSeries(orderedQuarters, 'donen_varliklar', 'kisa_vadeli_yukumlulukler', false, false, 1, 'x'),
                    roe: _buildAnnualizedRoeSeries(orderedQuarters, 'net_kar', 'ozkaynaklar', '%'),
                }
                : null,
        [orderedQuarters],
    );

    const activeRatios = useMemo(() => {
        if (!ratioSeries) return [];
        return [
            { title: 'Brüt Kâr Marjı', series: _takeLastSeries(ratioSeries.brutKarMarji, CHART_WINDOW_QUARTERS) },
            { title: 'FAVÖK Marjı', series: _takeLastSeries(ratioSeries.favokMarji, CHART_WINDOW_QUARTERS) },
            { title: 'Net Kâr Marjı', series: _takeLastSeries(ratioSeries.netKarMarji, CHART_WINDOW_QUARTERS) },
            { title: 'Cari Oran', series: _takeLastSeries(ratioSeries.cariOran, CHART_WINDOW_QUARTERS) },
            { title: 'Özkaynak Karlılığı (ROE)', series: _takeLastSeries(ratioSeries.roe, CHART_WINDOW_QUARTERS) },
        ].filter(r => r.series.length > 0);
    }, [ratioSeries]);

    return (
        <div className="kap-page">
            <header className="kap-header">
                <h1>KAP Dashboard</h1>
                <p>Kamuyu Aydınlatma Platformu — Mali Tablo Verileri</p>
            </header>

            <div className="kap-controls">
                <div className="kap-search-group">
                    <input
                        type="text"
                        className="kap-search"
                        placeholder="Şirket kodu yazın… (örn: THYAO)"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                    <select
                        className="kap-select"
                        value={selectedCompany}
                        onChange={(e) => {
                            setSelectedCompany(e.target.value);
                            setSearchTerm('');
                        }}
                    >
                        <option value="">— Şirket Seç —</option>
                        {filteredCompanies.map((c) => (
                            <option key={c} value={c}>
                                {c}
                            </option>
                        ))}
                    </select>
                </div>
                <div className="kap-btn-group">
                    <button className="btn-primary" disabled={!selectedCompany || loading} onClick={() => fetchSnapshot(false)}>
                        {loading ? 'Yükleniyor…' : 'Getir'}
                    </button>
                    <button
                        className="btn-secondary"
                        disabled={!selectedCompany || loading}
                        onClick={() => fetchSnapshot(true)}
                        title="Önbelleği yoksayarak canlı veri çek"
                        style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                    >
                        <RefreshCw size={16} /> Yenile
                    </button>
                </div>
            </div>

            {searchNoMatch && (
                <div className="kap-search-warn">
                    <strong>"{searchTerm}"</strong> listede bulunamadı. Lütfen geçerli bir BIST şirket kodu girin.
                </div>
            )}

            {error && (
                <div className="kap-error">
                    <strong>Hata:</strong> {error}
                </div>
            )}

            {loading && (
                <div className="kap-loading">
                    <div className="spinner" />
                    <span>KAP verileri cekiliyor. Ilk acilista 5-15 saniye surebilir, lutfen bekleyin…</span>
                </div>
            )}

            {!loading && !error && companiesLoading && (
                <div className="kap-loading">
                    <div className="spinner" />
                    <span>Sirket listesi hazirlaniyor. Backend yeni basladiysa kisa bir sure bekleyin…</span>
                </div>
            )}

            {!loading && !error && !companiesLoading && !snapshot && selectedCompany && (
                <div className="kap-empty">Veri hazirlaniyor. Gecikme olursa birkac saniye sonra "Yenile"ye basabilirsiniz.</div>
            )}

            {snapshot && !loading && (
                <>
                    <div className="kap-info-bar">
                        <span className="kap-company-name">{snapshot.company_title || snapshot.company}</span>
                        <span className="kap-stock-code">{snapshot.stock_code}</span>
                        {latestQuarter && (
                            <span className="kap-quarter-badge">
                                {_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)}
                            </span>
                        )}
                        {latestQuarter?.publish_date && (
                            <span className="kap-publish-date">
                                Yayın: {new Date(latestQuarter.publish_date).toLocaleDateString('tr-TR')}
                            </span>
                        )}
                        {snapshot.cache_hit && <span className="kap-cache-badge">Önbellek</span>}
                    </div>

                    <PriceTicker symbol={snapshot.stock_code || selectedCompany} />
                    <MultiplesRow snapshot={snapshot} quarters={orderedQuarters} />



                    {latestQuarter && (
                        <div className="kap-summary-panel panel">
                            <div className="kap-summary-head">
                                <h3>{snapshot.stock_code || snapshot.company} Özet Finansallar</h3>
                            </div>

                            <div className="kap-summary-grid">
                                <div className="kap-summary-table-wrap">
                                    <h4>
                                        Özet Gelir Tablosu <small>{_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)} YTD</small>
                                    </h4>
                                    <table className="kap-summary-table">
                                        <thead>
                                            <tr>
                                                <th>Kalem</th>
                                                <th>{_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)} YTD</th>
                                                <th>
                                                    {prevYearSameQuarter
                                                        ? `${_periodLabel(intSafe(prevYearSameQuarter.year), intSafe(prevYearSameQuarter.period), prevYearSameQuarter.quarter)} YTD`
                                                        : '-'}
                                                </th>
                                                <th>%</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {incomeSummaryRows.map((row) => {
                                                const currentValue = _resolveMetricYtdValue(latestQuarter, row.key);
                                                const baseValue =
                                                    prevYearSameQuarterIdx >= 0
                                                        ? _resolveMetricYtdValue(prevYearSameQuarter, row.key)
                                                        : null;
                                                if (currentValue === null && baseValue === null) {
                                                    return null;
                                                }
                                                const pct = _calcPctChange(currentValue, baseValue);
                                                return (
                                                    <tr key={`income-${row.key}`}>
                                                        <td>{row.label}</td>
                                                        <td>
                                                            {_resolveMetricYtdDisplay(latestQuarter, row.key)}
                                                        </td>
                                                        <td>
                                                            {prevYearSameQuarterIdx >= 0
                                                                ? _resolveMetricYtdDisplay(prevYearSameQuarter, row.key)
                                                                 : '-'}
                                                        </td>
                                                        <td className={_pctClass(pct, row.key)}>{_pctText(pct)}</td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>

                                <div className="kap-summary-table-wrap">
                                    <h4>
                                        Özet Bilanço <small>{_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)}</small>
                                    </h4>
                                    <table className="kap-summary-table">
                                        <thead>
                                            <tr>
                                                <th>Kalem</th>
                                                <th>{_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)}</th>
                                                <th>
                                                    {prevQuarter
                                                        ? _periodLabel(intSafe(prevQuarter.year), intSafe(prevQuarter.period), prevQuarter.quarter)
                                                        : '-'}
                                                </th>
                                                <th>%</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {balanceSummaryRows.map((row) => {
                                                const currentValue = _resolveMetricValue(orderedQuarters, latestQuarterIdx, row.key, false);
                                                const baseValue =
                                                    prevQuarterIdx >= 0
                                                        ? _resolveMetricValue(orderedQuarters, prevQuarterIdx, row.key, false)
                                                        : null;
                                                if (currentValue === null && baseValue === null) {
                                                    return null;
                                                }
                                                const pct = _calcPctChange(currentValue, baseValue);
                                                return (
                                                    <tr key={`balance-${row.key}`}>
                                                        <td>{row.label}</td>
                                                        <td>{_resolveMetricDisplay(orderedQuarters, latestQuarterIdx, row.key, false)}</td>
                                                        <td>
                                                            {prevQuarterIdx >= 0
                                                                ? _resolveMetricDisplay(orderedQuarters, prevQuarterIdx, row.key, false)
                                                                : '-'}
                                                        </td>
                                                        <td className={_pctClass(pct, row.key)}>{_pctText(pct)}</td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    )}

                    {barCharts.length > 0 && (
                        <div className="kap-charts-grid kap-charts-grid-3">
                            {barCharts.map((chart) => (
                                <BarChartCard
                                    key={chart.title}
                                    title={chart.title}
                                    series={chart.series}
                                    highlightedIndex={highlightedIndex}
                                    onHighlight={handleHighlight}
                                />
                            ))}
                        </div>
                    )}

                    {activeRatios.length > 0 && (
                        <div className="kap-charts-grid kap-charts-grid-3">
                            {activeRatios.map(ratio => (
                                <LineChartCard
                                    key={ratio.title}
                                    title={ratio.title}
                                    series={ratio.series}
                                    highlightedIndex={highlightedIndex}
                                    onHighlight={handleHighlight}
                                />
                            ))}
                        </div>
                    )}

                    {orderedQuarters.length > 0 ? (
                        <div className="kap-table-container">
                            <h3>Çeyrek Bazlı Karşılaştırma</h3>
                            <div className="kap-table-scroll">
                                <table className="kap-table">
                                    <thead>
                                        <tr>
                                            <th>Metrik</th>
                                            {orderedQuarters.map((q) => (
                                                <th key={q.quarter}>{_periodLabel(intSafe(q.year), intSafe(q.period), q.quarter)}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {KAP_FINANCIAL_TABLE_KEYS.map((key) => {
                                            const asQuarterlyFlow = FLOW_METRICS.has(key);
                                            const hasAny = orderedQuarters.some(
                                                (_, idx) => _resolveMetricValue(orderedQuarters, idx, key, asQuarterlyFlow) !== null,
                                            );
                                            if (!hasAny) return null;
                                            const label = _metricLabel(orderedQuarters[orderedQuarters.length - 1] || null, key);
                                            const isInverse = INVERSE_METRICS.has(key);
                                            return (
                                                <tr key={key}>
                                                    <td className="kap-row-label">{label}</td>
                                                    {orderedQuarters.map((q, idx) => {
                                                        const value = _resolveMetricValue(orderedQuarters, idx, key, asQuarterlyFlow);
                                                        const prevValue = idx > 0 ? _resolveMetricValue(orderedQuarters, idx - 1, key, asQuarterlyFlow) : null;
                                                        let heatClass = '';
                                                        if (value !== null && prevValue !== null && prevValue !== 0) {
                                                            const change = value - prevValue;
                                                            if (change > 0) heatClass = isInverse ? 'heatmap-down' : 'heatmap-up';
                                                            else if (change < 0) heatClass = isInverse ? 'heatmap-up' : 'heatmap-down';
                                                        }
                                                        const classes = [
                                                            value !== null && value < 0 ? 'negative' : '',
                                                            heatClass,
                                                        ].filter(Boolean).join(' ');
                                                        return (
                                                            <td key={`${q.quarter}-${key}`} className={classes}>
                                                                {_resolveMetricDisplay(orderedQuarters, idx, key, asQuarterlyFlow)}
                                                            </td>
                                                        );
                                                    })}
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    ) : (
                        !error && <div className="kap-empty">Çeyrek verisi bulunamadı.</div>
                    )}
                </>
            )}
        </div>
    );
}
