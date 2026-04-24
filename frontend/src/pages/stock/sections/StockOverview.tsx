import type { KapSnapshotResponse, KapQuarter } from '../../../api/types';
import { MultiplesRow } from '../../../components/stock/MultiplesRow';
import {
    _resolveMetricValueByPriority, _resolveMetricValue, _resolveMetricDisplayByPriority,
    _resolveMetricDisplay, _calcPctChange, _pctClass, _pctText, intSafe, _periodLabel,
    BANK_TICKERS, FLOW_METRICS,
} from '../../../utils/formatters';
import StockCharts from './StockCharts';
import './StockOverview.css';

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

export default function StockOverview({ snapshot, quarters }: { snapshot: KapSnapshotResponse, quarters: KapQuarter[] }) {
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

    const stockCodeNorm = (snapshot?.stock_code || '').toUpperCase();
    const companyTitleNorm = (snapshot?.company_title || '').toUpperCase();
    const hasAnyMetric = (metricKey: string, asQuarterlyFlow = FLOW_METRICS.has(metricKey)) =>
        quarters.some((_, idx) => _resolveMetricValue(quarters, idx, metricKey, asQuarterlyFlow) !== null);

    const isBankLike = BANK_TICKERS.has(stockCodeNorm) || companyTitleNorm.includes('BANK');
    const isInsuranceLike =
        !isBankLike &&
        (companyTitleNorm.includes('SIGORTA') ||
            hasAnyMetric('prim_uretimi', true) ||
            hasAnyMetric('teknik_denge', true) ||
            hasAnyMetric('esas_faaliyetlerden_alacaklar', false) ||
            hasAnyMetric('teknik_karsiliklar', false));

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

    return (
        <div className="section-overview fade-in">
            <MultiplesRow snapshot={snapshot} quarters={quarters} />

            {latestQuarter && (
                <div className="kap-summary-panel panel">
                    <div className="kap-summary-head">
                        <h3>Özet Finansallar</h3>
                        {snapshot.analysis_note ? (
                            <p className="kap-analysis-note">{snapshot.analysis_note}</p>
                        ) : null}
                    </div>

                    <div className="kap-summary-grid">
                        <div className="kap-summary-table-wrap">
                            <h4>
                                Özet Gelir Tablosu <small>{_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)}</small>
                            </h4>
                            <table className="kap-summary-table">
                                <thead>
                                    <tr>
                                        <th>Kalem</th>
                                        <th>{_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)}</th>
                                        <th>
                                            {prevYearSameQuarter
                                                ? _periodLabel(intSafe(prevYearSameQuarter.year), intSafe(prevYearSameQuarter.period), prevYearSameQuarter.quarter)
                                                : '-'}
                                        </th>
                                        <th>%</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {incomeSummaryRows.map((row) => {
                                        const currentValue = _resolveMetricValueByPriority(
                                            latestQuarter,
                                            row.key,
                                            ['metrics_ytd', 'metrics', 'metrics_quarterly'],
                                        );
                                        const baseValue =
                                            prevYearSameQuarter
                                                ? _resolveMetricValueByPriority(
                                                    prevYearSameQuarter,
                                                    row.key,
                                                    ['metrics_ytd', 'metrics', 'metrics_quarterly'],
                                                )
                                                : null;
                                        if (currentValue === null && baseValue === null) return null;
                                        const pct = _calcPctChange(currentValue, baseValue);
                                        return (
                                            <tr key={`income-${row.key}`}>
                                                <td>{row.label}</td>
                                                <td>
                                                    {_resolveMetricDisplayByPriority(
                                                        latestQuarter,
                                                        row.key,
                                                        ['metrics_ytd', 'metrics', 'metrics_quarterly'],
                                                    )}
                                                </td>
                                                <td>
                                                    {prevYearSameQuarter
                                                        ? _resolveMetricDisplayByPriority(
                                                            prevYearSameQuarter,
                                                            row.key,
                                                            ['metrics_ytd', 'metrics', 'metrics_quarterly'],
                                                        )
                                                        : '-'}
                                                </td>
                                                <td className={`kap-summary-pct-cell ${_pctClass(pct, row.key)}`}>{_pctText(pct)}</td>
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
                                        const currentValue = _resolveMetricValue(quarters, latestQuarterIdx, row.key, false);
                                        const baseValue =
                                            prevQuarterIdx >= 0
                                                ? _resolveMetricValue(quarters, prevQuarterIdx, row.key, false)
                                                : null;
                                        if (currentValue === null && baseValue === null) return null;
                                        const pct = _calcPctChange(currentValue, baseValue);
                                        return (
                                            <tr key={`balance-${row.key}`}>
                                                <td>{row.label}</td>
                                                <td>{_resolveMetricDisplay(quarters, latestQuarterIdx, row.key, false)}</td>
                                                <td>
                                                    {prevQuarterIdx >= 0
                                                        ? _resolveMetricDisplay(quarters, prevQuarterIdx, row.key, false)
                                                        : '-'}
                                                </td>
                                                <td className={`kap-summary-pct-cell ${_pctClass(pct, row.key)}`}>{_pctText(pct)}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {quarters.length > 0 && (
                <section className="overview-charts-shell">
                    <div className="overview-charts-head">
                        <h3>Grafikler ve Analiz</h3>
                        <p>İlk olarak çeyreklik bar grafikler, ardından marj ve oran trendleri.</p>
                    </div>
                    <StockCharts snapshot={snapshot} quarters={quarters} embedded />
                </section>
            )}
        </div>
    );
}
