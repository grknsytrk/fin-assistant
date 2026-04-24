import type { KapQuarter } from '../../../api/types';
import { 
    _resolveMetricValue, _resolveMetricDisplay, intSafe, _periodLabel,
    FLOW_METRICS, INVERSE_METRICS
} from '../../../utils/formatters';

const TABLE_KEYS = [
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
];

export default function StockFinancials({
    quarters,
    analysisNote,
}: {
    quarters: KapQuarter[];
    analysisNote?: string;
}) {
    if (!quarters.length) {
        return <div className="kap-empty">Çeyrek verisi bulunamadı.</div>;
    }

    return (
        <div className="section-financials fade-in">
            <div className="kap-table-container panel">
                <h3>Çeyrek Bazlı Karşılaştırma</h3>
                {analysisNote ? <p className="kap-analysis-note">{analysisNote}</p> : null}
                <div className="kap-table-scroll">
                    <table className="kap-table">
                        <thead>
                            <tr>
                                <th>Metrik</th>
                                {quarters.map((q) => (
                                    <th key={q.quarter}>{_periodLabel(intSafe(q.year), intSafe(q.period), q.quarter)}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {TABLE_KEYS.map((key) => {
                                const asQuarterlyFlow = FLOW_METRICS.has(key);
                                const hasAny = quarters.some(
                                    (_, idx) => _resolveMetricValue(quarters, idx, key, asQuarterlyFlow) !== null,
                                );
                                if (!hasAny) return null;
                                
                                const label = _metricLabel(quarters[quarters.length - 1] || null, key);
                                const isInverse = INVERSE_METRICS.has(key);
                                
                                return (
                                    <tr key={key}>
                                        <td className="kap-row-label">{label}</td>
                                        {quarters.map((q, idx) => {
                                            const value = _resolveMetricValue(quarters, idx, key, asQuarterlyFlow);
                                            const prevValue = idx > 0 ? _resolveMetricValue(quarters, idx - 1, key, asQuarterlyFlow) : null;
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
                                                    {_resolveMetricDisplay(quarters, idx, key, asQuarterlyFlow)}
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
        </div>
    );
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
