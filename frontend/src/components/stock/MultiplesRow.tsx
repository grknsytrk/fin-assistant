import { useMemo } from 'react';
import type { KapSnapshotResponse, KapQuarter } from '../../api/types';
import { _resolveMetricValueByPriority, _formatRatio, _formatMetric } from '../../utils/formatters';
import './MultiplesRow.css';

export function MultiplesRow({ snapshot, quarters }: { snapshot: KapSnapshotResponse; quarters: KapQuarter[] }) {
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
