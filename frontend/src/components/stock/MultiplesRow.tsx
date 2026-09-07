import { ttmSum } from '../../utils/financialMetrics';
import { useMemo } from 'react';
import type { KapSnapshotResponse, KapQuarter, MarketStockCardItem } from '../../api/types';
import { _resolveMetricValueByPriority, _formatRatio, _formatMetric } from '../../utils/formatters';
import './MultiplesRow.css';

type MarketCardMultiples = Pick<MarketStockCardItem, 'fk' | 'pd_dd' | 'fd_favok' | 'net_borc_favok'>;

export function MultiplesRow({
    snapshot,
    quarters,
    marketCard,
}: {
    snapshot: KapSnapshotResponse;
    quarters: KapQuarter[];
    marketCard?: MarketCardMultiples | null;
}) {
    const valuation = snapshot.valuation;

    const groupedMultiples = useMemo(() => {
        if (!quarters.length) return null;

        const latest = quarters[quarters.length - 1];
        const financialCompany = snapshot.company_kind === 'bank' || snapshot.company_kind === 'insurance';
        const ttmSatis = ttmSum(quarters, 'satis_gelirleri');

        const ozkaynaklar = _resolveMetricValueByPriority(latest, 'ozkaynaklar', ['metrics', 'metrics_ytd']);
        const netBorc = _resolveMetricValueByPriority(latest, 'net_borc', ['metrics', 'metrics_ytd']);
        const donenVarliklar = _resolveMetricValueByPriority(latest, 'donen_varliklar', ['metrics', 'metrics_ytd']);
        const kisaVadeli = _resolveMetricValueByPriority(latest, 'kisa_vadeli_yukumlulukler', ['metrics', 'metrics_ytd']);

        const ttmNetKar = valuation ? valuation.ttm_net_kar : ttmSum(quarters, 'net_kar');
        const ttmFavok = financialCompany ? null : (valuation ? valuation.ttm_favok : ttmSum(quarters, 'favok'));

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
        const fk = marketCard ? marketCard.fk : valuation?.fk;
        const pdDd = marketCard ? marketCard.pd_dd : valuation?.pd_dd;
        const fdFavok = marketCard ? marketCard.fd_favok : valuation?.fd_favok;
        const netBorcFavok = marketCard
            ? marketCard.net_borc_favok
            : (!financialCompany && netBorc !== null && ttmFavok !== null && Math.abs(ttmFavok) > 1e-12
                ? netBorc / ttmFavok
                : null);

        multipleItems.push({
            label: 'F/K',
            value: fk != null ? _formatRatio(fk, 'x') : '-',
            isNeg: fk != null ? fk < 0 : false,
        });
        multipleItems.push({
            label: 'PD/DD',
            value: pdDd != null ? _formatRatio(pdDd, 'x') : '-',
            isNeg: pdDd != null ? pdDd < 0 : false,
        });
        if (!financialCompany) multipleItems.push({
            label: 'FD/FAVÖK',
            value: fdFavok != null ? _formatRatio(fdFavok, 'x') : '-',
            isNeg: fdFavok != null ? fdFavok < 0 : false,
        });
        if (!financialCompany) multipleItems.push({
            label: 'Net Borç/FAVÖK',
            value: netBorcFavok != null ? _formatRatio(netBorcFavok, 'x') : '-',
            isNeg: netBorcFavok != null ? netBorcFavok < 0 : false,
        });

        if (!financialCompany && ttmSatis !== null && ttmSatis > 0 && ttmNetKar !== null) {
            const margin = (ttmNetKar / ttmSatis) * 100;
            profitabilityItems.push({ label: 'Net Kâr Marjı', value: _formatRatio(margin), isNeg: margin < 0 });
        }
        if (!financialCompany && ttmSatis !== null && ttmSatis > 0 && ttmFavok !== null) {
            const margin = (ttmFavok / ttmSatis) * 100;
            profitabilityItems.push({ label: 'FAVÖK Marjı', value: _formatRatio(margin), isNeg: margin < 0 });
        }
        if (ozkaynaklar && ozkaynaklar > 0 && ttmNetKar !== null) {
            const roe = (ttmNetKar / ozkaynaklar) * 100;
            profitabilityItems.push({ label: 'ROE', value: _formatRatio(roe), isNeg: roe < 0 });
        }
        if (!financialCompany && netBorc !== null && ozkaynaklar && ozkaynaklar > 0) {
            const ratio = netBorc / ozkaynaklar;
            balanceItems.push({ label: 'Borç/Özkaynak', value: _formatRatio(ratio, 'x'), isNeg: ratio > 1 });
        }
        if (!financialCompany && donenVarliklar && kisaVadeli && kisaVadeli !== 0) {
            const cari = donenVarliklar / kisaVadeli;
            balanceItems.push({ label: 'Cari Oran', value: _formatRatio(cari, 'x'), isNeg: cari < 1 });
        }

        if (ttmNetKar === null) profitabilityItems.push({ label: 'ROE (TTM) — eksik dönem', value: '—' });

        const groups = [
            { title: 'Piyasa', items: marketItems },
            { title: 'Çarpanlar', items: multipleItems },
            { title: 'Kârlılık', items: profitabilityItems },
            { title: 'Finansal Sağlık', items: balanceItems },
        ].filter((group) => group.items.length > 0);

        return groups.length ? groups : null;
    }, [marketCard, quarters, valuation, snapshot.company_kind]);

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
