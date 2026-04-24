import { useMemo, useState, useCallback } from 'react';
import type { KapSnapshotResponse, KapQuarter } from '../../../api/types';
import { BarChartCard, LineChartCard } from '../../../components/charts/BarChartCard';
import {
    _buildMetricSeries, _buildCustomSeries, _takeLastSeries, _buildRatioSeries, CHART_WINDOW_QUARTERS,
} from '../../../utils/chartBuilders';
import {
    _resolveMetricValue, FLOW_METRICS, BANK_TICKERS,
} from '../../../utils/formatters';

type StockChartsProps = {
    snapshot: KapSnapshotResponse;
    quarters: KapQuarter[];
    embedded?: boolean;
};

export default function StockCharts({ snapshot, quarters, embedded = false }: StockChartsProps) {
    const [highlightedIndex, setHighlightedIndex] = useState<number | null>(null);
    const handleHighlight = useCallback((idx: number | null) => {
        setHighlightedIndex((prev) => (prev === idx ? prev : idx));
    }, []);

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

    const barCharts = useMemo(() => {
        if (!quarters.length) return [];

        if (isBankLike) {
            return [
                {
                    title: 'Çeyreklik Net Faiz Geliri veya Gideri',
                    series: _takeLastSeries(_buildCustomSeries(quarters, 'net_faiz_geliri_gideri', (rows, idx) => {
                        const gelir = _resolveMetricValue(rows, idx, 'faiz_gelirleri', true);
                        const gider = _resolveMetricValue(rows, idx, 'faiz_giderleri', true);
                        if (gelir === null && gider === null) return null;
                        return (gelir ?? 0) + (gider ?? 0);
                    }, (val, curr) => `${val.toLocaleString('tr-TR')} ${curr}`), CHART_WINDOW_QUARTERS),
                },
                {
                    title: 'Çeyreklik Net Kar',
                    series: _takeLastSeries(_buildMetricSeries(quarters, 'net_kar', true), CHART_WINDOW_QUARTERS),
                },
                {
                    title: 'Krediler',
                    series: _takeLastSeries(_buildMetricSeries(quarters, 'krediler', false), CHART_WINDOW_QUARTERS),
                },
            ].filter((item) => item.series.length > 0);
        }

        if (isInsuranceLike) {
            return [
                {
                    title: 'Çeyreklik Prim Üretimi',
                    series: _takeLastSeries(_buildMetricSeries(quarters, 'prim_uretimi', true), CHART_WINDOW_QUARTERS),
                },
                {
                    title: 'Çeyreklik Teknik Denge',
                    series: _takeLastSeries(_buildMetricSeries(quarters, 'teknik_denge', true), CHART_WINDOW_QUARTERS),
                },
                {
                    title: 'Çeyreklik Net Kar',
                    series: _takeLastSeries(_buildMetricSeries(quarters, 'net_kar', true), CHART_WINDOW_QUARTERS),
                },
            ].filter((item) => item.series.length > 0);
        }

        return [
            {
                title: 'Çeyreklik Satışlar',
                series: _takeLastSeries(_buildMetricSeries(quarters, 'satis_gelirleri', true), CHART_WINDOW_QUARTERS),
            },
            {
                title: 'Çeyreklik FAVÖK',
                series: _takeLastSeries(_buildMetricSeries(quarters, 'favok', true), CHART_WINDOW_QUARTERS),
            },
            {
                title: 'Çeyreklik Net Kâr',
                series: _takeLastSeries(_buildMetricSeries(quarters, 'net_kar', true), CHART_WINDOW_QUARTERS),
            },
            {
                title: 'Çeyreklik Serbest Nakit Akışı',
                series: _takeLastSeries(_buildMetricSeries(quarters, 'serbest_nakit_akisi', true), CHART_WINDOW_QUARTERS),
            },
        ].filter((item) => item.series.length > 0);
    }, [quarters, isBankLike, isInsuranceLike]);

    const ratioSeries = useMemo(() => {
        if (!quarters.length) return null;
        return {
            brutKarMarji: _buildRatioSeries(quarters, 'brut_kar', 'satis_gelirleri', true, true, 100, v => `${v.toFixed(2)}%`),
            favokMarji: _buildRatioSeries(quarters, 'favok', 'satis_gelirleri', true, true, 100, v => `${v.toFixed(2)}%`),
            netKarMarji: _buildRatioSeries(quarters, 'net_kar', 'satis_gelirleri', true, true, 100, v => `${v.toFixed(2)}%`),
            cariOran: _buildRatioSeries(quarters, 'donen_varliklar', 'kisa_vadeli_yukumlulukler', false, false, 1, v => `${v.toFixed(2)}x`),
            roe: _buildRatioSeries(quarters, 'net_kar', 'ozkaynaklar', true, false, 100, v => `${v.toFixed(2)}%`),
        };
    }, [quarters]);

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
        <div className={`section-charts${embedded ? ' section-charts-embedded' : ' fade-in'}`}>
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
                <div className="kap-charts-grid kap-charts-grid-3 kap-charts-grid-lines">
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
        </div>
    );
}
