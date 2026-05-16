import { useMemo, useState, useCallback, useEffect } from 'react';
import type { KapSnapshotResponse, KapQuarter } from '../../../api/types';
import { BarChartCard, LineChartCard } from '../../../components/charts/BarChartCard';
import { buildOverviewChartGroups, type OverviewChartGroup } from '../../../utils/overviewPayload';

type StockChartsProps = {
    snapshot: KapSnapshotResponse;
    quarters: KapQuarter[];
    embedded?: boolean;
    chartWindowQuarters?: number;
};

export default function StockCharts({ snapshot, quarters, embedded = false, chartWindowQuarters }: StockChartsProps) {
    const [highlightedIndex, setHighlightedIndex] = useState<number | null>(null);
    const [expandedChart, setExpandedChart] = useState<OverviewChartGroup | null>(null);
    const handleHighlight = useCallback((idx: number | null) => {
        setHighlightedIndex((prev) => (prev === idx ? prev : idx));
    }, []);
    const closeExpandedChart = useCallback(() => {
        setExpandedChart(null);
        setHighlightedIndex(null);
    }, []);

    const chartGroups = useMemo(() => buildOverviewChartGroups(snapshot, quarters, chartWindowQuarters), [snapshot, quarters, chartWindowQuarters]);
    const barCharts = useMemo(() => chartGroups.filter((chart) => chart.kind === 'bar'), [chartGroups]);
    const activeRatios = useMemo(() => chartGroups.filter((chart) => chart.kind === 'line'), [chartGroups]);

    useEffect(() => {
        if (!expandedChart) {
            return;
        }
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                closeExpandedChart();
            }
        };
        document.addEventListener('keydown', onKeyDown);
        return () => document.removeEventListener('keydown', onKeyDown);
    }, [closeExpandedChart, expandedChart]);

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
                            onOpen={() => setExpandedChart(chart)}
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
                            onOpen={() => setExpandedChart(ratio)}
                        />
                    ))}
                </div>
            )}

            {expandedChart && (
                <div className="kap-chart-modal-backdrop" role="presentation" onClick={closeExpandedChart}>
                    <div
                        className="kap-chart-modal"
                        role="dialog"
                        aria-modal="true"
                        aria-label={`${expandedChart.title} grafiği`}
                        onClick={(event) => event.stopPropagation()}
                    >
                        <button className="kap-chart-modal-close" type="button" onClick={closeExpandedChart} aria-label="Grafiği kapat">
                            ×
                        </button>
                        {expandedChart.kind === 'bar' ? (
                            <BarChartCard
                                title={expandedChart.title}
                                series={expandedChart.series}
                                highlightedIndex={highlightedIndex}
                                onHighlight={handleHighlight}
                                className="kap-chart-card-expanded"
                            />
                        ) : (
                            <LineChartCard
                                title={expandedChart.title}
                                series={expandedChart.series}
                                highlightedIndex={highlightedIndex}
                                onHighlight={handleHighlight}
                                className="kap-chart-card-expanded"
                            />
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
