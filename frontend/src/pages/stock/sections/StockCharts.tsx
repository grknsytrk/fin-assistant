import { useMemo, useState, useCallback } from 'react';
import type { KapSnapshotResponse, KapQuarter } from '../../../api/types';
import { BarChartCard, LineChartCard } from '../../../components/charts/BarChartCard';
import { buildOverviewChartGroups } from '../../../utils/overviewPayload';

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

    const chartGroups = useMemo(() => buildOverviewChartGroups(snapshot, quarters), [snapshot, quarters]);
    const barCharts = useMemo(() => chartGroups.filter((chart) => chart.kind === 'bar'), [chartGroups]);
    const activeRatios = useMemo(() => chartGroups.filter((chart) => chart.kind === 'line'), [chartGroups]);

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
