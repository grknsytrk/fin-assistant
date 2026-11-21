import {
    Fragment,
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
    type CSSProperties,
    type PointerEvent as ReactPointerEvent,
    type ReactNode,
} from 'react';
import { ArrowUpDown, ChevronRight, Info, LineChart, LogIn, LogOut, ShieldAlert } from 'lucide-react';
import { Link } from 'react-router-dom';
import type {
    FundAllocation,
    FundAllocationHistoryDay,
    FundAllocationsHistoryResponse,
    FundHoldingsResponse,
    FundPortfolioPosition,
} from '../../../api/types';
import { FinLoader } from '../../../components/FinLoader';
import SymbolLogo from '../../../components/SymbolLogo';
import { toFundDetail, toStockDetail, type FundTab } from '../../../routing/routes';
import {
    donutSegmentPath,
    filterFundHoldings,
    finiteNumber,
    formatAllocationWeight,
    formatCompactWeightDelta,
    formatDate,
    formatHoldingPrice,
    formatHoldingWeight,
    formatMonthLegendLabel,
    formatPct,
    formatSignedCompactCurrency,
    holdingAssetCategory,
    holdingChangeStatus,
    holdingDisplayCode,
    holdingLogoKind,
    holdingLogoName,
    holdingLogoSymbol,
    holdingPositionKey,
    holdingWeightValue,
    holdingWeightWarning,
    latestHoldingReportMeta,
    normalizeCompareSymbol,
    pctClass,
    polarPoint,
    previousHoldingReportMeta,
    sortHoldingPositions,
    sortHoldingPositionsByWeight,
} from '../formatters';
import {
    FUND_DONUT_COLORS,
    FUND_HOLDINGS_LIVE_FLASH_MS,
    FUND_HOLDING_CATEGORY_FILTERS,
    FUND_HOLDING_CHANGE_FILTERS,
    FUND_HOLDING_GROUPS,
    FUND_HOLDING_SORT_COLUMNS,
    type AllocationHistoryPoint,
    type AllocationHistoryTrend,
    type FundHoldingCategoryFilter,
    type FundHoldingChangeFilter,
    type FundHoldingSortDirection,
    type FundHoldingSortKey,
} from '../types';
import './FundHoldingsSection.css';

export function HoldingWeightDisplay({
    position,
    weightKey = 'weight',
}: {
    position: FundPortfolioPosition;
    weightKey?: 'weight' | 'previous_weight';
}) {
    const value = formatHoldingWeight(position, weightKey);
    const warning = holdingWeightWarning(position, weightKey);
    return (
        <span
            className={`fund-holding-weight-value${warning ? ' has-warning' : ''}`}
            title={warning || undefined}
            aria-label={warning ? `${value}. ${warning}` : value}
        >
            {value}
            {warning ? <ShieldAlert size={12} aria-hidden="true" /> : null}
        </span>
    );
}



export function renderWeightDelta(value: number | null | undefined): ReactNode {
    if (value == null || !Number.isFinite(value)) return '-';
    const sign = value > 0 ? '+' : '';
    const numStr = `${sign}${value.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    return (
        <>
            <span style={{ fontFamily: 'var(--font-mono)' }}>{numStr}</span>
            <span style={{ fontFamily: 'var(--font-family)', marginLeft: '0.22rem' }}>puan</span>
        </>
    );
}

export function FundAllocationSummary({
    allocations,
    loading,
    onOpenHistory,
    historyLoading = false,
}: {
    allocations: FundAllocation[];
    loading: boolean;
    onOpenHistory?: () => void;
    historyLoading?: boolean;
}) {
    const [activeIndex, setActiveIndex] = useState<number | null>(null);
    const positiveAllocations = allocations.filter((item) => Number(item.weight) > 0).slice(0, FUND_DONUT_COLORS.length);
    const positiveTotal = positiveAllocations.reduce((sum, item) => sum + Number(item.weight || 0), 0);
    const netTotal = allocations.reduce((sum, item) => {
        const weight = Number(item.weight);
        return Number.isFinite(weight) ? sum + weight : sum;
    }, 0);
    let cursor = 0;
    const segments = positiveTotal > 0
        ? positiveAllocations.map((item, index) => {
            const weight = Number(item.weight || 0);
            const startAngle = cursor;
            const angle = (weight / positiveTotal) * 360;
            cursor += angle;
            const midAngle = startAngle + angle / 2;
            const tooltipPoint = polarPoint(50, 50, 36, midAngle);
            return {
                item,
                index,
                color: FUND_DONUT_COLORS[index % FUND_DONUT_COLORS.length],
                path: donutSegmentPath(startAngle, cursor),
                tooltipX: tooltipPoint.x,
                tooltipY: tooltipPoint.y,
            };
        })
        : [];
    const activeSegment = activeIndex == null ? null : segments[activeIndex] || null;

    return (
        <section className="fund-allocation-panel">
            <h2>Varlık Dağılımı</h2>
            {loading ? (
                <FinLoader message="Dağılım verisi yükleniyor" />
            ) : positiveAllocations.length ? (
                <>
                    <div className="fund-allocation-donut-wrap" onPointerLeave={() => setActiveIndex(null)}>
                        <svg className="fund-allocation-donut" viewBox="0 0 200 200" role="img" aria-label="Fon varlık dağılımı">
                            <circle cx="100" cy="100" r="92" className="fund-allocation-donut-track" />
                            {segments.map((segment) => {
                                const isMuted = activeIndex != null && activeIndex !== segment.index;
                                return (
                                    <path
                                        key={segment.item.allocation_type}
                                        d={segment.path}
                                        fill={segment.color}
                                        className={[
                                            'fund-allocation-segment',
                                            activeIndex === segment.index ? 'is-active' : '',
                                            isMuted ? 'is-muted' : '',
                                        ].filter(Boolean).join(' ')}
                                        tabIndex={0}
                                        aria-label={`${segment.item.label}: ${formatAllocationWeight(segment.item.weight)}`}
                                        onPointerEnter={() => setActiveIndex(segment.index)}
                                        onFocus={() => setActiveIndex(segment.index)}
                                        onBlur={() => setActiveIndex(null)}
                                    />
                                );
                            })}
                            <circle cx="100" cy="100" r="44" className="fund-allocation-donut-hole" />
                        </svg>
                        <div className="fund-allocation-donut-center">
                            <span>{activeSegment ? activeSegment.item.label : 'Toplam'}</span>
                            <strong>{activeSegment ? formatAllocationWeight(activeSegment.item.weight) : formatAllocationWeight(netTotal)}</strong>
                        </div>
                        {activeSegment && (
                            <div
                                className="fund-allocation-tooltip"
                                style={{
                                    left: `${activeSegment.tooltipX}%`,
                                    top: `${activeSegment.tooltipY}%`,
                                }}
                            >
                                <span>{activeSegment.item.label}</span>
                                <strong>{formatAllocationWeight(activeSegment.item.weight)}</strong>
                            </div>
                        )}
                    </div>
                    <div className="fund-allocation-list">
                        {positiveAllocations.map((item, index) => (
                            <div
                                key={`${item.allocation_type}-${index}`}
                                className={[
                                    'fund-allocation-row',
                                    activeIndex === index ? 'is-active' : '',
                                    activeIndex != null && activeIndex !== index ? 'is-muted' : '',
                                ].filter(Boolean).join(' ')}
                                onPointerEnter={() => setActiveIndex(index)}
                                onPointerLeave={() => setActiveIndex(null)}
                            >
                                <i style={{ background: FUND_DONUT_COLORS[index % FUND_DONUT_COLORS.length] }} />
                                <span>{item.label}</span>
                                <strong className={pctClass(item.weight)}>{formatAllocationWeight(item.weight)}</strong>
                            </div>
                        ))}
                    </div>
                    {onOpenHistory && (
                        <button
                            className="fund-allocation-history-button"
                            type="button"
                            onClick={onOpenHistory}
                            aria-busy={historyLoading}
                        >
                            <LineChart size={15} aria-hidden="true" />
                            {historyLoading ? 'Geçmiş yükleniyor...' : 'Geçmiş verilerde gör'}
                        </button>
                    )}
                </>
            ) : (
                <div className="funds-state">TEFAS dağılım verisi bu fon için yok.</div>
            )}
        </section>
    );
}

export function allocationWeightForType(day: FundAllocationHistoryDay, allocationType: string): number | null {
    const item = day.allocations.find((allocation) => allocation.allocation_type === allocationType);
    const weight = Number(item?.weight);
    return Number.isFinite(weight) ? weight : null;
}

export function buildAllocationHistoryTrends(history: FundAllocationHistoryDay[]): AllocationHistoryTrend[] {
    const ordered = [...history].sort((a, b) => a.date.localeCompare(b.date));
    const latest = ordered[ordered.length - 1];
    if (!latest) return [];
    const topTypes = latest.allocations
        .filter((item) => Number.isFinite(Number(item.weight)) && Number(item.weight) > 0)
        .slice(0, 8)
        .map((item) => item.allocation_type);
    return topTypes.map((allocationType, index) => {
        const latestItem = latest.allocations.find((item) => item.allocation_type === allocationType);
        const points = ordered.map((day) => ({
            date: day.date,
            weight: allocationWeightForType(day, allocationType),
        }));
        const firstPoint = points.find((point) => point.weight != null);
        const lastPoint = [...points].reverse().find((point) => point.weight != null);
        const startWeight = firstPoint?.weight ?? null;
        const endWeight = lastPoint?.weight ?? null;
        return {
            allocationType,
            label: latestItem?.label || allocationType.toUpperCase(),
            color: FUND_DONUT_COLORS[index % FUND_DONUT_COLORS.length],
            startWeight,
            endWeight,
            delta: startWeight != null && endWeight != null ? endWeight - startWeight : null,
            points,
        };
    });
}

export function allocationDeltaAtIndex(points: AllocationHistoryPoint[], index: number): number | null {
    const current = points[index]?.weight;
    if (current == null || !Number.isFinite(current)) return null;
    for (let previousIndex = index - 1; previousIndex >= 0; previousIndex -= 1) {
        const previous = points[previousIndex]?.weight;
        if (previous != null && Number.isFinite(previous)) {
            return current - previous;
        }
    }
    return null;
}

export function latestAllocationDelta(trend: AllocationHistoryTrend): number | null {
    for (let index = trend.points.length - 1; index >= 0; index -= 1) {
        if (trend.points[index]?.weight != null) {
            return allocationDeltaAtIndex(trend.points, index);
        }
    }
    return null;
}

export function AllocationSparkline({ trend }: { trend: AllocationHistoryTrend }) {
    const values = trend.points.map((point) => point.weight).filter((value): value is number => value != null && Number.isFinite(value));
    if (values.length < 2) return <span className="fund-allocation-sparkline-empty">-</span>;
    const width = 132;
    const height = 34;
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const span = Math.max(0.01, maxValue - minValue);
    const usablePoints = trend.points.filter((point) => point.weight != null);
    const path = usablePoints.map((point, index) => {
        const x = usablePoints.length === 1 ? 0 : (index / (usablePoints.length - 1)) * width;
        const y = height - ((Number(point.weight) - minValue) / span) * height;
        return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
    return (
        <svg className="fund-allocation-sparkline" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
            <path d={path} fill="none" stroke={trend.color} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
    );
}

export function AllocationHistoryChart({
    history,
    trends,
}: {
    history: FundAllocationHistoryDay[];
    trends: AllocationHistoryTrend[];
}) {
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const width = 980;
    const height = 330;
    const padding = { top: 24, right: 62, bottom: 38, left: 42 };
    const dates = history.map((day) => day.date);
    const values = trends.flatMap((trend) => trend.points.map((point) => point.weight)).filter((value): value is number => value != null && Number.isFinite(value));
    if (dates.length < 2 || values.length < 2) {
        return null;
    }
    const minValue = Math.min(0, ...values);
    const maxValue = Math.max(1, ...values);
    const span = Math.max(1, maxValue - minValue);
    const tickCount = 5;
    const ticks = Array.from({ length: tickCount }, (_, index) => minValue + (span * index) / (tickCount - 1));
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    const xForIndex = (index: number) => padding.left + (dates.length === 1 ? 0 : (index / (dates.length - 1)) * chartWidth);
    const yForValue = (value: number) => padding.top + (1 - (value - minValue) / span) * chartHeight;
    const pathForTrend = (trend: AllocationHistoryTrend) => {
        let started = false;
        return trend.points.reduce<string[]>((commands, point, index) => {
            if (point.weight == null || !Number.isFinite(point.weight)) {
                started = false;
                return commands;
            }
            commands.push(`${started ? 'L' : 'M'} ${xForIndex(index)} ${yForValue(point.weight)}`);
            started = true;
            return commands;
        }, []).join(' ');
    };
    const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        if (rect.width <= 0) return;
        const localX = event.clientX - rect.left;
        const scaledX = (localX / rect.width) * width;
        const clamped = Math.min(Math.max(scaledX, padding.left), width - padding.right);
        const ratio = chartWidth > 0 ? (clamped - padding.left) / chartWidth : 0;
        setHoverIndex(Math.max(0, Math.min(dates.length - 1, Math.round(ratio * (dates.length - 1)))));
    };
    const hoverDate = hoverIndex == null ? null : dates[hoverIndex];
    const hoverRows = hoverIndex == null
        ? []
        : trends
            .map((trend) => ({
                trend,
                weight: trend.points[hoverIndex]?.weight ?? null,
                delta: allocationDeltaAtIndex(trend.points, hoverIndex),
            }))
            .filter((item) => item.weight != null)
            .sort((a, b) => Number(b.weight) - Number(a.weight));
    const hoverAnchorWeight = hoverRows[0]?.weight ?? null;
    const hoverAnchorY = hoverAnchorWeight == null ? height / 2 : yForValue(Number(hoverAnchorWeight));
    const tooltipLeft = hoverIndex == null ? 0 : (xForIndex(hoverIndex) / width) * 100;
    const tooltipTop = hoverIndex == null ? 0 : (hoverAnchorY / height) * 100;
    const tooltipAlignRight = tooltipLeft > 62;
    const tooltipAlignUp = hoverAnchorY > height * 0.55;

    return (
        <div className="fund-allocation-history-chart-wrap">
            <svg
                className="fund-allocation-history-chart"
                viewBox={`0 0 ${width} ${height}`}
                preserveAspectRatio="none"
                role="img"
                aria-label="Varlık dağılımı zaman serisi"
                onPointerMove={handlePointerMove}
                onPointerLeave={() => setHoverIndex(null)}
            >
                {ticks.map((tick) => {
                    const y = yForValue(tick);
                    return (
                        <g key={tick}>
                            <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="fund-allocation-history-gridline" />
                            <text x={width - padding.right + 8} y={y + 4} className="fund-chart-axis">
                                {formatAllocationWeight(tick)}
                            </text>
                        </g>
                    );
                })}
                {dates.map((dateValue, index) => {
                    if (index !== 0 && index !== dates.length - 1 && index !== Math.floor((dates.length - 1) / 2)) {
                        return null;
                    }
                    return (
                        <text
                            key={dateValue}
                            x={xForIndex(index)}
                            y={height - 10}
                            className="fund-chart-axis"
                            textAnchor={index === 0 ? 'start' : index === dates.length - 1 ? 'end' : 'middle'}
                        >
                            {formatDate(dateValue)}
                        </text>
                    );
                })}
                {trends.map((trend) => (
                    <path
                        key={trend.allocationType}
                        d={pathForTrend(trend)}
                        fill="none"
                        stroke={trend.color}
                        strokeWidth="2.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />
                ))}
                {hoverIndex != null && (
                    <>
                        <line
                            x1={xForIndex(hoverIndex)}
                            x2={xForIndex(hoverIndex)}
                            y1={padding.top}
                            y2={height - padding.bottom}
                            className="fund-allocation-history-hoverline"
                        />
                        {hoverRows.map(({ trend, weight }) => (
                            <circle
                                key={trend.allocationType}
                                cx={xForIndex(hoverIndex)}
                                cy={yForValue(Number(weight))}
                                r="3.4"
                                fill={trend.color}
                                stroke="#0a0c0f"
                                strokeWidth="1.8"
                            />
                        ))}
                    </>
                )}
            </svg>
            {hoverDate && hoverRows.length ? (
                <div
                    className={[
                        'fund-allocation-history-tooltip',
                        tooltipAlignRight ? 'align-right' : '',
                        tooltipAlignUp ? 'align-up' : '',
                    ].filter(Boolean).join(' ')}
                    style={{ left: `${tooltipLeft}%`, top: `${tooltipTop}%` }}
                >
                    <strong>{formatDate(hoverDate)}</strong>
                    {hoverRows.slice(0, 4).map(({ trend, weight, delta }) => (
                        <span key={trend.allocationType}>
                            <i style={{ background: trend.color }} />
                            <em>{trend.label}</em>
                            <b>{formatAllocationWeight(weight)}</b>
                            <small className={pctClass(delta)}>{renderWeightDelta(delta)}</small>
                        </span>
                    ))}
                </div>
            ) : null}
            <div className="fund-allocation-history-legend">
                {trends.slice(0, 6).map((trend) => (
                    <span key={trend.allocationType}>
                        <i style={{ background: trend.color }} />
                        {trend.label}
                    </span>
                ))}
            </div>
        </div>
    );
}

export function FundAllocationHistoryPanel({
    history,
    loading,
    error,
}: {
    history: FundAllocationsHistoryResponse | null;
    loading: boolean;
    error: string | null;
}) {
    const orderedHistory = useMemo(() => [...(history?.history || [])].sort((a, b) => a.date.localeCompare(b.date)), [history]);
    const trends = useMemo(() => buildAllocationHistoryTrends(orderedHistory), [orderedHistory]);
    const firstDate = orderedHistory[0]?.date;
    const lastDate = orderedHistory[orderedHistory.length - 1]?.date;

    return (
        <section className="fund-allocation-history-panel" aria-label="Varlık dağılımı geçmişi">
            <div className="fund-allocation-history-head">
                <span>Son {history?.lookback_days || 30} gün</span>
                <h3>Varlık Dağılımı Geçmişi</h3>
            </div>
            {loading ? (
                <FinLoader message="Dağılım geçmişi yükleniyor" />
            ) : error ? (
                <div className="funds-state funds-state-error">{error}</div>
            ) : history?.refresh_pending && !trends.length ? (
                <div className="funds-state">Dağılım geçmişi arka planda hazırlanıyor. Bu ekran otomatik güncellenecek.</div>
            ) : trends.length ? (
                <>
                    <div className="fund-allocation-history-range">
                        <span>{formatDate(firstDate)}</span>
                        <strong>{orderedHistory.length} rapor günü</strong>
                        <span>{formatDate(lastDate)}</span>
                    </div>
                    <AllocationHistoryChart history={orderedHistory} trends={trends} />
                    <div className="fund-allocation-history-table-wrap">
                        <table className="fund-history-table fund-allocation-history-table">
                            <thead>
                                <tr>
                                    <th>Varlık</th>
                                    <th>İlk</th>
                                    <th>Son</th>
                                    <th>Son Gün</th>
                                    <th>Değişim</th>
                                    <th>Trend</th>
                                </tr>
                            </thead>
                            <tbody>
                                {trends.map((trend) => (
                                    <tr key={trend.allocationType}>
                                        <td>
                                            <span className="fund-allocation-history-name">
                                                <i style={{ background: trend.color }} />
                                                {trend.label}
                                            </span>
                                        </td>
                                        <td>{formatAllocationWeight(trend.startWeight)}</td>
                                        <td>{formatAllocationWeight(trend.endWeight)}</td>
                                        <td className={pctClass(latestAllocationDelta(trend))}>{renderWeightDelta(latestAllocationDelta(trend))}</td>
                                        <td className={pctClass(trend.delta)}>{renderWeightDelta(trend.delta)}</td>
                                        <td><AllocationSparkline trend={trend} /></td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            ) : (
                <div className="funds-state">Son 30 gün için dağılım geçmişi bulunamadı.</div>
            )}
        </section>
    );
}

export function holdingSortColumnLabel(key: FundHoldingSortKey): string {
    return FUND_HOLDING_SORT_COLUMNS.find((column) => column.key === key)?.label || key;
}

export function FundHoldingsSortHeader({
    columnKey,
    sortKey,
    sortDirection,
    onSort,
}: {
    columnKey: FundHoldingSortKey;
    sortKey: FundHoldingSortKey;
    sortDirection: FundHoldingSortDirection;
    onSort: (key: FundHoldingSortKey) => void;
}) {
    const label = holdingSortColumnLabel(columnKey);
    const active = sortKey === columnKey;
    const sortLabel = active ? (sortDirection === 'asc' ? 'artan' : 'azalan') : '';

    return (
        <th aria-sort={active ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'}>
            <button
                type="button"
                className={`fund-holdings-th-sort${active ? ' is-active' : ''}`}
                onClick={() => onSort(columnKey)}
            >
                <span>{label}</span>
                <ArrowUpDown size={13} aria-hidden="true" />
                <span className="sr-only">{sortLabel ? `Şu an ${sortLabel}` : `${label} sütununa göre sırala`}</span>
            </button>
        </th>
    );
}

export type FundHoldingsHighlightCardProps = {
    title: string;
    positions: FundPortfolioPosition[];
};

export function FundHoldingsHighlightCard({ title, positions }: FundHoldingsHighlightCardProps) {
    return (
        <section className="fund-holdings-highlight-card" aria-label={title}>
            <h4>{title}</h4>
            <div className="fund-holdings-highlight-head" aria-hidden="true">
                <span>Sembol</span>
                <span>Ağırlık/Değişim</span>
            </div>
            {positions.length ? (
                <div className="fund-holdings-highlight-list">
                    {positions.map((position) => {
                        const symbol = holdingLogoSymbol(position);
                        const category = holdingAssetCategory(position) || 'local_equity';
                        return (
                            <div className="fund-holdings-highlight-row" key={`${title}-${holdingPositionKey(position)}`}>
                                <span className="fund-holdings-highlight-symbol">
                                    <SymbolLogo
                                        symbol={symbol}
                                        name={holdingLogoName(position)}
                                        kind={category === 'fund' ? 'fund' : 'stock'}
                                        logoUrl={position.logo_url}
                                        size="sm"
                                    />
                                    <strong>{holdingDisplayCode(position)}</strong>
                                </span>
                                <span className="fund-holdings-highlight-values">
                                    <strong><HoldingWeightDisplay position={position} /></strong>
                                    <small className={pctClass(position.weight_change)}>{formatCompactWeightDelta(position.weight_change)}</small>
                                </span>
                            </div>
                        );
                    })}
                </div>
            ) : (
                <div className="fund-holdings-highlight-empty">Veri yok</div>
            )}
        </section>
    );
}

export function FundHoldingsSnapshot({
    positions,
    reportDate,
    reportUrl,
}: {
    positions: FundPortfolioPosition[];
    reportDate: string | null;
    reportUrl: string | null;
}) {
    const relevantPositions = useMemo(
        () => positions.filter((position) => {
            const weight = holdingWeightValue(position);
            return weight != null && weight > 0;
        }),
        [positions],
    );
    const biggestPositions = useMemo(
        () => sortHoldingPositions(relevantPositions, 'weight', 'desc').slice(0, 5),
        [relevantPositions],
    );
    const increasedPositions = useMemo(
        () => sortHoldingPositions(
            relevantPositions.filter((position) => Number(position.weight_change) > 0.005),
            'weight_change',
            'desc',
        ).slice(0, 5),
        [relevantPositions],
    );
    const decreasedPositions = useMemo(
        () => sortHoldingPositions(
            relevantPositions.filter((position) => Number(position.weight_change) < -0.005),
            'weight_change',
            'asc',
        ).slice(0, 5),
        [relevantPositions],
    );

    if (!positions.length) return null;

    return (
        <div className="fund-holdings-snapshot">
            <div className="fund-holdings-source-note">
                <Info size={15} aria-hidden="true" />
                <span>
                    Fon portföyü ile ilgili tüm veriler{' '}
                    <strong>{reportDate ? formatDate(reportDate) : 'son KAP'}</strong>{' '}
                    portföy dağılım raporu baz alınarak hesaplanmıştır.
                    {reportUrl ? (
                        <>
                            {' '}
                            <a href={reportUrl} target="_blank" rel="noreferrer">Detaylı Bilgi</a>
                        </>
                    ) : null}
                </span>
            </div>
            <div className="fund-holdings-highlight-grid">
                <FundHoldingsHighlightCard title="En Büyük Pozisyonlar" positions={biggestPositions} />
                <FundHoldingsHighlightCard title="Artırılan Pozisyonlar" positions={increasedPositions} />
                <FundHoldingsHighlightCard title="Azaltılan Pozisyonlar" positions={decreasedPositions} />
            </div>
        </div>
    );
}

export function FundOverviewHoldingsSection({
    holdings,
    loading,
    error,
}: {
    holdings: FundHoldingsResponse | null;
    loading: boolean;
    error: string | null;
}) {
    const positions = useMemo(
        () => (holdings?.positions || []).filter((position) => holdingAssetCategory(position)),
        [holdings],
    );
    const latestReport = latestHoldingReportMeta(holdings);
    const reportDate = latestReport?.report_date || holdings?.source_metadata?.as_of || positions.find((position) => position.report_date)?.report_date || null;
    const reportUrl = latestReport?.source_url || positions.find((position) => position.source_report_url)?.source_report_url || null;

    if (error) {
        return <div className="funds-state">{error}</div>;
    }
    if (loading || (!holdings && !error)) {
        return <FinLoader message="KAP fon içeriği yükleniyor" />;
    }
    if (!positions.length) {
        return null;
    }
    return (
        <FundHoldingsSnapshot
            positions={positions}
            reportDate={reportDate}
            reportUrl={reportUrl}
        />
    );
}

export function liveFlashNumber(value: number | null | undefined): number | null {
    return value != null && Number.isFinite(value) ? value : null;
}

export function FundLiveValueCell({
    value,
    className,
    children,
}: {
    value: number | null | undefined;
    className?: string;
    children: ReactNode;
}) {
    const previousValueRef = useRef<number | null | undefined>(undefined);
    const [flashClass, setFlashClass] = useState('');

    useEffect(() => {
        const currentValue = liveFlashNumber(value);
        const previousValue = previousValueRef.current;
        if (previousValue !== undefined && previousValue != null && currentValue != null && currentValue !== previousValue) {
            setFlashClass(currentValue > previousValue ? 'fund-live-flash-up' : 'fund-live-flash-down');
            const timer = window.setTimeout(() => setFlashClass(''), FUND_HOLDINGS_LIVE_FLASH_MS);
            previousValueRef.current = currentValue;
            return () => window.clearTimeout(timer);
        }
        previousValueRef.current = currentValue;
        return undefined;
    }, [value]);

    return (
        <td className={[className, flashClass].filter(Boolean).join(' ')}>
            {children}
        </td>
    );
}

export type FundHoldingsPanelProps = {
    holdings: FundHoldingsResponse | null;
    loading: boolean;
    error: string | null;
    categoryFilter: FundHoldingCategoryFilter;
    changeFilter: FundHoldingChangeFilter;
    onCategoryFilterChange: (filter: FundHoldingCategoryFilter) => void;
    onChangeFilterChange: (filter: FundHoldingChangeFilter) => void;
    onOpenTicker?: (ticker: string) => void;
    onOpenFund?: (fundCode: string, tab?: FundTab) => void;
};

export function FundHoldingsPanel({
    holdings,
    loading,
    error,
    categoryFilter,
    changeFilter,
    onCategoryFilterChange,
    onChangeFilterChange,
    onOpenTicker,
    onOpenFund,
}: FundHoldingsPanelProps) {
    const positions = holdings?.positions || [];
    const [holdingSort, setHoldingSort] = useState<{ key: FundHoldingSortKey; direction: FundHoldingSortDirection }>({
        key: 'weight',
        direction: 'desc',
    });
    const handleHoldingSort = useCallback((key: FundHoldingSortKey) => {
        setHoldingSort((current) => ({
            key,
            direction: current.key === key && current.direction === 'desc' ? 'asc' : 'desc',
        }));
    }, []);
    const displayablePositions = useMemo(
        () => positions.filter((position) => holdingAssetCategory(position)),
        [positions],
    );
    const activeDisplayablePositions = useMemo(
        () => displayablePositions.filter((position) => {
            const weight = holdingWeightValue(position);
            return weight != null && weight > 0;
        }),
        [displayablePositions],
    );
    const filteredPositions = useMemo(
        () => filterFundHoldings(displayablePositions, categoryFilter, changeFilter),
        [displayablePositions, categoryFilter, changeFilter],
    );
    const activeFilteredPositions = useMemo(
        () => filteredPositions.filter((position) => {
            const weight = holdingWeightValue(position);
            return weight != null && weight > 0;
        }),
        [filteredPositions],
    );
    const sortedFilteredPositions = useMemo(
        () => sortHoldingPositions(activeFilteredPositions, holdingSort.key, holdingSort.direction),
        [activeFilteredPositions, holdingSort],
    );
    const rankingPositions = useMemo(() => {
        const currentPositions = sortHoldingPositionsByWeight(
            filteredPositions.filter((position) => {
                const weight = holdingWeightValue(position);
                return weight != null && weight > 0;
            }),
            'desc',
        );
        const selectedKeys = new Set(currentPositions.map(holdingPositionKey));
        const removedPositions = sortHoldingPositions(
            filteredPositions.filter((position) => {
                if (selectedKeys.has(holdingPositionKey(position))) return false;
                const currentWeight = holdingWeightValue(position);
                const previousWeight = holdingWeightValue(position, 'previous_weight');
                return (currentWeight == null || currentWeight <= 0)
                    && previousWeight != null
                    && previousWeight > 0;
            }),
            'previous_weight',
            'desc',
        );
        return [...currentPositions, ...removedPositions];
    }, [filteredPositions]);
    const maxRankingWeight = Math.max(
        0,
        ...rankingPositions.flatMap((position) => [
            holdingWeightValue(position) || 0,
            holdingWeightValue(position, 'previous_weight') || 0,
        ]),
    );
    const totalCurrentRankingWeight = rankingPositions.reduce((sum, position) => {
        const weight = holdingWeightValue(position) || 0;
        return weight > 0 ? sum + weight : sum;
    }, 0);
    const totalPreviousRankingWeight = rankingPositions.reduce((sum, position) => {
        const weight = holdingWeightValue(position, 'previous_weight') || 0;
        return weight > 0 ? sum + weight : sum;
    }, 0);
    const previousRankByKey = useMemo(() => {
        const map = new Map<string, number>();
        sortHoldingPositions(
            filteredPositions.filter((position) => {
                const weight = holdingWeightValue(position, 'previous_weight');
                return weight != null && weight > 0;
            }),
            'previous_weight',
            'desc',
        ).forEach((position, index) => {
            map.set(holdingPositionKey(position), index + 1);
        });
        return map;
    }, [filteredPositions]);
    const groupedPositions = useMemo(
        () => FUND_HOLDING_GROUPS
            .map((group) => ({
                ...group,
                positions: sortedFilteredPositions.filter((position) => holdingAssetCategory(position) === group.key),
            }))
            .filter((group) => group.positions.length > 0),
        [sortedFilteredPositions],
    );
    const latestReport = latestHoldingReportMeta(holdings);
    const previousReport = previousHoldingReportMeta(holdings);
    const asOf = latestReport?.report_date || holdings?.source_metadata?.as_of || displayablePositions.find((position) => position.report_date)?.report_date || null;
    const portfolioEffect = holdings?.portfolio_effect || null;
    const holdingsQuality = holdings?.source_metadata?.holdings_quality || null;
    const currentMonthLabel = formatMonthLegendLabel(asOf, 'GÜNCEL');
    const previousMonthLabel = formatMonthLegendLabel(
        previousReport?.report_date || displayablePositions.find((position) => position.previous_report_date)?.previous_report_date,
        'ÖNCEKİ',
    );
    const normalizedCount = finiteNumber(holdingsQuality?.normalized_position_count);
    const fallbackCount = finiteNumber(holdingsQuality?.fallback_position_count);
    const invalidCount = finiteNumber(holdingsQuality?.invalid_position_count);
    const adjustedTotalWeight = finiteNumber(holdingsQuality?.adjusted_total_weight);
    const isGrossExposure = (holdingsQuality?.status === 'gross_exposure')
        || (adjustedTotalWeight != null && adjustedTotalWeight > 100.5);
    const qualityWarnings = [
        fallbackCount != null && fallbackCount > 0
            ? `${fallbackCount} ağırlık KAP satırındaki eksik yüzde nedeniyle yaklaşık değer olarak gösteriliyor.`
            : null,
        invalidCount != null && invalidCount > 0
            ? `${invalidCount} geçersiz KAP ağırlığı güvenlik nedeniyle gösterilmedi.`
            : null,
        normalizedCount != null && normalizedCount > 0
            ? `${normalizedCount} ağırlık ölçek hatası nedeniyle normalize edildi.`
            : null,
    ].filter((warning): warning is string => Boolean(warning));
    const qualityWarning = qualityWarnings.length ? qualityWarnings.join(' ') : null;
    const grossExposureWarning = isGrossExposure
        ? `Raporlanan toplam ağırlık ${formatAllocationWeight(adjustedTotalWeight)}. Brüt pozisyon (kaldıraç ya da KAP formatı) görünüyor; çubuklar oransal, yüzdeler ham değer.`
        : null;
    const hasPartialWarning = Boolean(holdings && displayablePositions.length && (holdings.status !== 'ok' || qualityWarning || grossExposureWarning));

    return (
        <section className="fund-holdings-panel" aria-label="Fon içeriği">
            <div className="fund-holdings-head">
                <div>
                    <h3>Fon İçeriği</h3>
                    <p>{asOf ? `${formatDate(asOf)} KAP portföy dağılım raporu` : 'KAP portföy dağılım raporu'}</p>
                </div>
                <div className="fund-holdings-head-actions">
                    <strong>{activeDisplayablePositions.length ? `${activeDisplayablePositions.length} varlık` : loading ? 'Yükleniyor' : '-'}</strong>
                </div>
            </div>

            {error ? (
                <div className="funds-state">{error}</div>
            ) : loading || (!holdings && !error) ? (
                <FinLoader message="KAP fon içeriği yükleniyor" />
            ) : displayablePositions.length ? (
                <>
                    {hasPartialWarning && (
                        <div className="fund-holdings-warning">
                            {grossExposureWarning || qualityWarning || 'KAP raporu kısmi parse edildi; bazı satırlar eksik olabilir.'}
                        </div>
                    )}
                    {portfolioEffect && (
                        <div className="fund-holdings-effect-summary" aria-label="KAP ağırlıklarıyla günlük tahmin">
                            <div className="fund-holdings-effect-card">
                                <span>Tahmini Günlük Getiri</span>
                                <strong className={pctClass(portfolioEffect.estimated_return_pct)}>
                                    {formatPct(portfolioEffect.estimated_return_pct)}
                                </strong>
                                <small>KAP ağırlıklarıyla</small>
                            </div>
                            <div className="fund-holdings-effect-card">
                                <span>Tahmini Günlük K/Z</span>
                                <strong className={pctClass(portfolioEffect.estimated_pnl_value)}>
                                    {formatSignedCompactCurrency(portfolioEffect.estimated_pnl_value)}
                                </strong>
                                <small>{portfolioEffect.as_of ? formatDate(portfolioEffect.as_of) : 'Günlük'}</small>
                            </div>
                            <div className="fund-holdings-effect-card">
                                <span>Fiyatlanan Ağırlık</span>
                                <strong>{formatAllocationWeight(portfolioEffect.priced_weight)}</strong>
                                <small>
                                    Eksik {formatAllocationWeight(portfolioEffect.missing_weight)}
                                </small>
                            </div>
                        </div>
                    )}
                    <div className="fund-holdings-toolbar">
                        <div className="fund-holdings-filters" aria-label="Varlık tipi filtresi">
                            {FUND_HOLDING_CATEGORY_FILTERS.map((item) => (
                                <button
                                    key={item.key}
                                    type="button"
                                    className={categoryFilter === item.key ? 'is-active' : ''}
                                    onClick={() => onCategoryFilterChange(item.key)}
                                >
                                    {item.label}
                                </button>
                            ))}
                        </div>
                        <div className="fund-holdings-filters fund-holdings-change-filters" aria-label="Değişim filtresi">
                            {FUND_HOLDING_CHANGE_FILTERS.map((item) => (
                                <button
                                    key={item.key}
                                    type="button"
                                    className={changeFilter === item.key ? 'is-active' : ''}
                                    onClick={() => onChangeFilterChange(item.key)}
                                >
                                    {item.label}
                                </button>
                            ))}
                        </div>
                    </div>
                    {rankingPositions.length > 0 && (
                        <div className="fund-holdings-rank-panel" aria-label="Güncel ağırlık sıralaması">
                            <div className="fund-holdings-rank-top">
                                <span className="fund-holdings-rank-title">
                                    Öne Çıkan Pozisyonlar ({Math.min(10, rankingPositions.length)}/{rankingPositions.length})
                                </span>
                                <div className="fund-holdings-rank-legend" aria-hidden="true">
                                    <span><i className="fund-holdings-rank-legend-current" />{currentMonthLabel}</span>
                                    <span><i className="fund-holdings-rank-legend-previous" />{previousMonthLabel}</span>
                                </div>
                            </div>
                            <div className="fund-holdings-rank-head" aria-hidden="true">
                                <span>#</span>
                                <span />
                                <span>Sembol</span>
                                <span>Güncel Ağırlık / Değişim</span>
                                <span>Önceki Sıra</span>
                            </div>
                            {rankingPositions.slice(0, 10).map((position, index) => {
                                const category = holdingAssetCategory(position) || 'local_equity';
                                const symbol = holdingLogoSymbol(position);
                                const logoKind = holdingLogoKind(category);
                                const detailClickable = position.detail_clickable !== false;
                                const stockTicker = category === 'local_equity' ? normalizeCompareSymbol(position.asset_code) : '';
                                const canOpenStock = Boolean(onOpenTicker && stockTicker && detailClickable);
                                const fundCode = category === 'fund' ? normalizeCompareSymbol(position.asset_code) : '';
                                const canOpenFund = Boolean(onOpenFund && fundCode && position.tefas_tradable && detailClickable);
                                const weight = holdingWeightValue(position) || 0;
                                const previousWeight = holdingWeightValue(position, 'previous_weight') || 0;
                                const currentDenominator = isGrossExposure && totalCurrentRankingWeight > 0
                                    ? totalCurrentRankingWeight
                                    : maxRankingWeight;
                                const previousDenominator = isGrossExposure && totalPreviousRankingWeight > 0
                                    ? totalPreviousRankingWeight
                                    : maxRankingWeight;
                                const widthPct = currentDenominator > 0 && weight > 0
                                    ? Math.max(3, Math.min(100, (weight / currentDenominator) * 100))
                                    : 0;
                                const previousWidthPct = previousDenominator > 0 && previousWeight > 0
                                    ? Math.max(3, Math.min(100, (previousWeight / previousDenominator) * 100))
                                    : 0;
                                const previousRank = previousRankByKey.get(holdingPositionKey(position));
                                const rankDelta = previousRank == null ? null : previousRank - (index + 1);
                                const rankTrend = rankDelta == null ? '' : rankDelta > 0 ? '▲' : rankDelta < 0 ? '▼' : '—';
                                const barWarning = holdingWeightWarning(position) || holdingWeightWarning(position, 'previous_weight');
                                const barTooltip = `${currentMonthLabel}: ${formatHoldingWeight(position)} · ${previousMonthLabel}: ${formatHoldingWeight(position, 'previous_weight')}${barWarning ? ` · ${barWarning}` : ''}`;
                                const status = holdingChangeStatus(position);
                                const statusIcon = status === 'new'
                                    ? <LogIn size={13} aria-label="Fona girdi" />
                                    : status === 'removed'
                                        ? <LogOut size={13} aria-label="Fondan çıktı" />
                                        : null;
                                const rankSymbolContent = (
                                    <>
                                        <SymbolLogo
                                            symbol={symbol}
                                            name={holdingLogoName(position)}
                                            kind={logoKind}
                                            logoUrl={position.logo_url}
                                            size="sm"
                                        />
                                        <strong>{holdingDisplayCode(position)}</strong>
                                    </>
                                );
                                return (
                                    <div className={`fund-holdings-rank-row${status === 'removed' ? ' is-removed' : ''}`} key={`rank-${holdingPositionKey(position)}`}>
                                        <span className="fund-holdings-rank-index">{index + 1}.</span>
                                        <span className={`fund-holdings-rank-flow${statusIcon ? ' is-active' : ''}`}>
                                            {statusIcon}
                                        </span>
                                        {canOpenStock ? (
                                            <Link
                                                to={toStockDetail(stockTicker)}
                                                className="fund-holdings-rank-symbol fund-holdings-rank-symbol-link"
                                                aria-label={`${stockTicker} hisse sayfasını aç`}
                                            >
                                                {rankSymbolContent}
                                            </Link>
                                        ) : canOpenFund ? (
                                            <Link
                                                to={toFundDetail(fundCode, 'overview')}
                                                className="fund-holdings-rank-symbol fund-holdings-rank-symbol-link"
                                                aria-label={`${fundCode} fonunu aç`}
                                            >
                                                {rankSymbolContent}
                                            </Link>
                                        ) : (
                                            <div className="fund-holdings-rank-symbol">{rankSymbolContent}</div>
                                        )}
                                        <div className="fund-holdings-rank-current">
                                            <div
                                                className="fund-holdings-rank-bars"
                                                data-tooltip={barTooltip}
                                                aria-label={barTooltip}
                                                tabIndex={0}
                                            >
                                                <i
                                                    className="fund-holdings-rank-bar-current"
                                                    style={{ '--holding-weight-width': `${widthPct}%` } as CSSProperties}
                                                />
                                                <i
                                                    className="fund-holdings-rank-bar-previous"
                                                    style={{ '--holding-weight-width': `${previousWidthPct}%` } as CSSProperties}
                                                />
                                            </div>
                                            <span className="fund-holdings-rank-weight"><HoldingWeightDisplay position={position} /></span>
                                            <span className={pctClass(position.weight_change)}>{renderWeightDelta(position.weight_change)}</span>
                                        </div>
                                        <span className="fund-holdings-rank-previous">
                                            {previousRank == null
                                                ? 'Yeni'
                                                : `#${previousRank} ${rankTrend}`}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                    <div className="fund-holdings-table-wrap">
                        <table className="fund-history-table fund-holdings-table">
                            <colgroup>
                                <col className="fund-holdings-col-asset" />
                                <col className="fund-holdings-col-price" />
                                <col className="fund-holdings-col-return" />
                                <col className="fund-holdings-col-pnl" />
                                <col className="fund-holdings-col-weight" />
                                <col className="fund-holdings-col-previous" />
                                <col className="fund-holdings-col-change" />
                                <col className="fund-holdings-col-history" />
                            </colgroup>
                            <thead>
                                <tr>
                                    <FundHoldingsSortHeader
                                        columnKey="asset"
                                        sortKey={holdingSort.key}
                                        sortDirection={holdingSort.direction}
                                        onSort={handleHoldingSort}
                                    />
                                    <FundHoldingsSortHeader
                                        columnKey="price"
                                        sortKey={holdingSort.key}
                                        sortDirection={holdingSort.direction}
                                        onSort={handleHoldingSort}
                                    />
                                    <FundHoldingsSortHeader
                                        columnKey="return_pct"
                                        sortKey={holdingSort.key}
                                        sortDirection={holdingSort.direction}
                                        onSort={handleHoldingSort}
                                    />
                                    <FundHoldingsSortHeader
                                        columnKey="estimated_pnl_value"
                                        sortKey={holdingSort.key}
                                        sortDirection={holdingSort.direction}
                                        onSort={handleHoldingSort}
                                    />
                                    <FundHoldingsSortHeader
                                        columnKey="weight"
                                        sortKey={holdingSort.key}
                                        sortDirection={holdingSort.direction}
                                        onSort={handleHoldingSort}
                                    />
                                    <FundHoldingsSortHeader
                                        columnKey="previous_weight"
                                        sortKey={holdingSort.key}
                                        sortDirection={holdingSort.direction}
                                        onSort={handleHoldingSort}
                                    />
                                    <FundHoldingsSortHeader
                                        columnKey="weight_change"
                                        sortKey={holdingSort.key}
                                        sortDirection={holdingSort.direction}
                                        onSort={handleHoldingSort}
                                    />
                                    <th>Geçmiş</th>
                                </tr>
                            </thead>
                            <tbody>
                                {groupedPositions.map((group) => (
                                    <Fragment key={group.key}>
                                        <tr className="fund-holdings-group-row">
                                            <td colSpan={8}>
                                                <div className="fund-holdings-group-label">
                                                    <span>{group.label}</span>
                                                    <strong>{group.positions.length}</strong>
                                                </div>
                                            </td>
                                        </tr>
                                        {group.positions.map((position) => {
                                            const category = holdingAssetCategory(position) || group.key;
                                            const symbol = holdingLogoSymbol(position);
                                            const logoKind = holdingLogoKind(category);
                                            const detailClickable = position.detail_clickable !== false;
                                            const stockTicker = category === 'local_equity' ? normalizeCompareSymbol(position.asset_code) : '';
                                            const canOpenStock = Boolean(onOpenTicker && stockTicker && detailClickable);
                                            const fundCode = category === 'fund' ? normalizeCompareSymbol(position.asset_code) : '';
                                            const canOpenFund = Boolean(onOpenFund && fundCode && position.tefas_tradable && detailClickable);
                                            const rowKey = `${position.asset_code || position.asset_name}-${position.report_date || ''}-${position.change_status || ''}`;
                                            const assetContent = (
                                                <>
                                                    <SymbolLogo
                                                        symbol={symbol}
                                                        name={holdingLogoName(position)}
                                                        kind={logoKind}
                                                        logoUrl={position.logo_url}
                                                        size="sm"
                                                    />
                                                    <span>
                                                        <strong title={position.asset_code || undefined}>{holdingDisplayCode(position)}</strong>
                                                        <small title={position.asset_name}>{position.asset_name}</small>
                                                    </span>
                                                </>
                                            );
                                            return (
                                                <tr key={rowKey}>
                                                    <td>
                                                        {canOpenStock ? (
                                                            <Link
                                                                to={toStockDetail(stockTicker)}
                                                                className="fund-holding-asset fund-holding-asset-link"
                                                                aria-label={`${stockTicker} hisse sayfasını aç`}
                                                            >
                                                                {assetContent}
                                                            </Link>
                                                        ) : canOpenFund ? (
                                                            <Link
                                                                to={toFundDetail(fundCode, 'overview')}
                                                                className="fund-holding-asset fund-holding-asset-link"
                                                                aria-label={`${fundCode} fonunu aç`}
                                                            >
                                                                {assetContent}
                                                            </Link>
                                                        ) : (
                                                            <div className="fund-holding-asset">{assetContent}</div>
                                                        )}
                                                    </td>
                                                    <FundLiveValueCell value={position.price} className="fund-holding-price">
                                                        {formatHoldingPrice(position.price, position.price_currency)}
                                                    </FundLiveValueCell>
                                                    <FundLiveValueCell value={position.return_pct} className={`fund-holding-return ${pctClass(position.return_pct)}`}>
                                                        {formatPct(position.return_pct)}
                                                    </FundLiveValueCell>
                                                    <FundLiveValueCell value={position.estimated_pnl_value} className={pctClass(position.estimated_pnl_value)}>
                                                        <span className="fund-holding-effect">
                                                            <strong>{formatSignedCompactCurrency(position.estimated_pnl_value)}</strong>
                                                            <small>
                                                                {position.estimated_fund_return_contribution_pct == null
                                                                    ? '-'
                                                                    : `${formatPct(position.estimated_fund_return_contribution_pct)} katkı`}
                                                            </small>
                                                        </span>
                                                    </FundLiveValueCell>
                                                    <td className="fund-holding-weight-current">
                                                        <HoldingWeightDisplay position={position} />
                                                    </td>
                                                    <td className="fund-holding-weight-previous">
                                                        <HoldingWeightDisplay position={position} weightKey="previous_weight" />
                                                    </td>
                                                    <td className={pctClass(position.weight_change)}>{renderWeightDelta(position.weight_change)}</td>
                                                    <td>
                                                        {position.source_report_url ? (
                                                            <a
                                                                className="fund-holdings-link"
                                                                href={position.source_report_url}
                                                                target="_blank"
                                                                rel="noreferrer"
                                                            >
                                                                KAP
                                                                <ChevronRight size={14} />
                                                            </a>
                                                        ) : (
                                                            <span className="fund-holdings-link is-disabled">-</span>
                                                        )}
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </Fragment>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    {!activeFilteredPositions.length && (
                        <div className="funds-state">Bu filtre için güncel portföyde varlık bulunamadı.</div>
                    )}
                </>
            ) : (
                <div className="funds-state">
                    {holdings?.message || 'KAP fon içeriği bu fon için bulunamadı.'}
                </div>
            )}
        </section>
    );
}
