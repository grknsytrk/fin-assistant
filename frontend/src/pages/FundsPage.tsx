import {
    memo,
    useCallback,
    useDeferredValue,
    useEffect,
    useMemo,
    useRef,
    useState,
    type CSSProperties,
    type PointerEvent as ReactPointerEvent,
} from 'react';
import {
    ArrowLeft,
    ArrowUpDown,
    BarChart3,
    BriefcaseBusiness,
    ExternalLink,
    LineChart,
    RefreshCw,
    Search,
    ShieldAlert,
    SlidersHorizontal,
} from 'lucide-react';
import { apiClient } from '../api/client';
import type {
    FundAllocation,
    FundAllocationsResponse,
    FundCategoriesResponse,
    FundDetail,
    FundPerformanceResponse,
    FundPricePoint,
    FundSummary,
    FundsResponse,
} from '../api/types';
import MarketsNavigation, { type MarketsNavigationSection } from '../components/MarketsNavigation';
import SymbolLogo from '../components/SymbolLogo';
import type { FundTab } from '../routing/routes';
import './FundsPage.css';

type FundSortKey =
    | 'fund_code'
    | 'name'
    | 'fund_type'
    | 'founder_company'
    | 'price'
    | 'daily_return'
    | 'risk_value'
    | 'aum'
    | 'as_of';

type FundsPageProps = {
    fundCode?: string | null;
    activeTab?: FundTab;
    onOpenFund: (fundCode: string, tab?: FundTab) => void;
    onTabChange: (tab: FundTab) => void;
    onBack: () => void;
    onNavigateSection: (section: MarketsNavigationSection) => void;
    onOpenTicker?: (ticker: string) => void;
};

const FUND_SORT_OPTIONS: Array<{ key: FundSortKey; label: string }> = [
    { key: 'fund_code', label: 'Kod' },
    { key: 'name', label: 'Ad' },
    { key: 'fund_type', label: 'Tür' },
    { key: 'founder_company', label: 'Kurucu' },
    { key: 'daily_return', label: 'Günlük getiri' },
    { key: 'risk_value', label: 'Risk' },
    { key: 'aum', label: 'Portföy' },
    { key: 'price', label: 'Fiyat' },
    { key: 'as_of', label: 'Tarih' },
];

const FUND_TABS: Array<{ key: FundTab; label: string; icon: typeof BarChart3 }> = [
    { key: 'overview', label: 'Genel Bakış', icon: BarChart3 },
    { key: 'allocation', label: 'Portföy Dağılımı', icon: BriefcaseBusiness },
    { key: 'history', label: 'Geçmiş Veriler', icon: LineChart },
];

type FundChartRange = '1w' | '1m' | '3m' | 'ytd' | '1y' | '5y' | 'custom';

const FUND_CHART_RANGES: Array<{ id: FundChartRange; label: string; title: string }> = [
    { id: '1w', label: '1H', title: '1 Hafta' },
    { id: '1m', label: '1A', title: '1 Ay' },
    { id: '3m', label: '3A', title: '3 Ay' },
    { id: 'ytd', label: 'YBB', title: 'Yılbaşından Beri' },
    { id: '1y', label: '1Y', title: '1 Yıl' },
    { id: '5y', label: '5Y', title: '5 Yıl' },
    { id: 'custom', label: 'Özel', title: 'Özel aralık' },
];

const FUND_DONUT_COLORS = ['#4f46e5', '#a3e635', '#818cf8', '#84cc16', '#f472b6', '#8b5cf6', '#c084fc', '#14b8a6'];

function useDebouncedValue<T>(value: T, delayMs: number): T {
    const [debounced, setDebounced] = useState(value);

    useEffect(() => {
        const timer = window.setTimeout(() => {
            setDebounced(value);
        }, delayMs);
        return () => window.clearTimeout(timer);
    }, [value, delayMs]);

    return debounced;
}

function formatDate(value: string | null | undefined): string {
    if (!value) return '-';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleDateString('tr-TR', { day: '2-digit', month: 'short', year: 'numeric' });
}

function isoDateDaysAgo(days: number): string {
    const date = new Date();
    date.setDate(date.getDate() - days);
    return date.toISOString().slice(0, 10);
}

function isoDateMonthsAgo(months: number): string {
    const date = new Date();
    date.setMonth(date.getMonth() - months);
    return date.toISOString().slice(0, 10);
}

function formatCurrency(value: number | null | undefined, currency = 'TRY'): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const prefix = currency === 'TRY' ? '₺' : `${currency} `;
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 4,
    })}`;
}

function formatCompactCurrency(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const abs = Math.abs(value);
    if (abs >= 1_000_000_000) {
        return `₺${(value / 1_000_000_000).toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })} mr`;
    }
    if (abs >= 1_000_000) {
        return `₺${(value / 1_000_000).toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })} mn`;
    }
    return formatCurrency(value);
}

function formatPct(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const sign = value > 0 ? '+' : '';
    return `% ${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function pctClass(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return 'funds-flat';
    if (value > 0) return 'funds-up';
    if (value < 0) return 'funds-down';
    return 'funds-flat';
}

function shouldRefreshSnapshot(payload: FundsResponse | null): boolean {
    if (!payload) return false;
    const total = payload.total_count ?? payload.count ?? 0;
    return total === 0 && (payload.degraded || payload.stale || payload.status === 'unavailable');
}

function compareFundRows(a: FundSummary, b: FundSummary, key: FundSortKey, order: 'asc' | 'desc'): number {
    const av = a[key];
    const bv = b[key];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    const direction = order === 'desc' ? -1 : 1;
    if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * direction;
    return String(av).localeCompare(String(bv), 'tr', { numeric: true }) * direction;
}

function returnBetween(latestPrice: number | null | undefined, basePrice: number | null | undefined): number | null {
    if (latestPrice == null || basePrice == null || !Number.isFinite(latestPrice) || !Number.isFinite(basePrice) || basePrice <= 0) {
        return null;
    }
    return ((latestPrice / basePrice) - 1) * 100;
}

function sortFundPoints(points: FundPerformanceResponse['points'] | undefined): FundPricePoint[] {
    return [...(points || [])]
        .filter((point) => Number.isFinite(Number(point.price)) && Number(point.price) > 0 && point.date)
        .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
}

function rangeStartDate(range: FundChartRange, endDateIso: string, customStartDate: string): string {
    const end = new Date(endDateIso);
    if (Number.isNaN(end.getTime())) return isoDateDaysAgo(30);
    const start = new Date(end);
    if (range === '1w') start.setDate(start.getDate() - 7);
    if (range === '1m') start.setMonth(start.getMonth() - 1);
    if (range === '3m') start.setMonth(start.getMonth() - 3);
    if (range === 'ytd') {
        start.setMonth(0, 1);
    }
    if (range === '1y') start.setFullYear(start.getFullYear() - 1);
    if (range === '5y') start.setFullYear(start.getFullYear() - 5);
    if (range === 'custom') return customStartDate || isoDateMonthsAgo(6);
    return start.toISOString().slice(0, 10);
}

function filterPointsForRange(
    points: FundPricePoint[],
    range: FundChartRange,
    endDateIso: string,
    customStartDate: string,
    customEndDate: string,
): FundPricePoint[] {
    const startIso = rangeStartDate(range, endDateIso, customStartDate);
    const effectiveEndIso = range === 'custom' ? (customEndDate || endDateIso) : endDateIso;
    const startTs = new Date(startIso).getTime();
    const endTs = new Date(effectiveEndIso).getTime();
    if (!Number.isFinite(startTs) || !Number.isFinite(endTs)) return points;
    return points.filter((point) => {
        const ts = new Date(point.date).getTime();
        return Number.isFinite(ts) && ts >= startTs && ts <= endTs;
    });
}

function hasUsableRangeCoverage(points: FundPricePoint[], startIso: string, endIso: string): boolean {
    const filtered = points.filter((point) => point.date >= startIso && point.date <= endIso);
    if (filtered.length < 2) return false;
    const latest = filtered[filtered.length - 1];
    const latestTs = new Date(latest.date).getTime();
    const endTs = new Date(endIso).getTime();
    return Number.isFinite(latestTs) && Number.isFinite(endTs) && latestTs >= endTs - 10 * 24 * 60 * 60 * 1000;
}

function formatChartDate(value: string | null | undefined, range: FundChartRange): string {
    if (!value) return '-';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return new Intl.DateTimeFormat('tr-TR', {
        day: '2-digit',
        month: 'short',
        ...(range === '1w' ? {} : { year: '2-digit' }),
    }).format(date);
}

function monthKey(value: string): string {
    return value.slice(0, 7);
}

function formatMonthLabel(value: string): string {
    const date = new Date(`${value}-01T00:00:00`);
    if (Number.isNaN(date.getTime())) return value;
    return new Intl.DateTimeFormat('tr-TR', { month: 'short', year: 'numeric' }).format(date);
}

type MonthlyReturn = {
    month: string;
    label: string;
    year: number;
    monthIndex: number;
    returnPct: number;
    startPrice: number;
    endPrice: number;
};

function monthlyReturnsFromPoints(points: FundPricePoint[]): MonthlyReturn[] {
    const monthEndPoints = new Map<string, FundPricePoint>();
    for (const point of points) {
        const key = monthKey(point.date);
        const current = monthEndPoints.get(key);
        if (!current || point.date > current.date) {
            monthEndPoints.set(key, point);
        }
    }
    const ordered = [...monthEndPoints.entries()].sort(([a], [b]) => a.localeCompare(b));
    const monthly: MonthlyReturn[] = [];
    for (let index = 1; index < ordered.length; index += 1) {
        const [month, endPoint] = ordered[index];
        const [, startPoint] = ordered[index - 1];
        const startPrice = Number(startPoint.price);
        const endPrice = Number(endPoint.price);
        const returnPct = returnBetween(endPrice, startPrice);
        if (returnPct == null) continue;
        const date = new Date(`${month}-01T00:00:00`);
        monthly.push({
            month,
            label: formatMonthLabel(month),
            year: date.getFullYear(),
            monthIndex: date.getMonth(),
            returnPct,
            startPrice,
            endPrice,
        });
    }
    return monthly;
}

function pointOnOrBefore(points: FundPerformanceResponse['points'], targetDate: Date): FundPerformanceResponse['points'][number] | null {
    const targetTs = targetDate.getTime();
    let best: FundPerformanceResponse['points'][number] | null = null;
    for (const point of points) {
        const price = Number(point.price);
        const ts = new Date(point.date).getTime();
        if (!Number.isFinite(price) || !Number.isFinite(ts) || ts > targetTs) continue;
        if (!best || new Date(point.date).getTime() > new Date(best.date).getTime()) {
            best = point;
        }
    }
    return best;
}

function periodReturnsFromPerformance(points: FundPerformanceResponse['points'] | undefined): FundSummary['period_returns'] | null {
    const usable = [...(points || [])]
        .filter((point) => Number.isFinite(Number(point.price)) && Number(point.price) > 0 && point.date)
        .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
    if (usable.length < 2) return null;

    const latest = usable[usable.length - 1];
    const latestDate = new Date(latest.date);
    const latestPrice = Number(latest.price);
    const target = (days: number) => {
        const value = new Date(latestDate);
        value.setDate(value.getDate() - days);
        return value;
    };
    const ytd = new Date(latestDate);
    ytd.setMonth(0, 1);

    return {
        '1w': returnBetween(latestPrice, pointOnOrBefore(usable, target(7))?.price),
        '1m': returnBetween(latestPrice, pointOnOrBefore(usable, target(30))?.price),
        '3m': returnBetween(latestPrice, pointOnOrBefore(usable, target(90))?.price),
        '6m': returnBetween(latestPrice, pointOnOrBefore(usable, target(180))?.price),
        ytd: returnBetween(latestPrice, pointOnOrBefore(usable, ytd)?.price),
        '1y': returnBetween(latestPrice, pointOnOrBefore(usable, target(365))?.price),
    };
}

type FundsTableProps = {
    rows: FundSummary[];
    sortKey: FundSortKey;
    sortOrder: 'asc' | 'desc';
    onSort: (key: FundSortKey) => void;
    onOpenFund: (fundCode: string, tab?: FundTab) => void;
};

const FundTableRow = memo(function FundTableRow({
    row,
    onOpenFund,
}: {
    row: FundSummary;
    onOpenFund: (fundCode: string, tab?: FundTab) => void;
}) {
    const handleOpen = useCallback(() => {
        onOpenFund(row.fund_code, 'overview');
    }, [onOpenFund, row.fund_code]);

    return (
        <tr onClick={handleOpen}>
            <td>
                <div className="funds-symbol">
                    <SymbolLogo
                        symbol={row.fund_code}
                        name={row.founder_company || row.manager_company || row.name}
                        kind="fund"
                        size="sm"
                    />
                    <span>
                        <strong>{row.fund_code}</strong>
                        <small>{row.name}</small>
                    </span>
                </div>
            </td>
            <td>{row.fund_type || '-'}</td>
            <td>{row.founder_company || row.manager_company || '-'}</td>
            <td className="funds-cell-right">{formatCurrency(row.price, row.currency)}</td>
            <td className={`funds-cell-right ${pctClass(row.daily_return)}`}>{formatPct(row.daily_return)}</td>
            <td className="funds-cell-right">{row.risk_value ?? '-'}</td>
            <td className="funds-cell-right">{formatCompactCurrency(row.aum)}</td>
            <td className="funds-cell-right">{formatDate(row.as_of)}</td>
        </tr>
    );
});

const FundsTable = memo(function FundsTable({
    rows,
    sortKey,
    sortOrder,
    onSort,
    onOpenFund,
}: FundsTableProps) {
    const sortLabel = sortOrder === 'asc' ? 'artan' : 'azalan';
    const renderSortHeader = (key: FundSortKey, label: string, align: 'left' | 'right' = 'left') => (
        <th className={align === 'right' ? 'funds-cell-right' : undefined} aria-sort={sortKey === key ? (sortOrder === 'asc' ? 'ascending' : 'descending') : 'none'}>
            <button
                type="button"
                className={`funds-th-sort${align === 'right' ? ' funds-th-sort-right' : ''}${sortKey === key ? ' active' : ''}`}
                onClick={() => onSort(key)}
                title={`${label} sütununa göre sırala`}
            >
                <span>{label}</span>
                <ArrowUpDown size={13} aria-hidden="true" />
                <span className="sr-only">{sortKey === key ? `Şu an ${sortLabel}` : ''}</span>
            </button>
        </th>
    );

    return (
        <div className="funds-table-wrap">
            <table className="funds-table">
                <thead>
                    <tr>
                        {renderSortHeader('fund_code', 'Fon')}
                        {renderSortHeader('fund_type', 'Tür')}
                        {renderSortHeader('founder_company', 'Kurucu')}
                        {renderSortHeader('price', 'Fiyat', 'right')}
                        {renderSortHeader('daily_return', 'Gün %', 'right')}
                        {renderSortHeader('risk_value', 'Risk', 'right')}
                        {renderSortHeader('aum', 'Portföy', 'right')}
                        {renderSortHeader('as_of', 'Tarih', 'right')}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row) => (
                        <FundTableRow key={row.fund_code} row={row} onOpenFund={onOpenFund} />
                    ))}
                </tbody>
            </table>
        </div>
    );
});

function formatAllocationWeight(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return `% ${value.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function FundPerformanceChart({
    fundCode,
    points,
    selectedRange,
    pendingRange,
    rangeError,
    customStartDate,
    customEndDate,
    onRangeSelect,
    onCustomStartDateChange,
    onCustomEndDateChange,
}: {
    fundCode: string;
    points: FundPricePoint[];
    selectedRange: FundChartRange;
    pendingRange: FundChartRange | null;
    rangeError: string | null;
    customStartDate: string;
    customEndDate: string;
    onRangeSelect: (range: FundChartRange) => void;
    onCustomStartDateChange: (value: string) => void;
    onCustomEndDateChange: (value: string) => void;
}) {
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const width = 860;
    const height = 330;
    const padding = { top: 24, right: 22, bottom: 34, left: 52 };
    const validPoints = points.filter((point) => Number.isFinite(Number(point.price)) && Number(point.price) > 0);
    const rangeReturn = validPoints.length >= 2
        ? returnBetween(Number(validPoints[validPoints.length - 1].price), Number(validPoints[0].price))
        : null;
    const color = rangeReturn == null || rangeReturn >= 0 ? '#22c55e' : '#ff4d5e';

    const controls = (
        <div className="fund-chart-toolbar">
            <div className="fund-chart-ranges">
                {FUND_CHART_RANGES.map((range) => (
                    <button
                        key={range.id}
                        type="button"
                        className={[
                            'fund-chart-range',
                            selectedRange === range.id ? 'is-active' : '',
                            pendingRange === range.id ? 'is-loading' : '',
                        ].filter(Boolean).join(' ')}
                        title={range.title}
                        aria-pressed={selectedRange === range.id}
                        onClick={() => onRangeSelect(range.id)}
                    >
                        {range.label}
                    </button>
                ))}
            </div>
            <div className={`fund-chart-return ${pctClass(rangeReturn)}`}>{formatPct(rangeReturn)}</div>
        </div>
    );

    if (validPoints.length < 2) {
        return (
            <section className="fund-chart-panel">
                {controls}
                {selectedRange === 'custom' && (
                    <div className="fund-custom-range">
                        <input type="date" value={customStartDate} onChange={(event) => onCustomStartDateChange(event.target.value)} />
                        <input type="date" value={customEndDate} onChange={(event) => onCustomEndDateChange(event.target.value)} />
                    </div>
                )}
                <div className="fund-chart-empty">{pendingRange ? 'Grafik verisi yükleniyor...' : 'Bu aralık için grafik verisi yok.'}</div>
                {rangeError && <div className="fund-chart-error">{rangeError}</div>}
            </section>
        );
    }

    const prices = validPoints.map((point) => Number(point.price));
    let minValue = Math.min(...prices);
    let maxValue = Math.max(...prices);
    const rawSpan = Math.max(0.01, maxValue - minValue);
    minValue -= rawSpan * 0.08;
    maxValue += rawSpan * 0.08;
    const span = Math.max(0.01, maxValue - minValue);
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const xFor = (index: number) => padding.left + (validPoints.length === 1 ? 0 : (index / (validPoints.length - 1)) * plotWidth);
    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;
    const pathData = validPoints.map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(index)} ${yFor(Number(point.price))}`).join(' ');
    const areaData = `${pathData} L ${xFor(validPoints.length - 1)} ${height - padding.bottom} L ${padding.left} ${height - padding.bottom} Z`;
    const gradientId = `fund-chart-area-${fundCode.replace(/[^a-zA-Z0-9]/g, '')}-${selectedRange}`;
    const tickIndexes = Array.from(new Set([0, Math.floor((validPoints.length - 1) / 2), validPoints.length - 1]));
    const yTicks = [0, 0.5, 1].map((ratio) => maxValue - ratio * span);

    const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        if (rect.width <= 0) return;
        const x = ((event.clientX - rect.left) / rect.width) * width;
        let closestIndex = 0;
        let minDiff = Infinity;
        for (let index = 0; index < validPoints.length; index += 1) {
            const diff = Math.abs(xFor(index) - x);
            if (diff < minDiff) {
                minDiff = diff;
                closestIndex = index;
            }
        }
        setHoverIndex(closestIndex);
    };
    const hoverPoint = hoverIndex == null ? null : validPoints[Math.max(0, Math.min(validPoints.length - 1, hoverIndex))];
    const hoverX = hoverPoint ? xFor(validPoints.indexOf(hoverPoint)) : null;
    const hoverY = hoverPoint ? yFor(Number(hoverPoint.price)) : null;
    const tooltipWidth = 152;
    const tooltipHeight = 58;
    const tooltipX = hoverX == null ? 0 : Math.min(Math.max(hoverX + 12, 8), width - tooltipWidth - 8);
    const tooltipY = hoverY == null ? 0 : Math.min(Math.max(hoverY - tooltipHeight / 2, padding.top), height - tooltipHeight - 8);

    return (
        <section className="fund-chart-panel">
            {controls}
            {selectedRange === 'custom' && (
                <div className="fund-custom-range">
                    <input type="date" value={customStartDate} onChange={(event) => onCustomStartDateChange(event.target.value)} />
                    <input type="date" value={customEndDate} onChange={(event) => onCustomEndDateChange(event.target.value)} />
                </div>
            )}
            <svg
                className="fund-detail-chart"
                viewBox={`0 0 ${width} ${height}`}
                preserveAspectRatio="none"
                role="img"
                aria-label={`${fundCode} fiyat grafiği`}
                onPointerMove={handlePointerMove}
                onPointerLeave={() => setHoverIndex(null)}
            >
                <defs>
                    <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor={color} stopOpacity="0.24" />
                        <stop offset="100%" stopColor={color} stopOpacity="0" />
                    </linearGradient>
                </defs>
                {yTicks.map((tick) => {
                    const y = yFor(tick);
                    return (
                        <g key={tick}>
                            <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="fund-chart-gridline" />
                            <text x={padding.left - 10} y={y + 4} className="fund-chart-axis" textAnchor="end">
                                {tick.toLocaleString('tr-TR', { maximumFractionDigits: 1 })}
                            </text>
                        </g>
                    );
                })}
                {tickIndexes.map((index) => (
                    <text
                        key={index}
                        x={xFor(index)}
                        y={height - 10}
                        className="fund-chart-axis"
                        textAnchor={index === 0 ? 'start' : index === validPoints.length - 1 ? 'end' : 'middle'}
                    >
                        {formatChartDate(validPoints[index].date, selectedRange)}
                    </text>
                ))}
                <path d={areaData} fill={`url(#${gradientId})`} />
                <path d={pathData} fill="none" stroke={color} strokeWidth="2.7" strokeLinecap="round" strokeLinejoin="round" />
                <circle cx={xFor(validPoints.length - 1)} cy={yFor(Number(validPoints[validPoints.length - 1].price))} r="4" fill={color} />
                {hoverPoint && hoverX != null && hoverY != null && (
                    <g>
                        <line x1={hoverX} x2={hoverX} y1={padding.top} y2={height - padding.bottom} className="fund-chart-hoverline" />
                        <circle cx={hoverX} cy={hoverY} r="4" fill={color} stroke="#0a0c0f" strokeWidth="2" />
                        <g transform={`translate(${tooltipX}, ${tooltipY})`}>
                            <rect width={tooltipWidth} height={tooltipHeight} rx="6" className="fund-chart-tooltip-bg" />
                            <text x="10" y="20" className="fund-chart-tooltip-muted">{formatDate(hoverPoint.date)}</text>
                            <text x="10" y="43" className="fund-chart-tooltip-value">{formatCurrency(hoverPoint.price, 'TRY')}</text>
                        </g>
                    </g>
                )}
            </svg>
            <div className="fund-chart-meta">
                <span>{formatDate(validPoints[0]?.date)}</span>
                <strong>{validPoints.length} veri noktası</strong>
                <span>{formatDate(validPoints[validPoints.length - 1]?.date)}</span>
            </div>
            {rangeError && <div className="fund-chart-error">{rangeError}</div>}
        </section>
    );
}

function FundAllocationSummary({
    allocations,
    loading,
}: {
    allocations: FundAllocation[];
    loading: boolean;
}) {
    const positiveAllocations = allocations.filter((item) => Number(item.weight) > 0).slice(0, FUND_DONUT_COLORS.length);
    const positiveTotal = positiveAllocations.reduce((sum, item) => sum + Number(item.weight || 0), 0);
    let cursor = 0;
    const gradient = positiveTotal > 0
        ? `conic-gradient(${positiveAllocations.map((item, index) => {
            const start = cursor;
            cursor += (Number(item.weight) / positiveTotal) * 100;
            return `${FUND_DONUT_COLORS[index % FUND_DONUT_COLORS.length]} ${start}% ${cursor}%`;
        }).join(', ')})`
        : undefined;

    return (
        <section className="fund-allocation-panel">
            <h2>Varlık Dağılımı</h2>
            {loading ? (
                <div className="funds-state">Dağılım verisi yükleniyor...</div>
            ) : allocations.length ? (
                <>
                    <div className="fund-allocation-donut" style={gradient ? { background: gradient } : undefined}>
                        <span />
                    </div>
                    <div className="fund-allocation-list">
                        {allocations.slice(0, 8).map((item, index) => (
                            <div key={`${item.allocation_type}-${index}`}>
                                <i style={{ background: FUND_DONUT_COLORS[index % FUND_DONUT_COLORS.length] }} />
                                <span>{item.label}</span>
                                <strong className={pctClass(item.weight)}>{formatAllocationWeight(item.weight)}</strong>
                            </div>
                        ))}
                    </div>
                </>
            ) : (
                <div className="funds-state">Fintables dağılım verisi bu fon için yok.</div>
            )}
        </section>
    );
}

function FundMonthlyHeatmap({ monthlyReturns }: { monthlyReturns: MonthlyReturn[] }) {
    const maxAbs = Math.max(1, ...monthlyReturns.map((item) => Math.abs(item.returnPct)));
    const byMonth = new Map(monthlyReturns.map((item) => [item.month, item]));
    const years = Array.from(new Set(monthlyReturns.map((item) => item.year))).sort((a, b) => a - b);
    const best = monthlyReturns.reduce<MonthlyReturn | null>((current, item) => (!current || item.returnPct > current.returnPct ? item : current), null);
    const worst = monthlyReturns.reduce<MonthlyReturn | null>((current, item) => (!current || item.returnPct < current.returnPct ? item : current), null);
    const positiveCount = monthlyReturns.filter((item) => item.returnPct > 0).length;
    const avg = monthlyReturns.length
        ? monthlyReturns.reduce((sum, item) => sum + item.returnPct, 0) / monthlyReturns.length
        : null;
    const monthNames = ['Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz', 'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara'];

    return (
        <section className="fund-heatmap-panel">
            <h2>Aylık Getiriler</h2>
            {monthlyReturns.length ? (
                <>
                    <div className="fund-heatmap">
                        <div className="fund-heatmap-months">
                            <span />
                            {monthNames.map((month) => <span key={month}>{month}</span>)}
                        </div>
                        {years.map((year) => (
                            <div className="fund-heatmap-row" key={year}>
                                <strong>{year}</strong>
                                {monthNames.map((_, index) => {
                                    const key = `${year}-${String(index + 1).padStart(2, '0')}`;
                                    const item = byMonth.get(key);
                                    const intensity = item ? Math.min(1, Math.abs(item.returnPct) / maxAbs) : 0;
                                    return (
                                        <span
                                            key={key}
                                            className={[
                                                'fund-heatmap-cell',
                                                item ? (item.returnPct >= 0 ? 'is-positive' : 'is-negative') : 'is-empty',
                                            ].join(' ')}
                                            style={{ '--heat-intensity': intensity.toFixed(2) } as CSSProperties}
                                            title={item ? `${item.label}: ${formatPct(item.returnPct)}` : `${monthNames[index]} ${year}: veri yok`}
                                        >
                                            {item ? item.returnPct.toLocaleString('tr-TR', { maximumFractionDigits: 1 }) : ''}
                                        </span>
                                    );
                                })}
                            </div>
                        ))}
                    </div>
                    <div className="fund-heatmap-stats">
                        <div><span>En iyi ay</span><strong>{best ? `${best.label} ${formatPct(best.returnPct)}` : '-'}</strong></div>
                        <div><span>En düşük ay</span><strong>{worst ? `${worst.label} ${formatPct(worst.returnPct)}` : '-'}</strong></div>
                        <div><span>Ay sayısı</span><strong>{monthlyReturns.length}</strong></div>
                        <div><span>Pozitif ay</span><strong>{positiveCount}</strong></div>
                        <div><span>Aylık ortalama</span><strong>{formatPct(avg)}</strong></div>
                    </div>
                </>
            ) : (
                <div className="funds-state">Aylık getiri için yeterli geçmiş veri yok.</div>
            )}
        </section>
    );
}

function FundHistoryTable({ points }: { points: FundPricePoint[] }) {
    const rows = [...points].sort((a, b) => b.date.localeCompare(a.date));
    return (
        <div className="fund-history-table-wrap">
            <table className="fund-history-table">
                <thead>
                    <tr>
                        <th>Tarih</th>
                        <th>Fiyat</th>
                        <th>Günlük getiri</th>
                        <th>Portföy</th>
                        <th>Yatırımcı</th>
                    </tr>
                </thead>
                <tbody>
                    {rows.map((point) => (
                        <tr key={point.date}>
                            <td>{formatDate(point.date)}</td>
                            <td>{formatCurrency(point.price, 'TRY')}</td>
                            <td className={pctClass(point.daily_return)}>{formatPct(point.daily_return)}</td>
                            <td>{formatCompactCurrency(point.aum)}</td>
                            <td>{point.investor_count?.toLocaleString('tr-TR') || '-'}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

export default function FundsPage({
    fundCode,
    activeTab = 'overview',
    onOpenFund,
    onTabChange,
    onBack,
    onNavigateSection,
    onOpenTicker,
}: FundsPageProps) {
    const [navCollapsed, setNavCollapsed] = useState(false);
    const [funds, setFunds] = useState<FundsResponse | null>(null);
    const [categories, setCategories] = useState<FundCategoriesResponse | null>(null);
    const [detail, setDetail] = useState<FundDetail | null>(null);
    const [performance, setPerformance] = useState<FundPerformanceResponse | null>(null);
    const [allocations, setAllocations] = useState<FundAllocationsResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [detailLoading, setDetailLoading] = useState(false);
    const [allocationLoading, setAllocationLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [refreshError, setRefreshError] = useState<string | null>(null);
    const [detailError, setDetailError] = useState<string | null>(null);
    const [chartRange, setChartRange] = useState<FundChartRange>('1m');
    const [pendingChartRange, setPendingChartRange] = useState<FundChartRange | null>(null);
    const [chartRangeError, setChartRangeError] = useState<string | null>(null);
    const [customStartDate, setCustomStartDate] = useState(isoDateMonthsAgo(6));
    const [customEndDate, setCustomEndDate] = useState(new Date().toISOString().slice(0, 10));
    const [searchTerm, setSearchTerm] = useState('');
    const [fundTypeFilter, setFundTypeFilter] = useState('');
    const [riskFilter, setRiskFilter] = useState('');
    const [sortKey, setSortKey] = useState<FundSortKey>('fund_code');
    const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
    const debouncedSearchTerm = useDebouncedValue(searchTerm, 160);
    const deferredSearchTerm = useDeferredValue(debouncedSearchTerm);
    const mountedRef = useRef(false);
    const autoRefreshAttemptedRef = useRef(false);
    const performanceRefreshAttemptedRef = useRef(new Set<string>());
    const allocationRefreshAttemptedRef = useRef(new Set<string>());

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
        };
    }, []);

    const refreshSnapshot = useCallback(async () => {
        setRefreshing(true);
        setRefreshError(null);
        try {
            const fundPayload = await apiClient.refreshFundsSnapshot();
            const categoryPayload = await apiClient.fundCategories();
            if (!mountedRef.current) return;
            setFunds(fundPayload);
            setCategories(categoryPayload);
        } catch (err) {
            if (!mountedRef.current) return;
            setRefreshError(err instanceof Error ? err.message : 'Fon snapshot yenilenemedi.');
        } finally {
            if (mountedRef.current) setRefreshing(false);
        }
    }, []);

    useEffect(() => {
        let alive = true;
        setLoading(true);
        setError(null);
        setRefreshError(null);
        Promise.all([apiClient.funds(), apiClient.fundCategories()])
            .then(([fundPayload, categoryPayload]) => {
                if (!alive) return;
                setFunds(fundPayload);
                setCategories(categoryPayload);
                if (shouldRefreshSnapshot(fundPayload) && !autoRefreshAttemptedRef.current) {
                    autoRefreshAttemptedRef.current = true;
                    void refreshSnapshot();
                }
            })
            .catch((err) => {
                if (!alive) return;
                setError(err instanceof Error ? err.message : 'Fon listesi alınamadı.');
            })
            .finally(() => {
                if (alive) setLoading(false);
            });
        return () => {
            alive = false;
        };
    }, [refreshSnapshot]);

    useEffect(() => {
        if (!fundCode) {
            setDetail(null);
            setDetailError(null);
            return;
        }
        let alive = true;
        setDetailLoading(true);
        setDetailError(null);
        apiClient
            .fundDetail(fundCode)
            .then((payload) => {
                if (alive) setDetail(payload);
            })
            .catch((err) => {
                if (alive) setDetailError(err instanceof Error ? err.message : 'Fon detayı alınamadı.');
            })
            .finally(() => {
                if (alive) setDetailLoading(false);
            });
        return () => {
            alive = false;
        };
    }, [fundCode]);

    useEffect(() => {
        if (!fundCode) return;
        setChartRange('1m');
        setPendingChartRange(null);
        setChartRangeError(null);
        setCustomStartDate(isoDateMonthsAgo(6));
        setCustomEndDate(new Date().toISOString().slice(0, 10));
    }, [fundCode]);

    useEffect(() => {
        if (!fundCode) {
            setPerformance(null);
            return;
        }
        let alive = true;
        const normalizedCode = fundCode.trim().toUpperCase();
        apiClient
            .fundPerformance(normalizedCode)
            .then((payload) => {
                if (!alive) return;
                setPerformance(payload);
                const shouldRefreshPerformance =
                    payload.status === 'unavailable' ||
                    payload.stale ||
                    (payload.points || []).filter((point) => Number.isFinite(Number(point.price)) && Number(point.price) > 0).length < 2;
                const refreshKey = `${normalizedCode}:1y`;
                if (shouldRefreshPerformance && !performanceRefreshAttemptedRef.current.has(refreshKey)) {
                    performanceRefreshAttemptedRef.current.add(refreshKey);
                    void apiClient
                        .refreshFundPerformance(normalizedCode, isoDateDaysAgo(370))
                        .then((freshPayload) => {
                            if (alive) setPerformance(freshPayload);
                        })
                        .catch(() => {
                            if (alive) setPerformance(payload);
                        });
                }
            })
            .catch(() => {
                if (alive) setPerformance(null);
            });
        return () => {
            alive = false;
        };
    }, [fundCode]);

    useEffect(() => {
        if (!fundCode) {
            setAllocations(null);
            return;
        }
        let alive = true;
        const normalizedCode = fundCode.trim().toUpperCase();
        setAllocationLoading(true);
        apiClient
            .fundAllocations(normalizedCode)
            .then((payload) => {
                if (!alive) return;
                setAllocations(payload);
                const shouldRefresh = payload.status === 'unavailable' || payload.stale;
                if (shouldRefresh && !allocationRefreshAttemptedRef.current.has(normalizedCode)) {
                    allocationRefreshAttemptedRef.current.add(normalizedCode);
                    void apiClient
                        .refreshFundAllocations(normalizedCode, detail?.as_of || undefined)
                        .then((freshPayload) => {
                            if (alive) setAllocations(freshPayload);
                        })
                        .catch(() => {
                            if (alive) setAllocations(payload);
                        });
                }
            })
            .catch(() => {
                if (alive) setAllocations(null);
            })
            .finally(() => {
                if (alive) setAllocationLoading(false);
            });
        return () => {
            alive = false;
        };
    }, [fundCode, detail?.as_of]);

    const filteredFunds = useMemo(() => {
        const rows = [...(funds?.rows || [])];
        const needle = deferredSearchTerm.trim().toLocaleLowerCase('tr-TR');
        const filtered = rows.filter((row) => {
            if (needle) {
                const haystack = `${row.fund_code} ${row.name} ${row.fund_type || ''} ${row.founder_company || ''}`.toLocaleLowerCase('tr-TR');
                if (!haystack.includes(needle)) return false;
            }
            if (fundTypeFilter && row.fund_type !== fundTypeFilter) return false;
            if (riskFilter && String(row.risk_value ?? '') !== riskFilter) return false;
            return true;
        });
        filtered.sort((a, b) => compareFundRows(a, b, sortKey, sortOrder));
        return filtered;
    }, [funds, deferredSearchTerm, fundTypeFilter, riskFilter, sortKey, sortOrder]);

    const selectedFund = detail || filteredFunds.find((row) => row.fund_code === fundCode) || null;
    const performancePoints = useMemo(() => sortFundPoints(performance?.points), [performance]);
    const chartEndDate = chartRange === 'custom'
        ? (customEndDate || selectedFund?.as_of || new Date().toISOString().slice(0, 10))
        : (selectedFund?.as_of || performance?.as_of || new Date().toISOString().slice(0, 10));
    const chartPoints = useMemo(
        () => filterPointsForRange(performancePoints, chartRange, chartEndDate, customStartDate, customEndDate),
        [performancePoints, chartRange, chartEndDate, customStartDate, customEndDate],
    );
    const monthlyReturns = useMemo(() => monthlyReturnsFromPoints(performancePoints), [performancePoints]);
    const allocationRows = useMemo(
        () => [...(allocations?.allocations || [])]
            .filter((item) => Number.isFinite(Number(item.weight)))
            .sort((a, b) => Math.abs(Number(b.weight || 0)) - Math.abs(Number(a.weight || 0))),
        [allocations],
    );
    const detailPeriodReturns = useMemo(
        () => periodReturnsFromPerformance(performance?.points) || selectedFund?.period_returns || {},
        [performance, selectedFund],
    );
    const visibleFundTypes = categories?.fund_types.length ? categories.fund_types : Array.from(new Set((funds?.rows || []).map((row) => row.fund_type).filter(Boolean))) as string[];
    const visibleRiskValues = categories?.risk_values.length ? categories.risk_values : Array.from(new Set((funds?.rows || []).map((row) => row.risk_value).filter((value): value is number => typeof value === 'number'))).sort((a, b) => a - b);
    const setTableSort = useCallback((key: FundSortKey) => {
        if (sortKey === key) {
            setSortOrder((currentOrder) => (currentOrder === 'asc' ? 'desc' : 'asc'));
            return;
        }
        setSortKey(key);
        setSortOrder('asc');
    }, [sortKey]);
    const refreshChartRange = useCallback((range: FundChartRange, nextCustomStart = customStartDate, nextCustomEnd = customEndDate) => {
        if (!fundCode) return;
        const normalizedCode = fundCode.trim().toUpperCase();
        const endIso = range === 'custom'
            ? (nextCustomEnd || chartEndDate)
            : (selectedFund?.as_of || performance?.as_of || new Date().toISOString().slice(0, 10));
        const startIso = rangeStartDate(range, endIso, nextCustomStart);
        if (hasUsableRangeCoverage(performancePoints, startIso, endIso)) {
            setChartRange(range);
            setChartRangeError(null);
            return;
        }
        setChartRange(range);
        setPendingChartRange(range);
        setChartRangeError(null);
        apiClient
            .refreshFundPerformance(normalizedCode, startIso, endIso)
            .then((payload) => {
                setPerformance(payload);
            })
            .catch((err) => {
                setChartRangeError(err instanceof Error ? err.message : 'Grafik verisi alınamadı.');
            })
            .finally(() => {
                setPendingChartRange(null);
            });
    }, [chartEndDate, customEndDate, customStartDate, fundCode, performance?.as_of, performancePoints, selectedFund?.as_of]);

    const handleChartRangeSelect = useCallback((range: FundChartRange) => {
        refreshChartRange(range);
    }, [refreshChartRange]);

    const handleCustomStartDateChange = useCallback((value: string) => {
        setCustomStartDate(value);
        refreshChartRange('custom', value, customEndDate);
    }, [customEndDate, refreshChartRange]);

    const handleCustomEndDateChange = useCallback((value: string) => {
        setCustomEndDate(value);
        refreshChartRange('custom', customStartDate, value);
    }, [customStartDate, refreshChartRange]);

    return (
        <div className={`mn-layout${navCollapsed ? ' mn-nav-collapsed' : ''}`}>
            <MarketsNavigation
                collapsed={navCollapsed}
                activeSection="funds"
                onCollapsedChange={setNavCollapsed}
                onSectionChange={onNavigateSection}
                onSelectTicker={onOpenTicker}
            />
            <div className="funds-workspace">
                <div className="funds-page">
                    {!fundCode ? (
                        <>
                            <header className="funds-header">
                                <div>
                                    <span className="funds-kicker">Fintables snapshot</span>
                                    <h1>Fonlar</h1>
                                    <p>Yatırım fonları, fiyat/getiri verisi ve kaynak durumuyla tek listede.</p>
                                </div>
                                <div className="funds-header-stats">
                                    <div>
                                        <span>Fon</span>
                                        <strong>{funds?.total_count ?? funds?.count ?? 0}</strong>
                                    </div>
                                    <div>
                                        <span>Kaynak</span>
                                        <strong>{funds?.source?.toUpperCase() || 'FINTABLES'}</strong>
                                    </div>
                                    <div>
                                        <span>Tarih</span>
                                        <strong>{formatDate(funds?.as_of)}</strong>
                                    </div>
                                </div>
                            </header>

                            {(funds?.stale || funds?.degraded || funds?.warnings?.length) && (
                                <div className="funds-source-warning">
                                    <ShieldAlert size={17} aria-hidden="true" />
                                    <span>
                                        {refreshError
                                            ? `Fon snapshot yenilenemedi: ${refreshError}`
                                            : refreshing
                                              ? 'Fon snapshot Fintables üzerinden yenileniyor.'
                                              : funds?.degraded
                                            ? 'Fon snapshot cache boş veya kullanılamıyor.'
                                            : 'Fon snapshot stale cache üzerinden gösteriliyor.'}
                                    </span>
                                    <button
                                        type="button"
                                        className="funds-refresh-button"
                                        onClick={refreshSnapshot}
                                        disabled={refreshing}
                                        title="Fon snapshot yenile"
                                    >
                                        <RefreshCw size={15} aria-hidden="true" className={refreshing ? 'funds-spin' : undefined} />
                                        {refreshing ? 'Yenileniyor' : 'Yenile'}
                                    </button>
                                </div>
                            )}

                            <section className="funds-panel">
                                <div className="funds-toolbar">
                                    <label className="funds-search">
                                        <Search size={16} aria-hidden="true" />
                                        <input
                                            value={searchTerm}
                                            onChange={(event) => setSearchTerm(event.target.value)}
                                            placeholder="Fon kodu, ad veya kurucu ara"
                                        />
                                    </label>
                                    <div className="funds-filter">
                                        <SlidersHorizontal size={16} aria-hidden="true" />
                                        <select value={fundTypeFilter} onChange={(event) => setFundTypeFilter(event.target.value)}>
                                            <option value="">Tüm türler</option>
                                            {visibleFundTypes.map((item) => (
                                                <option key={item} value={item}>{item}</option>
                                            ))}
                                        </select>
                                        <select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)}>
                                            <option value="">Tüm riskler</option>
                                            {visibleRiskValues.map((item) => (
                                                <option key={item} value={String(item)}>Risk {item}</option>
                                            ))}
                                        </select>
                                        <select value={sortKey} onChange={(event) => setSortKey(event.target.value as FundSortKey)}>
                                            {FUND_SORT_OPTIONS.map((option) => (
                                                <option key={option.key} value={option.key}>{option.label}</option>
                                            ))}
                                        </select>
                                        <button
                                            type="button"
                                            onClick={() => setSortOrder((order) => (order === 'asc' ? 'desc' : 'asc'))}
                                            className="funds-sort-order"
                                        >
                                            {sortOrder === 'asc' ? 'Artan' : 'Azalan'}
                                        </button>
                                    </div>
                                </div>

                                {loading ? (
                                    <div className="funds-state">Fon listesi yükleniyor...</div>
                                ) : error ? (
                                    <div className="funds-state funds-state-error">{error}</div>
                                ) : filteredFunds.length === 0 ? (
                                    <div className="funds-state">Fon bulunamadı.</div>
                                ) : (
                                    <FundsTable
                                        rows={filteredFunds}
                                        sortKey={sortKey}
                                        sortOrder={sortOrder}
                                        onSort={setTableSort}
                                        onOpenFund={onOpenFund}
                                    />
                                )}
                            </section>
                        </>
                    ) : (
                        <section className="fund-detail">
                            <button type="button" className="fund-back-button" onClick={onBack}>
                                <ArrowLeft size={16} aria-hidden="true" />
                                Fon listesi
                            </button>

                            {detailLoading ? (
                                <div className="funds-state">Fon detayı yükleniyor...</div>
                            ) : detailError ? (
                                <div className="funds-state funds-state-error">{detailError}</div>
                            ) : selectedFund ? (
                                <>
                                    <header className="fund-detail-hero">
                                        <div className="fund-detail-title">
                                            <SymbolLogo
                                                symbol={selectedFund.fund_code}
                                                name={selectedFund.founder_company || selectedFund.manager_company || selectedFund.name}
                                                kind="fund"
                                                size="lg"
                                            />
                                            <div>
                                                <span className="funds-kicker">{selectedFund.fund_code}</span>
                                                <h1>{selectedFund.name}</h1>
                                                <p>{selectedFund.fund_type || 'Fon'} · {selectedFund.founder_company || selectedFund.manager_company || 'Kurum bilgisi yok'}</p>
                                            </div>
                                        </div>
                                        <div className="fund-detail-actions">
                                            {detail?.fintables_url && (
                                                <a href={detail.fintables_url} target="_blank" rel="noreferrer">
                                                    Fintables
                                                    <ExternalLink size={14} aria-hidden="true" />
                                                </a>
                                            )}
                                        </div>
                                    </header>

                                    <div className="fund-metrics-grid">
                                        <div><span>Fiyat</span><strong>{formatCurrency(selectedFund.price, selectedFund.currency)}</strong></div>
                                        <div><span>Günlük getiri</span><strong className={pctClass(selectedFund.daily_return)}>{formatPct(selectedFund.daily_return)}</strong></div>
                                        <div><span>Risk</span><strong>{selectedFund.risk_value ?? '-'}</strong></div>
                                        <div><span>Portföy</span><strong>{formatCompactCurrency(selectedFund.aum)}</strong></div>
                                        <div><span>Yatırımcı</span><strong>{selectedFund.investor_count?.toLocaleString('tr-TR') || '-'}</strong></div>
                                        <div><span>Kaynak tarihi</span><strong>{formatDate(selectedFund.as_of)}</strong></div>
                                    </div>

                                    <nav className="fund-tabs">
                                        {FUND_TABS.map((tab) => {
                                            const Icon = tab.icon;
                                            return (
                                                <button
                                                    key={tab.key}
                                                    type="button"
                                                    className={activeTab === tab.key ? 'active' : ''}
                                                    onClick={() => onTabChange(tab.key)}
                                                >
                                                    <Icon size={16} aria-hidden="true" />
                                                    {tab.label}
                                                </button>
                                            );
                                        })}
                                    </nav>

                                    {activeTab === 'overview' && (
                                        <div className="fund-overview-stack">
                                            <div className="fund-overview-grid">
                                                <FundPerformanceChart
                                                    fundCode={selectedFund.fund_code}
                                                    points={chartPoints}
                                                    selectedRange={chartRange}
                                                    pendingRange={pendingChartRange}
                                                    rangeError={chartRangeError}
                                                    customStartDate={customStartDate}
                                                    customEndDate={customEndDate}
                                                    onRangeSelect={handleChartRangeSelect}
                                                    onCustomStartDateChange={handleCustomStartDateChange}
                                                    onCustomEndDateChange={handleCustomEndDateChange}
                                                />
                                                <FundAllocationSummary allocations={allocationRows} loading={allocationLoading && !allocations} />
                                            </div>
                                            <div className="fund-return-strip">
                                                <div><span>1H</span><strong className={pctClass(detailPeriodReturns?.['1w'])}>{formatPct(detailPeriodReturns?.['1w'])}</strong></div>
                                                <div><span>1A</span><strong className={pctClass(detailPeriodReturns?.['1m'])}>{formatPct(detailPeriodReturns?.['1m'])}</strong></div>
                                                <div><span>3A</span><strong className={pctClass(detailPeriodReturns?.['3m'])}>{formatPct(detailPeriodReturns?.['3m'])}</strong></div>
                                                <div><span>6A</span><strong className={pctClass(detailPeriodReturns?.['6m'])}>{formatPct(detailPeriodReturns?.['6m'])}</strong></div>
                                                <div><span>YBB</span><strong className={pctClass(detailPeriodReturns?.ytd)}>{formatPct(detailPeriodReturns?.ytd)}</strong></div>
                                                <div><span>1Y</span><strong className={pctClass(detailPeriodReturns?.['1y'])}>{formatPct(detailPeriodReturns?.['1y'])}</strong></div>
                                            </div>
                                            <FundMonthlyHeatmap monthlyReturns={monthlyReturns} />
                                        </div>
                                    )}

                                    {activeTab === 'allocation' && (
                                        <div className="fund-detail-panel">
                                            <h2>Portföy Dağılımı</h2>
                                            {allocationRows.length ? (
                                                <div className="fund-allocation-detail">
                                                    <FundAllocationSummary allocations={allocationRows} loading={allocationLoading && !allocations} />
                                                    <div className="fund-allocation-table-wrap">
                                                        <table className="fund-history-table">
                                                            <thead>
                                                                <tr>
                                                                    <th>Varlık</th>
                                                                    <th>Kod</th>
                                                                    <th>Ağırlık</th>
                                                                    <th>Tarih</th>
                                                                </tr>
                                                            </thead>
                                                            <tbody>
                                                                {allocationRows.map((item) => (
                                                                    <tr key={item.allocation_type}>
                                                                        <td>{item.label}</td>
                                                                        <td>{item.allocation_type.toUpperCase()}</td>
                                                                        <td className={pctClass(item.weight)}>{formatAllocationWeight(item.weight)}</td>
                                                                        <td>{formatDate(item.report_date)}</td>
                                                                    </tr>
                                                                ))}
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                </div>
                                            ) : (
                                                <div className="funds-state">
                                                    {allocationLoading ? 'Dağılım verisi yükleniyor...' : 'Fintables dağılım verisi bu fon için yok.'}
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {activeTab === 'history' && (
                                        <div className="fund-detail-panel">
                                            <h2>Geçmiş Veriler</h2>
                                            {performancePoints.length ? (
                                                <FundHistoryTable points={performancePoints} />
                                            ) : (
                                                <div className="funds-state">Geçmiş fiyat verisi henüz hazır değil.</div>
                                            )}
                                        </div>
                                    )}
                                </>
                            ) : (
                                <div className="funds-state">Fon bulunamadı.</div>
                            )}
                        </section>
                    )}
                </div>
            </div>
        </div>
    );
}
