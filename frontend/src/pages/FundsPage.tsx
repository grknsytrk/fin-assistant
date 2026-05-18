import {
    memo,
    useCallback,
    useDeferredValue,
    useEffect,
    useMemo,
    useRef,
    useState,
    type CSSProperties,
    type Dispatch,
    type PointerEvent as ReactPointerEvent,
    type RefObject,
    type SetStateAction,
} from 'react';
import {
    ArrowLeft,
    ArrowUpDown,
    BarChart3,
    BriefcaseBusiness,
    ChevronDown,
    ChevronRight,
    Database,
    LineChart,
    RefreshCw,
    Scale,
    Search,
    ShieldAlert,
    SlidersHorizontal,
    Star,
    X,
} from 'lucide-react';
import { apiClient } from '../api/client';
import type {
    FundAllocation,
    FundAllocationHistoryDay,
    FundAllocationsHistoryResponse,
    FundAllocationsResponse,
    FundCategoriesResponse,
    FundDetail,
    FundPerformanceResponse,
    FundPricePoint,
    FundSummary,
    FundYieldSummaryResponse,
    FundsResponse,
    FxQuote,
    MarketComparisonHistoryResponse,
    MarketIndexListRow,
    MarketStockRow,
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

type FundHistorySubtab = 'prices' | 'allocation';

const FUND_HISTORY_TABS: Array<{ key: FundHistorySubtab; label: string; icon: typeof LineChart }> = [
    { key: 'prices', label: 'Fiyat / Yatırımcı', icon: LineChart },
    { key: 'allocation', label: 'Fon Dağılımı', icon: BriefcaseBusiness },
];

type FundChartRange = 'all' | '1w' | '1m' | '3m' | '6m' | 'ytd' | '1y' | '5y' | 'custom';

const DETAIL_RETURN_PERIODS = [
    { key: '1w', label: '1H' },
    { key: '1m', label: '1A' },
    { key: '3m', label: '3A' },
    { key: '6m', label: '6A' },
    { key: 'ytd', label: 'YBB' },
    { key: '1y', label: '1Y' },
] as const;

type ComparisonPeriod = typeof DETAIL_RETURN_PERIODS[number]['key'];
type ComparisonAssetKind = 'fund' | 'stock' | 'index' | 'fx';
type FundComparisonControlSurface = 'chart' | 'returns';

type ComparisonAsset = {
    id: string;
    kind: ComparisonAssetKind;
    symbol: string;
    displaySymbol?: string;
    label: string;
    logoName?: string | null;
    logoUrl?: string | null;
    returns: Partial<Record<ComparisonPeriod, number | null>>;
};

type FundComparisonState = {
    baseAsset: ComparisonAsset | null;
    selectedAssets: ComparisonAsset[];
    selectedIds: string[];
    period: ComparisonPeriod;
    setPeriod: Dispatch<SetStateAction<ComparisonPeriod>>;
    query: string;
    setQuery: Dispatch<SetStateAction<string>>;
    searchOpen: boolean;
    setSearchOpen: Dispatch<SetStateAction<boolean>>;
    searchSurface: FundComparisonControlSurface | null;
    searchResults: ComparisonAsset[];
    addAsset: (asset: ComparisonAsset) => void;
    removeAsset: (assetId: string) => void;
    openSearch: (surface?: FundComparisonControlSurface) => void;
    fundSearchLoading: boolean;
    stocksLoading: boolean;
    indicesLoading: boolean;
    fxLoading: boolean;
    inputRef: RefObject<HTMLInputElement | null>;
};

type FundChartComparisonSeries = {
    asset: ComparisonAsset;
    points: Array<{ date: string; value: number }>;
    source?: string | null;
    error?: string | null;
};

type ComparisonReturnSource = {
    change_pct?: number | null;
    return_1w_pct?: number | null;
    return_1m_pct?: number | null;
    return_3m_pct?: number | null;
    return_6m_pct?: number | null;
    return_ytd_pct?: number | null;
    return_1y_pct?: number | null;
};

const FUND_DONUT_COLORS = ['#4f46e5', '#a3e635', '#818cf8', '#84cc16', '#f472b6', '#8b5cf6', '#c084fc', '#14b8a6'];
const DEFAULT_COMPARISON_IDS = ['index:XU100', 'index:XU030'];
const FUND_CHART_COMPARISON_COLORS = ['#22c55e', '#60a5fa', '#f59e0b', '#f472b6', '#a78bfa', '#14b8a6', '#f97316', '#e879f9'];

const STOCK_SEARCH_LABELS: Record<string, string[]> = {
    ASTOR: ['Astor Enerji A.Ş.'],
    BIMAS: ['BİM Birleşik Mağazalar A.Ş.', 'BIM'],
    MGROS: ['Migros Ticaret A.Ş.', 'Migros', 'MIGROS'],
    ORGE: ['Orge Enerji Elektrik Taahhüt A.Ş.'],
    SOKM: ['Şok Marketler Ticaret A.Ş.', 'SOK'],
    TAVHL: ['TAV Havalimanları Holding A.Ş.', 'TAV'],
    YEOTK: ['YEO Teknoloji Enerji ve Endüstri A.Ş.'],
};

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

function formatQuotePrice(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return value.toLocaleString('tr-TR', {
        minimumFractionDigits: 4,
        maximumFractionDigits: 6,
    });
}

function formatYieldBounds(
    period: FundYieldSummaryResponse['periods'][typeof DETAIL_RETURN_PERIODS[number]['key']] | undefined,
    currency = 'TRY',
): string {
    if (!period) return '';
    const low = formatCurrency(period.low, currency);
    const high = formatCurrency(period.high, currency);
    if (low === '-' && high === '-') return '';
    return `D ${low} / Y ${high}`;
}

function formatPct(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const sign = value > 0 ? '+' : '';
    return `% ${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatSignedInteger(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '';
    const sign = value > 0 ? '+' : '';
    return `${sign}${Math.trunc(value).toLocaleString('tr-TR')}`;
}

function pctClass(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return 'funds-flat';
    if (value > 0) return 'funds-up';
    if (value < 0) return 'funds-down';
    return 'funds-flat';
}

function normalizeCompareSymbol(value: string | null | undefined): string {
    return String(value || '').trim().toUpperCase().replace(/\s+/g, '');
}

function normalizeCompareSearch(value: string | null | undefined): string {
    return String(value || '')
        .toLocaleUpperCase('tr-TR')
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .replace(/[^A-Z0-9]+/g, ' ')
        .trim();
}

function compactCompareSearch(value: string | null | undefined): string {
    return normalizeCompareSearch(value).replace(/\s+/g, '');
}

function comparisonAssetId(kind: ComparisonAssetKind, symbol: string): string {
    return `${kind}:${normalizeCompareSymbol(symbol)}`;
}

function periodReturnFromAsset(asset: Pick<ComparisonAsset, 'returns'>, period: ComparisonPeriod): number | null {
    const value = asset.returns[period];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function marketReturnForPeriod(
    row: ComparisonReturnSource,
    period: ComparisonPeriod,
): number | null {
    const map: Record<ComparisonPeriod, keyof typeof row> = {
        '1w': 'return_1w_pct',
        '1m': 'return_1m_pct',
        '3m': 'return_3m_pct',
        '6m': 'return_6m_pct',
        ytd: 'return_ytd_pct',
        '1y': 'return_1y_pct',
    };
    const value = row[map[period]];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function returnsFromMarketRow(
    row: ComparisonReturnSource,
): Partial<Record<ComparisonPeriod, number | null>> {
    return {
        '1w': marketReturnForPeriod(row, '1w'),
        '1m': marketReturnForPeriod(row, '1m'),
        '3m': marketReturnForPeriod(row, '3m'),
        '6m': marketReturnForPeriod(row, '6m'),
        ytd: marketReturnForPeriod(row, 'ytd'),
        '1y': marketReturnForPeriod(row, '1y'),
    };
}

function fundToComparisonAsset(row: FundSummary): ComparisonAsset {
    const symbol = normalizeCompareSymbol(row.fund_code);
    return {
        id: comparisonAssetId('fund', symbol),
        kind: 'fund',
        symbol,
        label: row.name,
        logoName: row.founder_company || row.manager_company || row.name,
        returns: {
            '1w': row.period_returns?.['1w'] ?? null,
            '1m': row.period_returns?.['1m'] ?? null,
            '3m': row.period_returns?.['3m'] ?? null,
            '6m': row.period_returns?.['6m'] ?? null,
            ytd: row.period_returns?.ytd ?? null,
            '1y': row.period_returns?.['1y'] ?? null,
        },
    };
}

function stockToComparisonAsset(row: MarketStockRow): ComparisonAsset {
    const symbol = normalizeCompareSymbol(row.company);
    const label = STOCK_SEARCH_LABELS[symbol]?.[0] || 'Hisse';
    return {
        id: comparisonAssetId('stock', symbol),
        kind: 'stock',
        symbol,
        label,
        logoName: label,
        logoUrl: row.logo_url,
        returns: returnsFromMarketRow(row),
    };
}

function indexToComparisonAsset(row: MarketIndexListRow): ComparisonAsset {
    const symbol = normalizeCompareSymbol(row.symbol);
    return {
        id: comparisonAssetId('index', symbol),
        kind: 'index',
        symbol,
        label: row.label,
        logoName: row.label,
        returns: returnsFromMarketRow({
            change_pct: row.change_pct,
            return_1w_pct: row.return_1w_pct,
            return_1m_pct: row.return_1m_pct,
            return_3m_pct: row.return_3m_pct,
            return_6m_pct: row.return_6m_pct,
            return_ytd_pct: row.return_ytd_pct,
            return_1y_pct: row.return_1y_pct,
        }),
    };
}

function fxToComparisonAsset(row: FxQuote): ComparisonAsset | null {
    const rawSymbol = String(row.symbol || '').trim().toUpperCase();
    if (!rawSymbol.endsWith('/TRY')) return null;
    const returns = returnsFromMarketRow(row);
    const hasReturn = Object.values(returns).some((value) => typeof value === 'number' && Number.isFinite(value));
    if (!hasReturn) return null;
    const displaySymbol = rawSymbol.replace('/', '');
    return {
        id: comparisonAssetId('fx', rawSymbol),
        kind: 'fx',
        symbol: rawSymbol,
        displaySymbol,
        label: row.label,
        logoName: row.label,
        logoUrl: row.logo_url,
        returns,
    };
}

function comparisonAssetKindLabel(kind: ComparisonAssetKind): string {
    if (kind === 'fund') return 'Fon';
    if (kind === 'stock') return 'Hisse';
    if (kind === 'fx') return 'Döviz';
    return 'Endeks';
}

function comparisonLogoKind(kind: ComparisonAssetKind): 'fund' | 'stock' | 'index' | 'fx' {
    if (kind === 'fund') return 'fund';
    if (kind === 'stock') return 'stock';
    if (kind === 'fx') return 'fx';
    return 'index';
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
    if (range === '6m') start.setMonth(start.getMonth() - 6);
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
    if (range === 'all') return points;
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

function periodReturnsFromYieldSummary(
    summary: FundYieldSummaryResponse | null,
    latestPrice: number | null,
): FundSummary['period_returns'] {
    if (!summary?.periods || latestPrice == null || !Number.isFinite(latestPrice) || latestPrice <= 0) {
        return {};
    }
    const periodKeys = ['1w', '1m', '3m', '6m', 'ytd', '1y'] as const;
    return periodKeys.reduce<FundSummary['period_returns']>((returns, key) => {
        returns[key] = returnBetween(latestPrice, summary.periods[key]?.prev_close);
        return returns;
    }, {});
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

function formatWeightDelta(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} puan`;
}

function polarPoint(cx: number, cy: number, radius: number, angleDeg: number): { x: number; y: number } {
    const radians = ((angleDeg - 90) * Math.PI) / 180;
    return {
        x: cx + radius * Math.cos(radians),
        y: cy + radius * Math.sin(radians),
    };
}

function donutSegmentPath(startAngle: number, endAngle: number, outerRadius = 92, innerRadius = 44): string {
    const cx = 100;
    const cy = 100;
    const safeEndAngle = endAngle - startAngle >= 359.99 ? startAngle + 359.99 : endAngle;
    const outerStart = polarPoint(cx, cy, outerRadius, startAngle);
    const outerEnd = polarPoint(cx, cy, outerRadius, safeEndAngle);
    const innerEnd = polarPoint(cx, cy, innerRadius, safeEndAngle);
    const innerStart = polarPoint(cx, cy, innerRadius, startAngle);
    const largeArc = safeEndAngle - startAngle > 180 ? 1 : 0;
    return [
        `M ${outerStart.x} ${outerStart.y}`,
        `A ${outerRadius} ${outerRadius} 0 ${largeArc} 1 ${outerEnd.x} ${outerEnd.y}`,
        `L ${innerEnd.x} ${innerEnd.y}`,
        `A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${innerStart.x} ${innerStart.y}`,
        'Z',
    ].join(' ');
}

function chartDateMs(value: string): number {
    const ts = new Date(`${value}T00:00:00`).getTime();
    return Number.isFinite(ts) ? ts : 0;
}

function nearestChartPoint<T extends { date: string }>(points: T[], targetDate: string): T | null {
    if (!points.length) return null;
    const targetTs = chartDateMs(targetDate);
    let closest = points[0];
    let closestDiff = Math.abs(chartDateMs(points[0].date) - targetTs);
    for (const point of points.slice(1)) {
        const diff = Math.abs(chartDateMs(point.date) - targetTs);
        if (diff < closestDiff) {
            closest = point;
            closestDiff = diff;
        }
    }
    return closest;
}

function FundChartReturnStrip({
    periodReturns,
    yieldPeriods,
    currency,
    selectedRange,
    onRangeSelect,
}: {
    periodReturns: FundSummary['period_returns'] | undefined;
    yieldPeriods?: FundYieldSummaryResponse['periods'];
    currency?: string | null;
    selectedRange: FundChartRange;
    onRangeSelect: (range: FundChartRange) => void;
}) {
    return (
        <div className="fund-chart-return-strip" aria-label="Dönem getirileri">
            {DETAIL_RETURN_PERIODS.map((period) => {
                const value = periodReturns?.[period.key];
                const bounds = formatYieldBounds(yieldPeriods?.[period.key], currency || 'TRY');
                return (
                    <button
                        type="button"
                        key={period.key}
                        className={selectedRange === period.key ? 'is-active' : undefined}
                        aria-pressed={selectedRange === period.key}
                        onClick={() => onRangeSelect(period.key)}
                        title={bounds || period.label}
                    >
                        <span>{period.label}</span>
                        <strong className={pctClass(value)}>{formatPct(value)}</strong>
                    </button>
                );
            })}
        </div>
    );
}

function FundPerformanceChart({
    fundCode,
    points,
    comparisonAssets,
    comparisonHistory,
    comparisonLoading,
    comparisonError,
    selectedRange,
    pendingRange,
    rangeError,
    customStartDate,
    customEndDate,
    periodReturns,
    yieldPeriods,
    currency,
    comparison,
    onRangeSelect,
    onCustomStartDateChange,
    onCustomEndDateChange,
}: {
    fundCode: string;
    points: FundPricePoint[];
    comparisonAssets: ComparisonAsset[];
    comparisonHistory: MarketComparisonHistoryResponse | null;
    comparisonLoading: boolean;
    comparisonError: string | null;
    selectedRange: FundChartRange;
    pendingRange: FundChartRange | null;
    rangeError: string | null;
    customStartDate: string;
    customEndDate: string;
    periodReturns: FundSummary['period_returns'] | undefined;
    yieldPeriods?: FundYieldSummaryResponse['periods'];
    currency?: string | null;
    comparison: FundComparisonState;
    onRangeSelect: (range: FundChartRange) => void;
    onCustomStartDateChange: (value: string) => void;
    onCustomEndDateChange: (value: string) => void;
}) {
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const width = 860;
    const height = 430;
    const padding = { top: 34, right: 24, bottom: 50, left: 74 };
    const validPoints = points.filter((point) => Number.isFinite(Number(point.price)) && Number(point.price) > 0);
    const rangeReturn = validPoints.length >= 2
        ? returnBetween(Number(validPoints[validPoints.length - 1].price), Number(validPoints[0].price))
        : null;
    const color = rangeReturn == null || rangeReturn >= 0 ? '#22c55e' : '#ff4d5e';

    const returnStrip = (
        <FundChartReturnStrip
            periodReturns={periodReturns}
            yieldPeriods={yieldPeriods}
            currency={currency}
            selectedRange={selectedRange}
            onRangeSelect={onRangeSelect}
        />
    );
    const controls = (
        <div className="fund-chart-toolbar">
            <div className="fund-chart-range-row">
                {returnStrip}
            </div>
            <div className="fund-chart-custom-control">
                <button
                    type="button"
                    className={[
                        'fund-chart-range',
                        selectedRange === 'custom' ? 'is-active' : '',
                        pendingRange === 'custom' ? 'is-loading' : '',
                    ].filter(Boolean).join(' ')}
                    title="Özel aralık"
                    aria-pressed={selectedRange === 'custom'}
                    onClick={() => onRangeSelect('custom')}
                >
                    Özel
                </button>
                <div className={`fund-chart-return ${selectedRange === 'custom' ? pctClass(rangeReturn) : ''}`}>
                    {selectedRange === 'custom' ? formatPct(rangeReturn) : '-'}
                </div>
            </div>
        </div>
    );
    const customRangeInputs = selectedRange === 'custom' ? (
        <div className="fund-custom-range">
            <input type="date" value={customStartDate} onChange={(event) => onCustomStartDateChange(event.target.value)} />
            <input type="date" value={customEndDate} onChange={(event) => onCustomEndDateChange(event.target.value)} />
        </div>
    ) : null;

    if (validPoints.length < 2) {
        return (
            <section className="fund-chart-panel">
                {controls}
                {customRangeInputs}
                <div className="fund-chart-empty">{pendingRange ? 'Grafik verisi yükleniyor...' : 'Bu aralık için grafik verisi yok.'}</div>
                {rangeError && <div className="fund-chart-error">{rangeError}</div>}
                {comparisonError && <div className="fund-chart-error">{comparisonError}</div>}
            </section>
        );
    }

    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const gradientId = `fund-chart-area-${fundCode.replace(/[^a-zA-Z0-9]/g, '')}-${selectedRange}`;
    const tickIndexes = Array.from(new Set([0, Math.floor((validPoints.length - 1) / 2), validPoints.length - 1]));
    const firstChartDate = validPoints[0]?.date || null;
    const lastChartDate = validPoints[validPoints.length - 1]?.date || null;
    const selectedComparisonIds = comparisonAssets.map((asset) => asset.id);
    const comparisonHistoryAssets = comparisonHistory?.assets || [];
    const comparisonHistoryById = new Map(comparisonHistoryAssets.map((asset) => [asset.id, asset]));
    const comparisonHistoryMatches = comparisonAssets.length > 0
        && Boolean(comparisonHistory)
        && comparisonHistory?.start_date === firstChartDate
        && comparisonHistory?.end_date === lastChartDate
        && selectedComparisonIds.every((id) => comparisonHistoryById.has(id));
    const hasUsableComparisonSeries = comparisonHistoryMatches
        && selectedComparisonIds.some((id) => (
            (comparisonHistoryById.get(id)?.points || [])
                .filter((point) => Number.isFinite(Number(point.value)) && Number(point.value) > 0)
                .length >= 2
        ));
    const comparisonMode = comparisonHistoryMatches && hasUsableComparisonSeries;
    const waitingForComparison = comparisonAssets.length > 0
        && !comparisonMode
        && (comparisonLoading || !comparisonHistoryMatches)
        && !comparisonError;

    if (waitingForComparison) {
        const yTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => padding.top + ratio * plotHeight);
        return (
            <section className="fund-chart-panel">
                {controls}
                {customRangeInputs}
                <svg
                    className="fund-detail-chart"
                    viewBox={`0 0 ${width} ${height}`}
                    preserveAspectRatio="xMidYMid meet"
                    role="img"
                    aria-label={`${fundCode} karşılaştırmalı getiri grafiği yükleniyor`}
                >
                    {yTicks.map((y) => (
                        <line key={y} x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="fund-chart-gridline" />
                    ))}
                    {tickIndexes.map((index) => (
                        <text
                            key={index}
                            x={padding.left + (index / Math.max(1, validPoints.length - 1)) * plotWidth}
                            y={height - 10}
                            className="fund-chart-axis"
                            textAnchor={index === 0 ? 'start' : index === validPoints.length - 1 ? 'end' : 'middle'}
                        >
                            {formatChartDate(validPoints[index].date, selectedRange)}
                        </text>
                    ))}
                    <text x={width / 2} y={height / 2} className="fund-chart-loading-label" textAnchor="middle">
                        Karşılaştırma grafiği hazırlanıyor...
                    </text>
                </svg>
                <FundComparisonControls comparison={comparison} surface="chart" className="fund-chart-compare-controls" />
                {rangeError && <div className="fund-chart-error">{rangeError}</div>}
            </section>
        );
    }

    if (comparisonMode) {
        const historyById = comparisonHistoryById;
        const baseAsset: ComparisonAsset = {
            id: comparisonAssetId('fund', fundCode),
            kind: 'fund',
            symbol: fundCode,
            label: fundCode,
            returns: {},
        };
        const baseSeries: FundChartComparisonSeries = {
            asset: baseAsset,
            points: validPoints.map((point) => ({ date: point.date, value: Number(point.price) })),
            source: 'fund_performance',
        };
        const externalSeries: FundChartComparisonSeries[] = comparisonAssets.map((asset) => {
            const history = historyById.get(asset.id);
            return {
                asset,
                points: (history?.points || [])
                    .filter((point) => Number.isFinite(Number(point.value)) && Number(point.value) > 0)
                    .map((point) => ({ date: point.date, value: Number(point.value) })),
                source: history?.source,
                error: history?.error,
            };
        });
        const rawSeries = [baseSeries, ...externalSeries];
        const normalizedSeries = rawSeries
            .map((series, index) => {
                const ordered = [...series.points].sort((a, b) => a.date.localeCompare(b.date));
                const baseValue = ordered.find((point) => point.value > 0)?.value ?? null;
                return {
                    ...series,
                    color: FUND_CHART_COMPARISON_COLORS[index % FUND_CHART_COMPARISON_COLORS.length],
                    points: baseValue
                        ? ordered.map((point) => ({
                            date: point.date,
                            value: ((point.value / baseValue) - 1) * 100,
                        }))
                        : [],
                };
            })
            .filter((series) => series.points.length >= 2);
        const returnValues = normalizedSeries.flatMap((series) => series.points.map((point) => point.value));
        const rawMin = Math.min(0, ...returnValues);
        const rawMax = Math.max(0, ...returnValues);
        const rawSpan = Math.max(1, rawMax - rawMin);
        const minValue = rawMin - rawSpan * 0.08;
        const maxValue = rawMax + rawSpan * 0.08;
        const span = Math.max(1, maxValue - minValue);
        const startTs = chartDateMs(validPoints[0].date);
        const endTs = Math.max(startTs + 1, chartDateMs(validPoints[validPoints.length - 1].date));
        const xForDate = (date: string) => {
            const ratio = (chartDateMs(date) - startTs) / (endTs - startTs);
            return padding.left + Math.min(1, Math.max(0, ratio)) * plotWidth;
        };
        const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;
        const yTicks = Array.from({ length: 5 }, (_, index) => maxValue - (span * index) / 4);
        const zeroY = yFor(0);
        const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
            const rect = event.currentTarget.getBoundingClientRect();
            if (rect.width <= 0) return;
            const x = ((event.clientX - rect.left) / rect.width) * width;
            let closestIndex = 0;
            let minDiff = Infinity;
            for (let index = 0; index < validPoints.length; index += 1) {
                const diff = Math.abs(xForDate(validPoints[index].date) - x);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestIndex = index;
                }
            }
            setHoverIndex(closestIndex);
        };
        const hoverPoint = hoverIndex == null ? null : validPoints[Math.max(0, Math.min(validPoints.length - 1, hoverIndex))];
        const hoverX = hoverPoint ? xForDate(hoverPoint.date) : null;
        const tooltipRows = hoverPoint
            ? normalizedSeries
                .map((series) => {
                    const point = nearestChartPoint(series.points, hoverPoint.date);
                    return point ? { series, point } : null;
                })
                .filter((item): item is { series: typeof normalizedSeries[number]; point: { date: string; value: number } } => Boolean(item))
                .slice(0, 4)
            : [];
        const tooltipWidth = 172;
        const tooltipHeight = Math.max(48, 24 + tooltipRows.length * 15);
        const tooltipGap = 12;
        const tooltipShouldFlip = hoverX != null && hoverX + tooltipGap + tooltipWidth > width - padding.right;
        const tooltipX = hoverX == null
            ? 0
            : tooltipShouldFlip
                ? Math.max(8, hoverX - tooltipGap - tooltipWidth)
                : Math.min(hoverX + tooltipGap, width - tooltipWidth - 8);
        const tooltipY = padding.top + 8;

        return (
            <section className="fund-chart-panel">
                {controls}
                {customRangeInputs}
                <svg
                    key={`${fundCode}-${selectedRange}-${validPoints.length}-compare-${comparisonAssets.map((asset) => asset.id).join('-')}`}
                    className="fund-detail-chart"
                    viewBox={`0 0 ${width} ${height}`}
                    preserveAspectRatio="xMidYMid meet"
                    role="img"
                    aria-label={`${fundCode} karşılaştırmalı getiri grafiği`}
                    onPointerMove={handlePointerMove}
                    onPointerLeave={() => setHoverIndex(null)}
                >
                    <defs>
                        <clipPath id={`${gradientId}-compare-clip`}>
                            <rect x={padding.left} y={0} width={plotWidth} height={height} className="fund-chart-reveal" />
                        </clipPath>
                    </defs>
                    {yTicks.map((tick) => {
                        const y = yFor(tick);
                        return (
                            <g key={tick}>
                                <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="fund-chart-gridline" />
                                <text x={padding.left - 10} y={y + 4} className="fund-chart-axis" textAnchor="end">
                                    {formatPct(tick).replace('+', '')}
                                </text>
                            </g>
                        );
                    })}
                    <line x1={padding.left} x2={width - padding.right} y1={zeroY} y2={zeroY} className="fund-comparison-zero" />
                    {tickIndexes.map((index) => (
                        <text
                            key={index}
                            x={xForDate(validPoints[index].date)}
                            y={height - 10}
                            className="fund-chart-axis"
                            textAnchor={index === 0 ? 'start' : index === validPoints.length - 1 ? 'end' : 'middle'}
                        >
                            {formatChartDate(validPoints[index].date, selectedRange)}
                        </text>
                    ))}
                    <g clipPath={`url(#${gradientId}-compare-clip)`}>
                        {normalizedSeries.map((series) => {
                            const pathData = series.points
                                .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xForDate(point.date)} ${yFor(point.value)}`)
                                .join(' ');
                            return (
                                <path
                                    key={series.asset.id}
                                    d={pathData}
                                    fill="none"
                                    stroke={series.color}
                                    strokeWidth={series.asset.id === baseAsset.id ? 3 : 2.2}
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    className="fund-chart-line"
                                />
                            );
                        })}
                        {normalizedSeries.flatMap((series) => (
                            series.points.map((point, index) => (
                                <circle
                                    key={`${series.asset.id}-node-${index}`}
                                    className="fund-chart-node"
                                    cx={xForDate(point.date)}
                                    cy={yFor(point.value)}
                                    r={series.asset.id === baseAsset.id ? 1.9 : 1.6}
                                    fill={series.color}
                                />
                            ))
                        ))}
                    </g>
                    {normalizedSeries.map((series) => {
                        const lastPoint = series.points[series.points.length - 1];
                        return lastPoint ? (
                            <circle
                                key={`${series.asset.id}-endpoint`}
                                className="fund-chart-endpoint"
                                cx={xForDate(lastPoint.date)}
                                cy={yFor(lastPoint.value)}
                                r={series.asset.id === baseAsset.id ? 4 : 3.4}
                                fill={series.color}
                            />
                        ) : null;
                    })}
                    {hoverPoint && hoverX != null && tooltipRows.length > 0 && (
                        <g>
                            <line x1={hoverX} x2={hoverX} y1={padding.top} y2={height - padding.bottom} className="fund-chart-hoverline" />
                            {tooltipRows.map(({ series, point }) => (
                                <circle
                                    key={`${series.asset.id}-hover`}
                                    className="fund-chart-hover-point"
                                    cx={xForDate(point.date)}
                                    cy={yFor(point.value)}
                                    r={series.asset.id === baseAsset.id ? 4 : 3.5}
                                    fill={series.color}
                                    stroke="#0a0c0f"
                                    strokeWidth="1.6"
                                />
                            ))}
                            <g transform={`translate(${tooltipX}, ${tooltipY})`}>
                                <rect width={tooltipWidth} height={tooltipHeight} rx="6" className="fund-chart-tooltip-bg" />
                                <text x="8" y="17" className="fund-chart-tooltip-muted">{formatDate(hoverPoint.date)}</text>
                                {tooltipRows.map(({ series, point }, index) => (
                                    <g key={series.asset.id} transform={`translate(8, ${35 + index * 15})`}>
                                        <circle cx="3.5" cy="-4" r="2.6" fill={series.color} />
                                        <text x="12" y="0" className="fund-chart-tooltip-muted">
                                            {(series.asset.displaySymbol || series.asset.symbol).slice(0, 8)}
                                        </text>
                                        <text x={tooltipWidth - 10} y="0" className="fund-chart-tooltip-value" textAnchor="end">
                                            {formatPct(point.value)}
                                        </text>
                                    </g>
                                ))}
                            </g>
                        </g>
                    )}
                </svg>
                <div className="fund-chart-legend" aria-label="Grafik karşılaştırmaları">
                    {normalizedSeries.map((series) => (
                        <span key={series.asset.id}>
                            <i style={{ background: series.color }} />
                            {series.asset.displaySymbol || series.asset.symbol}
                        </span>
                    ))}
                </div>
                <FundComparisonControls comparison={comparison} surface="chart" className="fund-chart-compare-controls" />
                {rangeError && <div className="fund-chart-error">{rangeError}</div>}
                {comparisonError && <div className="fund-chart-error">{comparisonError}</div>}
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
    const xFor = (index: number) => padding.left + (validPoints.length === 1 ? 0 : (index / (validPoints.length - 1)) * plotWidth);
    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;
    const pathData = validPoints.map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(index)} ${yFor(Number(point.price))}`).join(' ');
    const areaData = `${pathData} L ${xFor(validPoints.length - 1)} ${height - padding.bottom} L ${padding.left} ${height - padding.bottom} Z`;
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
    const tooltipGap = 12;
    const tooltipShouldFlip = hoverX != null && hoverX + tooltipGap + tooltipWidth > width - padding.right;
    const tooltipX = hoverX == null
        ? 0
        : tooltipShouldFlip
            ? Math.max(8, hoverX - tooltipGap - tooltipWidth)
            : Math.min(hoverX + tooltipGap, width - tooltipWidth - 8);
    const tooltipY = hoverY == null ? 0 : Math.min(Math.max(hoverY - tooltipHeight / 2, padding.top), height - tooltipHeight - 8);

    return (
        <section className="fund-chart-panel">
            {controls}
            {customRangeInputs}
            <svg
                key={`${fundCode}-${selectedRange}-${validPoints.length}`}
                className="fund-detail-chart"
                viewBox={`0 0 ${width} ${height}`}
                preserveAspectRatio="xMidYMid meet"
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
                    <clipPath id={`${gradientId}-clip`}>
                        <rect x={padding.left} y={0} width={plotWidth} height={height} className="fund-chart-reveal" />
                    </clipPath>
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
                <g clipPath={`url(#${gradientId}-clip)`}>
                    <path d={areaData} fill={`url(#${gradientId})`} />
                    <path d={pathData} fill="none" stroke={color} strokeWidth="2.7" strokeLinecap="round" strokeLinejoin="round" />
                </g>
                <circle
                    className="fund-chart-endpoint"
                    cx={xFor(validPoints.length - 1)}
                    cy={yFor(Number(validPoints[validPoints.length - 1].price))}
                    r="4"
                    fill={color}
                />
                {hoverPoint && hoverX != null && hoverY != null && (
                    <g>
                        <line x1={hoverX} x2={hoverX} y1={padding.top} y2={height - padding.bottom} className="fund-chart-hoverline" />
                        <circle className="fund-chart-hover-point" cx={hoverX} cy={hoverY} r="3.6" fill={color} stroke="#0a0c0f" strokeWidth="1.6" />
                        <g transform={`translate(${tooltipX}, ${tooltipY})`}>
                            <rect width={tooltipWidth} height={tooltipHeight} rx="6" className="fund-chart-tooltip-bg" />
                            <text x="10" y="20" className="fund-chart-tooltip-muted">{formatDate(hoverPoint.date)}</text>
                            <text x="10" y="43" className="fund-chart-tooltip-value">{formatCurrency(hoverPoint.price, 'TRY')}</text>
                        </g>
                    </g>
                )}
            </svg>
            <FundComparisonControls comparison={comparison} surface="chart" className="fund-chart-compare-controls" />
            {comparisonLoading && comparisonAssets.length > 0 && (
                <div className="fund-chart-note">Karşılaştırma geçmişi yükleniyor...</div>
            )}
            {comparisonError && <div className="fund-chart-error">{comparisonError}</div>}
            {rangeError && <div className="fund-chart-error">{rangeError}</div>}
        </section>
    );
}

function FundAllocationSummary({
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
                <div className="funds-state">Dağılım verisi yükleniyor...</div>
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
                            <strong>{activeSegment ? formatAllocationWeight(activeSegment.item.weight) : formatAllocationWeight(positiveTotal)}</strong>
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

type AllocationHistoryTrend = {
    allocationType: string;
    label: string;
    color: string;
    startWeight: number | null;
    endWeight: number | null;
    delta: number | null;
    points: AllocationHistoryPoint[];
};

type AllocationHistoryPoint = {
    date: string;
    weight: number | null;
};

function allocationWeightForType(day: FundAllocationHistoryDay, allocationType: string): number | null {
    const item = day.allocations.find((allocation) => allocation.allocation_type === allocationType);
    const weight = Number(item?.weight);
    return Number.isFinite(weight) ? weight : null;
}

function buildAllocationHistoryTrends(history: FundAllocationHistoryDay[]): AllocationHistoryTrend[] {
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

function allocationDeltaAtIndex(points: AllocationHistoryPoint[], index: number): number | null {
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

function latestAllocationDelta(trend: AllocationHistoryTrend): number | null {
    for (let index = trend.points.length - 1; index >= 0; index -= 1) {
        if (trend.points[index]?.weight != null) {
            return allocationDeltaAtIndex(trend.points, index);
        }
    }
    return null;
}

function AllocationSparkline({ trend }: { trend: AllocationHistoryTrend }) {
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

function AllocationHistoryChart({
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
                            <small className={pctClass(delta)}>{formatWeightDelta(delta)}</small>
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

function FundAllocationHistoryPanel({
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
                <div className="funds-state">Dağılım geçmişi TEFAS üzerinden yükleniyor...</div>
            ) : error ? (
                <div className="funds-state funds-state-error">{error}</div>
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
                                        <td className={pctClass(latestAllocationDelta(trend))}>{formatWeightDelta(latestAllocationDelta(trend))}</td>
                                        <td className={pctClass(trend.delta)}>{formatWeightDelta(trend.delta)}</td>
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

function useFundComparisonState({
    selectedFund,
    baseReturns,
    funds,
    openSearchSignal,
}: {
    selectedFund: FundSummary | null;
    baseReturns: FundSummary['period_returns'];
    funds: FundSummary[];
    openSearchSignal: number;
}): FundComparisonState {
    const [period, setPeriod] = useState<ComparisonPeriod>('1m');
    const [selectedIds, setSelectedIds] = useState<string[]>(() => [...DEFAULT_COMPARISON_IDS]);
    const [query, setQuery] = useState('');
    const [searchOpen, setSearchOpen] = useState(false);
    const [searchSurface, setSearchSurface] = useState<FundComparisonControlSurface | null>(null);
    const [fundSearchRows, setFundSearchRows] = useState<FundSummary[]>([]);
    const [stockRows, setStockRows] = useState<MarketStockRow[]>([]);
    const [indexRows, setIndexRows] = useState<MarketIndexListRow[]>([]);
    const [fxRows, setFxRows] = useState<FxQuote[]>([]);
    const [fundSearchLoading, setFundSearchLoading] = useState(false);
    const [stocksLoading, setStocksLoading] = useState(false);
    const [indicesLoading, setIndicesLoading] = useState(false);
    const [fxLoading, setFxLoading] = useState(false);
    const inputRef = useRef<HTMLInputElement | null>(null);
    const stocksLoadedRef = useRef(false);

    const selectedFundCode = normalizeCompareSymbol(selectedFund?.fund_code);
    const baseAsset = useMemo<ComparisonAsset | null>(() => {
        if (!selectedFund || !selectedFundCode) return null;
        return {
            id: comparisonAssetId('fund', selectedFundCode),
            kind: 'fund',
            symbol: selectedFundCode,
            label: selectedFund.name,
            logoName: selectedFund.founder_company || selectedFund.manager_company || selectedFund.name,
            returns: {
                '1w': baseReturns?.['1w'] ?? selectedFund.period_returns?.['1w'] ?? null,
                '1m': baseReturns?.['1m'] ?? selectedFund.period_returns?.['1m'] ?? null,
                '3m': baseReturns?.['3m'] ?? selectedFund.period_returns?.['3m'] ?? null,
                '6m': baseReturns?.['6m'] ?? selectedFund.period_returns?.['6m'] ?? null,
                ytd: baseReturns?.ytd ?? selectedFund.period_returns?.ytd ?? null,
                '1y': baseReturns?.['1y'] ?? selectedFund.period_returns?.['1y'] ?? null,
            },
        };
    }, [baseReturns, selectedFund, selectedFundCode]);

    useEffect(() => {
        setSelectedIds([...DEFAULT_COMPARISON_IDS]);
        setQuery('');
        setSearchOpen(false);
        setSearchSurface(null);
    }, [selectedFundCode]);

    useEffect(() => {
        if (!selectedFundCode) return;
        let alive = true;
        setIndicesLoading(true);
        setFxLoading(true);
        Promise.allSettled([apiClient.marketIndices(), apiClient.marketFx()])
            .then(([indicesResult, fxResult]) => {
                if (!alive) return;
                setIndexRows(indicesResult.status === 'fulfilled' ? indicesResult.value.rows || [] : []);
                setFxRows(fxResult.status === 'fulfilled' ? fxResult.value.items || [] : []);
            })
            .finally(() => {
                if (alive) {
                    setIndicesLoading(false);
                    setFxLoading(false);
                }
            });
        return () => {
            alive = false;
        };
    }, [selectedFundCode]);

    const ensureStockRows = useCallback(() => {
        if (stocksLoadedRef.current || stocksLoading) return;
        stocksLoadedRef.current = true;
        setStocksLoading(true);
        apiClient
            .marketStocks({ index: 'XUTUM' })
            .then((payload) => {
                setStockRows(payload.rows || []);
            })
            .catch(() => {
                setStockRows([]);
                stocksLoadedRef.current = false;
            })
            .finally(() => setStocksLoading(false));
    }, [stocksLoading]);

    const openSearch = useCallback((surface: FundComparisonControlSurface = 'returns') => {
        setSearchSurface(surface);
        setSearchOpen(true);
        ensureStockRows();
        window.setTimeout(() => inputRef.current?.focus(), 0);
    }, [ensureStockRows]);

    useEffect(() => {
        if (openSearchSignal <= 0 || !selectedFundCode) return;
        openSearch('returns');
    }, [openSearch, openSearchSignal, selectedFundCode]);

    useEffect(() => {
        const trimmed = query.trim();
        if (!searchOpen || !trimmed) {
            setFundSearchRows([]);
            setFundSearchLoading(false);
            return;
        }
        let alive = true;
        setFundSearchLoading(true);
        const timer = window.setTimeout(() => {
            apiClient
                .fundSearch(trimmed, 10)
                .then((payload) => {
                    if (alive) setFundSearchRows(payload.rows || []);
                })
                .catch(() => {
                    if (alive) setFundSearchRows([]);
                })
                .finally(() => {
                    if (alive) setFundSearchLoading(false);
                });
        }, 160);
        return () => {
            alive = false;
            window.clearTimeout(timer);
        };
    }, [query, searchOpen]);

    const allAssets = useMemo(() => {
        if (!selectedFundCode) return [];
        const currentFundId = comparisonAssetId('fund', selectedFundCode);
        const mergedFunds = new Map<string, FundSummary>();
        for (const row of funds) {
            mergedFunds.set(normalizeCompareSymbol(row.fund_code), row);
        }
        for (const row of fundSearchRows) {
            mergedFunds.set(normalizeCompareSymbol(row.fund_code), row);
        }
        const fundAssets = [...mergedFunds.values()]
            .filter((row) => comparisonAssetId('fund', row.fund_code) !== currentFundId)
            .map(fundToComparisonAsset);
        return [
            ...fundAssets,
            ...stockRows.map(stockToComparisonAsset),
            ...indexRows.map(indexToComparisonAsset),
            ...fxRows.map(fxToComparisonAsset).filter((asset): asset is ComparisonAsset => Boolean(asset)),
        ];
    }, [fundSearchRows, funds, fxRows, indexRows, selectedFundCode, stockRows]);

    const assetById = useMemo(() => new Map(allAssets.map((asset) => [asset.id, asset])), [allAssets]);
    const selectedAssets = useMemo(
        () => selectedIds.map((id) => assetById.get(id)).filter((asset): asset is ComparisonAsset => Boolean(asset)),
        [assetById, selectedIds],
    );

    const searchResults = useMemo(() => {
        const normalizedQuery = normalizeCompareSearch(query);
        const compactQuery = compactCompareSearch(query);
        if (!normalizedQuery) return [];
        const selectedSet = new Set(selectedIds);
        return allAssets
            .filter((asset) => !selectedSet.has(asset.id))
            .map((asset) => {
                const searchParts = [
                    asset.symbol,
                    asset.displaySymbol || '',
                    asset.label,
                    comparisonAssetKindLabel(asset.kind),
                    ...(asset.kind === 'stock' ? STOCK_SEARCH_LABELS[asset.symbol] || [] : []),
                ];
                const normalizedParts = searchParts.map(normalizeCompareSearch);
                const compactParts = searchParts.map(compactCompareSearch);
                const starts = normalizedParts.some((item) => item.startsWith(normalizedQuery))
                    || compactParts.some((item) => item.startsWith(compactQuery));
                const includes = normalizedParts.some((item) => item.includes(normalizedQuery))
                    || compactParts.some((item) => item.includes(compactQuery));
                if (!starts && !includes) return null;
                return { asset, score: starts ? 0 : 1 };
            })
            .filter((item): item is { asset: ComparisonAsset; score: number } => Boolean(item))
            .sort((a, b) => a.score - b.score || a.asset.symbol.localeCompare(b.asset.symbol, 'tr'))
            .slice(0, 8)
            .map((item) => item.asset);
    }, [allAssets, query, selectedIds]);

    const addAsset = useCallback((asset: ComparisonAsset) => {
        setSelectedIds((current) => current.includes(asset.id) ? current : [...current, asset.id]);
        setQuery('');
        setSearchOpen(false);
    }, []);

    const removeAsset = useCallback((assetId: string) => {
        setSelectedIds((current) => current.filter((id) => id !== assetId));
    }, []);

    return {
        baseAsset,
        selectedAssets,
        selectedIds,
        period,
        setPeriod,
        query,
        setQuery,
        searchOpen,
        setSearchOpen,
        searchSurface,
        searchResults,
        addAsset,
        removeAsset,
        openSearch,
        fundSearchLoading,
        stocksLoading,
        indicesLoading,
        fxLoading,
        inputRef,
    };
}

function FundComparisonControls({
    comparison,
    surface,
    className,
}: {
    comparison: FundComparisonState;
    surface: FundComparisonControlSurface;
    className?: string;
}) {
    const {
        selectedAssets,
        query,
        setQuery,
        searchOpen,
        setSearchOpen,
        searchSurface,
        searchResults,
        addAsset,
        removeAsset,
        openSearch,
        fundSearchLoading,
        stocksLoading,
        indicesLoading,
        fxLoading,
        inputRef,
    } = comparison;
    const surfaceOpen = searchOpen && searchSurface === surface;

    return (
        <div className={`fund-comparison-controls${className ? ` ${className}` : ''}`}>
            <div className="fund-comparison-search-wrap">
                {surfaceOpen ? (
                    <div className="fund-comparison-search-active">
                        <Search size={15} aria-hidden="true" />
                        <input
                            ref={inputRef}
                            value={query}
                            onChange={(event) => setQuery(event.target.value)}
                            onKeyDown={(event) => {
                                if (event.key === 'Escape') {
                                    setSearchOpen(false);
                                    setQuery('');
                                }
                                if (event.key === 'Enter' && searchResults[0]) {
                                    event.preventDefault();
                                    addAsset(searchResults[0]);
                                }
                            }}
                            placeholder="Karşılaştır"
                            aria-label="Fon, hisse, endeks veya döviz ara"
                        />
                        <button
                            type="button"
                            onClick={() => {
                                setSearchOpen(false);
                                setQuery('');
                            }}
                            aria-label="Aramayı kapat"
                        >
                            <X size={15} aria-hidden="true" />
                        </button>
                    </div>
                ) : (
                    <button type="button" className="fund-comparison-search-button" onClick={() => openSearch(surface)}>
                        <Search size={15} aria-hidden="true" />
                        Karşılaştırma
                    </button>
                )}
                {surfaceOpen && (
                    <div className="fund-comparison-results">
                        {query.trim() ? (
                            searchResults.length ? (
                                searchResults.map((asset) => (
                                    <button key={asset.id} type="button" onClick={() => addAsset(asset)}>
                                        <SymbolLogo
                                            symbol={asset.symbol}
                                            name={asset.logoName || asset.label}
                                            kind={comparisonLogoKind(asset.kind)}
                                            logoUrl={asset.logoUrl}
                                            size="sm"
                                        />
                                        <strong>{asset.displaySymbol || asset.symbol}</strong>
                                        <span>{asset.label}</span>
                                    </button>
                                ))
                            ) : (
                                <div className="fund-comparison-result-empty">
                                    {fundSearchLoading || stocksLoading || indicesLoading || fxLoading ? 'Aranıyor...' : 'Sonuç bulunamadı.'}
                                </div>
                            )
                        ) : (
                            <div className="fund-comparison-result-empty">Fon, hisse, endeks veya döviz yazın.</div>
                        )}
                    </div>
                )}
            </div>

            <div className="fund-comparison-chips" aria-label="Seçili karşılaştırmalar">
                {selectedAssets.map((asset) => (
                    <button
                        key={asset.id}
                        type="button"
                        onClick={() => removeAsset(asset.id)}
                        title={`${asset.displaySymbol || asset.symbol} karşılaştırmadan çıkar`}
                    >
                        <i />
                        {asset.displaySymbol || asset.symbol}
                    </button>
                ))}
            </div>
        </div>
    );
}

function FundReturnComparison({
    comparison,
    containerRef,
}: {
    comparison: FundComparisonState;
    containerRef: { current: HTMLDivElement | null };
}) {
    const [hover, setHover] = useState<{ assetId: string; x: number; y: number; width: number; height: number } | null>(null);
    const {
        baseAsset,
        selectedAssets,
        selectedIds,
        period,
        setPeriod,
    } = comparison;

    const chartItems = useMemo(() => {
        if (!baseAsset) return [];
        const items = [baseAsset, ...selectedAssets]
            .map((asset) => ({
                asset,
                value: periodReturnFromAsset(asset, period),
            }))
            .filter((item) => item.value != null);
        return items.sort((a, b) => {
            const valueDiff = Number(b.value) - Number(a.value);
            if (valueDiff !== 0) return valueDiff;
            return (a.asset.displaySymbol || a.asset.symbol).localeCompare(b.asset.displaySymbol || b.asset.symbol, 'tr');
        });
    }, [baseAsset, period, selectedAssets]);

    useEffect(() => {
        setHover(null);
    }, [period, selectedIds]);

    if (!baseAsset) {
        return null;
    }

    const chartWidth = 1040;
    const chartHeight = 340;
    const padding = { top: 28, right: 22, bottom: 58, left: 64 };
    const plotWidth = chartWidth - padding.left - padding.right;
    const plotHeight = chartHeight - padding.top - padding.bottom;
    const values = chartItems.map((item) => Number(item.value));
    const rawMax = Math.max(0, ...values);
    const rawMin = Math.min(0, ...values);
    const maxValue = Math.max(1, Math.ceil(rawMax + Math.max(1, rawMax - rawMin) * 0.08));
    const minValue = rawMin < 0 ? Math.floor(rawMin - Math.max(1, rawMax - rawMin) * 0.08) : 0;
    const valueSpan = Math.max(1, maxValue - minValue);
    const yFor = (value: number) => padding.top + ((maxValue - value) / valueSpan) * plotHeight;
    const zeroY = yFor(0);
    const barGap = chartItems.length > 1 ? 46 : 0;
    const rawBarWidth = chartItems.length ? (plotWidth - barGap * (chartItems.length - 1)) / chartItems.length : 0;
    const barWidth = Math.max(46, Math.min(132, rawBarWidth));
    const totalBarsWidth = chartItems.length * barWidth + Math.max(0, chartItems.length - 1) * barGap;
    const startX = padding.left + Math.max(0, (plotWidth - totalBarsWidth) / 2);
    const ticks = Array.from({ length: 6 }, (_, index) => minValue + (valueSpan * index) / 5);
    const hoverItem = hover ? chartItems.find((item) => item.asset.id === hover.assetId) || null : null;
    const tooltipAlignX = hover && hover.x > hover.width - 180 ? 'left' : 'right';
    const tooltipAlignY = hover && hover.y > hover.height - 110 ? 'up' : 'down';

    const handleBarPointerMove = (event: ReactPointerEvent<SVGRectElement>, assetId: string) => {
        const rect = event.currentTarget.ownerSVGElement?.getBoundingClientRect();
        if (!rect) return;
        setHover({
            assetId,
            x: event.clientX - rect.left,
            y: event.clientY - rect.top,
            width: rect.width,
            height: rect.height,
        });
    };

    const handleSvgPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const target = event.target as Element | null;
        if (!target || !(target instanceof SVGElement)) return;
        // Only keep hover state while pointer is genuinely over a bar.
        if (!target.classList.contains('fund-comparison-bar')) {
            setHover(null);
        }
    };

    return (
        <section className="fund-comparison-panel" ref={containerRef} aria-label="Getiri karşılaştırma">
            <div className="fund-comparison-head">
                <h2>Getiri Karşılaştırma</h2>
                <label className="fund-comparison-period">
                    <select value={period} onChange={(event) => setPeriod(event.target.value as ComparisonPeriod)}>
                        {DETAIL_RETURN_PERIODS.map((item) => (
                            <option key={item.key} value={item.key}>{item.label}</option>
                        ))}
                    </select>
                    <ChevronDown size={15} aria-hidden="true" />
                </label>
            </div>

            <div className="fund-comparison-chart-wrap">
                {chartItems.length ? (
                    <svg
                        className="fund-comparison-chart"
                        viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                        role="img"
                        aria-label="Seçili varlıkların dönem getirileri"
                        onPointerMove={handleSvgPointerMove}
                        onPointerLeave={() => setHover(null)}
                    >
                        {ticks.map((tick) => {
                            const y = yFor(tick);
                            return (
                                <g key={tick}>
                                    <line x1={padding.left} x2={chartWidth - padding.right} y1={y} y2={y} className="fund-comparison-gridline" />
                                    <text x={padding.left - 12} y={y + 4} className="fund-chart-axis" textAnchor="end">
                                        {formatPct(tick).replace('+', '')}
                                    </text>
                                </g>
                            );
                        })}
                        <line x1={padding.left} x2={chartWidth - padding.right} y1={zeroY} y2={zeroY} className="fund-comparison-zero" />
                        {chartItems.map(({ asset, value }, index) => {
                            const numericValue = Number(value);
                            const x = startX + index * (barWidth + barGap);
                            const y = numericValue >= 0 ? yFor(numericValue) : zeroY;
                            const height = Math.max(2, Math.abs(zeroY - yFor(numericValue)));
                            const isBase = asset.id === baseAsset.id;
                            const isActive = hover?.assetId === asset.id;
                            const displaySymbol = asset.displaySymbol || asset.symbol;
                            return (
                                <g
                                    key={asset.id}
                                    className="fund-comparison-bar-group"
                                    style={{ transform: `translateX(${x}px)` } as CSSProperties}
                                >
                                    <rect
                                        x={0}
                                        y={y}
                                        width={barWidth}
                                        height={height}
                                        className={[
                                            'fund-comparison-bar',
                                            isBase ? 'is-base' : '',
                                            isActive ? 'is-active' : '',
                                            hover && !isActive ? 'is-muted' : '',
                                            numericValue < 0 ? 'is-negative' : '',
                                        ].filter(Boolean).join(' ')}
                                        style={{ y: `${y}px`, height: `${height}px` } as CSSProperties}
                                        onPointerMove={(event) => handleBarPointerMove(event, asset.id)}
                                        onPointerEnter={(event) => handleBarPointerMove(event, asset.id)}
                                    />
                                    <text
                                        x={barWidth / 2}
                                        y={numericValue >= 0 ? y - 8 : y + height + 18}
                                        className="fund-comparison-value-label"
                                        textAnchor="middle"
                                        style={{ y: `${numericValue >= 0 ? y - 8 : y + height + 18}px` } as CSSProperties}
                                    >
                                        {formatPct(numericValue).replace('+', '')}
                                    </text>
                                    <text
                                        x={barWidth / 2}
                                        y={chartHeight - 18}
                                        className="fund-comparison-x-label"
                                        textAnchor="middle"
                                        transform={`rotate(-32 ${barWidth / 2} ${chartHeight - 18})`}
                                    >
                                        {displaySymbol}
                                    </text>
                                </g>
                            );
                        })}
                    </svg>
                ) : (
                    <div className="fund-comparison-empty">Bu dönem için karşılaştırılabilir getiri yok.</div>
                )}
                {hover && hoverItem ? (
                    <div
                        className={[
                            'fund-comparison-tooltip',
                            tooltipAlignX === 'left' ? 'align-left' : '',
                            tooltipAlignY === 'up' ? 'align-up' : '',
                        ].filter(Boolean).join(' ')}
                        style={{ left: `${hover.x}px`, top: `${hover.y}px` }}
                    >
                        <span>{hoverItem.asset.displaySymbol || hoverItem.asset.symbol}</span>
                        <strong>{formatPct(hoverItem.value)}</strong>
                    </div>
                ) : null}
            </div>

            <FundComparisonControls comparison={comparison} surface="returns" />
        </section>
    );
}

function FundHistoryTable({ points }: { points: FundPricePoint[] }) {
    const orderedPoints = [...points].sort((a, b) => a.date.localeCompare(b.date));
    const latestPoint = orderedPoints[orderedPoints.length - 1] || null;
    const latestDate = latestPoint ? new Date(`${latestPoint.date}T00:00:00`) : null;
    const startDate = latestDate && !Number.isNaN(latestDate.getTime()) ? new Date(latestDate) : null;
    if (startDate) {
        startDate.setDate(startDate.getDate() - 29);
    }
    const rows = orderedPoints
        .filter((point) => {
            if (!startDate || !latestDate) return true;
            const pointDate = new Date(`${point.date}T00:00:00`);
            return !Number.isNaN(pointDate.getTime()) && pointDate >= startDate && pointDate <= latestDate;
        })
        .sort((a, b) => b.date.localeCompare(a.date));
    const investorDeltas = new Map<string, number>();
    let previousInvestorCount: number | null = null;

    rows.slice().reverse().forEach((point) => {
        const currentInvestorCount = point.investor_count;
        if (typeof currentInvestorCount !== 'number' || !Number.isFinite(currentInvestorCount)) {
            return;
        }
        if (previousInvestorCount != null) {
            investorDeltas.set(point.date, currentInvestorCount - previousInvestorCount);
        }
        previousInvestorCount = currentInvestorCount;
    });

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
                    {rows.map((point) => {
                        const investorDelta = investorDeltas.get(point.date);
                        return (
                            <tr key={point.date}>
                                <td>{formatDate(point.date)}</td>
                                <td>{formatCurrency(point.price, 'TRY')}</td>
                                <td className={pctClass(point.daily_return)}>{formatPct(point.daily_return)}</td>
                                <td>{formatCompactCurrency(point.aum)}</td>
                                <td>
                                    <span className="fund-investor-cell">
                                        <span>{point.investor_count?.toLocaleString('tr-TR') || '-'}</span>
                                        {investorDelta != null ? (
                                            <span className={`fund-investor-delta ${pctClass(investorDelta)}`}>
                                                ({formatSignedInteger(investorDelta)})
                                            </span>
                                        ) : null}
                                    </span>
                                </td>
                            </tr>
                        );
                    })}
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
    const [starredFundCodes, setStarredFundCodes] = useState<Set<string>>(() => new Set());
    const [compareFundCodes, setCompareFundCodes] = useState<Set<string>>(() => new Set());
    const [funds, setFunds] = useState<FundsResponse | null>(null);
    const [categories, setCategories] = useState<FundCategoriesResponse | null>(null);
    const [detail, setDetail] = useState<FundDetail | null>(null);
    const [performance, setPerformance] = useState<FundPerformanceResponse | null>(null);
    const [yieldSummary, setYieldSummary] = useState<FundYieldSummaryResponse | null>(null);
    const [allocations, setAllocations] = useState<FundAllocationsResponse | null>(null);
    const [allocationHistory, setAllocationHistory] = useState<FundAllocationsHistoryResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [detailLoading, setDetailLoading] = useState(false);
    const [performanceLoading, setPerformanceLoading] = useState(false);
    const [yieldLoading, setYieldLoading] = useState(false);
    const [allocationLoading, setAllocationLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [refreshError, setRefreshError] = useState<string | null>(null);
    const [detailError, setDetailError] = useState<string | null>(null);
    const [performanceError, setPerformanceError] = useState<string | null>(null);
    const [yieldError, setYieldError] = useState<string | null>(null);
    const [historySubtab, setHistorySubtab] = useState<FundHistorySubtab>('prices');
    const [allocationHistoryLoading, setAllocationHistoryLoading] = useState(false);
    const [allocationHistoryError, setAllocationHistoryError] = useState<string | null>(null);
    const [chartRange, setChartRange] = useState<FundChartRange>('all');
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
    const activeFundCodeRef = useRef('');
    const autoRefreshAttemptedRef = useRef(false);
    const allocationRefreshAttemptedRef = useRef(new Set<string>());
    const comparisonPanelRef = useRef<HTMLDivElement | null>(null);
    const [comparisonSearchOpenSignal, setComparisonSearchOpenSignal] = useState(0);
    const [comparisonHistory, setComparisonHistory] = useState<MarketComparisonHistoryResponse | null>(null);
    const [comparisonHistoryLoading, setComparisonHistoryLoading] = useState(false);
    const [comparisonHistoryError, setComparisonHistoryError] = useState<string | null>(null);

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
            setRefreshError(err instanceof Error ? err.message : 'Fon listesi yenilenemedi.');
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
        activeFundCodeRef.current = fundCode?.trim().toUpperCase() || '';
    }, [fundCode]);

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
        setChartRange('all');
        setPendingChartRange(null);
        setChartRangeError(null);
        setHistorySubtab('prices');
        setCustomStartDate(isoDateMonthsAgo(6));
        setCustomEndDate(new Date().toISOString().slice(0, 10));
    }, [fundCode]);

    useEffect(() => {
        if (!fundCode) {
            setPerformance(null);
            setPerformanceError(null);
            return;
        }
        let alive = true;
        const normalizedCode = fundCode.trim().toUpperCase();
        setPerformanceLoading(true);
        setPerformanceError(null);
        apiClient
            .fundPerformance(normalizedCode)
            .then((payload) => {
                if (!alive) return;
                setPerformance(payload);
            })
            .catch((err) => {
                if (!alive) return;
                setPerformance(null);
                setPerformanceError(err instanceof Error ? err.message : 'Grafik verisi alınamadı.');
            })
            .finally(() => {
                if (alive) setPerformanceLoading(false);
            });
        return () => {
            alive = false;
        };
    }, [fundCode]);

    useEffect(() => {
        if (!fundCode) {
            setYieldSummary(null);
            setYieldError(null);
            return;
        }
        let alive = true;
        const normalizedCode = fundCode.trim().toUpperCase();
        setYieldLoading(true);
        setYieldError(null);
        apiClient
            .fundYieldSummary(normalizedCode)
            .then((payload) => {
                if (alive) setYieldSummary(payload);
            })
            .catch((err) => {
                if (!alive) return;
                setYieldSummary(null);
                setYieldError(err instanceof Error ? err.message : 'Getiri özeti alınamadı.');
            })
            .finally(() => {
                if (alive) setYieldLoading(false);
            });
        return () => {
            alive = false;
        };
    }, [fundCode]);

    useEffect(() => {
        if (!fundCode) {
            setAllocations(null);
            setAllocationHistory(null);
            setAllocationHistoryError(null);
            setAllocationHistoryLoading(false);
            return;
        }
        setAllocationHistory(null);
        setAllocationHistoryError(null);
        setAllocationHistoryLoading(false);
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
    const latestPerformancePrice = performancePoints.length ? Number(performancePoints[performancePoints.length - 1].price) : null;
    const selectedFundPrice = Number(selectedFund?.price);
    const detailLatestPrice = latestPerformancePrice != null && Number.isFinite(latestPerformancePrice) && latestPerformancePrice > 0
        ? latestPerformancePrice
        : (Number.isFinite(selectedFundPrice) && selectedFundPrice > 0 ? selectedFundPrice : null);
    const detailPeriodReturns = useMemo(
        () => {
            const fromSummary = periodReturnsFromYieldSummary(yieldSummary, detailLatestPrice);
            return Object.keys(fromSummary).length ? fromSummary : selectedFund?.period_returns || {};
        },
        [detailLatestPrice, selectedFund, yieldSummary],
    );
    const comparisonState = useFundComparisonState({
        selectedFund,
        baseReturns: detailPeriodReturns,
        funds: funds?.rows || [],
        openSearchSignal: comparisonSearchOpenSignal,
    });
    const comparisonChartStartDate = chartPoints[0]?.date || null;
    const comparisonChartEndDate = chartPoints[chartPoints.length - 1]?.date || null;
    const comparisonAssets = comparisonState.selectedAssets;
    const comparisonAssetRequests = useMemo(
        () => comparisonAssets.map((asset) => ({
            id: asset.id,
            kind: asset.kind,
            symbol: asset.symbol,
            label: asset.displaySymbol || asset.label,
        })),
        [comparisonAssets],
    );
    const comparisonAssetRequestKey = comparisonAssetRequests
        .map((asset) => `${asset.id}:${asset.kind}:${asset.symbol}:${asset.label || ''}`)
        .join('|');
    useEffect(() => {
        if (!comparisonAssetRequests.length || !comparisonChartStartDate || !comparisonChartEndDate) {
            setComparisonHistory(null);
            setComparisonHistoryError(null);
            setComparisonHistoryLoading(false);
            return;
        }
        if (pendingChartRange) {
            setComparisonHistoryError(null);
            setComparisonHistoryLoading(true);
            return;
        }
        const controller = new AbortController();
        setComparisonHistoryError(null);
        setComparisonHistoryLoading(true);
        apiClient
            .marketComparisonHistory(
                {
                    start_date: comparisonChartStartDate,
                    end_date: comparisonChartEndDate,
                    assets: comparisonAssetRequests,
                },
                { signal: controller.signal },
            )
            .then((payload) => {
                setComparisonHistory(payload);
            })
            .catch((err) => {
                if ((err as Error)?.name === 'AbortError') return;
                setComparisonHistory(null);
                setComparisonHistoryError(err instanceof Error ? err.message : 'Karşılaştırma geçmişi alınamadı.');
            })
            .finally(() => {
                if (!controller.signal.aborted) {
                    setComparisonHistoryLoading(false);
                }
            });
        return () => {
            controller.abort();
        };
    }, [comparisonAssetRequestKey, comparisonChartEndDate, comparisonChartStartDate, pendingChartRange]);
    const detailSourceWarnings = useMemo(() => {
        if (!fundCode) return [];
        const warnings = [
            performanceError,
            yieldError,
            performanceLoading && !performance ? 'Grafik verisi TEFAS üzerinden yükleniyor.' : null,
            yieldLoading && !yieldSummary ? 'Getiri özeti TEFAS üzerinden yükleniyor.' : null,
            ...(performance?.source_metadata?.warnings || []),
            performance?.source_metadata?.warning,
            ...(yieldSummary?.source_metadata?.warnings || []),
            yieldSummary?.source_metadata?.warning,
        ];
        return Array.from(new Set(warnings.filter((item): item is string => Boolean(item))));
    }, [fundCode, performance, performanceError, performanceLoading, yieldError, yieldLoading, yieldSummary]);
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
    const selectedFundCode = selectedFund?.fund_code || fundCode?.trim().toUpperCase() || '';
    const isStarred = selectedFundCode ? starredFundCodes.has(selectedFundCode) : false;
    const isCompared = selectedFundCode ? compareFundCodes.has(selectedFundCode) : false;
    const toggleStarredFund = useCallback(() => {
        if (!selectedFundCode) return;
        setStarredFundCodes((current) => {
            const next = new Set(current);
            if (next.has(selectedFundCode)) {
                next.delete(selectedFundCode);
            } else {
                next.add(selectedFundCode);
            }
            return next;
        });
    }, [selectedFundCode]);
    const openComparisonSearch = useCallback(() => {
        if (!selectedFundCode) return;
        setCompareFundCodes((current) => {
            const next = new Set(current);
            next.add(selectedFundCode);
            return next;
        });
        onTabChange('overview');
        setComparisonSearchOpenSignal((value) => value + 1);
        window.setTimeout(() => {
            comparisonPanelRef.current?.scrollIntoView({ block: 'center', behavior: 'smooth' });
        }, 80);
    }, [onTabChange, selectedFundCode]);

    const loadAllocationHistory = useCallback(() => {
        if (!fundCode) return;
        const normalizedCode = fundCode.trim().toUpperCase();
        setAllocationHistoryError(null);
        if (allocationHistory?.fund_code === normalizedCode && allocationHistory.lookback_days === 30) {
            return;
        }
        if (allocationHistoryLoading) {
            return;
        }
        setAllocationHistoryLoading(true);
        apiClient
            .fundAllocationsHistory(normalizedCode, 30)
            .then((payload) => {
                if (!mountedRef.current || activeFundCodeRef.current !== normalizedCode) return;
                setAllocationHistory(payload);
            })
            .catch((err) => {
                if (!mountedRef.current || activeFundCodeRef.current !== normalizedCode) return;
                setAllocationHistory(null);
                setAllocationHistoryError(err instanceof Error ? err.message : 'Dağılım geçmişi alınamadı.');
            })
            .finally(() => {
                if (!mountedRef.current || activeFundCodeRef.current !== normalizedCode) return;
                setAllocationHistoryLoading(false);
            });
    }, [allocationHistory, allocationHistoryLoading, fundCode]);

    const openAllocationHistory = useCallback(() => {
        setHistorySubtab('allocation');
        onTabChange('history');
        loadAllocationHistory();
    }, [loadAllocationHistory, onTabChange]);

    useEffect(() => {
        if (
            activeTab === 'history'
            && historySubtab === 'allocation'
            && !allocationHistory
            && !allocationHistoryError
            && !allocationHistoryLoading
        ) {
            loadAllocationHistory();
        }
    }, [
        activeTab,
        allocationHistory,
        allocationHistoryError,
        allocationHistoryLoading,
        historySubtab,
        loadAllocationHistory,
    ]);

    const refreshChartRange = useCallback((range: FundChartRange, nextCustomStart = customStartDate, nextCustomEnd = customEndDate) => {
        if (!fundCode) return;
        const normalizedCode = fundCode.trim().toUpperCase();
        if (range === 'all') {
            setChartRange(range);
            setChartRangeError(null);
            if (performancePoints.length >= 2) return;
            setPendingChartRange(range);
            setPerformanceError(null);
            setPerformanceLoading(true);
            apiClient
                .fundPerformance(normalizedCode)
                .then((payload) => {
                    setPerformance(payload);
                })
                .catch((err) => {
                    const message = err instanceof Error ? err.message : 'Grafik verisi alınamadı.';
                    setChartRangeError(message);
                    setPerformanceError(message);
                })
                .finally(() => {
                    setPendingChartRange(null);
                    setPerformanceLoading(false);
                });
            return;
        }
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
        setPerformanceError(null);
        setPerformanceLoading(true);
        apiClient
            .refreshFundPerformance(normalizedCode, startIso, endIso)
            .then((payload) => {
                setPerformance(payload);
            })
            .catch((err) => {
                const message = err instanceof Error ? err.message : 'Grafik verisi alınamadı.';
                setChartRangeError(message);
                setPerformanceError(message);
            })
            .finally(() => {
                setPendingChartRange(null);
                setPerformanceLoading(false);
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
                onSelectFund={(nextFundCode) => onOpenFund(nextFundCode, 'overview')}
            />
            <div className="funds-workspace">
                <div className="funds-page">
                    {!fundCode ? (
                        <>
                            <header className="funds-header">
                                <div>
                                    <span className="funds-kicker">TEFAS snapshot</span>
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
                                        <strong>{funds?.source?.toUpperCase() || 'TEFAS'}</strong>
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
                                            ? `Fon listesi yenilenemedi: ${refreshError}`
                                            : refreshing
                                              ? 'Fon snapshot TEFAS üzerinden yenileniyor.'
                                              : funds?.degraded
                                            ? 'Fon snapshot cache boş veya kullanılamıyor.'
                                            : funds?.stale
                                            ? 'Fon snapshot stale cache üzerinden gösteriliyor.'
                                            : funds?.warnings?.length
                                            ? `Fon verisi TEFAS birincil kaynağından alınamadı; fallback sonucu gösteriliyor. ${funds.warnings[0]}`
                                            : 'Fon kaynak durumu uyarı verdi.'}
                                    </span>
                                    <button
                                        type="button"
                                        className="funds-refresh-button"
                                        onClick={refreshSnapshot}
                                        disabled={refreshing}
                                        title="Fon listesini yenile"
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
                            {detailLoading ? (
                                <div className="funds-state">Fon detayı yükleniyor...</div>
                            ) : detailError ? (
                                <div className="funds-state funds-state-error">{detailError}</div>
                            ) : selectedFund ? (
                                <>
                                    <header className="fund-market-shell">
                                        <div className="fund-market-breadcrumb" aria-label="Fon konumu">
                                            <button type="button" className="fund-breadcrumb-back" onClick={onBack}>
                                                <ArrowLeft size={15} aria-hidden="true" />
                                                Fonlar
                                            </button>
                                            <ChevronRight size={14} aria-hidden="true" />
                                            <span className="fund-breadcrumb-group">
                                                <Database size={17} aria-hidden="true" />
                                                Yatırım Fonları
                                            </span>
                                            <ChevronRight size={14} aria-hidden="true" />
                                            <span className="fund-breadcrumb-code">
                                                <SymbolLogo
                                                    symbol={selectedFund.fund_code}
                                                    name={selectedFund.founder_company || selectedFund.manager_company || selectedFund.name}
                                                    kind="fund"
                                                    size="xs"
                                                />
                                                {selectedFund.fund_code}
                                            </span>
                                        </div>

                                        <div className="fund-market-hero">
                                            <div className="fund-market-title">
                                                <SymbolLogo
                                                    symbol={selectedFund.fund_code}
                                                    name={selectedFund.founder_company || selectedFund.manager_company || selectedFund.name}
                                                    kind="fund"
                                                    size="lg"
                                                    className="fund-market-logo"
                                                />
                                                <div className="fund-market-copy">
                                                    <div className="fund-market-code-row">
                                                        <h1>{selectedFund.fund_code}</h1>
                                                        <button
                                                            type="button"
                                                            className={`fund-icon-action${isStarred ? ' active' : ''}`}
                                                            onClick={toggleStarredFund}
                                                            aria-pressed={isStarred}
                                                            title={isStarred ? 'Favoriden çıkar' : 'Favoriye ekle'}
                                                        >
                                                            <Star size={19} aria-hidden="true" />
                                                        </button>
                                                        <button
                                                            type="button"
                                                            className={`fund-compare-action${isCompared ? ' active' : ''}`}
                                                            onClick={openComparisonSearch}
                                                            aria-pressed={isCompared}
                                                        >
                                                            <Scale size={15} aria-hidden="true" />
                                                            Karşılaştır
                                                        </button>
                                                    </div>
                                                    <p>{selectedFund.name}</p>
                                                    <small>
                                                        {selectedFund.fund_type || 'Fon'}
                                                        {selectedFund.founder_company || selectedFund.manager_company
                                                            ? ` · ${selectedFund.founder_company || selectedFund.manager_company}`
                                                            : ''}
                                                    </small>
                                                </div>
                                            </div>

                                            <div className="fund-market-quote">
                                                <div>
                                                    <strong>{formatQuotePrice(selectedFund.price)}</strong>
                                                    <span className={pctClass(selectedFund.daily_return)}>{formatPct(selectedFund.daily_return)}</span>
                                                </div>
                                                <small>{formatDate(selectedFund.as_of)}</small>
                                            </div>
                                        </div>

                                        <div className="fund-market-stats">
                                            <div><span>Portföy</span><strong>{formatCompactCurrency(selectedFund.aum)}</strong></div>
                                            <div><span>Yatırımcı</span><strong>{selectedFund.investor_count?.toLocaleString('tr-TR') || '-'}</strong></div>
                                            <div><span>Risk</span><strong>{selectedFund.risk_value ?? '-'}</strong></div>
                                            <div><span>Kaynak</span><strong>{selectedFund.source?.toUpperCase() || 'TEFAS'}</strong></div>
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
                                    </header>

                                    {detailSourceWarnings.length > 0 && (
                                        <div className="funds-source-warning">
                                            <ShieldAlert size={17} aria-hidden="true" />
                                            <span>{detailSourceWarnings.slice(0, 2).join(' ')}</span>
                                        </div>
                                    )}

                                    {activeTab === 'overview' && (
                                        <div className="fund-overview-stack">
                                            <div className="fund-overview-grid">
                                                <FundPerformanceChart
                                                    fundCode={selectedFund.fund_code}
                                                    points={chartPoints}
                                                    comparisonAssets={comparisonState.selectedAssets}
                                                    comparisonHistory={comparisonHistory}
                                                    comparisonLoading={comparisonHistoryLoading}
                                                    comparisonError={comparisonHistoryError}
                                                    comparison={comparisonState}
                                                    selectedRange={chartRange}
                                                    pendingRange={pendingChartRange}
                                                    rangeError={chartRangeError}
                                                    customStartDate={customStartDate}
                                                    customEndDate={customEndDate}
                                                    periodReturns={detailPeriodReturns}
                                                    yieldPeriods={yieldSummary?.periods}
                                                    currency={selectedFund.currency}
                                                    onRangeSelect={handleChartRangeSelect}
                                                    onCustomStartDateChange={handleCustomStartDateChange}
                                                    onCustomEndDateChange={handleCustomEndDateChange}
                                                />
                                                <FundAllocationSummary
                                                    allocations={allocationRows}
                                                    loading={allocationLoading && !allocations}
                                                    onOpenHistory={openAllocationHistory}
                                                    historyLoading={allocationHistoryLoading}
                                                />
                                            </div>
                                            <FundReturnComparison
                                                comparison={comparisonState}
                                                containerRef={comparisonPanelRef}
                                            />
                                            {yieldLoading && !yieldSummary && (
                                                <div className="funds-state">Getiri kartları TEFAS üzerinden yükleniyor...</div>
                                            )}
                                            <FundMonthlyHeatmap monthlyReturns={monthlyReturns} />
                                        </div>
                                    )}

                                    {activeTab === 'allocation' && (
                                        <div className="fund-detail-panel">
                                            <h2>Portföy Dağılımı</h2>
                                            {allocationRows.length ? (
                                                <div className="fund-allocation-detail">
                                                    <FundAllocationSummary
                                                        allocations={allocationRows}
                                                        loading={allocationLoading && !allocations}
                                                        onOpenHistory={openAllocationHistory}
                                                        historyLoading={allocationHistoryLoading}
                                                    />
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
                                                    {allocationLoading ? 'Dağılım verisi yükleniyor...' : 'TEFAS dağılım verisi bu fon için yok.'}
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {activeTab === 'history' && (
                                        <div className="fund-detail-panel fund-history-panel">
                                            <div className="fund-history-head">
                                                <h2>Geçmiş Veriler</h2>
                                                <div className="fund-history-tabs" role="tablist" aria-label="Geçmiş veri türü">
                                                    {FUND_HISTORY_TABS.map((tab) => {
                                                        const Icon = tab.icon;
                                                        return (
                                                            <button
                                                                key={tab.key}
                                                                type="button"
                                                                role="tab"
                                                                aria-selected={historySubtab === tab.key}
                                                                className={historySubtab === tab.key ? 'active' : ''}
                                                                onClick={() => setHistorySubtab(tab.key)}
                                                            >
                                                                <Icon size={15} aria-hidden="true" />
                                                                {tab.label}
                                                            </button>
                                                        );
                                                    })}
                                                </div>
                                            </div>
                                            {historySubtab === 'prices' ? (
                                                performanceLoading && !performancePoints.length ? (
                                                    <div className="funds-state">Geçmiş fiyat verisi TEFAS üzerinden yükleniyor...</div>
                                                ) : performanceError ? (
                                                    <div className="funds-state funds-state-error">{performanceError}</div>
                                                ) : performancePoints.length ? (
                                                    <FundHistoryTable points={performancePoints} />
                                                ) : (
                                                    <div className="funds-state">Geçmiş fiyat verisi henüz hazır değil.</div>
                                                )
                                            ) : (
                                                <FundAllocationHistoryPanel
                                                    history={allocationHistory}
                                                    loading={allocationHistoryLoading}
                                                    error={allocationHistoryError}
                                                />
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
