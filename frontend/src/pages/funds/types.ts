import type { Dispatch, RefObject, SetStateAction } from 'react';
import { BarChart3, BriefcaseBusiness, LineChart } from 'lucide-react';
import type {
    FundAllocationsResponse,
    FundCategoriesResponse,
    FundDetail,
    FundHoldingsResponse,
    FundPerformanceResponse,
    FundsResponse,
} from '../../api/types';
import type { MarketsNavigationFundSection, MarketsNavigationSection } from '../../components/MarketsNavigation';
import type { FundTab } from '../../routing/routes';

export type FundSortKey =
    | 'fund_code'
    | 'name'
    | 'fund_type'
    | 'founder_company'
    | 'price'
    | 'daily_return'
    | 'return_1w'
    | 'return_1m'
    | 'return_3m'
    | 'return_6m'
    | 'return_ytd'
    | 'return_1y'
    | 'risk_value'
    | 'aum'
    | 'investor_count'
    | 'as_of';

export type FundTableMode = 'summary' | 'performance';
export type FundScreenerPresetId =
    | 'high_1y'
    | 'steady_momentum'
    | 'short_term_move'
    | 'low_risk'
    | 'large_aum'
    | 'popular_investors';
export type FundScreenerRangeKey =
    | 'min_daily' | 'max_daily'
    | 'min_1w' | 'max_1w'
    | 'min_1m' | 'max_1m'
    | 'min_3m' | 'max_3m'
    | 'min_6m' | 'max_6m'
    | 'min_ytd' | 'max_ytd'
    | 'min_1y' | 'max_1y'
    | 'min_risk' | 'max_risk'
    | 'min_aum_m' | 'max_aum_m'
    | 'min_investors' | 'max_investors';
export type FundScreenerFilterState = {
    founder: string;
} & Record<FundScreenerRangeKey, string>;

export const EMPTY_FUND_SCREENER_FILTERS: FundScreenerFilterState = {
    founder: '',
    min_daily: '', max_daily: '',
    min_1w: '', max_1w: '',
    min_1m: '', max_1m: '',
    min_3m: '', max_3m: '',
    min_6m: '', max_6m: '',
    min_ytd: '', max_ytd: '',
    min_1y: '', max_1y: '',
    min_risk: '', max_risk: '',
    min_aum_m: '', max_aum_m: '',
    min_investors: '', max_investors: '',
};

export const FUND_SCREENER_PRESETS: Array<{
    id: FundScreenerPresetId;
    label: string;
    filters: Partial<FundScreenerFilterState>;
    fundType?: string;
    risk?: string;
}> = [
    { id: 'high_1y', label: 'Yüksek Getiri (1Y %50+)', filters: { min_1y: '50' } },
    { id: 'steady_momentum', label: 'Güçlü Momentum (1A & 3A)', filters: { min_1m: '3', min_3m: '10' } },
    { id: 'short_term_move', label: 'Kısa Vadeli İvme (1H %2+)', filters: { min_1w: '2' } },
    { id: 'low_risk', label: 'Düşük Risk (1–3)', filters: { min_risk: '1', max_risk: '3' } },
    { id: 'large_aum', label: 'Büyük Ölçek (1 Mr ₺+)', filters: { min_aum_m: '1000' } },
    { id: 'popular_investors', label: 'Popüler Fonlar (10B+ Yatırımcı)', filters: { min_investors: '10000' } },
];

export type FundScreenerRangeSpec = {
    label: string;
    minKey: FundScreenerRangeKey;
    maxKey: FundScreenerRangeKey;
    minPh: string;
    maxPh: string;
    unit?: string;
    group: 'returns' | 'scale';
};

export const FUND_SCREENER_RANGE_SPECS: FundScreenerRangeSpec[] = [
    { label: 'Günlük Getiri', minKey: 'min_daily', maxKey: 'max_daily', minPh: '-5', maxPh: '5', unit: '%', group: 'returns' },
    { label: '1 Hafta Getiri', minKey: 'min_1w', maxKey: 'max_1w', minPh: '-10', maxPh: '15', unit: '%', group: 'returns' },
    { label: '1 Ay Getiri', minKey: 'min_1m', maxKey: 'max_1m', minPh: '0', maxPh: '25', unit: '%', group: 'returns' },
    { label: '3 Ay Getiri', minKey: 'min_3m', maxKey: 'max_3m', minPh: '0', maxPh: '40', unit: '%', group: 'returns' },
    { label: '6 Ay Getiri', minKey: 'min_6m', maxKey: 'max_6m', minPh: '0', maxPh: '80', unit: '%', group: 'returns' },
    { label: 'Yılbaşı Getiri', minKey: 'min_ytd', maxKey: 'max_ytd', minPh: '0', maxPh: '100', unit: '%', group: 'returns' },
    { label: '1 Yıl Getiri', minKey: 'min_1y', maxKey: 'max_1y', minPh: '20', maxPh: '150', unit: '%', group: 'returns' },
    { label: 'Risk Değeri', minKey: 'min_risk', maxKey: 'max_risk', minPh: '1', maxPh: '7', group: 'scale' },
    { label: 'Portföy Büyüklüğü', minKey: 'min_aum_m', maxKey: 'max_aum_m', minPh: '100', maxPh: '50000', unit: 'Mn ₺', group: 'scale' },
    { label: 'Yatırımcı Sayısı', minKey: 'min_investors', maxKey: 'max_investors', minPh: '1000', maxPh: '100000', unit: 'kişi', group: 'scale' },
];

export type FundsPageProps = {
    view?: 'list' | 'compare';
    fundCode?: string | null;
    activeTab?: FundTab;
    onOpenFund: (fundCode: string, tab?: FundTab) => void;
    onTabChange: (tab: FundTab) => void;
    onBack: () => void;
    onNavigateSection: (section: MarketsNavigationSection) => void;
    onNavigateFundSection?: (section: MarketsNavigationFundSection) => void;
    onOpenTicker?: (ticker: string) => void;
};

export const FUND_TABS: Array<{ key: FundTab; label: string; icon: typeof BarChart3 }> = [
    { key: 'overview', label: 'Genel Bakış', icon: BarChart3 },
    { key: 'allocation', label: 'Portföy Dağılımı', icon: BriefcaseBusiness },
    { key: 'history', label: 'Geçmiş Veriler', icon: LineChart },
];

export type FundHistorySubtab = 'prices' | 'allocation';

export const FUND_HOLDINGS_LIVE_REFRESH_MS = 30_000;
export const FUND_HOLDINGS_LIVE_MAX_BACKOFF_MS = 60_000;
// KAP portfolio reports publish monthly and have no push/webhook channel.
// Revalidate at a low cadence while visible so newly published reports still surface without polling storms.
export const FUND_HOLDINGS_REPORT_REFRESH_MS = 15 * 60_000;
export const FUND_HOLDINGS_LIVE_FLASH_MS = 900;
// The first chart view is six months. The backend may still warm the shared
// history job up to one year, but that wider fetch must not change the UI range.
export const FUND_INITIAL_PERFORMANCE_MONTHS = 6;

export const FUND_HISTORY_TABS: Array<{ key: FundHistorySubtab; label: string; icon: typeof LineChart }> = [
    { key: 'prices', label: 'Fiyat / Yatırımcı', icon: LineChart },
    { key: 'allocation', label: 'Fon Dağılımı', icon: BriefcaseBusiness },
];

export type FundHoldingAssetCategory = 'local_equity' | 'foreign_equity' | 'fund' | 'etf';
export type FundHoldingCategoryFilter = 'all' | FundHoldingAssetCategory;
export type FundHoldingChangeFilter = 'all' | 'increased' | 'decreased' | 'new' | 'removed';
export type FundHoldingSortKey =
    | 'asset'
    | 'price'
    | 'return_pct'
    | 'estimated_pnl_value'
    | 'weight'
    | 'previous_weight'
    | 'weight_change';
export type FundHoldingSortDirection = 'asc' | 'desc';

export const FUND_HOLDING_SORT_COLUMNS: Array<{ key: FundHoldingSortKey; label: string }> = [
    { key: 'asset', label: 'Varlık' },
    { key: 'price', label: 'Fiyat' },
    { key: 'return_pct', label: 'Getiri %' },
    { key: 'estimated_pnl_value', label: 'K/Z' },
    { key: 'weight', label: 'Ağırlık %' },
    { key: 'previous_weight', label: 'Önceki Ay %' },
    { key: 'weight_change', label: 'Fark' },
];

export const FUND_HOLDING_CATEGORY_FILTERS: Array<{ key: FundHoldingCategoryFilter; label: string }> = [
    { key: 'all', label: 'Tümü' },
    { key: 'local_equity', label: 'Hisseler' },
    { key: 'foreign_equity', label: 'Yabancı' },
    { key: 'fund', label: 'Fonlar/BYF' },
    { key: 'etf', label: 'ETF' },
];

export const FUND_HOLDING_GROUPS: Array<{ key: FundHoldingAssetCategory; label: string }> = [
    { key: 'local_equity', label: 'Hisseler' },
    { key: 'fund', label: 'BYF / Fonlar' },
    { key: 'foreign_equity', label: 'Yabancı Hisseler' },
    { key: 'etf', label: 'ETF' },
];

export const FUND_HOLDING_CHANGE_FILTERS: Array<{ key: FundHoldingChangeFilter; label: string }> = [
    { key: 'all', label: 'Tümü' },
    { key: 'increased', label: 'Artanlar' },
    { key: 'decreased', label: 'Azalanlar' },
    { key: 'new', label: 'Yeni' },
    { key: 'removed', label: 'Çıkarılan' },
];

export type FundChartRange = 'all' | '1w' | '1m' | '3m' | '6m' | 'ytd' | '1y' | '3y' | '5y' | 'custom';

export const DETAIL_RETURN_PERIODS = [
    { key: '1w', label: '1H' },
    { key: '1m', label: '1A' },
    { key: '3m', label: '3A' },
    { key: '6m', label: '6A' },
    { key: 'ytd', label: 'YBB' },
    { key: '1y', label: '1Y' },
] as const;

export type ComparisonPeriod = typeof DETAIL_RETURN_PERIODS[number]['key'];
export type ComparisonAssetKind = 'fund' | 'stock' | 'index' | 'fx';
export type FundComparisonControlSurface = 'chart' | 'returns';

export type ComparisonAsset = {
    id: string;
    kind: ComparisonAssetKind;
    symbol: string;
    displaySymbol?: string;
    label: string;
    logoName?: string | null;
    logoUrl?: string | null;
    returns: Partial<Record<ComparisonPeriod, number | null>>;
};

export type FundComparisonState = {
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

export type FundChartComparisonSeries = {
    asset: ComparisonAsset;
    points: Array<{ date: string; value: number }>;
    source?: string | null;
    error?: string | null;
};

export type ComparisonReturnSource = {
    change_pct?: number | null;
    return_1w_pct?: number | null;
    return_1m_pct?: number | null;
    return_3m_pct?: number | null;
    return_6m_pct?: number | null;
    return_ytd_pct?: number | null;
    return_1y_pct?: number | null;
};

export const FUND_DONUT_COLORS = ['#4f46e5', '#a3e635', '#818cf8', '#84cc16', '#f472b6', '#8b5cf6', '#c084fc', '#14b8a6'];
export const DEFAULT_COMPARISON_IDS: string[] = [];
export const FUND_CHART_COMPARISON_COLORS = ['#22c55e', '#60a5fa', '#f59e0b', '#f472b6', '#a78bfa', '#14b8a6', '#f97316', '#e879f9'];

export const STOCK_SEARCH_LABELS: Record<string, string[]> = {
    ASTOR: ['Astor Enerji A.Ş.'],
    BIMAS: ['BİM Birleşik Mağazalar A.Ş.', 'BIM'],
    MGROS: ['Migros Ticaret A.Ş.', 'Migros', 'MIGROS'],
    ORGE: ['Orge Enerji Elektrik Taahhüt A.Ş.'],
    SOKM: ['Şok Marketler Ticaret A.Ş.', 'SOK'],
    TAVHL: ['TAV Havalimanları Holding A.Ş.', 'TAV'],
    YEOTK: ['YEO Teknoloji Enerji ve Endüstri A.Ş.'],
};

export type FundsCatalogPayload = {
    funds: FundsResponse;
    categories: FundCategoriesResponse;
};

export type FundsCatalogMemoryCache = FundsCatalogPayload & { fetchedAt: number };

export type MonthlyReturn = {
    month: string;
    label: string;
    year: number;
    monthIndex: number;
    returnPct: number;
    startPrice: number;
    endPrice: number;
};

export type FundOverviewMetricPoint = {
    month: string;
    label: string;
    value: number | null;
};

export type FundOverviewMetricSeries = {
    aum: FundOverviewMetricPoint[];
    investors: FundOverviewMetricPoint[];
    cashFlow: FundOverviewMetricPoint[];
};

export const FUND_TABLE_ROW_HEIGHT = 44;

export type FundHoldingsReportMeta = {
    report_date?: string | null;
    source_url?: string | null;
    file_name?: string | null;
};

export type AllocationHistoryTrend = {
    allocationType: string;
    label: string;
    color: string;
    startWeight: number | null;
    endWeight: number | null;
    delta: number | null;
    points: AllocationHistoryPoint[];
};

export type AllocationHistoryPoint = {
    date: string;
    weight: number | null;
};

export type FundCompareRange = '1w' | '1m' | '3m' | 'ytd' | '6m' | '1y' | '3y' | '5y' | 'all';

export const FUND_COMPARE_MAX_FUNDS = 4;
export const FUND_COMPARE_COLORS = ['#1f6bff', '#f59e0b', '#22c55e', '#a855f7'];
export const FUND_COMPARE_RANGES: Array<{ key: FundCompareRange; label: string }> = [
    { key: '1w', label: '1 Hafta' },
    { key: '1m', label: '1 Ay' },
    { key: '3m', label: '3 Ay' },
    { key: 'ytd', label: String(new Date().getFullYear()) },
    { key: '6m', label: '6 Ay' },
    { key: '1y', label: '1 Yıl' },
    { key: '3y', label: '3 Yıl' },
    { key: '5y', label: '5 Yıl' },
    { key: 'all', label: 'Tüm' },
];

export type FundCompareDataset = {
    detail: FundDetail | null;
    performance: FundPerformanceResponse | null;
    allocations: FundAllocationsResponse | null;
    holdings: FundHoldingsResponse | null;
    detailLoading: boolean;
    performanceLoading: boolean;
    allocationsLoading: boolean;
    holdingsLoading: boolean;
    detailError: string | null;
    performanceError: string | null;
    allocationsError: string | null;
    holdingsError: string | null;
    loading: boolean;
    error: string | null;
};

export type FundCompareCell = {
    primary: string;
    secondary?: string;
    className?: string;
};

export type FundCompareTableRow = {
    label: string;
    values: FundCompareCell[];
};
