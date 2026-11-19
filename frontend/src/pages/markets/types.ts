import type {
    MarketIndexCode,
    MarketIndexConstituent,
    MarketIndexListRow,
    MarketStockCardsResponse,
    MarketStockIndex,
    MarketUniverseResponse,
} from '../../api/types';
import type {
    MarketsNavigationFundSection,
    MarketsNavigationSection,
    MarketsNavigationStockSection,
} from '../../components/MarketsNavigation';
import type { StockReturnMode } from '../../routing/routes';

export type MarketSection = MarketsNavigationSection;
export type SortDirection = 'asc' | 'desc';
export type IndexConstituentDataSortKey =
    | 'symbol'
    | 'price'
    | 'change_pct'
    | 'volume'
    | 'weight_pct'
    | 'point_effect';
export type IndexConstituentSortKey = IndexConstituentDataSortKey | 'impact_pct' | 'impact_abs';
export type StockCardDropPlacement = 'before' | 'after';

export type StockSortKey =
    | 'company'
    | 'sector'
    | 'fundamental_period'
    | 'fk'
    | 'pd_dd'
    | 'fd_favok'
    | 'fd_satis'
    | 'net_borc_favok'
    | 'roe_pct'
    | 'free_float_pct'
    | 'price'
    | 'change_pct'
    | 'volume'
    | 'market_cap'
    | 'return_1w_pct'
    | 'return_1m_pct'
    | 'return_3m_pct'
    | 'return_6m_pct'
    | 'return_ytd_pct'
    | 'return_1y_pct'
    | 'ttm_net_kar'
    | 'ttm_fcf'
    | 'ttm_satis_gelirleri'
    | 'ttm_favok'
    | 'net_borc'
    | 'ozkaynaklar'
    | 'hist_fk_avg'
    | 'hist_pd_dd_avg'
    | 'hist_fd_favok_avg'
    | 'fk_hist_diff_pct'
    | 'pd_dd_hist_diff_pct'
    | 'fd_favok_hist_diff_pct'
    | 'sma_50_diff_pct'
    | 'sma_200_diff_pct'
    | 'high_52w_dist_pct'
    | 'low_52w_dist_pct'
    | 'fk_sector_diff_pct';

export type IndexSortKey = keyof MarketIndexListRow;

export type StockReturnKey = Extract<
    StockSortKey,
    'return_1w_pct' | 'return_1m_pct' | 'return_3m_pct' | 'return_6m_pct' | 'return_ytd_pct' | 'return_1y_pct'
>;

export type StockTableMode = 'performance' | 'fundamentals';
export type StockScreenerCategory = 'all' | 'profile' | 'valuation' | 'financials' | 'market' | 'historical';

export type StockScreenerPresetId =
    | 'discount'
    | 'low_debt'
    | 'net_cash'
    | 'roe_value'
    | 'below_book'
    | 'positive_cash_flow'
    | 'high_margins'
    | 'momentum_leaders'
    | 'large_liquid'
    | 'hist_discount'
    | 'trend_above_200'
    | 'sector_discount';

export type StockScreenerRangeKey =
    | 'min_fk' | 'max_fk'
    | 'min_pd_dd' | 'max_pd_dd'
    | 'min_fd_favok' | 'max_fd_favok'
    | 'min_fd_satis' | 'max_fd_satis'
    | 'min_net_borc_favok' | 'max_net_borc_favok'
    | 'min_roe_pct' | 'max_roe_pct'
    | 'min_favok_margin_pct' | 'max_favok_margin_pct'
    | 'min_net_margin_pct' | 'max_net_margin_pct'
    | 'min_ttm_net_kar' | 'max_ttm_net_kar'
    | 'min_ttm_fcf' | 'max_ttm_fcf'
    | 'min_ttm_satis' | 'max_ttm_satis'
    | 'min_ttm_favok' | 'max_ttm_favok'
    | 'min_net_borc' | 'max_net_borc'
    | 'min_ozkaynaklar' | 'max_ozkaynaklar'
    | 'min_market_cap' | 'max_market_cap'
    | 'min_volume' | 'max_volume'
    | 'min_free_float_pct' | 'max_free_float_pct'
    | 'min_price' | 'max_price'
    | 'min_change_pct' | 'max_change_pct'
    | 'min_return_1w_pct' | 'max_return_1w_pct'
    | 'min_return_1m_pct' | 'max_return_1m_pct'
    | 'min_return_3m_pct' | 'max_return_3m_pct'
    | 'min_return_ytd_pct' | 'max_return_ytd_pct'
    | 'min_return_1y_pct' | 'max_return_1y_pct'
    | 'min_hist_fk_avg' | 'max_hist_fk_avg'
    | 'min_hist_pd_dd_avg' | 'max_hist_pd_dd_avg'
    | 'min_hist_fd_favok_avg' | 'max_hist_fd_favok_avg'
    | 'min_fk_hist_diff_pct' | 'max_fk_hist_diff_pct'
    | 'min_pd_dd_hist_diff_pct' | 'max_pd_dd_hist_diff_pct'
    | 'min_fd_favok_hist_diff_pct' | 'max_fd_favok_hist_diff_pct'
    | 'min_sma_50_diff_pct' | 'max_sma_50_diff_pct'
    | 'min_sma_200_diff_pct' | 'max_sma_200_diff_pct'
    | 'min_high_52w_dist_pct' | 'max_high_52w_dist_pct'
    | 'min_low_52w_dist_pct' | 'max_low_52w_dist_pct'
    | 'min_fk_sector_diff_pct' | 'max_fk_sector_diff_pct'
    | 'min_pd_dd_sector_diff_pct' | 'max_pd_dd_sector_diff_pct'
    | 'min_fd_favok_sector_diff_pct' | 'max_fd_favok_sector_diff_pct';

export type StockScreenerToggleKey =
    | 'positive_net_profit'
    | 'positive_ebitda'
    | 'positive_free_cash_flow'
    | 'net_cash_only'
    | 'positive_roe'
    | 'only_with_financials'
    | 'below_hist_fk'
    | 'below_hist_pd_dd'
    | 'above_sma_50'
    | 'above_sma_200'
    | 'below_sector_fk'
    | 'below_sector_pd_dd';

export type StockScreenerFilterState = { sector: string } & Record<StockScreenerRangeKey, string> & Record<StockScreenerToggleKey, boolean>;

export type StockSectorDiffFields = {
    fk_sector_diff_pct: number | null;
    pd_dd_sector_diff_pct: number | null;
    fd_favok_sector_diff_pct: number | null;
};

export type StockTableColumn = {
    key: StockSortKey;
    label: string;
    sublabel?: string;
    align?: 'left' | 'right';
    width?: number;
};

export type ScreenerRangeSpec = {
    card: 'valuation' | 'profitability' | 'scale' | 'market_scale' | 'returns' | 'hist_multiples' | 'technicals' | 'sector_relative';
    category: Exclude<StockScreenerCategory, 'all' | 'profile'>;
    field:
        | 'fk' | 'pd_dd' | 'fd_favok' | 'fd_satis'
        | 'roe_pct' | 'favok_margin_pct' | 'net_margin_pct' | 'net_borc_favok'
        | 'ttm_satis_gelirleri' | 'ttm_favok' | 'ttm_net_kar' | 'ttm_fcf' | 'ozkaynaklar' | 'net_borc'
        | 'market_cap' | 'volume' | 'free_float_pct' | 'price'
        | 'change_pct' | 'return_1w_pct' | 'return_1m_pct' | 'return_3m_pct' | 'return_ytd_pct' | 'return_1y_pct'
        | 'hist_fk_avg' | 'hist_pd_dd_avg' | 'hist_fd_favok_avg'
        | 'fk_hist_diff_pct' | 'pd_dd_hist_diff_pct' | 'fd_favok_hist_diff_pct'
        | 'sma_50_diff_pct' | 'sma_200_diff_pct' | 'high_52w_dist_pct' | 'low_52w_dist_pct'
        | 'fk_sector_diff_pct' | 'pd_dd_sector_diff_pct' | 'fd_favok_sector_diff_pct';
    minKey: StockScreenerRangeKey;
    maxKey: StockScreenerRangeKey;
    label: string;
    hint: string;
    unit: 'x' | '%' | 'mn ₺' | '₺';
    scale?: number;
};

export type IndexGroupFilter = 'all' | 'main' | 'sector';

export type MarketUniverseMemoryCache = { data: MarketUniverseResponse; fetchedAt: number };
export type MarketStockCardsMemoryCache = { data: MarketStockCardsResponse; fetchedAt: number };

export type HeatmapTile = {
    row: MarketIndexConstituent;
    impactPct: number | null;
    changePct: number | null;
    pointEffect: number | null;
    left: number;
    top: number;
    width: number;
    height: number;
    areaPct: number;
};

export type HeatmapLayoutItem = {
    row: MarketIndexConstituent;
    area: number;
    impactPct: number | null;
    changePct: number | null;
};

export type HeatmapLayoutRect = {
    x: number;
    y: number;
    width: number;
    height: number;
};

export type ScreenerActiveChip = {
    id: string;
    label: string;
    onClear: () => void;
};

export type SectorAverageSummary = {
    fk: number | null;
    pd_dd: number | null;
    fd_favok: number | null;
    count: number;
};

export type ScreenerComputedData = {
    rows: import('../../api/types').MarketStockRow[];
    sectors: string[];
    sectorCounts: Map<string, number>;
    sectorAverages: Map<string, SectorAverageSummary>;
    sectorDiffByCompany: Map<string, StockSectorDiffFields>;
    totalCount: number;
    financialCount: number;
};

export interface MarketsViewProps {
    routeSection?: MarketSection;
    routeStockIndex?: MarketStockIndex;
    routeSelectedIndex?: MarketIndexCode | null;
    routeReturnMode?: StockReturnMode;
    onNavigateSection?: (section: MarketSection) => void;
    onNavigateStockSection?: (section: MarketsNavigationStockSection) => void;
    onNavigateFundSection?: (section: MarketsNavigationFundSection) => void;
    onNavigateStockIndex?: (index: MarketStockIndex) => void;
    onNavigateReturnMode?: (mode: StockReturnMode) => void;
    onNavigateIndexDetail?: (index: MarketIndexCode | null) => void;
    onOpenTicker?: (ticker: string) => void;
    onOpenFund?: (fundCode: string) => void;
}
