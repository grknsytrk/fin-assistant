export interface KapCompanySearchItem {
  symbol: string;
  title: string | null;
  aliases?: string[];
  latest_quarter?: string | null;
  has_kap_cache?: boolean;
}

export interface KapCompaniesResponse {
  companies: string[];
  items?: KapCompanySearchItem[];
}

export interface MarketUniverseStats {
  index: MarketStockIndex;
  index_count: number;
  bist100_count: number;
  bist_all_count: number;
  kap_cache_count: number;
}

export interface MarketUniverseRow {
  company: string;
  latest_quarter: string | null;
  has_kap_cache: boolean;
  price: number | null;
  price_currency: string | null;
  change: number | null;
  change_pct: number | null;
  price_as_of: string | null;
  market_cap: number | null;
  logo_url?: string | null;
  logo_source?: 'kap' | null;
}

export interface MarketStockRow extends MarketUniverseRow {
  volume: number | null;
  return_1w_pct: number | null;
  return_1m_pct: number | null;
  return_3m_pct: number | null;
  return_6m_pct: number | null;
  return_ytd_pct: number | null;
  return_1y_pct: number | null;
}

export type MarketStockIndex = 'XUTUM' | 'XU100' | 'XU030';
export type MarketSectorIndexCode =
  | 'XUSIN'
  | 'XUHIZ'
  | 'XUMAL'
  | 'XUTEK'
  | 'XBANK'
  | 'XAKUR'
  | 'XBLSM'
  | 'XELKT'
  | 'XFINK'
  | 'XGMYO'
  | 'XGIDA'
  | 'XHOLD'
  | 'XILTM'
  | 'XINSA'
  | 'XKAGT'
  | 'XKMYA'
  | 'XMADN'
  | 'XMANA'
  | 'XMESY'
  | 'XSGRT'
  | 'XSPOR'
  | 'XTAST'
  | 'XTCRT'
  | 'XTEKS'
  | 'XTRZM'
  | 'XULAS'
  | 'XYORT';
export type MarketIndexCode = MarketStockIndex | MarketSectorIndexCode;

export interface MarketReturnBenchmark {
  return_1w_pct: number | null;
  return_1m_pct: number | null;
  return_3m_pct: number | null;
  return_6m_pct: number | null;
  return_ytd_pct: number | null;
  return_1y_pct: number | null;
  as_of: string | null;
}

export interface MarketUniverseResponse {
  stats: MarketUniverseStats;
  universe?: {
    index: MarketStockIndex;
    count: number;
    source?: string | null;
    source_url?: string | null;
    source_date?: string | null;
    fetched_at?: string | null;
    cache_hit?: boolean;
    fallback_used?: boolean;
  };
  rows: MarketUniverseRow[];
  coverage_rows: MarketUniverseRow[];
}

export interface MarketStocksResponse {
  index: MarketStockIndex;
  rows: MarketStockRow[];
  benchmarks: Record<MarketStockIndex, MarketReturnBenchmark>;
  source: string;
  universe?: {
    index: MarketStockIndex;
    count: number;
    source?: string | null;
    source_url?: string | null;
    source_date?: string | null;
    fetched_at?: string | null;
    cache_hit?: boolean;
    fallback_used?: boolean;
  };
  as_of: string;
}

export interface FundSourceMetadata {
  source: string;
  source_url?: string | null;
  fetched_at?: string | null;
  as_of?: string | null;
  cache_hit?: boolean;
  cache_policy?: string | null;
  stale?: boolean;
  parse_status?: string | null;
  snapshot_as_of?: string | null;
  price_history_as_of?: string | null;
  price_history_overlay_count?: number;
  price_history_overlay_source?: string | null;
  price_history_fetched_at?: string | null;
  warnings?: string[];
  warning?: string | null;
  history_source_used?: string | null;
  summary_source_used?: string | null;
  history_source_policy?: string | null;
  primary_source?: string | null;
  fallback_reason?: string | null;
  tefasfon_adapter_version?: string | null;
  final_points_count?: number | null;
  date_min?: string | null;
  date_max?: string | null;
  backfill_used?: boolean;
  full_history_requested?: boolean;
  requested_start_date?: string | null;
  requested_end_date?: string | null;
  fallback_used?: boolean;
  cached_fallback_points_present?: boolean;
  cached_fallback_point_count?: number | null;
  source_policy?: string | null;
  adapter_version?: string | null;
  holdings_quality?: {
    status?: string | null;
    normalized_position_count?: number | null;
    raw_total_weight?: number | null;
    adjusted_total_weight?: number | null;
    normalization?: {
      action?: string | null;
      factor?: number | null;
      reason?: string | null;
    } | null;
  } | null;
}

export interface FundPeriodReturns {
  '1w'?: number | null;
  '1m'?: number | null;
  '3m'?: number | null;
  '6m'?: number | null;
  ytd?: number | null;
  '1y'?: number | null;
}

export type FundYieldPeriodKey = '1w' | '1m' | '3m' | '6m' | 'ytd' | '1y' | '3y' | '5y' | 'oldest';

export interface FundYieldPeriodSummary {
  prev_close_date: string | null;
  prev_close: number | null;
  high: number | null;
  low: number | null;
}

export interface FundYieldSummaryResponse {
  fund_code: string;
  status: string;
  source: string;
  source_url?: string | null;
  periods: Partial<Record<FundYieldPeriodKey, FundYieldPeriodSummary>>;
  source_metadata: FundSourceMetadata & {
    purpose?: string;
    writes_fund_prices?: boolean;
  };
}

export interface FundSummary {
  fund_code: string;
  name: string;
  fund_type: string | null;
  founder_company: string | null;
  manager_company: string | null;
  price: number | null;
  daily_return: number | null;
  period_returns: FundPeriodReturns;
  risk_value: number | null;
  currency: string;
  as_of: string | null;
  source: string;
  aum?: number | null;
  investor_count?: number | null;
  share_count?: number | null;
  management_fee?: number | null;
  management_fee_applied?: number | null;
  management_fee_prospectus?: number | null;
  total_expense_ratio?: number | null;
  tax_info?: string | null;
  isin?: string | null;
}

export interface FundCategoryRankingItem {
  key: string;
  label: string;
  value: number | null;
  rank: number;
  total: number;
  top_percentile: number | null;
  direction: string;
}

export interface FundCategoryRankings {
  category: string | null;
  category_total: number;
  as_of: string | null;
  items: FundCategoryRankingItem[];
}

export interface FundDetail extends FundSummary {
  strategy: string | null;
  benchmark: string | null;
  management_fee: number | null;
  management_fee_applied?: number | null;
  management_fee_prospectus?: number | null;
  total_expense_ratio?: number | null;
  tax_info: string | null;
  fintables_url: string | null;
  kap_url: string | null;
  category_rankings?: FundCategoryRankings;
  source_metadata: FundSourceMetadata;
}

export interface FundPricePoint {
  fund_code: string;
  date: string;
  price: number | null;
  daily_return: number | null;
  aum: number | null;
  investor_count: number | null;
  share_count?: number | null;
  source: string;
}

export interface FundPeriodStat {
  key: 'current_month' | 'last_30_days' | 'previous_month' | string;
  label: string;
  start_date: string;
  end_date: string;
  trading_days: number;
  return_days: number;
  positive_days: number;
  negative_days: number;
  flat_days: number;
  average_daily_return: number | null;
  cumulative_return: number | null;
  best_day_return: number | null;
  best_day_date: string | null;
  worst_day_return: number | null;
  worst_day_date: string | null;
  basis: string;
}

export interface FundPeriodStats {
  as_of: string | null;
  periods: FundPeriodStat[];
}

export interface FundAllocation {
  fund_code: string;
  allocation_type: string;
  label: string;
  weight: number | null;
  report_date: string | null;
  source: string;
}

export interface FundPortfolioPosition {
  fund_code: string;
  asset_code: string | null;
  asset_name: string;
  asset_type: string | null;
  asset_region?: string | null;
  provider_symbol?: string | null;
  provider_name?: string | null;
  logo_symbol?: string | null;
  detail_clickable?: boolean | null;
  isin?: string | null;
  weight: number | null;
  raw_weight?: number | null;
  weight_quality?: 'ok' | 'normalized' | 'missing' | string | null;
  weight_warning?: string | null;
  previous_weight?: number | null;
  raw_previous_weight?: number | null;
  previous_weight_quality?: 'ok' | 'normalized' | 'missing' | string | null;
  previous_weight_warning?: string | null;
  weight_change?: number | null;
  raw_weight_change?: number | null;
  change_status?: 'new' | 'increased' | 'decreased' | 'removed' | 'unchanged' | string | null;
  amount: number | null;
  market_value: number | null;
  price?: number | null;
  price_currency?: string | null;
  sector_code?: string | null;
  sector_label?: string | null;
  logo_url?: string | null;
  logo_source?: string | null;
  return_pct?: number | null;
  return_source?: string | null;
  return_as_of?: string | null;
  estimated_exposure_value?: number | null;
  estimated_pnl_value?: number | null;
  estimated_fund_return_contribution_pct?: number | null;
  tefas_tradable?: boolean | null;
  report_date: string | null;
  previous_report_date?: string | null;
  source_report_url: string | null;
  source_type: string | null;
  parse_confidence: number | null;
}

export interface FundPortfolioEffect {
  period: 'daily' | string;
  estimated_return_pct: number | null;
  estimated_pnl_value: number | null;
  priced_weight: number | null;
  missing_weight: number | null;
  aum: number | null;
  as_of: string | null;
}

export interface FundsResponse {
  status: string;
  rows: FundSummary[];
  count: number;
  total_count: number;
  source: string;
  source_url?: string | null;
  as_of: string | null;
  fetched_at: string | null;
  stale: boolean;
  degraded: boolean;
  warnings: string[];
  source_metadata: FundSourceMetadata;
}

export interface FundCategoriesResponse {
  status: string;
  fund_types: string[];
  founder_companies: string[];
  manager_companies: string[];
  risk_values: number[];
  source_metadata: FundSourceMetadata;
}

export interface FundPerformanceResponse {
  fund_code: string;
  status: string;
  points: FundPricePoint[];
  source: string;
  source_url?: string | null;
  as_of: string | null;
  fetched_at: string | null;
  stale: boolean;
  period_stats?: FundPeriodStats;
  source_metadata: FundSourceMetadata;
}

export type MarketComparisonHistoryAssetKind = 'fund' | 'stock' | 'index' | 'fx';

export interface MarketComparisonHistoryAssetRequest {
  id?: string;
  kind: MarketComparisonHistoryAssetKind;
  symbol: string;
  label?: string | null;
}

export interface MarketComparisonHistoryPoint {
  date: string;
  value: number;
}

export interface MarketComparisonHistoryAssetSeries {
  id: string;
  kind: MarketComparisonHistoryAssetKind;
  symbol: string;
  label: string | null;
  points: MarketComparisonHistoryPoint[];
  source: string;
  error: string | null;
}

export interface MarketComparisonHistoryResponse {
  start_date: string;
  end_date: string;
  assets: MarketComparisonHistoryAssetSeries[];
  source: string;
  as_of: string;
}

export interface FundAllocationsResponse {
  fund_code: string;
  status: string;
  allocations: FundAllocation[];
  source: string;
  stale?: boolean;
  source_metadata: FundSourceMetadata;
}

export interface FundAllocationHistoryDay {
  date: string;
  allocations: FundAllocation[];
}

export interface FundAllocationsHistoryResponse {
  fund_code: string;
  status: string;
  lookback_days: number;
  history: FundAllocationHistoryDay[];
  source: string;
  stale?: boolean;
  source_metadata: FundSourceMetadata;
}

export interface FundHoldingsResponse {
  fund_code: string;
  status: 'unavailable' | 'not_parsed' | 'ok' | string;
  positions: FundPortfolioPosition[];
  portfolio_effect?: FundPortfolioEffect | null;
  source: string;
  message?: string;
  source_metadata: FundSourceMetadata;
}

export interface FundHoldingsLivePosition {
  asset_code: string | null;
  price: number | null;
  price_currency: string | null;
  return_pct: number | null;
  return_source: string | null;
  return_as_of: string | null;
  estimated_exposure_value: number | null;
  estimated_pnl_value: number | null;
  estimated_fund_return_contribution_pct: number | null;
}

export interface FundHoldingsLiveResponse {
  fund_code: string;
  status: 'unavailable' | 'not_parsed' | 'ok' | string;
  positions: FundHoldingsLivePosition[];
  portfolio_effect?: FundPortfolioEffect | null;
  source: string;
  as_of: string | null;
  source_metadata: FundSourceMetadata;
}

export interface MarketIndexListRow {
  symbol: MarketIndexCode;
  label: string;
  yahoo_symbol: string | null;
  price: number | null;
  prev_close: number | null;
  change: number | null;
  change_pct: number | null;
  high: number | null;
  low: number | null;
  volume: number | null;
  currency: string;
  market_state: string;
  as_of: string | null;
  error: string | null;
  return_1w_pct: number | null;
  return_1m_pct: number | null;
  return_3m_pct: number | null;
  return_6m_pct: number | null;
  return_ytd_pct: number | null;
  return_1y_pct: number | null;
  return_5y_pct: number | null;
}

export interface MarketIndexLinePoint {
  time: string;
  close: number;
  open?: number;
  high?: number;
  low?: number;
}

export type MarketStockCardChartRange = '1d' | '1w' | '1m' | '1y';

export interface MarketStockCardItem {
  symbol: string;
  company: string;
  yahoo_symbol: string | null;
  price: number | null;
  currency: string;
  change: number | null;
  change_pct: number | null;
  volume: number | null;
  volume_lot: number | null;
  volume_tl: number | null;
  market_cap: number | null;
  high: number | null;
  low: number | null;
  previous_close: number | null;
  fk: number | null;
  pd_dd: number | null;
  fd_favok: number | null;
  net_borc_favok: number | null;
  return_1w_pct: number | null;
  return_1m_pct: number | null;
  return_3m_pct: number | null;
  return_6m_pct: number | null;
  return_ytd_pct: number | null;
  return_1y_pct: number | null;
  base_1w?: number | null;
  high_1w?: number | null;
  low_1w?: number | null;
  base_1m?: number | null;
  high_1m?: number | null;
  low_1m?: number | null;
  base_3m?: number | null;
  high_3m?: number | null;
  low_3m?: number | null;
  base_6m?: number | null;
  high_6m?: number | null;
  low_6m?: number | null;
  base_ytd?: number | null;
  high_ytd?: number | null;
  low_ytd?: number | null;
  base_1y?: number | null;
  high_1y?: number | null;
  low_1y?: number | null;
  market_state: string;
  as_of: string | null;
  session_status?: 'open' | 'closed' | 'previous_session' | 'pre' | 'post' | 'unknown' | string | null;
  session_label?: string | null;
  is_live?: boolean | null;
  is_stale?: boolean | null;
  last_trade_at?: string | null;
  last_trade_date?: string | null;
  line_points: MarketIndexLinePoint[];
  error: string | null;
  logo_url?: string | null;
  logo_source?: 'kap' | null;
}

export interface MarketStockCardsResponse {
  items: MarketStockCardItem[];
  source: string;
  as_of: string;
}

export interface MarketStockCardChartResponse {
  symbol: string;
  range: MarketStockCardChartRange;
  yahoo_symbol: string | null;
  line_points: MarketIndexLinePoint[];
  source: 'yahoo_live' | 'yahoo_cache' | string;
  as_of: string | null;
  error: string | null;
  session_status?: 'open' | 'closed' | 'previous_session' | 'pre' | 'post' | 'unknown' | string | null;
  session_label?: string | null;
  is_live?: boolean | null;
  is_stale?: boolean | null;
  last_trade_at?: string | null;
  last_trade_date?: string | null;
}

export interface MarketIndexConstituent {
  symbol: string;
  price: number | null;
  price_currency: string | null;
  change_pct: number | null;
  volume: number | null;
  shares_outstanding: number | null;
  fdpo: number | null;
  weight_coefficient: number | null;
  free_float_market_value: number | null;
  weight_pct: number | null;
  point_effect: number | null;
  logo_url?: string | null;
  logo_source?: 'kap' | null;
}

export interface MarketIndicesResponse {
  rows: MarketIndexListRow[];
  source: string;
  as_of: string;
}

export interface MarketIndexDetailResponse extends MarketIndexListRow {
  line_points: MarketIndexLinePoint[];
  constituents: MarketIndexConstituent[];
  weight_status: 'available' | 'unavailable';
  weight_note: string | null;
  source: string;
}

export interface MarketFlowItem {
  id: string;
  source: string;
  symbol: string;
  stock_codes?: string[];
  title: string;
  subject?: string;
  published_at: string;
  category: string;
  kap_url?: string | null;
}

export interface MarketFlowResponse {
  items: MarketFlowItem[];
  as_of: string;
  source?: string;
  degraded_mode?: boolean;
  multi_category?: boolean;
  warning?: string | null;
  public_error?: string | null;
}

export interface MarketWatchItem {
  symbol: string;
  label: string;
  yahoo_symbol: string | null;
  price: number | null;
  prev_close: number | null;
  change: number | null;
  change_pct: number | null;
  currency: string;
  market_state: string;
  as_of: string | null;
  error: string | null;
  logo_url?: string | null;
  logo_source?: 'kap' | null;
}

export interface MarketWatchSections {
  indices: MarketWatchItem[];
  fx: MarketWatchItem[];
  commodities: MarketWatchItem[];
}

export interface MarketWatchResponse {
  sections: MarketWatchSections;
  source: string;
  delay_note: string;
  as_of: string;
}

export interface MarketWatchGlobalResponse {
  items: MarketWatchItem[];
  source: string;
  delay_note: string;
  as_of: string;
}

export interface FxQuote {
  symbol: string;
  label: string;
  yahoo_symbol: string;
  price: number | null;
  prev_close: number | null;
  change: number | null;
  change_pct: number | null;
  currency: string;
  market_state: string;
  as_of: string | null;
  error: string | null;
  logo_url?: string | null;
  logo_source?: 'kap' | null;
  return_1w_pct?: number | null;
  return_1m_pct?: number | null;
  return_3m_pct?: number | null;
  return_6m_pct?: number | null;
  return_ytd_pct?: number | null;
  return_1y_pct?: number | null;
}

export interface MarketFxResponse {
  items: FxQuote[];
  source: string;
  delay_note: string;
  as_of: string;
}

export interface CommodityQuote {
  symbol: string;
  label: string;
  yahoo_symbol: string;
  price: number | null;
  prev_close: number | null;
  change: number | null;
  change_pct: number | null;
  currency: string;
  market_state: string;
  as_of: string | null;
  error: string | null;
  logo_url?: string | null;
  logo_source?: 'kap' | null;
}

export interface MarketCommoditiesResponse {
  items: CommodityQuote[];
  source: string;
  delay_note: string;
  as_of: string;
}

export interface MarketIndexResponse {
  index: string;
  rows: MarketUniverseRow[];
  as_of: string;
}

export interface KapMetricValue {
  label: string;
  value: number | null;
  display: string;
}

export interface KapQuarter {
  quarter: string;
  year: number;
  period: number;
  currency: string;
  publish_date: string;
  metrics: Record<string, KapMetricValue>;
  metrics_quarterly: Record<string, KapMetricValue>;
  metrics_ytd: Record<string, KapMetricValue>;
  metrics_original?: Record<string, KapMetricValue>;
  metrics_quarterly_original?: Record<string, KapMetricValue>;
  metrics_ytd_original?: Record<string, KapMetricValue>;
  metrics_comparative?: Record<string, KapMetricValue>;
  metrics_quarterly_comparative?: Record<string, KapMetricValue>;
  metrics_ytd_comparative?: Record<string, KapMetricValue>;
  analysis_multiplier?: number;
  analysis_factor_source?: string;
}

export interface KapValuation {
  price: number | null;
  price_currency: string | null;
  price_as_of: string | null;
  price_source: string;
  shares_outstanding: number | null;
  share_source: string | null;
  share_nominal_value: number | null;
  market_cap: number | null;
  enterprise_value: number | null;
  ttm_net_kar: number | null;
  ttm_favok: number | null;
  fk: number | null;
  pd_dd: number | null;
  fd_favok: number | null;
  assumptions: string[];
}

export interface KapInsurancePremiumDisclosure {
  year: number | null;
  month: number | null;
  period_label: string | null;
  period_start: string | null;
  period_end: string | null;
  published_at: string | null;
  disclosure_index: number | null;
  summary: string | null;
  source_url: string | null;
  monthly_gross_premium: number | null;
  monthly_gross_premium_display: string;
  ytd_gross_premium: number | null;
  ytd_gross_premium_display: string;
  previous_year_monthly_gross_premium: number | null;
  previous_year_monthly_gross_premium_display: string;
  previous_year_ytd_gross_premium: number | null;
  previous_year_ytd_gross_premium_display: string;
  monthly_yoy_pct: number | null;
  monthly_yoy_pct_display: string;
  ytd_yoy_pct: number | null;
  ytd_yoy_pct_display: string;
}

export interface KapSnapshotResponse {
  ok: boolean;
  company: string;
  company_title: string;
  stock_code: string;
  company_kind?: 'generic' | 'bank' | 'insurance';
  fetched_at: string;
  cache_hit: boolean;
  cache_stale?: boolean;
  error?: string;
  analysis_basis?: string;
  analysis_note?: string;
  latest_quarter: string | null;
  summary: Record<string, KapMetricValue>;
  quarters: KapQuarter[];
  valuation?: KapValuation;
  insurance_premium_disclosures?: KapInsurancePremiumDisclosure[];
}

export interface KapOverviewCommentaryRequest {
  company: string;
  company_title: string;
  latest_period: string;
  model?: string;
  history_context: KapOverviewHistoryContext;
  overview_payload: {
    income_summary: unknown[];
    balance_summary: unknown[];
    charts: unknown[];
  };
}

export interface KapOverviewHistoryQuarter {
  label: string;
  year: number;
  period: number;
  metrics: Record<string, number | null>;
  ratios: Record<string, number | null>;
}

export interface KapOverviewHistoryContext {
  company_kind: 'generic' | 'bank' | 'insurance';
  quarters: KapOverviewHistoryQuarter[];
}

export interface KapOverviewScorecardSubscore {
  key: 'buyume' | 'karlilik' | 'bilanco' | 'nakit_akisi';
  label: string;
  score: number;
  summary: string;
}

export interface KapOverviewScorecard {
  overall_score: number;
  overall_label: string;
  summary: string;
  seasonality_note: string;
  score_source: 'deterministic_only' | 'ai_adjusted' | 'ai_failed_fallback';
  subscores: KapOverviewScorecardSubscore[];
}

export interface KapOverviewCommentaryResponse {
  ok: boolean;
  headline: string;
  bullets: string[];
  risk_note: string;
  watch_metrics: string[];
  model_used: string;
  scorecard: KapOverviewScorecard;
  error: string | null;
  debug_trace?: string[];
}
