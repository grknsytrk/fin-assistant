export interface AskRequest {
  question: string;
  retriever?: 'v1' | 'v2' | 'v3' | 'v5' | 'v6';
  mode?: 'single' | 'trend';
  company?: string;
}

export interface AskResponse {
  answer: {
    bullets: string[];
    answer_text?: string;
    found: boolean;
    confidence: number;
    verify_status: string;
  };
  parsed: {
    quarter: string | null;
    query_type: string | null;
    company: string | null;
    mentioned_companies?: string[];
  };
  evidence: EvidenceChunk[];
  debug: {
    retriever: string;
    latency_ms: number;
    top_k: number;
  };
  trend?: {
    rows: any[];
  };
  comparison?: {
    mode: string;
    target: string;
    best_company: string | null;
    best_value: number | null;
    best_confidence: number | null;
    rows: ComparisonRow[];
  };
}

export interface EvidenceChunk {
  doc_id: string;
  company: string | null;
  year: number | null;
  quarter: string;
  page: number;
  section_title: string;
  excerpt: string;
  block_type: string;
  confidence: number | null;
  verify_status: string | null;
  verify_warnings: string[];
}

export interface ComparisonRow {
  company: string;
  target: string;
  quarter: string | null;
  value: number | null;
  confidence: number | null;
}

export interface StatsResponse {
  pdf_count: number;
  page_count: number;
  chunk_count_v1: number;
  chunk_count_v2: number;
  collection_count_v1: number | null;
  collection_count_v2: number | null;
  companies: string[];
}

export interface CompanyBreakdownRow {
  company: string;
  chunks: number;
  quarters: string[];
  quarter_count: number;
}

export interface CompanyBreakdownResponse {
  rows: CompanyBreakdownRow[];
}

export interface MarketUniverseStats {
  bist100_count: number;
  rag_ready_count: number;
  kap_only_count: number;
  kap_cache_count: number;
  pdf_count: number;
  page_count: number;
}

export interface MarketUniverseRow {
  company: string;
  chunks: number;
  quarter_count: number;
  latest_quarter: string | null;
  has_rag: boolean;
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

export type MarketStockIndex = 'XU100' | 'XU030';
export type MarketIndexCode = 'XU100' | 'XU030';

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
  rows: MarketUniverseRow[];
  coverage_rows: MarketUniverseRow[];
}

export interface MarketStocksResponse {
  index: MarketStockIndex;
  rows: MarketStockRow[];
  benchmarks: Record<MarketStockIndex, MarketReturnBenchmark>;
  source: string;
  as_of: string;
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

export interface FeedbackRequest {
  timestamp?: string;
  company?: string;
  quarter?: string;
  metric: string;
  extracted_value?: string;
  user_value?: string;
  evidence_ref?: string;
  verdict: 'dogru' | 'yanlis';
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

export interface KapSnapshotResponse {
  ok: boolean;
  company: string;
  company_title: string;
  stock_code: string;
  fetched_at: string;
  cache_hit: boolean;
  error?: string;
  analysis_basis?: string;
  analysis_note?: string;
  latest_quarter: string | null;
  summary: Record<string, KapMetricValue>;
  quarters: KapQuarter[];
  valuation?: KapValuation;
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
