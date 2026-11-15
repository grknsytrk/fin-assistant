import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from 'react';
import { Activity, AlertCircle, BarChart3, Building2, Check, Clock3, Coins, GripHorizontal, History, Maximize2, Percent, Plus, RefreshCw, RotateCcw, Search, SlidersHorizontal, Trash2, TrendingUp, X } from 'lucide-react';
import { apiClient } from '../api/client';
import type {
    MarketIndexCode,
    MarketIndexConstituent,
    MarketIndexDetailResponse,
    MarketIndexListRow,
    MarketIndicesResponse,
    MarketIndexLinePoint,
    MarketStockCardChartRange,
    MarketStockCardItem,
    MarketStockCardsResponse,
    MarketReturnBenchmark,
    MarketStockIndex,
    MarketStockRow,
    MarketStocksResponse,
    MarketUniverseResponse,
    MarketUniverseRow,
    FundSummary,
} from '../api/types';
import { DEFAULT_STOCK_RETURN_MODE, toStockDetail } from '../routing/routes';
import type { StockReturnMode } from '../routing/routes';
import MarketWatchRail from '../components/MarketWatchRail';
import MarketWatchStrip from '../components/MarketWatchStrip';
import MarketsNavigation, { type MarketsNavigationFundSection, type MarketsNavigationSection } from '../components/MarketsNavigation';
import SymbolLogo from '../components/SymbolLogo';
import { TerminalClock } from '../components/TerminalClock';
import { buildDocumentTitle, formatTitleNumber, formatTitlePct, useDocumentTitle } from '../hooks/useDocumentTitle';
import { MAX_WATCHLIST_ITEMS, normalizeWatchlistSymbol, useWatchlist, type WatchlistItem } from '../hooks/useWatchlist';
import { useTableVirtualizer } from '../hooks/useTableVirtualizer';
import {
    getMarketStockCardSymbolsNeedingLoad,
    retainMarketStockCardsForSymbols,
} from '../utils/stockCardData';
import { searchWatchlistFunds } from '../utils/watchlistData';
import { isBistTradingHours } from '../utils/marketHours';
import { BoundedMemoryCache } from '../utils/lruCache';
import '../styles/market-shared.css';
import './MarketsView.css';

type MarketSection = MarketsNavigationSection;
type SortDirection = 'asc' | 'desc';
type IndexConstituentDataSortKey = 'symbol' | 'price' | 'change_pct' | 'volume' | 'weight_pct' | 'point_effect';
type IndexConstituentSortKey = IndexConstituentDataSortKey | 'impact_pct' | 'impact_abs';
type StockCardDropPlacement = 'before' | 'after';
type StockSortKey =
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
type IndexSortKey = keyof MarketIndexListRow;

type StockReturnKey = Extract<
    StockSortKey,
    'return_1w_pct' | 'return_1m_pct' | 'return_3m_pct' | 'return_6m_pct' | 'return_ytd_pct' | 'return_1y_pct'
>;
type StockTableMode = 'performance' | 'fundamentals';
type StockScreenerCategory = 'all' | 'profile' | 'valuation' | 'financials' | 'market' | 'historical';
type StockScreenerPresetId =
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
type StockScreenerRangeKey =
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
type StockScreenerToggleKey =
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
type StockScreenerFilterState = { sector: string } & Record<StockScreenerRangeKey, string> & Record<StockScreenerToggleKey, boolean>;
type StockSectorDiffFields = {
    fk_sector_diff_pct: number | null;
    pd_dd_sector_diff_pct: number | null;
    fd_favok_sector_diff_pct: number | null;
};
type StockTableColumn = { key: StockSortKey; label: string; sublabel?: string; align?: 'left' | 'right'; width?: number };

type ScreenerRangeSpec = {
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

const SCREENER_RANGE_SPECS: ScreenerRangeSpec[] = [
    // Değerleme Çarpanları
    { card: 'valuation', category: 'valuation', field: 'fk', minKey: 'min_fk', maxKey: 'max_fk', label: 'F/K', hint: 'Fiyat / Kazanç', unit: 'x' },
    { card: 'valuation', category: 'valuation', field: 'pd_dd', minKey: 'min_pd_dd', maxKey: 'max_pd_dd', label: 'PD/DD', hint: 'Piyasa / Defter', unit: 'x' },
    { card: 'valuation', category: 'valuation', field: 'fd_favok', minKey: 'min_fd_favok', maxKey: 'max_fd_favok', label: 'FD/FAVÖK', hint: 'Firma / FAVÖK', unit: 'x' },
    { card: 'valuation', category: 'valuation', field: 'fd_satis', minKey: 'min_fd_satis', maxKey: 'max_fd_satis', label: 'FD/Satış', hint: 'Firma / Satış', unit: 'x' },
    // Kârlılık & Kaldıraç
    { card: 'profitability', category: 'financials', field: 'roe_pct', minKey: 'min_roe_pct', maxKey: 'max_roe_pct', label: 'ROE', hint: 'Özsermaye Kârlılığı', unit: '%' },
    { card: 'profitability', category: 'financials', field: 'favok_margin_pct', minKey: 'min_favok_margin_pct', maxKey: 'max_favok_margin_pct', label: 'FAVÖK Marjı', hint: 'Esas Faaliyet Kârı', unit: '%' },
    { card: 'profitability', category: 'financials', field: 'net_margin_pct', minKey: 'min_net_margin_pct', maxKey: 'max_net_margin_pct', label: 'Net Kâr Marjı', hint: 'Net Kâr / Satışlar', unit: '%' },
    { card: 'profitability', category: 'financials', field: 'net_borc_favok', minKey: 'min_net_borc_favok', maxKey: 'max_net_borc_favok', label: 'Net Borç / FAVÖK', hint: 'Kaldıraç Oranı', unit: 'x' },
    // Bilanço & TTM Ölçek
    { card: 'scale', category: 'financials', field: 'ttm_satis_gelirleri', minKey: 'min_ttm_satis', maxKey: 'max_ttm_satis', label: 'TTM Satışlar', hint: 'Son 12 Ay Ciro', unit: 'mn ₺', scale: 1_000_000 },
    { card: 'scale', category: 'financials', field: 'ttm_favok', minKey: 'min_ttm_favok', maxKey: 'max_ttm_favok', label: 'TTM FAVÖK', hint: 'Son 12 Ay FAVÖK', unit: 'mn ₺', scale: 1_000_000 },
    { card: 'scale', category: 'financials', field: 'ttm_net_kar', minKey: 'min_ttm_net_kar', maxKey: 'max_ttm_net_kar', label: 'TTM Net Kâr', hint: 'Son 12 Ay Net Kâr', unit: 'mn ₺', scale: 1_000_000 },
    { card: 'scale', category: 'financials', field: 'ttm_fcf', minKey: 'min_ttm_fcf', maxKey: 'max_ttm_fcf', label: 'Serbest Nakit', hint: 'Son 12 Ay FCF', unit: 'mn ₺', scale: 1_000_000 },
    { card: 'scale', category: 'financials', field: 'ozkaynaklar', minKey: 'min_ozkaynaklar', maxKey: 'max_ozkaynaklar', label: 'Özkaynaklar', hint: 'Toplam Özkaynak', unit: 'mn ₺', scale: 1_000_000 },
    { card: 'scale', category: 'financials', field: 'net_borc', minKey: 'min_net_borc', maxKey: 'max_net_borc', label: 'Net Borç', hint: 'Negatif = Net Nakit', unit: 'mn ₺', scale: 1_000_000 },
    // Piyasa, Likidite & Fiyat
    { card: 'market_scale', category: 'market', field: 'market_cap', minKey: 'min_market_cap', maxKey: 'max_market_cap', label: 'Piyasa Değeri', hint: 'Toplam Değer', unit: 'mn ₺', scale: 1_000_000 },
    { card: 'market_scale', category: 'market', field: 'volume', minKey: 'min_volume', maxKey: 'max_volume', label: 'Günlük Hacim', hint: 'İşlem Hacmi', unit: 'mn ₺', scale: 1_000_000 },
    { card: 'market_scale', category: 'market', field: 'free_float_pct', minKey: 'min_free_float_pct', maxKey: 'max_free_float_pct', label: 'Halka Açıklık', hint: 'Fiili Dolaşım Oranı', unit: '%' },
    { card: 'market_scale', category: 'market', field: 'price', minKey: 'min_price', maxKey: 'max_price', label: 'Hisse Fiyatı', hint: 'Son İşlem Fiyatı', unit: '₺' },
    // Fiyat Performansı & Getiri
    { card: 'returns', category: 'market', field: 'change_pct', minKey: 'min_change_pct', maxKey: 'max_change_pct', label: 'Günlük Değişim', hint: 'Gün İçi Getiri', unit: '%' },
    { card: 'returns', category: 'market', field: 'return_1w_pct', minKey: 'min_return_1w_pct', maxKey: 'max_return_1w_pct', label: '1 Hafta Getiri', hint: 'Haftalık Performans', unit: '%' },
    { card: 'returns', category: 'market', field: 'return_1m_pct', minKey: 'min_return_1m_pct', maxKey: 'max_return_1m_pct', label: '1 Ay Getiri', hint: 'Aylık Performans', unit: '%' },
    { card: 'returns', category: 'market', field: 'return_3m_pct', minKey: 'min_return_3m_pct', maxKey: 'max_return_3m_pct', label: '3 Ay Getiri', hint: 'Çeyreklik Performans', unit: '%' },
    { card: 'returns', category: 'market', field: 'return_ytd_pct', minKey: 'min_return_ytd_pct', maxKey: 'max_return_ytd_pct', label: 'Yılbaşı (YTD)', hint: 'Yıl Başından Bugüne', unit: '%' },
    { card: 'returns', category: 'market', field: 'return_1y_pct', minKey: 'min_return_1y_pct', maxKey: 'max_return_1y_pct', label: '1 Yıl Getiri', hint: 'Yıllık Performans', unit: '%' },
    // Tarihsel Çarpan Ortalamaları (KAP)
    { card: 'hist_multiples', category: 'historical', field: 'fk_hist_diff_pct', minKey: 'min_fk_hist_diff_pct', maxKey: 'max_fk_hist_diff_pct', label: 'F/K Tarihsel Fark', hint: 'Negatif = KAP ort. iskontolu', unit: '%' },
    { card: 'hist_multiples', category: 'historical', field: 'pd_dd_hist_diff_pct', minKey: 'min_pd_dd_hist_diff_pct', maxKey: 'max_pd_dd_hist_diff_pct', label: 'PD/DD Tarihsel Fark', hint: 'Negatif = KAP ort. iskontolu', unit: '%' },
    { card: 'hist_multiples', category: 'historical', field: 'fd_favok_hist_diff_pct', minKey: 'min_fd_favok_hist_diff_pct', maxKey: 'max_fd_favok_hist_diff_pct', label: 'FD/FAVÖK Tarihsel Fark', hint: 'Negatif = KAP ort. iskontolu', unit: '%' },
    { card: 'hist_multiples', category: 'historical', field: 'hist_fk_avg', minKey: 'min_hist_fk_avg', maxKey: 'max_hist_fk_avg', label: 'Tarihsel Ort. F/K', hint: 'Mevcut KAP çeyrek ortalaması', unit: 'x' },
    { card: 'hist_multiples', category: 'historical', field: 'hist_pd_dd_avg', minKey: 'min_hist_pd_dd_avg', maxKey: 'max_hist_pd_dd_avg', label: 'Tarihsel Ort. PD/DD', hint: 'Mevcut KAP çeyrek ortalaması', unit: 'x' },
    { card: 'hist_multiples', category: 'historical', field: 'hist_fd_favok_avg', minKey: 'min_hist_fd_favok_avg', maxKey: 'max_hist_fd_favok_avg', label: 'Tarihsel Ort. FD/FAVÖK', hint: 'Mevcut KAP çeyrek ortalaması', unit: 'x' },
    // Fiyat Hareketli Ortalamaları & 52H Zirve/Dip
    { card: 'technicals', category: 'historical', field: 'sma_50_diff_pct', minKey: 'min_sma_50_diff_pct', maxKey: 'max_sma_50_diff_pct', label: '50G Ort. Fark', hint: 'Fiyat / 50G Ort. Uzaklık', unit: '%' },
    { card: 'technicals', category: 'historical', field: 'sma_200_diff_pct', minKey: 'min_sma_200_diff_pct', maxKey: 'max_sma_200_diff_pct', label: '200G Ort. Fark', hint: 'Fiyat / 200G Ort. Uzaklık', unit: '%' },
    { card: 'technicals', category: 'historical', field: 'high_52w_dist_pct', minKey: 'min_high_52w_dist_pct', maxKey: 'max_high_52w_dist_pct', label: '52H Zirveye Uzaklık', hint: '0% = 52H zirve seviyesi', unit: '%' },
    { card: 'technicals', category: 'historical', field: 'low_52w_dist_pct', minKey: 'min_low_52w_dist_pct', maxKey: 'max_low_52w_dist_pct', label: '52H Dibe Uzaklık', hint: 'Düşük pozitif = dibe yakın', unit: '%' },
    // Sektör Ortalaması Kıyaslaması
    { card: 'sector_relative', category: 'historical', field: 'fk_sector_diff_pct', minKey: 'min_fk_sector_diff_pct', maxKey: 'max_fk_sector_diff_pct', label: 'Sektör F/K Fark', hint: 'Negatif = sektörden ucuz', unit: '%' },
    { card: 'sector_relative', category: 'historical', field: 'pd_dd_sector_diff_pct', minKey: 'min_pd_dd_sector_diff_pct', maxKey: 'max_pd_dd_sector_diff_pct', label: 'Sektör PD/DD Fark', hint: 'Negatif = sektörden ucuz', unit: '%' },
    { card: 'sector_relative', category: 'historical', field: 'fd_favok_sector_diff_pct', minKey: 'min_fd_favok_sector_diff_pct', maxKey: 'max_fd_favok_sector_diff_pct', label: 'Sektör FD/FAVÖK Fark', hint: 'Negatif = sektörden ucuz', unit: '%' },
];

const SCREENER_TOGGLE_SPECS: Array<{ key: StockScreenerToggleKey; label: string; category: 'profile' | 'historical' }> = [
    { key: 'positive_net_profit', label: 'Pozitif TTM Net Kâr', category: 'profile' },
    { key: 'positive_ebitda', label: 'Pozitif TTM FAVÖK', category: 'profile' },
    { key: 'positive_free_cash_flow', label: 'Pozitif Serbest Nakit Akışı', category: 'profile' },
    { key: 'net_cash_only', label: 'Net Nakit (Borçsuz)', category: 'profile' },
    { key: 'positive_roe', label: 'Pozitif ROE', category: 'profile' },
    { key: 'only_with_financials', label: 'Bilanço Verisi Mevcut', category: 'profile' },
    { key: 'below_hist_fk', label: 'Tarihsel F/K Ort. Altında', category: 'historical' },
    { key: 'below_hist_pd_dd', label: 'Tarihsel PD/DD Ort. Altında', category: 'historical' },
    { key: 'above_sma_50', label: '50G Ort. Üstünde', category: 'historical' },
    { key: 'above_sma_200', label: '200G Ort. Üstünde', category: 'historical' },
    { key: 'below_sector_fk', label: 'Sektör F/K Ort. Altında', category: 'historical' },
    { key: 'below_sector_pd_dd', label: 'Sektör PD/DD Ort. Altında', category: 'historical' },
];

const EMPTY_STOCK_SCREENER_FILTERS: StockScreenerFilterState = {
    sector: '',
    min_fk: '', max_fk: '',
    min_pd_dd: '', max_pd_dd: '',
    min_fd_favok: '', max_fd_favok: '',
    min_fd_satis: '', max_fd_satis: '',
    min_net_borc_favok: '', max_net_borc_favok: '',
    min_roe_pct: '', max_roe_pct: '',
    min_favok_margin_pct: '', max_favok_margin_pct: '',
    min_net_margin_pct: '', max_net_margin_pct: '',
    min_ttm_net_kar: '', max_ttm_net_kar: '',
    min_ttm_fcf: '', max_ttm_fcf: '',
    min_ttm_satis: '', max_ttm_satis: '',
    min_ttm_favok: '', max_ttm_favok: '',
    min_net_borc: '', max_net_borc: '',
    min_ozkaynaklar: '', max_ozkaynaklar: '',
    min_market_cap: '', max_market_cap: '',
    min_volume: '', max_volume: '',
    min_free_float_pct: '', max_free_float_pct: '',
    min_price: '', max_price: '',
    min_change_pct: '', max_change_pct: '',
    min_return_1w_pct: '', max_return_1w_pct: '',
    min_return_1m_pct: '', max_return_1m_pct: '',
    min_return_3m_pct: '', max_return_3m_pct: '',
    min_return_ytd_pct: '', max_return_ytd_pct: '',
    min_return_1y_pct: '', max_return_1y_pct: '',
    min_hist_fk_avg: '', max_hist_fk_avg: '',
    min_hist_pd_dd_avg: '', max_hist_pd_dd_avg: '',
    min_hist_fd_favok_avg: '', max_hist_fd_favok_avg: '',
    min_fk_hist_diff_pct: '', max_fk_hist_diff_pct: '',
    min_pd_dd_hist_diff_pct: '', max_pd_dd_hist_diff_pct: '',
    min_fd_favok_hist_diff_pct: '', max_fd_favok_hist_diff_pct: '',
    min_sma_50_diff_pct: '', max_sma_50_diff_pct: '',
    min_sma_200_diff_pct: '', max_sma_200_diff_pct: '',
    min_high_52w_dist_pct: '', max_high_52w_dist_pct: '',
    min_low_52w_dist_pct: '', max_low_52w_dist_pct: '',
    min_fk_sector_diff_pct: '', max_fk_sector_diff_pct: '',
    min_pd_dd_sector_diff_pct: '', max_pd_dd_sector_diff_pct: '',
    min_fd_favok_sector_diff_pct: '', max_fd_favok_sector_diff_pct: '',
    positive_net_profit: false,
    positive_ebitda: false,
    positive_free_cash_flow: false,
    net_cash_only: false,
    positive_roe: false,
    only_with_financials: false,
    below_hist_fk: false,
    below_hist_pd_dd: false,
    above_sma_50: false,
    above_sma_200: false,
    below_sector_fk: false,
    below_sector_pd_dd: false,
};

const STOCK_COLUMNS: StockTableColumn[] = [
    { key: 'company', label: 'Hisse', align: 'left', width: 130 },
    { key: 'price', label: 'Fiyat (₺)', align: 'right', width: 100 },
    { key: 'change_pct', label: 'Gün %', align: 'right', width: 90 },
    { key: 'volume', label: 'Hacim', align: 'right', width: 110 },
    { key: 'market_cap', label: 'Piyasa Değeri', align: 'right', width: 130 },
    { key: 'return_1w_pct', label: '1 Hafta', align: 'right', width: 90 },
    { key: 'return_1m_pct', label: '1 Ay', align: 'right', width: 90 },
    { key: 'return_3m_pct', label: '3 Ay', align: 'right', width: 90 },
    { key: 'return_6m_pct', label: '6 Ay', align: 'right', width: 90 },
    { key: 'return_ytd_pct', label: 'Yılbaşı', align: 'right', width: 90 },
    { key: 'return_1y_pct', label: '1 Yıl', align: 'right', width: 90 },
];
const STOCK_FUNDAMENTAL_COLUMNS: StockTableColumn[] = [
    { key: 'company', label: 'Hisse', align: 'left', width: 120 },
    { key: 'sector', label: 'Sektör', align: 'left', width: 180 },
    { key: 'fundamental_period', label: 'Dönem', align: 'left', width: 80 },
    { key: 'price', label: 'Fiyat', align: 'right', width: 95 },
    { key: 'market_cap', label: 'Piyasa Değeri', align: 'right', width: 125 },
    { key: 'fk', label: 'F/K', align: 'right', width: 85 },
    { key: 'pd_dd', label: 'PD/DD', align: 'right', width: 85 },
    { key: 'fd_favok', label: 'FD/FAVÖK', align: 'right', width: 95 },
    { key: 'fd_satis', label: 'FD/Satış', align: 'right', width: 96 },
    { key: 'hist_fk_avg', label: 'Ort. F/K', sublabel: 'KAP Ort.', align: 'right', width: 124 },
    { key: 'fk_hist_diff_pct', label: 'F/K Fark %', sublabel: 'vs KAP Ort.', align: 'right', width: 108 },
    { key: 'hist_pd_dd_avg', label: 'Ort. PD/DD', sublabel: 'KAP Ort.', align: 'right', width: 124 },
    { key: 'pd_dd_hist_diff_pct', label: 'PD/DD Fark %', sublabel: 'vs KAP Ort.', align: 'right', width: 118 },
    { key: 'hist_fd_favok_avg', label: 'Ort. FD/FAVÖK', sublabel: 'KAP Ort.', align: 'right', width: 140 },
    { key: 'fk_sector_diff_pct', label: 'Sektör F/K %', align: 'right', width: 120 },
    { key: 'sma_50_diff_pct', label: '50G Ort. %', align: 'right', width: 112 },
    { key: 'sma_200_diff_pct', label: '200G Ort. %', align: 'right', width: 114 },
    { key: 'high_52w_dist_pct', label: '52H Zirve %', align: 'right', width: 116 },
    { key: 'net_borc_favok', label: 'Net Borç/FAVÖK', align: 'right', width: 130 },
    { key: 'ttm_net_kar', label: 'TTM Net Kâr', align: 'right', width: 125 },
    { key: 'ttm_fcf', label: 'TTM Serbest Nakit Akışı', align: 'right', width: 150 },
    { key: 'ttm_satis_gelirleri', label: 'TTM Satışlar', align: 'right', width: 125 },
    { key: 'ttm_favok', label: 'TTM FAVÖK', align: 'right', width: 125 },
    { key: 'net_borc', label: 'Net Borç', align: 'right', width: 120 },
    { key: 'ozkaynaklar', label: 'Özkaynaklar', align: 'right', width: 125 },
    { key: 'roe_pct', label: 'ROE %', align: 'right', width: 85 },
    { key: 'free_float_pct', label: 'Halka Açıklık %', align: 'right', width: 110 },
];
const STOCK_SCREENER_PRESETS: Array<{
    id: StockScreenerPresetId;
    label: string;
    filters: Partial<StockScreenerFilterState>;
}> = [
    { id: 'discount', label: 'Çarpan İskontosu', filters: { min_fk: '0.01', max_fk: '12', min_fd_favok: '0.01', max_fd_favok: '8' } },
    { id: 'low_debt', label: 'Düşük Borçluluk', filters: { max_net_borc_favok: '1', positive_ebitda: true } },
    { id: 'net_cash', label: 'Net Nakit Şirketler', filters: { net_cash_only: true, positive_net_profit: true } },
    { id: 'roe_value', label: 'Yüksek ROE & Makul PD/DD', filters: { min_roe_pct: '25', min_pd_dd: '0.01', max_pd_dd: '4' } },
    { id: 'below_book', label: 'Defter Değerinin Altında', filters: { min_pd_dd: '0.01', max_pd_dd: '1' } },
    { id: 'high_margins', label: 'Yüksek Kâr Marjı', filters: { min_favok_margin_pct: '20', min_net_margin_pct: '10' } },
    { id: 'positive_cash_flow', label: 'Pozitif Serbest Nakit', filters: { positive_free_cash_flow: true } },
    { id: 'momentum_leaders', label: 'Güçlü Momentum (1A & 1Y)', filters: { min_return_1m_pct: '5', min_return_1y_pct: '30' } },
    { id: 'large_liquid', label: 'Büyük Ölçek & Halka Açık', filters: { min_market_cap: '10000', min_free_float_pct: '25' } },
    { id: 'hist_discount', label: 'Tarihsel İskontolu', filters: { below_hist_fk: true, below_hist_pd_dd: true, positive_net_profit: true } },
    { id: 'trend_above_200', label: '200G Ort. Üstü Trend', filters: { above_sma_50: true, above_sma_200: true } },
    { id: 'sector_discount', label: 'Sektörüne Göre Ucuz', filters: { below_sector_fk: true, below_sector_pd_dd: true, positive_roe: true } },
];
const STOCK_INDEX_OPTIONS: MarketStockIndex[] = ['XUTUM', 'XU100', 'XU030'];
const PRIMARY_MARKET_INDEX_RANK = new Map<string, number>(
    ['XU100', 'XU030', 'XUTUM'].map((symbol, index) => [symbol, index]),
);
const RETURN_MODE_OPTIONS: Array<{ id: StockReturnMode; label: string }> = [
    { id: 'absolute', label: 'Mutlak' },
    { id: 'relative_xu100', label: "XU100'a göre" },
    { id: 'relative_xu030', label: "XU030'a göre" },
];
const RETURN_KEYS: StockReturnKey[] = [
    'return_1w_pct',
    'return_1m_pct',
    'return_3m_pct',
    'return_6m_pct',
    'return_ytd_pct',
    'return_1y_pct',
];
const STOCK_CARD_STORAGE_KEY = 'ragfin_market_stock_cards';
const MAX_STOCK_CARDS = 12;
// The first desktop row becomes useful as soon as possible. Remaining cards
// are fetched in small batches so a slow quote/valuation provider cannot hold
// the whole panel behind one spinner.
const STOCK_CARD_INITIAL_BATCH_SIZE = 3;
const STOCK_CARD_BACKGROUND_BATCH_SIZE = 3;
const STOCK_CARD_BACKGROUND_DELAY_MS = 120;
const MARKET_QUOTE_REFRESH_MS = 3000;
const MARKET_LIST_REFRESH_DESKTOP_MS = MARKET_QUOTE_REFRESH_MS;
const MARKET_LIST_REFRESH_MOBILE_MS = MARKET_QUOTE_REFRESH_MS;
const MARKET_DETAIL_REFRESH_DESKTOP_MS = 15000;
const MARKET_DETAIL_REFRESH_MOBILE_MS = 30000;
const STOCK_CARD_CHART_RANGES: Array<{ id: MarketStockCardChartRange; label: string; title: string }> = [
    { id: '1d', label: 'G', title: 'Gün içi' },
    { id: '1w', label: '1H', title: '1 Hafta' },
    { id: '1m', label: '1A', title: '1 Ay' },
    { id: 'ytd', label: 'YTD', title: 'Yılbaşından bugüne' },
    { id: '1y', label: '1Y', title: '1 Yıl' },
    { id: '5y', label: '5Y', title: '5 Yıl' },
];
const INDEX_COLUMNS: Array<{ key: keyof MarketIndexListRow; label: string; align?: 'left' | 'right' }> = [
    { key: 'symbol', label: 'Endeks', align: 'left' },
    { key: 'price', label: 'Fiyat', align: 'right' },
    { key: 'change_pct', label: 'Gün %', align: 'right' },
    { key: 'volume', label: 'Hacim', align: 'right' },
    { key: 'return_1w_pct', label: '1 Hafta', align: 'right' },
    { key: 'return_1m_pct', label: '1 Ay', align: 'right' },
    { key: 'return_3m_pct', label: '3 Ay', align: 'right' },
    { key: 'return_6m_pct', label: '6 Ay', align: 'right' },
    { key: 'return_ytd_pct', label: 'Yılbaşı', align: 'right' },
    { key: 'return_1y_pct', label: '1 Yıl', align: 'right' },
];
type IndexGroupFilter = 'all' | 'main' | 'sector';
const INDEX_GROUP_OPTIONS: Array<{ id: IndexGroupFilter; label: string }> = [
    { id: 'all', label: 'Tümü' },
    { id: 'main', label: 'Ana Endeksler' },
    { id: 'sector', label: 'Sektör' },
];
const MAIN_MARKET_INDICES = new Set<string>([
    'XUTUM',
    'XU100',
    'XU030',
    'XUSIN',
    'XUHIZ',
    'XUMAL',
    'XUTEK',
]);
const INDEX_CONSTITUENT_COLUMNS: Array<{ key: Exclude<IndexConstituentSortKey, 'impact_abs'>; label: string; align?: 'left' | 'right' }> = [
    { key: 'symbol', label: 'Şirket', align: 'left' },
    { key: 'price', label: 'Son Fiyat', align: 'right' },
    { key: 'change_pct', label: '%', align: 'right' },
    { key: 'volume', label: 'Hacim', align: 'right' },
    { key: 'weight_pct', label: 'Endeks Ağırlığı', align: 'right' },
    { key: 'point_effect', label: 'Puan Etkisi', align: 'right' },
    { key: 'impact_pct', label: 'Etki %', align: 'right' },
];
const DETAIL_RETURN_KEYS: Array<{ key: keyof MarketIndexListRow; label: string }> = [
    { key: 'change_pct', label: 'Gün içi' },
    { key: 'return_1w_pct', label: '1 Hafta' },
    { key: 'return_1m_pct', label: '1 Ay' },
    { key: 'return_ytd_pct', label: 'YTA' },
    { key: 'return_6m_pct', label: '6 Ay' },
    { key: 'return_1y_pct', label: '1 Yıl' },
    { key: 'return_5y_pct', label: '5 Yıl' },
];

function formatStockPrice(row: MarketStockRow): string {
    if (row.price == null) return '—';
    const currencyPrefix = row.price_currency && row.price_currency !== 'TRY' ? `${row.price_currency} ` : '';
    return `${currencyPrefix}${row.price.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatIndexPrice(value: number | null): string {
    if (value == null) return '—';
    return value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

function formatMaybeCurrency(value: number | null, currency?: string | null): string {
    if (value == null) return '—';
    const prefix = currency && currency !== 'TRY' ? `${currency} ` : '';
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatTablePct(value: number | null): string {
    if (value == null) return '—';
    const sign = value > 0 ? '+' : '';
    return `%${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatVolume(value: number | null): string {
    if (value == null) return '—';
    const abs = Math.abs(value);
    if (abs >= 1_000_000_000) {
        return `${(value / 1_000_000_000).toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })} mr`;
    }
    if (abs >= 1_000_000) {
        return `${(value / 1_000_000).toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })} mn`;
    }
    return value.toLocaleString('tr-TR', { maximumFractionDigits: 0 });
}

function formatMarketCap(value: number | null): string {
    if (value == null) return '—';
    return formatVolume(value);
}

function formatFundamentalRatio(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '—';
    return `${value.toLocaleString('tr-TR', { maximumFractionDigits: 2 })}x`;
}

function formatFundamentalPercent(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '—';
    return `%${value.toLocaleString('tr-TR', { maximumFractionDigits: 2 })}`;
}

function parseScreenerNumber(raw: string): number | undefined {
    const text = String(raw || '').trim().replace(',', '.');
    if (!text) return undefined;
    const value = Number(text);
    return Number.isFinite(value) ? value : undefined;
}

function hasStockFinancialData(row: MarketStockRow): boolean {
    return Boolean(
        row.has_kap_cache
        || row.fundamental_period
        || row.fk != null
        || row.pd_dd != null
        || row.fd_favok != null
        || row.fd_satis != null
        || row.net_borc_favok != null
        || row.roe_pct != null
        || row.favok_margin_pct != null
        || row.net_margin_pct != null
        || row.ttm_net_kar != null
        || row.ttm_satis_gelirleri != null
        || row.ttm_favok != null
        || row.ttm_fcf != null
        || row.net_borc != null
        || row.ozkaynaklar != null,
    );
}

function formatCardCurrency(value: number | null | undefined, currency?: string | null): string {
    if (value == null) return '-';
    const prefix = currency && currency !== 'TRY' ? `${currency} ` : '₺';
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatCardPct(value: number | null | undefined): string {
    if (value == null) return '% -';
    const sign = value > 0 ? '+' : '';
    return `%${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatCardRatio(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

function formatCardPositiveRatio(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value) || value <= 0) return '-';
    return formatCardRatio(value);
}

function hasStockCardLoadedData(item: MarketStockCardItem | null | undefined): boolean {
    if (!item) return false;
    if (item.error) return true;
    const hasMarketField = [
        item.price,
        item.change_pct,
        item.high,
        item.low,
        item.previous_close,
        item.volume,
        item.volume_lot,
        item.volume_tl,
        item.market_cap,
    ].some((value) => Number.isFinite(value));
    const hasChart = (item.line_points ?? []).some((point) => Number.isFinite(point.close));
    return hasMarketField || hasChart;
}

function formatCardFullNumber(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return value.toLocaleString('tr-TR', { maximumFractionDigits: 0 });
}

function formatCardFullCurrency(value: number | null | undefined, currency?: string | null): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const prefix = currency && currency !== 'TRY' ? `${currency} ` : '₺';
    return `${prefix}${formatCardFullNumber(value)}`;
}

function istanbulDateKey(date: Date): string {
    if (Number.isNaN(date.getTime())) return '';
    return new Intl.DateTimeFormat('en-CA', {
        timeZone: 'Europe/Istanbul',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
    }).format(date);
}

function formatStockCardChartDate(iso: string | null | undefined, range: MarketStockCardChartRange): string {
    if (!iso) return '-';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '-';

    const timeStr = new Intl.DateTimeFormat('tr-TR', {
        timeZone: 'Europe/Istanbul',
        hour: '2-digit',
        minute: '2-digit',
    }).format(dt);

    const isToday = istanbulDateKey(dt) === istanbulDateKey(new Date());

    if (range === '1d' || isToday) {
        return `Bugün ${timeStr}`;
    }

    const withTime = range === '1w' || range === '1m';
    return new Intl.DateTimeFormat('tr-TR', {
        timeZone: 'Europe/Istanbul',
        day: '2-digit',
        month: 'short',
        ...(range === '5y' ? { year: '2-digit' } : {}),
        ...(withTime ? { hour: '2-digit', minute: '2-digit' } : {}),
    }).format(dt);
}

function formatStockCardAxisDate(iso: string | null | undefined, range: MarketStockCardChartRange): string {
    if (!iso) return '';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '';
    if (range === '1d') {
        return new Intl.DateTimeFormat('tr-TR', {
            timeZone: 'Europe/Istanbul',
            hour: '2-digit',
            minute: '2-digit',
        }).format(dt);
    }
    if (range === '5y') {
        return new Intl.DateTimeFormat('tr-TR', {
            timeZone: 'Europe/Istanbul',
            month: 'short',
            year: '2-digit',
        }).format(dt);
    }
    return new Intl.DateTimeFormat('tr-TR', {
        timeZone: 'Europe/Istanbul',
        day: '2-digit',
        month: 'short',
    }).format(dt);
}

function formatStockCardAxisTime(date: Date): string {
    if (Number.isNaN(date.getTime())) return '';
    return new Intl.DateTimeFormat('tr-TR', {
        timeZone: 'Europe/Istanbul',
        hour: '2-digit',
        minute: '2-digit',
    }).format(date);
}

function formatUpdateTime(iso: string | null | undefined): string {
    if (!iso) return '--:--';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '--:--';
    return dt.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatStockCardTradeTime(iso: string | null | undefined): string {
    if (!iso) return '--:--';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '--:--';
    const today = new Date();
    const isToday = dt.toLocaleDateString('tr-TR', { timeZone: 'Europe/Istanbul' })
        === today.toLocaleDateString('tr-TR', { timeZone: 'Europe/Istanbul' });
    return new Intl.DateTimeFormat('tr-TR', {
        timeZone: 'Europe/Istanbul',
        ...(isToday ? {} : { day: '2-digit', month: 'short' }),
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    }).format(dt);
}

function latestStockCardPointTime(points: MarketIndexLinePoint[] | null | undefined): string | null {
    const valid = (points || [])
        .map((point) => point.time)
        .filter((time): time is string => Boolean(time))
        .sort();
    return valid[valid.length - 1] || null;
}

function isPreviousIstanbulDate(iso: string | null | undefined): boolean {
    if (!iso) return false;
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return false;
    return istanbulDateKey(dt) < istanbulDateKey(new Date());
}

function formatDateTime(iso: string | null | undefined): string {
    if (!iso) return '-';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '-';
    return dt.toLocaleString('tr-TR', {
        day: '2-digit',
        month: 'long',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
}

function normalizeStockCardSymbol(raw: string): string {
    return raw.trim().toUpperCase().replace(/\.IS$/, '');
}

function readStoredStockCards(): string[] {
    if (typeof window === 'undefined') return [];
    try {
        const raw = window.localStorage.getItem(STOCK_CARD_STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        if (!Array.isArray(parsed)) return [];
        const seen = new Set<string>();
        const symbols: string[] = [];
        for (const item of parsed) {
            if (typeof item !== 'string') continue;
            const symbol = normalizeStockCardSymbol(item);
            if (!/^[A-Z0-9]{2,12}$/.test(symbol) || seen.has(symbol)) continue;
            symbols.push(symbol);
            seen.add(symbol);
            if (symbols.length >= MAX_STOCK_CARDS) break;
        }
        return symbols;
    } catch {
        return [];
    }
}

function mobileSparklinePath(points: MarketIndexLinePoint[]): string {
    const valid = points.filter((point) => Number.isFinite(point.close));
    if (valid.length < 2) return '';
    const sampled = valid.length > 36
        ? valid.filter((_, index) => index % Math.ceil(valid.length / 36) === 0)
        : valid;
    const values = sampled.map((point) => point.close);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(max - min, Math.abs(max) * 0.001, 0.0001);
    return sampled
        .map((point, index) => {
            const x = (index / (sampled.length - 1)) * 220;
            const y = 46 - ((point.close - min) / span) * 38;
            return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(' ');
}

function MobileMarketOverview({
    index,
    rows,
    stockQuoteRows,
    stockCardItems,
    watchlistItems,
    fundRows,
    isMobileViewport,
    onSelectTicker,
    onSelectIndex,
    onOpenFund,
    onAddStock,
    onAddFund,
    onRemoveWatchlistItem,
    onSearchOpenChange,
}: {
    index: MarketIndexDetailResponse | null;
    rows: MarketUniverseRow[];
    stockQuoteRows: MarketStockRow[];
    stockCardItems: MarketStockCardItem[];
    watchlistItems: WatchlistItem[];
    fundRows: Record<string, FundSummary | null>;
    isMobileViewport: boolean;
    onSelectTicker: (ticker: string) => void;
    onSelectIndex: (index: MarketIndexCode) => void;
    onOpenFund?: (fundCode: string) => void;
    onAddStock: (symbol: string) => void;
    onAddFund: (fund: FundSummary) => void;
    onRemoveWatchlistItem: (item: WatchlistItem) => void;
    onSearchOpenChange: (open: boolean) => void;
}) {
    const [searchOpen, setSearchOpen] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [companySearchItems, setCompanySearchItems] = useState<MarketUniverseRow[]>([]);
    const [companySearchLoading, setCompanySearchLoading] = useState(false);
    const [fundSearchItems, setFundSearchItems] = useState<FundSummary[]>([]);
    const [fundSearchLoading, setFundSearchLoading] = useState(false);
    const [watchlistChartPoints, setWatchlistChartPoints] = useState<Record<string, MarketIndexLinePoint[] | null>>({});
    const [swipedWatchlistKey, setSwipedWatchlistKey] = useState<string | null>(null);
    const swipeStartRef = useRef<{ key: string; x: number; y: number } | null>(null);
    const suppressRowClickRef = useRef(false);
    const rowBySymbol = new Map(rows.map((row) => [normalizeWatchlistSymbol(row.company), row]));
    const quoteRowBySymbol = new Map(stockQuoteRows.map((row) => [normalizeWatchlistSymbol(row.symbol || row.company), row]));
    const cardBySymbol = useMemo(
        () => new Map(stockCardItems.map((row) => [normalizeWatchlistSymbol(row.symbol), row])),
        [stockCardItems],
    );
    const indexRowBySymbol = new Map((index?.constituents || []).map((row) => [normalizeWatchlistSymbol(row.symbol), row]));
    const visibleWatchlistChartSymbols = useMemo(
        () => watchlistItems
            .slice(0, 5)
            .filter((item) => item.kind === 'stock')
            .map((item) => normalizeWatchlistSymbol(item.symbol))
            .filter(Boolean),
        [watchlistItems],
    );
    const watchlistChartSymbolsToLoad = useMemo(
        () => visibleWatchlistChartSymbols.filter((symbol) => (
            !mobileSparklinePath(cardBySymbol.get(symbol)?.line_points ?? [])
            && !(symbol in watchlistChartPoints)
        )),
        [cardBySymbol, visibleWatchlistChartSymbols, watchlistChartPoints],
    );

    useEffect(() => {
        if (!isMobileViewport || watchlistChartSymbolsToLoad.length === 0) return undefined;
        const controller = new AbortController();
        let active = true;
        void Promise.all(watchlistChartSymbolsToLoad.map(async (symbol) => {
            try {
                const payload = await apiClient.marketStockCardChart(symbol, '1d', { signal: controller.signal });
                return [symbol, payload.line_points ?? []] as const;
            } catch {
                return [symbol, null] as const;
            }
        })).then((entries) => {
            if (!active || controller.signal.aborted) return;
            setWatchlistChartPoints((current) => ({
                ...current,
                ...Object.fromEntries(entries),
            }));
        });
        return () => {
            active = false;
            controller.abort();
        };
    }, [isMobileViewport, watchlistChartSymbolsToLoad]);

    useEffect(() => {
        onSearchOpenChange(searchOpen);
        return () => onSearchOpenChange(false);
    }, [onSearchOpenChange, searchOpen]);
    const savedStockSymbols = new Set(
        watchlistItems
            .filter((item) => item.kind === 'stock')
            .map((item) => normalizeWatchlistSymbol(item.symbol)),
    );
    const isWatchlistFull = watchlistItems.length >= MAX_WATCHLIST_ITEMS;
    const normalizedSearch = searchTerm.trim().toUpperCase();
    useEffect(() => {
        const query = searchTerm.trim();
        if (!searchOpen || !query) {
            setCompanySearchItems([]);
            setCompanySearchLoading(false);
            return;
        }
        let active = true;
        const controller = new AbortController();
        setCompanySearchLoading(true);
        const timer = window.setTimeout(() => {
            apiClient.marketStockSearch({ q: query, index: 'XUTUM', limit: 20, signal: controller.signal })
                .then((payload) => {
                    if (active) setCompanySearchItems(payload.rows || []);
                })
                .catch((error) => {
                    if ((error as Error)?.name === 'AbortError') return;
                    if (active) setCompanySearchItems([]);
                })
                .finally(() => {
                    if (active) setCompanySearchLoading(false);
                });
        }, 160);
        return () => {
            active = false;
            window.clearTimeout(timer);
            controller.abort();
        };
    }, [searchOpen, searchTerm]);

    useEffect(() => {
        const query = searchTerm.trim();
        if (!searchOpen || !query) {
            setFundSearchItems([]);
            setFundSearchLoading(false);
            return undefined;
        }

        let active = true;
        setFundSearchLoading(true);
        const timer = window.setTimeout(() => {
            searchWatchlistFunds(query)
                .then((payload) => {
                    if (active) setFundSearchItems(payload);
                })
                .catch(() => {
                    if (active) setFundSearchItems([]);
                })
                .finally(() => {
                    if (active) setFundSearchLoading(false);
                });
        }, 180);
        return () => {
            active = false;
            window.clearTimeout(timer);
        };
    }, [searchOpen, searchTerm]);

    const searchCandidates = new Map<string, { symbol: string; name: string; logoUrl?: string | null }>();
    for (const row of rows) {
        const symbol = normalizeWatchlistSymbol(row.company);
        searchCandidates.set(symbol, { symbol, name: row.name || row.company, logoUrl: row.logo_url });
    }
    for (const item of companySearchItems) {
        const symbol = normalizeWatchlistSymbol(item.symbol || item.company);
        if (!symbol || searchCandidates.has(symbol)) continue;
        searchCandidates.set(symbol, { symbol, name: item.name || symbol, logoUrl: item.logo_url });
    }
    const searchResults = normalizedSearch
        ? [...searchCandidates.values()]
            .filter((item) => !savedStockSymbols.has(item.symbol))
            .filter((item) => (
                item.symbol.includes(normalizedSearch)
                || item.name.toLocaleUpperCase('tr-TR').includes(normalizedSearch)
            ))
            .slice(0, 8)
        : [];
    const savedFundCodes = new Set(
        watchlistItems
            .filter((item) => item.kind === 'fund')
            .map((item) => normalizeWatchlistSymbol(item.symbol)),
    );
    const fundSearchResults = fundSearchItems
        .filter((fund) => !savedFundCodes.has(normalizeWatchlistSymbol(fund.fund_code)))
        .slice(0, 8);
    const visibleWatchlist = watchlistItems.slice(0, 5).map((item) => ({
        item,
        row: item.kind === 'stock' ? rowBySymbol.get(normalizeWatchlistSymbol(item.symbol)) || null : null,
    }));
    const sparkline = mobileSparklinePath(index?.line_points || []);

    const handleWatchlistPointerDown = (event: ReactPointerEvent<HTMLButtonElement>, key: string) => {
        if (event.pointerType === 'mouse' && event.button !== 0) return;
        event.currentTarget.setPointerCapture?.(event.pointerId);
        swipeStartRef.current = { key, x: event.clientX, y: event.clientY };
    };

    const handleWatchlistPointerUp = (event: ReactPointerEvent<HTMLButtonElement>, key: string) => {
        const start = swipeStartRef.current;
        swipeStartRef.current = null;
        if (!start || start.key !== key) return;

        const deltaX = event.clientX - start.x;
        const deltaY = event.clientY - start.y;
        if (Math.abs(deltaX) < 45 || Math.abs(deltaX) <= Math.abs(deltaY)) return;

        suppressRowClickRef.current = true;
        setSwipedWatchlistKey(deltaX < 0 ? key : null);
        window.setTimeout(() => {
            suppressRowClickRef.current = false;
        }, 0);
    };

    const handleWatchlistRowClick = (item: WatchlistItem, symbol: string) => {
        if (suppressRowClickRef.current) return;
        if (swipedWatchlistKey) {
            setSwipedWatchlistKey(null);
            return;
        }
        if (item.kind === 'fund') onOpenFund?.(symbol);
        else onSelectTicker(symbol);
    };

    return (
        <section className={`mobile-market-overview${searchOpen ? ' is-watchlist-search-open' : ''}`} aria-label="Mobil piyasa özeti">
            <button
                type="button"
                className="mobile-market-index-card"
                onClick={() => onSelectIndex('XU100')}
                aria-label="XU100 endeks detayını aç"
            >
                <div className="mobile-market-index-head">
                    <div className="mobile-market-index-identity">
                        <SymbolLogo symbol="XU100" name="BIST 100" kind="index" size="md" />
                        <div>
                            <strong>XU100</strong>
                            <span>BIST 100</span>
                            <small>{index ? formatUpdateTime(index.as_of) : 'Güncelleniyor…'}</small>
                        </div>
                    </div>
                    <div className="mobile-market-index-quote">
                        <span title="Günlük">Günlük</span>
                        <strong>{formatIndexPrice(index?.price ?? null)}</strong>
                        <b className={getTableChangeClass(index?.change_pct ?? null)}>
                            {formatTablePct(index?.change_pct ?? null)}
                        </b>
                    </div>
                </div>
                <div className={`mobile-market-index-chart ${getTableChangeClass(index?.change_pct ?? null)}${!sparkline && !index ? ' is-loading' : ''}`} aria-hidden="true">
                    {sparkline ? (
                        <svg viewBox="0 0 220 52" preserveAspectRatio="none">
                            <path d={sparkline} />
                        </svg>
                    ) : (
                        <span />
                    )}
                </div>
            </button>

            <div className="mobile-market-watchlist">
                <div className="mobile-market-watchlist-head">
                    <div>
                        <h2>İzleme listesi</h2>
                        <span>Favori hisselerini hızlıca takip et</span>
                    </div>
                </div>

                {visibleWatchlist.length ? (
                    <div className="mobile-market-watchlist-list">
                        {visibleWatchlist.map(({ item, row }) => {
                            const symbol = normalizeWatchlistSymbol(item.symbol);
                            const itemKey = `${item.kind}:${symbol}`;
                            const fundRow = item.kind === 'fund' ? fundRows[symbol] : null;
                            const quoteRow = item.kind === 'stock' ? quoteRowBySymbol.get(symbol) : null;
                            const cardRow = item.kind === 'stock' ? cardBySymbol.get(symbol) : null;
                            const indexRow = item.kind === 'stock' ? indexRowBySymbol.get(symbol) : null;
                            const price = quoteRow?.price ?? cardRow?.price ?? indexRow?.price ?? row?.price ?? fundRow?.price ?? null;
                            const changePct = quoteRow?.change_pct ?? cardRow?.change_pct ?? indexRow?.change_pct ?? row?.change_pct ?? fundRow?.daily_return ?? null;
                            const asOf = quoteRow?.price_as_of ?? cardRow?.as_of ?? row?.price_as_of ?? fundRow?.as_of ?? null;
                            const chartPoints = cardRow?.line_points?.length
                                ? cardRow.line_points
                                : (watchlistChartPoints[symbol] || []);
                            const chartPath = item.kind === 'stock' ? mobileSparklinePath(chartPoints) : '';
                            return (
                                <div
                                    key={itemKey}
                                    className={`mobile-market-watchlist-row-shell${swipedWatchlistKey === itemKey ? ' is-swiped' : ''}`}
                                >
                                    <button
                                        type="button"
                                        className="mobile-market-watchlist-row"
                                        onClick={() => handleWatchlistRowClick(item, symbol)}
                                        onPointerDown={(event) => handleWatchlistPointerDown(event, itemKey)}
                                        onPointerUp={(event) => handleWatchlistPointerUp(event, itemKey)}
                                        onPointerCancel={() => { swipeStartRef.current = null; }}
                                    >
                                        <SymbolLogo
                                            symbol={symbol}
                                            name={item.label || quoteRow?.company || cardRow?.company || indexRow?.symbol || row?.company || fundRow?.name || symbol}
                                            kind={item.kind}
                                            logoUrl={quoteRow?.logo_url ?? cardRow?.logo_url ?? indexRow?.logo_url ?? row?.logo_url}
                                            size="sm"
                                        />
                                        <span className="mobile-market-watchlist-symbol">
                                            <strong>{symbol}</strong>
                                            <small>{formatUpdateTime(asOf)}</small>
                                        </span>
                                        <span className="mobile-market-watchlist-price">
                                            <small title="Günlük">Günlük</small>
                                            {formatIndexPrice(price)}
                                        </span>
                                        <span className={`mobile-market-watchlist-chart ${getTableChangeClass(changePct)}`} aria-hidden="true">
                                            {chartPath ? (
                                                <svg viewBox="0 0 220 52" preserveAspectRatio="none">
                                                    <path d={chartPath} />
                                                </svg>
                                            ) : null}
                                        </span>
                                        <span className={`mobile-market-watchlist-change ${getTableChangeClass(changePct)}`}>
                                            {formatTablePct(changePct)}
                                        </span>
                                    </button>
                                    <button
                                        type="button"
                                        className="mobile-market-watchlist-delete"
                                        onClick={() => {
                                            onRemoveWatchlistItem(item);
                                            setSwipedWatchlistKey(null);
                                        }}
                                        aria-label={`${symbol} izleme listesinden sil`}
                                        title="İzleme listesinden sil"
                                    >
                                        <Trash2 size={19} aria-hidden="true" />
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                ) : (
                    <div className="mobile-market-watchlist-empty">
                        İzleme listene hisse veya fon eklemek için Sembol ekle butonuna dokun.
                    </div>
                )}

                {searchOpen ? (
                    <div className="mobile-market-watchlist-search-area">
                        <div className="mobile-market-watchlist-search">
                            <Search size={18} aria-hidden="true" />
                            <input
                                type="search"
                                autoFocus
                                value={searchTerm}
                                onChange={(event) => setSearchTerm(event.target.value)}
                                placeholder="Hisse veya fon ara..."
                                aria-label="İzleme listesine eklenecek hisse veya fonu ara"
                            />
                            <button
                                type="button"
                                onClick={() => {
                                    setSearchOpen(false);
                                    setSearchTerm('');
                                }}
                                aria-label="Aramayı kapat"
                                title="Kapat"
                            >
                                <X size={17} aria-hidden="true" />
                            </button>
                        </div>
                        {searchTerm.trim() && (
                            <div className="mobile-market-watchlist-search-results">
                                {companySearchLoading && fundSearchLoading && searchResults.length === 0 && fundSearchResults.length === 0 ? (
                                    <span className="mobile-market-watchlist-search-empty">Varlıklar aranıyor...</span>
                                ) : searchResults.length > 0 || fundSearchResults.length > 0 ? (
                                    <>
                                        {searchResults.map((result) => (
                                            <button
                                                key={`stock-${result.symbol}`}
                                                type="button"
                                                onClick={() => {
                                                    onAddStock(result.symbol);
                                                    setSearchTerm('');
                                                    setSearchOpen(false);
                                                }}
                                            >
                                                <SymbolLogo
                                                    symbol={result.symbol}
                                                    name={result.name}
                                                    kind="stock"
                                                    logoUrl={result.logoUrl}
                                                    size="sm"
                                                />
                                                <strong>{result.symbol}</strong>
                                                <span>Hisse · Ekle</span>
                                            </button>
                                        ))}
                                        {fundSearchResults.map((fund) => (
                                            <button
                                                key={`fund-${fund.fund_code}`}
                                                type="button"
                                                onClick={() => {
                                                    onAddFund(fund);
                                                    setSearchTerm('');
                                                    setSearchOpen(false);
                                                }}
                                            >
                                                <SymbolLogo
                                                    symbol={fund.fund_code}
                                                    name={fund.name}
                                                    kind="fund"
                                                    size="sm"
                                                />
                                                <strong>{fund.fund_code}</strong>
                                                <span>Fon · Ekle</span>
                                            </button>
                                        ))}
                                    </>
                                ) : (
                                    <span className="mobile-market-watchlist-search-empty">Sonuç bulunamadı.</span>
                                )}
                            </div>
                        )}
                    </div>
                ) : (
                    <button
                        type="button"
                        className="mobile-market-watchlist-add"
                        onClick={() => setSearchOpen(true)}
                        disabled={isWatchlistFull}
                    >
                        <Plus size={20} aria-hidden="true" />
                        {isWatchlistFull ? `İzleme listesi dolu (${MAX_WATCHLIST_ITEMS}/${MAX_WATCHLIST_ITEMS})` : 'Sembol ekle'}
                    </button>
                )}
            </div>
        </section>
    );
}

type MarketUniverseMemoryCache = { data: MarketUniverseResponse; fetchedAt: number };
type MarketStockCardsMemoryCache = { data: MarketStockCardsResponse; fetchedAt: number };

// Keep the last visible market data while the user changes sections.  The
// backend still refreshes in the background, so this is a stale-while-
// revalidate UX cache rather than a replacement for server-side freshness.
let marketUniverseMemoryCache: MarketUniverseMemoryCache | null = null;
const marketStockCardsMemoryCache = new BoundedMemoryCache<string, MarketStockCardsMemoryCache>(50);

function stockCardsMemoryKey(symbols: string[]): string {
    return symbols.join(',');
}

function emptyMarketReturnBenchmark(): MarketReturnBenchmark {
    return {
        return_1w_pct: null,
        return_1m_pct: null,
        return_3m_pct: null,
        return_6m_pct: null,
        return_ytd_pct: null,
        return_1y_pct: null,
        as_of: null,
    };
}

function quickStocksFromUniverse(
    index: MarketStockIndex,
    universe: MarketUniverseResponse,
): MarketStocksResponse {
    const rows: MarketStockRow[] = (universe.rows || []).map((row) => ({
        ...row,
        symbol: row.symbol || row.company,
        volume: null,
        return_1w_pct: null,
        return_1m_pct: null,
        return_3m_pct: null,
        return_6m_pct: null,
        return_ytd_pct: null,
        return_1y_pct: null,
    }));
    const benchmark = emptyMarketReturnBenchmark();
    return {
        index,
        rows,
        benchmarks: {
            XUTUM: benchmark,
            XU100: benchmark,
            XU030: benchmark,
        },
        source: 'market_universe_quick',
        universe: universe.universe,
        as_of: universe.universe?.fetched_at || new Date().toISOString(),
        cache_status: 'hit',
        quote_status: 'fresh',
        stale: false,
        partial: true,
        partial_message: 'Detaylı hacim ve getiri verileri arka planda yükleniyor…',
    };
}

function mergeStockCardPayloads(
    current: MarketStockCardsResponse | null,
    next: MarketStockCardsResponse,
    orderedSymbols: string[],
): MarketStockCardsResponse {
    const quoteTimestamp = (item: MarketStockCardItem | undefined): number => {
        const timestamp = item?.as_of ? Date.parse(item.as_of) : Number.NaN;
        return Number.isFinite(timestamp) ? timestamp : 0;
    };
    const shouldUseNextQuote = (currentItem: MarketStockCardItem, nextItem: MarketStockCardItem): boolean => {
        const currentTimestamp = quoteTimestamp(currentItem);
        const nextTimestamp = quoteTimestamp(nextItem);
        if (currentTimestamp === 0) return true;
        if (nextTimestamp === 0) return false;
        return nextTimestamp >= currentTimestamp;
    };
    const mergeQuoteIntoCard = (
        card: MarketStockCardItem,
        quote: MarketStockCardItem,
    ): MarketStockCardItem => ({
        ...card,
        price: quote.price ?? card.price,
        currency: quote.currency || card.currency,
        change: quote.change ?? card.change,
        change_pct: quote.change_pct ?? card.change_pct,
        volume: quote.volume ?? card.volume,
        volume_tl: quote.volume_tl ?? card.volume_tl,
        market_cap: card.market_cap ?? quote.market_cap,
        market_state: quote.market_state || card.market_state,
        as_of: quote.as_of || card.as_of,
        error: quote.price != null ? null : card.error || quote.error,
    });
    const itemsBySymbol = new Map<string, MarketStockCardItem>();
    for (const item of current?.items || []) itemsBySymbol.set(item.symbol, item);
    for (const item of next.items || []) {
        const currentItem = itemsBySymbol.get(item.symbol);
        if (!currentItem) {
            itemsBySymbol.set(item.symbol, item);
            continue;
        }

        const latestQuote = shouldUseNextQuote(currentItem, item) ? item : currentItem;
        // Quick polling updates the quote fields in place while preserving the
        // chart, valuation and return data already loaded for a full card.
        if (item.card_stage === 'quick' && currentItem.card_stage !== 'quick') {
            itemsBySymbol.set(item.symbol, mergeQuoteIntoCard(currentItem, latestQuote));
        } else if (item.card_stage === 'full') {
            // A slower full response must not overwrite a newer 3-second quote
            // response that finished while its enrichment was running.
            itemsBySymbol.set(item.symbol, mergeQuoteIntoCard(item, latestQuote));
        } else if (shouldUseNextQuote(currentItem, item)) {
            itemsBySymbol.set(item.symbol, item);
        }
    }
    return {
        ...next,
        items: orderedSymbols
            .map((symbol) => itemsBySymbol.get(symbol))
            .filter((item): item is MarketStockCardItem => Boolean(item)),
    };
}

function splitStockCardBatches(symbols: string[], size: number): string[][] {
    const batches: string[][] = [];
    for (let index = 0; index < symbols.length; index += size) {
        batches.push(symbols.slice(index, index + size));
    }
    return batches;
}

function getTableChangeClass(value: number | null): string {
    if (value == null || value === 0) return 'stocks-flat';
    return value > 0 ? 'stocks-up' : 'stocks-down';
}

function numericOrNull(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function indexCellValue(row: MarketIndexListRow, key: keyof MarketIndexListRow): string {
    if (key === 'symbol') return row.symbol;
    if (key === 'price') return formatIndexPrice(row.price);
    if (key === 'volume') return formatVolume(row.volume);
    if (String(key).includes('return') || key === 'change_pct') {
        return formatTablePct(numericOrNull(row[key]));
    }
    const value = row[key];
    return value == null ? '-' : String(value);
}

function constituentPrice(row: MarketIndexConstituent): string {
    return formatMaybeCurrency(row.price, row.price_currency);
}

function formatWeight(value: number | null): string {
    if (value == null) return '-';
    return `% ${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatPointEffect(value: number | null): string {
    if (value == null) return '-';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatPointEffectShort(value: number | null): string {
    const formatted = formatPointEffect(value);
    return formatted === '-' ? formatted : `${formatted}p`;
}

function getImpactPct(row: MarketIndexConstituent, indexLevel: number | null): number | null {
    if (row.point_effect == null || indexLevel == null || indexLevel <= 0) return null;
    return (row.point_effect / indexLevel) * 100;
}

function formatImpactPct(value: number | null): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}%`;
}

function formatHeatmapChangePct(value: number | null): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}%`;
}

function isReturnKey(key: StockSortKey): key is StockReturnKey {
    return RETURN_KEYS.includes(key as StockReturnKey);
}

function getBenchmarkIndex(returnMode: StockReturnMode): MarketStockIndex | null {
    if (returnMode === 'relative_xu100') return 'XU100';
    if (returnMode === 'relative_xu030') return 'XU030';
    return null;
}

function getReturnValue(
    row: MarketStockRow,
    key: StockReturnKey,
    benchmarks: Record<MarketStockIndex, MarketReturnBenchmark> | undefined,
    returnMode: StockReturnMode,
): number | null {
    const rawValue = row[key];
    const benchmarkIndex = getBenchmarkIndex(returnMode);
    if (!benchmarkIndex) return rawValue;

    const benchmarkValue = benchmarks?.[benchmarkIndex]?.[key];
    if (rawValue == null || benchmarkValue == null) return null;
    return Math.round((rawValue - benchmarkValue) * 100) / 100;
}

function getColumnLabel(key: StockSortKey, _returnMode: StockReturnMode): string {
    return STOCK_COLUMNS.find((column) => column.key === key)?.label || '';
}

function stockSortValue(
    row: MarketStockRow,
    key: StockSortKey,
    benchmarks: Record<MarketStockIndex, MarketReturnBenchmark> | undefined,
    returnMode: StockReturnMode,
    sectorDiff?: StockSectorDiffFields,
): string | number | null | undefined {
    if (key === 'company') return row.company;
    if (key === 'fk_sector_diff_pct') return sectorDiff?.fk_sector_diff_pct ?? null;
    if (isReturnKey(key)) return getReturnValue(row, key, benchmarks, returnMode);
    return row[key];
}

function constituentSortValue(
    row: MarketIndexConstituent,
    key: IndexConstituentSortKey,
    indexLevel: number | null,
): string | number | null {
    if (key === 'symbol') return row.symbol;
    if (key === 'impact_abs') return row.point_effect == null ? null : Math.abs(row.point_effect);
    if (key === 'impact_pct') return getImpactPct(row, indexLevel);
    if (key === 'price') return row.price;
    if (key === 'change_pct') return row.change_pct;
    if (key === 'volume') return row.volume;
    if (key === 'weight_pct') return row.weight_pct;
    if (key === 'point_effect') return row.point_effect;
    return null;
}

const StockTableRow = memo(function StockTableRow({
    row,
    rank,
    mode,
    returnMode,
    benchmarks,
    sectorDiff,
    onCompanyClick,
}: {
    row: MarketStockRow;
    rank: number;
    mode: 'performance' | 'fundamentals';
    returnMode: StockReturnMode;
    benchmarks?: MarketStocksResponse['benchmarks'];
    sectorDiff?: StockSectorDiffFields;
    onCompanyClick: (symbol: string) => void;
}) {
    const prevPriceRef = useRef(row.price);
    const [flashClass, setFlashClass] = useState('');

    useEffect(() => {
        if (row.price != null && prevPriceRef.current != null && row.price !== prevPriceRef.current) {
            setFlashClass(row.price > prevPriceRef.current ? 'stocks-flash-up' : 'stocks-flash-down');
            const timer = window.setTimeout(() => setFlashClass(''), 1100);
            prevPriceRef.current = row.price;
            return () => window.clearTimeout(timer);
        }
        prevPriceRef.current = row.price;
    }, [row.price]);

    const symbol = row.symbol || row.company;

    return (
        <tr
            aria-rowindex={rank + 1}
            className={flashClass}
            onClick={(event: ReactMouseEvent<HTMLTableRowElement>) => {
                if (event.button !== 0) return;
                onCompanyClick(row.company);
            }}
            onAuxClick={(event: ReactMouseEvent<HTMLTableRowElement>) => {
                if (event.button !== 1) return;
                if ((event.target as Element).closest('a[href]')) return;
                event.preventDefault();
                window.open(toStockDetail(symbol), '_blank', 'noopener,noreferrer');
            }}
        >
            <td className="stocks-rank">{rank}</td>
            <td className="stocks-symbol-cell">
                <a
                    className="stocks-symbol-link"
                    href={toStockDetail(row.company)}
                    onClick={(event) => {
                        event.stopPropagation();
                        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                        event.preventDefault();
                        onCompanyClick(row.company);
                    }}
                >
                    <span className="stocks-symbol-main">
                        <SymbolLogo
                            symbol={row.company}
                            name={row.company}
                            kind="stock"
                            logoUrl={row.logo_url}
                            size="xs"
                            className="stocks-symbol-logo"
                        />
                        <span>{row.company}</span>
                    </span>
                </a>
            </td>
            {mode === 'fundamentals' ? (
                <>
                    <td title={row.sector || undefined}>{row.sector || '—'}</td>
                    <td>{row.fundamental_period || row.latest_quarter || '—'}</td>
                    <td className="stocks-cell-right stocks-price-cell">{formatStockPrice(row)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.market_cap)}</td>
                    <td className="stocks-cell-right">{formatFundamentalRatio(row.fk)}</td>
                    <td className="stocks-cell-right">{formatFundamentalRatio(row.pd_dd)}</td>
                    <td className="stocks-cell-right">{formatFundamentalRatio(row.fd_favok)}</td>
                    <td className="stocks-cell-right">{formatFundamentalRatio(row.fd_satis)}</td>
                    <td className="stocks-cell-right" title={(row.hist_fk_quarters ?? row.hist_quarters) != null ? `Son ${row.hist_fk_quarters ?? row.hist_quarters} çeyrek KAP ortalaması` : undefined}>
                        {formatFundamentalRatio(row.hist_fk_avg)}
                        {row.hist_fk_avg != null && (row.hist_fk_quarters ?? row.hist_quarters) != null ? (
                            <span className="stocks-cell-muted"> ({row.hist_fk_quarters ?? row.hist_quarters}Ç)</span>
                        ) : null}
                    </td>
                    <td className={`stocks-cell-right ${row.fk_hist_diff_pct == null ? 'stocks-cell-muted' : row.fk_hist_diff_pct < 0 ? 'stocks-up' : row.fk_hist_diff_pct > 0 ? 'stocks-down' : 'stocks-flat'}`}>
                        {formatTablePct(row.fk_hist_diff_pct ?? null)}
                    </td>
                    <td className="stocks-cell-right" title={(row.hist_pd_dd_quarters ?? row.hist_quarters) != null ? `Son ${row.hist_pd_dd_quarters ?? row.hist_quarters} çeyrek KAP ortalaması` : undefined}>
                        {formatFundamentalRatio(row.hist_pd_dd_avg)}
                        {row.hist_pd_dd_avg != null && (row.hist_pd_dd_quarters ?? row.hist_quarters) != null ? (
                            <span className="stocks-cell-muted"> ({row.hist_pd_dd_quarters ?? row.hist_quarters}Ç)</span>
                        ) : null}
                    </td>
                    <td className={`stocks-cell-right ${row.pd_dd_hist_diff_pct == null ? 'stocks-cell-muted' : row.pd_dd_hist_diff_pct < 0 ? 'stocks-up' : row.pd_dd_hist_diff_pct > 0 ? 'stocks-down' : 'stocks-flat'}`}>
                        {formatTablePct(row.pd_dd_hist_diff_pct ?? null)}
                    </td>
                    <td className="stocks-cell-right" title={(row.hist_fd_favok_quarters ?? row.hist_quarters) != null ? `Son ${row.hist_fd_favok_quarters ?? row.hist_quarters} çeyrek KAP ortalaması` : undefined}>
                        {formatFundamentalRatio(row.hist_fd_favok_avg)}
                        {row.hist_fd_favok_avg != null && (row.hist_fd_favok_quarters ?? row.hist_quarters) != null ? (
                            <span className="stocks-cell-muted"> ({row.hist_fd_favok_quarters ?? row.hist_quarters}Ç)</span>
                        ) : null}
                    </td>
                    <td className={`stocks-cell-right ${sectorDiff?.fk_sector_diff_pct == null ? 'stocks-cell-muted' : sectorDiff.fk_sector_diff_pct < 0 ? 'stocks-up' : sectorDiff.fk_sector_diff_pct > 0 ? 'stocks-down' : 'stocks-flat'}`}>
                        {formatTablePct(sectorDiff?.fk_sector_diff_pct ?? null)}
                    </td>
                    <td className={`stocks-cell-right ${row.sma_50_diff_pct == null ? 'stocks-cell-muted' : getTableChangeClass(row.sma_50_diff_pct)}`}>
                        {formatTablePct(row.sma_50_diff_pct ?? null)}
                    </td>
                    <td className={`stocks-cell-right ${row.sma_200_diff_pct == null ? 'stocks-cell-muted' : getTableChangeClass(row.sma_200_diff_pct)}`}>
                        {formatTablePct(row.sma_200_diff_pct ?? null)}
                    </td>
                    <td className={`stocks-cell-right ${row.high_52w_dist_pct == null ? 'stocks-cell-muted' : getTableChangeClass(row.high_52w_dist_pct)}`}>
                        {formatTablePct(row.high_52w_dist_pct ?? null)}
                    </td>
                    <td className="stocks-cell-right">{formatFundamentalRatio(row.net_borc_favok)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.ttm_net_kar ?? null)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.ttm_fcf ?? null)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.ttm_satis_gelirleri ?? null)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.ttm_favok ?? null)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.net_borc ?? null)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.ozkaynaklar ?? null)}</td>
                    <td className="stocks-cell-right">{formatFundamentalPercent(row.roe_pct)}</td>
                    <td className="stocks-cell-right">{formatFundamentalPercent(row.free_float_pct)}</td>
                </>
            ) : (
                <>
                    <td className="stocks-cell-right stocks-price-cell">{formatStockPrice(row)}</td>
                    <td className={`stocks-cell-right ${row.change_pct == null ? 'stocks-cell-muted' : getTableChangeClass(row.change_pct)}`}>
                        {formatTablePct(row.change_pct)}
                    </td>
                    <td className="stocks-cell-right stocks-volume-cell">{formatVolume(row.volume)}</td>
                    <td className="stocks-cell-right stocks-market-cap-cell">
                        {formatMarketCap(row.market_cap)}
                    </td>
                    {RETURN_KEYS.map((k) => {
                        const val = getReturnValue(row, k, benchmarks, returnMode);
                        const isNull = val == null || !Number.isFinite(val);
                        return (
                            <td
                                key={k}
                                className={`stocks-cell-right ${isNull ? 'stocks-cell-muted' : getTableChangeClass(val)}`}
                            >
                                {formatTablePct(val)}
                            </td>
                        );
                    })}
                </>
            )}
        </tr>
    );
});


const isBistSymbol = (sym?: string) => {
    if (!sym) return true;
    const globalSymbols = ['SP500', 'NASDAQ', 'DOW', 'DAX', 'FTSE', 'CAC40', 'NIKKEI', 'HANGSENG', 'VIX', 'DXY'];
    if (globalSymbols.includes(sym.toUpperCase())) return false;
    if (sym.includes('/')) return false;
    return true;
};

function IndexLineChart({
    symbol,
    points,
    prevClose,
    changePct,
}: {
    symbol?: string;
    points: MarketIndexDetailResponse['line_points'];
    prevClose: number | null;
    changePct: number | null;
}) {
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const [measureAnchorIndex, setMeasureAnchorIndex] = useState<number | null>(null);
    const width = 1120;
    const height = 400;
    const padding = { top: 30, right: 65, bottom: 40, left: 16 };
    const validPoints = points.filter((point) => Number.isFinite(point.close));
    const pointWindowKey = `${symbol || ''}:${validPoints.length}:${validPoints[0]?.time || ''}:${validPoints[validPoints.length - 1]?.time || ''}`;

    useEffect(() => {
        setHoverIndex(null);
        setMeasureAnchorIndex(null);
    }, [pointWindowKey]);

    if (validPoints.length < 2) {
        return <div className="indices-chart-empty">Grafik verisi bekleniyor.</div>;
    }

    const values = validPoints.map((point) => point.close);
    if (prevClose != null) values.push(prevClose);
    
    // Add small visual padding around min/max values
    let minValue = Math.min(...values);
    let maxValue = Math.max(...values);
    const spanRaw = Math.max(1, maxValue - minValue);
    minValue -= spanRaw * 0.05;
    maxValue += spanRaw * 0.05;
    const span = maxValue - minValue;

    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const isBist = isBistSymbol(symbol);
    const useTimeScale = isBist && validPoints.length > 0;
    let startTimeMs = 0;
    let endTimeMs = 0;
    if (useTimeScale) {
        const d = new Date(validPoints[0].time);
        const start = new Date(d); start.setHours(10, 0, 0, 0);
        const end = new Date(d); end.setHours(18, 0, 0, 0);
        startTimeMs = start.getTime();
        endTimeMs = end.getTime();
    }

    const xFor = (index: number) => {
        if (useTimeScale) {
            const pointTime = new Date(validPoints[index].time).getTime();
            let ratio = (pointTime - startTimeMs) / (endTimeMs - startTimeMs);
            ratio = Math.max(0, Math.min(1, ratio));
            return padding.left + ratio * plotWidth;
        }
        return padding.left + (validPoints.length === 1 ? 0 : (index / (validPoints.length - 1)) * plotWidth);
    };
    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;

    const last = validPoints[validPoints.length - 1].close;

    // Y ekseni çizgileri (5 adet)
    const tickCount = 6;
    const tickValues = Array.from({ length: tickCount }).map((_, i) => minValue + (span * i) / (tickCount - 1));

    // X ekseni (saat başları için tahmini çizim veya eşit dağılımlı)
    const timeTickLabels: Array<{x: number, label: string, key: string}> = [];
    if (useTimeScale) {
        for (let h = 10; h <= 18; h += 1) {
            const d = new Date(validPoints[0].time);
            d.setHours(h, 0, 0, 0);
            timeTickLabels.push({
                x: padding.left + ((d.getTime() - startTimeMs) / (endTimeMs - startTimeMs)) * plotWidth,
                label: `${h.toString().padStart(2, '0')}:00`,
                key: `fixed-${h}`
            });
        }
    } else {
        const timeTickCount = 8;
        const timeTicks = Array.from({ length: timeTickCount }).map((_, i) => Math.floor((validPoints.length - 1) * (i / (timeTickCount - 1))));
        timeTicks.forEach(index => {
            const dt = new Date(validPoints[index].time);
            const label = Number.isNaN(dt.getTime()) ? '' : dt.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
            timeTickLabels.push({ x: xFor(index), label, key: `dyn-${index}` });
        });
    }

    const chartColor = changePct != null && changePct < 0 ? '#ff4d5e' : '#22c55e';

    const pathData = validPoints
        .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(index)} ${yFor(point.close)}`)
        .join(' ');
        
    const areaData = `${pathData} L ${xFor(validPoints.length - 1)} ${height - padding.bottom} L ${padding.left} ${height - padding.bottom} Z`;

    const closestIndexForX = (x: number) => {
        let closestIndex = 0;
        let minDiff = Infinity;
        for (let i = 0; i < validPoints.length; i++) {
            const diff = Math.abs(xFor(i) - x);
            if (diff < minDiff) {
                minDiff = diff;
                closestIndex = i;
            }
        }
        return closestIndex;
    };

    const chartXFromEvent = (event: ReactPointerEvent<SVGSVGElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return null;
        return Math.min(Math.max(((event.clientX - rect.left) / rect.width) * width, 0), width);
    };

    const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const x = chartXFromEvent(event);
        if (x == null) return;
        setHoverIndex(closestIndexForX(x));
    };

    const handlePointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
        if (event.pointerType === 'mouse' && event.button !== 0) return;
        event.preventDefault();
        const x = chartXFromEvent(event);
        if (x == null) return;
        const closestIndex = closestIndexForX(x);
        setHoverIndex(closestIndex);
        setMeasureAnchorIndex(closestIndex);
        event.currentTarget.setPointerCapture?.(event.pointerId);
    };

    const handlePointerUp = (event: ReactPointerEvent<SVGSVGElement>) => {
        setMeasureAnchorIndex(null);
        try {
            if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
            }
        } catch {
            // Ignore release errors when the browser has already cancelled capture.
        }
    };

    const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);
    const activeHoverIndex = hoverIndex == null ? null : clamp(hoverIndex, 0, validPoints.length - 1);
    const hoverPoint = activeHoverIndex == null ? null : validPoints[activeHoverIndex];
    const hoverX = activeHoverIndex == null ? null : xFor(activeHoverIndex);
    const hoverY = hoverPoint ? yFor(hoverPoint.close) : null;
    const measureAnchorPoint = measureAnchorIndex == null
        ? null
        : validPoints[clamp(measureAnchorIndex, 0, validPoints.length - 1)];
    const measureEndPoint = measureAnchorPoint && hoverPoint && activeHoverIndex != null && activeHoverIndex !== measureAnchorIndex
        ? hoverPoint
        : null;
    const measureAnchorX = measureAnchorIndex == null ? null : xFor(clamp(measureAnchorIndex, 0, validPoints.length - 1));
    const measureAnchorY = measureAnchorPoint ? yFor(measureAnchorPoint.close) : null;
    const measureEndX = measureEndPoint && activeHoverIndex != null ? xFor(activeHoverIndex) : null;
    const measureEndY = measureEndPoint ? yFor(measureEndPoint.close) : null;
    const measureStartValue = measureAnchorPoint?.close ?? null;
    const measureEndValue = measureEndPoint?.close ?? null;
    const measureDelta = measureStartValue != null && measureEndValue != null
        ? measureEndValue - measureStartValue
        : null;
    const measureReturn = measureStartValue != null && measureEndValue != null && measureStartValue !== 0
        ? ((measureEndValue - measureStartValue) / measureStartValue) * 100
        : null;
    const measureLineY = measureAnchorY != null && measureEndY != null
        ? Math.max(padding.top + 12, Math.min(measureAnchorY, measureEndY) - 18)
        : null;
    const measureTooltipWidth = 190;
    const measureTooltipHeight = 70;
    const measureTooltipGap = 14;
    const measureTooltipAnchorX = hoverX ?? measureEndX;
    const measureTooltipAnchorY = hoverY ?? measureEndY;
    const measureTooltipShouldFlip = measureTooltipAnchorX != null
        && measureTooltipAnchorX + measureTooltipGap + measureTooltipWidth > width - 8;
    const measureTooltipX = measureTooltipAnchorX == null
        ? 0
        : measureTooltipShouldFlip
            ? Math.max(8, measureTooltipAnchorX - measureTooltipGap - measureTooltipWidth)
            : Math.min(measureTooltipAnchorX + measureTooltipGap, width - measureTooltipWidth - 8);
    const measureTooltipY = measureTooltipAnchorY == null
        ? 0
        : measureTooltipAnchorY - measureTooltipGap - measureTooltipHeight >= padding.top
            ? measureTooltipAnchorY - measureTooltipGap - measureTooltipHeight
            : Math.min(measureTooltipAnchorY + measureTooltipGap, height - measureTooltipHeight - 8);
    const measureDeltaText = measureDelta == null
        ? '-'
        : `${measureDelta > 0 ? '+' : measureDelta < 0 ? '-' : ''}${formatIndexPrice(Math.abs(measureDelta))} puan`;
    const measureClass = measureReturn == null ? '' : measureReturn >= 0 ? 'is-positive' : 'is-negative';
    const measureLeftX = measureAnchorX != null && measureEndX != null ? Math.min(measureAnchorX, measureEndX) : null;
    const measureRightX = measureAnchorX != null && measureEndX != null ? Math.max(measureAnchorX, measureEndX) : null;
    const tooltipWidth = 145;
    const tooltipHeight = 60;
    const tooltipX = hoverX == null ? 0 : clamp(hoverX + 16, padding.left, width - tooltipWidth - padding.right);
    const tooltipY = hoverY == null ? 0 : clamp(hoverY - tooltipHeight / 2, padding.top, height - tooltipHeight - padding.bottom);

    return (
        <div style={{ backgroundColor: '#0f1214', borderRadius: '8px', border: '1px solid #1e2327', position: 'relative', overflow: 'hidden' }}>
        <svg 
            className="indices-line-chart" 
            viewBox={`0 0 ${width} ${height}`} 
            role="img" 
            aria-label="Endeks çizgi grafiği" 
            style={{ display: 'block', width: '100%', height: 'auto', borderBottom: 'none' }}
            onPointerMove={handlePointerMove}
            onPointerDown={handlePointerDown}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            onPointerLeave={() => {
                setHoverIndex(null);
                setMeasureAnchorIndex(null);
            }}
        >
            <defs>
                <linearGradient id="areaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor={chartColor} stopOpacity="0.24" />
                    <stop offset="100%" stopColor={chartColor} stopOpacity="0" />
                </linearGradient>
            </defs>
            {/* Koyu Arkaplan (Uygulamanın karanlık temasına uygun) */}
            <rect x="0" y="0" width={width} height={height} fill="#0d1113" rx="4" />
            
            {/* Yatay Grid ve Değerler */}
            {tickValues.map((value) => (
                <g key={value}>
                    <line
                        x1={padding.left}
                        x2={width - padding.right}
                        y1={yFor(value)}
                        y2={yFor(value)}
                        stroke="rgba(255,255,255,0.05)"
                        strokeWidth="1"
                    />
                    <text x={width - padding.right + 10} y={yFor(value) + 4} fill="rgba(255,255,255,0.4)" fontSize="11" fontFamily="monospace">
                        {formatIndexPrice(value)}
                    </text>
                </g>
            ))}

            {/* Önceki Kapanış Referans Çizgisi */}
            {prevClose != null && (
                <line
                    x1={padding.left}
                    x2={width - padding.right}
                    y1={yFor(prevClose)}
                    y2={yFor(prevClose)}
                    stroke="rgba(180, 180, 180, 0.3)"
                    strokeWidth="1"
                    strokeDasharray="4 4"
                />
            )}

            {/* Dikey Grid Zaman Etiketleri */}
            {timeTickLabels.map(({ x, label, key }) => (
                <g key={key}>
                    <line
                        x1={x}
                        x2={x}
                        y1={padding.top}
                        y2={height - padding.bottom}
                        stroke="rgba(255,255,255,0.03)"
                        strokeWidth="1"
                    />
                    <text x={x} y={height - 15} fill="rgba(255,255,255,0.5)" fontSize="11" fontFamily="monospace" textAnchor="middle">
                        {label}
                    </text>
                </g>
            ))}

            {/* Alan ve Çizgi (Ağ/Line) (Açılış, yüksek, düşük kullanılmıyor, SADECE CLOSE) */}
            <path d={areaData} fill="url(#areaGrad)" />
            <path d={pathData} fill="none" stroke={chartColor} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
            
            {/* Uç Noktası (Kapanış) Dot ve Pulse */}
            <circle cx={xFor(validPoints.length - 1)} cy={yFor(last)} r="4" fill={chartColor} opacity="0.6">
                <animate attributeName="r" values="4; 14; 14" keyTimes="0; 0.5; 1" dur="2s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.6; 0; 0" keyTimes="0; 0.5; 1" dur="2s" repeatCount="indefinite" />
            </circle>
            <circle cx={xFor(validPoints.length - 1)} cy={yFor(last)} r="4" fill={chartColor} />

            {/* Anlık Fiyat İşareti */}
            <g transform={`translate(${width - padding.right}, ${yFor(last)})`}>
                <rect x="0" y="-10" width="65" height="20" fill={chartColor} rx="2" />
                <path d="M 0 0 L 6 -6 L 6 6 Z" fill={chartColor} transform="translate(-5, 0)" />
                <text x="32" y="3" fill="#ffffff" fontSize="11" fontFamily="monospace" textAnchor="middle" fontWeight="bold">
                    {formatIndexPrice(last)}
                                </text>
            </g>

            {/* Drag-to-measure */}
            {measureAnchorPoint
                && measureEndPoint
                && measureAnchorX != null
                && measureAnchorY != null
                && measureEndX != null
                && measureEndY != null
                && measureLeftX != null
                && measureRightX != null
                && measureLineY != null
                && measureReturn != null
                && measureDelta != null && (
                <g className="indices-chart-measure">
                    <line
                        x1={measureAnchorX}
                        y1={measureAnchorY}
                        x2={measureEndX}
                        y2={measureEndY}
                        className="indices-chart-measure-path"
                    />
                    <line
                        x1={measureLeftX}
                        y1={measureLineY}
                        x2={measureRightX}
                        y2={measureLineY}
                        className="indices-chart-measure-line"
                    />
                    <line
                        x1={measureAnchorX}
                        y1={measureLineY - 5}
                        x2={measureAnchorX}
                        y2={measureLineY + 5}
                        className="indices-chart-measure-line"
                    />
                    <line
                        x1={measureEndX}
                        y1={measureLineY - 5}
                        x2={measureEndX}
                        y2={measureLineY + 5}
                        className="indices-chart-measure-line"
                    />
                    <circle cx={measureAnchorX} cy={measureAnchorY} r="4.2" className="indices-chart-measure-point" fill={chartColor} />
                    <circle cx={measureEndX} cy={measureEndY} r="4.2" className="indices-chart-measure-point" fill={chartColor} />
                    <g
                        className="indices-chart-measure-tooltip"
                        style={{ transform: `translate(${measureTooltipX}px, ${measureTooltipY}px)` }}
                    >
                        <rect width={measureTooltipWidth} height={measureTooltipHeight} rx="7" className="indices-chart-measure-tooltip-bg" />
                        <text x="10" y="18" className="indices-chart-measure-muted">
                            {new Intl.DateTimeFormat('tr-TR', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }).format(new Date(measureAnchorPoint.time))}
                            {' - '}
                            {new Intl.DateTimeFormat('tr-TR', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }).format(new Date(measureEndPoint.time))}
                        </text>
                        <text x="10" y="42" className={`indices-chart-measure-value ${measureClass}`}>
                            {formatTablePct(measureReturn)}
                        </text>
                        <text x="10" y="60" className="indices-chart-measure-muted">
                            {measureDeltaText}
                        </text>
                    </g>
                </g>
            )}

            {/* Hover Tooltip */}
            {measureAnchorIndex == null && hoverPoint && hoverX != null && hoverY != null && (
                <g className="indices-chart-hover">
                    <line
                        x1={hoverX}
                        x2={hoverX}
                        y1={padding.top}
                        y2={height - padding.bottom}
                        stroke="rgba(255,255,255,0.42)"
                        strokeDasharray="4 4"
                    />
                    <line
                        x1={padding.left}
                        x2={width - padding.right}
                        y1={hoverY}
                        y2={hoverY}
                        stroke="rgba(255,255,255,0.24)"
                        strokeDasharray="5 5"
                    />
                    <circle cx={hoverX} cy={hoverY} r="4" fill={chartColor} stroke="#0a0c0f" strokeWidth="2" />
                    <g transform={`translate(${tooltipX}, ${tooltipY})`}>
                        <rect width={tooltipWidth} height={tooltipHeight} rx="4" fill="#07090b" stroke="rgba(255,255,255,0.1)" />
                        <text x="12" y="22" fill="#d8dee9" fontSize="13" fontFamily="monospace">
                            {new Intl.DateTimeFormat('tr-TR', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }).format(new Date(hoverPoint.time))}
                        </text>
                        <text x="12" y="46" fill="#f8fafc" fontSize="14" fontFamily="monospace" fontWeight="700">
                            {symbol || 'Endeks'}
                        </text>
                        <text x={tooltipWidth - 12} y="46" fill="#f8fafc" fontSize="14" fontFamily="monospace" fontWeight="700" textAnchor="end">
                            {formatIndexPrice(hoverPoint.close)}
                        </text>
                    </g>
                </g>
            )}
        </svg>
        </div>
    );
}

function StockCardMiniChart({
    symbol,
    points,
    previousClose,
    changePct,
    currency,
    selectedRange,
    pendingRange,
    rangeLoading,
    rangeError,
    onRangeSelect,
    onOpenDetail,
    isLoading = false,
    isLive = false,
}: {
    symbol: string;
    points: MarketIndexLinePoint[] | undefined;
    previousClose?: number | null;
    changePct: number | null;
    currency: string | null;
    selectedRange: MarketStockCardChartRange;
    pendingRange: MarketStockCardChartRange | null;
    rangeLoading: Partial<Record<MarketStockCardChartRange, boolean>>;
    rangeError: Partial<Record<MarketStockCardChartRange, string | null>>;
    onRangeSelect: (range: MarketStockCardChartRange) => void;
    onOpenDetail?: () => void;
    isLoading?: boolean;
    isLive?: boolean;
}) {
    const canvasWrapRef = useRef<HTMLDivElement>(null);
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const [dimensions, setDimensions] = useState<{ width: number; height: number }>({
        width: 340,
        height: 125,
    });

    useLayoutEffect(() => {
        const el = canvasWrapRef.current;
        if (!el) return;

        let frameId: number | null = null;
        const updateFromRect = (rect: DOMRectReadOnly | DOMRect) => {
            const w = Math.round(rect.width);
            const h = Math.round(rect.height);
            if (w <= 0 || h <= 0) return;
            if (frameId !== null) cancelAnimationFrame(frameId);
            frameId = requestAnimationFrame(() => {
                setDimensions((prev) => (prev.width === w && prev.height === h ? prev : { width: w, height: h }));
            });
        };

        updateFromRect(el.getBoundingClientRect());

        if (typeof ResizeObserver !== 'undefined') {
            const observer = new ResizeObserver((entries) => {
                const entry = entries[0];
                if (entry) updateFromRect(entry.contentRect);
            });
            observer.observe(el);
            return () => {
                if (frameId !== null) cancelAnimationFrame(frameId);
                observer.disconnect();
            };
        }

        const onResize = () => updateFromRect(el.getBoundingClientRect());
        window.addEventListener('resize', onResize);
        return () => {
            if (frameId !== null) cancelAnimationFrame(frameId);
            window.removeEventListener('resize', onResize);
        };
    }, []);

    const { width, height } = dimensions;
    const padding = { top: 12, right: 6, bottom: 20, left: 6 };
    const validPoints = (points ?? []).filter((point) => Number.isFinite(point.close));
    const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);
    const rangeControls = (
        <div className="stock-card-chart-ranges" onClick={(event) => event.stopPropagation()}>
            {STOCK_CARD_CHART_RANGES.map((range) => {
                const isActive = selectedRange === range.id;
                const isPending = pendingRange === range.id || Boolean(rangeLoading[range.id]);
                const error = rangeError[range.id];
                return (
                    <button
                        key={range.id}
                        type="button"
                        className={[
                            'stock-card-chart-range',
                            isActive ? 'is-active' : '',
                            isPending ? 'is-loading' : '',
                            error ? 'has-error' : '',
                        ]
                            .filter(Boolean)
                            .join(' ')}
                        title={error ? `${range.title}: ${error}` : range.title}
                        aria-pressed={isActive}
                        onClick={(event) => {
                            event.stopPropagation();
                            onRangeSelect(range.id);
                        }}
                    >
                        {range.label}
                    </button>
                );
            })}
        </div>
    );
    const chartToolbar = (
        <div className="stock-card-chart-toolbar">
            {rangeControls}
            {onOpenDetail && (
                <button
                    type="button"
                    className="stock-card-chart-action-btn"
                    title={`${symbol} detaylı grafik ve analiz`}
                    aria-label={`${symbol} detaylı analiz`}
                    onClick={(event) => {
                        event.stopPropagation();
                        onOpenDetail();
                    }}
                >
                    <Maximize2 size={13} aria-hidden="true" />
                </button>
            )}
        </div>
    );
    if (isLoading) {
        return (
            <div className="stock-card-chart-shell">
                {chartToolbar}
                <div ref={canvasWrapRef} className="stock-card-chart-canvas-wrap">
                    <div className="stock-card-inline-chart-skeleton" role="status" aria-label="Grafik yükleniyor">
                        <svg
                            className="stock-card-inline-chart-svg"
                            viewBox={`0 0 ${width} ${height}`}
                            preserveAspectRatio="none"
                            aria-hidden="true"
                        >
                            {[0.28, 0.52, 0.76].map((ratio) => (
                                <line
                                    key={ratio}
                                    className="stock-card-inline-chart-gridline"
                                    x1={0}
                                    x2={width}
                                    y1={height * ratio}
                                    y2={height * ratio}
                                />
                            ))}
                        </svg>
                        <div className="stock-card-inline-chart-baseline stock-card-inline-skeleton-pulse" aria-hidden="true" />
                    </div>
                </div>
            </div>
        );
    }

    if (validPoints.length < 2) {
        return (
            <div className="stock-card-chart-shell">
                {chartToolbar}
                <div ref={canvasWrapRef} className="stock-card-chart-canvas-wrap">
                    <div className="stock-card-chart-empty">Veri yok</div>
                </div>
            </div>
        );
    }

    const values = validPoints.map((point) => point.close);
    let minValue = Math.min(...values);
    let maxValue = Math.max(...values);
    const spanRaw = Math.max(0.01, maxValue - minValue);
    minValue -= spanRaw * 0.08;
    maxValue += spanRaw * 0.08;
    const span = Math.max(0.01, maxValue - minValue);
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const isBist = isBistSymbol(symbol);
    const useTimeScale = isBist && selectedRange === '1d' && validPoints.length > 0;
    let startTimeMs = 0;
    let endTimeMs = 0;
    if (useTimeScale) {
        const d = new Date(validPoints[0].time);
        const start = new Date(d); start.setHours(10, 0, 0, 0);
        const end = new Date(d);
        const lastPointDate = new Date(validPoints[validPoints.length - 1].time);
        const hasTodayPoint = istanbulDateKey(lastPointDate) === istanbulDateKey(new Date());
        if (isLive || hasTodayPoint) {
            end.setHours(18, 0, 0, 0);
        } else {
            const lastMinutes = lastPointDate.getHours() * 60 + lastPointDate.getMinutes();
            const roundedEndHour = Math.min(18, Math.max(13, Math.ceil(lastMinutes / 60)));
            end.setHours(roundedEndHour, 0, 0, 0);
            if (end.getTime() <= lastPointDate.getTime()) {
                end.setHours(Math.min(18, end.getHours() + 1), 0, 0, 0);
            }
        }
        startTimeMs = start.getTime();
        endTimeMs = Math.max(end.getTime(), startTimeMs + 60 * 60 * 1000);
    }

    const xFor = (index: number) => {
        if (useTimeScale) {
            const pointTime = new Date(validPoints[index].time).getTime();
            let ratio = (pointTime - startTimeMs) / (endTimeMs - startTimeMs);
            ratio = Math.max(0, Math.min(1, ratio));
            return padding.left + ratio * plotWidth;
        }
        return padding.left + (validPoints.length === 1 ? 0 : (index / (validPoints.length - 1)) * plotWidth);
    };

    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;
    const pathData = validPoints
        .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(index)} ${yFor(point.close)}`)
        .join(' ');
    const areaData = `${pathData} L ${xFor(validPoints.length - 1)} ${height - 2} L ${padding.left} ${height - 2} Z`;
    const color = changePct == null || changePct >= 0 ? '#22c55e' : '#ff4d5e';
    const gradientId = `stock-card-area-${symbol.replace(/[^a-zA-Z0-9]/g, '')}-${selectedRange}`;
    const baselineValue =
        selectedRange === '1d' && Number.isFinite(previousClose)
            ? Number(previousClose)
            : validPoints.length >= 2
                ? validPoints[validPoints.length - 2].close
                : validPoints[0].close;
    const baselineY = clamp(yFor(baselineValue), padding.top, height - padding.bottom);
    const timeTickIndexes = Array.from(
        new Set([0, Math.floor((validPoints.length - 1) / 2), validPoints.length - 1]),
    );
    const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        if (rect.width <= 0) return;
        const x = ((event.clientX - rect.left) / rect.width) * width;
        let closestIndex = 0;
        let minDiff = Infinity;
        for (let i = 0; i < validPoints.length; i++) {
            const diff = Math.abs(xFor(i) - x);
            if (diff < minDiff) {
                minDiff = diff;
                closestIndex = i;
            }
        }
        setHoverIndex(closestIndex);
    };
    const activeHoverIndex = hoverIndex == null ? null : clamp(hoverIndex, 0, validPoints.length - 1);
    const hoverPoint = activeHoverIndex == null ? null : validPoints[activeHoverIndex];
    const hoverX = activeHoverIndex == null ? null : xFor(activeHoverIndex);
    const hoverY = hoverPoint ? yFor(hoverPoint.close) : null;
    const tooltipWidth = 100;
    const tooltipHeight = 36;
    const isLeftHalf = hoverX != null && hoverX < width / 2;
    const horizontalOffset = 12;
    let tooltipX = 0;
    let tooltipY = 0;
    if (hoverX != null && hoverY != null) {
        if (isLeftHalf) {
            tooltipX = Math.min(width - tooltipWidth - 6, hoverX + horizontalOffset);
        } else {
            tooltipX = Math.max(6, hoverX - tooltipWidth - horizontalOffset);
        }
        const minTop = 4;
        const maxTop = Math.max(minTop, height - tooltipHeight - 4);
        tooltipY = clamp(hoverY - tooltipHeight / 2, minTop, maxTop);
    }

    return (
        <div className="stock-card-chart-shell">
            {chartToolbar}
            <div ref={canvasWrapRef} className="stock-card-chart-canvas-wrap">
                <svg
                    className="stock-card-mini-chart"
                    viewBox={`0 0 ${width} ${height}`}
                    preserveAspectRatio="none"
                    role="img"
                    aria-label={`${symbol} ${selectedRange} çizgi grafiği`}
                    onPointerMove={handlePointerMove}
                    onPointerLeave={() => setHoverIndex(null)}
                    shapeRendering="geometricPrecision"
                    textRendering="geometricPrecision"
                >
                    <defs>
                        <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stopColor={color} stopOpacity="0.22" />
                            <stop offset="100%" stopColor={color} stopOpacity="0" />
                        </linearGradient>
                    </defs>
                    <line
                        x1={0}
                        x2={width}
                        y1={baselineY}
                        y2={baselineY}
                        stroke="rgba(255,255,255,0.12)"
                        strokeDasharray="3 3"
                    />
                    {useTimeScale ? (
                        [startTimeMs, startTimeMs + (endTimeMs - startTimeMs) / 2, endTimeMs].map((timeMs, index) => {
                            const x = index === 0 ? padding.left : index === 2 ? width - padding.right : padding.left + plotWidth / 2;
                            return (
                                <text
                                    key={timeMs}
                                    x={x}
                                    y={height - 6}
                                    fill="#71717a"
                                    fontSize="10"
                                    fontFamily="var(--font-mono)"
                                    textAnchor={index === 0 ? 'start' : index === 2 ? 'end' : 'middle'}
                                >
                                    {formatStockCardAxisTime(new Date(timeMs))}
                                </text>
                            );
                        })
                    ) : (
                        timeTickIndexes.map((index) => {
                            const x = xFor(index);
                            return (
                                <text
                                    key={index}
                                    x={x}
                                    y={height - 6}
                                    fill="#71717a"
                                    fontSize="10"
                                    fontFamily="var(--font-mono)"
                                    textAnchor={index === 0 ? 'start' : index === validPoints.length - 1 ? 'end' : 'middle'}
                                >
                                    {formatStockCardAxisDate(validPoints[index].time, selectedRange)}
                                </text>
                            );
                        })
                    )}
                    <path d={areaData} fill={`url(#${gradientId})`} />
                    <path d={pathData} fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
                    
                    {selectedRange === '1d' && validPoints.length > 0 && (
                        <circle
                            cx={xFor(validPoints.length - 1)}
                            cy={yFor(validPoints[validPoints.length - 1].close)}
                            r="3"
                            fill={color}
                        />
                    )}

                    {hoverPoint && hoverX != null && hoverY != null && (
                        <g className="stock-card-chart-hover">
                            <line
                                x1={hoverX}
                                x2={hoverX}
                                y1={padding.top}
                                y2={height - padding.bottom}
                                stroke="rgba(255,255,255,0.28)"
                                strokeDasharray="3 3"
                            />
                            <line
                                x1={padding.left}
                                x2={width - padding.right}
                                y1={hoverY}
                                y2={hoverY}
                                stroke="rgba(255,255,255,0.18)"
                                strokeDasharray="4 4"
                            />
                            <circle cx={hoverX} cy={hoverY} r="5" fill={color} opacity="0.25" />
                            <circle cx={hoverX} cy={hoverY} r="3" fill={color} stroke="#141413" strokeWidth="1.5" />
                            <g transform={`translate(${tooltipX}, ${tooltipY})`} style={{ pointerEvents: 'none' }}>
                                <rect
                                    width={tooltipWidth}
                                    height={tooltipHeight}
                                    rx="5"
                                    fill="#18181b"
                                    stroke="rgba(255,255,255,0.12)"
                                    strokeWidth="1"
                                    style={{ filter: 'drop-shadow(0 4px 10px rgba(0,0,0,0.5))' }}
                                />
                                <text x="8" y="14" fill="#a1a1aa" fontSize="9.5" fontFamily="var(--font-mono)">
                                    {formatStockCardChartDate(hoverPoint.time, selectedRange)}
                                </text>
                                <text x="8" y="28" fill="#f8fafc" fontSize="12" fontFamily="var(--font-mono)" fontWeight="700">
                                    {formatCardCurrency(hoverPoint.close, currency)}
                                </text>
                            </g>
                        </g>
                    )}
                </svg>
            </div>
        </div>
    );
}

function emptyStockCardItem(symbol: string): MarketStockCardItem {
    return {
        symbol,
        company: symbol,
        yahoo_symbol: null,
        price: null,
        currency: 'TRY',
        change: null,
        change_pct: null,
        volume: null,
        volume_lot: null,
        volume_tl: null,
        market_cap: null,
        high: null,
        low: null,
        previous_close: null,
        fk: null,
        pd_dd: null,
        fd_favok: null,
        net_borc_favok: null,
        return_1w_pct: null,
        return_1m_pct: null,
        return_3m_pct: null,
        return_6m_pct: null,
        return_ytd_pct: null,
        return_1y_pct: null,
        market_state: '',
        as_of: null,
        session_status: 'unknown',
        session_label: 'Veri bekleniyor',
        is_live: false,
        is_stale: false,
        last_trade_at: null,
        last_trade_date: null,
        line_points: [],
        error: 'data_unavailable',
        logo_url: null,
        logo_source: null,
    };
}

const STOCK_CARD_PERFORMANCE_KEYS: Array<{
    key: 'return_1w_pct' | 'return_1m_pct' | 'return_3m_pct' | 'return_6m_pct' | 'return_ytd_pct' | 'return_1y_pct';
    label: string;
}> = [
    { key: 'return_1w_pct', label: '1H' },
    { key: 'return_1m_pct', label: '1A' },
    { key: 'return_3m_pct', label: '3A' },
    { key: 'return_6m_pct', label: '6A' },
    { key: 'return_ytd_pct', label: 'YTD' },
    { key: 'return_1y_pct', label: '1Y' },
];

function MarketStockCard({
    item,
    onOpen,
    onRemove,
    onMoveStart,
    onMoveOver,
    onMoveEnd,
    isLoading = false,
    isDragging = false,
}: {
    item: MarketStockCardItem;
    onOpen: () => void;
    onRemove: () => void;
    onMoveStart: () => void;
    onMoveOver: (placement: StockCardDropPlacement) => void;
    onMoveEnd: () => void;
    isLoading?: boolean;
    isDragging?: boolean;
}) {
    const isDetailLoading = !isLoading && item.card_stage === 'quick';
    const isCardDataLoading = isLoading || isDetailLoading;
    const [selectedRange, setSelectedRange] = useState<MarketStockCardChartRange>('1d');
    const [pendingRange, setPendingRange] = useState<MarketStockCardChartRange | null>(null);
    const [chartDataByRange, setChartDataByRange] = useState<Partial<Record<MarketStockCardChartRange, MarketIndexLinePoint[]>>>(
        () => ({ '1d': item.line_points ?? [] }),
    );
    const [rangeLoading, setRangeLoading] = useState<Partial<Record<MarketStockCardChartRange, boolean>>>({});
    const [rangeError, setRangeError] = useState<Partial<Record<MarketStockCardChartRange, string | null>>>({});
    const chartAbortRef = useRef<AbortController | null>(null);
    const chartRequestIdRef = useRef(0);

    useEffect(() => {
        setChartDataByRange((previous) => ({ ...previous, '1d': item.line_points ?? [] }));
        setRangeError((previous) => ({ ...previous, '1d': null }));
    }, [item.line_points]);

    useEffect(() => {
        return () => {
            chartAbortRef.current?.abort();
        };
    }, []);

    const [hoveredData, setHoveredData] = useState<{
        base: number | null | undefined;
        high: number | null | undefined;
        low: number | null | undefined;
        anchorX: number;
    } | null>(null);
    const [tooltipActive, setTooltipActive] = useState(false);

    const performanceRef = useRef<HTMLDivElement>(null);
    const [tooltipStyles, setTooltipStyles] = useState<React.CSSProperties>({});
    const tooltipCloseTimerRef = useRef<number | null>(null);
    const tooltipOpenFrameRef = useRef<number | null>(null);

    useLayoutEffect(() => {
        if (!isCardDataLoading && hoveredData && performanceRef.current) {
            const performanceRect = performanceRef.current.getBoundingClientRect();
            const cardElement = performanceRef.current.closest('.stock-card');
            const cardRect = cardElement?.getBoundingClientRect() ?? performanceRect;
            const halfTooltipWidth = cardRect.width / 2;
            const viewportPadding = 12;
            const minCenterX = viewportPadding + halfTooltipWidth;
            const maxCenterX = window.innerWidth - viewportPadding - halfTooltipWidth;
            const tooltipCenterX = Math.min(Math.max(hoveredData.anchorX, minCenterX), maxCenterX);
            setTooltipStyles({
                top: `${performanceRect.top - 8}px`,
                left: `${tooltipCenterX}px`,
                width: `${cardRect.width}px`,
                maxWidth: `${cardRect.width}px`,
            });
        }
    }, [hoveredData, isCardDataLoading]);

    useEffect(() => {
        return () => {
            if (tooltipCloseTimerRef.current != null) {
                window.clearTimeout(tooltipCloseTimerRef.current);
            }
            if (tooltipOpenFrameRef.current != null) {
                window.cancelAnimationFrame(tooltipOpenFrameRef.current);
            }
        };
    }, []);

    useEffect(() => {
        if (isCardDataLoading) {
            if (tooltipCloseTimerRef.current != null) {
                window.clearTimeout(tooltipCloseTimerRef.current);
                tooltipCloseTimerRef.current = null;
            }
            if (tooltipOpenFrameRef.current != null) {
                window.cancelAnimationFrame(tooltipOpenFrameRef.current);
                tooltipOpenFrameRef.current = null;
            }
            setTooltipActive(false);
            setHoveredData(null);
        }
    }, [isCardDataLoading]);

    const openPerformanceTooltip = (
        data: {
            base: number | null | undefined;
            high: number | null | undefined;
            low: number | null | undefined;
            anchorX: number;
        },
    ) => {
        if (tooltipCloseTimerRef.current != null) {
            window.clearTimeout(tooltipCloseTimerRef.current);
            tooltipCloseTimerRef.current = null;
        }
        if (tooltipOpenFrameRef.current != null) {
            window.cancelAnimationFrame(tooltipOpenFrameRef.current);
            tooltipOpenFrameRef.current = null;
        }
        setTooltipActive(false);
        setHoveredData(data);
        tooltipOpenFrameRef.current = window.requestAnimationFrame(() => {
            tooltipOpenFrameRef.current = window.requestAnimationFrame(() => {
                setTooltipActive(true);
                tooltipOpenFrameRef.current = null;
            });
        });
    };

    const closePerformanceTooltip = () => {
        if (tooltipOpenFrameRef.current != null) {
            window.cancelAnimationFrame(tooltipOpenFrameRef.current);
            tooltipOpenFrameRef.current = null;
        }
        setTooltipActive(false);
        if (tooltipCloseTimerRef.current != null) {
            window.clearTimeout(tooltipCloseTimerRef.current);
        }
        tooltipCloseTimerRef.current = window.setTimeout(() => {
            setHoveredData(null);
            tooltipCloseTimerRef.current = null;
        }, 180);
    };

    const handleRangeSelect = (nextRange: MarketStockCardChartRange) => {
        if (isCardDataLoading) {
            return;
        }
        if (nextRange === selectedRange || rangeLoading[nextRange]) {
            return;
        }

        const cachedPoints = chartDataByRange[nextRange];
        if (cachedPoints !== undefined && !rangeError[nextRange]) {
            setSelectedRange(nextRange);
            setPendingRange(null);
            return;
        }

        chartAbortRef.current?.abort();
        const controller = new AbortController();
        chartAbortRef.current = controller;
        const requestId = chartRequestIdRef.current + 1;
        chartRequestIdRef.current = requestId;
        setPendingRange(nextRange);
        setRangeLoading((previous) => ({ ...previous, [nextRange]: true }));
        setRangeError((previous) => ({ ...previous, [nextRange]: null }));

        apiClient
            .marketStockCardChart(item.symbol, nextRange, { signal: controller.signal })
            .then((payload) => {
                if (controller.signal.aborted || requestId !== chartRequestIdRef.current) {
                    return;
                }
                const nextPoints = payload.line_points ?? [];
                setChartDataByRange((previous) => ({ ...previous, [nextRange]: nextPoints }));
                if (payload.error || nextPoints.length < 2) {
                    setRangeError((previous) => ({
                        ...previous,
                        [nextRange]: payload.error || 'Veri yok',
                    }));
                    return;
                }
                setSelectedRange(nextRange);
            })
            .catch((error) => {
                if ((error as Error)?.name === 'AbortError' || requestId !== chartRequestIdRef.current) {
                    return;
                }
                setRangeError((previous) => ({
                    ...previous,
                    [nextRange]: (error as Error)?.message || 'Grafik verisi alınamadı',
                }));
            })
            .finally(() => {
                if (requestId !== chartRequestIdRef.current) {
                    return;
                }
                setRangeLoading((previous) => ({ ...previous, [nextRange]: false }));
                setPendingRange(null);
                if (chartAbortRef.current === controller) {
                    chartAbortRef.current = null;
                }
            });
    };

    const formatTooltipMetricValue = (value: number | null | undefined): string => {
        if (value == null || !Number.isFinite(value)) return '-';
        return value.toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        });
    };
    const latestPointTime = latestStockCardPointTime(item.line_points);
    const hasPreviousSessionData = item.is_stale === true || item.session_status === 'previous_session' || isPreviousIstanbulDate(item.last_trade_at || latestPointTime);
    const isCardLive = !hasPreviousSessionData && (item.is_live === true || item.session_status === 'open');
    const sessionLabel = item.session_label || (isCardLive ? 'Canlı' : hasPreviousSessionData ? 'Piyasa kapalı' : 'Piyasa kapalı');
    const sessionStatusClass = isCardLive
        ? 'is-live'
        : hasPreviousSessionData
            ? 'is-previous'
            : 'is-closed';
    const sessionTime = formatStockCardTradeTime(
        isCardLive
            ? item.as_of || item.last_trade_at || latestPointTime
            : item.last_trade_at || latestPointTime || item.as_of,
    );
    const footerLabel = isCardLive ? 'Canlı güncelleme' : hasPreviousSessionData ? 'Son işlem' : sessionLabel;

    return (
        <article
            className={`stock-card${isDragging ? ' stock-card-dragging' : ''}`}
            data-stock-card-symbol={item.symbol}
            onClick={onOpen}
            onDragEnter={(event) => {
                event.preventDefault();
            }}
            onDragOver={(event) => {
                event.preventDefault();
                event.stopPropagation();
                event.dataTransfer.dropEffect = 'move';
                const rect = event.currentTarget.getBoundingClientRect();
                const placement: StockCardDropPlacement =
                    event.clientX >= rect.left + rect.width / 2 ? 'after' : 'before';
                onMoveOver(placement);
            }}
            onDrop={(event) => {
                event.preventDefault();
                event.stopPropagation();
                onMoveEnd();
            }}
        >
            <div
                className="stock-card-controls"
                onClick={(event) => event.stopPropagation()}
            >
                <button
                    type="button"
                    className="stock-card-control stock-card-drag-handle"
                    draggable
                    onDragStart={(event) => {
                        event.stopPropagation();
                        event.dataTransfer.effectAllowed = 'move';
                        event.dataTransfer.setData('text/plain', item.symbol);
                        const cardElement = event.currentTarget.closest('.stock-card') as HTMLElement | null;
                        if (cardElement) {
                            event.dataTransfer.setDragImage(cardElement, cardElement.offsetWidth / 2, 24);
                        }
                        onMoveStart();
                    }}
                    onDragEnd={(event) => {
                        event.stopPropagation();
                        onMoveEnd();
                    }}
                    aria-label={`${item.symbol} kartını taşı`}
                    title="Kartı taşı"
                >
                    <GripHorizontal size={14} aria-hidden="true" />
                </button>
                <button
                    type="button"
                    className="stock-card-control stock-card-control-remove"
                    onClick={(event) => {
                        event.stopPropagation();
                        onRemove();
                    }}
                    aria-label={`${item.symbol} kartını kaldır`}
                    title="Kartı kaldır"
                >
                    <X size={13} aria-hidden="true" />
                </button>
            </div>
            <div className="stock-card-head">
                <div className="stock-card-identity">
                    <SymbolLogo
                        symbol={item.symbol}
                        name={item.company}
                        kind="stock"
                        logoUrl={item.logo_url}
                        size="md"
                        className="stock-card-logo"
                    />
                    <div>
                        <h3>{item.symbol}</h3>
                        <span>{item.company}</span>
                    </div>
                </div>
                {!isCardDataLoading && (
                    <span 
                        className={`stock-card-session-dot ${sessionStatusClass}`}
                        title={sessionLabel}
                    />
                )}
            </div>

            <div className="stock-card-price-row">
                {isLoading ? (
                    <>
                        <span className="stock-card-inline-skeleton-price stock-card-inline-skeleton-pulse" aria-hidden="true" />
                        <span className="stock-card-inline-skeleton-change stock-card-inline-skeleton-pulse" aria-hidden="true" />
                    </>
                ) : (
                    <>
                        <span className="stock-card-price">{formatCardCurrency(item.price, item.currency)}</span>
                        <span className={`stock-card-change ${getTableChangeClass(item.change_pct)}`}>
                            {formatCardPct(item.change_pct)}
                        </span>
                    </>
                )}
            </div>

            <div className="stock-card-metrics">
                <div>
                    <span>Yüksek</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardCurrency(item.high, item.currency)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Düşük</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardCurrency(item.low, item.currency)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Önc.Kap.</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardCurrency(item.previous_close, item.currency)
                        )}
                    </strong>
                </div>
            </div>

            <StockCardMiniChart
                symbol={item.symbol}
                points={chartDataByRange[selectedRange]}
                previousClose={item.previous_close}
                changePct={(() => {
                    if (selectedRange === '1d') return item.change_pct;
                    if (selectedRange === '1w') return item.return_1w_pct;
                    if (selectedRange === '1m') return item.return_1m_pct;
                    if (selectedRange === 'ytd') return item.return_ytd_pct;
                    if (selectedRange === '1y') return item.return_1y_pct;
                    if (selectedRange === '5y') {
                        const points = chartDataByRange['5y'];
                        if (points && points.length >= 2 && points[0].close > 0) {
                            return ((points[points.length - 1].close - points[0].close) / points[0].close) * 100;
                        }
                        return null;
                    }
                    return item.change_pct;
                })()}
                currency={item.currency}
                selectedRange={selectedRange}
                pendingRange={pendingRange}
                rangeLoading={rangeLoading}
                rangeError={rangeError}
                onRangeSelect={handleRangeSelect}
                onOpenDetail={onOpen}
                isLoading={isCardDataLoading}
                isLive={isCardLive}
            />

            <div className="stock-card-detail-metrics">
                <div>
                    <span>F/K</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardPositiveRatio(item.fk)
                        )}
                    </strong>
                </div>
                <div>
                    <span>FD/FAVÖK</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardPositiveRatio(item.fd_favok)
                        )}
                    </strong>
                </div>
                <div>
                    <span>PD/DD</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardPositiveRatio(item.pd_dd)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Net Borç/FAVÖK</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardRatio(item.net_borc_favok)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Hacim Lot</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardFullNumber(item.volume_lot)
                        )}
                    </strong>
                </div>
                <div>
                    <span>Hacim TL</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric" aria-hidden="true" />
                        ) : (
                            formatCardFullCurrency(item.volume_tl, item.currency)
                        )}
                    </strong>
                </div>
                <div className="stock-card-detail-wide">
                    <span>Piyasa Değeri</span>
                    <strong>
                        {isCardDataLoading ? (
                            <span className="stock-card-inline-skeleton-value stock-card-inline-skeleton-pulse stock-card-inline-skeleton-metric-wide" aria-hidden="true" />
                        ) : (
                            formatCardFullCurrency(item.market_cap, item.currency)
                        )}
                    </strong>
                </div>
            </div>


            <div className="stock-card-performance" ref={performanceRef}>
                {!isCardDataLoading && hoveredData && createPortal(
                    <div
                        className={`stock-card-performance-tooltip${tooltipActive ? ' active' : ''}`}
                        style={tooltipStyles}
                    >
                        <div className="tooltip-row">
                            <div className="tooltip-metric">
                                <span className="tooltip-metric-label">Önc.Kap.:</span>
                                <strong className="tooltip-metric-value">{formatTooltipMetricValue(hoveredData.base)}</strong>
                            </div>
                            <div className="tooltip-metric">
                                <span className="tooltip-metric-label">Düşük:</span>
                                <strong className="tooltip-metric-value">{formatTooltipMetricValue(hoveredData.low)}</strong>
                            </div>
                            <div className="tooltip-metric">
                                <span className="tooltip-metric-label">Yüksek:</span>
                                <strong className="tooltip-metric-value">{formatTooltipMetricValue(hoveredData.high)}</strong>
                            </div>
                        </div>
                    </div>,
                    document.body
                )}
                {isCardDataLoading
                    ? STOCK_CARD_PERFORMANCE_KEYS.map(({ key }) => (
                          <span key={key} className="stock-card-performance-chip stock-card-performance-chip-skeleton" aria-hidden="true">
                              <span className="stock-card-inline-skeleton-chip-line stock-card-inline-skeleton-pulse" />
                          </span>
                      ))
                    : STOCK_CARD_PERFORMANCE_KEYS.map(({ key, label }) => {
                          const value = item[key];
                          const period = key.replace('return_', '').replace('_pct', '');
                          const base = (item as any)[`base_${period}`];
                          const high = (item as any)[`high_${period}`];
                          const low = (item as any)[`low_${period}`];

                          return (
                              <span
                                  key={key}
                                  className={`stock-card-performance-chip ${getTableChangeClass(value)}`}
                                  onMouseEnter={(event) => {
                                      const chipRect = event.currentTarget.getBoundingClientRect();
                                      openPerformanceTooltip({
                                          base,
                                          high,
                                          low,
                                          anchorX: chipRect.left + chipRect.width / 2,
                                      });
                                  }}
                                  onMouseLeave={closePerformanceTooltip}
                              >
                                  <strong>{label}</strong>
                                  {formatCardPct(value)}
                              </span>
                          );
                      })}
            </div>

            <div className="stock-card-foot">
                {isCardDataLoading ? (
                    <span className="stock-card-inline-skeleton-time stock-card-inline-skeleton-pulse" aria-hidden="true" />
                ) : (
                    <span>{footerLabel} {sessionTime}</span>
                )}
            </div>
        </article>
    );
}

const TREEMAP_LAYOUT_WIDTH = 100;
const TREEMAP_LAYOUT_HEIGHT = 62;

type HeatmapTile = {
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

type HeatmapLayoutItem = {
    row: MarketIndexConstituent;
    area: number;
    impactPct: number | null;
    changePct: number | null;
};

type HeatmapLayoutRect = {
    x: number;
    y: number;
    width: number;
    height: number;
};

function treemapValue(row: MarketIndexConstituent): number {
    if (row.weight_pct != null && row.weight_pct > 0) return row.weight_pct;
    const impact = Math.abs(row.point_effect ?? 0);
    return impact > 0 ? impact : 0.01;
}

function worstTreemapAspect(row: HeatmapLayoutItem[], shortSide: number): number {
    if (!row.length || shortSide <= 0) return Number.POSITIVE_INFINITY;
    const areas = row.map((item) => item.area);
    const sum = areas.reduce((total, area) => total + area, 0);
    const min = Math.min(...areas);
    const max = Math.max(...areas);
    if (sum <= 0 || min <= 0) return Number.POSITIVE_INFINITY;
    const sideSquared = shortSide * shortSide;
    return Math.max((sideSquared * max) / (sum * sum), (sum * sum) / (sideSquared * min));
}

function heatmapTileFromRect(item: HeatmapLayoutItem, rect: HeatmapLayoutRect): HeatmapTile {
    const width = Math.max(0, rect.width);
    const height = Math.max(0, rect.height);
    return {
        row: item.row,
        impactPct: item.impactPct,
        changePct: item.changePct,
        pointEffect: item.row.point_effect,
        left: (rect.x / TREEMAP_LAYOUT_WIDTH) * 100,
        top: (rect.y / TREEMAP_LAYOUT_HEIGHT) * 100,
        width: (width / TREEMAP_LAYOUT_WIDTH) * 100,
        height: (height / TREEMAP_LAYOUT_HEIGHT) * 100,
        areaPct: (width * height) / (TREEMAP_LAYOUT_WIDTH * TREEMAP_LAYOUT_HEIGHT) * 100,
    };
}

function layoutTreemapRow(
    row: HeatmapLayoutItem[],
    remaining: HeatmapLayoutRect,
    tiles: HeatmapTile[],
): void {
    if (!row.length || remaining.width <= 0 || remaining.height <= 0) return;
    const rowArea = row.reduce((total, item) => total + item.area, 0);
    if (rowArea <= 0) return;

    if (remaining.width >= remaining.height) {
        const columnWidth = Math.min(remaining.width, rowArea / remaining.height);
        let y = remaining.y;
        row.forEach((item, index) => {
            const height = index === row.length - 1 ? remaining.y + remaining.height - y : item.area / columnWidth;
            tiles.push(heatmapTileFromRect(item, {
                x: remaining.x,
                y,
                width: columnWidth,
                height: Math.max(0, height),
            }));
            y += height;
        });
        remaining.x += columnWidth;
        remaining.width = Math.max(0, remaining.width - columnWidth);
        return;
    }

    const rowHeight = Math.min(remaining.height, rowArea / remaining.width);
    let x = remaining.x;
    row.forEach((item, index) => {
        const width = index === row.length - 1 ? remaining.x + remaining.width - x : item.area / rowHeight;
        tiles.push(heatmapTileFromRect(item, {
            x,
            y: remaining.y,
            width: Math.max(0, width),
            height: rowHeight,
        }));
        x += width;
    });
    remaining.y += rowHeight;
    remaining.height = Math.max(0, remaining.height - rowHeight);
}

function buildHeatmapTiles(items: MarketIndexConstituent[], indexLevel: number | null): HeatmapTile[] {
    const sorted = [...items].sort((a, b) => {
        const result = treemapValue(b) - treemapValue(a);
        return result === 0 ? a.symbol.localeCompare(b.symbol, 'tr') : result;
    });
    const totalValue = sorted.reduce((total, row) => total + treemapValue(row), 0);
    if (totalValue <= 0) return [];

    const totalArea = TREEMAP_LAYOUT_WIDTH * TREEMAP_LAYOUT_HEIGHT;
    const layoutItems: HeatmapLayoutItem[] = sorted.map((row) => {
        const value = treemapValue(row);
        return {
            row,
            area: (value / totalValue) * totalArea,
            impactPct: getImpactPct(row, indexLevel),
            changePct: numericOrNull(row.change_pct),
        };
    });
    const remaining: HeatmapLayoutRect = {
        x: 0,
        y: 0,
        width: TREEMAP_LAYOUT_WIDTH,
        height: TREEMAP_LAYOUT_HEIGHT,
    };
    const tiles: HeatmapTile[] = [];
    let row: HeatmapLayoutItem[] = [];

    layoutItems.forEach((item) => {
        const nextRow = [...row, item];
        const shortSide = Math.min(remaining.width, remaining.height);
        if (!row.length || worstTreemapAspect(nextRow, shortSide) <= worstTreemapAspect(row, shortSide)) {
            row = nextRow;
            return;
        }
        layoutTreemapRow(row, remaining, tiles);
        row = [item];
    });
    layoutTreemapRow(row, remaining, tiles);
    return tiles;
}

function formatTreemapScalePct(value: number): string {
    const rounded = Math.abs(value) >= 1 ? Math.round(value) : Number(value.toFixed(2));
    if (rounded === 0) return '0%';
    return `${rounded.toLocaleString('tr-TR', {
        minimumFractionDigits: Math.abs(rounded) >= 1 ? 0 : 2,
        maximumFractionDigits: Math.abs(rounded) >= 1 ? 0 : 2,
    })}%`;
}

function getTreemapTileColor(changePct: number | null, scaledMove: number): string {
    if (changePct == null || changePct === 0) return 'hsl(146 10% 28%)';
    if (changePct > 0) {
        return `hsl(123 92% ${23 + scaledMove * 24}%)`;
    }
    return `hsl(0 40% ${25 + scaledMove * 18}%)`;
}

interface MarketsViewProps {
    routeSection?: MarketSection;
    routeStockIndex?: MarketStockIndex;
    routeSelectedIndex?: MarketIndexCode | null;
    routeReturnMode?: StockReturnMode;
    onNavigateSection?: (section: MarketSection) => void;
    onNavigateFundSection?: (section: MarketsNavigationFundSection) => void;
    onNavigateStockIndex?: (index: MarketStockIndex) => void;
    onNavigateReturnMode?: (mode: StockReturnMode) => void;
    onNavigateIndexDetail?: (index: MarketIndexCode | null) => void;
    onOpenTicker?: (ticker: string) => void;
    onOpenFund?: (fundCode: string) => void;
}

export default function MarketsView({
    routeSection = 'stocks',
    routeStockIndex = 'XUTUM',
    routeSelectedIndex = null,
    routeReturnMode = DEFAULT_STOCK_RETURN_MODE,
    onNavigateSection,
    onNavigateFundSection,
    onNavigateStockIndex,
    onNavigateReturnMode,
    onNavigateIndexDetail,
    onOpenTicker,
    onOpenFund,
}: MarketsViewProps) {
    const [market, setMarket] = useState<MarketUniverseResponse | null>(() => marketUniverseMemoryCache?.data || null);
    const [stocks, setStocks] = useState<MarketStocksResponse | null>(null);
    const [indices, setIndices] = useState<MarketIndicesResponse | null>(null);
    const [indexDetail, setIndexDetail] = useState<MarketIndexDetailResponse | null>(null);
    const [loading, setLoading] = useState(() => !marketUniverseMemoryCache);
    const [error, setError] = useState<string | null>(null);
    const [stocksLoading, setStocksLoading] = useState(false);
    const [stocksError, setStocksError] = useState<string | null>(null);
    const [indicesLoading, setIndicesLoading] = useState(false);
    const [indicesError, setIndicesError] = useState<string | null>(null);
    const [indexDetailLoading, setIndexDetailLoading] = useState(false);
    const [indexDetailError, setIndexDetailError] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [indexGroup, setIndexGroup] = useState<IndexGroupFilter>('all');
    const [stockCardSearchTerm, setStockCardSearchTerm] = useState('');
    const [navCollapsed, setNavCollapsed] = useState(false);
    const [railCollapsed, setRailCollapsed] = useState(false);
    const collapsedMenuCount = (navCollapsed ? 1 : 0) + (railCollapsed ? 1 : 0);
    const [activeSection, setActiveSection] = useState<MarketSection>(routeSection);
    const [isMobileViewport, setIsMobileViewport] = useState(() => (
        typeof window !== 'undefined' && window.matchMedia('(max-width: 760px)').matches
    ));
    const [mobileMarketPanelOpen, setMobileMarketPanelOpen] = useState(false);
    const [mobileWatchlistSearchOpen, setMobileWatchlistSearchOpen] = useState(false);
    const [mobileIndexDetail, setMobileIndexDetail] = useState<MarketIndexDetailResponse | null>(null);
    const [mobileWatchlistFundRows, setMobileWatchlistFundRows] = useState<Record<string, FundSummary | null>>({});
    const [selectedIndex, setSelectedIndex] = useState<MarketIndexCode | null>(routeSelectedIndex);
    const [stockIndex, setStockIndex] = useState<MarketStockIndex>(routeStockIndex);
    const [returnMode, setReturnMode] = useState<StockReturnMode>(routeReturnMode);
    const [stockTableMode, setStockTableMode] = useState<StockTableMode>('performance');
    const [screenerFilters, setScreenerFilters] = useState<StockScreenerFilterState>({ ...EMPTY_STOCK_SCREENER_FILTERS });
    const [activeScreenerPreset, setActiveScreenerPreset] = useState<StockScreenerPresetId | null>(null);
    const [screenerModalOpen, setScreenerModalOpen] = useState(false);
    const [screenerCategory, setScreenerCategory] = useState<StockScreenerCategory>('all');

    const { activeFilterCount, screenerCategoryCounts, screenerActiveChips } = useMemo(() => {
        const counts: Record<StockScreenerCategory, number> = {
            all: 0,
            profile: 0,
            valuation: 0,
            financials: 0,
            market: 0,
            historical: 0,
        };
        const chips: Array<{ id: string; label: string; onClear: () => void }> = [];

        if (screenerFilters.sector) {
            counts.profile += 1;
            chips.push({
                id: 'sector',
                label: `Sektör: ${screenerFilters.sector}`,
                onClear: () => {
                    setScreenerFilters((c) => ({ ...c, sector: '' }));
                    setActiveScreenerPreset(null);
                },
            });
        }

        for (const toggle of SCREENER_TOGGLE_SPECS) {
            if (screenerFilters[toggle.key]) {
                counts[toggle.category] += 1;
                chips.push({
                    id: toggle.key,
                    label: toggle.label,
                    onClear: () => {
                        setScreenerFilters((c) => ({ ...c, [toggle.key]: false }));
                        setActiveScreenerPreset(null);
                    },
                });
            }
        }

        for (const spec of SCREENER_RANGE_SPECS) {
            const minRaw = screenerFilters[spec.minKey].trim();
            const maxRaw = screenerFilters[spec.maxKey].trim();
            const hasMin = minRaw !== '';
            const hasMax = maxRaw !== '';
            if (hasMin) counts[spec.category] += 1;
            if (hasMax) counts[spec.category] += 1;
            if (hasMin || hasMax) {
                const boundText = hasMin && hasMax
                    ? `${minRaw} – ${maxRaw} ${spec.unit}`
                    : hasMin
                        ? `≥ ${minRaw} ${spec.unit}`
                        : `≤ ${maxRaw} ${spec.unit}`;
                chips.push({
                    id: spec.field,
                    label: `${spec.label}: ${boundText}`,
                    onClear: () => {
                        setScreenerFilters((c) => ({ ...c, [spec.minKey]: '', [spec.maxKey]: '' }));
                        setActiveScreenerPreset(null);
                    },
                });
            }
        }

        counts.all = counts.profile + counts.valuation + counts.financials + counts.market + counts.historical;
        return {
            activeFilterCount: counts.all,
            screenerCategoryCounts: counts,
            screenerActiveChips: chips,
        };
    }, [screenerFilters]);

    const isScreenerActive = activeFilterCount > 0;

    useEffect(() => {
        if (!screenerModalOpen) return;
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') setScreenerModalOpen(false);
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [screenerModalOpen]);
    const [stockCardSymbols, setStockCardSymbols] = useState<string[]>(readStoredStockCards);
    const [stockCards, setStockCards] = useState<MarketStockCardsResponse | null>(() => {
        const symbols = readStoredStockCards();
        return marketStockCardsMemoryCache.get(stockCardsMemoryKey(symbols))?.data || null;
    });
    const [stockCardsLoading, setStockCardsLoading] = useState(false);
    const [stockCardsError, setStockCardsError] = useState<string | null>(null);
    const [stockCardPendingSymbols, setStockCardPendingSymbols] = useState<string[]>(() => [...readStoredStockCards()]);
    const [stockCardPickerOpen, setStockCardPickerOpen] = useState(false);
    const [stockSort, setStockSort] = useState<{ key: StockSortKey; direction: SortDirection }>({
        key: 'company',
        direction: 'asc',
    });
    const [indexSort, setIndexSort] = useState<{ key: IndexSortKey; direction: SortDirection }>({
        key: 'symbol',
        direction: 'asc',
    });
    const [indexConstituentSort, setIndexConstituentSort] = useState<{
        key: IndexConstituentSortKey;
        direction: SortDirection;
    }>({
        key: 'impact_abs',
        direction: 'desc',
    });
    const stocksInFlightRef = useRef(false);
    const indicesInFlightRef = useRef(false);
    const indexDetailInFlightRef = useRef(false);
    const stockCardQuickInFlightRef = useRef<Promise<void> | null>(null);
    const stockCardFullInFlightRef = useRef<Promise<void> | null>(null);
    const stockCardsRef = useRef<MarketStockCardsResponse | null>(stockCards);
    const marketPageRef = useRef<HTMLDivElement | null>(null);
    const pendingMarketScrollResetRef = useRef(true);
    const marketScrollResetFrameRef = useRef<number | null>(null);
    const latestStockIndexRef = useRef<MarketStockIndex>(stockIndex);
    const latestSelectedIndexRef = useRef<MarketIndexCode | null>(selectedIndex);
    const latestStockCardSymbolsRef = useRef(stockCardSymbols.join(','));
    const previousStockCardSymbolsRef = useRef<string[]>(stockCardSymbols);
    const draggingStockCardSymbolRef = useRef<string | null>(null);
    const [draggingStockCardSymbol, setDraggingStockCardSymbol] = useState<string | null>(null);
    const stockCardSymbolsKey = stockCardSymbols.join(',');
    const marketListRefreshMs = isMobileViewport
        ? MARKET_LIST_REFRESH_MOBILE_MS
        : MARKET_LIST_REFRESH_DESKTOP_MS;
    const marketDetailRefreshMs = isMobileViewport
        ? MARKET_DETAIL_REFRESH_MOBILE_MS
        : MARKET_DETAIL_REFRESH_DESKTOP_MS;
    const watchlist = useWatchlist();
    const mobileWatchlistFundCodes = useMemo(
        () => watchlist.items
            .filter((item) => item.kind === 'fund')
            .map((item) => normalizeWatchlistSymbol(item.symbol)),
        [watchlist.items],
    );

    useEffect(() => {
        stockCardsRef.current = stockCards;
    }, [stockCards]);

    useEffect(() => {
        const mediaQuery = window.matchMedia('(max-width: 760px)');
        const updateViewport = () => setIsMobileViewport(mediaQuery.matches);
        updateViewport();
        mediaQuery.addEventListener('change', updateViewport);
        return () => mediaQuery.removeEventListener('change', updateViewport);
    }, []);

    useEffect(() => {
        if (activeSection !== 'markets' || !isMobileViewport) return;
        let alive = true;
        const loadMobileIndex = () => {
            apiClient
                .marketIndexDetail('XU100')
                .then((payload) => {
                    if (alive) setMobileIndexDetail(payload);
                })
                .catch(() => {
                    if (alive) setMobileIndexDetail(null);
                });
        };
        loadMobileIndex();
        const intervalId = window.setInterval(() => {
            if (document.visibilityState === 'visible' && isBistTradingHours()) {
                loadMobileIndex();
            }
        }, marketDetailRefreshMs);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && isBistTradingHours()) loadMobileIndex();
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            alive = false;
            window.clearInterval(intervalId);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [activeSection, isMobileViewport, marketDetailRefreshMs]);

    useEffect(() => {
        if (activeSection !== 'markets' || !isMobileViewport) return;
        const missingCodes = mobileWatchlistFundCodes.filter(
            (code) => !(code in mobileWatchlistFundRows),
        );
        if (missingCodes.length === 0) return;

        let cancelled = false;
        Promise.all(
            missingCodes.map((code) => apiClient
                .fundDetail(code)
                .then((detail) => [code, detail] as const)
                .catch(() => [code, null] as const)),
        ).then((entries) => {
            if (cancelled) return;
            setMobileWatchlistFundRows((current) => {
                const next = { ...current };
                for (const [code, detail] of entries) next[code] = detail;
                return next;
            });
        });

        return () => {
            cancelled = true;
        };
    }, [
        activeSection,
        isMobileViewport,
        mobileWatchlistFundCodes,
        mobileWatchlistFundRows,
    ]);

    useEffect(() => {
        setActiveSection(routeSection);
        if (routeSection !== 'markets') {
            setMobileMarketPanelOpen(false);
        }
    }, [routeSection]);

    useEffect(() => {
        setStockIndex(routeStockIndex);
    }, [routeStockIndex]);

    useEffect(() => {
        setSelectedIndex(routeSelectedIndex);
    }, [routeSelectedIndex]);

    useEffect(() => {
        setReturnMode(routeReturnMode);
    }, [routeReturnMode]);

    useEffect(() => {
        if (activeSection === 'markets') {
            pendingMarketScrollResetRef.current = true;
        }
    }, [activeSection]);

    useLayoutEffect(() => {
        if (activeSection !== 'markets' || loading || error || !market) return;
        if (!pendingMarketScrollResetRef.current) return;
        if (stockCardSymbols.length > 0 && !stockCards && !stockCardsError) return;

        const resetScroll = () => {
            marketPageRef.current?.scrollTo({ top: 0, left: 0, behavior: 'auto' });
            window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
            document.documentElement.scrollTop = 0;
            document.body.scrollTop = 0;
        };

        resetScroll();
        marketScrollResetFrameRef.current = window.requestAnimationFrame(() => {
            resetScroll();
            marketScrollResetFrameRef.current = window.requestAnimationFrame(resetScroll);
        });
        pendingMarketScrollResetRef.current = false;

        return () => {
            if (marketScrollResetFrameRef.current != null) {
                window.cancelAnimationFrame(marketScrollResetFrameRef.current);
                marketScrollResetFrameRef.current = null;
            }
        };
    }, [activeSection, loading, error, market, stockCardSymbols.length, stockCards, stockCardsError]);

    useEffect(() => {
        latestStockIndexRef.current = stockIndex;
    }, [stockIndex]);

    useEffect(() => {
        latestSelectedIndexRef.current = selectedIndex;
    }, [selectedIndex]);

    useEffect(() => {
        latestStockCardSymbolsRef.current = stockCardSymbolsKey;
        try {
            window.localStorage.setItem(STOCK_CARD_STORAGE_KEY, JSON.stringify(stockCardSymbols));
        } catch {
            // localStorage may be unavailable in private or restricted contexts.
        }
    }, [stockCardSymbols, stockCardSymbolsKey]);

    useEffect(() => {
        const previousSymbols = previousStockCardSymbolsRef.current;
        const previousSet = new Set(previousSymbols);
        const selectedSet = new Set(stockCardSymbols);
        const addedSymbols = stockCardSymbols.filter((symbol) => !previousSet.has(symbol));

        setStockCardPendingSymbols((previous) => {
            if (stockCardSymbols.length === 0) return [];
            const next = previous.filter((symbol) => selectedSet.has(symbol));
            const existing = new Set(next);
            for (const symbol of addedSymbols) {
                if (!existing.has(symbol)) {
                    next.push(symbol);
                }
            }
            return next;
        });

        previousStockCardSymbolsRef.current = stockCardSymbols;
    }, [stockCardSymbols, stockCardSymbolsKey]);

    useEffect(() => {
        if (activeSection !== 'markets') return;
        loadStats();
        const intervalId = window.setInterval(() => {
            if (document.visibilityState === 'visible') {
                loadStats(true);
            }
        }, 5 * 60 * 1000);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible') loadStats(true);
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            window.clearInterval(intervalId);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [activeSection]);

    useEffect(() => {
        return apiClient.subscribeMarketStocks((index, payload) => {
            if (activeSection === 'stocks' && index === latestStockIndexRef.current) {
                setStocks(payload);
                setStocksError(null);
            }
        });
    }, [activeSection]);

    useEffect(() => {
        if (activeSection !== 'stocks') return;
        setStocks(null);
        setStocksError(null);
        loadStocks(false, false, stockIndex);
        const releaseStream = apiClient.retainMarketStocksStream(stockIndex);
        const intervalId = window.setInterval(() => {
            if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
            if (apiClient.isMarketStocksStreamActive(stockIndex)) return;
            loadStocks(true, false, stockIndex);
        }, marketListRefreshMs);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && isBistTradingHours()) {
                loadStocks(true, false, stockIndex);
            }
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            releaseStream();
            window.clearInterval(intervalId);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [activeSection, marketListRefreshMs, stockIndex]);

    useEffect(() => {
        if (activeSection !== 'markets' || isMobileViewport) return;
        if (stockCardSymbols.length === 0) {
            stockCardsRef.current = null;
            setStockCards(null);
            setStockCardsError(null);
            setStockCardsLoading(false);
            setStockCardPendingSymbols([]);
            return;
        }
        const cacheKey = stockCardsMemoryKey(stockCardSymbols);
        const cached = marketStockCardsMemoryCache.get(cacheKey);
        if (cached) {
            stockCardsRef.current = cached.data;
            setStockCards(cached.data);
            setStockCardPendingSymbols((previous) => previous.filter(
                (symbol) => !(cached.data.items || []).some(
                    (item) => item.symbol === symbol && item.card_stage !== 'quick',
                ),
            ));
        } else if (stockCardsRef.current) {
            // Adding or reordering a card must not blank cards that are
            // already visible. Keep the current payload and request only the
            // symbols that are not present in it.
            const retained = retainMarketStockCardsForSymbols(stockCardsRef.current, stockCardSymbols);
            stockCardsRef.current = retained;
            setStockCards(retained);
        } else {
            setStockCards(null);
        }

        const controller = new AbortController();
        let delayTimer: number | null = null;
        let detailBatchesLoading = false;
        let quoteBatchesLoading = false;
        const initialSymbols = getMarketStockCardSymbolsNeedingLoad(stockCardSymbols, stockCardsRef.current);
        const buildBatches = (symbols: string[]) => {
            const initialBatch = symbols.slice(0, STOCK_CARD_INITIAL_BATCH_SIZE);
            return initialBatch.length > 0
                ? [initialBatch, ...splitStockCardBatches(symbols.slice(STOCK_CARD_INITIAL_BATCH_SIZE), STOCK_CARD_BACKGROUND_BATCH_SIZE)]
                : [];
        };
        const loadBatches = async (symbols: string[], includeFull = true) => {
            if (includeFull ? detailBatchesLoading : quoteBatchesLoading) return;
            const batches = includeFull
                ? buildBatches(symbols)
                : symbols.length > 0 ? [symbols] : [];
            if (batches.length === 0) return;
            if (includeFull) detailBatchesLoading = true;
            else quoteBatchesLoading = true;
            try {
                for (const [index, batch] of batches.entries()) {
                    if (controller.signal.aborted) return;
                    if (includeFull && index > 0) {
                        await new Promise<void>((resolve) => {
                            delayTimer = window.setTimeout(resolve, STOCK_CARD_BACKGROUND_DELAY_MS);
                        });
                        if (controller.signal.aborted) return;
                    }
                    await loadStockCards(includeFull && (index > 0 || Boolean(cached)), false, batch, controller.signal, 'quick');
                    if (includeFull) {
                        await loadStockCards(true, false, batch, controller.signal, 'full');
                    }
                }
            } finally {
                if (includeFull) detailBatchesLoading = false;
                else quoteBatchesLoading = false;
            }
        };

        // When the selection changes, load only new or incomplete cards. The
        // periodic refresh below still revalidates the complete selection.
        void loadBatches(initialSymbols);
        const intervalId = window.setInterval(() => {
            if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
            void loadBatches(stockCardSymbols, true);
        }, marketDetailRefreshMs);
        const quoteIntervalId = window.setInterval(() => {
            if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
            void loadBatches(stockCardSymbols, false);
        }, MARKET_QUOTE_REFRESH_MS);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && isBistTradingHours()) {
                void loadBatches(stockCardSymbols, true);
            }
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            controller.abort();
            if (delayTimer !== null) window.clearTimeout(delayTimer);
            window.clearInterval(intervalId);
            window.clearInterval(quoteIntervalId);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [activeSection, isMobileViewport, marketDetailRefreshMs, stockCardSymbols, stockCardSymbolsKey]);

    useEffect(() => {
        if (activeSection !== 'indices') return;
        loadIndices(false, false);
        const intervalId = window.setInterval(() => {
            if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
            loadIndices(true, false);
        }, marketListRefreshMs);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && isBistTradingHours()) loadIndices(true, false);
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            window.clearInterval(intervalId);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [activeSection, marketListRefreshMs]);

    useEffect(() => {
        if (activeSection !== 'indices' || !selectedIndex) return;
        setIndexDetail(null);
        setIndexDetailError(null);
        loadIndexDetail(false, false, selectedIndex);
        const intervalId = window.setInterval(() => {
            if (document.visibilityState !== 'visible' || !isBistTradingHours()) return;
            loadIndexDetail(true, false, selectedIndex);
        }, marketDetailRefreshMs);
        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible' && isBistTradingHours()) loadIndexDetail(true, false, selectedIndex);
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            window.clearInterval(intervalId);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [activeSection, marketDetailRefreshMs, selectedIndex]);

    async function loadStats(silent = false) {
        const cached = marketUniverseMemoryCache?.data;
        if (cached) setMarket(cached);
        if (!silent) setLoading(!cached);
        if (!silent) setError(null);
        try {
            const marketPayload = await apiClient.marketUniverse();
            marketUniverseMemoryCache = { data: marketPayload, fetchedAt: Date.now() };
            setMarket(marketPayload);
        } catch (err: any) {
            if (!silent) setError(err.message || 'Veriler yüklenemedi.');
        } finally {
            if (!silent) setLoading(false);
        }
    }

    async function loadStocks(silent = false, refresh = false, requestedIndex: MarketStockIndex = stockIndex) {
        if (stocksInFlightRef.current) return;
        stocksInFlightRef.current = true;
        const cachedStocks = apiClient.getCachedMarketStocks(requestedIndex);
        if (cachedStocks) {
            setStocks(cachedStocks);
        } else if (!silent) {
            void apiClient
                .marketUniverse({ index: requestedIndex })
                .then((universePayload) => {
                    if (latestStockIndexRef.current !== requestedIndex) return;
                    setStocks((current) => (
                        current?.index === requestedIndex && !current.partial
                            ? current
                            : quickStocksFromUniverse(requestedIndex, universePayload)
                    ));
                })
                .catch(() => {
                    // The detailed stock request remains the source of truth.
                });
        }
        if (!silent) setStocksLoading(true);
        if (!silent) setStocksError(null);
        try {
            const stocksPayload = await apiClient.marketStocks({ index: requestedIndex, refresh });
            if (latestStockIndexRef.current !== requestedIndex) return;
            setStocks(stocksPayload);
            setStocksError(null);
        } catch (err: any) {
            if (!silent || !stocks) {
                setStocksError(err.message || 'Hisse verileri yüklenemedi.');
            }
        } finally {
            stocksInFlightRef.current = false;
            if (!silent) setStocksLoading(false);
        }
    }

    async function loadStockCards(
        silent = false,
        refresh = false,
        requestedSymbols: string[] = stockCardSymbols,
        signal?: AbortSignal,
        tier: 'quick' | 'full' = 'full',
    ) {
        if (requestedSymbols.length === 0) return;
        const inFlightRef = tier === 'quick' ? stockCardQuickInFlightRef : stockCardFullInFlightRef;
        const previousRequest = inFlightRef.current;
        const request = (async () => {
            if (previousRequest) await previousRequest;
            if (signal?.aborted) return;

            if (!silent) setStockCardsLoading(true);
            if (!silent) setStockCardsError(null);
            try {
                const payload = await apiClient.marketStockCards({ symbols: requestedSymbols, refresh, tier, signal });
                if (signal?.aborted) return;
                const latestSymbols = latestStockCardSymbolsRef.current.split(',').filter(Boolean);
                const latestSymbolSet = new Set(latestSymbols);
                if (!requestedSymbols.every((symbol) => latestSymbolSet.has(symbol))) return;
                const mergedPayload = mergeStockCardPayloads(stockCardsRef.current, payload, latestSymbols);
                stockCardsRef.current = mergedPayload;
                marketStockCardsMemoryCache.set(latestSymbols.join(','), { data: mergedPayload, fetchedAt: Date.now() });
                setStockCards(mergedPayload);
                setStockCardsError(null);
                const readySymbols = new Set(
                    (mergedPayload.items ?? [])
                        .filter((item) => item.card_stage !== 'quick' && hasStockCardLoadedData(item))
                        .map((item) => item.symbol),
                );
                if (readySymbols.size > 0) {
                    setStockCardPendingSymbols((previous) => previous.filter((symbol) => !readySymbols.has(symbol)));
                }
            } catch (err: any) {
                if (err?.name === 'AbortError' || signal?.aborted) return;
                if (!silent || !stockCardsRef.current) {
                    setStockCardsError(err.message || 'Hisse kartları şu an hazırlanıyor.');
                }
            } finally {
                if (!silent) setStockCardsLoading(false);
            }
        })();

        inFlightRef.current = request;
        try {
            await request;
        } finally {
            if (inFlightRef.current === request) {
                inFlightRef.current = null;
            }
        }
    }

    async function loadIndices(silent = false, refresh = false) {
        if (indicesInFlightRef.current) return;
        indicesInFlightRef.current = true;
        if (!silent) setIndicesLoading(true);
        if (!silent) setIndicesError(null);
        try {
            const payload = await apiClient.marketIndices({ refresh });
            setIndices(payload);
            setIndicesError(null);
        } catch (err: any) {
            if (!silent || !indices) {
                setIndicesError(err.message || 'Endeks verileri yüklenemedi.');
            }
        } finally {
            indicesInFlightRef.current = false;
            if (!silent) setIndicesLoading(false);
        }
    }

    async function loadIndexDetail(
        silent = false,
        refresh = false,
        requestedIndex: MarketIndexCode = selectedIndex || 'XUTUM',
    ) {
        if (indexDetailInFlightRef.current) return;
        indexDetailInFlightRef.current = true;
        if (!silent) setIndexDetailLoading(true);
        if (!silent) setIndexDetailError(null);
        try {
            const payload = await apiClient.marketIndexDetail(requestedIndex, { refresh });
            if (latestSelectedIndexRef.current !== requestedIndex) return;
            setIndexDetail(payload);
            setIndexDetailError(null);
        } catch (err: any) {
            if (!silent || !indexDetail) {
                setIndexDetailError(err.message || 'Endeks detayı yüklenemedi.');
            }
        } finally {
            indexDetailInFlightRef.current = false;
            if (!silent) setIndexDetailLoading(false);
        }
    }

    const normalizedSearch = searchTerm.trim().toLowerCase();
    const normalizedStockCardSearch = stockCardSearchTerm.trim().toLowerCase();
    const stockCardCandidates = useMemo(
        () =>
            (market?.rows || [])
                .filter((row) => !stockCardSymbols.includes(row.company))
                .filter((row) => {
                    if (!normalizedStockCardSearch) return true;
                    return row.company.toLowerCase().includes(normalizedStockCardSearch)
                        || String(row.name || '').toLowerCase().includes(normalizedStockCardSearch);
                })
                .slice(0, 18),
        [market?.rows, normalizedStockCardSearch, stockCardSymbols],
    );

    const screenerData = useMemo(() => {
        const allRows = stocks?.rows || [];
        const sectorCounts = new Map<string, number>();
        const sectorBuckets = new Map<string, { fk: number[]; pd_dd: number[]; fd_favok: number[]; count: number }>();
        let financialCount = 0;
        for (const row of allRows) {
            const sectorName = String(row.sector || '').trim();
            if (sectorName) {
                sectorCounts.set(sectorName, (sectorCounts.get(sectorName) || 0) + 1);
                let bucket = sectorBuckets.get(sectorName);
                if (!bucket) {
                    bucket = { fk: [], pd_dd: [], fd_favok: [], count: 0 };
                    sectorBuckets.set(sectorName, bucket);
                }
                bucket.count += 1;
                if (row.fk != null && Number.isFinite(row.fk) && row.fk > 0) bucket.fk.push(row.fk);
                if (row.pd_dd != null && Number.isFinite(row.pd_dd) && row.pd_dd > 0) bucket.pd_dd.push(row.pd_dd);
                if (row.fd_favok != null && Number.isFinite(row.fd_favok) && row.fd_favok > 0) bucket.fd_favok.push(row.fd_favok);
            }
            if (hasStockFinancialData(row)) {
                financialCount += 1;
            }
        }
        const avgOf = (arr: number[]): number | null => (
            arr.length >= 2 ? Math.round((arr.reduce((sum, v) => sum + v, 0) / arr.length) * 100) / 100 : null
        );
        const pctDiff = (val: number | null | undefined, benchmark: number | null): number | null => {
            if (val == null || !Number.isFinite(val) || val <= 0 || benchmark == null || benchmark <= 0) return null;
            return Math.round(((val - benchmark) / benchmark) * 10000) / 100;
        };
        const sectorAverages = new Map<string, { fk: number | null; pd_dd: number | null; fd_favok: number | null; count: number }>();
        for (const [sectorName, bucket] of sectorBuckets.entries()) {
            sectorAverages.set(sectorName, {
                fk: avgOf(bucket.fk),
                pd_dd: avgOf(bucket.pd_dd),
                fd_favok: avgOf(bucket.fd_favok),
                count: bucket.count,
            });
        }
        const sectorDiffByCompany = new Map<string, StockSectorDiffFields>();
        for (const row of allRows) {
            const sectorName = String(row.sector || '').trim();
            const sectorAvg = sectorName ? sectorAverages.get(sectorName) : undefined;
            sectorDiffByCompany.set(row.company, {
                fk_sector_diff_pct: pctDiff(row.fk, sectorAvg?.fk ?? null),
                pd_dd_sector_diff_pct: pctDiff(row.pd_dd, sectorAvg?.pd_dd ?? null),
                fd_favok_sector_diff_pct: pctDiff(row.fd_favok, sectorAvg?.fd_favok ?? null),
            });
        }
        const sectors = [...sectorCounts.keys()].sort((a, b) => a.localeCompare(b, 'tr'));

        const sectorFilter = screenerFilters.sector.trim().toLocaleLowerCase('tr-TR');
        const rows = allRows.filter((row) => {
            if (sectorFilter && String(row.sector || '').trim().toLocaleLowerCase('tr-TR') !== sectorFilter) return false;
            const sectorDiff = sectorDiffByCompany.get(row.company);
            for (const spec of SCREENER_RANGE_SPECS) {
                const scale = spec.scale || 1;
                const minimumInput = parseScreenerNumber(screenerFilters[spec.minKey]);
                const maximumInput = parseScreenerNumber(screenerFilters[spec.maxKey]);
                if (minimumInput === undefined && maximumInput === undefined) continue;
                let value: number | null | undefined;
                if (
                    spec.field === 'fk_sector_diff_pct'
                    || spec.field === 'pd_dd_sector_diff_pct'
                    || spec.field === 'fd_favok_sector_diff_pct'
                ) {
                    value = sectorDiff?.[spec.field];
                } else {
                    value = row[spec.field];
                }
                if (value == null || !Number.isFinite(value)) return false;
                if (minimumInput !== undefined && value < minimumInput * scale) return false;
                if (maximumInput !== undefined && value > maximumInput * scale) return false;
            }
            if (screenerFilters.positive_net_profit && !(row.ttm_net_kar != null && row.ttm_net_kar > 0)) return false;
            if (screenerFilters.positive_ebitda && !(row.ttm_favok != null && row.ttm_favok > 0)) return false;
            if (screenerFilters.positive_free_cash_flow && !(row.ttm_fcf != null && row.ttm_fcf > 0)) return false;
            if (screenerFilters.net_cash_only && !(row.net_borc != null && row.net_borc < 0)) return false;
            if (screenerFilters.positive_roe && !(row.roe_pct != null && row.roe_pct > 0)) return false;
            if (screenerFilters.only_with_financials && !hasStockFinancialData(row)) return false;
            if (screenerFilters.below_hist_fk && !(row.fk_hist_diff_pct != null && row.fk_hist_diff_pct < 0)) return false;
            if (screenerFilters.below_hist_pd_dd && !(row.pd_dd_hist_diff_pct != null && row.pd_dd_hist_diff_pct < 0)) return false;
            if (screenerFilters.above_sma_50 && !(row.sma_50_diff_pct != null && row.sma_50_diff_pct > 0)) return false;
            if (screenerFilters.above_sma_200 && !(row.sma_200_diff_pct != null && row.sma_200_diff_pct > 0)) return false;
            if (screenerFilters.below_sector_fk && !(sectorDiff?.fk_sector_diff_pct != null && sectorDiff.fk_sector_diff_pct < 0)) return false;
            if (screenerFilters.below_sector_pd_dd && !(sectorDiff?.pd_dd_sector_diff_pct != null && sectorDiff.pd_dd_sector_diff_pct < 0)) return false;
            return true;
        });
        return {
            rows,
            sectors,
            sectorCounts,
            sectorAverages,
            sectorDiffByCompany,
            totalCount: allRows.length,
            financialCount,
        };
    }, [stocks?.rows, screenerFilters]);

    const stockRowsForMode = useMemo(() => {
        if (isScreenerActive) return screenerData.rows;
        return stocks?.rows || [];
    }, [isScreenerActive, screenerData.rows, stocks?.rows]);

    const filteredStocks = useMemo(
        () =>
            stockRowsForMode.filter((row) =>
                row.company.toLowerCase().includes(normalizedSearch),
            ),
        [stockRowsForMode, normalizedSearch],
    );

    const filteredIndices = useMemo(
        () =>
            (indices?.rows || []).filter((row) => {
                if (indexGroup === 'main' && !MAIN_MARKET_INDICES.has(row.symbol)) return false;
                if (indexGroup === 'sector' && MAIN_MARKET_INDICES.has(row.symbol)) return false;
                const query = normalizedSearch;
                if (!query) return true;
                return (
                    row.symbol.toLowerCase().includes(query) ||
                    row.label.toLowerCase().includes(query)
                );
            }),
        [indices?.rows, indexGroup, normalizedSearch],
    );
    const isIndicesFiltered = indexGroup !== 'all' || Boolean(normalizedSearch);

    const sortedIndices = useMemo(() => {
        const arr = [...filteredIndices];
        arr.sort((a, b) => {
            const aPrimaryRank = PRIMARY_MARKET_INDEX_RANK.get(a.symbol);
            const bPrimaryRank = PRIMARY_MARKET_INDEX_RANK.get(b.symbol);
            if (aPrimaryRank !== undefined || bPrimaryRank !== undefined) {
                if (aPrimaryRank === undefined) return 1;
                if (bPrimaryRank === undefined) return -1;
                return aPrimaryRank - bPrimaryRank;
            }

            const av = a[indexSort.key];
            const bv = b[indexSort.key];
            const aMissing = av == null || av === '';
            const bMissing = bv == null || bv === '';

            if (aMissing && bMissing) return String(a.symbol).localeCompare(String(b.symbol), 'tr');
            if (aMissing) return 1;
            if (bMissing) return -1;

            let result = 0;
            if (typeof av === 'string' || typeof bv === 'string') {
                result = String(av).localeCompare(String(bv), 'tr');
            } else {
                result = Number(av) - Number(bv);
            }
            if (result === 0) result = String(a.symbol).localeCompare(String(b.symbol), 'tr');
            return indexSort.direction === 'asc' ? result : -result;
        });
        return arr;
    }, [filteredIndices, indexSort]);

    const sortedStocks = useMemo(() => {
        const arr = [...filteredStocks];
        arr.sort((a, b) => {
            const av = stockSortValue(
                a,
                stockSort.key,
                stocks?.benchmarks,
                returnMode,
                screenerData.sectorDiffByCompany.get(a.company),
            );
            const bv = stockSortValue(
                b,
                stockSort.key,
                stocks?.benchmarks,
                returnMode,
                screenerData.sectorDiffByCompany.get(b.company),
            );
            const aMissing = av == null || av === '';
            const bMissing = bv == null || bv === '';
            if (aMissing && bMissing) return a.company.localeCompare(b.company, 'tr');
            if (aMissing) return 1;
            if (bMissing) return -1;

            let result = 0;
            if (typeof av === 'string' || typeof bv === 'string') {
                result = String(av).localeCompare(String(bv), 'tr');
            } else {
                result = Number(av) - Number(bv);
            }
            if (result === 0) result = a.company.localeCompare(b.company, 'tr');
            return stockSort.direction === 'asc' ? result : -result;
        });
        return arr;
    }, [filteredStocks, stockSort, stocks?.benchmarks, returnMode, screenerData.sectorDiffByCompany]);

    const stockTableWrapRef = useRef<HTMLDivElement>(null);
    const {
        startIndex: stockStartIndex,
        endIndex: stockEndIndex,
        topSpacerHeight: stockTopSpacerHeight,
        bottomSpacerHeight: stockBottomSpacerHeight,
        containerRef: stockTableWrapCallbackRef,
    } = useTableVirtualizer({
        totalRows: sortedStocks.length,
        rowHeight: 38,
        overscan: 12,
        containerRef: stockTableWrapRef,
        resetDeps: [
            searchTerm,
            stockIndex,
            stockTableMode,
            returnMode,
            stockSort.key,
            stockSort.direction,
            activeFilterCount,
            screenerFilters,
        ],
    });

    const visibleStocks = useMemo(() => {
        return sortedStocks.slice(stockStartIndex, stockEndIndex);
    }, [sortedStocks, stockStartIndex, stockEndIndex]);

    const activeStockColumns = stockTableMode === 'fundamentals' ? STOCK_FUNDAMENTAL_COLUMNS : STOCK_COLUMNS;
    const stockTableTotalWidth = useMemo(
        () => 48 + activeStockColumns.reduce((sum, column) => sum + (column.width || 100), 0),
        [activeStockColumns],
    );

    const stockCardsBySymbol = useMemo(() => {
        const map = new Map<string, MarketStockCardItem>();
        for (const item of stockCards?.items || []) {
            map.set(item.symbol, item);
        }
        return map;
    }, [stockCards?.items]);
    const stockCardPendingSet = useMemo(() => new Set(stockCardPendingSymbols), [stockCardPendingSymbols]);
    const canAddStockCards = stockCardSymbols.length < MAX_STOCK_CARDS;
    const activeBenchmarkIndex = getBenchmarkIndex(returnMode);
    const totalStocksCount = stocks?.rows?.length ?? sortedStocks.length;
    const isStocksFiltered = sortedStocks.length !== totalStocksCount;
    const indexConstituents = useMemo(() => indexDetail?.constituents ?? [], [indexDetail?.constituents]);
    const indexImpactLevel = numericOrNull(indexDetail?.price) ?? numericOrNull(indexDetail?.prev_close);
    const positiveConstituents = indexConstituents.filter((row) => (row.point_effect || 0) > 0).length;
    const negativeConstituents = indexConstituents.filter((row) => (row.point_effect || 0) < 0).length;
    const neutralConstituents = indexConstituents.length - positiveConstituents - negativeConstituents;
    const sortedIndexConstituents = useMemo(() => {
        const arr = [...indexConstituents];
        arr.sort((a, b) => {
            const av = constituentSortValue(a, indexConstituentSort.key, indexImpactLevel);
            const bv = constituentSortValue(b, indexConstituentSort.key, indexImpactLevel);
            const aMissing = av == null || av === '';
            const bMissing = bv == null || bv === '';
            if (aMissing && bMissing) return a.symbol.localeCompare(b.symbol, 'tr');
            if (aMissing) return 1;
            if (bMissing) return -1;

            let result = 0;
            if (typeof av === 'string' || typeof bv === 'string') {
                result = String(av).localeCompare(String(bv), 'tr');
            } else {
                result = Number(av) - Number(bv);
            }
            if (result === 0) result = a.symbol.localeCompare(b.symbol, 'tr');
            return indexConstituentSort.direction === 'asc' ? result : -result;
        });
        return arr;
    }, [indexConstituents, indexConstituentSort, indexImpactLevel]);

    const indexConstituentWrapRef = useRef<HTMLDivElement>(null);
    const {
        startIndex: indexConstituentStartIndex,
        endIndex: indexConstituentEndIndex,
        topSpacerHeight: indexConstituentTopSpacerHeight,
        bottomSpacerHeight: indexConstituentBottomSpacerHeight,
        containerRef: indexConstituentTableWrapCallbackRef,
    } = useTableVirtualizer({
        totalRows: sortedIndexConstituents.length,
        rowHeight: isMobileViewport ? 95 : 38,
        overscan: 12,
        containerRef: indexConstituentWrapRef,
        resetDeps: [
            activeSection,
            selectedIndex,
            indexConstituentSort.key,
            indexConstituentSort.direction,
            isMobileViewport,
        ],
    });

    const visibleIndexConstituents = useMemo(() => {
        return sortedIndexConstituents.slice(indexConstituentStartIndex, indexConstituentEndIndex);
    }, [sortedIndexConstituents, indexConstituentStartIndex, indexConstituentEndIndex]);
    const heatmapConstituents = useMemo(
        () =>
            indexConstituents
                .filter((row) => row.weight_pct != null && row.weight_pct > 0)
                .sort((a, b) => treemapValue(b) - treemapValue(a)),
        [indexConstituents],
    );
    const maxHeatmapMovePct = Math.max(
        0.01,
        ...heatmapConstituents.map((row) => Math.abs(row.change_pct ?? 0)),
    );
    const heatmapTiles = useMemo(
        () => buildHeatmapTiles(heatmapConstituents, indexImpactLevel),
        [heatmapConstituents, indexImpactLevel],
    );
    const hasPointEffects = useMemo(
        () => indexConstituents.some((row) => row.point_effect != null),
        [indexConstituents],
    );
    const netPointEffect = useMemo<number | null>(
        () => {
            if (!hasPointEffects) return null;
            return indexConstituents.reduce((total, row) => {
                if (row.point_effect == null || !Number.isFinite(row.point_effect)) return total;
                return total + row.point_effect;
            }, 0);
        },
        [indexConstituents, hasPointEffects],
    );
    const netImpactPct = netPointEffect != null && indexImpactLevel && indexImpactLevel > 0 ? (netPointEffect / indexImpactLevel) * 100 : null;
    const pageTitle =
        activeSection === 'indices'
            ? selectedIndex
                ? indexDetail
                    ? `${indexDetail.symbol} — ${indexDetail.label}`
                    : `${selectedIndex} Endeks Detayı`
                : 'Borsa İstanbul Endeksleri'
            : activeSection === 'stocks'
              ? 'BIST Hisseleri'
              : 'Piyasa Görünümü';
    const pageDescription =
        activeSection === 'indices'
            ? 'BIST ana ve sektör endekslerini, getirileri ve endeks içi şirket hareketlerini takip edin.'
            : activeSection === 'stocks'
              ? 'XUTUM, XU100 ve XU030 hisselerini fiyat, hacim ve piyasa değeriyle karşılaştırın.'
              : 'Güncel fiyatlar, finansal görünüm ve analiz erişimi tek ekranda.';
    const documentTitle = useMemo(() => {
        if (activeSection === 'indices' && selectedIndex) {
            const indexQuoteTitle = [
                formatTitleNumber(indexDetail?.price),
                formatTitlePct(indexDetail?.change_pct),
            ].filter(Boolean).join(' ');
            return buildDocumentTitle(
                indexDetail?.symbol || selectedIndex,
                indexQuoteTitle,
                indexDetail?.label || 'Endeks Detayı',
            );
        }
        if (activeSection === 'indices') {
            const indexCount = sortedIndices.length || indices?.rows.length;
            return buildDocumentTitle('Endeksler', indexCount ? `${indexCount} endeks` : null);
        }
        if (activeSection === 'stocks') {
            const stockCount = sortedStocks.length || stocks?.rows.length;
            return buildDocumentTitle('BIST Hisseleri', stocks?.index || stockIndex, stockCount ? `${stockCount} hisse` : null);
        }
        return buildDocumentTitle('Piyasa Özeti', market?.rows.length ? `${market.rows.length} hisse` : null);
    }, [
        activeSection,
        indexDetail?.change_pct,
        indexDetail?.label,
        indexDetail?.price,
        indexDetail?.symbol,
        indices?.rows.length,
        market?.rows.length,
        selectedIndex,
        sortedIndices.length,
        sortedStocks.length,
        stockIndex,
        stocks?.index,
        stocks?.rows.length,
    ]);
    useDocumentTitle(documentTitle);

    const onCompanyClick = useCallback(
        (ticker: string) => {
            const normalizedTicker = String(ticker || '').trim().toUpperCase();
            if (!normalizedTicker) return;
            if (onOpenTicker) {
                onOpenTicker(normalizedTicker);
                return;
            }
            window.location.href = `/?ticker=${normalizedTicker}`;
        },
        [onOpenTicker]
    );

    const handleStockIndexChange = (index: MarketStockIndex) => {
        setStockIndex(index);
        onNavigateStockIndex?.(index);
    };

    const handleReturnModeChange = (mode: StockReturnMode) => {
        if (mode === returnMode) return;
        setReturnMode(mode);
        onNavigateReturnMode?.(mode);
    };

    const handleStockTableModeChange = (mode: StockTableMode) => {
        if (mode === stockTableMode) return;
        setStockTableMode(mode);
        setStockSort({ key: mode === 'fundamentals' ? 'fk' : 'company', direction: 'asc' });
    };

    const handleScreenerFilterChange = <K extends keyof StockScreenerFilterState,>(
        key: K,
        value: StockScreenerFilterState[K],
    ) => {
        setScreenerFilters((current) => ({ ...current, [key]: value }));
        setActiveScreenerPreset(null);
    };

    const applyScreenerPreset = (presetId: StockScreenerPresetId) => {
        if (activeScreenerPreset === presetId) {
            setScreenerFilters({ ...EMPTY_STOCK_SCREENER_FILTERS });
            setActiveScreenerPreset(null);
            return;
        }
        const preset = STOCK_SCREENER_PRESETS.find((option) => option.id === presetId);
        if (!preset) return;
        setScreenerFilters({ ...EMPTY_STOCK_SCREENER_FILTERS, ...preset.filters });
        setActiveScreenerPreset(presetId);
    };

    const clearScreenerFilters = () => {
        setScreenerFilters({ ...EMPTY_STOCK_SCREENER_FILTERS });
        setActiveScreenerPreset(null);
    };

    const renderScreenerRangeRows = (card: ScreenerRangeSpec['card']) => (
        SCREENER_RANGE_SPECS
            .filter((spec) => spec.card === card)
            .map((spec) => {
                const minRaw = screenerFilters[spec.minKey];
                const maxRaw = screenerFilters[spec.maxKey];
                const minNum = parseScreenerNumber(minRaw);
                const maxNum = parseScreenerNumber(maxRaw);
                const minInvalid = minRaw.trim() !== '' && minNum === undefined;
                const maxInvalid = maxRaw.trim() !== '' && maxNum === undefined;
                const isReversed = minNum !== undefined && maxNum !== undefined && minNum > maxNum;
                const isFilled = minRaw.trim() !== '' || maxRaw.trim() !== '';

                return (
                    <div
                        className={`screener-range-row${isReversed ? ' is-reversed' : ''}`}
                        key={spec.field}
                        title={isReversed ? 'Minimum değer maksimum değerden büyük' : undefined}
                    >
                        <div className="screener-range-label">
                            <div className="screener-range-title">
                                <strong>{spec.label}</strong>
                                <small>{spec.hint}</small>
                            </div>
                            {isFilled && (
                                <button
                                    type="button"
                                    className="screener-range-row-clear"
                                    onClick={() => {
                                        setScreenerFilters((c) => ({ ...c, [spec.minKey]: '', [spec.maxKey]: '' }));
                                        setActiveScreenerPreset(null);
                                    }}
                                    title={`${spec.label} filtresini temizle`}
                                    aria-label={`${spec.label} filtresini temizle`}
                                >
                                    <X size={11} aria-hidden="true" />
                                </button>
                            )}
                        </div>
                        <div className="screener-range-inputs">
                            <label
                                className={`screener-num-field${minRaw.trim() !== '' ? ' is-filled' : ''}${minInvalid ? ' is-invalid' : ''}`}
                            >
                                <span className="screener-num-tag">Min</span>
                                <input
                                    type="text"
                                    inputMode="decimal"
                                    value={minRaw}
                                    aria-label={`${spec.label} minimum`}
                                    onChange={(e) => handleScreenerFilterChange(spec.minKey, e.currentTarget.value)}
                                />
                                <span className="screener-num-unit">{spec.unit}</span>
                            </label>
                            <span className="screener-range-sep" aria-hidden="true">—</span>
                            <label
                                className={`screener-num-field${maxRaw.trim() !== '' ? ' is-filled' : ''}${maxInvalid ? ' is-invalid' : ''}`}
                            >
                                <span className="screener-num-tag">Maks</span>
                                <input
                                    type="text"
                                    inputMode="decimal"
                                    value={maxRaw}
                                    aria-label={`${spec.label} maksimum`}
                                    onChange={(e) => handleScreenerFilterChange(spec.maxKey, e.currentTarget.value)}
                                />
                                <span className="screener-num-unit">{spec.unit}</span>
                            </label>
                        </div>
                    </div>
                );
            })
    );

    const handleSelectIndex = (index: MarketIndexCode | null) => {
        setSelectedIndex(index);
        if (index) {
            setActiveSection('indices');
        }
        onNavigateIndexDetail?.(index);
    };

    const handleAddStockCard = (symbol: string) => {
        const normalized = normalizeStockCardSymbol(symbol);
        setStockCardSymbols((prev) => {
            if (prev.includes(normalized) || prev.length >= MAX_STOCK_CARDS) return prev;
            return [...prev, normalized];
        });
        setStockCardSearchTerm('');
        setStockCardPickerOpen(false);
    };

    const handleStockCardMoveStart = (symbol: string) => {
        draggingStockCardSymbolRef.current = symbol;
        setDraggingStockCardSymbol(symbol);
    };

    const moveStockCard = (
        sourceSymbol: string,
        targetSymbol: string,
        placement: StockCardDropPlacement,
    ) => {
        if (sourceSymbol === targetSymbol) return;

        setStockCardSymbols((prev) => {
            if (!prev.includes(sourceSymbol) || !prev.includes(targetSymbol)) return prev;
            const next = prev.filter((symbol) => symbol !== sourceSymbol);
            const targetIndex = next.indexOf(targetSymbol);
            if (targetIndex < 0) return prev;
            next.splice(targetIndex + (placement === 'after' ? 1 : 0), 0, sourceSymbol);
            const didChange = next.some((symbol, index) => symbol !== prev[index]);
            return didChange ? next : prev;
        });
    };

    const handleStockCardMoveOver = (targetSymbol: string, placement: StockCardDropPlacement) => {
        const sourceSymbol = draggingStockCardSymbolRef.current;
        if (!sourceSymbol || sourceSymbol === targetSymbol) return;
        moveStockCard(sourceSymbol, targetSymbol, placement);
    };

    const handleStockCardMoveToEnd = () => {
        const sourceSymbol = draggingStockCardSymbolRef.current;
        if (!sourceSymbol) return;
        setStockCardSymbols((prev) => {
            if (!prev.includes(sourceSymbol) || prev[prev.length - 1] === sourceSymbol) return prev;
            const next = prev.filter((symbol) => symbol !== sourceSymbol);
            next.push(sourceSymbol);
            return next;
        });
    };

    const handleStockCardGridDragOver = (event: React.DragEvent<HTMLDivElement>) => {
        const sourceSymbol = draggingStockCardSymbolRef.current;
        if (!sourceSymbol) return;

        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';

        const cardElements = Array.from(
            event.currentTarget.querySelectorAll<HTMLElement>('.stock-card:not(.stock-card-dragging)'),
        );
        let closestDrop: {
            symbol: string;
            placement: StockCardDropPlacement;
            distance: number;
        } | null = null;

        for (const cardElement of cardElements) {
            const targetSymbol = cardElement.dataset.stockCardSymbol;
            if (!targetSymbol || targetSymbol === sourceSymbol) continue;

            const rect = cardElement.getBoundingClientRect();
            const outsideX = event.clientX < rect.left
                ? rect.left - event.clientX
                : event.clientX > rect.right
                  ? event.clientX - rect.right
                  : 0;
            const outsideY = event.clientY < rect.top
                ? rect.top - event.clientY
                : event.clientY > rect.bottom
                  ? event.clientY - rect.bottom
                  : 0;
            const placement: StockCardDropPlacement =
                event.clientX >= rect.left + rect.width / 2 ? 'after' : 'before';
            const distance = outsideY * 1000 + outsideX;

            if (!closestDrop || distance < closestDrop.distance) {
                closestDrop = { symbol: targetSymbol, placement, distance };
            }
        }

        if (closestDrop) {
            moveStockCard(sourceSymbol, closestDrop.symbol, closestDrop.placement);
        } else {
            handleStockCardMoveToEnd();
        }
    };

    const handleStockCardMoveEnd = () => {
        draggingStockCardSymbolRef.current = null;
        setDraggingStockCardSymbol(null);
    };

    const handleRemoveStockCard = (symbol: string) => {
        if (draggingStockCardSymbolRef.current === symbol) {
            handleStockCardMoveEnd();
        }
        setStockCardSymbols((prev) => prev.filter((item) => item !== symbol));
    };

    const handleSectionChange = (section: MarketSection) => {
        if (onNavigateSection) {
            onNavigateSection(section);
            return;
        }
        setActiveSection(section);
        setSelectedIndex(null);
    };

    const handleStockSort = (key: StockSortKey) => {
        setStockSort((prev) => {
            if (prev.key === key) {
                return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
            }
            return { key, direction: key === 'company' ? 'asc' : 'desc' };
        });
    };

    const handleIndexSort = (key: IndexSortKey) => {
        setIndexSort((prev) => {
            if (prev.key === key) {
                return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
            }
            return { key, direction: key === 'symbol' ? 'asc' : 'desc' };
        });
    };

    const handleIndexConstituentSort = (key: IndexConstituentSortKey) => {
        setIndexConstituentSort((prev) => {
            if (prev.key === key) {
                return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
            }
            return { key, direction: key === 'symbol' ? 'asc' : 'desc' };
        });
    };

    const getTreemapTileScale = (row: MarketIndexConstituent): number => {
        const movePct = Math.abs(row.change_pct ?? 0);
        return Math.min(1, Math.sqrt(movePct / maxHeatmapMovePct));
    };

    const getHeatmapTileFontSize = (tile: HeatmapTile): string => {
        if (tile.areaPct >= 16 && tile.width >= 22 && tile.height >= 22) return '2rem';
        if (tile.areaPct >= 8) return '1.45rem';
        if (tile.areaPct >= 3.5) return '1rem';
        if (tile.areaPct >= 1.2) return '0.78rem';
        return '0.64rem';
    };

    return (
        <div className={`mn-layout${navCollapsed ? ' mn-nav-collapsed' : ''}`}>
            <MarketsNavigation
                collapsed={navCollapsed}
                activeSection={activeSection}
                onCollapsedChange={setNavCollapsed}
                onSectionChange={handleSectionChange}
                onFundSectionChange={onNavigateFundSection}
                onSelectTicker={onCompanyClick}
                onSelectFund={onOpenFund}
            />
            <div
                className={`markets-workspace${activeSection === 'markets' ? ' markets-workspace-with-rail' : ''}${navCollapsed ? ' is-nav-collapsed' : ''}${railCollapsed ? ' is-rail-collapsed' : ''}`}
                data-collapsed-menus={collapsedMenuCount}
            >
                <div className="market-page" ref={marketPageRef}>
            {activeSection !== 'markets' && (
                <header className="market-header">
                    <div className="market-title">
                        <h1>{pageTitle}</h1>
                        <p>{pageDescription}</p>
                    </div>
                </header>
            )}

            {activeSection === 'markets' && (
                <div className="market-terminal-shell">
                    <div className="market-overview-head">
                        <div className="market-title">
                            <h1>Piyasa Görünümü</h1>
                            <p>Güncel fiyatlar, finansal görünüm ve seçtiğiniz hisse kartları tek ekranda.</p>
                        </div>
                    </div>
                    <div className="market-session-bar">
                        <div className="market-session-date">
                            <span className="market-session-dot" aria-label="Piyasa izleme aktif" />
                            <TerminalClock intervalMs={isMobileViewport ? 10000 : 1000} />
                        </div>
                    </div>
                    <MarketWatchStrip variant="compact" onSelectIndex={handleSelectIndex} />
                </div>
            )}

            {activeSection === 'markets' && loading && (
                <div className="market-content market-content-terminal" role="status" aria-live="polite" aria-label="Piyasa verileri yükleniyor">
                    <div className="market-main-column">
                        <div className="stock-card-skeleton-grid" aria-hidden="true">
                            {[0, 1, 2].map((cardIdx) => (
                                <div key={cardIdx} className="stock-card-skeleton">
                                    <div className="stock-card-skeleton-head">
                                        <div className="stock-card-skeleton-logo stock-card-skeleton-pulse" />
                                        <div className="stock-card-skeleton-id">
                                            <div className="stock-card-skeleton-symbol stock-card-skeleton-pulse" />
                                            <div className="stock-card-skeleton-company stock-card-skeleton-pulse" />
                                        </div>
                                    </div>
                                    <div className="stock-card-skeleton-price-row">
                                        <div className="stock-card-skeleton-price stock-card-skeleton-pulse" />
                                        <div className="stock-card-skeleton-change stock-card-skeleton-pulse" />
                                    </div>
                                    <div className="stock-card-skeleton-metrics">
                                        <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                        <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                        <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                    </div>
                                    <div className="stock-card-skeleton-chart stock-card-skeleton-pulse" />
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
            
            {activeSection === 'markets' && error && (
                <div className="error-message">
                    <strong>Hata:</strong> {error}
                    <button onClick={() => loadStats()} className="btn-secondary" style={{ marginLeft: '1rem' }}>Tekrar Dene</button>
                </div>
            )}

            {activeSection === 'markets' && !loading && !error && market && (
                <div className={`market-content market-content-terminal${mobileWatchlistSearchOpen ? ' mobile-watchlist-search-open' : ''}`}>
                    <div className="market-main-column">
                        <MobileMarketOverview
                            index={mobileIndexDetail}
                            rows={market.rows}
                            stockQuoteRows={stocks?.rows ?? []}
                            stockCardItems={stockCards?.items ?? []}
                            watchlistItems={watchlist.items}
                            fundRows={mobileWatchlistFundRows}
                            isMobileViewport={isMobileViewport}
                            onSelectTicker={onCompanyClick}
                            onSelectIndex={handleSelectIndex}
                            onOpenFund={onOpenFund}
                            onAddStock={(symbol) => watchlist.addItem({ kind: 'stock', symbol })}
                            onAddFund={(fund) => {
                                setMobileWatchlistFundRows((current) => ({ ...current, [fund.fund_code]: fund }));
                                watchlist.addItem({ kind: 'fund', symbol: fund.fund_code, label: fund.name });
                            }}
                            onRemoveWatchlistItem={(item) => watchlist.removeItem(item.kind, item.symbol)}
                            onSearchOpenChange={setMobileWatchlistSearchOpen}
                        />
                        {!isMobileViewport && (
                        <section className="panel stock-cards-panel">
                            <div className="stock-cards-toolbar">
                                <div className="stock-cards-title-row">
                                    <h2>Hisse Kartları</h2>
                                    <span className="panel-kicker">
                                        {stockCardSymbols.length}/{MAX_STOCK_CARDS} kart
                                    </span>
                                    <button
                                        type="button"
                                        className="stock-card-add-link"
                                        onClick={() => setStockCardPickerOpen((open) => !open)}
                                        disabled={!canAddStockCards}
                                    >
                                        <span>Ekle</span>
                                        <Plus size={16} aria-hidden="true" />
                                    </button>
                                </div>
                            </div>

                            {stockCardPickerOpen && (
                                <div className="stock-card-picker">
                                    <div className="stock-card-picker-search">
                                        <Search size={16} aria-hidden="true" />
                                        <input
                                            type="text"
                                            placeholder="Hisse ara..."
                                            value={stockCardSearchTerm}
                                            onChange={(event) => setStockCardSearchTerm(event.target.value)}
                                            autoFocus
                                        />
                                        <button
                                            type="button"
                                            className="stock-card-picker-close"
                                            onClick={() => {
                                                setStockCardPickerOpen(false);
                                                setStockCardSearchTerm('');
                                            }}
                                            aria-label="Hisse seçiciyi kapat"
                                            title="Kapat"
                                        >
                                            <X size={16} aria-hidden="true" />
                                        </button>
                                    </div>
                                    <div className="stock-card-picker-heading">Hisse Senetleri</div>
                                    <div className="stock-card-picker-list">
                                        {stockCardCandidates.length > 0 ? (
                                            stockCardCandidates.map((row) => (
                                                <button
                                                    key={row.company}
                                                    type="button"
                                                    onClick={() => handleAddStockCard(row.company)}
                                                >
                                                    <SymbolLogo
                                                        symbol={row.company}
                                                        name={row.company}
                                                        kind="stock"
                                                        logoUrl={row.logo_url}
                                                        size="sm"
                                                        className="stock-card-picker-logo"
                                                    />
                                                    <span className="stock-card-picker-copy">
                                                        <strong>{row.company}</strong>
                                                    </span>
                                                </button>
                                            ))
                                        ) : (
                                            <div className="stock-card-picker-empty">
                                                Eklenebilecek hisse bulunamadı.
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {stockCardsError && (
                                <div className="stock-cards-inline-alert" role="status" aria-live="polite">
                                    <span>Hisse kartları güncellenemedi.</span>
                                    <button
                                        type="button"
                                        onClick={() => loadStockCards(false, true, stockCardSymbols.slice(0, STOCK_CARD_INITIAL_BATCH_SIZE))}
                                    >
                                        Tekrar dene
                                    </button>
                                </div>
                            )}

                            {stockCardSymbols.length === 0 ? (
                                <div className="stock-card-empty-state">
                                    <h3>Henüz hisse kartı yok</h3>
                                    <p>Takip etmek istediğiniz hisseleri ekleyin; fiyat, gün içi grafik ve temel piyasa verileri burada görünsün.</p>
                                    <button
                                        type="button"
                                        className="stock-card-empty-action"
                                        onClick={() => setStockCardPickerOpen(true)}
                                    >
                                        <Plus size={16} aria-hidden="true" />
                                        Hisse kartı ekle
                                    </button>
                                </div>
                            ) : !stockCards && !stockCardsError ? (
                                <div className="stock-card-skeleton-grid">
                                    {stockCardSymbols.map((symbol) => (
                                        <div key={symbol} className="stock-card-skeleton">
                                            <div className="stock-card-skeleton-head">
                                                <div className="stock-card-skeleton-logo stock-card-skeleton-pulse" />
                                                <div className="stock-card-skeleton-id">
                                                    <div className="stock-card-skeleton-symbol stock-card-skeleton-pulse" />
                                                    <div className="stock-card-skeleton-company stock-card-skeleton-pulse" />
                                                </div>
                                            </div>
                                            <div className="stock-card-skeleton-price-row">
                                                <div className="stock-card-skeleton-price stock-card-skeleton-pulse" />
                                                <div className="stock-card-skeleton-change stock-card-skeleton-pulse" />
                                            </div>
                                            <div className="stock-card-skeleton-metrics">
                                                <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                                <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                                <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                            </div>
                                            <div className="stock-card-skeleton-chart stock-card-skeleton-pulse" />
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div
                                    className="stock-card-grid"
                                    onDragOver={handleStockCardGridDragOver}
                                    onDrop={(event) => {
                                        event.preventDefault();
                                        handleStockCardMoveEnd();
                                    }}
                                >
                                    {stockCardSymbols.map((symbol) => {
                                        const item = stockCardsBySymbol.get(symbol) || emptyStockCardItem(symbol);
                                        const hasLoadedItem = stockCardsBySymbol.has(symbol);
                                        const isCardLoading =
                                            !hasLoadedItem
                                            && (stockCardPendingSet.has(symbol) || stockCardsLoading);
                                        return (
                                            <MarketStockCard
                                                key={symbol}
                                                item={item}
                                                onOpen={() => onCompanyClick(symbol)}
                                                onRemove={() => handleRemoveStockCard(symbol)}
                                                onMoveStart={() => handleStockCardMoveStart(symbol)}
                                                onMoveOver={(placement) => handleStockCardMoveOver(symbol, placement)}
                                                onMoveEnd={handleStockCardMoveEnd}
                                                isLoading={isCardLoading}
                                                isDragging={draggingStockCardSymbol === symbol}
                                            />
                                        );
                                    })}
                                </div>
                            )}
                        </section>
                        )}
                    </div>
                </div>
            )}

            {activeSection === 'stocks' && (
                <div className="stocks-view">
                    <div className="panel stocks-table-panel">
                        <div className="panel-header stocks-panel-header">
                            <div className="stocks-panel-header-top">
                                <div className="stocks-panel-title-group">
                                    <h2>Hisseler</h2>
                                    <span className="stocks-count-badge">
                                        {isStocksFiltered ? `${sortedStocks.length} / ${totalStocksCount}` : sortedStocks.length} hisse
                                    </span>
                                </div>
                                <div className="stocks-header-actions">
                                    <div className="stocks-search-wrap">
                                        <Search size={14} className="stocks-search-icon" aria-hidden="true" />
                                        <input
                                            type="text"
                                            placeholder="Hisse kodu ile ara..."
                                            value={searchTerm}
                                            onChange={(e) => setSearchTerm(e.target.value)}
                                            className="stocks-search-input"
                                        />
                                        {searchTerm && (
                                            <button
                                                type="button"
                                                className="stocks-search-clear"
                                                onClick={() => setSearchTerm('')}
                                                aria-label="Aramayı temizle"
                                            >
                                                <X size={12} aria-hidden="true" />
                                            </button>
                                        )}
                                    </div>
                                    <button
                                        type="button"
                                        className={`stocks-screener-trigger-btn${isScreenerActive ? ' is-active' : ''}`}
                                        onClick={() => setScreenerModalOpen(true)}
                                        aria-label="Hisse Tarayıcısı ve Filtreler"
                                    >
                                        <SlidersHorizontal size={13} aria-hidden="true" />
                                        <span>Hisse Tarayıcısı</span>
                                        {activeFilterCount > 0 && (
                                            <span className="screener-count-badge">{activeFilterCount}</span>
                                        )}
                                    </button>
                                </div>
                            </div>
                            <div className="stocks-panel-toolbar">
                                <div className="stocks-toolbar-left">
                                    <div className="stocks-control-group">
                                        <span className="stocks-control-label">Endeks</span>
                                        <div
                                            className="stocks-segment"
                                            aria-label="Endeks seçimi"
                                            style={{
                                                '--item-count': STOCK_INDEX_OPTIONS.length,
                                                '--active-index': STOCK_INDEX_OPTIONS.indexOf(stockIndex)
                                            } as React.CSSProperties}
                                        >
                                            {STOCK_INDEX_OPTIONS.map((option) => (
                                                <button
                                                    key={option}
                                                    type="button"
                                                    className={stockIndex === option ? 'active' : ''}
                                                    aria-pressed={stockIndex === option}
                                                    onClick={() => handleStockIndexChange(option)}
                                                >
                                                    {option}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="stocks-control-group">
                                        <span className="stocks-control-label">Görünüm</span>
                                        <div
                                            className="stocks-segment stocks-table-mode-segment"
                                            aria-label="Tablo görünümü"
                                            style={{
                                                '--item-count': 2,
                                                '--active-index': stockTableMode === 'performance' ? 0 : 1,
                                            } as React.CSSProperties}
                                        >
                                            {([['performance', 'Getiri'], ['fundamentals', 'Temel Analiz']] as const).map(([mode, label]) => (
                                                <button
                                                    key={mode}
                                                    type="button"
                                                    className={stockTableMode === mode ? 'active' : ''}
                                                    aria-pressed={stockTableMode === mode}
                                                    onClick={() => handleStockTableModeChange(mode)}
                                                >
                                                    {label}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                    {stockTableMode === 'performance' && (
                                        <div className="stocks-control-group">
                                            <span className="stocks-control-label">Kıyas</span>
                                            <div
                                                className="stocks-segment stocks-return-segment"
                                                aria-label="Getiri kıyas modu"
                                                style={{
                                                    '--item-count': RETURN_MODE_OPTIONS.length,
                                                    '--active-index': RETURN_MODE_OPTIONS.findIndex(o => o.id === returnMode),
                                                } as React.CSSProperties}
                                            >
                                                {RETURN_MODE_OPTIONS.map((option) => (
                                                    <button
                                                        key={option.id}
                                                        type="button"
                                                        className={returnMode === option.id ? 'active' : ''}
                                                        aria-pressed={returnMode === option.id}
                                                        onClick={() => handleReturnModeChange(option.id)}
                                                    >
                                                        {option.id === 'absolute' ? 'Mutlak' : option.id === 'relative_xu100' ? 'XU100' : 'XU030'}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                                {stocks && (
                                    <div className="stocks-toolbar-meta">
                                        {stockTableMode === 'performance' && activeBenchmarkIndex && (
                                            <span className="stocks-meta-chip" title={`${activeBenchmarkIndex} göreceli getiri bazı`}>
                                                Baz: <strong>{activeBenchmarkIndex}</strong> · {formatUpdateTime(stocks.benchmarks?.[activeBenchmarkIndex]?.as_of)}
                                            </span>
                                        )}
                                        {stocks.partial && (
                                            <span
                                                className="stocks-status-pill is-info"
                                                title={stocks.partial_message || 'Detaylı veriler hazırlanıyor.'}
                                            >
                                                <span>{stocks.partial_message || 'Veriler hazırlanıyor'}</span>
                                            </span>
                                        )}
                                        {stocks.stale && (
                                            <span
                                                className="stocks-status-pill is-warning"
                                                title={stocks.quote_error || 'Son veri alınamadı; son sağlıklı fiyat gösteriliyor.'}
                                            >
                                                <AlertCircle size={12} aria-hidden="true" />
                                                <span>Son sağlıklı fiyat</span>
                                            </span>
                                        )}
                                        {stocksError && (
                                            <span className="stocks-status-pill is-danger" title={stocksError}>
                                                <AlertCircle size={12} aria-hidden="true" />
                                                <span>Yenileme başarısız</span>
                                            </span>
                                        )}
                                        <span className="stocks-meta-time" title="Son fiyat güncellemesi">
                                            <Clock3 size={12} aria-hidden="true" />
                                            <span>{formatUpdateTime(stocks.as_of)}</span>
                                        </span>
                                        <button
                                            type="button"
                                            className={`stocks-refresh-btn${stocksLoading ? ' is-spinning' : ''}`}
                                            onClick={() => loadStocks(false, true, stockIndex)}
                                            disabled={stocksLoading}
                                            title="Verileri yenile"
                                            aria-label="Hisse verilerini yenile"
                                        >
                                            <RefreshCw size={12} aria-hidden="true" />
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>

                        {isScreenerActive && (
                            <div className="stocks-screener-active-banner">
                                <div className="stocks-screener-active-banner-left">
                                    <SlidersHorizontal size={13} aria-hidden="true" />
                                    <span>
                                        <strong>{activeFilterCount} filtre devrede:</strong> {screenerData.rows.length} / {screenerData.totalCount} hisse gösteriliyor
                                    </span>
                                </div>
                                <div className="stocks-screener-active-banner-actions">
                                    <button
                                        type="button"
                                        className="stocks-screener-banner-btn"
                                        onClick={() => setScreenerModalOpen(true)}
                                    >
                                        Filtreleri Düzenle
                                    </button>
                                    <button
                                        type="button"
                                        className="stocks-screener-banner-btn stocks-screener-banner-clear"
                                        onClick={clearScreenerFilters}
                                    >
                                        <X size={12} aria-hidden="true" /> Temizle
                                    </button>
                                </div>
                            </div>
                        )}

                        {screenerModalOpen && createPortal(
                            <div
                                className="screener-modal-backdrop"
                                role="dialog"
                                aria-modal="true"
                                aria-labelledby="screener-modal-title"
                                onClick={() => setScreenerModalOpen(false)}
                            >
                                <div className="screener-modal" onClick={(e) => e.stopPropagation()}>
                                    <div className="screener-modal-header">
                                        <div className="screener-modal-title-group">
                                            <div>
                                                <h3 id="screener-modal-title">Hisse Tarayıcısı</h3>
                                                <p>
                                                    {stockIndex} · {screenerData.totalCount} hisse · {screenerData.financialCount} bilanço verisi
                                                </p>
                                            </div>
                                        </div>
                                        <button
                                            type="button"
                                            className="screener-modal-close-btn"
                                            onClick={() => setScreenerModalOpen(false)}
                                            aria-label="Kapat"
                                        >
                                            <X size={17} aria-hidden="true" />
                                        </button>
                                    </div>

                                    <div className="screener-modal-subbar">
                                        <div className="screener-modal-tabs" role="tablist" aria-label="Tarayıcı kategorileri">
                                            {([
                                                { id: 'all', label: 'Tümü' },
                                                { id: 'profile', label: 'Sektör & Nitelik' },
                                                { id: 'valuation', label: 'Değerleme' },
                                                { id: 'historical', label: 'Tarihsel Ortalamalar' },
                                                { id: 'financials', label: 'Kârlılık & Bilanço' },
                                                { id: 'market', label: 'Piyasa & Getiri' },
                                            ] as const).map(({ id, label }) => {
                                                const count = screenerCategoryCounts[id];
                                                return (
                                                    <button
                                                        key={id}
                                                        type="button"
                                                        role="tab"
                                                        aria-selected={screenerCategory === id}
                                                        className={`screener-tab-btn${screenerCategory === id ? ' is-active' : ''}`}
                                                        onClick={() => setScreenerCategory(id)}
                                                    >
                                                        <span>{label}</span>
                                                        {id !== 'all' && count > 0 && (
                                                            <span className="screener-tab-count">{count}</span>
                                                        )}
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    </div>

                                    <div className="screener-modal-body">
                                        <div className="screener-modal-presets-section">
                                            <div className="screener-modal-section-label">
                                                <span>Hazır Stratejiler</span>
                                            </div>
                                            <div className="screener-modal-presets-row">
                                                {STOCK_SCREENER_PRESETS.map((preset) => (
                                                    <button
                                                        key={preset.id}
                                                        type="button"
                                                        aria-pressed={activeScreenerPreset === preset.id}
                                                        className={`screener-preset-chip${activeScreenerPreset === preset.id ? ' is-active' : ''}`}
                                                        onClick={() => applyScreenerPreset(preset.id)}
                                                    >
                                                        {preset.label}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        {screenerActiveChips.length > 0 && (
                                            <div className="screener-active-chips-row" aria-label="Aktif filtreler">
                                                {screenerActiveChips.map((chip) => (
                                                    <span key={chip.id} className="screener-active-chip">
                                                        <span>{chip.label}</span>
                                                        <button
                                                            type="button"
                                                            onClick={chip.onClear}
                                                            aria-label={`${chip.label} filtresini kaldır`}
                                                        >
                                                            <X size={11} aria-hidden="true" />
                                                        </button>
                                                    </span>
                                                ))}
                                            </div>
                                        )}

                                        <div className={`screener-modal-grid${screenerCategory === 'valuation' ? ' is-single' : ''}`}>
                                            {(screenerCategory === 'all' || screenerCategory === 'profile') && (
                                                <div className="screener-modal-card">
                                                    <div className="screener-card-header">
                                                        <Building2 size={14} aria-hidden="true" />
                                                        <span>Sektör & Nitelik</span>
                                                    </div>
                                                    <div className="screener-card-content">
                                                        <div className="screener-field">
                                                            <label className="screener-field-label" htmlFor="screener-sector-select">
                                                                Sektör
                                                            </label>
                                                            <select
                                                                id="screener-sector-select"
                                                                className="screener-select"
                                                                value={screenerFilters.sector}
                                                                onChange={(e) => handleScreenerFilterChange('sector', e.currentTarget.value)}
                                                            >
                                                                <option value="">Tüm Sektörler ({screenerData.totalCount} hisse)</option>
                                                                {screenerData.sectors.map((s) => (
                                                                    <option key={s} value={s}>
                                                                        {s} ({screenerData.sectorCounts.get(s) || 0})
                                                                    </option>
                                                                ))}
                                                            </select>
                                                        </div>
                                                        <div className="screener-toggles-grid">
                                                            {SCREENER_TOGGLE_SPECS.filter((spec) => spec.category === 'profile').map(({ key, label }) => {
                                                                const checked = screenerFilters[key];
                                                                return (
                                                                    <label
                                                                        key={key}
                                                                        className={`screener-toggle-item${checked ? ' is-checked' : ''}`}
                                                                    >
                                                                        <input
                                                                            type="checkbox"
                                                                            checked={checked}
                                                                            onChange={(e) => handleScreenerFilterChange(key, e.currentTarget.checked)}
                                                                        />
                                                                        <span className="screener-checkbox-box" aria-hidden="true">
                                                                            {checked && <Check size={11} strokeWidth={2.8} />}
                                                                        </span>
                                                                        <span>{label}</span>
                                                                    </label>
                                                                );
                                                            })}
                                                        </div>
                                                    </div>
                                                </div>
                                            )}

                                            {(screenerCategory === 'all' || screenerCategory === 'valuation') && (
                                                <div className="screener-modal-card">
                                                    <div className="screener-card-header">
                                                        <TrendingUp size={14} aria-hidden="true" />
                                                        <span>Değerleme Çarpanları</span>
                                                    </div>
                                                    <div className="screener-card-content">
                                                        {renderScreenerRangeRows('valuation')}
                                                    </div>
                                                </div>
                                            )}

                                            {(screenerCategory === 'all' || screenerCategory === 'financials') && (
                                                <div className="screener-modal-card">
                                                    <div className="screener-card-header">
                                                        <Percent size={14} aria-hidden="true" />
                                                        <span>Kârlılık & Kaldıraç</span>
                                                    </div>
                                                    <div className="screener-card-content">
                                                        {renderScreenerRangeRows('profitability')}
                                                    </div>
                                                </div>
                                            )}

                                            {(screenerCategory === 'all' || screenerCategory === 'market') && (
                                                <div className="screener-modal-card">
                                                    <div className="screener-card-header">
                                                        <BarChart3 size={14} aria-hidden="true" />
                                                        <span>Piyasa, Likidite & Fiyat</span>
                                                    </div>
                                                    <div className="screener-card-content">
                                                        {renderScreenerRangeRows('market_scale')}
                                                    </div>
                                                </div>
                                            )}

                                            {(screenerCategory === 'all' || screenerCategory === 'financials') && (
                                                <div className="screener-modal-card">
                                                    <div className="screener-card-header">
                                                        <Coins size={14} aria-hidden="true" />
                                                        <span>Bilanço & TTM Ölçek</span>
                                                    </div>
                                                    <div className="screener-card-content">
                                                        {renderScreenerRangeRows('scale')}
                                                    </div>
                                                </div>
                                            )}

                                            {(screenerCategory === 'all' || screenerCategory === 'market') && (
                                                <div className="screener-modal-card">
                                                    <div className="screener-card-header">
                                                        <TrendingUp size={14} aria-hidden="true" />
                                                        <span>Fiyat Performansı & Getiri</span>
                                                    </div>
                                                    <div className="screener-card-content">
                                                        {renderScreenerRangeRows('returns')}
                                                    </div>
                                                </div>
                                            )}

                                            {(screenerCategory === 'all' || screenerCategory === 'historical') && (
                                                <div className="screener-modal-card">
                                                    <div className="screener-card-header">
                                                        <History size={14} aria-hidden="true" />
                                                        <span>Tarihsel Çarpan Ortalamaları (KAP)</span>
                                                    </div>
                                                    <div className="screener-card-content">
                                                        {renderScreenerRangeRows('hist_multiples')}
                                                    </div>
                                                </div>
                                            )}

                                            {(screenerCategory === 'all' || screenerCategory === 'historical') && (
                                                <div className="screener-modal-card">
                                                    <div className="screener-card-header">
                                                        <Activity size={14} aria-hidden="true" />
                                                        <span>Fiyat Ortalamaları & 52H Zirve/Dip</span>
                                                    </div>
                                                    <div className="screener-card-content">
                                                        {renderScreenerRangeRows('technicals')}
                                                    </div>
                                                </div>
                                            )}

                                            {(screenerCategory === 'all' || screenerCategory === 'historical') && (
                                                <div className="screener-modal-card is-span-2">
                                                    <div className="screener-card-header">
                                                        <BarChart3 size={14} aria-hidden="true" />
                                                        <span>Sektör Ortalaması & Hızlı Seçimler</span>
                                                    </div>
                                                    <div className="screener-card-content">
                                                        {(() => {
                                                            const selectedSectorAvg = screenerFilters.sector
                                                                ? screenerData.sectorAverages.get(screenerFilters.sector)
                                                                : undefined;
                                                            if (selectedSectorAvg) {
                                                                return (
                                                                    <div className="screener-sector-benchmark-strip">
                                                                        <span><strong>{screenerFilters.sector}</strong> ortalaması ({selectedSectorAvg.count} hisse):</span>
                                                                        <span>F/K: <strong>{formatFundamentalRatio(selectedSectorAvg.fk)}</strong></span>
                                                                        <span>PD/DD: <strong>{formatFundamentalRatio(selectedSectorAvg.pd_dd)}</strong></span>
                                                                        <span>FD/FAVÖK: <strong>{formatFundamentalRatio(selectedSectorAvg.fd_favok)}</strong></span>
                                                                    </div>
                                                                );
                                                            }
                                                            return (
                                                                <div className="screener-sector-benchmark-strip">
                                                                    <span>Her hisse, kendi sektörünün pozitif güncel çarpan ortalamasına göre kıyaslanır ({screenerData.sectorAverages.size} sektör).</span>
                                                                </div>
                                                            );
                                                        })()}
                                                        {renderScreenerRangeRows('sector_relative')}
                                                        <div className="screener-toggles-grid is-three-col">
                                                            {SCREENER_TOGGLE_SPECS.filter((spec) => spec.category === 'historical').map(({ key, label }) => {
                                                                const checked = screenerFilters[key];
                                                                return (
                                                                    <label
                                                                        key={key}
                                                                        className={`screener-toggle-item${checked ? ' is-checked' : ''}`}
                                                                    >
                                                                        <input
                                                                            type="checkbox"
                                                                            checked={checked}
                                                                            onChange={(e) => handleScreenerFilterChange(key, e.currentTarget.checked)}
                                                                        />
                                                                        <span className="screener-checkbox-box" aria-hidden="true">
                                                                            {checked && <Check size={11} strokeWidth={2.8} />}
                                                                        </span>
                                                                        <span>{label}</span>
                                                                    </label>
                                                                );
                                                            })}
                                                        </div>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="screener-modal-footer">
                                        <button
                                            type="button"
                                            className="screener-modal-clear-btn"
                                            onClick={clearScreenerFilters}
                                            disabled={!isScreenerActive}
                                        >
                                            <RotateCcw size={13} aria-hidden="true" />
                                            <span>Filtreleri Sıfırla</span>
                                        </button>
                                        <div className="screener-modal-footer-right">
                                            <span className="screener-modal-footer-info">
                                                <strong>{screenerData.rows.length}</strong> / {screenerData.totalCount} hisse
                                            </span>
                                            <button
                                                type="button"
                                                className="screener-modal-apply-btn"
                                                onClick={() => setScreenerModalOpen(false)}
                                            >
                                                Sonuçları Göster
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>,
                            document.body
                        )}

                        {stocksLoading && !stocks && (
                            <div className="mv-table-skeleton" role="status" aria-live="polite" aria-label="Hisseler yükleniyor">
                                <div className="mv-table-skeleton-head" aria-hidden="true">
                                    <span className="mv-sk-cell is-xs stock-card-skeleton-pulse" />
                                    <span className="mv-sk-cell is-lg stock-card-skeleton-pulse" />
                                    <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                    <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                    <span className="mv-sk-cell is-md stock-card-skeleton-pulse" />
                                    <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                    <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                </div>
                                {Array.from({ length: 10 }, (_, idx) => (
                                    <div key={idx} className="mv-table-skeleton-row" aria-hidden="true">
                                        <span className="mv-sk-cell is-xs stock-card-skeleton-pulse" />
                                        <div className="mv-sk-symbol">
                                            <span className="mv-sk-avatar stock-card-skeleton-pulse" />
                                            <span className="mv-sk-cell is-lg stock-card-skeleton-pulse" />
                                        </div>
                                        <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                        <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                        <span className="mv-sk-cell is-md stock-card-skeleton-pulse" />
                                        <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                        <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                    </div>
                                ))}
                            </div>
                        )}

                        {stocksError && !stocks && (
                            <div className="error-message">
                                <strong>Hata:</strong> {stocksError}
                                <button
                                    onClick={() => loadStocks(false, true, stockIndex)}
                                    className="btn-secondary"
                                    style={{ marginLeft: '1rem' }}
                                >
                                    Tekrar Dene
                                </button>
                            </div>
                        )}

                        {stocks && (
                            <>
                                {sortedStocks.length === 0 ? (
                                    <div className="no-results">Aranan kritere uyan hisse bulunamadı.</div>
                                ) : (
                                    <div className="stocks-table-wrap" ref={stockTableWrapCallbackRef}>
                                        <table
                                            className={`stocks-table${stockTableMode === 'fundamentals' ? ' stocks-fundamental-table' : ''}`}
                                            style={{
                                                width: `${stockTableTotalWidth}px`,
                                                minWidth: stockTableMode === 'fundamentals' ? `${stockTableTotalWidth}px` : '100%',
                                            }}
                                            aria-rowcount={sortedStocks.length + 1}
                                        >
                                            <colgroup>
                                                <col style={{ width: '48px', minWidth: '48px' }} />
                                                {activeStockColumns.map((column) => (
                                                    <col
                                                        key={column.key}
                                                        style={{ width: `${column.width || 100}px`, minWidth: `${column.width || 100}px` }}
                                                    />
                                                ))}
                                            </colgroup>
                                            <thead>
                                                <tr>
                                                    <th className="stocks-rank">#</th>
                                                    {activeStockColumns.map((column) => (
                                                        <th
                                                            key={column.key}
                                                            className={column.align === 'right' ? 'stocks-cell-right' : undefined}
                                                        >
                                                            <button
                                                                type="button"
                                                                className="stocks-sort-button"
                                                                onClick={() => handleStockSort(column.key)}
                                                                aria-sort={
                                                                    stockSort.key === column.key
                                                                        ? stockSort.direction === 'asc'
                                                                            ? 'ascending'
                                                                            : 'descending'
                                                                        : 'none'
                                                                }
                                                            >
                                                                <span className="stocks-sort-label-stack">
                                                                    <span>{stockTableMode === 'fundamentals' ? column.label : getColumnLabel(column.key, returnMode)}</span>
                                                                    {column.sublabel && <small>{column.sublabel}</small>}
                                                                </span>
                                                                {stockSort.key === column.key && (
                                                                    <span className="stocks-sort-indicator">
                                                                        {stockSort.direction === 'asc' ? '↑' : '↓'}
                                                                    </span>
                                                                )}
                                                            </button>
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {stockTopSpacerHeight > 0 && (
                                                    <tr className="stocks-spacer-row" style={{ height: `${stockTopSpacerHeight}px` }} aria-hidden="true">
                                                        <td
                                                            colSpan={activeStockColumns.length + 1}
                                                            style={{ height: `${stockTopSpacerHeight}px` }}
                                                        />
                                                    </tr>
                                                )}
                                                {visibleStocks.map((row, index) => {
                                                    const actualRank = stockStartIndex + index + 1;
                                                    return (
                                                        <StockTableRow
                                                            key={row.company}
                                                            row={row}
                                                            rank={actualRank}
                                                            mode={stockTableMode}
                                                            returnMode={returnMode}
                                                            benchmarks={stocks?.benchmarks}
                                                            sectorDiff={screenerData.sectorDiffByCompany.get(row.company)}
                                                            onCompanyClick={onCompanyClick}
                                                        />
                                                    );
                                                })}
                                                {stockBottomSpacerHeight > 0 && (
                                                    <tr className="stocks-spacer-row" style={{ height: `${stockBottomSpacerHeight}px` }} aria-hidden="true">
                                                        <td
                                                            colSpan={activeStockColumns.length + 1}
                                                            style={{ height: `${stockBottomSpacerHeight}px` }}
                                                        />
                                                    </tr>
                                                )}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>
            )}

            {activeSection === 'indices' && !selectedIndex && (
                <div className="indices-view">
                    <div className="panel stocks-table-panel">
                        <div className="panel-header stocks-panel-header">
                            <div className="stocks-panel-header-top">
                                <div className="stocks-panel-title-group">
                                    <h2>Endeksler</h2>
                                    <span className="stocks-count-badge">
                                        {indicesLoading && !indices
                                            ? 'Yükleniyor…'
                                            : isIndicesFiltered
                                                ? `${sortedIndices.length} / ${indices?.rows.length || 0} endeks`
                                                : `${sortedIndices.length} endeks`}
                                    </span>
                                </div>
                                <div className="stocks-header-actions">
                                    <div className="stocks-search-wrap">
                                        <Search size={14} className="stocks-search-icon" aria-hidden="true" />
                                        <input
                                            type="text"
                                            placeholder="Endeks kodu veya adı ile ara..."
                                            value={searchTerm}
                                            onChange={(e) => setSearchTerm(e.target.value)}
                                            className="stocks-search-input"
                                        />
                                        {searchTerm && (
                                            <button
                                                type="button"
                                                className="stocks-search-clear"
                                                onClick={() => setSearchTerm('')}
                                                aria-label="Aramayı temizle"
                                            >
                                                <X size={12} aria-hidden="true" />
                                            </button>
                                        )}
                                    </div>
                                </div>
                            </div>
                            <div className="stocks-panel-toolbar">
                                <div className="stocks-toolbar-left">
                                    <div className="stocks-control-group">
                                        <span className="stocks-control-label">Kapsam</span>
                                        <div
                                            className="stocks-segment"
                                            aria-label="Endeks kapsamı"
                                            style={{
                                                '--item-count': INDEX_GROUP_OPTIONS.length,
                                                '--active-index': INDEX_GROUP_OPTIONS.findIndex((opt) => opt.id === indexGroup),
                                            } as React.CSSProperties}
                                        >
                                            {INDEX_GROUP_OPTIONS.map((option) => (
                                                <button
                                                    key={option.id}
                                                    type="button"
                                                    className={indexGroup === option.id ? 'active' : ''}
                                                    aria-pressed={indexGroup === option.id}
                                                    onClick={() => setIndexGroup(option.id)}
                                                >
                                                    {option.label}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                                {indices && (
                                    <div className="stocks-toolbar-meta">
                                        {indicesError && (
                                            <span className="stocks-status-pill is-danger" title={indicesError}>
                                                <AlertCircle size={12} aria-hidden="true" />
                                                <span>Yenileme başarısız</span>
                                            </span>
                                        )}
                                        <span className="stocks-meta-time" title="Son endeks güncellemesi">
                                            <Clock3 size={12} aria-hidden="true" />
                                            <span>{formatUpdateTime(indices.as_of)}</span>
                                        </span>
                                        <button
                                            type="button"
                                            className={`stocks-refresh-btn${indicesLoading ? ' is-spinning' : ''}`}
                                            onClick={() => loadIndices(false, true)}
                                            disabled={indicesLoading}
                                            title="Verileri yenile"
                                            aria-label="Endeks verilerini yenile"
                                        >
                                            <RefreshCw size={12} aria-hidden="true" />
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>

                        {indicesLoading && !indices && (
                            <div className="mv-table-skeleton" role="status" aria-live="polite" aria-label="Endeksler yükleniyor">
                                <div className="mv-table-skeleton-head is-indices" aria-hidden="true">
                                    <span className="mv-sk-cell is-xs stock-card-skeleton-pulse" />
                                    <span className="mv-sk-cell is-lg stock-card-skeleton-pulse" />
                                    <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                    <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                    <span className="mv-sk-cell is-md stock-card-skeleton-pulse" />
                                    <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                </div>
                                {Array.from({ length: 8 }, (_, idx) => (
                                    <div key={idx} className="mv-table-skeleton-row is-indices" aria-hidden="true">
                                        <span className="mv-sk-cell is-xs stock-card-skeleton-pulse" />
                                        <div className="mv-sk-symbol">
                                            <span className="mv-sk-avatar stock-card-skeleton-pulse" />
                                            <span className="mv-sk-cell is-lg stock-card-skeleton-pulse" />
                                        </div>
                                        <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                        <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                        <span className="mv-sk-cell is-md stock-card-skeleton-pulse" />
                                        <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                    </div>
                                ))}
                            </div>
                        )}

                        {indicesError && !indices && (
                            <div className="error-message">
                                <strong>Hata:</strong> {indicesError}
                                <button
                                    onClick={() => loadIndices(false, true)}
                                    className="btn-secondary"
                                    style={{ marginLeft: '1rem' }}
                                >
                                    Tekrar Dene
                                </button>
                            </div>
                        )}

                        {indices && (
                            <>
                                {filteredIndices.length === 0 ? (
                                    <div className="no-results">Aranan kritere uyan endeks bulunamadı.</div>
                                ) : (
                                    <div className="stocks-table-wrap indices-table-wrap">
                                        <table className="stocks-table indices-table" aria-rowcount={sortedIndices.length + 1}>
                                            <thead>
                                                <tr>
                                                    <th className="stocks-rank">#</th>
                                                    {INDEX_COLUMNS.map((column) => (
                                                        <th
                                                            key={column.key}
                                                            className={column.align === 'right' ? 'stocks-cell-right' : undefined}
                                                        >
                                                            <button
                                                                type="button"
                                                                className="stocks-sort-button"
                                                                onClick={() => handleIndexSort(column.key)}
                                                                aria-sort={
                                                                    indexSort.key === column.key
                                                                        ? indexSort.direction === 'asc'
                                                                            ? 'ascending'
                                                                            : 'descending'
                                                                        : 'none'
                                                                }
                                                            >
                                                                <span>{column.label}</span>
                                                                {indexSort.key === column.key && (
                                                                    <span className="stocks-sort-indicator">
                                                                        {indexSort.direction === 'asc' ? '↑' : '↓'}
                                                                    </span>
                                                                )}
                                                            </button>
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {sortedIndices.map((row, index) => (
                                                    <tr
                                                        key={row.symbol}
                                                        aria-rowindex={index + 2}
                                                        onClick={() => handleSelectIndex(row.symbol)}
                                                    >
                                                        <td className="stocks-rank">{index + 1}</td>
                                                        {INDEX_COLUMNS.map((column) => {
                                                            const rawVal = row[column.key];
                                                            const value = numericOrNull(rawVal);
                                                            const isPct =
                                                                column.key === 'change_pct' ||
                                                                String(column.key).startsWith('return_');
                                                            const isMuted = column.key !== 'symbol' && (
                                                                isPct ? value == null : rawVal == null
                                                            );
                                                            return (
                                                                <td
                                                                    key={column.key}
                                                                    className={[
                                                                        column.align === 'right' ? 'stocks-cell-right' : '',
                                                                        isPct ? getTableChangeClass(value) : '',
                                                                        isMuted ? 'stocks-cell-muted' : '',
                                                                        column.key === 'symbol' ? 'stocks-symbol-cell' : '',
                                                                        column.key === 'price' ? 'stocks-price-cell' : '',
                                                                        column.key === 'volume' ? 'stocks-volume-cell' : '',
                                                                    ].filter(Boolean).join(' ')}
                                                                >
                                                                    {column.key === 'symbol' ? (
                                                                        <span className="indices-symbol-main">
                                                                            <span className="indices-symbol-head">
                                                                                <SymbolLogo
                                                                                    symbol={row.symbol}
                                                                                    name={row.label}
                                                                                    kind="index"
                                                                                    size="xs"
                                                                                    className="indices-symbol-logo"
                                                                                />
                                                                                <span>{row.symbol}</span>
                                                                            </span>
                                                                            <small title={row.label}>{row.label}</small>
                                                                        </span>
                                                                    ) : (
                                                                        indexCellValue(row, column.key)
                                                                    )}
                                                                </td>
                                                            );
                                                        })}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>
            )}

            {activeSection === 'indices' && selectedIndex && (
                <div className="indices-detail-view">
                    <div className="indices-breadcrumb">
                        <button type="button" onClick={() => handleSelectIndex(null)}>Endeksler</button>
                        <span>/</span>
                        <strong>{selectedIndex}</strong>
                    </div>

                    {indexDetailLoading && !indexDetail && (
                        <div className="mv-index-detail-skeleton" role="status" aria-live="polite" aria-label="Endeks detayı yükleniyor">
                            <div className="mv-index-hero-skeleton" aria-hidden="true">
                                <div className="mv-index-hero-left">
                                    <span className="mv-index-logo-sk stock-card-skeleton-pulse" />
                                    <div className="mv-index-title-sk">
                                        <span className="mv-sk-cell is-md stock-card-skeleton-pulse" style={{ height: '24px' }} />
                                        <span className="mv-sk-cell is-lg stock-card-skeleton-pulse" />
                                    </div>
                                </div>
                                <div className="mv-index-hero-right">
                                    <span className="mv-sk-cell is-md stock-card-skeleton-pulse" style={{ height: '24px' }} />
                                    <span className="mv-sk-cell is-sm stock-card-skeleton-pulse" />
                                </div>
                            </div>
                            <div className="mv-index-returns-sk" aria-hidden="true">
                                {[0, 1, 2, 3].map((i) => (
                                    <span key={i} className="mv-index-return-card-sk stock-card-skeleton-pulse" />
                                ))}
                            </div>
                            <div className="mv-index-chart-sk stock-card-skeleton-pulse" aria-hidden="true" />
                        </div>
                    )}

                    {indexDetailError && !indexDetail && (
                        <div className="error-message">
                            <strong>Hata:</strong> {indexDetailError}
                            <button
                                onClick={() => loadIndexDetail(false, true, selectedIndex)}
                                className="btn-secondary"
                                style={{ marginLeft: '1rem' }}
                            >
                                Tekrar Dene
                            </button>
                        </div>
                    )}

                    {indexDetail && (
                        <>
                            <section className="indices-hero">
                                <div className="indices-hero-main">
                                    <SymbolLogo
                                        symbol={indexDetail.symbol}
                                        name={indexDetail.label}
                                        kind="index"
                                        size="lg"
                                        className="indices-logo"
                                    />
                                    <div>
                                        <h2>{indexDetail.symbol}</h2>
                                        <p>{indexDetail.label}</p>
                                    </div>
                                </div>
                                <div className="indices-hero-price">
                                    <strong>{formatIndexPrice(indexDetail.price)}</strong>
                                    <span className={getTableChangeClass(indexDetail.change_pct)}>
                                        {formatTablePct(indexDetail.change_pct)}
                                    </span>
                                    <small>{formatDateTime(indexDetail.as_of)}</small>
                                </div>
                            </section>

                            <div className="indices-stat-row">
                                <span>Yüksek: <strong>{formatIndexPrice(indexDetail.high)}</strong></span>
                                <span>Düşük: <strong>{formatIndexPrice(indexDetail.low)}</strong></span>
                                <span>Hacim: <strong>{formatVolume(indexDetail.volume)}</strong></span>
                                <span>Önc.Kap.: <strong>{formatIndexPrice(indexDetail.prev_close)}</strong></span>
                            </div>

                            <div className="indices-return-strip">
                                {DETAIL_RETURN_KEYS.map((item) => {
                                    const value = numericOrNull(indexDetail[item.key]);
                                    return (
                                        <div key={item.key} className="indices-return-card">
                                            <span>{item.label}</span>
                                            <strong className={getTableChangeClass(value)}>{formatTablePct(value)}</strong>
                                        </div>
                                    );
                                })}
                            </div>

                            <section className="indices-chart-panel">
                                <IndexLineChart
                                    symbol={indexDetail.symbol}
                                    points={indexDetail.line_points}
                                    prevClose={indexDetail.prev_close}
                                    changePct={indexDetail.change_pct}
                                />
                            </section>

                            <section className="indices-impact-panel">
                                <div className="indices-impact-head">
                                    <div className="stocks-panel-title-group">
                                        <h3>{indexDetail.symbol} Endeksindeki Şirketler ve Etkileri</h3>
                                        <span className="stocks-count-badge">
                                            {indexDetail.constituents.length} hisse · {indexDetail.weight_status === 'available' ? 'ağırlık hesaplandı' : 'ağırlık bekleniyor'}
                                        </span>
                                    </div>
                                    {indexDetailError && (
                                        <span className="stocks-status-pill is-danger">
                                            <AlertCircle size={12} aria-hidden="true" />
                                            <span>Son yenileme başarısız</span>
                                        </span>
                                    )}
                                </div>

                                {indexDetail.weight_note && (
                                    <div className="indices-weight-note">{indexDetail.weight_note}</div>
                                )}

                                {indexDetail.weight_status === 'available' && (
                                    <>
                                        <div className="indices-treemap-scale" aria-hidden="true">
                                            <div className="indices-treemap-scale-labels">
                                                <span>{formatTreemapScalePct(-maxHeatmapMovePct)}</span>
                                                <span>0%</span>
                                                <span>{formatTreemapScalePct(maxHeatmapMovePct)}</span>
                                            </div>
                                            <div className="indices-treemap-scale-bar" />
                                        </div>
                                        <div className="indices-treemap">
                                            {heatmapTiles.map((tile) => {
                                                const row = tile.row;
                                                const impactPct = tile.impactPct;
                                                const scaledMove = getTreemapTileScale(row);
                                                const showChange = tile.areaPct >= 1.05 && tile.width >= 5 && tile.height >= 4.5;
                                                const showSymbol = tile.areaPct >= 0.45 && tile.width >= 3.5 && tile.height >= 3;
                                                const tileDetail = `${row.symbol} · Değişim ${formatHeatmapChangePct(tile.changePct)} · Ağırlık ${formatWeight(row.weight_pct)} · Endeks etkisi ${formatImpactPct(impactPct)} · ${formatPointEffectShort(tile.pointEffect)}`;
                                                const tileDensityClass = tile.areaPct >= 14
                                                    ? 'is-hero'
                                                    : showChange
                                                    ? 'is-full'
                                                    : showSymbol
                                                        ? 'is-label-only'
                                                        : 'is-tiny';
                                                return (
                                                    <div
                                                        key={row.symbol}
                                                        className={`indices-tree-tile ${tileDensityClass} ${getTableChangeClass(tile.changePct)}`}
                                                        style={{
                                                            left: `${tile.left}%`,
                                                            top: `${tile.top}%`,
                                                            width: `${tile.width}%`,
                                                            height: `${tile.height}%`,
                                                            backgroundColor: getTreemapTileColor(tile.changePct, scaledMove),
                                                            fontSize: getHeatmapTileFontSize(tile),
                                                        }}
                                                        title={tileDetail}
                                                        aria-label={tileDetail}
                                                    >
                                                        {showSymbol && <strong>{row.symbol}</strong>}
                                                        {showChange && (
                                                            <span className="indices-tree-impact-pct">{formatHeatmapChangePct(tile.changePct)}</span>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </>
                                )}

                                <div className="indices-impact-summary">
                                    <div className="indices-impact-counts">
                                        <span className="indices-impact-count stocks-up">
                                            <span className="indices-impact-dot" />
                                            {positiveConstituents} pozitif
                                        </span>
                                        <span className="indices-impact-count stocks-flat">
                                            <span className="indices-impact-dot" />
                                            {neutralConstituents} nötr
                                        </span>
                                        <span className="indices-impact-count stocks-down">
                                            <span className="indices-impact-dot" />
                                            {negativeConstituents} negatif
                                        </span>
                                    </div>
                                    <div className="indices-net-impact">
                                        <span>{indexDetail.symbol} tahmini katkı:</span>
                                        <strong className={getTableChangeClass(netPointEffect)}>
                                            {formatPointEffectShort(netPointEffect)}
                                        </strong>
                                        <span>·</span>
                                        <strong className={getTableChangeClass(netImpactPct)}>
                                            {formatImpactPct(netImpactPct)}
                                        </strong>
                                    </div>
                                </div>

                                <div className="stocks-table-wrap indices-constituent-wrap" ref={indexConstituentTableWrapCallbackRef}>
                                    <table
                                        className="stocks-table indices-constituent-table"
                                        aria-rowcount={sortedIndexConstituents.length + 1}
                                    >
                                        <thead>
                                            <tr>
                                                {INDEX_CONSTITUENT_COLUMNS.map((column) => {
                                                    const isActive = indexConstituentSort.key === column.key;
                                                    return (
                                                        <th
                                                            key={column.key}
                                                            className={column.align === 'right' ? 'stocks-cell-right' : undefined}
                                                            scope="col"
                                                            aria-sort={
                                                                isActive
                                                                    ? indexConstituentSort.direction === 'asc'
                                                                        ? 'ascending'
                                                                        : 'descending'
                                                                    : 'none'
                                                            }
                                                        >
                                                            <button
                                                                type="button"
                                                                className="stocks-sort-button"
                                                                onClick={() => handleIndexConstituentSort(column.key)}
                                                            >
                                                                <span>{column.label}</span>
                                                                {isActive && (
                                                                    <span className="stocks-sort-indicator">
                                                                        {indexConstituentSort.direction === 'asc' ? '↑' : '↓'}
                                                                    </span>
                                                                )}
                                                            </button>
                                                        </th>
                                                    );
                                                })}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {indexConstituentTopSpacerHeight > 0 && (
                                                <tr className="stocks-spacer-row" style={{ height: `${indexConstituentTopSpacerHeight}px` }} aria-hidden="true">
                                                    <td
                                                        colSpan={INDEX_CONSTITUENT_COLUMNS.length}
                                                        style={{ height: `${indexConstituentTopSpacerHeight}px` }}
                                                    />
                                                </tr>
                                            )}
                                            {visibleIndexConstituents.map((row, index) => (
                                                <tr
                                                    key={row.symbol}
                                                    aria-rowindex={indexConstituentStartIndex + index + 2}
                                                    onClick={(event: ReactMouseEvent<HTMLTableRowElement>) => {
                                                        if (event.button !== 0) return;
                                                        onCompanyClick(row.symbol);
                                                    }}
                                                    onAuxClick={(event: ReactMouseEvent<HTMLTableRowElement>) => {
                                                        if (event.button !== 1) return;
                                                        if ((event.target as Element).closest('a[href]')) return;
                                                        event.preventDefault();
                                                        window.open(toStockDetail(row.symbol), '_blank', 'noopener,noreferrer');
                                                    }}
                                                >
                                                    <td className="stocks-symbol-cell">
                                                        <a
                                                            className="stocks-symbol-link"
                                                            href={toStockDetail(row.symbol)}
                                                            onClick={(event) => {
                                                                event.stopPropagation();
                                                                if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                                                                event.preventDefault();
                                                                onCompanyClick(row.symbol);
                                                            }}
                                                        >
                                                            <span className="stocks-symbol-main">
                                                                <SymbolLogo
                                                                    symbol={row.symbol}
                                                                    name={row.symbol}
                                                                    kind="stock"
                                                                    logoUrl={row.logo_url}
                                                                    size="xs"
                                                                    className="stocks-symbol-logo"
                                                                />
                                                                <span>{row.symbol}</span>
                                                            </span>
                                                        </a>
                                                    </td>
                                                    <td data-label="Fiyat" className="stocks-cell-right stocks-price-cell">{constituentPrice(row)}</td>
                                                    <td data-label="Değişim" className={`stocks-cell-right ${getTableChangeClass(row.change_pct)}`}>
                                                        {formatTablePct(row.change_pct)}
                                                    </td>
                                                    <td data-label="Hacim" className="stocks-cell-right stocks-volume-cell">{formatVolume(row.volume)}</td>
                                                    <td data-label="Ağırlık" className="stocks-cell-right">{formatWeight(row.weight_pct)}</td>
                                                    <td data-label="Puan" className={`stocks-cell-right ${getTableChangeClass(row.point_effect)}`}>
                                                        {formatPointEffect(row.point_effect)}
                                                    </td>
                                                    <td data-label="Endeks etkisi" className={`stocks-cell-right ${getTableChangeClass(getImpactPct(row, indexImpactLevel))}`}>
                                                        {formatImpactPct(getImpactPct(row, indexImpactLevel))}
                                                    </td>
                                                </tr>
                                            ))}
                                            {indexConstituentBottomSpacerHeight > 0 && (
                                                <tr className="stocks-spacer-row" style={{ height: `${indexConstituentBottomSpacerHeight}px` }} aria-hidden="true">
                                                    <td
                                                        colSpan={INDEX_CONSTITUENT_COLUMNS.length}
                                                        style={{ height: `${indexConstituentBottomSpacerHeight}px` }}
                                                    />
                                                </tr>
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </section>
                        </>
                    )}
                </div>
            )}
                </div>
                {activeSection === 'markets' && (
                    <MarketWatchRail
                        xu100Rows={market?.rows ?? []}
                        stockQuoteRows={stocks?.rows ?? []}
                        stockCardItems={stockCards?.items ?? []}
                        onSelectTicker={onCompanyClick}
                        onSelectFund={(fundCode) => onOpenFund?.(fundCode)}
                        mobilePanelOpen={isMobileViewport ? mobileMarketPanelOpen : undefined}
                        onMobilePanelOpenChange={isMobileViewport ? setMobileMarketPanelOpen : undefined}
                        onCollapseChange={setRailCollapsed}
                    />
                )}
            </div>
        </div>
    );
}
