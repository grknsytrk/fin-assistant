import type {
    MarketIndexConstituent,
    MarketIndexLinePoint,
    MarketIndexListRow,
    MarketReturnBenchmark,
    MarketStockCardChartRange,
    MarketStockCardItem,
    MarketStockCardsResponse,
    MarketStockIndex,
    MarketStockRow,
    MarketStocksResponse,
    MarketUniverseResponse,
} from '../../api/types';
import type { StockReturnMode } from '../../routing/routes';
import type {
    HeatmapLayoutItem,
    HeatmapLayoutRect,
    HeatmapTile,
    IndexConstituentSortKey,
    IndexGroupFilter,
    ScreenerComputedData,
    ScreenerRangeSpec,
    StockReturnKey,
    StockScreenerFilterState,
    StockScreenerPresetId,
    StockScreenerToggleKey,
    StockSectorDiffFields,
    StockSortKey,
    StockTableColumn,
} from './types';

export const SCREENER_RANGE_SPECS: ScreenerRangeSpec[] = [
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

export const SCREENER_TOGGLE_SPECS: Array<{ key: StockScreenerToggleKey; label: string; category: 'profile' | 'historical' }> = [
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

export const EMPTY_STOCK_SCREENER_FILTERS: StockScreenerFilterState = {
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

export const STOCK_COLUMNS: StockTableColumn[] = [
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

export const STOCK_FUNDAMENTAL_COLUMNS: StockTableColumn[] = [
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

export const STOCK_SCREENER_PRESETS: Array<{
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

export const STOCK_INDEX_OPTIONS: MarketStockIndex[] = ['XUTUM', 'XU100', 'XU030'];

export const PRIMARY_MARKET_INDEX_RANK = new Map<string, number>(
    ['XU100', 'XU030', 'XUTUM'].map((symbol, index) => [symbol, index]),
);

export const RETURN_MODE_OPTIONS: Array<{ id: StockReturnMode; label: string }> = [
    { id: 'absolute', label: 'Mutlak' },
    { id: 'relative_xu100', label: "XU100'a göre" },
    { id: 'relative_xu030', label: "XU030'a göre" },
];

export const RETURN_KEYS: StockReturnKey[] = [
    'return_1w_pct',
    'return_1m_pct',
    'return_3m_pct',
    'return_6m_pct',
    'return_ytd_pct',
    'return_1y_pct',
];

export const STOCK_CARD_STORAGE_KEY = 'ragfin_market_stock_cards';
export const MAX_STOCK_CARDS = 12;
// The first desktop row becomes useful as soon as possible. Remaining cards
// are fetched in small batches so a slow quote/valuation provider cannot hold
// the whole panel behind one spinner.
export const STOCK_CARD_INITIAL_BATCH_SIZE = 3;
export const STOCK_CARD_BACKGROUND_BATCH_SIZE = 3;
export const STOCK_CARD_BACKGROUND_DELAY_MS = 120;
export const MARKET_QUOTE_REFRESH_MS = 3000;
export const MARKET_LIST_REFRESH_DESKTOP_MS = MARKET_QUOTE_REFRESH_MS;
export const MARKET_LIST_REFRESH_MOBILE_MS = MARKET_QUOTE_REFRESH_MS;
export const MARKET_DETAIL_REFRESH_DESKTOP_MS = 15000;
export const MARKET_DETAIL_REFRESH_MOBILE_MS = 30000;

export const STOCK_CARD_CHART_RANGES: Array<{ id: MarketStockCardChartRange; label: string; title: string }> = [
    { id: '1d', label: 'G', title: 'Gün içi' },
    { id: '1w', label: '1H', title: '1 Hafta' },
    { id: '1m', label: '1A', title: '1 Ay' },
    { id: 'ytd', label: 'YTD', title: 'Yılbaşından bugüne' },
    { id: '1y', label: '1Y', title: '1 Yıl' },
    { id: '5y', label: '5Y', title: '5 Yıl' },
];

export const INDEX_COLUMNS: Array<{ key: keyof MarketIndexListRow; label: string; align?: 'left' | 'right' }> = [
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

export const INDEX_GROUP_OPTIONS: Array<{ id: IndexGroupFilter; label: string }> = [
    { id: 'all', label: 'Tümü' },
    { id: 'main', label: 'Ana Endeksler' },
    { id: 'sector', label: 'Sektör' },
];

export const MAIN_MARKET_INDICES = new Set<string>([
    'XUTUM',
    'XU100',
    'XU030',
    'XUSIN',
    'XUHIZ',
    'XUMAL',
    'XUTEK',
]);

export const INDEX_CONSTITUENT_COLUMNS: Array<{ key: Exclude<IndexConstituentSortKey, 'impact_abs'>; label: string; align?: 'left' | 'right' }> = [
    { key: 'symbol', label: 'Şirket', align: 'left' },
    { key: 'price', label: 'Son Fiyat', align: 'right' },
    { key: 'change_pct', label: '%', align: 'right' },
    { key: 'volume', label: 'Hacim', align: 'right' },
    { key: 'weight_pct', label: 'Endeks Ağırlığı', align: 'right' },
    { key: 'point_effect', label: 'Puan Etkisi', align: 'right' },
    { key: 'impact_pct', label: 'Etki %', align: 'right' },
];

export const DETAIL_RETURN_KEYS: Array<{ key: keyof MarketIndexListRow; label: string }> = [
    { key: 'change_pct', label: 'Gün içi' },
    { key: 'return_1w_pct', label: '1 Hafta' },
    { key: 'return_1m_pct', label: '1 Ay' },
    { key: 'return_ytd_pct', label: 'YTA' },
    { key: 'return_6m_pct', label: '6 Ay' },
    { key: 'return_1y_pct', label: '1 Yıl' },
    { key: 'return_5y_pct', label: '5 Yıl' },
];

export const STOCK_CARD_PERFORMANCE_KEYS: Array<{
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

export function formatStockPrice(row: MarketStockRow): string {
    if (row.price == null) return '—';
    const currencyPrefix = row.price_currency && row.price_currency !== 'TRY' ? `${row.price_currency} ` : '';
    return `${currencyPrefix}${row.price.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

export function formatIndexPrice(value: number | null): string {
    if (value == null) return '—';
    return value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

export function formatMaybeCurrency(value: number | null, currency?: string | null): string {
    if (value == null) return '—';
    const prefix = currency && currency !== 'TRY' ? `${currency} ` : '';
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

export function formatTablePct(value: number | null): string {
    if (value == null) return '—';
    const sign = value > 0 ? '+' : '';
    return `%${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

export function formatVolume(value: number | null): string {
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

export function formatMarketCap(value: number | null): string {
    if (value == null) return '—';
    return formatVolume(value);
}

export function formatFundamentalRatio(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '—';
    return `${value.toLocaleString('tr-TR', { maximumFractionDigits: 2 })}x`;
}

export function formatFundamentalPercent(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '—';
    return `%${value.toLocaleString('tr-TR', { maximumFractionDigits: 2 })}`;
}

export function parseScreenerNumber(raw: string): number | undefined {
    const text = String(raw || '').trim().replace(',', '.');
    if (!text) return undefined;
    const value = Number(text);
    return Number.isFinite(value) ? value : undefined;
}

export function hasStockFinancialData(row: MarketStockRow): boolean {
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

export function formatCardCurrency(value: number | null | undefined, currency?: string | null): string {
    if (value == null) return '-';
    const prefix = currency && currency !== 'TRY' ? `${currency} ` : '₺';
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

export function formatCardPct(value: number | null | undefined): string {
    if (value == null) return '% -';
    const sign = value > 0 ? '+' : '';
    return `%${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

export function formatCardRatio(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

export function formatCardPositiveRatio(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value) || value <= 0) return '-';
    return formatCardRatio(value);
}

export function hasStockCardLoadedData(item: MarketStockCardItem | null | undefined): boolean {
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

export function formatCardFullNumber(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return value.toLocaleString('tr-TR', { maximumFractionDigits: 0 });
}

export function formatCardFullCurrency(value: number | null | undefined, currency?: string | null): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const prefix = currency && currency !== 'TRY' ? `${currency} ` : '₺';
    return `${prefix}${formatCardFullNumber(value)}`;
}

export function istanbulDateKey(date: Date): string {
    if (Number.isNaN(date.getTime())) return '';
    return new Intl.DateTimeFormat('en-CA', {
        timeZone: 'Europe/Istanbul',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
    }).format(date);
}

export function formatStockCardChartDate(iso: string | null | undefined, range: MarketStockCardChartRange): string {
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

export function formatStockCardAxisDate(iso: string | null | undefined, range: MarketStockCardChartRange): string {
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

export function formatStockCardAxisTime(date: Date): string {
    if (Number.isNaN(date.getTime())) return '';
    return new Intl.DateTimeFormat('tr-TR', {
        timeZone: 'Europe/Istanbul',
        hour: '2-digit',
        minute: '2-digit',
    }).format(date);
}

export function formatUpdateTime(iso: string | null | undefined): string {
    if (!iso) return '--:--';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '--:--';
    return dt.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function formatStockCardTradeTime(iso: string | null | undefined): string {
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

export function latestStockCardPointTime(points: MarketIndexLinePoint[] | null | undefined): string | null {
    const valid = (points || [])
        .map((point) => point.time)
        .filter((time): time is string => Boolean(time))
        .sort();
    return valid[valid.length - 1] || null;
}

export function isPreviousIstanbulDate(iso: string | null | undefined): boolean {
    if (!iso) return false;
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return false;
    return istanbulDateKey(dt) < istanbulDateKey(new Date());
}

export function formatDateTime(iso: string | null | undefined): string {
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

export function normalizeStockCardSymbol(raw: string): string {
    return raw.trim().toUpperCase().replace(/\.IS$/, '');
}

export function readStoredStockCards(): string[] {
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

export function mobileSparklinePath(points: MarketIndexLinePoint[]): string {
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

export function stockCardsMemoryKey(symbols: string[]): string {
    return symbols.join(',');
}

export function emptyMarketReturnBenchmark(): MarketReturnBenchmark {
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

export function quickStocksFromUniverse(
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

export function mergeStockCardPayloads(
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

export function splitStockCardBatches(symbols: string[], size: number): string[][] {
    const batches: string[][] = [];
    for (let index = 0; index < symbols.length; index += size) {
        batches.push(symbols.slice(index, index + size));
    }
    return batches;
}

export function getTableChangeClass(value: number | null): string {
    if (value == null || value === 0) return 'stocks-flat';
    return value > 0 ? 'stocks-up' : 'stocks-down';
}

export function numericOrNull(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function indexCellValue(row: MarketIndexListRow, key: keyof MarketIndexListRow): string {
    if (key === 'symbol') return row.symbol;
    if (key === 'price') return formatIndexPrice(row.price);
    if (key === 'volume') return formatVolume(row.volume);
    if (String(key).includes('return') || key === 'change_pct') {
        return formatTablePct(numericOrNull(row[key]));
    }
    const value = row[key];
    return value == null ? '-' : String(value);
}

export function constituentPrice(row: MarketIndexConstituent): string {
    return formatMaybeCurrency(row.price, row.price_currency);
}

export function formatWeight(value: number | null): string {
    if (value == null) return '-';
    return `% ${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

export function formatPointEffect(value: number | null): string {
    if (value == null) return '-';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

export function formatPointEffectShort(value: number | null): string {
    const formatted = formatPointEffect(value);
    return formatted === '-' ? formatted : `${formatted}p`;
}

export function getImpactPct(row: MarketIndexConstituent, indexLevel: number | null): number | null {
    if (row.point_effect == null || indexLevel == null || indexLevel <= 0) return null;
    return (row.point_effect / indexLevel) * 100;
}

export function formatImpactPct(value: number | null): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}%`;
}

export function formatHeatmapChangePct(value: number | null): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}%`;
}

export function isReturnKey(key: StockSortKey): key is StockReturnKey {
    return RETURN_KEYS.includes(key as StockReturnKey);
}

export function getBenchmarkIndex(returnMode: StockReturnMode): MarketStockIndex | null {
    if (returnMode === 'relative_xu100') return 'XU100';
    if (returnMode === 'relative_xu030') return 'XU030';
    return null;
}

export function getReturnValue(
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

export function getColumnLabel(key: StockSortKey, _returnMode: StockReturnMode): string {
    return STOCK_COLUMNS.find((column) => column.key === key)?.label || '';
}

export function stockSortValue(
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

export function constituentSortValue(
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

export const isBistSymbol = (sym?: string): boolean => {
    if (!sym) return true;
    const globalSymbols = ['SP500', 'NASDAQ', 'DOW', 'DAX', 'FTSE', 'CAC40', 'NIKKEI', 'HANGSENG', 'VIX', 'DXY'];
    if (globalSymbols.includes(sym.toUpperCase())) return false;
    if (sym.includes('/')) return false;
    return true;
};

export function emptyStockCardItem(symbol: string): MarketStockCardItem {
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

export const TREEMAP_LAYOUT_WIDTH = 100;
export const TREEMAP_LAYOUT_HEIGHT = 62;

export function treemapValue(row: MarketIndexConstituent): number {
    if (row.weight_pct != null && row.weight_pct > 0) return row.weight_pct;
    const impact = Math.abs(row.point_effect ?? 0);
    return impact > 0 ? impact : 0.01;
}

export function worstTreemapAspect(row: HeatmapLayoutItem[], shortSide: number): number {
    if (!row.length || shortSide <= 0) return Number.POSITIVE_INFINITY;
    const areas = row.map((item) => item.area);
    const sum = areas.reduce((total, area) => total + area, 0);
    const min = Math.min(...areas);
    const max = Math.max(...areas);
    if (sum <= 0 || min <= 0) return Number.POSITIVE_INFINITY;
    const sideSquared = shortSide * shortSide;
    return Math.max((sideSquared * max) / (sum * sum), (sum * sum) / (sideSquared * min));
}

export function heatmapTileFromRect(item: HeatmapLayoutItem, rect: HeatmapLayoutRect): HeatmapTile {
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
        areaPct: ((width * height) / (TREEMAP_LAYOUT_WIDTH * TREEMAP_LAYOUT_HEIGHT)) * 100,
    };
}

export function layoutTreemapRow(
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

export function buildHeatmapTiles(items: MarketIndexConstituent[], indexLevel: number | null): HeatmapTile[] {
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

export function formatTreemapScalePct(value: number): string {
    const rounded = Math.abs(value) >= 1 ? Math.round(value) : Number(value.toFixed(2));
    if (rounded === 0) return '0%';
    return `${rounded.toLocaleString('tr-TR', {
        minimumFractionDigits: Math.abs(rounded) >= 1 ? 0 : 2,
        maximumFractionDigits: Math.abs(rounded) >= 1 ? 0 : 2,
    })}%`;
}

export function getTreemapTileColor(changePct: number | null, scaledMove: number): string {
    if (changePct == null || changePct === 0) return 'hsl(146 10% 28%)';
    if (changePct > 0) {
        return `hsl(123 92% ${23 + scaledMove * 24}%)`;
    }
    return `hsl(0 40% ${25 + scaledMove * 18}%)`;
}

export function computeScreenerData(
    allRows: MarketStockRow[],
    screenerFilters: StockScreenerFilterState,
): ScreenerComputedData {
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
}
