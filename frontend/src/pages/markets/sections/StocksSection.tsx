import type { CSSProperties } from 'react';
import { AlertCircle, Clock3, RefreshCw, Search, SlidersHorizontal, X } from 'lucide-react';
import type { MarketStockIndex, MarketStockRow, MarketStocksResponse } from '../../../api/types';
import type { StockReturnMode } from '../../../routing/routes';
import { StockScreenerModal } from '../components/StockScreenerModal';
import { StockTableRow } from '../components/StockTableRow';
import {
    RETURN_MODE_OPTIONS,
    STOCK_INDEX_OPTIONS,
    formatUpdateTime,
    getColumnLabel,
} from '../stockScreenerConfig';
import type {
    ScreenerActiveChip,
    ScreenerComputedData,
    SortDirection,
    StockScreenerCategory,
    StockScreenerFilterState,
    StockScreenerPresetId,
    StockScreenerRangeKey,
    StockSortKey,
    StockTableColumn,
    StockTableMode,
} from '../types';

export interface StocksSectionProps {
    stocks: MarketStocksResponse | null;
    stocksLoading: boolean;
    stocksError: string | null;
    searchTerm: string;
    stockIndex: MarketStockIndex;
    stockTableMode: StockTableMode;
    returnMode: StockReturnMode;
    activeBenchmarkIndex: MarketStockIndex | null;
    stockSort: { key: StockSortKey; direction: SortDirection };
    sortedStocks: MarketStockRow[];
    visibleStocks: MarketStockRow[];
    totalStocksCount: number;
    isStocksFiltered: boolean;
    activeStockColumns: StockTableColumn[];
    stockTableTotalWidth: number;
    stockStartIndex: number;
    stockTopSpacerHeight: number;
    stockBottomSpacerHeight: number;
    stockTableWrapCallbackRef: (node: HTMLDivElement | null) => void;
    screenerFilters: StockScreenerFilterState;
    screenerCategory: StockScreenerCategory;
    activeScreenerPreset: StockScreenerPresetId | null;
    screenerModalOpen: boolean;
    activeFilterCount: number;
    isScreenerActive: boolean;
    screenerCategoryCounts: Record<StockScreenerCategory, number>;
    screenerActiveChips: ScreenerActiveChip[];
    screenerData: ScreenerComputedData;
    onSearchTermChange: (value: string) => void;
    onStockIndexChange: (index: MarketStockIndex) => void;
    onStockTableModeChange: (mode: StockTableMode) => void;
    onReturnModeChange: (mode: StockReturnMode) => void;
    onRefreshStocks: () => void;
    onStockSort: (key: StockSortKey) => void;
    onCompanyClick: (ticker: string) => void;
    onOpenScreenerModal: () => void;
    onCloseScreenerModal: () => void;
    onScreenerCategoryChange: (category: StockScreenerCategory) => void;
    onScreenerFilterChange: <K extends keyof StockScreenerFilterState>(key: K, value: StockScreenerFilterState[K]) => void;
    onClearScreenerRangeSpec: (minKey: StockScreenerRangeKey, maxKey: StockScreenerRangeKey) => void;
    onApplyScreenerPreset: (presetId: StockScreenerPresetId) => void;
    onClearScreenerFilters: () => void;
}

export function StocksSection({
    stocks,
    stocksLoading,
    stocksError,
    searchTerm,
    stockIndex,
    stockTableMode,
    returnMode,
    activeBenchmarkIndex,
    stockSort,
    sortedStocks,
    visibleStocks,
    totalStocksCount,
    isStocksFiltered,
    activeStockColumns,
    stockTableTotalWidth,
    stockStartIndex,
    stockTopSpacerHeight,
    stockBottomSpacerHeight,
    stockTableWrapCallbackRef,
    screenerFilters,
    screenerCategory,
    activeScreenerPreset,
    screenerModalOpen,
    activeFilterCount,
    isScreenerActive,
    screenerCategoryCounts,
    screenerActiveChips,
    screenerData,
    onSearchTermChange,
    onStockIndexChange,
    onStockTableModeChange,
    onReturnModeChange,
    onRefreshStocks,
    onStockSort,
    onCompanyClick,
    onOpenScreenerModal,
    onCloseScreenerModal,
    onScreenerCategoryChange,
    onScreenerFilterChange,
    onClearScreenerRangeSpec,
    onApplyScreenerPreset,
    onClearScreenerFilters,
}: StocksSectionProps) {
    return (
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
                                    onChange={(e) => onSearchTermChange(e.target.value)}
                                    className="stocks-search-input"
                                />
                                {searchTerm && (
                                    <button
                                        type="button"
                                        className="stocks-search-clear"
                                        onClick={() => onSearchTermChange('')}
                                        aria-label="Aramayı temizle"
                                    >
                                        <X size={12} aria-hidden="true" />
                                    </button>
                                )}
                            </div>
                            <button
                                type="button"
                                className={`stocks-screener-trigger-btn${isScreenerActive ? ' is-active' : ''}`}
                                onClick={onOpenScreenerModal}
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
                                        '--active-index': STOCK_INDEX_OPTIONS.indexOf(stockIndex),
                                    } as CSSProperties}
                                >
                                    {STOCK_INDEX_OPTIONS.map((option) => (
                                        <button
                                            key={option}
                                            type="button"
                                            className={stockIndex === option ? 'active' : ''}
                                            aria-pressed={stockIndex === option}
                                            onClick={() => onStockIndexChange(option)}
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
                                    } as CSSProperties}
                                >
                                    {([['performance', 'Getiri'], ['fundamentals', 'Temel Analiz']] as const).map(([mode, label]) => (
                                        <button
                                            key={mode}
                                            type="button"
                                            className={stockTableMode === mode ? 'active' : ''}
                                            aria-pressed={stockTableMode === mode}
                                            onClick={() => onStockTableModeChange(mode)}
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
                                            '--active-index': RETURN_MODE_OPTIONS.findIndex((o) => o.id === returnMode),
                                        } as CSSProperties}
                                    >
                                        {RETURN_MODE_OPTIONS.map((option) => (
                                            <button
                                                key={option.id}
                                                type="button"
                                                className={returnMode === option.id ? 'active' : ''}
                                                aria-pressed={returnMode === option.id}
                                                onClick={() => onReturnModeChange(option.id)}
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
                                    onClick={onRefreshStocks}
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
                                onClick={onOpenScreenerModal}
                            >
                                Filtreleri Düzenle
                            </button>
                            <button
                                type="button"
                                className="stocks-screener-banner-btn stocks-screener-banner-clear"
                                onClick={onClearScreenerFilters}
                            >
                                <X size={12} aria-hidden="true" /> Temizle
                            </button>
                        </div>
                    </div>
                )}

                <StockScreenerModal
                    isOpen={screenerModalOpen}
                    stockIndex={stockIndex}
                    screenerFilters={screenerFilters}
                    screenerCategory={screenerCategory}
                    activeScreenerPreset={activeScreenerPreset}
                    isScreenerActive={isScreenerActive}
                    screenerCategoryCounts={screenerCategoryCounts}
                    screenerActiveChips={screenerActiveChips}
                    screenerData={screenerData}
                    onClose={onCloseScreenerModal}
                    onCategoryChange={onScreenerCategoryChange}
                    onFilterChange={onScreenerFilterChange}
                    onClearRangeSpec={onClearScreenerRangeSpec}
                    onApplyPreset={onApplyScreenerPreset}
                    onClearAll={onClearScreenerFilters}
                />

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
                            onClick={onRefreshStocks}
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
                                                        onClick={() => onStockSort(column.key)}
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
    );
}
