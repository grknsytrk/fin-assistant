import { useEffect, useState, type CSSProperties, type MouseEvent as ReactMouseEvent } from 'react';
import { AlertCircle, ArrowUpRight, Clock3, RefreshCw, Search, X } from 'lucide-react';
import type {
    MarketIndexCode,
    MarketIndexConstituent,
    MarketIndexDetailResponse,
    MarketIndexListRow,
    MarketIndicesResponse,
} from '../../../api/types';
import { toStockDetail } from '../../../routing/routes';
import SymbolLogo from '../../../components/SymbolLogo';
import AnimatedChartSkeleton from '../../../components/charts/AnimatedChartSkeleton';
import { IndexLineChart } from '../components/IndexLineChart';
import {
    DETAIL_RETURN_KEYS,
    INDEX_COLUMNS,
    INDEX_CONSTITUENT_COLUMNS,
    INDEX_GROUP_OPTIONS,
    constituentPrice,
    formatDateTime,
    formatHeatmapChangePct,
    formatImpactPct,
    formatIndexPrice,
    formatPointEffect,
    formatPointEffectShort,
    formatTablePct,
    formatTreemapScalePct,
    formatUpdateTime,
    formatVolume,
    formatWeight,
    getImpactPct,
    getTableChangeClass,
    getTreemapTileColor,
    indexCellValue,
    numericOrNull,
} from '../stockScreenerConfig';
import type {
    HeatmapTile,
    IndexConstituentSortKey,
    IndexGroupFilter,
    IndexSortKey,
    SortDirection,
} from '../types';
import './IndicesSection.css';

export interface IndicesSectionProps {
    selectedIndex: MarketIndexCode | null;
    indices: MarketIndicesResponse | null;
    indicesLoading: boolean;
    indicesError: string | null;
    searchTerm: string;
    indexGroup: IndexGroupFilter;
    isIndicesFiltered: boolean;
    indexSort: { key: IndexSortKey; direction: SortDirection };
    filteredIndices: MarketIndexListRow[];
    sortedIndices: MarketIndexListRow[];
    indexDetail: MarketIndexDetailResponse | null;
    indexDetailLoading: boolean;
    indexDetailError: string | null;
    indexImpactLevel: number | null;
    positiveConstituents: number;
    neutralConstituents: number;
    negativeConstituents: number;
    sortedIndexConstituents: MarketIndexConstituent[];
    visibleIndexConstituents: MarketIndexConstituent[];
    indexConstituentSort: { key: IndexConstituentSortKey; direction: SortDirection };
    indexConstituentFilter?: 'all' | 'positive' | 'neutral' | 'negative';
    indexConstituentStartIndex: number;
    indexConstituentTopSpacerHeight: number;
    indexConstituentBottomSpacerHeight: number;
    indexConstituentTableWrapCallbackRef: (node: HTMLDivElement | null) => void;
    maxHeatmapMovePct: number;
    heatmapTiles: HeatmapTile[];
    netPointEffect: number | null;
    netImpactPct: number | null;
    onSearchTermChange: (value: string) => void;
    onIndexGroupChange: (group: IndexGroupFilter) => void;
    onRefreshIndices: () => void;
    onIndexSort: (key: IndexSortKey) => void;
    onSelectIndex: (index: MarketIndexCode | null) => void;
    onRetryIndexDetail: (index: MarketIndexCode) => void;
    onIndexConstituentSort: (key: IndexConstituentSortKey) => void;
    onIndexConstituentPreset?: (preset: 'all' | 'weight' | 'positive' | 'negative' | 'neutral') => void;
    onCompanyClick: (ticker: string) => void;
    getTreemapTileScale: (row: MarketIndexConstituent) => number;
    getHeatmapTileFontSize: (tile: HeatmapTile) => string;
}

export function IndicesSection({
    selectedIndex,
    indices,
    indicesLoading,
    indicesError,
    searchTerm,
    indexGroup,
    isIndicesFiltered,
    indexSort,
    filteredIndices,
    sortedIndices,
    indexDetail,
    indexDetailLoading,
    indexDetailError,
    indexImpactLevel,
    positiveConstituents,
    neutralConstituents,
    negativeConstituents,
    sortedIndexConstituents,
    visibleIndexConstituents,
    indexConstituentSort,
    indexConstituentFilter = 'all',
    indexConstituentStartIndex,
    indexConstituentTopSpacerHeight,
    indexConstituentBottomSpacerHeight,
    indexConstituentTableWrapCallbackRef,
    maxHeatmapMovePct,
    heatmapTiles,
    netPointEffect,
    netImpactPct,
    onSearchTermChange,
    onIndexGroupChange,
    onRefreshIndices,
    onIndexSort,
    onSelectIndex,
    onRetryIndexDetail,
    onIndexConstituentSort,
    onIndexConstituentPreset,
    onCompanyClick,
    getTreemapTileScale,
    getHeatmapTileFontSize,
}: IndicesSectionProps) {
    const [selectedHeatmapSymbol, setSelectedHeatmapSymbol] = useState<string | null>(null);

    useEffect(() => {
        setSelectedHeatmapSymbol(null);
    }, [selectedIndex]);

    const selectedHeatmapTile = selectedHeatmapSymbol
        ? heatmapTiles.find((tile) => tile.row.symbol === selectedHeatmapSymbol) || null
        : null;

    if (!selectedIndex) {
        return (
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
                                        } as CSSProperties}
                                    >
                                        {INDEX_GROUP_OPTIONS.map((option) => (
                                            <button
                                                key={option.id}
                                                type="button"
                                                className={indexGroup === option.id ? 'active' : ''}
                                                aria-pressed={indexGroup === option.id}
                                                onClick={() => onIndexGroupChange(option.id)}
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
                                        onClick={onRefreshIndices}
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
                                onClick={onRefreshIndices}
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
                                                            onClick={() => onIndexSort(column.key)}
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
                                                    onClick={() => onSelectIndex(row.symbol)}
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
        );
    }

    return (
        <div className="indices-detail-view">
            <div className="indices-breadcrumb">
                <button type="button" onClick={() => onSelectIndex(null)}>Endeksler</button>
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
                    <div className="mv-index-chart-sk">
                        <AnimatedChartSkeleton
                            seed={`index-${selectedIndex}`}
                            width={1120}
                            height={400}
                            padding={{ top: 30, right: 65, bottom: 40, left: 16 }}
                            ariaLabel={`${selectedIndex} endeks grafiği yükleniyor`}
                        />
                    </div>
                </div>
            )}

            {indexDetailError && !indexDetail && (
                <div className="error-message">
                    <strong>Hata:</strong> {indexDetailError}
                    <button
                        onClick={() => onRetryIndexDetail(selectedIndex)}
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
                            isLoading={indexDetailLoading}
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
                                        const isSelectedTile = selectedHeatmapSymbol === row.symbol;
                                        return (
                                            <div
                                                key={row.symbol}
                                                role="button"
                                                tabIndex={0}
                                                className={`indices-tree-tile ${tileDensityClass} ${getTableChangeClass(tile.changePct)}${isSelectedTile ? ' is-selected' : ''}`}
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
                                                aria-pressed={isSelectedTile}
                                                onClick={() => {
                                                    setSelectedHeatmapSymbol((current) => (current === row.symbol ? null : row.symbol));
                                                }}
                                                onKeyDown={(event) => {
                                                    if (event.key === 'Enter' || event.key === ' ') {
                                                        event.preventDefault();
                                                        setSelectedHeatmapSymbol((current) => (current === row.symbol ? null : row.symbol));
                                                    }
                                                }}
                                            >
                                                {showSymbol && <strong>{row.symbol}</strong>}
                                                {showChange && (
                                                    <span className="indices-tree-impact-pct">{formatHeatmapChangePct(tile.changePct)}</span>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                                {selectedHeatmapTile && (
                                    <div className="indices-heatmap-inspector" role="region" aria-label={`${selectedHeatmapTile.row.symbol} özet bilgisi`}>
                                        <div className="indices-heatmap-inspector-main">
                                            <div className="indices-heatmap-inspector-head">
                                                <strong>{selectedHeatmapTile.row.symbol}</strong>
                                                <span className={getTableChangeClass(selectedHeatmapTile.changePct)}>
                                                    {formatHeatmapChangePct(selectedHeatmapTile.changePct)}
                                                </span>
                                            </div>
                                            <div className="indices-heatmap-inspector-meta">
                                                <span>Ağırlık: <b>{formatWeight(selectedHeatmapTile.row.weight_pct)}</b></span>
                                                <span>Etki: <b className={getTableChangeClass(selectedHeatmapTile.impactPct)}>{formatImpactPct(selectedHeatmapTile.impactPct)}</b></span>
                                                <span>Puan: <b className={getTableChangeClass(selectedHeatmapTile.pointEffect)}>{formatPointEffectShort(selectedHeatmapTile.pointEffect)}</b></span>
                                            </div>
                                        </div>
                                        <div className="indices-heatmap-inspector-actions">
                                            <button
                                                type="button"
                                                className="indices-heatmap-inspector-go"
                                                onClick={() => onCompanyClick(selectedHeatmapTile.row.symbol)}
                                            >
                                                <span>Detay</span>
                                                <ArrowUpRight size={13} aria-hidden="true" />
                                            </button>
                                            <button
                                                type="button"
                                                className="indices-heatmap-inspector-close"
                                                onClick={() => setSelectedHeatmapSymbol(null)}
                                                aria-label="Seçimi kapat"
                                            >
                                                <X size={14} aria-hidden="true" />
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </>
                        )}

                        <div className="indices-impact-summary">
                            <div className="indices-impact-counts">
                                <button
                                    type="button"
                                    className={`indices-impact-count stocks-up${indexConstituentFilter === 'positive' ? ' is-active' : ''}`}
                                    aria-pressed={indexConstituentFilter === 'positive'}
                                    onClick={() => onIndexConstituentPreset?.('positive')}
                                >
                                    <span className="indices-impact-dot" />
                                    {positiveConstituents} pozitif
                                </button>
                                <button
                                    type="button"
                                    className={`indices-impact-count stocks-flat${indexConstituentFilter === 'neutral' ? ' is-active' : ''}`}
                                    aria-pressed={indexConstituentFilter === 'neutral'}
                                    onClick={() => onIndexConstituentPreset?.('neutral')}
                                >
                                    <span className="indices-impact-dot" />
                                    {neutralConstituents} nötr
                                </button>
                                <button
                                    type="button"
                                    className={`indices-impact-count stocks-down${indexConstituentFilter === 'negative' ? ' is-active' : ''}`}
                                    aria-pressed={indexConstituentFilter === 'negative'}
                                    onClick={() => onIndexConstituentPreset?.('negative')}
                                >
                                    <span className="indices-impact-dot" />
                                    {negativeConstituents} negatif
                                </button>
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
                            {onIndexConstituentPreset && (
                                <div className="indices-constituent-presets" role="group" aria-label="Bileşen sıralama ve filtre">
                                    <button
                                        type="button"
                                        className={`indices-preset-btn${indexConstituentFilter === 'all' && indexConstituentSort.key === 'impact_abs' ? ' is-active' : ''}`}
                                        onClick={() => onIndexConstituentPreset('all')}
                                    >
                                        En Çok Etkileyen
                                    </button>
                                    <button
                                        type="button"
                                        className={`indices-preset-btn${indexConstituentFilter === 'all' && indexConstituentSort.key === 'weight_pct' ? ' is-active' : ''}`}
                                        onClick={() => onIndexConstituentPreset('weight')}
                                    >
                                        En Ağırlar
                                    </button>
                                    <button
                                        type="button"
                                        className={`indices-preset-btn is-up${indexConstituentFilter === 'positive' ? ' is-active' : ''}`}
                                        onClick={() => onIndexConstituentPreset('positive')}
                                    >
                                        Pozitif Etki
                                    </button>
                                    <button
                                        type="button"
                                        className={`indices-preset-btn is-down${indexConstituentFilter === 'negative' ? ' is-active' : ''}`}
                                        onClick={() => onIndexConstituentPreset('negative')}
                                    >
                                        Negatif Etki
                                    </button>
                                </div>
                            )}
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
                                                        onClick={() => onIndexConstituentSort(column.key)}
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
    );
}
