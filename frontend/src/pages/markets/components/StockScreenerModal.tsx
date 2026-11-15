import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import {
    Activity,
    BarChart3,
    Building2,
    Check,
    Coins,
    History,
    Percent,
    RotateCcw,
    TrendingUp,
    X,
} from 'lucide-react';
import type { MarketStockIndex } from '../../../api/types';
import {
    SCREENER_RANGE_SPECS,
    SCREENER_TOGGLE_SPECS,
    STOCK_SCREENER_PRESETS,
    formatFundamentalRatio,
    parseScreenerNumber,
} from '../stockScreenerConfig';
import type {
    ScreenerActiveChip,
    ScreenerComputedData,
    ScreenerRangeSpec,
    StockScreenerCategory,
    StockScreenerFilterState,
    StockScreenerPresetId,
    StockScreenerRangeKey,
} from '../types';
import './StockScreenerModal.css';

export interface StockScreenerModalProps {
    isOpen: boolean;
    stockIndex: MarketStockIndex;
    screenerFilters: StockScreenerFilterState;
    screenerCategory: StockScreenerCategory;
    activeScreenerPreset: StockScreenerPresetId | null;
    isScreenerActive: boolean;
    screenerCategoryCounts: Record<StockScreenerCategory, number>;
    screenerActiveChips: ScreenerActiveChip[];
    screenerData: ScreenerComputedData;
    onClose: () => void;
    onCategoryChange: (category: StockScreenerCategory) => void;
    onFilterChange: <K extends keyof StockScreenerFilterState>(key: K, value: StockScreenerFilterState[K]) => void;
    onClearRangeSpec: (minKey: StockScreenerRangeKey, maxKey: StockScreenerRangeKey) => void;
    onApplyPreset: (presetId: StockScreenerPresetId) => void;
    onClearAll: () => void;
}

export function StockScreenerModal({
    isOpen,
    stockIndex,
    screenerFilters,
    screenerCategory,
    activeScreenerPreset,
    isScreenerActive,
    screenerCategoryCounts,
    screenerActiveChips,
    screenerData,
    onClose,
    onCategoryChange,
    onFilterChange,
    onClearRangeSpec,
    onApplyPreset,
    onClearAll,
}: StockScreenerModalProps) {
    useEffect(() => {
        if (!isOpen) return;
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') onClose();
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isOpen, onClose]);

    if (!isOpen) return null;

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
                                    onClick={() => onClearRangeSpec(spec.minKey, spec.maxKey)}
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
                                    onChange={(e) => onFilterChange(spec.minKey, e.currentTarget.value)}
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
                                    onChange={(e) => onFilterChange(spec.maxKey, e.currentTarget.value)}
                                />
                                <span className="screener-num-unit">{spec.unit}</span>
                            </label>
                        </div>
                    </div>
                );
            })
    );

    return createPortal(
        <div
            className="screener-modal-backdrop"
            role="dialog"
            aria-modal="true"
            aria-labelledby="screener-modal-title"
            onClick={onClose}
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
                        onClick={onClose}
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
                                    onClick={() => onCategoryChange(id)}
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
                                    onClick={() => onApplyPreset(preset.id)}
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
                                            onChange={(e) => onFilterChange('sector', e.currentTarget.value)}
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
                                                        onChange={(e) => onFilterChange(key, e.currentTarget.checked)}
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
                                                        onChange={(e) => onFilterChange(key, e.currentTarget.checked)}
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
                        onClick={onClearAll}
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
                            onClick={onClose}
                        >
                            Sonuçları Göster
                        </button>
                    </div>
                </div>
            </div>
        </div>,
        document.body,
    );
}
