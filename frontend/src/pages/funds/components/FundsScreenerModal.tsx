import { createPortal } from 'react-dom';
import { Building2, Percent, RotateCcw, TrendingUp, X } from 'lucide-react';
import { parseFundScreenerNumber } from '../formatters';
import {
    FUND_SCREENER_PRESETS,
    FUND_SCREENER_RANGE_SPECS,
    type FundScreenerFilterState,
    type FundScreenerPresetId,
    type FundScreenerRangeKey,
    type FundScreenerRangeSpec,
} from '../types';
import './FundsScreenerModal.css';

export type FundsScreenerActiveChip = {
    id: string;
    label: string;
    onClear: () => void;
};

export type FundsScreenerModalProps = {
    open: boolean;
    onClose: () => void;
    totalFundsCount: number;
    filteredFundsCount: number;
    activePreset: FundScreenerPresetId | null;
    onApplyPreset: (presetId: FundScreenerPresetId) => void;
    activeChips: FundsScreenerActiveChip[];
    fundTypeFilter: string;
    onFundTypeFilterChange: (value: string) => void;
    riskFilter: string;
    onRiskFilterChange: (value: string) => void;
    visibleFundTypes: string[];
    visibleRiskValues: number[];
    visibleFounders: string[];
    filters: FundScreenerFilterState;
    onFilterChange: <K extends keyof FundScreenerFilterState>(key: K, value: FundScreenerFilterState[K]) => void;
    isScreenerActive: boolean;
    onClearFilters: () => void;
};

export function FundsScreenerModal({
    open,
    onClose,
    totalFundsCount,
    filteredFundsCount,
    activePreset,
    onApplyPreset,
    activeChips,
    fundTypeFilter,
    onFundTypeFilterChange,
    riskFilter,
    onRiskFilterChange,
    visibleFundTypes,
    visibleRiskValues,
    visibleFounders,
    filters,
    onFilterChange,
    isScreenerActive,
    onClearFilters,
}: FundsScreenerModalProps) {
    if (!open) return null;

    const hasNumericFilterError = (key: FundScreenerRangeKey): boolean => {
        const raw = filters[key].trim();
        return raw !== '' && parseFundScreenerNumber(raw) === undefined;
    };

    const renderRangeRows = (group: FundScreenerRangeSpec['group']) =>
        FUND_SCREENER_RANGE_SPECS.filter((spec) => spec.group === group).map((spec) => {
            const minRaw = filters[spec.minKey];
            const maxRaw = filters[spec.maxKey];
            const minVal = parseFundScreenerNumber(minRaw);
            const maxVal = parseFundScreenerNumber(maxRaw);
            const isReversed = minVal !== undefined && maxVal !== undefined && minVal > maxVal;
            const minInvalid = hasNumericFilterError(spec.minKey);
            const maxInvalid = hasNumericFilterError(spec.maxKey);

            return (
                <div key={spec.minKey} className={`funds-screener-range-row${isReversed ? ' is-reversed' : ''}`}>
                    <div className="funds-screener-range-header">
                        <span className="funds-screener-field-label">{spec.label}</span>
                        {isReversed && <span className="funds-screener-range-warn">Min &gt; Max</span>}
                    </div>
                    <div className="funds-screener-range-inputs">
                        <label className={`funds-screener-num-field${minRaw ? ' is-filled' : ''}${minInvalid ? ' is-invalid' : ''}`}>
                            <span className="funds-screener-num-tag">Min</span>
                            <input
                                type="text"
                                inputMode="decimal"
                                placeholder={spec.minPh}
                                aria-label={`${spec.label} en az`}
                                value={minRaw}
                                onChange={(e) => onFilterChange(spec.minKey, e.currentTarget.value)}
                            />
                            {spec.unit && <span className="funds-screener-num-unit">{spec.unit}</span>}
                        </label>
                        <span className="funds-screener-range-sep" aria-hidden="true">–</span>
                        <label className={`funds-screener-num-field${maxRaw ? ' is-filled' : ''}${maxInvalid ? ' is-invalid' : ''}`}>
                            <span className="funds-screener-num-tag">Max</span>
                            <input
                                type="text"
                                inputMode="decimal"
                                placeholder={spec.maxPh}
                                aria-label={`${spec.label} en çok`}
                                value={maxRaw}
                                onChange={(e) => onFilterChange(spec.maxKey, e.currentTarget.value)}
                            />
                            {spec.unit && <span className="funds-screener-num-unit">{spec.unit}</span>}
                        </label>
                    </div>
                </div>
            );
        });

    return createPortal(
        <div
            className="funds-screener-backdrop"
            role="dialog"
            aria-modal="true"
            aria-labelledby="fund-screener-modal-title"
            onClick={onClose}
        >
            <div className="funds-screener-modal" onClick={(event) => event.stopPropagation()}>
                <div className="funds-screener-header">
                    <div className="funds-screener-title-group">
                        <div>
                            <h3 id="fund-screener-modal-title">Fon Tarayıcısı</h3>
                            <p>
                                {totalFundsCount} fon · TEFAS kataloğu
                            </p>
                        </div>
                    </div>
                    <button
                        type="button"
                        className="funds-screener-close-btn"
                        onClick={onClose}
                        aria-label="Kapat"
                    >
                        <X size={17} aria-hidden="true" />
                    </button>
                </div>

                <div className="funds-screener-body">
                    <div className="funds-screener-presets-section">
                        <div className="funds-screener-section-label">
                            <span>Hazır Stratejiler</span>
                        </div>
                        <div className="funds-screener-presets-row">
                            {FUND_SCREENER_PRESETS.map((preset) => (
                                <button
                                    key={preset.id}
                                    type="button"
                                    aria-pressed={activePreset === preset.id}
                                    className={`funds-screener-preset-chip${activePreset === preset.id ? ' is-active' : ''}`}
                                    onClick={() => onApplyPreset(preset.id)}
                                >
                                    {preset.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {activeChips.length > 0 && (
                        <div className="funds-screener-active-chips" aria-label="Aktif filtreler">
                            {activeChips.map((chip) => (
                                <span key={chip.id} className="funds-screener-active-chip">
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

                    <div className="funds-screener-grid">
                        <div className="funds-screener-card">
                            <div className="funds-screener-card-header">
                                <Building2 size={14} aria-hidden="true" />
                                <span>Tür, Risk & Kurucu</span>
                            </div>
                            <div className="funds-screener-card-content">
                                <div className="funds-screener-field">
                                    <label className="funds-screener-field-label" htmlFor="fund-screener-type">
                                        Fon Türü
                                    </label>
                                    <select
                                        id="fund-screener-type"
                                        className="funds-screener-select"
                                        value={fundTypeFilter}
                                        onChange={(e) => onFundTypeFilterChange(e.currentTarget.value)}
                                    >
                                        <option value="">Tüm Türler</option>
                                        {visibleFundTypes.map((item) => (
                                            <option key={item} value={item}>{item}</option>
                                        ))}
                                    </select>
                                </div>
                                <div className="funds-screener-field">
                                    <label className="funds-screener-field-label" htmlFor="fund-screener-risk">
                                        Risk Seviyesi
                                    </label>
                                    <select
                                        id="fund-screener-risk"
                                        className="funds-screener-select"
                                        value={riskFilter}
                                        onChange={(e) => onRiskFilterChange(e.currentTarget.value)}
                                    >
                                        <option value="">Tüm Riskler</option>
                                        {visibleRiskValues.map((item) => (
                                            <option key={item} value={String(item)}>Risk {item}</option>
                                        ))}
                                    </select>
                                </div>
                                <div className="funds-screener-field">
                                    <label className="funds-screener-field-label" htmlFor="fund-screener-founder">
                                        Kurucu / Portföy Yönetimi
                                    </label>
                                    <select
                                        id="fund-screener-founder"
                                        className="funds-screener-select"
                                        value={filters.founder}
                                        onChange={(e) => onFilterChange('founder', e.currentTarget.value)}
                                    >
                                        <option value="">Tüm Kurucular</option>
                                        {visibleFounders.map((item) => (
                                            <option key={item} value={item}>{item}</option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div className="funds-screener-card">
                            <div className="funds-screener-card-header">
                                <Percent size={14} aria-hidden="true" />
                                <span>Ölçek & Yatırımcı</span>
                            </div>
                            <div className="funds-screener-card-content">
                                {renderRangeRows('scale')}
                            </div>
                        </div>

                        <div className="funds-screener-card funds-screener-card-wide">
                            <div className="funds-screener-card-header">
                                <TrendingUp size={14} aria-hidden="true" />
                                <span>Getiri Filtreleri (%)</span>
                            </div>
                            <div className="funds-screener-card-content funds-screener-returns-cols">
                                {renderRangeRows('returns')}
                            </div>
                        </div>
                    </div>
                </div>

                <div className="funds-screener-footer">
                    <button
                        type="button"
                        className="funds-screener-clear-btn"
                        onClick={onClearFilters}
                        disabled={!isScreenerActive}
                    >
                        <RotateCcw size={13} aria-hidden="true" />
                        <span>Filtreleri Sıfırla</span>
                    </button>
                    <div className="funds-screener-footer-right">
                        <span className="funds-screener-footer-info">
                            <strong>{filteredFundsCount}</strong> / {totalFundsCount} fon
                        </span>
                        <button
                            type="button"
                            className="funds-screener-apply-btn"
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
