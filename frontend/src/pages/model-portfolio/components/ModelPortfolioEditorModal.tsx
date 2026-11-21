import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import {
    Check,
    ChevronDown,
    FileText,
    Layers,
    Pencil,
    Search,
    SlidersHorizontal,
    Tag,
    TrendingUp,
    X,
} from 'lucide-react';
import { apiClient } from '../../../api/client';
import type {
    MarketStockRow,
    MarketUniverseRow,
    ModelPortfolioPosition,
    ModelPortfolioResponse,
} from '../../../api/types';
import SymbolLogo from '../../../components/SymbolLogo';
import {
    CATALYST_OPTIONS,
    HORIZON_OPTIONS,
    RATING_OPTIONS,
    SECTOR_OPTIONS,
} from '../constants';
import { useAccessibleDialog } from '../hooks/useAccessibleDialog';
import { formatPct, formatPrice, todayIsoDate, toneClass } from '../utils';
import { MpExplainedDropdown } from './MpExplainedDropdown';
import './ModelPortfolioModals.css';

export { MpExplainedDropdown };

export function MpSectorDropdown({
    value,
    onChange,
}: {
    value: string;
    onChange: (nextSector: string) => void;
}) {
    const [open, setOpen] = useState(false);
    const wrapRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        if (!open) return;
        const onDocClick = (e: MouseEvent) => {
            if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
                setOpen(false);
            }
        };
        document.addEventListener('mousedown', onDocClick);
        return () => document.removeEventListener('mousedown', onDocClick);
    }, [open]);

    return (
        <div className="mp-field mp-explained-select-wrap" ref={wrapRef}>
            <div className="mp-field-label-row">
                <span className="mp-field-label">Sektör & Tema Etiketi</span>
            </div>
            <div className="mp-sector-input-box">
                <input
                    type="text"
                    aria-label="Sektör ve tema etiketi"
                    value={value}
                    onFocus={() => setOpen(true)}
                    onChange={(e) => {
                        onChange(e.target.value);
                        setOpen(true);
                    }}
                    placeholder="Sektör seçin veya yazın..."
                />
                <button
                    type="button"
                    className="mp-sector-toggle-btn"
                    onClick={() => setOpen((p) => !p)}
                    aria-label="Sektör listesini aç"
                >
                    <ChevronDown size={14} className={`mp-explained-chevron${open ? ' is-open' : ''}`} />
                </button>
            </div>

            {open && (
                <div className="mp-explained-menu mp-sector-menu" role="listbox">
                    {SECTOR_OPTIONS.map((sec) => {
                        const isSelected = sec.value === value;
                        return (
                            <button
                                key={sec.value}
                                type="button"
                                className={`mp-explained-option${isSelected ? ' is-selected' : ''}`}
                                onClick={() => {
                                    onChange(sec.value);
                                    setOpen(false);
                                }}
                            >
                                <div className="mp-explained-option-body">
                                    <div className="mp-explained-option-top">
                                        <strong>{sec.value}</strong>
                                        <span className="mp-explained-badge is-blue">{sec.badge}</span>
                                    </div>
                                    <p>{sec.description}</p>
                                </div>
                                <span className="mp-explained-check">
                                    {isSelected && <Check size={14} />}
                                </span>
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
}

export function ModelPortfolioEditorModal({
    adminToken,
    editingPosition,
    activePositions = [],
    availableWeightPct,
    universeRows,
    onClose,
    onSaved,
}: {
    adminToken: string;
    editingPosition: ModelPortfolioPosition | null;
    activePositions?: ModelPortfolioPosition[];
    availableWeightPct: number;
    universeRows: MarketUniverseRow[];
    onClose: () => void;
    onSaved: (nextData: ModelPortfolioResponse) => void;
}) {
    const isEdit = Boolean(editingPosition);
    const activePositionsBySymbol = useMemo(
        () => new Map(activePositions.map((pos) => [pos.symbol.trim().toUpperCase(), pos])),
        [activePositions],
    );
    const [symbol, setSymbol] = useState(editingPosition?.symbol || '');
    const [symbolQuery, setSymbolQuery] = useState(editingPosition?.symbol || '');
    const [companyName, setCompanyName] = useState(editingPosition?.company_name || '');
    const [sector, setSector] = useState(editingPosition?.sector || '');
    const [entryDate, setEntryDate] = useState(editingPosition?.entry_date || todayIsoDate());
    const [entryPrice, setEntryPrice] = useState(
        editingPosition ? String(editingPosition.entry_price) : '',
    );
    const [entryXu100, setEntryXu100] = useState(
        editingPosition?.entry_xu100 != null ? String(editingPosition.entry_xu100) : '',
    );
    const [weightPct, setWeightPct] = useState(
        editingPosition ? String(editingPosition.weight_pct) : String(Math.min(15, availableWeightPct)),
    );
    const [targetPrice, setTargetPrice] = useState(
        editingPosition ? String(editingPosition.target_price) : '',
    );
    const [horizon, setHorizon] = useState(editingPosition?.horizon || '6-12 Ay');
    const [rating, setRating] = useState(editingPosition?.rating || 'Endeks Üstü Getiri');
    const [thesis, setThesis] = useState(editingPosition?.thesis || '');
    const [catalysts, setCatalysts] = useState<string[]>(editingPosition?.catalysts || []);
    const [customCatalyst, setCustomCatalyst] = useState('');
    const [riskNote, setRiskNote] = useState(editingPosition?.risk_note || '');
    const [revisionNote, setRevisionNote] = useState('');

    // Entry multiples (auto-filled on symbol selection, collapsible manual override)
    const [fk, setFk] = useState(
        editingPosition?.entry_multiples?.fk != null ? String(editingPosition.entry_multiples.fk) : '',
    );
    const [fdFavok, setFdFavok] = useState(
        editingPosition?.entry_multiples?.fd_favok != null ? String(editingPosition.entry_multiples.fd_favok) : '',
    );
    const [pdDd, setPdDd] = useState(
        editingPosition?.entry_multiples?.pd_dd != null ? String(editingPosition.entry_multiples.pd_dd) : '',
    );
    const [showMultiplesOverride, setShowMultiplesOverride] = useState(false);

    const [prefillLoading, setPrefillLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [improvingField, setImprovingField] = useState<'thesis' | 'risk_note' | 'revision_note' | null>(null);
    const [improveError, setImproveError] = useState<{
        field: 'thesis' | 'risk_note' | 'revision_note';
        message: string;
    } | null>(null);
    const [formError, setFormError] = useState<string | null>(null);
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const prefillAbortRef = useRef<AbortController | null>(null);
    const selectedSymbolRef = useRef((editingPosition?.symbol || '').toUpperCase());
    const sectorEditVersionRef = useRef(0);
    const dialogRef = useAccessibleDialog<HTMLDivElement>(true, onClose);

    const improveText = useCallback(async (field: 'thesis' | 'risk_note' | 'revision_note') => {
        const text = field === 'thesis' ? thesis : field === 'risk_note' ? riskNote : revisionNote;
        if (!text.trim() || improvingField) return;
        setImprovingField(field);
        setImproveError(null);
        const activePos = !isEdit ? activePositionsBySymbol.get(symbol.trim().toUpperCase()) : undefined;
        try {
            const result = await apiClient.modelPortfolioImproveText(
                {
                    field,
                    text,
                    symbol: activePos?.symbol || symbol,
                    company_name: activePos?.company_name || companyName,
                    sector: activePos?.sector || sector,
                },
                adminToken,
            );
            if (field === 'thesis') setThesis(result.text);
            else if (field === 'risk_note') setRiskNote(result.text);
            else setRevisionNote(result.text);
        } catch (error) {
            setImproveError({
                field,
                message: error instanceof Error ? error.message : 'Metin şu anda iyileştirilemedi. Lütfen tekrar deneyin.',
            });
        } finally {
            setImprovingField(null);
        }
    }, [activePositionsBySymbol, adminToken, companyName, improvingField, isEdit, revisionNote, riskNote, sector, symbol, thesis]);

    const renderImproveButton = (field: 'thesis' | 'risk_note' | 'revision_note', textVal: string) => (
        <button
            type="button"
            className="mp-btn mp-btn-subtle mp-ai-refine-button"
            onClick={() => void improveText(field)}
            disabled={!textVal.trim() || improvingField !== null || saving}
        >
            <Pencil size={13} />
            {improvingField === field ? 'Düzenleniyor...' : 'AI ile iyileştir'}
        </button>
    );

    const matchingUniverse = useMemo(() => {
        const q = symbolQuery.trim().toUpperCase();
        if (!q) return universeRows.slice(0, 10);
        return universeRows
            .filter(
                (r) =>
                    String(r.symbol || r.company).toUpperCase().includes(q)
                    || String(r.name || '').toUpperCase().includes(q),
            )
            .slice(0, 10);
    }, [symbolQuery, universeRows]);

    const triggerPrefill = useCallback((targetSym: string, targetDate: string) => {
        const cleanSym = targetSym.trim().toUpperCase();
        if (!cleanSym) return;
        prefillAbortRef.current?.abort();
        const controller = new AbortController();
        prefillAbortRef.current = controller;
        setPrefillLoading(true);
        apiClient
            .modelPortfolioPrefill(cleanSym, targetDate, controller.signal)
            .then((res) => {
                setCompanyName(res.company_name || cleanSym);
                if (res.entry_price != null) {
                    setEntryPrice(String(res.entry_price));
                }
                if (res.entry_xu100 != null) setEntryXu100(String(res.entry_xu100));
                if (res.entry_multiples) {
                    setFk(res.entry_multiples.fk != null ? String(res.entry_multiples.fk) : '');
                    setFdFavok(res.entry_multiples.fd_favok != null ? String(res.entry_multiples.fd_favok) : '');
                    setPdDd(res.entry_multiples.pd_dd != null ? String(res.entry_multiples.pd_dd) : '');
                }
            })
            .catch(() => {})
            .finally(() => setPrefillLoading(false));
    }, []);

    const existingActivePosition = useMemo(
        () => (!isEdit ? activePositionsBySymbol.get(symbol.trim().toUpperCase()) || null : null),
        [activePositionsBySymbol, isEdit, symbol],
    );
    const isAddToExisting = Boolean(existingActivePosition);

    useEffect(() => {
        setFormError(null);
        if (!existingActivePosition) return;
        setEntryPrice((prev) => {
            if (prev.trim() && Number(prev) > 0) return prev;
            const fallback = existingActivePosition.current_price ?? existingActivePosition.entry_price;
            return fallback != null ? String(fallback) : prev;
        });
        if (availableWeightPct > 0) {
            setWeightPct((prev) => {
                const num = Number(prev);
                if (!prev.trim() || !Number.isFinite(num) || num <= 0 || num > availableWeightPct) {
                    return String(Number(Math.min(5, availableWeightPct).toFixed(2)));
                }
                return prev;
            });
        }
    }, [availableWeightPct, existingActivePosition]);

    const autoSelectSector = useCallback(async (targetSym: string) => {
        const normalizedSymbol = targetSym.trim().toUpperCase();
        if (!normalizedSymbol) return;
        const editVersion = sectorEditVersionRef.current;
        const findSector = (rows: MarketStockRow[]) => rows.find(
            (row) => String(row.symbol || row.company || '').toUpperCase() === normalizedSymbol,
        )?.sector?.trim();

        let resolvedSector = findSector(apiClient.getCachedMarketStocks('XUTUM')?.rows || []);
        if (!resolvedSector) {
            try {
                const payload = await apiClient.marketStocks({ index: 'XUTUM' });
                resolvedSector = findSector(payload.rows || []);
            } catch {
                return;
            }
        }

        if (
            resolvedSector
            && selectedSymbolRef.current === normalizedSymbol
            && sectorEditVersionRef.current === editVersion
        ) {
            setSector(resolvedSector);
        }
    }, []);

    const selectStock = useCallback((targetSym: string, targetName: string) => {
        const normalizedSymbol = targetSym.trim().toUpperCase();
        selectedSymbolRef.current = normalizedSymbol;
        sectorEditVersionRef.current += 1;
        setSymbol(normalizedSymbol);
        setSymbolQuery(normalizedSymbol);
        setCompanyName(targetName || normalizedSymbol);
        setSector('');
        setFormError(null);
        setDropdownOpen(false);
        const matchedExisting = !isEdit ? activePositionsBySymbol.get(normalizedSymbol) : undefined;
        if (matchedExisting) {
            const fallbackPrice = matchedExisting.current_price ?? matchedExisting.entry_price;
            if (fallbackPrice != null) {
                setEntryPrice(String(fallbackPrice));
            }
        } else {
            void autoSelectSector(normalizedSymbol);
        }
        triggerPrefill(normalizedSymbol, entryDate);
    }, [activePositionsBySymbol, autoSelectSector, entryDate, isEdit, triggerPrefill]);

    const addPreview = useMemo(() => {
        if (!isAddToExisting || !existingActivePosition) return null;
        const oldWeight = Number(existingActivePosition.weight_pct) || 0;
        const parsedAddWeight = Number(weightPct);
        const addWeight = Number.isFinite(parsedAddWeight) && parsedAddWeight > 0 ? parsedAddWeight : 0;
        const newWeight = Number((oldWeight + addWeight).toFixed(2));
        const oldEntryPrice = Number(existingActivePosition.entry_price) || 0;
        const parsedAddPrice = Number(entryPrice);
        const addPrice = Number.isFinite(parsedAddPrice) && parsedAddPrice > 0 ? parsedAddPrice : null;
        const newAvgPrice =
            addPrice != null && addWeight > 0 && newWeight > 0
                ? Number((((oldWeight * oldEntryPrice) + (addWeight * addPrice)) / newWeight).toFixed(2))
                : oldEntryPrice;
        return {
            oldWeight,
            addWeight,
            newWeight,
            oldEntryPrice,
            addPrice,
            newAvgPrice,
        };
    }, [entryPrice, existingActivePosition, isAddToExisting, weightPct]);

    const impliedUpsidePct = useMemo(() => {
        const ep = Number(entryPrice);
        const tp = Number(targetPrice);
        if (!Number.isFinite(ep) || !Number.isFinite(tp) || ep <= 0 || tp <= 0) return null;
        return Number((((tp - ep) / ep) * 100).toFixed(1));
    }, [entryPrice, targetPrice]);

    const applyTargetUpsidePreset = (pct: number) => {
        const ep = Number(entryPrice);
        if (!Number.isFinite(ep) || ep <= 0) return;
        const computed = Number((ep * (1 + pct / 100)).toFixed(2));
        setTargetPrice(String(computed));
    };

    const remainingCashAfterWeight = useMemo(() => {
        const wp = Number(weightPct);
        if (!Number.isFinite(wp)) return availableWeightPct;
        return Math.max(0, Number((availableWeightPct - wp).toFixed(1)));
    }, [availableWeightPct, weightPct]);

    const toggleCatalyst = (label: string) => {
        setCatalysts((prev) =>
            prev.includes(label) ? prev.filter((c) => c !== label) : [...prev, label],
        );
    };

    const addCustomCatalyst = () => {
        const clean = customCatalyst.trim();
        if (!clean) return;
        if (!catalysts.includes(clean)) {
            setCatalysts((prev) => [...prev, clean]);
        }
        setCustomCatalyst('');
    };

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (improvingField) return;
        setFormError(null);
        const ep = Number(entryPrice);
        const tp = Number(targetPrice);
        const wp = Number(weightPct);
        if (!symbol.trim()) {
            setFormError('Lütfen bir hisse kodu seçin.');
            return;
        }
        if (isAddToExisting && existingActivePosition) {
            if (!Number.isFinite(ep) || ep <= 0) {
                setFormError('Geçerli bir ekleme fiyatı girin.');
                return;
            }
            if (availableWeightPct <= 0) {
                setFormError('Portföy ağırlıkları zaten %100. Ekleme yapmak için önce mevcut ağırlıkları azaltın.');
                return;
            }
            if (!Number.isFinite(wp) || wp <= 0 || wp > availableWeightPct + 1e-6) {
                setFormError(
                    `Kullanılabilir portföy payı %${availableWeightPct.toFixed(1)}. En fazla bu kadar ekleme yapabilirsiniz.`,
                );
                return;
            }
            setSaving(true);
            try {
                const added = await apiClient.modelPortfolioAddToPosition(
                    existingActivePosition.id,
                    {
                        add_weight_pct: wp,
                        add_price: ep,
                        add_date: entryDate,
                        add_xu100: entryXu100.trim() ? Number(entryXu100) : null,
                        note: revisionNote.trim() || undefined,
                    },
                    adminToken,
                );
                onSaved(added);
            } catch (err: unknown) {
                setFormError(err instanceof Error ? err.message : 'Ekleme sırasında hata oluştu.');
            } finally {
                setSaving(false);
            }
            return;
        }
        if (!Number.isFinite(ep) || ep <= 0) {
            setFormError('Geçerli bir giriş fiyatı girin.');
            return;
        }
        if (!Number.isFinite(tp) || tp <= 0) {
            setFormError('Geçerli bir hedef fiyat girin.');
            return;
        }
        if (!isEdit && availableWeightPct <= 0) {
            setFormError('Portföy ağırlıkları zaten %100. Yeni hisse eklemek için önce mevcut ağırlıkları azaltın.');
            return;
        }
        if (!Number.isFinite(wp) || wp <= 0 || wp > 100) {
            setFormError('Ağırlık %0 ile %100 arasında olmalıdır.');
            return;
        }
        if (wp > availableWeightPct + 1e-6) {
            setFormError(
                `Kullanılabilir portföy payı %${availableWeightPct.toFixed(1)}. Yeni hisse eklemek için önce mevcut ağırlıkları azaltın.`,
            );
            return;
        }

        const multiplesPayload = {
            fk: fk.trim() ? Number(fk) : null,
            fd_favok: fdFavok.trim() ? Number(fdFavok) : null,
            pd_dd: pdDd.trim() ? Number(pdDd) : null,
        };

        setSaving(true);
        try {
            if (isEdit && editingPosition) {
                const updated = await apiClient.modelPortfolioUpdatePosition(
                    editingPosition.id,
                    {
                        entry_date: entryDate,
                        entry_price: ep,
                        entry_xu100: entryXu100.trim() ? Number(entryXu100) : null,
                        weight_pct: wp,
                        target_price: tp,
                        sector: sector.trim() || null,
                        horizon,
                        rating,
                        thesis,
                        catalysts,
                        risk_note: riskNote,
                        revision_note: revisionNote.trim() || undefined,
                        entry_multiples: multiplesPayload,
                    },
                    adminToken,
                );
                onSaved(updated);
            } else {
                const created = await apiClient.modelPortfolioCreatePosition(
                    {
                        symbol: symbol.trim().toUpperCase(),
                        company_name: companyName.trim() || symbol.trim().toUpperCase(),
                        sector: sector.trim() || null,
                        entry_date: entryDate,
                        entry_price: ep,
                        entry_xu100: entryXu100.trim() ? Number(entryXu100) : null,
                        weight_pct: wp,
                        target_price: tp,
                        horizon,
                        rating,
                        thesis,
                        catalysts,
                        risk_note: riskNote,
                        entry_multiples: multiplesPayload,
                    },
                    adminToken,
                );
                onSaved(created);
            }
        } catch (err: unknown) {
            setFormError(err instanceof Error ? err.message : 'Kayıt sırasında hata oluştu.');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="mp-modal-backdrop" onClick={onClose}>
            <div
                ref={dialogRef}
                className="mp-modal mp-modal-lg mp-editor-modal"
                role="dialog"
                aria-modal="true"
                aria-labelledby="mp-editor-dialog-title"
                tabIndex={-1}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Sticky Header */}
                <div className="mp-modal-header">
                    <div className="mp-editor-title-group">
                        <h3 id="mp-editor-dialog-title">
                            {isEdit
                                ? `${editingPosition?.symbol} Pozisyonunu Düzenle`
                                : isAddToExisting && existingActivePosition
                                    ? `${existingActivePosition.symbol} Pozisyonuna Ekleme Yap`
                                    : 'Hisse Ekle'}
                        </h3>
                        <p className="mp-editor-subtitle">
                            {isEdit && editingPosition
                                ? `Mevcut ağırlık %${editingPosition.weight_pct.toFixed(1)} · Giriş ${editingPosition.entry_date}`
                                : isAddToExisting && existingActivePosition
                                    ? `Portföyde %${existingActivePosition.weight_pct.toFixed(1)} · Kullanılabilir nakit %${availableWeightPct.toFixed(1)}`
                                    : `Model Portföy · Kullanılabilir nakit payı %${availableWeightPct.toFixed(1)}`}
                        </p>
                    </div>
                    <button type="button" className="mp-editor-close-btn" aria-label="Pozisyon formunu kapat" onClick={onClose}>
                        <X size={16} />
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="mp-editor-shell">
                    <div className="mp-modal-body mp-editor-form">
                        {/* CARD 1: Stock Selection, Date & Sector */}
                        <div className="mp-editor-card">
                            <div className="mp-editor-card-header">
                                <Layers size={13} aria-hidden="true" />
                                <span>{isAddToExisting ? 'Pozisyon & Ekleme Tarihi' : 'Hisse & Konumlandırma'}</span>
                            </div>
                            <div className="mp-editor-card-content">
                            <div className={isAddToExisting ? 'mp-form-grid-2' : 'mp-form-grid-3'}>
                                <div className="mp-field mp-symbol-picker">
                                    <div className="mp-field-label-row">
                                        <span className="mp-field-label">Hisse Kodu *</span>
                                    </div>
                                    <div className="mp-search-input-wrap">
                                        <Search size={14} className="mp-search-icon" />
                                        <input
                                            type="text"
                                            aria-label="Hisse kodu"
                                            data-dialog-autofocus={!isEdit ? 'true' : undefined}
                                            role="combobox"
                                            aria-autocomplete="list"
                                            aria-expanded={dropdownOpen}
                                            aria-controls="mp-symbol-options"
                                            value={symbolQuery}
                                            disabled={isEdit}
                                            onFocus={() => setDropdownOpen(true)}
                                            onChange={(e) => {
                                                const val = e.target.value.toUpperCase();
                                                const normalizedSymbol = val.trim();
                                                if (normalizedSymbol !== selectedSymbolRef.current) {
                                                    selectedSymbolRef.current = normalizedSymbol;
                                                    sectorEditVersionRef.current += 1;
                                                    setSector('');
                                                    setFormError(null);
                                                    if (!isEdit) setEntryPrice('');
                                                }
                                                setSymbolQuery(val);
                                                setSymbol(val);
                                                setDropdownOpen(true);
                                            }}
                                            onBlur={() => {
                                                window.setTimeout(() => setDropdownOpen(false), 180);
                                                const normalizedSymbol = selectedSymbolRef.current;
                                                if (!isEdit && normalizedSymbol) {
                                                    if (!activePositionsBySymbol.has(normalizedSymbol)) {
                                                        void autoSelectSector(normalizedSymbol);
                                                    }
                                                    triggerPrefill(normalizedSymbol, entryDate);
                                                }
                                            }}
                                            placeholder="THYAO, BIMAS, TUPRS..."
                                        />
                                    </div>
                                    {!isEdit && dropdownOpen && matchingUniverse.length > 0 && (
                                        <div className="mp-symbol-dropdown" id="mp-symbol-options" role="listbox" aria-label="Hisse seçenekleri">
                                            {matchingUniverse.map((row) => {
                                                const sym = String(row.symbol || row.company).toUpperCase();
                                                const existingPos = activePositionsBySymbol.get(sym);
                                                return (
                                                    <button
                                                        key={sym}
                                                        type="button"
                                                        className="mp-symbol-option"
                                                        role="option"
                                                        aria-selected={symbol === sym}
                                                        onMouseDown={(e) => {
                                                            e.preventDefault();
                                                            selectStock(sym, row.name || sym);
                                                        }}
                                                    >
                                                        <div className="mp-symbol-option-left">
                                                            <SymbolLogo
                                                                symbol={sym}
                                                                name={row.name || sym}
                                                                kind="stock"
                                                                size="xs"
                                                            />
                                                            <strong>{sym}</strong>
                                                            {existingPos && (
                                                                <span className="mp-symbol-active-pill mp-mono">
                                                                    Portföyde · %{existingPos.weight_pct.toFixed(1).replace(/\.0$/, '')}
                                                                </span>
                                                            )}
                                                        </div>
                                                        <span title={row.name || undefined}>{row.name}</span>
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>

                                <div className="mp-field">
                                    <div className="mp-field-label-row">
                                        <span className="mp-field-label">
                                            {isAddToExisting ? 'Ekleme Tarihi *' : 'Giriş Tarihi *'}
                                        </span>
                                        <button
                                            type="button"
                                            className="mp-mini-preset-btn"
                                            onClick={() => {
                                                const d = todayIsoDate();
                                                setEntryDate(d);
                                                if (symbol.trim()) triggerPrefill(symbol, d);
                                            }}
                                        >
                                            Bugün
                                        </button>
                                    </div>
                                    <input
                                        type="date"
                                        aria-label={isAddToExisting ? 'Ekleme tarihi' : 'Giriş tarihi'}
                                        value={entryDate}
                                        onChange={(e) => {
                                            const nextDate = e.target.value;
                                            setEntryDate(nextDate);
                                            if (symbol.trim()) triggerPrefill(symbol, nextDate);
                                        }}
                                    />
                                </div>

                                {!isAddToExisting && (
                                    <MpSectorDropdown
                                        value={sector}
                                        onChange={(nextSector) => {
                                            sectorEditVersionRef.current += 1;
                                            setSector(nextSector);
                                        }}
                                    />
                                )}
                            </div>

                            {/* Existing Position Info Banner vs New Stock Reference Strip */}
                            {isAddToExisting && existingActivePosition ? (
                                <div className="mp-selected-stock-banner mp-add-existing-banner">
                                    <div className="mp-selected-stock-left">
                                        <SymbolLogo
                                            symbol={existingActivePosition.symbol}
                                            name={existingActivePosition.company_name}
                                            kind="stock"
                                            size="sm"
                                        />
                                        <div>
                                            <div className="mp-selected-stock-title">
                                                <strong className="mp-mono">{existingActivePosition.symbol}</strong>
                                                <span>{existingActivePosition.company_name}</span>
                                                <span className="mp-add-existing-badge">
                                                    Portföyde · Mevcut analiz ve hedef fiyat korunur
                                                </span>
                                            </div>
                                            <div className="mp-selected-stock-multiples mp-mono">
                                                <span>
                                                    Mevcut Ağırlık: <b>%{existingActivePosition.weight_pct.toFixed(1)}</b>
                                                </span>
                                                <span>
                                                    Mevcut Maliyet: <b>{formatPrice(existingActivePosition.entry_price)}</b>
                                                </span>
                                                <span>
                                                    Hedef Fiyat: <b>{formatPrice(existingActivePosition.target_price)}</b>
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                symbol.trim() && (
                                    <div className="mp-selected-stock-banner">
                                        <div className="mp-selected-stock-left">
                                            <SymbolLogo
                                                symbol={symbol.trim().toUpperCase()}
                                                name={companyName || symbol}
                                                kind="stock"
                                                size="sm"
                                            />
                                            <div>
                                                <div className="mp-selected-stock-title">
                                                    <strong className="mp-mono">{symbol.trim().toUpperCase()}</strong>
                                                    <span>{companyName || 'Şirket seçildi'}</span>
                                                </div>
                                                <div className="mp-selected-stock-multiples mp-mono">
                                                    <span>
                                                        XU100: <b>{entryXu100 ? Number(entryXu100).toLocaleString('tr-TR') : '—'}</b>
                                                    </span>
                                                    <span>
                                                        F/K: <b>{fk ? `${Number(fk).toFixed(2)}x` : '—'}</b>
                                                    </span>
                                                    <span>
                                                        FD/FAVÖK: <b>{fdFavok ? `${Number(fdFavok).toFixed(2)}x` : '—'}</b>
                                                    </span>
                                                    <span>
                                                        PD/DD: <b>{pdDd ? `${Number(pdDd).toFixed(2)}x` : '—'}</b>
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <button
                                            type="button"
                                            className="mp-btn mp-btn-subtle mp-btn-sm"
                                            onClick={() => setShowMultiplesOverride((p) => !p)}
                                        >
                                            <SlidersHorizontal size={12} />
                                            <span>{showMultiplesOverride ? 'Çarpanları Gizle' : 'Çarpanları Düzenle'}</span>
                                        </button>
                                    </div>
                                )
                            )}

                            {!isAddToExisting && showMultiplesOverride && (
                                <div className="mp-form-grid-4 mp-override-grid">
                                    <label className="mp-field">
                                        <span className="mp-field-label">Giriş Günü XU100</span>
                                        <input
                                            type="number"
                                            step="0.01"
                                            value={entryXu100}
                                            onChange={(e) => setEntryXu100(e.target.value)}
                                            placeholder="Otomatik"
                                        />
                                    </label>
                                    <label className="mp-field">
                                        <span className="mp-field-label">Giriş F/K</span>
                                        <input
                                            type="number"
                                            step="0.01"
                                            value={fk}
                                            onChange={(e) => setFk(e.target.value)}
                                            placeholder="Otomatik"
                                        />
                                    </label>
                                    <label className="mp-field">
                                        <span className="mp-field-label">Giriş FD/FAVÖK</span>
                                        <input
                                            type="number"
                                            step="0.01"
                                            value={fdFavok}
                                            onChange={(e) => setFdFavok(e.target.value)}
                                            placeholder="Otomatik"
                                        />
                                    </label>
                                    <label className="mp-field">
                                        <span className="mp-field-label">Giriş PD/DD</span>
                                        <input
                                            type="number"
                                            step="0.01"
                                            value={pdDd}
                                            onChange={(e) => setPdDd(e.target.value)}
                                            placeholder="Otomatik"
                                        />
                                    </label>
                                </div>
                            )}
                            </div>
                        </div>

                        {/* CARD 2+: Price, Weight, Strategy & Analysis */}
                        {isAddToExisting && existingActivePosition ? (
                            <div className="mp-editor-card">
                                <div className="mp-editor-card-header">
                                    <TrendingUp size={13} aria-hidden="true" />
                                    <span>Ekleme Detayı & Not</span>
                                </div>
                                <div className="mp-editor-card-content">
                                    <div className="mp-form-grid-2">
                                        <div className="mp-field">
                                            <div className="mp-field-label-row">
                                                <span className="mp-field-label">Ekleme Fiyatı (₺) *</span>
                                                {prefillLoading && <em className="mp-prefill-tag">Çekiliyor...</em>}
                                            </div>
                                            <input
                                                type="number"
                                                aria-label="Ekleme fiyatı"
                                                step="0.01"
                                                value={entryPrice}
                                                onChange={(e) => setEntryPrice(e.target.value)}
                                                placeholder="0.00"
                                            />
                                        </div>

                                        <div className="mp-field">
                                            <div className="mp-field-label-row">
                                                <span className="mp-field-label">Eklenecek Ağırlık (%) *</span>
                                                <span className="mp-field-micro-hint mp-mono">
                                                    Kalan: %{remainingCashAfterWeight.toFixed(1)}
                                                </span>
                                            </div>
                                            <input
                                                type="number"
                                                aria-label="Eklenecek ağırlık yüzde"
                                                step="0.5"
                                                min="0.5"
                                                max="100"
                                                value={weightPct}
                                                onChange={(e) => setWeightPct(e.target.value)}
                                            />
                                            <div className="mp-quick-pills-row">
                                                {[2.5, 5, 10].map((w) => (
                                                    <button
                                                        key={w}
                                                        type="button"
                                                        className={`mp-quick-pill${Number(weightPct) === w ? ' active' : ''}`}
                                                        onClick={() => setWeightPct(String(Math.min(w, availableWeightPct)))}
                                                    >
                                                        +%{w}
                                                    </button>
                                                ))}
                                                <button
                                                    type="button"
                                                    className="mp-quick-pill"
                                                    title={`Kullanılabilir maksimum: %${availableWeightPct.toFixed(1)}`}
                                                    onClick={() => setWeightPct(String(availableWeightPct))}
                                                >
                                                    Maks
                                                </button>
                                            </div>
                                            {availableWeightPct <= 0 && (
                                                <span className="mp-field-micro-hint mp-field-warning-text">
                                                    Portföy %100 dolu; ekleme yapmak için önce diğer pozisyonların ağırlığını azaltın.
                                                </span>
                                            )}
                                        </div>
                                    </div>

                                    {addPreview && (
                                        <div className="mp-add-preview-grid">
                                            <div className="mp-add-preview-item">
                                                <span>Yeni Toplam Ağırlık</span>
                                                <strong className="mp-mono">
                                                    %{addPreview.oldWeight.toFixed(1)} → %{addPreview.newWeight.toFixed(1)}
                                                </strong>
                                            </div>
                                            <div className="mp-add-preview-item">
                                                <span>Yeni Ortalama Maliyet</span>
                                                <strong className="mp-mono">
                                                    {formatPrice(addPreview.oldEntryPrice)} → {formatPrice(addPreview.newAvgPrice)}
                                                </strong>
                                            </div>
                                            <div className="mp-add-preview-item">
                                                <span>Hedef Fiyat (Korunur)</span>
                                                <strong className="mp-mono">
                                                    {formatPrice(existingActivePosition.target_price)}
                                                </strong>
                                            </div>
                                        </div>
                                    )}

                                    <div className="mp-field">
                                        <div className="mp-field-label-row">
                                            <span className="mp-field-label">Ekleme Notu (İsteğe bağlı)</span>
                                            {renderImproveButton('revision_note', revisionNote)}
                                        </div>
                                        <input
                                            type="text"
                                            aria-label="Ekleme notu"
                                            value={revisionNote}
                                            onChange={(e) => setRevisionNote(e.target.value)}
                                            placeholder="Örn: Geri çekilmede kademeli alım / ağırlık artışı"
                                            disabled={improvingField === 'revision_note'}
                                        />
                                        {improveError?.field === 'revision_note' && (
                                            <span className="mp-ai-refine-error" role="alert">{improveError.message}</span>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ) : (
                        <>
                        <div className="mp-editor-cards-grid">
                            {/* Left Card: Price & Weight */}
                            <div className="mp-editor-card">
                                <div className="mp-editor-card-header">
                                    <TrendingUp size={13} aria-hidden="true" />
                                    <span>Fiyat & Ağırlık</span>
                                </div>
                                <div className="mp-editor-card-content">
                                    <div className="mp-form-grid-2">
                                        <div className="mp-field">
                                            <div className="mp-field-label-row">
                                                <span className="mp-field-label">Giriş Fiyatı (₺) *</span>
                                                {prefillLoading && <em className="mp-prefill-tag">Çekiliyor...</em>}
                                            </div>
                                            <input
                                                type="number"
                                                aria-label="Giriş fiyatı"
                                                step="0.01"
                                                value={entryPrice}
                                                onChange={(e) => setEntryPrice(e.target.value)}
                                                placeholder="0.00"
                                            />
                                        </div>

                                        <div className="mp-field">
                                            <div className="mp-field-label-row">
                                                <span className="mp-field-label">Hedef Fiyat (₺) *</span>
                                                {impliedUpsidePct != null && (
                                                    <span className={`mp-upside-live-pill ${toneClass(impliedUpsidePct)}`}>
                                                        {formatPct(impliedUpsidePct)}
                                                    </span>
                                                )}
                                            </div>
                                            <input
                                                type="number"
                                                aria-label="Hedef fiyat"
                                                step="0.01"
                                                value={targetPrice}
                                                onChange={(e) => setTargetPrice(e.target.value)}
                                                placeholder="0.00"
                                            />
                                            <div className="mp-quick-pills-row">
                                                {[25, 35, 50, 75].map((pct) => (
                                                    <button
                                                        key={pct}
                                                        type="button"
                                                        className="mp-quick-pill"
                                                        disabled={!Number(entryPrice)}
                                                        onClick={() => applyTargetUpsidePreset(pct)}
                                                        title={`Giriş fiyatının %${pct} üzerini hesapla`}
                                                    >
                                                        +%{pct}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="mp-field">
                                        <div className="mp-field-label-row">
                                            <span className="mp-field-label">Portföy Ağırlığı (%) *</span>
                                            <span className="mp-field-micro-hint mp-mono">
                                                Kalan: %{remainingCashAfterWeight.toFixed(1)}
                                            </span>
                                        </div>
                                        <input
                                            type="number"
                                            aria-label="Portföy ağırlığı yüzde"
                                            step="0.5"
                                            min="0.5"
                                            max="100"
                                            value={weightPct}
                                            onChange={(e) => setWeightPct(e.target.value)}
                                        />
                                        <div className="mp-quick-pills-row">
                                            {[5, 10, 15, 20].map((w) => (
                                                <button
                                                    key={w}
                                                    type="button"
                                                    className={`mp-quick-pill${Number(weightPct) === w ? ' active' : ''}`}
                                                    onClick={() => setWeightPct(String(Math.min(w, availableWeightPct)))}
                                                >
                                                    %{w}
                                                </button>
                                            ))}
                                            <button
                                                type="button"
                                                className="mp-quick-pill"
                                                title={`Kullanılabilir maksimum: %${availableWeightPct.toFixed(0)}`}
                                                onClick={() => setWeightPct(String(availableWeightPct))}
                                            >
                                                Maks
                                            </button>
                                        </div>
                                        {!isEdit && availableWeightPct <= 0 && (
                                            <span className="mp-field-micro-hint mp-field-warning-text">
                                                Portföy %100 dolu; yeni hisse için pay açın.
                                            </span>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Right Card: Strategy & Catalysts */}
                            <div className="mp-editor-card">
                                <div className="mp-editor-card-header">
                                    <Tag size={13} aria-hidden="true" />
                                    <span>Strateji & Beklenti Etiketleri</span>
                                </div>
                                <div className="mp-editor-card-content">
                                    <div className="mp-form-grid-2">
                                        <MpExplainedDropdown
                                            label="Tavsiye"
                                            value={rating}
                                            options={RATING_OPTIONS}
                                            onChange={setRating}
                                        />
                                        <MpExplainedDropdown
                                            label="Vade"
                                            value={horizon}
                                            options={HORIZON_OPTIONS}
                                            onChange={setHorizon}
                                        />
                                    </div>

                                    <div className="mp-field">
                                        <span className="mp-field-label">Beklenti Etiketleri</span>
                                        <div className="mp-catalyst-choices" role="group" aria-label="Beklenti etiketleri">
                                            {CATALYST_OPTIONS.map((item) => {
                                                const active = catalysts.includes(item.value);
                                                return (
                                                    <button
                                                        key={item.value}
                                                        type="button"
                                                        className={`mp-catalyst-choice${active ? ' active' : ''}`}
                                                        aria-pressed={active}
                                                        onClick={() => toggleCatalyst(item.value)}
                                                    >
                                                        {active && <Check size={12} />}
                                                        {item.label}
                                                    </button>
                                                );
                                            })}
                                            {catalysts
                                                .filter((item) => !CATALYST_OPTIONS.some((option) => option.value === item))
                                                .map((item) => (
                                                    <button
                                                        key={item}
                                                        type="button"
                                                        className="mp-catalyst-choice active"
                                                        aria-pressed="true"
                                                        onClick={() => toggleCatalyst(item)}
                                                    >
                                                        <Check size={12} />
                                                        {item}
                                                    </button>
                                                ))}
                                        </div>
                                        <div className="mp-custom-catalyst-row">
                                            <input
                                                type="text"
                                                aria-label="Başka bir beklenti etiketi ekle"
                                                value={customCatalyst}
                                                onChange={(e) => setCustomCatalyst(e.target.value)}
                                                placeholder="Özel etiket..."
                                                onKeyDown={(e) => {
                                                    if (e.key === 'Enter') {
                                                        e.preventDefault();
                                                        addCustomCatalyst();
                                                    }
                                                }}
                                            />
                                            <button
                                                type="button"
                                                className="mp-btn mp-btn-subtle"
                                                onClick={addCustomCatalyst}
                                                disabled={!customCatalyst.trim()}
                                            >
                                                Ekle
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Card 4: Analysis & Notes */}
                        <div className="mp-editor-card">
                            <div className="mp-editor-card-header">
                                <FileText size={13} aria-hidden="true" />
                                <span>Analiz & Notlar</span>
                            </div>
                            <div className="mp-editor-card-content">
                                <div className="mp-form-grid-2">
                                    <div className="mp-field">
                                        <div className="mp-field-label-row">
                                            <span className="mp-field-label">Yatırım Tezi</span>
                                            {renderImproveButton('thesis', thesis)}
                                        </div>
                                        <textarea
                                            rows={3}
                                            aria-label="Yatırım tezi"
                                            value={thesis}
                                            onChange={(e) => setThesis(e.target.value)}
                                            placeholder="Katalizörler ve temel beklentiler..."
                                            disabled={improvingField === 'thesis'}
                                        />
                                        {improveError?.field === 'thesis' && (
                                            <span className="mp-ai-refine-error" role="alert">{improveError.message}</span>
                                        )}
                                    </div>

                                    <div className="mp-field">
                                        <div className="mp-field-label-row">
                                            <span className="mp-field-label">Temel Riskler</span>
                                            {renderImproveButton('risk_note', riskNote)}
                                        </div>
                                        <textarea
                                            rows={3}
                                            aria-label="Temel riskler"
                                            value={riskNote}
                                            onChange={(e) => setRiskNote(e.target.value)}
                                            placeholder="Marj baskısı, kur, talep riskleri..."
                                            disabled={improvingField === 'risk_note'}
                                        />
                                        {improveError?.field === 'risk_note' && (
                                            <span className="mp-ai-refine-error" role="alert">{improveError.message}</span>
                                        )}
                                    </div>
                                </div>

                                {isEdit && (
                                    <div className="mp-field">
                                        <div className="mp-field-label-row">
                                            <span className="mp-field-label">Revizyon Notu</span>
                                            {renderImproveButton('revision_note', revisionNote)}
                                        </div>
                                        <input
                                            type="text"
                                            aria-label="Revizyon notu"
                                            value={revisionNote}
                                            onChange={(e) => setRevisionNote(e.target.value)}
                                            placeholder="Örn: 3Ç bilançosu sonrası hedef revizesi"
                                            disabled={improvingField === 'revision_note'}
                                        />
                                        {improveError?.field === 'revision_note' && (
                                            <span className="mp-ai-refine-error" role="alert">{improveError.message}</span>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                        </>
                        )}

                        {formError && <div className="mp-form-error">{formError}</div>}
                    </div>

                    {/* Sticky Footer */}
                    <div className="mp-modal-footer mp-modal-footer-sticky">
                        <div className="mp-footer-summary">
                            {isAddToExisting && existingActivePosition && addPreview ? (
                                <span>
                                    <b>{existingActivePosition.symbol}</b> · +%{addPreview.addWeight.toFixed(1)} →{' '}
                                    <b>%{addPreview.newWeight.toFixed(1)}</b> · Ort. Maliyet:{' '}
                                    <b>{formatPrice(addPreview.newAvgPrice)}</b>
                                </span>
                            ) : symbol.trim() ? (
                                <span>
                                    <b>{symbol.trim().toUpperCase()}</b> · Ağırlık: <b>%{weightPct || 0}</b>
                                    {impliedUpsidePct != null && (
                                        <>
                                            {' '}
                                            · Hedef Potansiyel: <b>{formatPct(impliedUpsidePct)}</b>
                                        </>
                                    )}
                                </span>
                            ) : (
                                <span>Hisse kodu seçin</span>
                            )}
                        </div>
                        <div className="mp-footer-btns">
                            <button type="button" className="mp-btn mp-btn-ghost" onClick={onClose}>
                                İptal
                            </button>
                            <button type="submit" className="mp-btn mp-btn-primary" disabled={saving || improvingField !== null}>
                                {saving
                                    ? (isAddToExisting ? 'Ekleniyor...' : 'Kaydediliyor...')
                                    : isEdit
                                        ? 'Kaydet'
                                        : isAddToExisting
                                            ? 'Ağırlık Artır'
                                            : 'Portföye Ekle'}
                            </button>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    );
}

export default ModelPortfolioEditorModal;
