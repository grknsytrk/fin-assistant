import { useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { CheckCircle2, X } from 'lucide-react';
import { apiClient } from '../../../api/client';
import type { ModelPortfolioPosition, ModelPortfolioResponse } from '../../../api/types';
import { CLOSE_REASON_OPTIONS } from '../constants';
import { useAccessibleDialog } from '../hooks/useAccessibleDialog';
import { formatPct, formatPrice, todayIsoDate, toneClass } from '../utils';
import { MpExplainedDropdown } from './MpExplainedDropdown';
import './ModelPortfolioModals.css';

export function ModelPortfolioCloseModal({
    adminToken,
    position,
    currentXu100,
    onClose,
    onClosed,
}: {
    adminToken: string;
    position: ModelPortfolioPosition;
    currentXu100: number | null;
    onClose: () => void;
    onClosed: (nextData: ModelPortfolioResponse) => void;
}) {
    const [exitPrice, setExitPrice] = useState(
        String(position.current_price ?? position.entry_price),
    );
    const [closedDate, setClosedDate] = useState(todayIsoDate());
    const [closeReason, setCloseReason] = useState(CLOSE_REASON_OPTIONS[0].value);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const dialogRef = useAccessibleDialog<HTMLDivElement>(true, onClose);

    const previewReturnPct = useMemo(() => {
        const ep = Number(exitPrice);
        if (!Number.isFinite(ep) || ep <= 0 || position.entry_price <= 0) return null;
        return Number((((ep - position.entry_price) / position.entry_price) * 100).toFixed(2));
    }, [exitPrice, position.entry_price]);

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        const ep = Number(exitPrice);
        if (!Number.isFinite(ep) || ep <= 0) {
            setError('Geçerli bir çıkış fiyatı girin.');
            return;
        }
        setSaving(true);
        setError(null);
        try {
            const res = await apiClient.modelPortfolioClosePosition(
                position.id,
                {
                    exit_price: ep,
                    closed_date: closedDate,
                    exit_xu100: currentXu100,
                    close_reason: closeReason,
                },
                adminToken,
            );
            onClosed(res);
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'Pozisyon kapatılamadı.');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="mp-modal-backdrop" onClick={onClose}>
            <div
                ref={dialogRef}
                className="mp-modal mp-modal-md"
                role="dialog"
                aria-modal="true"
                aria-labelledby="mp-close-dialog-title"
                tabIndex={-1}
                onClick={(e) => e.stopPropagation()}
            >
                <div className="mp-modal-header">
                    <div className="mp-modal-title-row">
                        <div className="mp-modal-icon-badge is-close">
                            <CheckCircle2 size={15} />
                        </div>
                        <div>
                            <h3 id="mp-close-dialog-title">{position.symbol} Pozisyonunu Kapat</h3>
                            <span className="mp-modal-subtitle">
                                Gerçekleşen getiri ve BIST 100 rölatif alfa ile geçmiş işlemlere arşivlenir.
                            </span>
                        </div>
                    </div>
                    <button type="button" className="mp-icon-btn" aria-label="Kapatma penceresini kapat" onClick={onClose}>
                        <X size={15} />
                    </button>
                </div>
                <form onSubmit={handleSubmit} className="mp-editor-shell">
                    <div className="mp-modal-body">
                        <div className="mp-form-grid-2">
                            <label className="mp-field">
                                <div className="mp-field-label-row">
                                    <span className="mp-field-label">Çıkış Fiyatı (₺) *</span>
                                    {previewReturnPct != null && (
                                        <span className={`mp-upside-live-pill ${toneClass(previewReturnPct)}`}>
                                            Net Getiri: {formatPct(previewReturnPct)}
                                        </span>
                                    )}
                                </div>
                                <input
                                    type="number"
                                    data-dialog-autofocus
                                    step="0.01"
                                    value={exitPrice}
                                    onChange={(e) => setExitPrice(e.target.value)}
                                />
                            </label>
                            <label className="mp-field">
                                <span className="mp-field-label">Kapanış Tarihi *</span>
                                <input
                                    type="date"
                                    value={closedDate}
                                    onChange={(e) => setClosedDate(e.target.value)}
                                />
                            </label>
                        </div>

                        <MpExplainedDropdown
                            label="Kapanış Gerekçesi"
                            value={closeReason}
                            options={CLOSE_REASON_OPTIONS}
                            onChange={setCloseReason}
                        />

                        {error && <div className="mp-form-error">{error}</div>}
                    </div>
                    <div className="mp-modal-footer mp-modal-footer-sticky">
                        <div className="mp-footer-summary">
                            <span>Giriş: <b>{formatPrice(position.entry_price)}</b></span>
                        </div>
                        <div className="mp-footer-btns">
                            <button type="button" className="mp-btn mp-btn-ghost" onClick={onClose}>
                                İptal
                            </button>
                            <button type="submit" className="mp-btn mp-btn-primary" disabled={saving}>
                                {saving ? 'Kapatılıyor...' : 'Pozisyonu Kapat'}
                            </button>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    );
}

export default ModelPortfolioCloseModal;
