import { CheckCircle2, ExternalLink, Pencil, Trash2, X } from 'lucide-react';
import { Link } from 'react-router-dom';
import type { ModelPortfolioPosition } from '../../../api/types';
import SymbolLogo from '../../../components/SymbolLogo';
import { DEFAULT_STOCK_TAB, toStockDetail } from '../../../routing/routes';
import { useAccessibleDialog } from '../hooks/useAccessibleDialog';
import {
    formatDateTR,
    formatMultiple,
    formatPct,
    formatPrice,
    toneClass,
} from '../utils';
import './ModelPortfolioThesisDrawer.css';

export function ModelPortfolioThesisDrawer({
    position,
    isAdmin,
    onClose,
    onEdit,
    onClosePosition,
    onDeletePosition,
}: {
    position: ModelPortfolioPosition;
    isAdmin: boolean;
    onClose: () => void;
    onEdit: (pos: ModelPortfolioPosition) => void;
    onClosePosition: (pos: ModelPortfolioPosition) => void;
    onDeletePosition: (pos: ModelPortfolioPosition) => void;
}) {
    const progress = position.target_progress_pct ?? 0;
    const clampedBar = Math.max(0, Math.min(100, progress));
    const dialogRef = useAccessibleDialog<HTMLElement>(true, onClose);

    const multipleRows = [
        {
            label: 'F/K',
            title: 'Fiyat / Kazanç',
            entry: position.entry_multiples?.fk,
            current: position.current_multiples?.fk,
        },
        {
            label: 'FD/FAVÖK',
            title: 'Firma Değeri / FAVÖK',
            entry: position.entry_multiples?.fd_favok,
            current: position.current_multiples?.fd_favok,
        },
        {
            label: 'PD/DD',
            title: 'Piyasa Değeri / Defter Değeri',
            entry: position.entry_multiples?.pd_dd,
            current: position.current_multiples?.pd_dd,
        },
        {
            label: 'Net Borç/FAVÖK',
            title: 'Net Borç / FAVÖK',
            entry: position.entry_multiples?.net_borc_favok,
            current: position.current_multiples?.net_borc_favok,
        },
    ];

    return (
        <div className="mp-drawer-backdrop" onClick={onClose}>
            <aside
                ref={dialogRef}
                className="mp-drawer"
                role="dialog"
                aria-modal="true"
                aria-labelledby="mp-thesis-dialog-title"
                tabIndex={-1}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Drawer Header */}
                <div className="mp-drawer-header">
                    <div className="mp-drawer-identity">
                        <SymbolLogo
                            symbol={position.symbol}
                            name={position.company_name}
                            kind="stock"
                            logoUrl={position.logo_url}
                            size="md"
                        />
                        <div>
                            <div className="mp-drawer-title-row">
                                <h2 id="mp-thesis-dialog-title">{position.symbol}</h2>
                                <span className="mp-rating-badge">{position.rating || 'Endeks Üstü Getiri'}</span>
                                {position.status === 'closed' && (
                                    <span className="mp-closed-badge">KAPATILDI</span>
                                )}
                            </div>
                            <p className="mp-drawer-company">{position.sector ? `${position.company_name} · ${position.sector}` : position.company_name}</p>
                        </div>
                    </div>
                    <div className="mp-drawer-header-btns">
                        <Link
                            to={toStockDetail(position.symbol, DEFAULT_STOCK_TAB)}
                            className="mp-btn mp-btn-subtle"
                        >
                            <span>Hisse Detayı</span>
                            <ExternalLink size={13} />
                        </Link>
                        <button type="button" className="mp-icon-btn" onClick={onClose} aria-label="Detay penceresini kapat">
                            <X size={18} />
                        </button>
                    </div>
                </div>

                {/* Drawer Body */}
                <div className="mp-drawer-body">
                    {/* Key figures */}
                    <div className="mp-drawer-summary">
                        <div className="mp-drawer-figure">
                            <span>Giriş fiyatı</span>
                            <strong className="mp-mono">{formatPrice(position.entry_price)}</strong>
                            <small>{formatDateTR(position.entry_date)}</small>
                        </div>
                        <div className="mp-drawer-figure">
                            <span>{position.status === 'closed' ? 'Çıkış fiyatı' : 'Güncel fiyat'}</span>
                            <strong className="mp-mono">{formatPrice(position.current_price)}</strong>
                            {position.daily_change_pct != null && position.status !== 'closed' && (
                                <small className={toneClass(position.daily_change_pct)}>
                                    {formatPct(position.daily_change_pct)} bugün
                                </small>
                            )}
                        </div>
                        <div className="mp-drawer-figure">
                            <span>Hedef fiyat</span>
                            <strong className="mp-mono">{formatPrice(position.target_price)}</strong>
                            {position.horizon && <small>{position.horizon}</small>}
                        </div>
                        <div className="mp-drawer-figure">
                            <span>Net getiri</span>
                            <strong className={`mp-mono ${toneClass(position.return_pct)}`}>{formatPct(position.return_pct)}</strong>
                        </div>
                        <div className="mp-drawer-figure">
                            <span>XU100’e göre</span>
                            <strong className={`mp-mono ${toneClass(position.relative_return_pct)}`}>
                                {formatPct(position.relative_return_pct)}
                            </strong>
                        </div>
                        {position.status === 'active' && (
                            <div className="mp-drawer-figure">
                                <span>Hedef potansiyeli</span>
                                <strong className={`mp-mono ${toneClass(position.target_upside_pct)}`}>
                                    {formatPct(position.target_upside_pct)}
                                </strong>
                            </div>
                        )}
                    </div>

                    {position.status === 'active' && (
                        <div className="mp-drawer-progress-box">
                            <div className="mp-drawer-progress-labels">
                                <span>Hedefe ilerleme</span>
                                <span className="mp-mono">
                                    {progress < 0
                                        ? `Giriş fiyatının altında (${formatPct(progress, false)})`
                                        : `%${progress.toLocaleString('tr-TR', { maximumFractionDigits: 1 })}`}
                                </span>
                            </div>
                            <div className="mp-progress-track mp-progress-track-lg">
                                <div
                                    className={`mp-progress-fill${position.target_reached ? ' is-reached' : ''}`}
                                    style={{ width: `${clampedBar}%` }}
                                />
                            </div>
                        </div>
                    )}

                    {/* Investment Thesis Section */}
                    <section className="mp-drawer-section">
                        <h3>Yatırım tezi</h3>
                        <div className="mp-thesis-box">
                            {position.thesis ? (
                                position.thesis.split('\n').map((para, i) => (
                                    <p key={i}>{para}</p>
                                ))
                            ) : (
                                <p className="mp-muted">Bu pozisyon için henüz detaylı yatırım tezi metni girilmemiş.</p>
                            )}
                        </div>
                    </section>

                    {/* Catalysts / Expectations */}
                    {position.catalysts && position.catalysts.length > 0 && (
                        <section className="mp-drawer-section">
                            <h3>Katalizörler</h3>
                            <div className="mp-catalyst-tags">
                                {position.catalysts.map((cat) => (
                                    <span key={cat} className="mp-catalyst-chip">{cat}</span>
                                ))}
                            </div>
                        </section>
                    )}

                    {/* Entry Day vs Current Valuation Multiples */}
                    <section className="mp-drawer-section">
                        <h3>Çarpanlar</h3>
                        <div className="mp-multiples-compare-table-wrap">
                            <table className="mp-multiples-table">
                                <thead>
                                    <tr>
                                        <th>Çarpan</th>
                                        <th className="num">Girişte</th>
                                        <th className="num">Güncel</th>
                                        <th className="num">Fark</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {multipleRows.map((row) => {
                                        const diff =
                                            row.entry != null && row.current != null
                                                ? row.current - row.entry
                                                : null;
                                        return (
                                            <tr key={row.label}>
                                                <td title={row.title}>{row.label}</td>
                                                <td className="num mp-mono">{formatMultiple(row.entry)}</td>
                                                <td className="num mp-mono mp-bold">{formatMultiple(row.current)}</td>
                                                <td className="num mp-mono">
                                                    {diff != null ? (
                                                        <span className={diff <= 0 ? 'mp-tone-up' : 'mp-tone-flat'}>
                                                            {diff > 0 ? '+' : ''}
                                                            {diff.toFixed(2)}x
                                                        </span>
                                                    ) : (
                                                        '—'
                                                    )}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </section>

                    {/* Risk Note */}
                    {position.risk_note && (
                        <section className="mp-drawer-section">
                            <h3>Riskler</h3>
                            <div className="mp-risk-box">
                                <p>{position.risk_note}</p>
                            </div>
                        </section>
                    )}

                    {/* Target Price & Weight Revision History */}
                    {position.revisions && position.revisions.length > 0 && (
                        <section className="mp-drawer-section">
                            <h3>Revizyon geçmişi</h3>
                            <div className="mp-revisions-list">
                                {position.revisions.map((rev) => (
                                    <div key={rev.id} className="mp-revision-item">
                                        <div className="mp-revision-top">
                                            <span className="mp-mono mp-revision-date">{formatDateTR(rev.date)}</span>
                                            <span className="mp-mono">
                                                Hedef: <s>{formatPrice(rev.prev_target_price)}</s> →{' '}
                                                <b>{formatPrice(rev.new_target_price)}</b>
                                            </span>
                                            {rev.prev_weight_pct !== rev.new_weight_pct && (
                                                <span className="mp-mono">
                                                    Ağırlık: %{rev.prev_weight_pct} → <b>%{rev.new_weight_pct}</b>
                                                </span>
                                            )}
                                        </div>
                                        {rev.note && <p className="mp-revision-note">{rev.note}</p>}
                                    </div>
                                ))}
                            </div>
                        </section>
                    )}
                </div>

                {/* Drawer Admin Footer */}
                {isAdmin && (
                    <div className="mp-drawer-footer">
                        {position.status === 'active' && (
                            <>
                                <button
                                    type="button"
                                    className="mp-btn mp-btn-subtle"
                                    onClick={() => onEdit(position)}
                                >
                                    <Pencil size={14} /> Düzenle / Revize Et
                                </button>
                                <button
                                    type="button"
                                    className="mp-btn mp-btn-primary"
                                    onClick={() => onClosePosition(position)}
                                >
                                    <CheckCircle2 size={14} /> Pozisyonu Kapat
                                </button>
                            </>
                        )}
                        <button
                            type="button"
                            className="mp-btn mp-btn-danger"
                            onClick={() => onDeletePosition(position)}
                        >
                            <Trash2 size={14} /> Sil
                        </button>
                    </div>
                )}
            </aside>
        </div>
    );
}
