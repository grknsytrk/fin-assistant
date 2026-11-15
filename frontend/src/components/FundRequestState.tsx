import { AlertCircle } from 'lucide-react';
import './FundRequestState.css';

interface FundRequestStateProps {
    mode: 'loading' | 'error';
    symbol?: string | null;
    fundName?: string | null;
    compact?: boolean;
    onRetry?: () => void;
}

export default function FundRequestState({
    mode,
    symbol,
    fundName,
    compact = false,
    onRetry,
}: FundRequestStateProps) {
    const isLoading = mode === 'loading';
    const normalizedSymbol = String(symbol || '').trim().toUpperCase();

    if (isLoading) {
        const statusLabel = normalizedSymbol
            ? `${normalizedSymbol} fon bilgileri yükleniyor`
            : 'Fon bilgileri yükleniyor';

        return (
            <section
                className={`fund-request-state loading${compact ? ' compact' : ''}`}
                role="status"
                aria-live="polite"
                aria-label={statusLabel}
            >
                <div className="fund-request-skeleton-hero" aria-hidden="true">
                    <div className="fund-request-skeleton-title-group">
                        <div className="fund-request-skeleton-badge-row">
                            {normalizedSymbol ? (
                                <span className="fund-request-symbol-chip">{normalizedSymbol}</span>
                            ) : (
                                <span className="fund-request-sk-block fund-request-sk-code" />
                            )}
                            <span className="fund-request-sk-block fund-request-sk-kicker" />
                        </div>
                        {fundName ? (
                            <p className="fund-request-loading-name">{fundName}</p>
                        ) : (
                            <span className="fund-request-sk-block fund-request-sk-title" />
                        )}
                        <span className="fund-request-sk-block fund-request-sk-subtitle" />
                    </div>
                    <div className="fund-request-skeleton-quote">
                        <span className="fund-request-sk-block fund-request-sk-price" />
                        <span className="fund-request-sk-block fund-request-sk-delta" />
                    </div>
                </div>

                <div className="fund-request-skeleton-stats" aria-hidden="true">
                    <div className="fund-request-skeleton-stat-card">
                        <span className="fund-request-sk-block fund-request-sk-stat-label" />
                        <span className="fund-request-sk-block fund-request-sk-stat-value" />
                    </div>
                    <div className="fund-request-skeleton-stat-card">
                        <span className="fund-request-sk-block fund-request-sk-stat-label" />
                        <span className="fund-request-sk-block fund-request-sk-stat-value" />
                    </div>
                    <div className="fund-request-skeleton-stat-card">
                        <span className="fund-request-sk-block fund-request-sk-stat-label" />
                        <span className="fund-request-sk-block fund-request-sk-stat-value" />
                    </div>
                    <div className="fund-request-skeleton-stat-card">
                        <span className="fund-request-sk-block fund-request-sk-stat-label" />
                        <span className="fund-request-sk-block fund-request-sk-stat-value" />
                    </div>
                </div>

                {!compact ? (
                    <div className="fund-request-skeleton-grid" aria-hidden="true">
                        <div className="fund-request-skeleton-panel">
                            <span className="fund-request-sk-block fund-request-sk-panel-head" />
                            <span className="fund-request-sk-block fund-request-sk-panel-body" />
                        </div>
                        <div className="fund-request-skeleton-panel">
                            <span className="fund-request-sk-block fund-request-sk-panel-head" />
                            <span className="fund-request-sk-block fund-request-sk-panel-body" />
                        </div>
                    </div>
                ) : null}
            </section>
        );
    }

    return (
        <section
            className={`fund-request-state error${compact ? ' compact' : ''}`}
            role="alert"
            aria-live="assertive"
        >
            <div className="fund-request-error-icon" aria-hidden="true">
                <AlertCircle size={18} />
            </div>
            <div className="fund-request-copy">
                <p className="fund-request-kicker">
                    Yatırım Fonu
                    {normalizedSymbol ? (
                        <span className="fund-request-symbol-chip">{normalizedSymbol}</span>
                    ) : null}
                </p>
                <h2>Fon bilgisine şu an ulaşılamıyor</h2>
                {fundName ? <p className="fund-request-name">{fundName}</p> : null}
                <p className="fund-request-description">
                    İstek zamanında tamamlanamadı. Sayfayı yenilemeden tekrar deneyebilirsiniz.
                </p>
            </div>
            {onRetry ? (
                <div className="fund-request-actions">
                    <button type="button" className="fund-request-retry" onClick={onRetry}>
                        Tekrar Dene
                    </button>
                </div>
            ) : null}
        </section>
    );
}
