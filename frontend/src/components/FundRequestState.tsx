import './FundRequestState.css';

interface FundRequestStateProps {
    mode: 'loading' | 'error';
    symbol?: string | null;
    fundName?: string | null;
    compact?: boolean;
    onRetry?: () => void;
}

function FundPortfolioAnimation() {
    return (
        <svg
            className="fund-request-art"
            viewBox="0 0 228 144"
            aria-hidden="true"
            focusable="false"
        >
            <rect className="fund-request-card" x="42" y="13" width="126" height="104" rx="10" />
            <rect className="fund-request-mark" x="54" y="25" width="24" height="24" rx="6" />
            <path className="fund-request-mark-line" d="M61 42V34l5 3 5-6v11" />
            <path className="fund-request-copy-line" d="M87 31h48M87 40h34" />

            <rect className="fund-request-chart-panel" x="54" y="57" width="102" height="47" rx="5" />
            <path className="fund-request-grid-line" d="M63 91h84M63 78h84M63 65h84" />
            <g className="fund-request-bars">
                <rect x="66" y="84" width="8" height="10" rx="2" />
                <rect x="81" y="77" width="8" height="17" rx="2" />
                <rect x="96" y="80" width="8" height="14" rx="2" />
                <rect x="111" y="69" width="8" height="25" rx="2" />
                <rect x="126" y="63" width="8" height="31" rx="2" />
            </g>
            <path className="fund-request-trend" d="m65 83 18-8 18 4 17-14 17 4" />

            <g className="fund-request-units">
                <path d="M177 93c0-3.6 8.1-6.5 18-6.5s18 2.9 18 6.5v17c0 3.6-8.1 6.5-18 6.5s-18-2.9-18-6.5V93Z" />
                <ellipse cx="195" cy="93" rx="18" ry="6.5" />
                <path d="M177 101c0 3.6 8.1 6.5 18 6.5s18-2.9 18-6.5" />
                <path d="M177 109c0 3.6 8.1 6.5 18 6.5s18-2.9 18-6.5" />
            </g>
        </svg>
    );
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

    return (
        <section
            className={`fund-request-state ${mode}${compact ? ' compact' : ''}`}
            role={isLoading ? 'status' : 'alert'}
            aria-live={isLoading ? 'polite' : 'assertive'}
        >
            <div className="fund-request-illustration">
                <FundPortfolioAnimation />
            </div>
            <div className="fund-request-copy">
                <p className="fund-request-kicker">
                    Yatırım fonu{normalizedSymbol ? <span>{normalizedSymbol}</span> : null}
                </p>
                <h2>{isLoading ? 'Fon bilgileri hazırlanıyor' : 'Fon bilgisine şu an ulaşamadık'}</h2>
                {fundName ? <p className="fund-request-name">{fundName}</p> : null}
                <p className="fund-request-description">
                    {isLoading
                        ? 'Fonun fiyat ve portföy verileri getiriliyor.'
                        : 'İstek zamanında tamamlanmadı. Sayfayı yenilemeden tekrar deneyebilirsiniz.'}
                </p>
                {!isLoading && onRetry ? (
                    <button type="button" className="fund-request-retry" onClick={onRetry}>
                        Yeniden dene
                    </button>
                ) : null}
            </div>
        </section>
    );
}
