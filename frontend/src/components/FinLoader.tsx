interface FinLoaderProps {
    message?: string;
    className?: string;
}

export function FinLoader({ message = 'Yükleniyor', className = '' }: FinLoaderProps) {
    return (
        <div
            className={`fin-loader-container${className ? ` ${className}` : ''}`}
            role="status"
            aria-live="polite"
        >
            <div className="fin-loader-spinner-wrap" aria-hidden="true">
                <div className="fin-loader-ring" />
            </div>
            <span className="fin-loader-text">{message}</span>
        </div>
    );
}
