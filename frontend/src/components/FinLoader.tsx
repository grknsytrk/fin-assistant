
interface FinLoaderProps {
    message?: string;
    className?: string;
}

export function FinLoader({ message = 'Yükleniyor', className = '' }: FinLoaderProps) {
    return (
        <div className={`fin-loader-container ${className}`}>
            <div className="fin-loader-spinner-wrap">
                <div className="fin-loader-pulse-ring" />
                <div className="fin-loader-ring-outer" />
                <div className="fin-loader-ring-inner" />
            </div>
            <span className="fin-loader-text">{message}</span>
        </div>
    );
}
