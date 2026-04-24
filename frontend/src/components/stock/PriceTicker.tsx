import { useEffect, useState } from 'react';
import { Star } from 'lucide-react';
import { apiClient } from '../../api/client';
import './PriceTicker.css';

type PriceData = {
    ok: boolean;
    symbol: string;
    price: number | null;
    change: number | null;
    change_pct: number | null;
    currency: string;
    market_state: string;
    as_of?: string | null;
    error?: string;
};

function formatAsOf(value?: string | null): string | null {
    if (!value) {
        return null;
    }

    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
        return null;
    }

    const datePart = new Intl.DateTimeFormat('tr-TR', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
    }).format(parsed);

    const timePart = new Intl.DateTimeFormat('tr-TR', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    }).format(parsed);

    return `${datePart}, ${timePart}`;
}

export function PriceTicker({
    symbol,
    companyName,
    priceAsOf,
}: {
    symbol: string;
    companyName?: string;
    priceAsOf?: string | null;
}) {
    const [data, setData] = useState<PriceData | null>(null);

    useEffect(() => {
        if (!symbol) {
            return;
        }

        let cancelled = false;
        apiClient.kapPrice(symbol)
            .then((response) => {
                if (!cancelled) {
                    setData(response as PriceData);
                }
            })
            .catch(() => {
                if (!cancelled) {
                    setData(null);
                }
            });

        return () => {
            cancelled = true;
        };
    }, [symbol]);

    const isUp = (data?.change ?? 0) >= 0;
    const colorClass = isUp ? 'price-up' : 'price-down';
    const resolvedAsOf = formatAsOf(data?.as_of ?? priceAsOf) ?? 'Veri güncelleniyor';
    const currencyLabel =
        !data?.currency || data.currency === 'TRY' || data.currency === 'TL'
            ? '₺'
            : data.currency;

    return (
        <div className="stock-header-band">
            <div className="sh-left">
                <div className="sh-titles">
                    <div className="sh-symbol-row">
                        <h1>{symbol.toUpperCase()}</h1>
                        <button type="button" className="sh-star-btn" aria-label={`${symbol} favori`}>
                            <Star size={15} />
                        </button>
                    </div>
                    <p className="sh-company-name">{companyName || 'Şirket bilgisi yükleniyor...'}</p>
                </div>
            </div>

            <div className="sh-right">
                {data?.ok && data.price != null ? (
                    <>
                        <div className="sh-price-row">
                            <span className="sh-price-currency">{currencyLabel}</span>
                            <span className="sh-price-value">
                                {data.price.toLocaleString('tr-TR', {
                                    minimumFractionDigits: 2,
                                    maximumFractionDigits: 2,
                                })}
                            </span>
                            {data.change_pct != null && (
                                <span className={`sh-price-change ${colorClass}`}>
                                    % {data.change_pct.toLocaleString('tr-TR', {
                                        minimumFractionDigits: 2,
                                        maximumFractionDigits: 2,
                                    })}
                                </span>
                            )}
                        </div>
                        <div className="sh-time-row">{resolvedAsOf}</div>
                    </>
                ) : (
                    <>
                        <div className="sh-price-row">
                            <span className="sh-price-currency">₺</span>
                            <span className="sh-price-value">-</span>
                        </div>
                        <div className="sh-time-row">Fiyat verisi yükleniyor</div>
                    </>
                )}
            </div>
        </div>
    );
}
