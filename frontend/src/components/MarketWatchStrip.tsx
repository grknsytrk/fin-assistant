import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiClient } from '../api/client';
import type { MarketWatchItem, MarketWatchResponse } from '../api/types';
import './MarketWatchStrip.css';

const INDEX_SYMBOLS = ['XU100', 'XU030', 'VIX', 'S&P 500', 'NASDAQ', 'DOW'] as const;
const FX_SYMBOLS = ['USD/TRY', 'EUR/TRY'] as const;
const COMMODITY_SYMBOLS = ['BRENT', 'ALTIN', 'GUMUS', 'DOGALGAZ'] as const;
const WATCHLIST_SYMBOLS = ['BIMAS', 'MGROS', 'SOKM'] as const;
const DEFAULT_DELAY_NOTE = 'Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).';

const FALLBACK_LABELS: Record<string, string> = {
    XU100: 'BIST 100',
    XU030: 'BIST 30',
    'USD/TRY': 'Amerikan Doları',
    'EUR/TRY': 'Euro',
    BRENT: 'Brent Petrol',
    ALTIN: 'Altın (Ons)',
    GUMUS: 'Gümüş (Ons)',
    DOGALGAZ: 'Doğal Gaz',
    VIX: 'Korku Endeksi (VIX)',
    'S&P 500': 'S&P 500',
    NASDAQ: 'NASDAQ Composite',
    DOW: 'Dow Jones Industrial',
    BIMAS: 'BİM Birleşik Mağazalar',
    MGROS: 'Migros Ticaret',
    SOKM: 'Şok Marketler',
};

function normalizeSymbol(raw: string): string {
    return String(raw || '').trim().toUpperCase();
}

function placeholderItem(symbol: string): MarketWatchItem {
    return {
        symbol,
        label: FALLBACK_LABELS[symbol] || symbol,
        yahoo_symbol: null,
        price: null,
        prev_close: null,
        change: null,
        change_pct: null,
        currency: '',
        market_state: '',
        as_of: null,
        error: 'data_unavailable',
    };
}

function pickDecimals(value: number | null): number {
    if (value == null) return 2;
    const abs = Math.abs(value);
    if (abs < 10) return 4;
    if (abs < 100) return 3;
    return 2;
}

function formatPrice(item: MarketWatchItem): string {
    if (item.price == null) return '-';

    let prefix = '';
    if (item.currency === 'TRY') prefix = '₺';
    else if (item.currency === 'USD') prefix = '$';
    else if (item.currency === 'EUR') prefix = '€';
    else if (item.currency) prefix = `${item.currency} `;

    const decimals = pickDecimals(item.price);
    return `${prefix}${item.price.toLocaleString('tr-TR', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    })}`;
}

function formatPct(changePct: number | null): string {
    if (changePct == null) return 'Veri bekleniyor';
    const sign = changePct > 0 ? '+' : '';
    return `% ${sign}${changePct.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function changeClass(changePct: number | null): string {
    if (changePct == null || changePct === 0) return 'mw-change-flat';
    return changePct > 0 ? 'mw-change-up' : 'mw-change-down';
}

function formatClock(rawIso: string | null | undefined): string {
    if (!rawIso) return '--:--';
    const dt = new Date(rawIso);
    if (Number.isNaN(dt.getTime())) return '--:--';
    return dt.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
}

function itemMetaLabel(item: MarketWatchItem): string {
    if (item.error) return 'Veri alınamadı';
    if (item.as_of) return `Saat ${formatClock(item.as_of)}`;
    return 'Zaman bilgisi yok';
}

function FlashMarketCard({
    item,
    className = '',
    children,
}: React.PropsWithChildren<{ item: MarketWatchItem; className?: string }>) {
    const prevPriceRef = useRef(item.price);
    const [flashClass, setFlashClass] = useState('');

    useEffect(() => {
        if (item.price != null && prevPriceRef.current != null && item.price !== prevPriceRef.current) {
            setFlashClass(item.price > prevPriceRef.current ? 'mw-flash-up' : 'mw-flash-down');
            const timer = window.setTimeout(() => setFlashClass(''), 1100);
            prevPriceRef.current = item.price;
            return () => window.clearTimeout(timer);
        }
        prevPriceRef.current = item.price;
    }, [item.price]);

    return (
        <article
            className={`mw-card ${className} ${item.price == null ? 'mw-card-muted' : ''} ${flashClass}`.trim()}
        >
            {children}
        </article>
    );
}

export default function MarketWatchStrip() {
    const [payload, setPayload] = useState<MarketWatchResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const inFlightRef = useRef(false);

    const loadWatch = useCallback(async (options?: { refresh?: boolean }) => {
        if (inFlightRef.current) return;
        inFlightRef.current = true;
        setLoading(true);
        setError(null);
        try {
            const response = await apiClient.marketWatch({ refresh: options?.refresh });
            setPayload(response);
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : 'Borsa izleme verisi alınamadı.';
            setError(msg);
        } finally {
            inFlightRef.current = false;
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        void loadWatch();
        const timer = window.setInterval(() => {
            void loadWatch();
        }, 3000);
        return () => {
            window.clearInterval(timer);
        };
    }, [loadWatch]);

    const lookup = useMemo(() => {
        const map = new Map<string, MarketWatchItem>();
        if (!payload?.sections) return map;

        const merged = [
            ...(payload.sections.indices || []),
            ...(payload.sections.fx || []),
            ...(payload.sections.commodities || []),
        ];

        for (const item of merged) {
            const key = normalizeSymbol(item.symbol);
            if (!key) continue;
            map.set(key, item);
        }
        return map;
    }, [payload]);

    const indexItems = useMemo(
        () => INDEX_SYMBOLS.map((symbol) => lookup.get(normalizeSymbol(symbol)) || placeholderItem(symbol)),
        [lookup],
    );
    const fxItems = useMemo(
        () => FX_SYMBOLS.map((symbol) => lookup.get(normalizeSymbol(symbol)) || placeholderItem(symbol)),
        [lookup],
    );
    const commodityItems = useMemo(
        () => COMMODITY_SYMBOLS.map((symbol) => lookup.get(normalizeSymbol(symbol)) || placeholderItem(symbol)),
        [lookup],
    );
    const watchlistItems = useMemo(
        () => WATCHLIST_SYMBOLS.map((symbol) => lookup.get(normalizeSymbol(symbol)) || placeholderItem(symbol)),
        [lookup],
    );

    const isInitialLoading = loading && !payload;
    const delayNote = payload?.delay_note || DEFAULT_DELAY_NOTE;

    return (
        <section className="mw-strip" aria-label="Borsa izleme bandı">
            <div className="mw-strip-header">
                <div className="mw-copy">
                    <h2>Borsa İzleme Bandı</h2>
                    <p>XU100, XU030, döviz ve emtia özetleri 3 saniyede bir otomatik güncellenir.</p>
                </div>
                <div className="mw-meta">
                    <span className="mw-updated">Son güncelleme: {formatClock(payload?.as_of)}</span>
                    <button
                        type="button"
                        className="mw-refresh"
                        onClick={() => {
                            void loadWatch({ refresh: true });
                        }}
                        disabled={loading}
                    >
                        Yenile
                    </button>
                </div>
            </div>

            {isInitialLoading && (
                <div className="mw-state">Borsa izleme verisi yükleniyor...</div>
            )}

            {!isInitialLoading && !payload && error && (
                <div className="mw-state mw-state-error">
                    <span>{error}</span>
                    <button
                        type="button"
                        className="mw-refresh"
                        onClick={() => {
                            void loadWatch({ refresh: true });
                        }}
                        disabled={loading}
                    >
                        Tekrar dene
                    </button>
                </div>
            )}

            {payload && (
                <>
                    <div className="mw-section-title">Endeksler</div>
                    <div className="mw-grid mw-grid-indices">
                        {indexItems.map((item) => (
                            <FlashMarketCard key={item.symbol} item={item}>
                                <div className="mw-card-head">
                                    <span className="mw-symbol">{item.symbol}</span>
                                    <span className="mw-label">{item.label}</span>
                                </div>
                                <div className="mw-card-price">{formatPrice(item)}</div>
                                <div className={`mw-card-change ${changeClass(item.change_pct)}`}>
                                    {formatPct(item.change_pct)}
                                </div>
                                <div className="mw-card-meta">{itemMetaLabel(item)}</div>
                            </FlashMarketCard>
                        ))}
                    </div>

                    <div className="mw-section-title">Döviz</div>
                    <div className="mw-grid mw-grid-fx">
                        {fxItems.map((item) => (
                            <FlashMarketCard key={item.symbol} item={item}>
                                <div className="mw-card-head">
                                    <span className="mw-symbol">{item.symbol}</span>
                                    <span className="mw-label">{item.label}</span>
                                </div>
                                <div className="mw-card-price">{formatPrice(item)}</div>
                                <div className={`mw-card-change ${changeClass(item.change_pct)}`}>
                                    {formatPct(item.change_pct)}
                                </div>
                                <div className="mw-card-meta">{itemMetaLabel(item)}</div>
                            </FlashMarketCard>
                        ))}
                    </div>

                    <div className="mw-section-title">Emtia</div>
                    <div className="mw-grid mw-grid-commodities">
                        {commodityItems.map((item) => (
                            <FlashMarketCard key={item.symbol} item={item}>
                                <div className="mw-card-head">
                                    <span className="mw-symbol">{item.symbol}</span>
                                    <span className="mw-label">{item.label}</span>
                                </div>
                                <div className="mw-card-price">{formatPrice(item)}</div>
                                <div className={`mw-card-change ${changeClass(item.change_pct)}`}>
                                    {formatPct(item.change_pct)}
                                </div>
                                <div className="mw-card-meta">{itemMetaLabel(item)}</div>
                            </FlashMarketCard>
                        ))}
                    </div>

                    <div className="mw-section-title">İzleme Listesi (Demo Şirketleri)</div>
                    <div className="mw-grid mw-grid-watchlist">
                        {watchlistItems.map((item) => (
                            <FlashMarketCard key={item.symbol} item={item}>
                                <div className="mw-card-head">
                                    <span className="mw-symbol">{item.symbol}</span>
                                    <span className="mw-label">{item.label}</span>
                                </div>
                                <div className="mw-card-price">{formatPrice(item)}</div>
                                <div className={`mw-card-change ${changeClass(item.change_pct)}`}>
                                    {formatPct(item.change_pct)}
                                </div>
                                <div className="mw-card-meta">{itemMetaLabel(item)}</div>
                            </FlashMarketCard>
                        ))}
                    </div>

                    <div className="mw-foot">
                        <span className="mw-delay-note">{delayNote}</span>
                        {loading && <span className="mw-syncing">Güncelleniyor...</span>}
                        {!loading && error && (
                            <span className="mw-soft-error">Son yenileme başarısız oldu.</span>
                        )}
                    </div>
                </>
            )}
        </section>
    );
}
