import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiClient } from '../api/client';
import type { MarketWatchItem, MarketWatchResponse } from '../api/types';
import SymbolLogo, { type SymbolLogoKind } from './SymbolLogo';
import './MarketWatchStrip.css';

const PRIMARY_SYMBOLS = ['XUTUM', 'XU100', 'XU030', 'USD/TRY', 'EUR/TRY'] as const;
const COMMODITY_SYMBOLS = ['BRENT', 'ALTIN', 'GUMUS', 'DOGALGAZ'] as const;
const DEFAULT_DELAY_NOTE = 'Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).';

const FALLBACK_LABELS: Record<string, string> = {
    XUTUM: 'BIST Tüm',
    XU100: 'BIST 100',
    XU030: 'BIST 30',
    'USD/TRY': 'Amerikan Doları',
    'EUR/TRY': 'Euro',
    BRENT: 'Brent Petrol',
    ALTIN: 'Altın (Ons)',
    GUMUS: 'Gümüş (Ons)',
    DOGALGAZ: 'Doğal Gaz',
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
        logo_url: null,
        logo_source: null,
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
    return `%${sign}${changePct.toLocaleString('tr-TR', {
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

function compactSymbol(symbol: string): string {
    return symbol.replace('/', '');
}

function watchItemKind(symbol: string): SymbolLogoKind {
    const normalized = normalizeSymbol(symbol);
    if (COMMODITY_SYMBOLS.includes(normalized as (typeof COMMODITY_SYMBOLS)[number])) {
        return 'commodity';
    }
    if (normalized.includes('/') || normalized === 'DXY') {
        return 'fx';
    }
    return 'index';
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

interface MarketWatchStripProps {
    variant?: 'panel' | 'compact';
}

export default function MarketWatchStrip({ variant = 'panel' }: MarketWatchStripProps) {
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

    const primaryItems = useMemo(
        () => PRIMARY_SYMBOLS.map((symbol) => lookup.get(normalizeSymbol(symbol)) || placeholderItem(symbol)),
        [lookup],
    );

    const commodityItems = useMemo(
        () => COMMODITY_SYMBOLS.map((symbol) => lookup.get(normalizeSymbol(symbol)) || placeholderItem(symbol)),
        [lookup],
    );

    const isInitialLoading = loading && !payload;
    const delayNote = payload?.delay_note || DEFAULT_DELAY_NOTE;

    if (variant === 'compact') {
        return (
            <section className="mw-strip mw-strip-compact" aria-label="Kompakt piyasa bandı">
                {isInitialLoading && (
                    <div className="mw-compact-skeleton-grid">
                        {[...Array(PRIMARY_SYMBOLS.length)].map((_, i) => (
                            <div key={i} className="mw-compact-skeleton-card">
                                <div className="mw-compact-skeleton-head">
                                    <div className="mw-compact-skeleton-logo mw-compact-skeleton-pulse" />
                                    <div className="mw-compact-skeleton-symbol mw-compact-skeleton-pulse" />
                                </div>
                                <div className="mw-compact-skeleton-price-row">
                                    <div className="mw-compact-skeleton-price mw-compact-skeleton-pulse" />
                                    <div className="mw-compact-skeleton-change mw-compact-skeleton-pulse" />
                                </div>
                            </div>
                        ))}
                    </div>
                )}
                {!isInitialLoading && !payload && error && (
                    <div className="mw-compact-state mw-compact-error">{error}</div>
                )}
                {(payload || !isInitialLoading) && (
                    <div className="mw-compact-grid">
                        {primaryItems.map((item) => (
                            <FlashMarketCard
                                key={item.symbol}
                                item={item}
                                className="mw-card-compact"
                            >
                                <div className="mw-compact-head">
                                    <SymbolLogo
                                        symbol={item.symbol}
                                        name={item.label}
                                        kind={watchItemKind(item.symbol)}
                                        logoUrl={item.logo_url}
                                        size="xs"
                                        className="mw-symbol-logo"
                                    />
                                    <div className="mw-compact-symbol">{compactSymbol(item.symbol)}</div>
                                </div>
                                <div className="mw-compact-price-row">
                                    <span className="mw-compact-price">{formatPrice(item)}</span>
                                    <span className={`mw-compact-change ${changeClass(item.change_pct)}`}>
                                        {formatPct(item.change_pct)}
                                    </span>
                                </div>
                            </FlashMarketCard>
                        ))}
                    </div>
                )}
            </section>
        );
    }

    return (
        <section className="mw-strip" aria-label="Borsa izleme bandı">
            <div className="mw-strip-header">
                <div className="mw-copy">
                    <h2>Borsa İzleme Bandı</h2>
                    <p>XUTUM, XU100, XU030, döviz ve emtia özetleri 3 saniyede bir otomatik güncellenir.</p>
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
                <div className="mw-skeleton-full">
                    <div className="mw-grid mw-grid-main">
                        {[...Array(4)].map((_, i) => (
                            <div key={i} className="mw-card mw-card-main mw-card-skeleton-pulse">
                                <div className="mw-card-head">
                                    <div className="mw-skeleton-rect mw-skeleton-logo-full mw-compact-skeleton-pulse" />
                                    <div className="mw-skeleton-rect mw-skeleton-label-full mw-compact-skeleton-pulse" />
                                </div>
                                <div className="mw-skeleton-rect mw-skeleton-price-full mw-compact-skeleton-pulse" />
                                <div className="mw-skeleton-rect mw-skeleton-change-full mw-compact-skeleton-pulse" />
                            </div>
                        ))}
                    </div>
                    <div className="mw-grid mw-grid-commodities">
                        {[...Array(4)].map((_, i) => (
                            <div key={i} className="mw-card mw-card-skeleton-pulse">
                                <div className="mw-card-head">
                                    <div className="mw-skeleton-rect mw-skeleton-logo-full mw-compact-skeleton-pulse" />
                                    <div className="mw-skeleton-rect mw-skeleton-label-full mw-compact-skeleton-pulse" />
                                </div>
                                <div className="mw-skeleton-rect mw-skeleton-price-full mw-compact-skeleton-pulse" />
                                <div className="mw-skeleton-rect mw-skeleton-change-full mw-compact-skeleton-pulse" />
                            </div>
                        ))}
                    </div>
                </div>
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
                    <div className="mw-grid mw-grid-main">
                        {primaryItems.map((item) => (
                            <FlashMarketCard
                                key={item.symbol}
                                item={item}
                                className="mw-card-main"
                            >
                                <div className="mw-card-head">
                                    <span className="mw-card-symbol">
                                        <SymbolLogo
                                            symbol={item.symbol}
                                            name={item.label}
                                            kind={watchItemKind(item.symbol)}
                                            logoUrl={item.logo_url}
                                            size="xs"
                                            className="mw-symbol-logo"
                                        />
                                        <span className="mw-symbol">{item.symbol}</span>
                                    </span>
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

                    <div className="mw-grid mw-grid-commodities">
                        {commodityItems.map((item) => (
                            <FlashMarketCard
                                key={item.symbol}
                                item={item}
                            >
                                <div className="mw-card-head">
                                    <span className="mw-card-symbol">
                                        <SymbolLogo
                                            symbol={item.symbol}
                                            name={item.label}
                                            kind="commodity"
                                            logoUrl={item.logo_url}
                                            size="xs"
                                            className="mw-symbol-logo"
                                        />
                                        <span className="mw-symbol">{item.symbol}</span>
                                    </span>
                                    <span className="mw-label">{item.label}</span>
                                </div>
                                <div className="mw-card-price">{formatPrice(item)}</div>
                                <div className={`mw-card-change ${changeClass(item.change_pct)}`}>
                                    {formatPct(item.change_pct)}
                                </div>
                            </FlashMarketCard>
                        ))}
                    </div>

                    <div className="mw-foot">
                        <span className="mw-delay-note">{delayNote}</span>
                        {!loading && error && (
                            <span className="mw-soft-error">Son yenileme başarısız oldu.</span>
                        )}
                    </div>
                </>
            )}
        </section>
    );
}
