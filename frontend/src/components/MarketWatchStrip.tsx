import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiClient } from '../api/client';
import type { MarketWatchItem, MarketWatchResponse } from '../api/types';
import SymbolLogo, { type SymbolLogoKind } from './SymbolLogo';
import { getAdaptiveRefreshMs } from '../utils/marketHours';
import './MarketWatchStrip.css';

const PRIMARY_SYMBOLS = ['XUTUM', 'XU100', 'XU030', 'USD/TRY', 'EUR/TRY'] as const;
const COMMODITY_SYMBOLS = ['BRENT', 'ALTIN', 'GUMUS', 'DOGALGAZ'] as const;
const DEFAULT_DELAY_NOTE = 'Yahoo Finance sağlayıcı gecikmeli veri (ortalama ~15dk).';
let marketWatchMemoryCache: MarketWatchResponse | null = null;

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
    if (item.price == null) return '—';

    let prefix = '';
    let suffix = '';
    if (item.currency === 'TRY') suffix = ' ₺';
    else if (item.currency === 'USD') prefix = '$';
    else if (item.currency === 'EUR') prefix = '€';
    else if (item.currency) suffix = ` ${item.currency}`;

    const decimals = pickDecimals(item.price);
    return `${prefix}${item.price.toLocaleString('tr-TR', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    })}${suffix}`;
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
    if (
        normalized.includes('/') ||
        normalized === 'DXY' ||
        normalized.endsWith('TRY') ||
        normalized === 'EURUSD' ||
        normalized === 'GBPUSD' ||
        normalized === 'USDJPY'
    ) {
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
    const [payload, setPayload] = useState<MarketWatchResponse | null>(() => marketWatchMemoryCache);
    const [loading, setLoading] = useState(() => !marketWatchMemoryCache);
    const [error, setError] = useState<string | null>(null);
    const inFlightRef = useRef(false);

    const loadWatch = useCallback(async (options?: { refresh?: boolean; silent?: boolean }) => {
        if (inFlightRef.current) return;
        inFlightRef.current = true;
        const silent = Boolean(options?.silent);
        if (!silent) {
            setLoading(true);
            setError(null);
        }
        try {
            const response = await apiClient.marketWatch({ refresh: options?.refresh });
            marketWatchMemoryCache = response;
            setPayload(response);
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : 'Borsa izleme verisi alınamadı.';
            if (!silent) setError(msg);
        } finally {
            inFlightRef.current = false;
            if (!silent) setLoading(false);
        }
    }, []);

    useEffect(() => {
        void loadWatch(marketWatchMemoryCache ? { silent: true } : undefined);
        let pollTimer: number | null = null;
        const scheduleAdaptiveTimer = () => {
            if (pollTimer !== null) {
                window.clearInterval(pollTimer);
                pollTimer = null;
            }
            const intervalMs = getAdaptiveRefreshMs('global_fx');
            if (intervalMs !== null) {
                pollTimer = window.setInterval(() => {
                    if (document.visibilityState === 'visible') {
                        void loadWatch({ silent: true });
                    }
                }, intervalMs);
            }
        };
        scheduleAdaptiveTimer();
        const phaseCheckTimer = window.setInterval(scheduleAdaptiveTimer, 60_000);
        const onVisibilityChange = () => {
            scheduleAdaptiveTimer();
            if (document.visibilityState === 'visible' && getAdaptiveRefreshMs('global_fx') !== null) {
                void loadWatch({ silent: true });
            }
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        const timer = pollTimer ?? undefined;
        return () => {
            if (pollTimer !== null) window.clearInterval(pollTimer);
            window.clearInterval(phaseCheckTimer);
            document.removeEventListener('visibilitychange', onVisibilityChange);
            void timer;
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

    type TickerSegment =
        | { type: 'badge'; category: 'indices' | 'fx' | 'commodities'; label: string }
        | { type: 'item'; item: MarketWatchItem };

    const groupedSegments = useMemo(() => {
        const indexOrder: Record<string, number> = {
            XU100: 1,
            XU030: 2,
            XUTUM: 3,
        };

        const fxOrder: Record<string, number> = {
            'USD/TRY': 1,
            USDTRY: 1,
            'EUR/TRY': 2,
            EURTRY: 2,
            'GBP/TRY': 3,
            GBPTRY: 3,
            'EUR/USD': 4,
            EURUSD: 4,
            DXY: 5,
        };

        const commodityOrder: Record<string, number> = {
            ALTIN: 1,
            GUMUS: 2,
            BRENT: 3,
            DOGALGAZ: 4,
        };

        // 1. Indices (BIST)
        const indices: MarketWatchItem[] = [];
        const seenIndices = new Set<string>();
        if (payload?.sections?.indices) {
            for (const item of payload.sections.indices) {
                const norm = normalizeSymbol(item.symbol);
                if (norm && !seenIndices.has(norm)) {
                    seenIndices.add(norm);
                    indices.push(item);
                }
            }
        }
        for (const sym of ['XU100', 'XU030', 'XUTUM']) {
            const norm = normalizeSymbol(sym);
            if (!seenIndices.has(norm)) {
                seenIndices.add(norm);
                indices.push(lookup.get(norm) || placeholderItem(sym));
            }
        }
        indices.sort((a, b) => {
            const pA = indexOrder[normalizeSymbol(a.symbol)] ?? 99;
            const pB = indexOrder[normalizeSymbol(b.symbol)] ?? 99;
            return pA - pB;
        });

        // 2. Forex / Currencies
        const fx: MarketWatchItem[] = [];
        const seenFx = new Set<string>();
        if (payload?.sections?.fx) {
            for (const item of payload.sections.fx) {
                const norm = normalizeSymbol(item.symbol);
                if (norm && !seenFx.has(norm)) {
                    seenFx.add(norm);
                    fx.push(item);
                }
            }
        }
        for (const sym of ['USD/TRY', 'EUR/TRY', 'GBP/TRY', 'EUR/USD']) {
            const norm = normalizeSymbol(sym);
            if (!seenFx.has(norm)) {
                seenFx.add(norm);
                fx.push(lookup.get(norm) || placeholderItem(sym));
            }
        }
        fx.sort((a, b) => {
            const pA = fxOrder[normalizeSymbol(a.symbol)] ?? 99;
            const pB = fxOrder[normalizeSymbol(b.symbol)] ?? 99;
            return pA - pB;
        });

        // 3. Commodities
        const commodities: MarketWatchItem[] = [];
        const seenCommodities = new Set<string>();
        if (payload?.sections?.commodities) {
            for (const item of payload.sections.commodities) {
                const norm = normalizeSymbol(item.symbol);
                if (norm && !seenCommodities.has(norm)) {
                    seenCommodities.add(norm);
                    commodities.push(item);
                }
            }
        }
        for (const sym of ['ALTIN', 'GUMUS', 'BRENT', 'DOGALGAZ']) {
            const norm = normalizeSymbol(sym);
            if (!seenCommodities.has(norm)) {
                seenCommodities.add(norm);
                commodities.push(lookup.get(norm) || placeholderItem(sym));
            }
        }
        commodities.sort((a, b) => {
            const pA = commodityOrder[normalizeSymbol(a.symbol)] ?? 99;
            const pB = commodityOrder[normalizeSymbol(b.symbol)] ?? 99;
            return pA - pB;
        });

        const singleSequence: TickerSegment[] = [];

        if (indices.length > 0) {
            singleSequence.push({ type: 'badge', category: 'indices', label: 'ENDEKS' });
            for (const item of indices) {
                singleSequence.push({ type: 'item', item });
            }
        }

        if (fx.length > 0) {
            singleSequence.push({ type: 'badge', category: 'fx', label: 'DÖVİZ' });
            for (const item of fx) {
                singleSequence.push({ type: 'item', item });
            }
        }

        if (commodities.length > 0) {
            singleSequence.push({ type: 'badge', category: 'commodities', label: 'EMTİA' });
            for (const item of commodities) {
                singleSequence.push({ type: 'item', item });
            }
        }

        return singleSequence;
    }, [payload, lookup]);

    const marqueeSegments = useMemo(() => {
        if (groupedSegments.length === 0) return [];
        const repeatCount = Math.max(2, Math.ceil(24 / groupedSegments.length));
        const oneHalf: TickerSegment[] = [];
        for (let i = 0; i < repeatCount; i++) {
            for (const seg of groupedSegments) {
                oneHalf.push(seg);
            }
        }
        return [...oneHalf, ...oneHalf];
    }, [groupedSegments]);

    const isInitialLoading = loading && !payload;
    const delayNote = payload?.delay_note || DEFAULT_DELAY_NOTE;

    if (variant === 'compact') {
        return (
            <div className="mw-strip mw-strip-compact" aria-label="Canlı borsa ve döviz akış bandı">
                {isInitialLoading && (
                    <div className="mw-ticker-skeleton">
                        {[...Array(6)].map((_, i) => (
                            <div key={i} className="mw-ticker-skeleton-item">
                                <div className="mw-compact-skeleton-symbol mw-compact-skeleton-pulse" />
                                <div className="mw-compact-skeleton-price mw-compact-skeleton-pulse" />
                                <div className="mw-compact-skeleton-change mw-compact-skeleton-pulse" />
                            </div>
                        ))}
                    </div>
                )}
                {!isInitialLoading && !payload && error && (
                    <div className="mw-compact-state mw-compact-error">{error}</div>
                )}
                {(payload || !isInitialLoading) && (
                    <div
                        className="mw-ticker-viewport"
                        tabIndex={0}
                        role="region"
                        aria-label="Canlı borsa ve döviz akış bandı. İmleç ile durdurabilirsiniz."
                    >
                        <div className="mw-ticker-track">
                            {marqueeSegments.map((seg, index) => {
                                if (seg.type === 'badge') {
                                    return (
                                        <div
                                            key={`seg-badge-${seg.category}-${index}`}
                                            className={`mw-ticker-badge mw-badge-${seg.category}`}
                                            aria-hidden="true"
                                        >
                                            <span className="mw-badge-dot" />
                                            <span>{seg.label}</span>
                                        </div>
                                    );
                                }

                                const item = seg.item;
                                const isPositive = item.change_pct != null && item.change_pct > 0;
                                const isNegative = item.change_pct != null && item.change_pct < 0;
                                return (
                                    <div
                                        key={`seg-item-${item.symbol}-${index}`}
                                        className={`mw-ticker-item ${item.price == null ? 'is-muted' : ''}`}
                                        title={`${item.label} · Son güncelleme: ${itemMetaLabel(item)}`}
                                    >
                                        <div className="mw-ticker-identity">
                                            <SymbolLogo
                                                symbol={item.symbol}
                                                name={item.label}
                                                kind={watchItemKind(item.symbol)}
                                                logoUrl={item.logo_url}
                                                size="xs"
                                                className="mw-ticker-logo"
                                            />
                                            <span className="mw-ticker-symbol">{compactSymbol(item.symbol)}</span>
                                        </div>
                                        <span className="mw-ticker-price">{formatPrice(item)}</span>
                                        <span className={`mw-ticker-change ${changeClass(item.change_pct)}`}>
                                            {isPositive ? '▲ ' : isNegative ? '▼ ' : ''}
                                            {formatPct(item.change_pct)}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>
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
