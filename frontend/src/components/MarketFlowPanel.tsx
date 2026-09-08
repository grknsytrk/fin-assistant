import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { UIEvent } from 'react';
import { ChevronRight } from 'lucide-react';
import { apiClient } from '../api/client';
import type { MarketFlowItem } from '../api/types';
import { normalizeWatchlistSymbol, useWatchlist } from '../hooks/useWatchlist';

type FlowFilter = 'all' | 'watchlist' | 'ozel_durum' | 'finansal_rapor' | 'kar_payi' | 'genel_kurul' | 'diger';

const FLOW_FILTERS: Array<{ value: FlowFilter; label: string }> = [
    { value: 'all', label: 'Tümü' },
    { value: 'watchlist', label: 'Favorilerim' },
    { value: 'ozel_durum', label: 'Özel Durum' },
    { value: 'finansal_rapor', label: 'Finansal Rapor' },
    { value: 'kar_payi', label: 'Kâr Payı' },
    { value: 'genel_kurul', label: 'Genel Kurul' },
    { value: 'diger', label: 'Diğer' },
];

const FLOW_PAGE_SIZE = 50;
const FLOW_INITIAL_LOAD_SIZE = FLOW_PAGE_SIZE * 2;
const FLOW_MAX_ITEMS = 500;
const FLOW_FAVORITES_LOAD_SIZE = 500;

function getFlowSymbols(item: MarketFlowItem): string[] {
    return [item.symbol, ...(item.stock_codes || []), ...(item.related_symbols || [])]
        .filter(Boolean)
        .map(normalizeWatchlistSymbol);
}

export function getMatchingFavoriteSymbols(item: MarketFlowItem, favoriteSymbols: Set<string>): string[] {
    const itemSymbols = new Set(getFlowSymbols(item));
    return Array.from(favoriteSymbols).filter((symbol) => itemSymbols.has(symbol));
}

function matchesFlowFilter(item: MarketFlowItem, filter: FlowFilter, favoriteSymbols: Set<string>): boolean {
    if (filter === 'all') return true;
    if (filter === 'watchlist') {
        if (favoriteSymbols.size === 0) return false;
        return getMatchingFavoriteSymbols(item, favoriteSymbols).length > 0;
    }
    if (filter === 'ozel_durum') return item.category === 'ozel_durum' || item.source === 'Özel Durum';
    if (filter === 'finansal_rapor') return item.category === 'finansal_rapor';
    if (filter === 'kar_payi') return item.category === 'kar_payi';
    if (filter === 'genel_kurul') return item.category === 'genel_kurul';
    return !['ozel_durum', 'finansal_rapor', 'kar_payi', 'genel_kurul'].includes(item.category);
}

function formatFlowTime(iso: string): string {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return '--:--';
    return date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
}

function formatFlowDate(iso: string): string {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return '';
    return date.toLocaleDateString('tr-TR', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
    });
}

function formatFlowCodes(item: MarketFlowItem, preferredSymbols?: string[]): string {
    const codes = preferredSymbols?.length
        ? preferredSymbols
        : item.stock_codes?.filter(Boolean) || [];
    if (codes.length === 0) return item.symbol || '';
    if (codes.length <= 2) return codes.join(' ');
    return `${codes.slice(0, 2).join(' ')} +${codes.length - 2} şirket`;
}

export default function MarketFlowPanel({
    onSelectTicker,
}: {
    onSelectTicker?: (ticker: string) => void;
}) {
    const watchlist = useWatchlist();
    const favoriteSymbols = useMemo(
        () => new Set(watchlist.items.map((item) => normalizeWatchlistSymbol(item.symbol))),
        [watchlist.items],
    );
    const [items, setItems] = useState<MarketFlowItem[] | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [filter, setFilter] = useState<FlowFilter>('all');
    const filterRef = useRef<FlowFilter>('all');
    const [loadingMore, setLoadingMore] = useState(false);
    const [hasMore, setHasMore] = useState(true);
    const [visibleLimit, setVisibleLimit] = useState(FLOW_PAGE_SIZE);
    const visibleLimitRef = useRef(FLOW_PAGE_SIZE);
    const requestLimitRef = useRef(FLOW_INITIAL_LOAD_SIZE);
    const loadingMoreRef = useRef(false);
    const prefetchInFlightRef = useRef(false);
    const [warning, setWarning] = useState<string | null>(null);

    const prefetchMore = useCallback((requestedLimit: number, expectedFilter = filterRef.current) => {
        if (
            prefetchInFlightRef.current
            || expectedFilter === 'watchlist'
            || requestedLimit <= requestLimitRef.current
            || requestedLimit > FLOW_MAX_ITEMS
        ) return;

        prefetchInFlightRef.current = true;
        apiClient
            .marketFlow(requestedLimit)
            .then((payload) => {
                if (filterRef.current !== expectedFilter) return;
                const nextItems = payload.items || [];
                requestLimitRef.current = requestedLimit;
                setItems(nextItems);
                setWarning(payload.warning || null);
                setHasMore(nextItems.length >= requestedLimit && requestedLimit < FLOW_MAX_ITEMS);
            })
            .catch(() => {
                // The visible page remains usable when a speculative request fails.
            })
            .finally(() => {
                prefetchInFlightRef.current = false;
            });
    }, []);

    const load = useCallback((refresh = false, requestedLimit?: number) => {
        const requestLimit = requestedLimit
            ?? (filterRef.current === 'watchlist' ? FLOW_FAVORITES_LOAD_SIZE : requestLimitRef.current || FLOW_INITIAL_LOAD_SIZE);
        const expectedFilter = filterRef.current;
        requestLimitRef.current = requestLimit;
        setHasMore(requestLimit < FLOW_MAX_ITEMS);
        setLoading(true);
        setError(null);
        apiClient
            .marketFlow(requestLimit, undefined, { refresh })
            .then((payload) => {
                if (filterRef.current !== expectedFilter) return;
                const nextItems = payload.items || [];
                setItems(nextItems);
                setWarning(payload.warning || null);
                setHasMore(nextItems.length >= requestLimit && requestLimit < FLOW_MAX_ITEMS);
            })
            .catch((requestError: unknown) => {
                setError(requestError instanceof Error ? requestError.message : 'Akış verisi alınamadı.');
            })
            .finally(() => setLoading(false));
    }, []);

    const filteredItems = useMemo(
        () => (items || [])
            .filter((item) => matchesFlowFilter(item, filter, favoriteSymbols)),
        [favoriteSymbols, filter, items],
    );

    const loadMore = useCallback(() => {
        if (
            loadingMoreRef.current
            || !hasMore
            || filterRef.current === 'watchlist'
        ) return;

        const currentVisibleLimit = visibleLimitRef.current;
        if (filteredItems.length > currentVisibleLimit) {
            const nextVisibleLimit = Math.min(currentVisibleLimit + FLOW_PAGE_SIZE, filteredItems.length);
            visibleLimitRef.current = nextVisibleLimit;
            setVisibleLimit(nextVisibleLimit);
            window.setTimeout(() => {
                prefetchMore(requestLimitRef.current + FLOW_PAGE_SIZE);
            }, 0);
            return;
        }

        const currentLimit = requestLimitRef.current;
        const nextLimit = Math.min(currentLimit + FLOW_PAGE_SIZE, FLOW_MAX_ITEMS);
        if (nextLimit <= currentLimit) {
            setHasMore(false);
            return;
        }

        loadingMoreRef.current = true;
        setLoadingMore(true);
        apiClient
            .marketFlow(nextLimit)
            .then((payload) => {
                const nextItems = payload.items || [];
                requestLimitRef.current = nextLimit;
                setItems(nextItems);
                setWarning(payload.warning || null);
                setHasMore(nextItems.length >= nextLimit && nextLimit < FLOW_MAX_ITEMS);
                const nextVisibleLimit = Math.min(currentVisibleLimit + FLOW_PAGE_SIZE, nextItems.length);
                visibleLimitRef.current = nextVisibleLimit;
                setVisibleLimit(nextVisibleLimit);
            })
            .catch((requestError: unknown) => {
                setError(requestError instanceof Error ? requestError.message : 'Daha fazla akış verisi alınamadı.');
            })
            .finally(() => {
                loadingMoreRef.current = false;
                setLoadingMore(false);
            });
    }, [filteredItems.length, hasMore, prefetchMore]);

    useEffect(() => {
        load();
    }, [load]);

    useEffect(() => {
        const timer = window.setInterval(() => {
            if (document.visibilityState === 'visible') load(true);
        }, 30000);
        return () => window.clearInterval(timer);
    }, [load]);

    const handleFilterChange = (nextFilter: FlowFilter) => {
        filterRef.current = nextFilter;
        visibleLimitRef.current = FLOW_PAGE_SIZE;
        setVisibleLimit(FLOW_PAGE_SIZE);
        setFilter(nextFilter);
        if (nextFilter === 'watchlist') {
            load(false, FLOW_FAVORITES_LOAD_SIZE);
        } else {
            load(false, FLOW_INITIAL_LOAD_SIZE);
        }
    };

    const handleFlowScroll = (event: UIEvent<HTMLDivElement>) => {
        const target = event.currentTarget;
        const remaining = target.scrollHeight - target.scrollTop - target.clientHeight;
        if (remaining < 220) loadMore();
    };

    const handleItemClick = (item: MarketFlowItem) => {
        if (item.kap_url) {
            window.open(item.kap_url, '_blank', 'noopener,noreferrer');
            return;
        }
        if (item.symbol) onSelectTicker?.(item.symbol);
    };

    const visibleItems = useMemo(
        () => filteredItems.slice(0, filter === 'watchlist' ? FLOW_PAGE_SIZE : visibleLimit),
        [filter, filteredItems, visibleLimit],
    );

    return (
        <div className="mwr-flow-panel">
            <div className="mwr-flow-toolbar">
                <div className="mwr-flow-heading">
                    <h2 className="mwr-flow-title">Akış</h2>
                    <ChevronRight size={16} aria-hidden="true" />
                </div>
                <label className="mwr-flow-select-wrap">
                    <select
                        className="mwr-flow-select"
                        value={filter}
                        onChange={(event) => handleFilterChange(event.target.value as FlowFilter)}
                        aria-label="Akış filtresi"
                    >
                        {FLOW_FILTERS.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                    </select>
                </label>
            </div>

            {warning && <div className="mwr-flow-warning" role="status">{warning}</div>}
            {loading && !items && <div className="mwr-flow-state">Akış yükleniyor…</div>}
            {error && <div className="mwr-flow-state mwr-flow-error">{error}</div>}
            {!loading && !error && items && filteredItems.length === 0 && (
                <div className="mwr-flow-state">
                    {filter === 'watchlist' && favoriteSymbols.size === 0
                        ? 'Önce yıldız simgesinden favori ekle.'
                        : 'Bu filtreye uygun bildirim yok.'}
                </div>
            )}

            <div className="mwr-flow-list" onScroll={handleFlowScroll}>
                {visibleItems.map((item) => (
                    <button
                        key={item.id}
                        type="button"
                        className="mwr-flow-item"
                        onClick={() => handleItemClick(item)}
                        title={`${item.title} · ${formatFlowDate(item.published_at)}`}
                    >
                        <div className="mwr-flow-item-meta">
                            <span className="mwr-flow-source">KAP</span>
                            {formatFlowCodes(item, filter === 'watchlist'
                                ? getMatchingFavoriteSymbols(item, favoriteSymbols)
                                : undefined) && (
                                <>
                                    <span className="mwr-flow-dot" aria-hidden="true">·</span>
                                    <span className="mwr-flow-codes">
                                        {formatFlowCodes(item, filter === 'watchlist'
                                            ? getMatchingFavoriteSymbols(item, favoriteSymbols)
                                            : undefined)}
                                    </span>
                                </>
                            )}
                            <time dateTime={item.published_at}>{formatFlowTime(item.published_at)}</time>
                        </div>
                        <div className="mwr-flow-item-title">{item.title}</div>
                    </button>
                ))}
                {loadingMore && <div className="mwr-flow-load-state">Daha fazla bildirim yükleniyor…</div>}
                {!loadingMore && !hasMore && filteredItems.length > 0 && (
                    <div className="mwr-flow-load-state">Akışın sonu</div>
                )}
            </div>
        </div>
    );
}
