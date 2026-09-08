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
const FLOW_INITIAL_LOAD_SIZE = FLOW_PAGE_SIZE;
const FLOW_CATEGORY_LOAD_SIZE = 100;
const FLOW_BACKFILL_PAGE_SIZE = 500;
const FLOW_MAX_ITEMS = 2_500;
const FLOW_FAVORITES_LOAD_SIZE = 500;
const FLOW_NEW_ITEM_HIGHLIGHT_MS = 5_000;
const FLOW_HEAD_POLL_MS = 15_000;

function getServerFlowCategory(filter: FlowFilter): string | undefined {
    if (filter === 'ozel_durum' || filter === 'finansal_rapor' || filter === 'kar_payi' || filter === 'genel_kurul') {
        return filter;
    }
    return undefined;
}

function getFlowSymbols(item: MarketFlowItem): string[] {
    return [item.symbol, ...(item.stock_codes || []), ...(item.related_symbols || [])]
        .filter(Boolean)
        .map(normalizeWatchlistSymbol);
}

export function getMatchingFavoriteSymbols(item: MarketFlowItem, favoriteSymbols: Set<string>): string[] {
    const itemSymbols = new Set(getFlowSymbols(item));
    return Array.from(favoriteSymbols).filter((symbol) => itemSymbols.has(symbol));
}

export function getNewFlowItemIds(previousIds: Set<string> | null, items: MarketFlowItem[]): string[] {
    if (!previousIds) return [];
    return items.filter((item) => !previousIds.has(item.id)).map((item) => item.id);
}

function matchesFlowFilter(item: MarketFlowItem, filter: FlowFilter, favoriteSymbols: Set<string>): boolean {
    if (filter === 'all') return true;
    if (filter === 'watchlist') {
        if (favoriteSymbols.size === 0) return false;
        return getMatchingFavoriteSymbols(item, favoriteSymbols).length > 0;
    }
    if (filter === 'ozel_durum') return item.category === 'ozel_durum' || (!item.category && item.source === 'Özel Durum');
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
    const requestCategoryRef = useRef<string | undefined>(undefined);
    const nextCursorRef = useRef<string | null>(null);
    const latestCursorRef = useRef<string | null>(null);
    const loadingMoreRef = useRef(false);
    const observedFlowIdsRef = useRef<Set<string> | null>(null);
    const highlightTimersRef = useRef(new Map<string, number>());
    const [highlightedItemIds, setHighlightedItemIds] = useState<Set<string>>(new Set());
    const [warning, setWarning] = useState<string | null>(null);

    const markNewItems = useCallback((itemIds: string[]) => {
        if (itemIds.length === 0) return;

        setHighlightedItemIds((current) => {
            const next = new Set(current);
            itemIds.forEach((id) => next.add(id));
            return next;
        });

        itemIds.forEach((id) => {
            const previousTimer = highlightTimersRef.current.get(id);
            if (previousTimer !== undefined) window.clearTimeout(previousTimer);
            const timer = window.setTimeout(() => {
                setHighlightedItemIds((current) => {
                    if (!current.has(id)) return current;
                    const next = new Set(current);
                    next.delete(id);
                    return next;
                });
                highlightTimersRef.current.delete(id);
            }, FLOW_NEW_ITEM_HIGHLIGHT_MS);
            highlightTimersRef.current.set(id, timer);
        });
    }, []);

    useEffect(() => () => {
        highlightTimersRef.current.forEach((timer) => window.clearTimeout(timer));
        highlightTimersRef.current.clear();
    }, []);

    const load = useCallback((requestedLimit?: number, requestedCategory?: string) => {
        const expectedFilter = filterRef.current;
        const serverCategory = requestedCategory ?? getServerFlowCategory(expectedFilter);
        const requestLimit = requestedLimit
            ?? (expectedFilter === 'watchlist'
                ? FLOW_FAVORITES_LOAD_SIZE
                : serverCategory ? FLOW_CATEGORY_LOAD_SIZE : requestLimitRef.current || FLOW_INITIAL_LOAD_SIZE);
        requestLimitRef.current = requestLimit;
        requestCategoryRef.current = serverCategory;
        nextCursorRef.current = null;
        latestCursorRef.current = null;
        setHasMore(true);
        setLoading(true);
        setError(null);
        apiClient
            .marketFlow(requestLimit, serverCategory)
            .then((payload) => {
                if (filterRef.current !== expectedFilter) return;
                const nextItems = payload.items || [];
                observedFlowIdsRef.current = new Set(nextItems.map((item) => item.id));
                nextCursorRef.current = payload.next_cursor || null;
                latestCursorRef.current = payload.latest_cursor || null;
                setItems(nextItems);
                setWarning(payload.warning || null);
                setHasMore(typeof payload.has_more === 'boolean'
                    ? payload.has_more
                    : nextItems.length >= requestLimit && requestLimit < FLOW_MAX_ITEMS);
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

    const loadMore = useCallback((requestedPageSize = FLOW_PAGE_SIZE, revealMore = true) => {
        if (loadingMoreRef.current) return;

        const currentVisibleLimit = visibleLimitRef.current;
        if (revealMore && filteredItems.length > currentVisibleLimit) {
            const nextVisibleLimit = Math.min(currentVisibleLimit + FLOW_PAGE_SIZE, filteredItems.length);
            visibleLimitRef.current = nextVisibleLimit;
            setVisibleLimit(nextVisibleLimit);
            return;
        }
        if (!hasMore) return;

        const nextCursor = nextCursorRef.current;
        const currentLimit = requestLimitRef.current;
        const pageSize = Math.min(Math.max(requestedPageSize, FLOW_PAGE_SIZE), FLOW_BACKFILL_PAGE_SIZE);
        const nextLimit = Math.min(currentLimit + pageSize, FLOW_MAX_ITEMS);
        if (!nextCursor && nextLimit <= currentLimit) {
            setHasMore(false);
            return;
        }

        loadingMoreRef.current = true;
        setLoadingMore(true);
        apiClient
            .marketFlow(
                nextCursor ? pageSize : nextLimit,
                requestCategoryRef.current,
                nextCursor ? { before: nextCursor } : undefined,
            )
            .then((payload) => {
                const olderItems = payload.items || [];
                requestLimitRef.current = nextCursor ? currentLimit + olderItems.length : nextLimit;
                nextCursorRef.current = payload.next_cursor || null;
                latestCursorRef.current = latestCursorRef.current || payload.latest_cursor || null;
                setItems((currentItems) => {
                    const current = currentItems || [];
                    const seen = new Set(current.map((item) => item.id));
                    return [...current, ...olderItems.filter((item) => !seen.has(item.id))];
                });
                setWarning(payload.warning || null);
                setHasMore(typeof payload.has_more === 'boolean'
                    ? payload.has_more
                    : olderItems.length >= (nextCursor ? pageSize : nextLimit)
                        && requestLimitRef.current < FLOW_MAX_ITEMS);
                const nextVisibleLimit = revealMore
                    ? currentVisibleLimit + FLOW_PAGE_SIZE
                    : currentVisibleLimit;
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
    }, [filteredItems.length, hasMore]);

    useEffect(() => {
        const needsBackfill = filter === 'watchlist' || filter === 'diger';
        if (!needsBackfill || loading || loadingMore || !items || !hasMore) return;
        if (filteredItems.length >= FLOW_PAGE_SIZE) return;

        // A filtered view should not depend on the user reaching an empty
        // scrollbar to discover older matches. Fetch one bounded 500-row page
        // at a time until we have 50 matches or the durable feed ends.
        loadMore(FLOW_BACKFILL_PAGE_SIZE, false);
    }, [filter, filteredItems.length, hasMore, items, loading, loadingMore, loadMore]);

    useEffect(() => {
        load();
    }, [load]);

    useEffect(() => {
        const timer = window.setInterval(() => {
            if (document.visibilityState !== 'visible') return;
            const serverCategory = requestCategoryRef.current;
            apiClient
                .marketFlowHead(serverCategory)
                .then((head) => {
                    const previousCursor = latestCursorRef.current;
                    if (!head.latest_cursor) return;
                    if (!previousCursor) {
                        latestCursorRef.current = head.latest_cursor;
                        return;
                    }
                    if (head.latest_cursor === previousCursor) return;

                    return apiClient
                        .marketFlow(FLOW_PAGE_SIZE, serverCategory, { after: previousCursor })
                        .then((payload) => {
                            const incoming = payload.items || [];
                            const newIds = getNewFlowItemIds(observedFlowIdsRef.current, incoming);
                            if (incoming.length > 0) {
                                setItems((currentItems) => {
                                    const current = currentItems || [];
                                    const incomingIds = new Set(incoming.map((item) => item.id));
                                    return [
                                        ...incoming,
                                        ...current.filter((item) => !incomingIds.has(item.id)),
                                    ];
                                });
                                const observed = observedFlowIdsRef.current || new Set<string>();
                                incoming.forEach((item) => observed.add(item.id));
                                observedFlowIdsRef.current = observed;
                                markNewItems(newIds);
                                setWarning(payload.warning || null);
                            }
                            latestCursorRef.current = head.latest_cursor;
                        });
                })
                .catch(() => {
                    // The last successful flow remains visible while the head check retries later.
                });
        }, FLOW_HEAD_POLL_MS);
        return () => window.clearInterval(timer);
    }, [markNewItems]);

    const handleFilterChange = (nextFilter: FlowFilter) => {
        filterRef.current = nextFilter;
        visibleLimitRef.current = FLOW_PAGE_SIZE;
        setVisibleLimit(FLOW_PAGE_SIZE);
        setFilter(nextFilter);
        if (nextFilter === 'watchlist') {
            load(FLOW_FAVORITES_LOAD_SIZE);
        } else if (getServerFlowCategory(nextFilter)) {
            load(FLOW_CATEGORY_LOAD_SIZE, getServerFlowCategory(nextFilter));
        } else {
            load(FLOW_INITIAL_LOAD_SIZE);
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
        () => filteredItems.slice(0, visibleLimit),
        [filteredItems, visibleLimit],
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
                        className={`mwr-flow-item${highlightedItemIds.has(item.id) ? ' is-new' : ''}`}
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
                            <time dateTime={item.published_at}>
                                {formatFlowDate(item.published_at)} · {formatFlowTime(item.published_at)}
                            </time>
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
