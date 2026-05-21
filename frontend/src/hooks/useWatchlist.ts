import { useCallback, useEffect, useMemo, useState } from 'react';

export type WatchlistItemKind = 'stock' | 'fund';

export interface WatchlistItem {
    kind: WatchlistItemKind;
    symbol: string;
    label?: string;
}

export const WATCHLIST_STORAGE_KEY = 'mwr_watchlist';
const WATCHLIST_EVENT = 'ragfin:watchlist-changed';

type WatchlistEventDetail = {
    items: WatchlistItem[];
};

function canUseStorage(): boolean {
    return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
}

export function normalizeWatchlistSymbol(symbol: string): string {
    return symbol.trim().toUpperCase();
}

export function watchlistItemKey(item: Pick<WatchlistItem, 'kind' | 'symbol'>): string {
    return `${item.kind}:${normalizeWatchlistSymbol(item.symbol)}`;
}

function normalizeItem(value: unknown): WatchlistItem | null {
    if (typeof value === 'string') {
        const symbol = normalizeWatchlistSymbol(value);
        return symbol ? { kind: 'stock', symbol } : null;
    }

    if (!value || typeof value !== 'object') return null;
    const candidate = value as Partial<WatchlistItem>;
    if (typeof candidate.symbol !== 'string') return null;
    const symbol = normalizeWatchlistSymbol(candidate.symbol);
    if (!symbol) return null;
    const kind: WatchlistItemKind = candidate.kind === 'fund' ? 'fund' : 'stock';
    const label = typeof candidate.label === 'string' && candidate.label.trim()
        ? candidate.label.trim()
        : undefined;
    return { kind, symbol, label };
}

export function normalizeWatchlistItems(value: unknown): WatchlistItem[] {
    if (!Array.isArray(value)) return [];
    const seen = new Set<string>();
    const items: WatchlistItem[] = [];

    for (const rawItem of value) {
        const item = normalizeItem(rawItem);
        if (!item) continue;
        const key = watchlistItemKey(item);
        if (seen.has(key)) continue;
        seen.add(key);
        items.push(item);
    }

    return items;
}

export function readWatchlistItems(): WatchlistItem[] {
    if (!canUseStorage()) return [];
    try {
        const raw = window.localStorage.getItem(WATCHLIST_STORAGE_KEY);
        if (!raw) return [];
        return normalizeWatchlistItems(JSON.parse(raw));
    } catch {
        return [];
    }
}

function writeWatchlistItems(items: WatchlistItem[]): void {
    if (!canUseStorage()) return;
    window.localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(normalizeWatchlistItems(items)));
}

function dispatchWatchlistEvent(items: WatchlistItem[]): void {
    if (typeof window === 'undefined') return;
    window.dispatchEvent(new CustomEvent<WatchlistEventDetail>(WATCHLIST_EVENT, {
        detail: { items: normalizeWatchlistItems(items) },
    }));
}

export function useWatchlist() {
    const [items, setItemsState] = useState<WatchlistItem[]>(readWatchlistItems);

    useEffect(() => {
        const normalized = readWatchlistItems();
        writeWatchlistItems(normalized);
        setItemsState(normalized);

        const handleStorage = (event: StorageEvent) => {
            if (event.key === WATCHLIST_STORAGE_KEY) {
                setItemsState(readWatchlistItems());
            }
        };

        const handleLocalChange = (event: Event) => {
            const detail = (event as CustomEvent<WatchlistEventDetail>).detail;
            setItemsState(normalizeWatchlistItems(detail?.items || readWatchlistItems()));
        };

        window.addEventListener('storage', handleStorage);
        window.addEventListener(WATCHLIST_EVENT, handleLocalChange);
        return () => {
            window.removeEventListener('storage', handleStorage);
            window.removeEventListener(WATCHLIST_EVENT, handleLocalChange);
        };
    }, []);

    const replaceItems = useCallback((nextItems: WatchlistItem[]) => {
        const normalized = normalizeWatchlistItems(nextItems);
        writeWatchlistItems(normalized);
        setItemsState(normalized);
        dispatchWatchlistEvent(normalized);
    }, []);

    const addItem = useCallback((item: WatchlistItem) => {
        setItemsState((current) => {
            const normalizedItem = normalizeItem(item);
            if (!normalizedItem) return current;
            if (current.some((existing) => watchlistItemKey(existing) === watchlistItemKey(normalizedItem))) {
                return current;
            }
            const next = normalizeWatchlistItems([...current, normalizedItem]);
            writeWatchlistItems(next);
            dispatchWatchlistEvent(next);
            return next;
        });
    }, []);

    const removeItem = useCallback((kind: WatchlistItemKind, symbol: string) => {
        const key = `${kind}:${normalizeWatchlistSymbol(symbol)}`;
        setItemsState((current) => {
            const next = current.filter((item) => watchlistItemKey(item) !== key);
            writeWatchlistItems(next);
            dispatchWatchlistEvent(next);
            return next;
        });
    }, []);

    const toggleItem = useCallback((item: WatchlistItem) => {
        const normalizedItem = normalizeItem(item);
        if (!normalizedItem) return;
        const key = watchlistItemKey(normalizedItem);
        setItemsState((current) => {
            const exists = current.some((existing) => watchlistItemKey(existing) === key);
            const next = exists
                ? current.filter((existing) => watchlistItemKey(existing) !== key)
                : normalizeWatchlistItems([...current, normalizedItem]);
            writeWatchlistItems(next);
            dispatchWatchlistEvent(next);
            return next;
        });
    }, []);

    const hasItem = useCallback((kind: WatchlistItemKind, symbol: string) => {
        const key = `${kind}:${normalizeWatchlistSymbol(symbol)}`;
        return items.some((item) => watchlistItemKey(item) === key);
    }, [items]);

    return useMemo(() => ({
        items,
        addItem,
        removeItem,
        replaceItems,
        toggleItem,
        hasItem,
    }), [addItem, hasItem, items, removeItem, replaceItems, toggleItem]);
}
