/**
 * Lightweight, zero-dependency generic bounded in-memory cache
 * using native JavaScript Map insertion-order semantics for O(1) LRU eviction.
 */

export interface BoundedMemoryCacheOptions {
    maxEntries?: number;
    ttlMs?: number;
}

interface CacheEntry<V> {
    value: V;
    expiresAt?: number;
}

export class BoundedMemoryCache<K, V> {
    private readonly maxEntries: number;
    private readonly ttlMs?: number;
    private readonly map = new Map<K, CacheEntry<V>>();

    constructor(options: BoundedMemoryCacheOptions | number = 50) {
        if (typeof options === 'number') {
            this.maxEntries = Math.max(1, options);
            this.ttlMs = undefined;
        } else {
            this.maxEntries = Math.max(1, options?.maxEntries ?? 50);
            this.ttlMs = options?.ttlMs;
        }
    }

    /**
     * Retrieves the item associated with the key.
     * Touches entry to refresh LRU recency (moves it to most-recently-used position).
     * If entry has expired due to TTL, removes it and returns undefined.
     */
    get(key: K): V | undefined {
        const entry = this.map.get(key);
        if (!entry) {
            return undefined;
        }

        if (entry.expiresAt !== undefined && Date.now() >= entry.expiresAt) {
            this.map.delete(key);
            return undefined;
        }

        // Refresh LRU recency: delete and re-set moves key to the end of Map iteration order
        this.map.delete(key);
        this.map.set(key, entry);
        return entry.value;
    }

    /**
     * Stores or updates a key-value pair, with optional custom TTL in milliseconds.
     * When capacity exceeds maxEntries, evicts the least recently used entry (first in Map order).
     */
    set(key: K, value: V, customTtlMs?: number): void {
        const ttl = customTtlMs !== undefined ? customTtlMs : this.ttlMs;
        const expiresAt = ttl !== undefined ? Date.now() + ttl : undefined;

        this.map.delete(key);
        this.map.set(key, { value, expiresAt });

        while (this.map.size > this.maxEntries) {
            const oldestKey = this.map.keys().next().value;
            if (oldestKey !== undefined) {
                this.map.delete(oldestKey);
            } else {
                break;
            }
        }
    }

    /**
     * Checks if a non-expired entry exists for the given key.
     * Cleans up the entry if expired. Does not alter LRU recency.
     */
    has(key: K): boolean {
        const entry = this.map.get(key);
        if (!entry) {
            return false;
        }
        if (entry.expiresAt !== undefined && Date.now() >= entry.expiresAt) {
            this.map.delete(key);
            return false;
        }
        return true;
    }

    /**
     * Removes an entry by key. Returns true if key was present, false otherwise.
     */
    delete(key: K): boolean {
        return this.map.delete(key);
    }

    /**
     * Clears all entries from the cache.
     */
    clear(): void {
        this.map.clear();
    }

    /**
     * Current number of entries stored in the cache.
     */
    get size(): number {
        return this.map.size;
    }
}
