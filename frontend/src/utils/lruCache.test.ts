import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BoundedMemoryCache } from './lruCache';

describe('BoundedMemoryCache', () => {
    describe('Upper bound enforcement (LRU eviction)', () => {
        it('respects maxEntries passed as a number and evicts oldest items', () => {
            const cache = new BoundedMemoryCache<string, number>(3);

            cache.set('a', 1);
            cache.set('b', 2);
            cache.set('c', 3);

            expect(cache.size).toBe(3);
            expect(cache.get('a')).toBe(1);
            expect(cache.get('b')).toBe(2);
            expect(cache.get('c')).toBe(3);

            // Adding a 4th item must evict the least recently used ('a' was touched first among the three, but let's check order:
            // In the lines above, we called get('a'), get('b'), get('c').
            // Order is now: a (least recent), b, c (most recent).
            cache.set('d', 4);

            expect(cache.size).toBe(3);
            expect(cache.get('a')).toBeUndefined();
            expect(cache.get('b')).toBe(2);
            expect(cache.get('c')).toBe(3);
            expect(cache.get('d')).toBe(4);
        });

        it('respects maxEntries passed via options object', () => {
            const cache = new BoundedMemoryCache<string, string>({ maxEntries: 2 });

            cache.set('k1', 'v1');
            cache.set('k2', 'v2');
            cache.set('k3', 'v3');

            expect(cache.size).toBe(2);
            expect(cache.get('k1')).toBeUndefined();
            expect(cache.get('k2')).toBe('v2');
            expect(cache.get('k3')).toBe('v3');
        });

        it('defaults to 50 maxEntries when unspecified', () => {
            const cache = new BoundedMemoryCache<number, number>();

            for (let i = 0; i < 60; i++) {
                cache.set(i, i * 10);
            }

            expect(cache.size).toBe(50);
            // First 10 items (0..9) must be evicted
            for (let i = 0; i < 10; i++) {
                expect(cache.get(i)).toBeUndefined();
            }
            // Last 50 items (10..59) must be present
            for (let i = 10; i < 60; i++) {
                expect(cache.get(i)).toBe(i * 10);
            }
        });
    });

    describe('LRU touch refresh', () => {
        it('refreshes recency on get(), preventing premature eviction', () => {
            const cache = new BoundedMemoryCache<string, string>(3);

            cache.set('first', '1');
            cache.set('second', '2');
            cache.set('third', '3');

            // 'first' is currently the oldest. Accessing it touches it and moves it to most recent.
            expect(cache.get('first')).toBe('1');

            // Order is now: 'second' (oldest), 'third', 'first' (newest).
            // Adding 'fourth' should evict 'second', NOT 'first'.
            cache.set('fourth', '4');

            expect(cache.size).toBe(3);
            expect(cache.get('second')).toBeUndefined();
            expect(cache.get('first')).toBe('1');
            expect(cache.get('third')).toBe('3');
            expect(cache.get('fourth')).toBe('4');
        });

        it('refreshes recency when re-setting an existing key', () => {
            const cache = new BoundedMemoryCache<string, number>(2);

            cache.set('x', 10);
            cache.set('y', 20);

            // Re-setting 'x' updates value and moves it to most recent
            cache.set('x', 100);

            // Now 'y' is the oldest. Adding 'z' must evict 'y'.
            cache.set('z', 30);

            expect(cache.size).toBe(2);
            expect(cache.get('y')).toBeUndefined();
            expect(cache.get('x')).toBe(100);
            expect(cache.get('z')).toBe(30);
        });
    });

    describe('TTL expiration', () => {
        beforeEach(() => {
            vi.useFakeTimers();
        });

        afterEach(() => {
            vi.useRealTimers();
        });

        it('expires entries after default ttlMs', () => {
            const cache = new BoundedMemoryCache<string, string>({ maxEntries: 10, ttlMs: 1000 });

            cache.set('msg', 'hello');

            // Advance time before expiry
            vi.advanceTimersByTime(500);
            expect(cache.get('msg')).toBe('hello');
            expect(cache.has('msg')).toBe(true);

            // Advance time past expiry
            vi.advanceTimersByTime(501);
            expect(cache.get('msg')).toBeUndefined();
            expect(cache.has('msg')).toBe(false);
        });

        it('supports customTtlMs on individual set() calls', () => {
            const cache = new BoundedMemoryCache<string, string>({ maxEntries: 10, ttlMs: 5000 });

            // Override with shorter TTL of 200ms
            cache.set('short', 'quick', 200);
            // Default TTL 5000ms
            cache.set('long', 'slow');

            vi.advanceTimersByTime(250);

            expect(cache.get('short')).toBeUndefined();
            expect(cache.has('short')).toBe(false);
            expect(cache.get('long')).toBe('slow');
            expect(cache.has('long')).toBe(true);

            vi.advanceTimersByTime(4800);

            expect(cache.get('long')).toBeUndefined();
            expect(cache.has('long')).toBe(false);
        });

        it('purges expired entry from cache upon get() or has()', () => {
            const cache = new BoundedMemoryCache<string, number>({ maxEntries: 5, ttlMs: 100 });

            cache.set('item', 42);
            expect(cache.size).toBe(1);

            vi.advanceTimersByTime(101);

            // has() detects expiry and removes entry
            expect(cache.has('item')).toBe(false);
            expect(cache.size).toBe(0);
        });
    });

    describe('Standard Map-like operations and edge cases', () => {
        it('supports has(), delete(), and clear()', () => {
            const cache = new BoundedMemoryCache<string, number>(5);

            cache.set('a', 10);
            cache.set('b', 20);

            expect(cache.has('a')).toBe(true);
            expect(cache.has('c')).toBe(false);

            expect(cache.delete('a')).toBe(true);
            expect(cache.delete('a')).toBe(false);
            expect(cache.has('a')).toBe(false);
            expect(cache.size).toBe(1);

            cache.clear();
            expect(cache.size).toBe(0);
            expect(cache.get('b')).toBeUndefined();
        });

        it('safely handles falsy values (0, empty string, false, null)', () => {
            const cache = new BoundedMemoryCache<string, any>(10);

            cache.set('zero', 0);
            cache.set('empty', '');
            cache.set('bool', false);
            cache.set('nul', null);

            expect(cache.get('zero')).toBe(0);
            expect(cache.get('empty')).toBe('');
            expect(cache.get('bool')).toBe(false);
            expect(cache.get('nul')).toBe(null);

            expect(cache.has('zero')).toBe(true);
            expect(cache.has('empty')).toBe(true);
            expect(cache.has('bool')).toBe(true);
            expect(cache.has('nul')).toBe(true);
        });

        it('gracefully handles maxEntries <= 0 by clamping to 1', () => {
            const cache = new BoundedMemoryCache<string, number>(0);

            cache.set('a', 1);
            cache.set('b', 2);

            expect(cache.size).toBe(1);
            expect(cache.get('a')).toBeUndefined();
            expect(cache.get('b')).toBe(2);
        });
    });
});
