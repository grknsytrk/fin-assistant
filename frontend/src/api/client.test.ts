import { afterEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from './client';
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
describe('request wait budget', () => {
    it('stops retrying HTTP failures after two attempts', async () => {
        vi.useFakeTimers();
        const fetch = vi.fn().mockResolvedValue(new Response('{}', { status: 503 }));
        vi.stubGlobal('fetch', fetch);
        const result = expect(apiClient.health()).rejects.toThrow();
        await vi.runAllTimersAsync(); await result;
        expect(fetch).toHaveBeenCalledTimes(2);
    });
    it('does not wait out a long Retry-After', async () => {
        const fetch = vi.fn().mockResolvedValue(new Response('{}', { status: 429, headers: { 'Retry-After': '120' } }));
        vi.stubGlobal('fetch', fetch);
        await expect(apiClient.health()).rejects.toThrow('suresi');
        expect(fetch).toHaveBeenCalledTimes(1);
    });
    it('bounds a hung request by the total deadline', async () => {
        vi.useFakeTimers();
        const fetch = vi.fn((_url, options) => new Promise((_resolve, reject) => {
            options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
        }));
        vi.stubGlobal('fetch', fetch);
        const result = expect(apiClient.health()).rejects.toThrow('suresi');
        await vi.advanceTimersByTimeAsync(15001); await result;
        expect(fetch).toHaveBeenCalledTimes(1);
    });
    it('cancels snapshot retries when leaving the page', async () => {
        vi.useFakeTimers();
        const fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
        vi.stubGlobal('fetch', fetch);
        const controller = new AbortController();
        const result = expect(apiClient.kapSnapshot('TEST', false, 5, controller.signal)).rejects.toMatchObject({ name: 'AbortError' });
        await vi.advanceTimersByTimeAsync(1);
        controller.abort(); await result;
        await vi.runAllTimersAsync();
        expect(fetch).toHaveBeenCalledTimes(1);
    });
});
