import { afterEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from './client';
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
describe('request wait budget', () => {
    it('coalesces and briefly caches identical flow requests', async () => {
        const payload = { items: [], as_of: '2026-09-08T00:00:00Z' };
        const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(payload)));
        vi.stubGlobal('fetch', fetch);

        const [first, second] = await Promise.all([
            apiClient.marketFlow(123),
            apiClient.marketFlow(123),
        ]);
        expect(first).toEqual(payload);
        expect(second).toEqual(payload);
        expect(await apiClient.marketFlow(123)).toEqual(payload);
        expect(fetch).toHaveBeenCalledTimes(1);
    });

    it('coalesces and briefly caches identical market stock requests', async () => {
        const payload = {
            index: 'XU100',
            rows: [],
            benchmarks: {},
            source: 'test',
            as_of: '2026-09-08T00:00:00Z',
        };
        const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(payload)));
        vi.stubGlobal('fetch', fetch);

        const [first, second] = await Promise.all([
            apiClient.marketStocks({ index: 'XU100' }),
            apiClient.marketStocks({ index: 'XU100' }),
        ]);
        expect(first).toEqual(payload);
        expect(second).toEqual(payload);
        expect(await apiClient.marketStocks({ index: 'XU100' })).toEqual(payload);
        expect(fetch).toHaveBeenCalledTimes(1);
    });

    it('keeps cold fund responses loading until the background result arrives', async () => {
        vi.useFakeTimers();
        const ready = { status: 'ok', rows: [{ fund_code: 'AAL' }] };
        const fetch = vi.fn()
            .mockResolvedValueOnce(new Response(JSON.stringify({ status: 'pending', rows: [] })))
            .mockResolvedValueOnce(new Response(JSON.stringify(ready)));
        vi.stubGlobal('fetch', fetch);
        const result = apiClient.funds();
        await vi.runAllTimersAsync();
        expect(await result).toEqual(ready);
        expect(fetch).toHaveBeenCalledTimes(2);
    });
    it('returns stale fund data immediately without waiting for its refresh', async () => {
        const stale = { status: 'ok', refresh_pending: true, rows: [{ fund_code: 'AAL' }] };
        const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(stale)));
        vi.stubGlobal('fetch', fetch);
        expect(await apiClient.funds()).toEqual(stale);
        expect(fetch).toHaveBeenCalledTimes(1);
    });
    it('reuses market stock payload on 304 without notifying subscribers again', async () => {
        const payload = { index: 'XU030', rows: [{ company: 'AAA', price: 10 }] };
        const etag = 'W/"market-stock-test"';
        const notModifiedJson = vi.fn();
        const notModifiedResponse = {
            status: 304,
            ok: false,
            headers: new Headers({ ETag: etag }),
            json: notModifiedJson,
        } as unknown as Response;
        const fetch = vi.fn()
            .mockResolvedValueOnce(new Response(JSON.stringify(payload), { headers: { ETag: etag } }))
            .mockResolvedValueOnce(notModifiedResponse);
        vi.stubGlobal('fetch', fetch);
        const listener = vi.fn();
        const unsubscribe = apiClient.subscribeMarketStocks(listener);

        const first = await apiClient.marketStocks({ index: 'XU030', refresh: true });
        const second = await apiClient.marketStocks({ index: 'XU030', refresh: true });

        unsubscribe();
        expect(second).toBe(first);
        expect(listener).toHaveBeenCalledTimes(1);
        expect(notModifiedJson).not.toHaveBeenCalled();
        expect(new Headers(fetch.mock.calls[1]?.[1]?.headers).get('If-None-Match')).toBe(etag);
    });
    it('reuses the fund catalogue on 304 without notifying subscribers again', async () => {
        const payload = { status: 'ok', rows: [{ fund_code: 'ETAG_TEST' }] };
        const etag = 'W/"fund-catalog-test"';
        const notModifiedJson = vi.fn();
        const notModifiedResponse = {
            status: 304,
            ok: false,
            headers: new Headers({ ETag: etag }),
            json: notModifiedJson,
        } as unknown as Response;
        const fetch = vi.fn()
            .mockResolvedValueOnce(new Response(JSON.stringify(payload), { headers: { ETag: etag } }))
            .mockResolvedValueOnce(notModifiedResponse);
        vi.stubGlobal('fetch', fetch);
        const listener = vi.fn();
        const unsubscribe = apiClient.subscribeFundsCatalog(listener);

        const first = await apiClient.funds();
        const second = await apiClient.funds();

        unsubscribe();
        expect(second).toBe(first);
        expect(listener).toHaveBeenCalledTimes(1);
        expect(notModifiedJson).not.toHaveBeenCalled();
        expect(new Headers(fetch.mock.calls[1]?.[1]?.headers).get('If-None-Match')).toBe(etag);
    });
    it('cancels cold fund polling when its consumer leaves', async () => {
        vi.useFakeTimers();
        const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: 'pending' })));
        vi.stubGlobal('fetch', fetch);
        const controller = new AbortController();
        const result = expect(apiClient.fundYieldSummary('AAL', { signal: controller.signal }))
            .rejects.toMatchObject({ name: 'AbortError' });
        await vi.advanceTimersByTimeAsync(1);
        controller.abort();
        await result;
        await vi.runAllTimersAsync();
        expect(fetch).toHaveBeenCalledTimes(1);
    });
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
