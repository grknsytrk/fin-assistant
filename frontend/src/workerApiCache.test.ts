import { describe, expect, it, vi } from 'vitest';
import {
    applyListCachePolicy,
    cacheControlForListRequest,
    cacheTagsForList,
    canonicalizeCacheRequest,
    createUpstreamRequest,
    getCacheableListRoute,
    marketStocksCacheTag,
    purgeMarketCacheForRefresh,
    shouldPurgeMarketCache,
    shouldUseCachedList,
} from '../../worker/api-cache';

function makeRequest(path: string, init?: RequestInit): Request {
    return new Request(`https://fin-rag.example${path}`, init);
}

describe('Worker edge cache routing', () => {
    it('recognizes only the exact public list paths', () => {
        expect(getCacheableListRoute(new URL('https://fin-rag.example/api/market/stocks?index=XU100')))
            .toBe('market-stocks');
        expect(getCacheableListRoute(new URL('https://fin-rag.example/api/funds?sort=name')))
            .toBe('funds');
        expect(getCacheableListRoute(new URL('https://fin-rag.example/api/funds/search?q=bank')))
            .toBeNull();
        expect(getCacheableListRoute(new URL('https://fin-rag.example/funds')))
            .toBeNull();
    });

    it('bypasses cache for credentials, cross-origin requests, non-GETs, and market refreshes', () => {
        expect(shouldUseCachedList(makeRequest('/api/funds'))).toBe(true);
        expect(shouldUseCachedList(makeRequest('/api/funds', { headers: { Authorization: 'Bearer token' } })))
            .toBe(false);
        expect(shouldUseCachedList(makeRequest('/api/funds', { headers: { Cookie: 'session=1' } })))
            .toBe(false);
        expect(shouldUseCachedList(makeRequest('/api/funds', { headers: { Origin: 'https://other.example' } })))
            .toBe(false);
        expect(shouldUseCachedList(makeRequest('/api/funds', { method: 'POST', body: '{}' }))).toBe(false);
        expect(shouldUseCachedList(makeRequest('/api/market/stocks?index=XU100&refresh=true'))).toBe(false);
        expect(shouldPurgeMarketCache(makeRequest('/api/market/stocks?index=XU100&refresh=true'))).toBe(true);
    });

    it('canonicalizes query order and partitions entries by public host', () => {
        const first = canonicalizeCacheRequest(makeRequest('/api/funds?sort=name&order=desc'));
        const second = canonicalizeCacheRequest(makeRequest('/api/funds?order=desc&sort=name'));
        expect(new URL(first.url).search).toBe(new URL(second.url).search);
        expect(new URL(first.url).searchParams.get('__finrag_edge_host')).toBe('fin-rag.example');

        const otherHost = canonicalizeCacheRequest(new Request('https://preview.example/api/funds?sort=name&order=desc'));
        expect(new URL(otherHost.url).searchParams.get('__finrag_edge_host')).toBe('preview.example');
        expect(first.headers.get('Accept')).toBe(second.headers.get('Accept'));
    });

    it('rewrites API paths while preserving query, validators, method, and body', async () => {
        const getRequest = canonicalizeCacheRequest(new Request(
            'https://fin-rag.example/api/funds?sort=name&__finrag_edge_host=spoofed',
            { headers: { 'If-None-Match': 'W/"catalog"' } },
        ));
        const upstreamGet = createUpstreamRequest(getRequest, 'https://api.example');
        expect(upstreamGet.url).toBe('https://api.example/funds?sort=name');
        expect(upstreamGet.headers.get('If-None-Match')).toBe('W/"catalog"');

        const postRequest = makeRequest('/api/admin/example?force=true', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ value: 7 }),
        });
        const upstreamPost = createUpstreamRequest(postRequest, 'https://api.example');
        expect(upstreamPost.method).toBe('POST');
        expect(upstreamPost.url).toBe('https://api.example/admin/example?force=true');
        expect(await upstreamPost.text()).toBe('{"value":7}');
    });
});

describe('Worker edge cache policies', () => {
    it('uses open-session, pre-open, and fund catalogue TTLs', () => {
        const marketRequest = makeRequest('/api/market/stocks?index=XU100');
        expect(cacheControlForListRequest(marketRequest, new Date('2026-10-19T07:00:00.000Z')))
            .toBe('public, max-age=3, stale-while-revalidate=30, stale-if-error=0');
        expect(cacheControlForListRequest(marketRequest, new Date('2026-10-19T06:48:00.000Z')))
            .toBe('public, max-age=120, stale-if-error=0');

        expect(cacheControlForListRequest(makeRequest('/api/funds?sort=name')))
            .toBe('public, max-age=300, stale-while-revalidate=3600, stale-if-error=0');
    });

    it('tags the fund list and each market index variant', () => {
        const url = new URL('https://fin-rag.example/api/market/stocks?index=xu100');
        expect(marketStocksCacheTag(url)).toBe('market-stocks-XU100');
        expect(cacheTagsForList('market-stocks', url)).toEqual(['market-stocks', 'market-stocks-XU100']);
        expect(cacheTagsForList('funds', new URL('https://fin-rag.example/api/funds?sort=name')))
            .toEqual(['funds-list']);
    });

    it('purges the refreshed market index tag and leaves other requests alone', async () => {
        const purge = vi.fn(async () => undefined);
        const refreshed = await purgeMarketCacheForRefresh(
            makeRequest('/api/market/stocks?index=XU100&refresh=true'),
            purge,
        );
        expect(refreshed).toBe(true);
        expect(purge).toHaveBeenCalledWith('market-stocks-XU100');

        const untouched = await purgeMarketCacheForRefresh(makeRequest('/api/funds'), purge);
        expect(untouched).toBe(false);
        expect(purge).toHaveBeenCalledTimes(1);
    });

    it('keeps origin ETags and browser cache headers while adding edge policy', async () => {
        const request = makeRequest('/api/funds');
        const originResponse = new Response(JSON.stringify({
            status: 'ok',
            refresh_pending: true,
            rows: [{ fund_code: 'AAL' }],
        }), {
            headers: {
                'Content-Type': 'application/json',
                'Cache-Control': 'private, no-cache',
                ETag: 'W/"funds-v1"',
            },
        });

        const response = await applyListCachePolicy(originResponse, request, 'funds');
        expect(response.headers.get('Cache-Control')).toBe('private, no-cache');
        expect(response.headers.get('Cloudflare-CDN-Cache-Control'))
            .toBe('public, max-age=300, stale-while-revalidate=3600, stale-if-error=0');
        expect(response.headers.get('ETag')).toBe('W/"funds-v1"');
        expect(response.headers.get('Cache-Tag')).toBe('funds-list');
    });

    it('does not cache unavailable, degraded, empty market, or error responses', async () => {
        const request = makeRequest('/api/funds');
        const pending = await applyListCachePolicy(
            new Response(JSON.stringify({ status: 'pending', rows: [] }), {
                headers: { 'Content-Type': 'application/json', 'Cache-Control': 'private, no-cache' },
            }),
            request,
            'funds',
        );
        expect(pending.headers.get('Cache-Control')).toBe('private, no-store');
        expect(pending.headers.has('Cloudflare-CDN-Cache-Control')).toBe(false);
        expect(pending.headers.has('Cache-Tag')).toBe(false);

        const degraded = await applyListCachePolicy(
            new Response(JSON.stringify({ status: 'ok', stale: true, rows: [{ fund_code: 'AAL' }] }), {
                headers: { 'Content-Type': 'application/json' },
            }),
            request,
            'funds',
        );
        expect(degraded.headers.get('Cache-Control')).toBe('private, no-store');

        const emptyMarket = await applyListCachePolicy(
            new Response(JSON.stringify({ rows: [] }), { headers: { 'Content-Type': 'application/json' } }),
            makeRequest('/api/market/stocks'),
            'market-stocks',
        );
        expect(emptyMarket.headers.has('Cloudflare-CDN-Cache-Control')).toBe(false);

        const failed = await applyListCachePolicy(new Response('{}', { status: 503 }), request, 'funds');
        expect(failed.headers.get('Cache-Control')).toBe('private, no-store');
    });

    it('keeps 304 validators usable during cache revalidation', async () => {
        const response = await applyListCachePolicy(
            new Response(null, { status: 304, headers: { ETag: 'W/"funds-v1"', 'Cache-Control': 'private, no-cache' } }),
            makeRequest('/api/funds'),
            'funds',
        );
        expect(response.status).toBe(304);
        expect(response.headers.get('ETag')).toBe('W/"funds-v1"');
        expect(response.headers.get('Cloudflare-CDN-Cache-Control'))
            .toBe('public, max-age=300, stale-while-revalidate=3600, stale-if-error=0');
    });
});
