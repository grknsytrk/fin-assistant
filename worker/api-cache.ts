import { isBistTradingHours } from "../frontend/src/utils/marketHours";

export const API_PREFIX = "/api";
export const MARKET_STOCKS_PATH = `${API_PREFIX}/market/stocks`;
export const FUNDS_LIST_PATH = `${API_PREFIX}/funds`;
export const MARKET_FLOW_HEAD_PATH = `${API_PREFIX}/market/flow/head`;
export const KAP_COMPANIES_PATH = `${API_PREFIX}/kap/companies`;
const EDGE_HOST_QUERY_PARAM = "__finrag_edge_host";
const MAX_MARKET_CACHE_AGE_SECONDS = 300;

export type CacheableListRoute = "market-stocks" | "funds" | "market-flow-head" | "kap-companies";

interface IstanbulDateParts {
    year: number;
    month: number;
    day: number;
    weekday: string;
    hour: number;
    minute: number;
    second: number;
}

const istanbulDateTimeFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "Europe/Istanbul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
});

function getIstanbulDateParts(date: Date): IstanbulDateParts {
    const parts = istanbulDateTimeFormatter.formatToParts(date);
    const values = new Map(parts.map(({ type, value }) => [type, value]));
    return {
        year: Number(values.get("year")),
        month: Number(values.get("month")),
        day: Number(values.get("day")),
        weekday: values.get("weekday") || "Mon",
        hour: Number(values.get("hour")) % 24,
        minute: Number(values.get("minute")),
        second: Number(values.get("second")),
    };
}

function istanbulWallTimeToUtc(
    year: number,
    month: number,
    day: number,
    hour: number,
    minute: number,
    referenceDate: Date,
): number {
    const targetWallTime = Date.UTC(year, month - 1, day, hour, minute);
    const referenceParts = getIstanbulDateParts(referenceDate);
    const referenceWallTime = Date.UTC(
        referenceParts.year,
        referenceParts.month - 1,
        referenceParts.day,
        referenceParts.hour,
        referenceParts.minute,
        referenceParts.second,
    );
    const referenceUtcTime = Math.floor(referenceDate.getTime() / 1_000) * 1_000;
    let candidate = targetWallTime - (referenceWallTime - referenceUtcTime);

    // Recalculate the offset at the target date as well, so a future timezone
    // transition cannot leave a pre-open cache entry fresh into the session.
    for (let attempt = 0; attempt < 3; attempt += 1) {
        const candidateParts = getIstanbulDateParts(new Date(candidate));
        const candidateWallTime = Date.UTC(
            candidateParts.year,
            candidateParts.month - 1,
            candidateParts.day,
            candidateParts.hour,
            candidateParts.minute,
            candidateParts.second,
        );
        const candidateUtcTime = Math.floor(candidate / 1_000) * 1_000;
        const nextCandidate = targetWallTime - (candidateWallTime - candidateUtcTime);
        if (nextCandidate === candidate) break;
        candidate = nextCandidate;
    }

    return candidate;
}

function secondsUntilNextBistSession(now: Date): number {
    const localNow = getIstanbulDateParts(now);
    const localMinutes = localNow.hour * 60 + localNow.minute;
    const sessionStartMinutes = 9 * 60 + 50;

    for (let dayOffset = 0; dayOffset <= 7; dayOffset += 1) {
        const localDate = new Date(Date.UTC(localNow.year, localNow.month - 1, localNow.day + dayOffset));
        const weekday = localDate.getUTCDay();
        if (weekday === 0 || weekday === 6) continue;
        if (dayOffset === 0 && localMinutes >= sessionStartMinutes) continue;

        const openAt = istanbulWallTimeToUtc(
            localDate.getUTCFullYear(),
            localDate.getUTCMonth() + 1,
            localDate.getUTCDate(),
            9,
            50,
            now,
        );
        return Math.max(0, Math.floor((openAt - now.getTime()) / 1_000));
    }

    return MAX_MARKET_CACHE_AGE_SECONDS;
}

export function getCacheableListRoute(url: URL): CacheableListRoute | null {
    if (url.pathname === MARKET_STOCKS_PATH) return "market-stocks";
    if (url.pathname === FUNDS_LIST_PATH) return "funds";
    if (url.pathname === MARKET_FLOW_HEAD_PATH) return "market-flow-head";
    if (url.pathname === KAP_COMPANIES_PATH) return "kap-companies";
    return null;
}

export function isMarketRefreshRequest(url: URL): boolean {
    return getCacheableListRoute(url) === "market-stocks"
        && url.searchParams.getAll("refresh").some((value) => value.trim().toLowerCase() === "true");
}

export function shouldUseCachedList(request: Request): boolean {
    const url = new URL(request.url);
    const route = getCacheableListRoute(url);
    if (request.method !== "GET" || !route || isMarketRefreshRequest(url)) return false;
    if (request.headers.has("Authorization") || request.headers.has("Cookie")) return false;

    const requestOrigin = request.headers.get("Origin");
    if (requestOrigin && requestOrigin !== url.origin) return false;
    return true;
}

export function shouldPurgeMarketCache(request: Request): boolean {
    return request.method === "GET" && isMarketRefreshRequest(new URL(request.url));
}

export async function purgeMarketCacheForRefresh(
    request: Request,
    purge: (tag: string) => Promise<void>,
): Promise<boolean> {
    if (!shouldPurgeMarketCache(request)) return false;
    await purge(marketStocksCacheTag(new URL(request.url)));
    return true;
}

export function marketStocksCacheTag(requestUrl: URL): string {
    const suppliedIndex = requestUrl.searchParams.getAll("index").at(-1)?.trim().toUpperCase() || "XUTUM";
    const safeIndex = suppliedIndex.replace(/[^A-Z0-9_-]/g, "_").slice(0, 32) || "UNKNOWN";
    return `market-stocks-${safeIndex}`;
}

export function cacheTagsForList(route: CacheableListRoute, requestUrl: URL): string[] {
    if (route === "funds") return ["funds-list"];
    if (route === "market-flow-head") return ["market-flow-head"];
    if (route === "kap-companies") return ["kap-companies"];
    return ["market-stocks", marketStocksCacheTag(requestUrl)];
}

export function canonicalizeCacheRequest(request: Request): Request {
    const url = new URL(request.url);
    url.searchParams.delete(EDGE_HOST_QUERY_PARAM);
    url.searchParams.sort();
    // Workers Cache keys do not include the request hostname by default. Add
    // the trusted gateway host to the URL query before entering the cached
    // entrypoint so separate public hosts cannot share a response.
    url.searchParams.set(EDGE_HOST_QUERY_PARAM, url.host.toLowerCase());
    return new Request(url, request);
}

export function createUpstreamRequest(request: Request, apiBaseUrl: string): Request {
    const requestUrl = new URL(request.url);
    if (!requestUrl.pathname.startsWith(`${API_PREFIX}/`)) {
        throw new Error("API proxy only accepts /api/* paths");
    }

    const apiPath = requestUrl.pathname.slice(API_PREFIX.length);
    requestUrl.searchParams.delete(EDGE_HOST_QUERY_PARAM);
    const upstreamUrl = new URL(`${apiPath}${requestUrl.search}`, apiBaseUrl);
    const init: RequestInit & { duplex?: "half" } = {
        method: request.method,
        headers: new Headers(request.headers),
        redirect: "manual",
        signal: request.signal,
    };
    if (request.method !== "GET" && request.method !== "HEAD" && request.body) {
        init.body = request.body;
        init.duplex = "half";
    }
    return new Request(upstreamUrl, init);
}

export function cacheControlForListRequest(request: Request, now: Date = new Date()): string {
    const url = new URL(request.url);
    const route = getCacheableListRoute(url);
    if (route === "funds") {
        return "public, max-age=300, stale-while-revalidate=3600, stale-if-error=0";
    }
    if (route === "market-flow-head") {
        // The KAP flow is refreshed by cron once per minute, and the panel
        // polls every 5s. A 10s edge TTL is the smallest value that can serve
        // at least every other poll for a single client.
        return "public, max-age=10, stale-while-revalidate=10, stale-if-error=0";
    }
    if (route === "kap-companies") {
        return "public, max-age=3600, stale-while-revalidate=86400, stale-if-error=0";
    }
    if (route !== "market-stocks") return "no-store";

    if (isBistTradingHours(now)) {
        return "public, max-age=3, stale-while-revalidate=30, stale-if-error=0";
    }

    const freshSeconds = Math.min(
        MAX_MARKET_CACHE_AGE_SECONDS,
        secondsUntilNextBistSession(now),
    );
    // Keep the closed-session entry from remaining fresh after the next BIST
    // open. Revalidation happens synchronously at expiry in this window.
    return `public, max-age=${freshSeconds}, stale-if-error=0`;
}

export async function isCacheableListResponse(
    response: Response,
    route: CacheableListRoute,
): Promise<boolean> {
    if (response.status !== 200 || response.headers.has("Set-Cookie")) return false;
    if (!response.headers.get("Content-Type")?.toLowerCase().includes("application/json")) return false;

    let payload: Record<string, unknown>;
    try {
        payload = await response.clone().json() as Record<string, unknown>;
    } catch {
        return false;
    }

    const status = String(payload.status || "").trim().toLowerCase();
    if (["pending", "unavailable", "error", "empty", "failed"].includes(status)) return false;
    // A queued background refresh does not make the current published snapshot
    // partial: keep serving/cache the last complete rows while it runs.
    if (payload.pending === true || payload.degraded === true || payload.stale === true) {
        return false;
    }
    if (route === "funds" && status && status !== "ok") return false;
    if (route === "market-flow-head") {
        const refreshStatus = String(payload.refresh_status || "").trim().toLowerCase();
        if (["empty", "error", "failed", "unavailable"].includes(refreshStatus)) return false;
        if (!payload.latest_cursor) return false;
    }
    if (route === "market-stocks" && Array.isArray(payload.rows) && payload.rows.length === 0) return false;
    if (route === "kap-companies" && (!Array.isArray(payload.companies) || payload.companies.length === 0)) return false;
    return true;
}

export async function applyListCachePolicy(
    response: Response,
    request: Request,
    route: CacheableListRoute,
    now: Date = new Date(),
): Promise<Response> {
    // A 304 has no body; caching it as a standalone entry is not meaningful and
    // would make the edge serve an empty response without revalidation.
    if (response.status === 304) return response;

    const cacheable = await isCacheableListResponse(response, route);
    const headers = new Headers(response.headers);
    headers.delete("Cache-Tag");
    headers.delete("Cloudflare-CDN-Cache-Control");

    if (!cacheable) {
        headers.set("Cache-Control", "private, no-store");
        return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers,
        });
    }

    headers.set("Cloudflare-CDN-Cache-Control", cacheControlForListRequest(request, now));
    headers.set("Cache-Tag", cacheTagsForList(route, new URL(request.url)).join(","));
    return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
    });
}
