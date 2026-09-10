const FAST_REFRESH_PATH = "/admin/kap/refresh?fast=1";
const DEEP_REFRESH_PATH = "/admin/kap/refresh";
const REFRESH_TIMEOUT_MS = 90_000;
const REFRESH_ATTEMPTS = 2;
const FINTABLES_GATE_BASE_URL = "https://gate.fintables.com";
const FINTABLES_PROXY_PATHS = new Set([
    "/internal/fintables/udf/history",
    "/internal/fintables/yield-summary",
]);

function asErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

async function wait(milliseconds: number): Promise<void> {
    await new Promise<void>((resolve) => setTimeout(resolve, milliseconds));
}

function jsonResponse(body: Record<string, unknown>, status: number): Response {
    return new Response(JSON.stringify(body), {
        status,
        headers: {
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "no-store",
        },
    });
}

async function secretsMatch(provided: string, expected: string): Promise<boolean> {
    if (!provided || !expected) return false;
    const encoder = new TextEncoder();
    const [providedHash, expectedHash] = await Promise.all([
        crypto.subtle.digest("SHA-256", encoder.encode(provided)),
        crypto.subtle.digest("SHA-256", encoder.encode(expected)),
    ]);
    return crypto.subtle.timingSafeEqual(providedHash, expectedHash);
}

function copyQueryParam(source: URL, target: URL, name: string): string | null {
    const value = source.searchParams.get(name)?.trim() || "";
    if (!value) return null;
    target.searchParams.set(name, value);
    return value;
}

async function handleFintablesProxy(request: Request, env: Env): Promise<Response | null> {
    const requestUrl = new URL(request.url);
    if (!FINTABLES_PROXY_PATHS.has(requestUrl.pathname)) return null;
    if (request.method !== "GET") {
        return jsonResponse({ error: "method_not_allowed" }, 405);
    }

    const expectedToken = String(env.FIN_API_ADMIN_TOKEN || "").trim();
    const providedToken = request.headers.get("Authorization") || "";
    if (!expectedToken) {
        return jsonResponse({ error: "proxy_not_configured" }, 503);
    }
    if (!(await secretsMatch(providedToken, `Bearer ${expectedToken}`))) {
        return jsonResponse({ error: "unauthorized" }, 401);
    }

    const target = new URL(
        requestUrl.pathname === "/internal/fintables/udf/history"
            ? `${FINTABLES_GATE_BASE_URL}/barbar/udf/history`
            : `${FINTABLES_GATE_BASE_URL}/barbar/server/yield`,
    );
    if (requestUrl.pathname === "/internal/fintables/udf/history") {
        const symbol = copyQueryParam(requestUrl, target, "symbol");
        const resolution = copyQueryParam(requestUrl, target, "resolution");
        const from = copyQueryParam(requestUrl, target, "from");
        const to = copyQueryParam(requestUrl, target, "to");
        if (!symbol || !/^[A-Z0-9._-]{1,20}$/.test(symbol) || resolution !== "D" ||
            !from || !/^\d{8,12}$/.test(from) || !to || !/^\d{8,12}$/.test(to)) {
            return jsonResponse({ error: "invalid_history_query" }, 400);
        }
    } else {
        const code = copyQueryParam(requestUrl, target, "code");
        if (!code || !/^[A-Z0-9._-]{1,20}$/.test(code)) {
            return jsonResponse({ error: "invalid_yield_query" }, 400);
        }
    }

    try {
        const upstream = await fetch(target, {
            headers: {
                Accept: "*/*",
                "Cache-Control": "no-cache",
                "User-Agent": "PostmanRuntime/7.51.0",
            },
        });
        const headers = new Headers(upstream.headers);
        headers.set("Cache-Control", "no-store");
        headers.delete("Set-Cookie");
        return new Response(upstream.body, {
            status: upstream.status,
            statusText: upstream.statusText,
            headers,
        });
    } catch (error) {
        console.error(JSON.stringify({
            message: "Fintables proxy upstream request failed",
            path: requestUrl.pathname,
            error: asErrorMessage(error),
        }));
        return jsonResponse({ error: "upstream_unavailable" }, 502);
    }
}

async function requestKapRefresh(baseUrl: string, token: string, path: string): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REFRESH_TIMEOUT_MS);
    try {
        return await fetch(new URL(path, baseUrl), {
            method: "POST",
            headers: {
                Authorization: `Bearer ${token}`,
                Accept: "application/json",
            },
            signal: controller.signal,
        });
    } finally {
        clearTimeout(timeoutId);
    }
}

async function refreshKapFlow(controller: ScheduledController, env: Env): Promise<void> {
    const refreshPath = controller.cron === "* * * * *" ? FAST_REFRESH_PATH : DEEP_REFRESH_PATH;
    let response: Response | null = null;
    let lastError: string | null = null;
    let attemptsMade = 0;

    for (let attempt = 1; attempt <= REFRESH_ATTEMPTS; attempt += 1) {
        attemptsMade = attempt;
        try {
            response = await requestKapRefresh(env.FIN_API_BASE_URL, env.FIN_API_ADMIN_TOKEN, refreshPath);
            if (response.ok || (response.status !== 429 && response.status < 500)) break;
            lastError = `backend returned HTTP ${response.status}`;
        } catch (error) {
            lastError = asErrorMessage(error);
        }

        if (attempt < REFRESH_ATTEMPTS) await wait(3_000);
    }

    if (!response) {
        throw new Error(`KAP refresh request failed: ${lastError || "unknown error"}`);
    }

    const body = await response.text();
    if (!response.ok) {
        const logPayload = {
            message: "KAP refresh backend rejected the request",
            status: response.status,
            attemptCount: attemptsMade,
            refreshPath,
        };
        if (response.status === 401 || response.status === 403) {
            controller.noRetry();
            console.error(JSON.stringify(logPayload));
            return;
        }
        throw new Error(`${logPayload.message}: HTTP ${response.status}`);
    }

    let payload: { status?: string; stored_count?: number; database_written?: boolean } = {};
    try {
        payload = JSON.parse(body) as typeof payload;
    } catch {
        throw new Error("KAP refresh backend returned invalid JSON");
    }

    const status = String(payload.status || "");
    if (status === "ok" || status === "already_running") {
        console.log(JSON.stringify({
            message: "KAP flow refresh completed",
            cron: controller.cron,
            refreshPath,
            scheduledTime: controller.scheduledTime,
            status,
            storedCount: payload.stored_count ?? 0,
            databaseWritten: payload.database_written ?? false,
        }));
        return;
    }

    if (status === "empty") {
        controller.noRetry();
        console.warn(JSON.stringify({
            message: "KAP flow refresh returned no rows; existing data remains authoritative",
            cron: controller.cron,
            refreshPath,
            scheduledTime: controller.scheduledTime,
        }));
        return;
    }

    throw new Error(`KAP refresh returned unexpected status: ${status || "missing"}`);
}

export default {
    async fetch(request: Request, env: Env): Promise<Response> {
        const url = new URL(request.url);
        const fintablesResponse = await handleFintablesProxy(request, env);
        if (fintablesResponse) return fintablesResponse;
        if (url.pathname === "/__scheduled") {
            return new Response("Not Found", { status: 404 });
        }
        return env.ASSETS.fetch(request);
    },

    async scheduled(controller: ScheduledController, env: Env): Promise<void> {
        try {
            await refreshKapFlow(controller, env);
        } catch (error) {
            console.error(JSON.stringify({
                message: "KAP flow refresh failed",
                cron: controller.cron,
                refreshPath: controller.cron === "* * * * *" ? FAST_REFRESH_PATH : DEEP_REFRESH_PATH,
                scheduledTime: controller.scheduledTime,
                error: asErrorMessage(error),
            }));
            throw error;
        }
    },
} satisfies ExportedHandler<Env>;
