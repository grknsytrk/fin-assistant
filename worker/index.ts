const REFRESH_PATH = "/admin/kap/refresh";
const REFRESH_TIMEOUT_MS = 90_000;
const REFRESH_ATTEMPTS = 2;

function asErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

async function wait(milliseconds: number): Promise<void> {
    await new Promise<void>((resolve) => setTimeout(resolve, milliseconds));
}

async function requestKapRefresh(baseUrl: string, token: string): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REFRESH_TIMEOUT_MS);
    try {
        return await fetch(new URL(REFRESH_PATH, baseUrl), {
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
    let response: Response | null = null;
    let lastError: string | null = null;
    let attemptsMade = 0;

    for (let attempt = 1; attempt <= REFRESH_ATTEMPTS; attempt += 1) {
        attemptsMade = attempt;
        try {
            response = await requestKapRefresh(env.FIN_API_BASE_URL, env.FIN_API_ADMIN_TOKEN);
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
            scheduledTime: controller.scheduledTime,
        }));
        return;
    }

    throw new Error(`KAP refresh returned unexpected status: ${status || "missing"}`);
}

export default {
    async fetch(request: Request, env: Env): Promise<Response> {
        const url = new URL(request.url);
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
                scheduledTime: controller.scheduledTime,
                error: asErrorMessage(error),
            }));
            throw error;
        }
    },
} satisfies ExportedHandler<Env>;
