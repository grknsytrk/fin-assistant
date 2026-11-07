export const FAST_REFRESH_PATH = "/admin/kap/refresh?fast=1";
export const DEEP_REFRESH_PATH = "/admin/kap/refresh";

export function resolveScheduledRefreshPath(cron: string, scheduledTime: number): string | null {
    const scheduledMs = Number.isFinite(scheduledTime) ? scheduledTime : Date.now();
    const minute = new Date(scheduledMs).getUTCMinutes();

    // Prevent collision: on 5-minute boundaries, yield to deep refresh
    if (cron === "* * * * *" && minute % 5 === 0) {
        return null;
    }
    return cron === "* * * * *" ? FAST_REFRESH_PATH : DEEP_REFRESH_PATH;
}
