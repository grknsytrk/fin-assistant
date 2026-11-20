export const FAST_REFRESH_PATH = "/admin/kap/refresh?fast=1";
export const DEEP_REFRESH_PATH = "/admin/kap/refresh";

const FAST_CRON_EXPRESSIONS = new Set(["* * * * *", "*/2 * * * *"]);

const istanbulScheduleFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "Europe/Istanbul",
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
});

function getIstanbulScheduleParts(scheduledMs: number): { weekday: string; hour: number; minute: number } {
    const parts = istanbulScheduleFormatter.formatToParts(new Date(scheduledMs));
    const values = new Map(parts.map(({ type, value }) => [type, value]));
    return {
        weekday: values.get("weekday") || "Mon",
        hour: Number(values.get("hour")) % 24,
        minute: Number(values.get("minute")),
    };
}

export function resolveScheduledRefreshPath(cron: string, scheduledTime: number): string | null {
    const scheduledMs = Number.isFinite(scheduledTime) ? scheduledTime : Date.now();
    const { weekday, hour, minute } = getIstanbulScheduleParts(scheduledMs);
    const isWeekend = weekday === "Sat" || weekday === "Sun";
    const isNightHours = hour < 7;
    const isQuietWindow = isWeekend || isNightHours;
    const isPeakWindow = !isWeekend && hour >= 9 && hour < 19;

    if (FAST_CRON_EXPRESSIONS.has(cron)) {
        // Run fast KAP polling only during weekday peak BIST hours (09:00-18:59 Istanbul).
        if (!isPeakWindow) {
            return null;
        }
        // Poll every 2 minutes on even minutes and yield to deep refresh on 10-minute boundaries.
        if (minute % 2 !== 0 || minute % 10 === 0) {
            return null;
        }
        return FAST_REFRESH_PATH;
    }

    // Keep the backend warm 24/7 with hourly deep refreshes during nights and weekends.
    if (isQuietWindow) {
        return minute === 0 ? DEEP_REFRESH_PATH : null;
    }

    // During active weekday hours (07:00-23:59 Istanbul), run deep refresh every 10 minutes.
    if (minute % 10 !== 0) {
        return null;
    }

    return DEEP_REFRESH_PATH;
}
