export const FAST_REFRESH_PATH = "/admin/kap/refresh?fast=1";
export const DEEP_REFRESH_PATH = "/admin/kap/refresh";

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

    if (cron === "* * * * *") {
        // Skip 1-minute fast cron during Istanbul night hours (00:00-07:00) and weekends.
        if (isQuietWindow) {
            return null;
        }
        // Prevent collision: on 5-minute boundaries, yield to deep refresh.
        if (minute % 5 === 0) {
            return null;
        }
        return FAST_REFRESH_PATH;
    }

    // Throttle deep refresh (*/5) during night/weekends to run only at :00 and :30.
    if (isQuietWindow && minute !== 0 && minute !== 30) {
        return null;
    }

    return DEEP_REFRESH_PATH;
}
