const ISTANBUL_TIMEZONE = 'Europe/Istanbul';

const istanbulPartsFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: ISTANBUL_TIMEZONE,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    weekday: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
});

/**
 * BIST full-day closures, Europe/Istanbul calendar dates.
 * Source: Borsa İstanbul official holiday calendar.
 * Keep in sync with app/fund_service.py::_TURKEY_MARKET_FULL_DAY_HOLIDAYS.
 * Half-days are intentionally NOT listed: polling a few extra hours is harmless.
 */
const BIST_FULL_DAY_HOLIDAYS: ReadonlySet<string> = new Set([
    // 2026 — mirrors app/fund_service.py:3776-3786
    '2026-01-01',
    '2026-03-20',
    '2026-04-23',
    '2026-05-01',
    '2026-05-19',
    '2026-05-27',
    '2026-05-28',
    '2026-05-29',
    '2026-07-15',
    '2026-08-30',
    '2026-10-29',
    // TODO(2027): add verified BIST holiday dates before 2027-01-01.
]);

export interface IstanbulClockParts {
    year: number;
    month: number;
    day: number;
    weekday: string;
    hour: number;
    minute: number;
    minutesOfDay: number;
    /** YYYY-MM-DD in Europe/Istanbul. */
    isoDate: string;
}

export function getIstanbulClockParts(now: Date = new Date()): IstanbulClockParts {
    const parts = istanbulPartsFormatter.formatToParts(now);
    let weekday = 'Mon';
    let hour = 0;
    let minute = 0;
    let year = 1970;
    let month = 1;
    let day = 1;

    for (const part of parts) {
        if (part.type === 'weekday') weekday = part.value;
        else if (part.type === 'year') year = Number(part.value);
        else if (part.type === 'month') month = Number(part.value);
        else if (part.type === 'day') day = Number(part.value);
        else if (part.type === 'hour') hour = Number(part.value) % 24;
        else if (part.type === 'minute') minute = Number(part.value);
    }

    const isoDate = `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;

    return {
        year,
        month,
        day,
        weekday,
        hour,
        minute,
        minutesOfDay: hour * 60 + minute,
        isoDate,
    };
}

/**
 * Returns true during Borsa Istanbul active session window (Mon-Fri 09:50 - 18:20 Europe/Istanbul),
 * covering the 09:55 opening auction through the 18:10 closing auction + settlement buffer.
 */
export function isBistTradingHours(now: Date = new Date()): boolean {
    const { weekday, minutesOfDay, isoDate } = getIstanbulClockParts(now);
    if (weekday === 'Sat' || weekday === 'Sun') {
        return false;
    }
    if (BIST_FULL_DAY_HOLIDAYS.has(isoDate)) {
        return false;
    }
    const sessionStart = 9 * 60 + 50; // 09:50
    const sessionEnd = 18 * 60 + 20;  // 18:20
    return minutesOfDay >= sessionStart && minutesOfDay <= sessionEnd;
}

/**
 * Returns true during 24/5 Global Indices, FX, and Commodity market hours
 * (Closes Saturday 01:00 Istanbul time, reopens Monday 00:00 / Sunday 23:55 Istanbul time).
 */
export function isGlobalFxTradingHours(now: Date = new Date()): boolean {
    const { weekday, minutesOfDay } = getIstanbulClockParts(now);
    if (weekday === 'Sat' && minutesOfDay >= 60) {
        return false;
    }
    if (weekday === 'Sun' && minutesOfDay < 23 * 60 + 55) {
        return false;
    }
    return true;
}

export type MarketPollingKind = 'bist' | 'global_fx';

/**
 * Computes the adaptive polling interval (in milliseconds) based on market session hours
 * and browser tab visibility (`document.visibilityState`).
 * Returns `null` when the market session is closed so callers pause periodic polling.
 */
export function getAdaptiveRefreshMs(
    kind: MarketPollingKind,
    options?: {
        activeMs?: number;
        hiddenMs?: number;
        now?: Date;
    },
): number | null {
    const now = options?.now ?? new Date();
    const isOpen = kind === 'bist' ? isBistTradingHours(now) : isGlobalFxTradingHours(now);
    if (!isOpen) {
        return null;
    }

    const isHidden = typeof document !== 'undefined' && document.visibilityState === 'hidden';
    if (isHidden) {
        return options?.hiddenMs ?? 30_000;
    }

    return options?.activeMs ?? (kind === 'bist' ? 3_000 : 5_000);
}
