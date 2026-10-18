const ISTANBUL_TIMEZONE = 'Europe/Istanbul';

const istanbulPartsFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: ISTANBUL_TIMEZONE,
    weekday: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
});

export interface IstanbulClockParts {
    weekday: string;
    hour: number;
    minute: number;
    minutesOfDay: number;
}

export function getIstanbulClockParts(now: Date = new Date()): IstanbulClockParts {
    const parts = istanbulPartsFormatter.formatToParts(now);
    let weekday = 'Mon';
    let hour = 0;
    let minute = 0;

    for (const part of parts) {
        if (part.type === 'weekday') weekday = part.value;
        else if (part.type === 'hour') hour = Number(part.value) % 24;
        else if (part.type === 'minute') minute = Number(part.value);
    }

    return {
        weekday,
        hour,
        minute,
        minutesOfDay: hour * 60 + minute,
    };
}

/**
 * Returns true during Borsa Istanbul active session window (Mon-Fri 09:50 - 18:20 Europe/Istanbul),
 * covering the 09:55 opening auction through the 18:10 closing auction + settlement buffer.
 */
export function isBistTradingHours(now: Date = new Date()): boolean {
    const { weekday, minutesOfDay } = getIstanbulClockParts(now);
    if (weekday === 'Sat' || weekday === 'Sun') {
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
