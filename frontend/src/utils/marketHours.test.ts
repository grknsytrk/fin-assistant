import { describe, expect, it } from 'vitest';
import {
    getAdaptiveRefreshMs,
    getIstanbulClockParts,
    isBistTradingHours,
} from './marketHours';

describe('isBistTradingHours — Baseline Plan Specifications', () => {
    it('is open on a regular weekday during the session', () => {
        // Wednesday 2026-11-04 12:00 Istanbul (09:00 UTC)
        expect(isBistTradingHours(new Date('2026-11-04T09:00:00.000Z'))).toBe(true);
    });

    it('is closed on weekends', () => {
        expect(isBistTradingHours(new Date('2026-10-25T11:00:00.000Z'))).toBe(false);
    });

    it('is closed outside session hours', () => {
        expect(isBistTradingHours(new Date('2026-11-04T19:00:00.000Z'))).toBe(false);
    });

    it('is closed on a full-day BIST holiday', () => {
        // 2026-10-29 Republic Day, 12:00 Istanbul (09:00 UTC)
        expect(isBistTradingHours(new Date('2026-10-29T09:00:00.000Z'))).toBe(false);
    });
});

describe('isBistTradingHours — Session Boundary Edge Cases', () => {
    // Istanbul is UTC+3. Session window: 09:50 - 18:20 Istanbul time.

    it('is closed 1 minute before session opening (09:49 Istanbul / 06:49 UTC)', () => {
        expect(isBistTradingHours(new Date('2026-11-04T06:49:00.000Z'))).toBe(false);
    });

    it('is open exactly at session opening buffer (09:50 Istanbul / 06:50 UTC)', () => {
        expect(isBistTradingHours(new Date('2026-11-04T06:50:00.000Z'))).toBe(true);
    });

    it('is open during continuous trading (14:30 Istanbul / 11:30 UTC)', () => {
        expect(isBistTradingHours(new Date('2026-11-04T11:30:00.000Z'))).toBe(true);
    });

    it('is open exactly at session closing buffer (18:20 Istanbul / 15:20 UTC)', () => {
        expect(isBistTradingHours(new Date('2026-11-04T15:20:00.000Z'))).toBe(true);
    });

    it('is closed 1 minute after session closing (18:21 Istanbul / 15:21 UTC)', () => {
        expect(isBistTradingHours(new Date('2026-11-04T15:21:00.000Z'))).toBe(false);
    });

    it('is closed during early morning pre-market (08:00 Istanbul / 05:00 UTC)', () => {
        expect(isBistTradingHours(new Date('2026-11-04T05:00:00.000Z'))).toBe(false);
    });

    it('is closed at night / after hours (22:00 Istanbul / 19:00 UTC)', () => {
        expect(isBistTradingHours(new Date('2026-11-04T19:00:00.000Z'))).toBe(false);
    });
});

describe('isBistTradingHours — Weekends', () => {
    it('is closed on Saturday midday (12:00 Istanbul / 09:00 UTC)', () => {
        expect(isBistTradingHours(new Date('2026-10-24T09:00:00.000Z'))).toBe(false);
    });

    it('is closed on Sunday midday (12:00 Istanbul / 09:00 UTC)', () => {
        expect(isBistTradingHours(new Date('2026-10-25T09:00:00.000Z'))).toBe(false);
    });

    it('is closed on Saturday morning at 09:55 (opening auction time on weekdays)', () => {
        expect(isBistTradingHours(new Date('2026-10-24T06:55:00.000Z'))).toBe(false);
    });
});

describe('isBistTradingHours — All 11 Verified 2026 BIST Full-Day Holidays', () => {
    const verified2026Holidays: Array<{ date: string; name: string }> = [
        { date: '2026-01-01', name: "New Year's Day (Thursday)" },
        { date: '2026-03-20', name: 'Ramazan Bayramı Day 1 (Friday)' },
        { date: '2026-04-23', name: 'National Sovereignty & Children Day (Thursday)' },
        { date: '2026-05-01', name: 'Labour and Solidarity Day (Friday)' },
        { date: '2026-05-19', name: 'Commemoration of Atatürk, Youth & Sports Day (Tuesday)' },
        { date: '2026-05-27', name: 'Kurban Bayramı Day 1 (Wednesday)' },
        { date: '2026-05-28', name: 'Kurban Bayramı Day 2 (Thursday)' },
        { date: '2026-05-29', name: 'Kurban Bayramı Day 3 (Friday)' },
        { date: '2026-07-15', name: 'Democracy & National Unity Day (Wednesday)' },
        { date: '2026-08-30', name: 'Victory Day (Sunday)' },
        { date: '2026-10-29', name: 'Republic Day (Thursday)' },
    ];

    it.each(verified2026Holidays)(
        'is closed during midday session hours on holiday $date ($name)',
        ({ date }) => {
            // 12:00 Istanbul time = 09:00 UTC
            const testDate = new Date(`${date}T09:00:00.000Z`);
            expect(isBistTradingHours(testDate)).toBe(false);
        },
    );
});

describe('isBistTradingHours — Adjacent Regular Trading Days & Half-Days', () => {
    it('is open on the regular business day immediately after New Year (Friday 2026-01-02 12:00 Istanbul)', () => {
        expect(isBistTradingHours(new Date('2026-01-02T09:00:00.000Z'))).toBe(true);
    });

    it('is open on the regular business day immediately after Republic Day (Friday 2026-10-30 12:00 Istanbul)', () => {
        expect(isBistTradingHours(new Date('2026-10-30T09:00:00.000Z'))).toBe(true);
    });

    it('remains open during morning session on half-day eve of Republic Day (Wednesday 2026-10-28 10:30 Istanbul / 07:30 UTC)', () => {
        // Half-days are intentionally NOT listed in full-day closure calendar.
        expect(isBistTradingHours(new Date('2026-10-28T07:30:00.000Z'))).toBe(true);
    });
});

describe('getIstanbulClockParts — Date & Time Breakdown', () => {
    it('correctly parses Europe/Istanbul date parts including isoDate', () => {
        // 2026-11-04 14:45:00 Istanbul = 11:45:00 UTC
        const parts = getIstanbulClockParts(new Date('2026-11-04T11:45:00.000Z'));
        expect(parts).toEqual({
            year: 2026,
            month: 11,
            day: 4,
            weekday: 'Wed',
            hour: 14,
            minute: 45,
            minutesOfDay: 14 * 60 + 45,
            isoDate: '2026-11-04',
        });
    });

    it('correctly handles UTC-to-Istanbul midnight rollover', () => {
        // 2026-10-28 22:30:00 UTC is 2026-10-29 01:30:00 in Europe/Istanbul (UTC+3)
        const parts = getIstanbulClockParts(new Date('2026-10-28T22:30:00.000Z'));
        expect(parts.isoDate).toBe('2026-10-29');
        expect(parts.weekday).toBe('Thu');
        expect(parts.hour).toBe(1);
        expect(parts.minute).toBe(30);
    });
});

describe('getAdaptiveRefreshMs integration with BIST trading hours', () => {
    it('returns active polling rate (3,000 ms) during active session on regular trading day', () => {
        const openDate = new Date('2026-11-04T09:00:00.000Z');
        expect(getAdaptiveRefreshMs('bist', { now: openDate })).toBe(3_000);
    });

    it('returns null on a full-day BIST holiday', () => {
        const holidayDate = new Date('2026-10-29T09:00:00.000Z');
        expect(getAdaptiveRefreshMs('bist', { now: holidayDate })).toBeNull();
    });

    it('returns null on weekends', () => {
        const weekendDate = new Date('2026-10-25T09:00:00.000Z');
        expect(getAdaptiveRefreshMs('bist', { now: weekendDate })).toBeNull();
    });

    it('returns null outside session hours on a regular trading day', () => {
        const closedDate = new Date('2026-11-04T19:00:00.000Z');
        expect(getAdaptiveRefreshMs('bist', { now: closedDate })).toBeNull();
    });
});

describe('isBistTradingHours — Parameter-less default call', () => {
    it('executes with default current Date without throwing', () => {
        const result = isBistTradingHours();
        expect(typeof result).toBe('boolean');
    });
});
