import { afterEach, describe, expect, it, vi } from 'vitest';

import { resolveScheduledRefreshPath, FAST_REFRESH_PATH, DEEP_REFRESH_PATH } from '../../worker/cron-schedule';

describe('Worker Cron Collision Resolution', () => {
    afterEach(() => {
        vi.useRealTimers();
    });

    it('skips fast cron on 10-minute boundaries and odd minutes during peak BIST hours', () => {
        // Monday 2026-11-09 12:xx UTC = 15:xx Istanbul (weekday peak 09:00-18:59)
        for (const minute of [0, 1, 3, 5, 10, 15, 20, 30, 50, 55]) {
            const ts = Date.UTC(2026, 10, 9, 12, minute, 0);
            expect(resolveScheduledRefreshPath('*/2 * * * *', ts)).toBeNull();
            expect(resolveScheduledRefreshPath('* * * * *', ts)).toBeNull();
        }
    });

    it('executes fast refresh on even non-10-minute boundaries during peak Istanbul hours', () => {
        // Monday 2026-11-09 08:xx UTC = 11:xx Istanbul (weekday peak 09:00-18:59)
        for (let minute = 0; minute < 60; minute++) {
            const ts = new Date(Date.UTC(2026, 10, 9, 8, minute, 15)).getTime();
            const expected = (minute % 2 === 0 && minute % 10 !== 0) ? FAST_REFRESH_PATH : null;
            expect(resolveScheduledRefreshPath('*/2 * * * *', ts)).toBe(expected);
            expect(resolveScheduledRefreshPath('* * * * *', ts)).toBe(expected);
        }
    });

    it('skips fast cron and runs deep refresh every 10 minutes during shoulder hours (07:00-08:59 and 19:00-23:59 Istanbul)', () => {
        // Monday 2026-11-09 04:xx UTC = 07:xx Istanbul (early shoulder)
        expect(resolveScheduledRefreshPath('*/2 * * * *', Date.parse('2026-11-09T04:02:00Z'))).toBeNull();
        expect(resolveScheduledRefreshPath('*/10 * * * *', Date.parse('2026-11-09T04:00:00Z'))).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/10 * * * *', Date.parse('2026-11-09T04:10:00Z'))).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/5 * * * *', Date.parse('2026-11-09T04:05:00Z'))).toBeNull();

        // Monday 2026-11-09 17:xx UTC = 20:xx Istanbul (evening shoulder)
        expect(resolveScheduledRefreshPath('*/2 * * * *', Date.parse('2026-11-09T17:04:00Z'))).toBeNull();
        expect(resolveScheduledRefreshPath('*/10 * * * *', Date.parse('2026-11-09T17:20:00Z'))).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/5 * * * *', Date.parse('2026-11-09T17:15:00Z'))).toBeNull();
    });

    it('executes deep refresh only on 10-minute boundaries during peak Istanbul hours', () => {
        // Monday 2026-11-09 12:xx UTC = 15:xx Istanbul
        expect(resolveScheduledRefreshPath('*/10 * * * *', Date.parse('2026-11-09T12:00:00Z'))).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/10 * * * *', Date.parse('2026-11-09T12:10:00Z'))).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/5 * * * *', Date.parse('2026-11-09T12:05:00Z'))).toBeNull();
        expect(resolveScheduledRefreshPath('*/5 * * * *', Date.parse('2026-11-09T12:10:00Z'))).toBe(DEEP_REFRESH_PATH);
    });

    it('skips fast cron and throttles deep cron to hourly (:00) during Istanbul night hours and weekends', () => {
        // Monday 2026-11-09 01:xx UTC = 04:xx Istanbul (night quiet window 00:00-06:59)
        expect(resolveScheduledRefreshPath('*/2 * * * *', Date.parse('2026-11-09T01:02:00Z'))).toBeNull();
        expect(resolveScheduledRefreshPath('*/10 * * * *', Date.parse('2026-11-09T01:00:00Z'))).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/10 * * * *', Date.parse('2026-11-09T01:10:00Z'))).toBeNull();
        expect(resolveScheduledRefreshPath('*/10 * * * *', Date.parse('2026-11-09T01:30:00Z'))).toBeNull();

        // Saturday 2026-11-07 12:xx UTC = 15:xx Istanbul (weekend quiet window)
        expect(resolveScheduledRefreshPath('*/2 * * * *', Date.parse('2026-11-07T12:02:00Z'))).toBeNull();
        expect(resolveScheduledRefreshPath('*/10 * * * *', Date.parse('2026-11-07T12:00:00Z'))).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/10 * * * *', Date.parse('2026-11-07T12:10:00Z'))).toBeNull();
        expect(resolveScheduledRefreshPath('*/10 * * * *', Date.parse('2026-11-07T12:30:00Z'))).toBeNull();
    });

    it('handles non-finite or missing scheduledTime safely', () => {
        // When scheduledTime is not finite, resolveScheduledRefreshPath falls back to Date.now()
        vi.useFakeTimers();
        vi.setSystemTime(new Date('2026-11-09T12:02:00Z'));
        expect(resolveScheduledRefreshPath('*/2 * * * *', Number.NaN)).toBe(FAST_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('* * * * *', Number.NaN)).toBe(FAST_REFRESH_PATH);
        vi.setSystemTime(new Date('2026-11-09T12:10:00Z'));
        expect(resolveScheduledRefreshPath('*/10 * * * *', Number.NaN)).toBe(DEEP_REFRESH_PATH);
    });

    it('exports the expected path constants', () => {
        expect(FAST_REFRESH_PATH).toBe('/admin/kap/refresh?fast=1');
        expect(DEEP_REFRESH_PATH).toBe('/admin/kap/refresh');
    });
});
