import { afterEach, describe, expect, it, vi } from 'vitest';

import { resolveScheduledRefreshPath, FAST_REFRESH_PATH, DEEP_REFRESH_PATH } from '../../worker/cron-schedule';

describe('Worker Cron Collision Resolution', () => {
    afterEach(() => {
        vi.useRealTimers();
    });

    it('skips 1-minute cron on 5-minute boundaries during active hours', () => {
        // Monday 2026-11-09 12:xx UTC = 15:xx Istanbul (active weekday daytime)
        const atMinute00 = Date.parse('2026-11-09T12:00:00Z');
        const atMinute05 = Date.parse('2026-11-09T12:05:00Z');
        const atMinute55 = Date.parse('2026-11-09T12:55:00Z');

        expect(resolveScheduledRefreshPath('* * * * *', atMinute00)).toBeNull();
        expect(resolveScheduledRefreshPath('* * * * *', atMinute05)).toBeNull();
        expect(resolveScheduledRefreshPath('* * * * *', atMinute55)).toBeNull();
    });

    it('verifies all 5-minute boundaries across an entire hour yield for 1-minute cron', () => {
        const fiveMinuteBoundaries = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55];
        for (const minute of fiveMinuteBoundaries) {
            const date = new Date(Date.UTC(2026, 10, 9, 14, minute, 0));
            expect(
                resolveScheduledRefreshPath('* * * * *', date.getTime()),
                `Minute ${minute} should yield to deep refresh (return null)`
            ).toBeNull();
        }
    });

    it('executes 1-minute fast refresh on non-5-minute boundaries during active Istanbul hours', () => {
        // Monday 2026-11-09 12:xx UTC = 15:xx Istanbul
        const atMinute01 = Date.parse('2026-11-09T12:01:00Z');
        const atMinute04 = Date.parse('2026-11-09T12:04:00Z');

        expect(resolveScheduledRefreshPath('* * * * *', atMinute01)).toBe(FAST_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('* * * * *', atMinute04)).toBe(FAST_REFRESH_PATH);
    });

    it('verifies all non-5-minute minutes return FAST_REFRESH_PATH for 1-minute cron during active Istanbul hours', () => {
        // Monday 2026-11-09 08:xx UTC = 11:xx Istanbul
        for (let minute = 0; minute < 60; minute++) {
            if (minute % 5 !== 0) {
                const date = new Date(Date.UTC(2026, 10, 9, 8, minute, 15));
                expect(
                    resolveScheduledRefreshPath('* * * * *', date.getTime()),
                    `Minute ${minute} should execute fast refresh`
                ).toBe(FAST_REFRESH_PATH);
            }
        }
    });

    it('executes deep refresh on 5-minute cron during active Istanbul hours', () => {
        // Monday 2026-11-09 12:xx UTC = 15:xx Istanbul
        const atMinute00 = Date.parse('2026-11-09T12:00:00Z');
        const atMinute05 = Date.parse('2026-11-09T12:05:00Z');
        const atMinute10 = Date.parse('2026-11-09T12:10:00Z');

        expect(resolveScheduledRefreshPath('*/5 * * * *', atMinute00)).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/5 * * * *', atMinute05)).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/5 * * * *', atMinute10)).toBe(DEEP_REFRESH_PATH);
    });

    it('skips fast cron and throttles deep cron to :00 and :30 during Istanbul night hours and weekends', () => {
        // Monday 2026-11-09 01:xx UTC = 04:xx Istanbul (night quiet window 00:00-07:00)
        expect(resolveScheduledRefreshPath('* * * * *', Date.parse('2026-11-09T01:01:00Z'))).toBeNull();
        expect(resolveScheduledRefreshPath('*/5 * * * *', Date.parse('2026-11-09T01:00:00Z'))).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/5 * * * *', Date.parse('2026-11-09T01:05:00Z'))).toBeNull();
        expect(resolveScheduledRefreshPath('*/5 * * * *', Date.parse('2026-11-09T01:30:00Z'))).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/5 * * * *', Date.parse('2026-11-09T01:35:00Z'))).toBeNull();

        // Saturday 2026-11-07 12:xx UTC = 15:xx Istanbul (weekend quiet window)
        expect(resolveScheduledRefreshPath('* * * * *', Date.parse('2026-11-07T12:01:00Z'))).toBeNull();
        expect(resolveScheduledRefreshPath('*/5 * * * *', Date.parse('2026-11-07T12:00:00Z'))).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/5 * * * *', Date.parse('2026-11-07T12:05:00Z'))).toBeNull();
        expect(resolveScheduledRefreshPath('*/5 * * * *', Date.parse('2026-11-07T12:30:00Z'))).toBe(DEEP_REFRESH_PATH);
    });

    it('handles non-finite or missing scheduledTime safely', () => {
        // When scheduledTime is not finite, resolveScheduledRefreshPath falls back to Date.now()
        vi.useFakeTimers();
        vi.setSystemTime(new Date('2026-11-09T12:01:00Z'));
        expect(resolveScheduledRefreshPath('* * * * *', Number.NaN)).toBe(FAST_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/5 * * * *', Number.NaN)).toBe(DEEP_REFRESH_PATH);
    });

    it('exports the expected path constants', () => {
        expect(FAST_REFRESH_PATH).toBe('/admin/kap/refresh?fast=1');
        expect(DEEP_REFRESH_PATH).toBe('/admin/kap/refresh');
    });
});
