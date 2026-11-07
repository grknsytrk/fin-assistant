import { describe, expect, it } from 'vitest';

import { resolveScheduledRefreshPath, FAST_REFRESH_PATH, DEEP_REFRESH_PATH } from '../../worker/cron-schedule';

describe('Worker Cron Collision Resolution', () => {
    it('skips 1-minute cron on 5-minute boundaries', () => {
        const atMinute00 = Date.parse('2026-11-07T12:00:00Z');
        const atMinute05 = Date.parse('2026-11-07T12:05:00Z');
        const atMinute55 = Date.parse('2026-11-07T12:55:00Z');

        expect(resolveScheduledRefreshPath('* * * * *', atMinute00)).toBeNull();
        expect(resolveScheduledRefreshPath('* * * * *', atMinute05)).toBeNull();
        expect(resolveScheduledRefreshPath('* * * * *', atMinute55)).toBeNull();
    });

    it('verifies all 5-minute boundaries across an entire hour yield for 1-minute cron', () => {
        const fiveMinuteBoundaries = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55];
        for (const minute of fiveMinuteBoundaries) {
            const date = new Date(Date.UTC(2026, 10, 7, 14, minute, 0));
            expect(
                resolveScheduledRefreshPath('* * * * *', date.getTime()),
                `Minute ${minute} should yield to deep refresh (return null)`
            ).toBeNull();
        }
    });

    it('executes 1-minute fast refresh on non-5-minute boundaries', () => {
        const atMinute01 = Date.parse('2026-11-07T12:01:00Z');
        const atMinute04 = Date.parse('2026-11-07T12:04:00Z');

        expect(resolveScheduledRefreshPath('* * * * *', atMinute01)).toBe(FAST_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('* * * * *', atMinute04)).toBe(FAST_REFRESH_PATH);
    });

    it('verifies all non-5-minute minutes return FAST_REFRESH_PATH for 1-minute cron', () => {
        for (let minute = 0; minute < 60; minute++) {
            if (minute % 5 !== 0) {
                const date = new Date(Date.UTC(2026, 10, 7, 8, minute, 15));
                expect(
                    resolveScheduledRefreshPath('* * * * *', date.getTime()),
                    `Minute ${minute} should execute fast refresh`
                ).toBe(FAST_REFRESH_PATH);
            }
        }
    });

    it('always executes deep refresh on 5-minute cron', () => {
        const atMinute00 = Date.parse('2026-11-07T12:00:00Z');
        const atMinute05 = Date.parse('2026-11-07T12:05:00Z');
        const atMinute10 = Date.parse('2026-11-07T12:10:00Z');

        expect(resolveScheduledRefreshPath('*/5 * * * *', atMinute00)).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/5 * * * *', atMinute05)).toBe(DEEP_REFRESH_PATH);
        expect(resolveScheduledRefreshPath('*/5 * * * *', atMinute10)).toBe(DEEP_REFRESH_PATH);
    });

    it('handles non-finite or missing scheduledTime safely', () => {
        // When scheduledTime is not finite, resolveScheduledRefreshPath falls back to Date.now()
        const nowMinute = new Date().getUTCMinutes();
        const expected = (nowMinute % 5 === 0) ? null : FAST_REFRESH_PATH;
        expect(resolveScheduledRefreshPath('* * * * *', Number.NaN)).toBe(expected);
        expect(resolveScheduledRefreshPath('*/5 * * * *', Number.NaN)).toBe(DEEP_REFRESH_PATH);
    });

    it('exports the expected path constants', () => {
        expect(FAST_REFRESH_PATH).toBe('/admin/kap/refresh?fast=1');
        expect(DEEP_REFRESH_PATH).toBe('/admin/kap/refresh');
    });
});
