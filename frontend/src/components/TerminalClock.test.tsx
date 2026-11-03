// @vitest-environment jsdom
import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { formatTerminalClock, formatTerminalDate, TerminalClock } from './TerminalClock';

afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.useRealTimers();
});

describe('TerminalClock', () => {
    it('renders initial formatted date and time', () => {
        const testDate = new Date('2026-11-03T14:30:15');
        vi.useFakeTimers();
        vi.setSystemTime(testDate);

        render(<TerminalClock />);

        expect(screen.getByText(formatTerminalDate(testDate))).toBeDefined();
        expect(screen.getByText(formatTerminalClock(testDate))).toBeDefined();
    });

    it('updates time on 1-second interval tick', async () => {
        vi.useFakeTimers();
        vi.setSystemTime(new Date('2026-11-03T10:00:00'));

        render(<TerminalClock intervalMs={1000} />);

        expect(screen.getByText('10:00:00')).toBeDefined();

        await act(async () => {
            vi.advanceTimersByTime(1000);
        });

        expect(screen.getByText('10:00:01')).toBeDefined();
    });

    it('pauses timer updates when document is hidden and resumes on visible', async () => {
        vi.useFakeTimers();
        vi.setSystemTime(new Date('2026-11-03T10:00:00'));

        const visibilitySpy = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible');
        render(<TerminalClock intervalMs={1000} />);

        expect(screen.getByText('10:00:00')).toBeDefined();

        // Switch to hidden
        await act(async () => {
            visibilitySpy.mockReturnValue('hidden');
            document.dispatchEvent(new Event('visibilitychange'));
        });

        // Advance 5 seconds while hidden
        await act(async () => {
            vi.advanceTimersByTime(5000);
        });

        // Still 10:00:00 because tick was halted while hidden
        expect(screen.getByText('10:00:00')).toBeDefined();

        // Switch to visible
        await act(async () => {
            visibilitySpy.mockReturnValue('visible');
            document.dispatchEvent(new Event('visibilitychange'));
        });

        // Immediately syncs to current time
        expect(screen.getByText('10:00:05')).toBeDefined();

        // Subsequent ticks work normally
        await act(async () => {
            vi.advanceTimersByTime(1000);
        });

        expect(screen.getByText('10:00:06')).toBeDefined();
    });

    it('cleans up interval on unmount', () => {
        vi.useFakeTimers();
        const clearIntervalSpy = vi.spyOn(window, 'clearInterval');

        const { unmount } = render(<TerminalClock />);
        unmount();

        expect(clearIntervalSpy).toHaveBeenCalled();
    });

    it('supports custom intervalMs and applies custom className', async () => {
        vi.useFakeTimers();
        vi.setSystemTime(new Date('2026-11-03T10:00:00'));

        const { container } = render(<TerminalClock intervalMs={5000} className="custom-class" />);
        const timeEl = container.querySelector('time');
        expect(timeEl?.className).toBe('market-session-clock custom-class');

        await act(async () => {
            vi.advanceTimersByTime(1000);
        });
        // Has not ticked at 1000ms because interval is 5000ms
        expect(screen.getByText('10:00:00')).toBeDefined();

        await act(async () => {
            vi.advanceTimersByTime(4000);
        });
        expect(screen.getByText('10:00:05')).toBeDefined();
    });
});
