import React, { useEffect, useState } from 'react';

export function formatTerminalDate(date: Date): string {
    const dayMonth = date.toLocaleDateString('tr-TR', {
        day: 'numeric',
        month: 'long',
    });
    const weekday = date.toLocaleDateString('tr-TR', { weekday: 'long' });
    return `${dayMonth}, ${weekday.charAt(0).toUpperCase()}${weekday.slice(1)}`;
}

export function formatTerminalClock(date: Date): string {
    return date.toLocaleTimeString('tr-TR', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
}

export interface TerminalClockProps {
    intervalMs?: number;
    className?: string;
}

/**
 * Isolated leaf component for the live market terminal clock.
 * Updates internal clock state at `intervalMs` (default 1000ms) without
 * triggering re-renders in parent views or data tables.
 * Automatically halts timer ticks when tab is hidden or minimized.
 */
export const TerminalClock = React.memo(function TerminalClock({
    intervalMs = 1000,
    className,
}: TerminalClockProps) {
    const [now, setNow] = useState(() => new Date());

    useEffect(() => {
        let timer: ReturnType<typeof setInterval> | null = null;
        const tick = () => setNow(new Date());

        const start = () => {
            if (timer === null && typeof document !== 'undefined' && document.visibilityState === 'visible') {
                timer = window.setInterval(tick, intervalMs);
            }
        };

        const stop = () => {
            if (timer !== null) {
                window.clearInterval(timer);
                timer = null;
            }
        };

        const onVisibilityChange = () => {
            if (typeof document !== 'undefined' && document.visibilityState === 'visible') {
                tick();
                start();
            } else {
                stop();
            }
        };

        start();
        if (typeof document !== 'undefined') {
            document.addEventListener('visibilitychange', onVisibilityChange);
        }

        return () => {
            stop();
            if (typeof document !== 'undefined') {
                document.removeEventListener('visibilitychange', onVisibilityChange);
            }
        };
    }, [intervalMs]);

    const clockClass = className ? `market-session-clock ${className}` : 'market-session-clock';

    return (
        <>
            <span>{formatTerminalDate(now)}</span>
            <time className={clockClass}>{formatTerminalClock(now)}</time>
        </>
    );
});

export default TerminalClock;
