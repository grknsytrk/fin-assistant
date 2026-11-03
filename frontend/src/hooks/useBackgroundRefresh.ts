import { useEffect, useRef } from 'react';

/**
 * Quietly refresh background queries with exponential backoff on failure.
 * Halts timer loops when tab is hidden or minimized to prevent CPU thrashing.
 * Immediately resumes polling when tab returns to visible state.
 */
export function useBackgroundRefresh(
    enabled: boolean,
    refresh: (signal: AbortSignal) => Promise<unknown>,
    baseDelay = 2000,
): void {
    const latest = useRef(refresh);
    latest.current = refresh;

    useEffect(() => {
        if (!enabled) return;
        const controller = new AbortController();
        let delay = baseDelay;
        let timer: ReturnType<typeof setTimeout> | null = null;
        let inFlight = false;

        const clear = () => {
            if (timer !== null) {
                clearTimeout(timer);
                timer = null;
            }
        };

        const tick = async () => {
            if (
                controller.signal.aborted ||
                typeof document === 'undefined' ||
                document.visibilityState !== 'visible'
            ) {
                return;
            }
            if (inFlight) return;
            inFlight = true;

            try {
                await latest.current(controller.signal);
            } catch {
                /* Keep the last usable response while the source recovers. */
            } finally {
                inFlight = false;
            }

            delay = Math.min(30_000, delay * 1.5);

            if (
                !controller.signal.aborted &&
                typeof document !== 'undefined' &&
                document.visibilityState === 'visible'
            ) {
                clear();
                timer = setTimeout(tick, delay);
            }
        };

        const onVisibilityChange = () => {
            if (typeof document !== 'undefined' && document.visibilityState === 'visible') {
                clear();
                delay = baseDelay;
                void tick();
            } else {
                clear();
            }
        };

        if (typeof document !== 'undefined') {
            document.addEventListener('visibilitychange', onVisibilityChange);
            if (document.visibilityState === 'visible') {
                timer = setTimeout(tick, delay);
            }
        } else {
            timer = setTimeout(tick, delay);
        }

        return () => {
            controller.abort();
            clear();
            if (typeof document !== 'undefined') {
                document.removeEventListener('visibilitychange', onVisibilityChange);
            }
        };
    }, [enabled, baseDelay]);
}
