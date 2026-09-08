import { useEffect, useRef } from 'react';

/** Quietly replace stale data; preserve it on failure and pause hidden pages. */
export function useBackgroundRefresh(enabled: boolean, refresh: (signal: AbortSignal) => Promise<void>) {
    const latest = useRef(refresh);
    latest.current = refresh;
    useEffect(() => {
        if (!enabled) return;
        const controller = new AbortController();
        let delay = 2000;
        let timer: ReturnType<typeof setTimeout>;
        const tick = async () => {
            if (controller.signal.aborted) return;
            if (document.visibilityState !== 'hidden') {
                try { await latest.current(controller.signal); }
                catch { /* Keep the last usable response while the source recovers. */ }
                delay = Math.min(30_000, delay * 1.5);
            }
            if (!controller.signal.aborted) timer = setTimeout(tick, delay);
        };
        timer = setTimeout(tick, delay);
        return () => { controller.abort(); clearTimeout(timer); };
    }, [enabled]);
}
