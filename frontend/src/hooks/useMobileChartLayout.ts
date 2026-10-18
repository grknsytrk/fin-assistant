import { useEffect, useState } from 'react';

const MOBILE_CHART_QUERY = '(max-width: 640px)';

export function useMobileChartLayout(): boolean {
    const [isMobile, setIsMobile] = useState(() => (
        typeof window !== 'undefined'
        && typeof window.matchMedia === 'function'
        && window.matchMedia(MOBILE_CHART_QUERY).matches
    ));

    useEffect(() => {
        if (typeof window.matchMedia !== 'function') return undefined;
        const mediaQuery = window.matchMedia(MOBILE_CHART_QUERY);
        const update = () => setIsMobile(mediaQuery.matches);
        update();
        if (typeof mediaQuery.addEventListener === 'function') {
            mediaQuery.addEventListener('change', update);
            return () => mediaQuery.removeEventListener('change', update);
        }
        mediaQuery.addListener(update);
        return () => mediaQuery.removeListener(update);
    }, []);

    return isMobile;
}
