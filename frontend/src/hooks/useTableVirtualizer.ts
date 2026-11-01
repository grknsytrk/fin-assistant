import { useEffect, useRef, useState } from 'react';

export interface UseTableVirtualizerOptions {
    totalRows: number;
    rowHeight?: number;
    overscan?: number;
    containerRef: React.RefObject<HTMLDivElement | null>;
    resetDeps?: unknown[];
}

export interface UseTableVirtualizerReturn {
    startIndex: number;
    endIndex: number;
    topSpacerHeight: number;
    bottomSpacerHeight: number;
    totalHeight: number;
    resetScroll: () => void;
}

export function useTableVirtualizer({
    totalRows,
    rowHeight = 38,
    overscan = 12,
    containerRef,
    resetDeps,
}: UseTableVirtualizerOptions): UseTableVirtualizerReturn {
    const [scrollTop, setScrollTop] = useState(0);
    const [viewportHeight, setViewportHeight] = useState(600);
    const rafIdRef = useRef<number | null>(null);

    const resetScroll = () => {
        if (containerRef.current) {
            containerRef.current.scrollTop = 0;
        }
        setScrollTop(0);
    };

    // Reset scroll when requested dependencies change
    const resetDepsRef = useRef(resetDeps);
    useEffect(() => {
        if (resetDeps && resetDeps.length > 0) {
            resetScroll();
        }
        resetDepsRef.current = resetDeps;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, resetDeps ?? []);

    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;

        const handleScroll = () => {
            if (rafIdRef.current !== null) return;
            rafIdRef.current = window.requestAnimationFrame(() => {
                const currentScrollTop = el.scrollTop;
                // Only trigger React state update if vertical scroll position actually changed.
                // This keeps horizontal-only scrolling 100% on the GPU with zero JS re-renders!
                setScrollTop((prev) => (prev !== currentScrollTop ? currentScrollTop : prev));
                rafIdRef.current = null;
            });
        };

        const updateHeight = () => {
            if (el.clientHeight > 0 && el.clientHeight !== viewportHeight) {
                setViewportHeight(el.clientHeight);
            }
        };

        updateHeight();
        el.addEventListener('scroll', handleScroll, { passive: true });

        let ro: ResizeObserver | null = null;
        if (typeof ResizeObserver !== 'undefined') {
            ro = new ResizeObserver(() => {
                updateHeight();
            });
            ro.observe(el);
        }

        return () => {
            if (rafIdRef.current !== null) {
                window.cancelAnimationFrame(rafIdRef.current);
                rafIdRef.current = null;
            }
            el.removeEventListener('scroll', handleScroll);
            if (ro) ro.disconnect();
        };
    }, [containerRef, viewportHeight]);

    const safeTotalRows = Math.max(0, totalRows);
    const totalHeight = safeTotalRows * rowHeight;

    const startRow = Math.floor(scrollTop / rowHeight);
    const visibleRowCount = Math.ceil(viewportHeight / rowHeight);

    const startIndex = Math.max(0, startRow - overscan);
    const endIndex = Math.min(safeTotalRows, startRow + visibleRowCount + overscan);

    const topSpacerHeight = startIndex * rowHeight;
    const bottomSpacerHeight = Math.max(0, (safeTotalRows - endIndex) * rowHeight);

    return {
        startIndex,
        endIndex,
        topSpacerHeight,
        bottomSpacerHeight,
        totalHeight,
        resetScroll,
    };
}
