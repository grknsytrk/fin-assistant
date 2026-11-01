import { useCallback, useEffect, useRef, useState } from 'react';

export interface UseTableVirtualizerOptions {
    totalRows: number;
    rowHeight?: number;
    overscan?: number;
    containerRef?: React.RefObject<HTMLDivElement | null>;
    resetDeps?: unknown[];
}

export interface UseTableVirtualizerReturn {
    startIndex: number;
    endIndex: number;
    topSpacerHeight: number;
    bottomSpacerHeight: number;
    totalHeight: number;
    resetScroll: () => void;
    containerRef: (node: HTMLDivElement | null) => void;
}

interface VirtualRange {
    startIndex: number;
    endIndex: number;
}

export function useTableVirtualizer({
    totalRows,
    rowHeight = 38,
    overscan = 12,
    containerRef,
    resetDeps,
}: UseTableVirtualizerOptions): UseTableVirtualizerReturn {
    const [element, setElement] = useState<HTMLDivElement | null>(() => containerRef?.current ?? null);
    const viewportHeightRef = useRef<number>(600);
    const totalRowsRef = useRef<number>(totalRows);
    totalRowsRef.current = totalRows;
    const rafIdRef = useRef<number | null>(null);

    // Keep element in sync if containerRef is a standard RefObject that attached after mount
    useEffect(() => {
        if (containerRef && containerRef.current && containerRef.current !== element) {
            setElement(containerRef.current);
        }
    });

    const attachRef = useCallback((node: HTMLDivElement | null) => {
        if (containerRef) {
            (containerRef as React.MutableRefObject<HTMLDivElement | null>).current = node;
        }
        setElement(node);
    }, [containerRef]);

    const computeRange = useCallback(
        (sTop: number, vHeight: number, total: number): VirtualRange => {
            const safeTotal = Math.max(0, total);
            if (safeTotal === 0) {
                return { startIndex: 0, endIndex: 0 };
            }
            const startRow = Math.floor(sTop / rowHeight);
            const visibleRowCount = Math.ceil(vHeight / rowHeight);
            const startIndex = Math.max(0, startRow - overscan);
            const endIndex = Math.min(safeTotal, startRow + visibleRowCount + overscan);
            return { startIndex, endIndex };
        },
        [rowHeight, overscan]
    );

    const [range, setRange] = useState<VirtualRange>(() =>
        computeRange(0, 600, totalRows)
    );
    const rangeRef = useRef<VirtualRange>(range);
    rangeRef.current = range;

    const resetScroll = useCallback(() => {
        const el = element || containerRef?.current;
        if (el) {
            el.scrollTop = 0;
        }
        const nextRange = computeRange(0, viewportHeightRef.current, totalRowsRef.current);
        rangeRef.current = nextRange;
        setRange(nextRange);
    }, [element, containerRef, computeRange]);

    // Recalculate when totalRows or element changes
    useEffect(() => {
        const el = element || containerRef?.current;
        const sTop = el ? el.scrollTop : 0;
        const vHeight = el && el.clientHeight > 0 ? el.clientHeight : viewportHeightRef.current;
        if (el && el.clientHeight > 0) {
            viewportHeightRef.current = el.clientHeight;
        }
        const nextRange = computeRange(sTop, vHeight, totalRows);
        if (nextRange.startIndex !== rangeRef.current.startIndex || nextRange.endIndex !== rangeRef.current.endIndex) {
            rangeRef.current = nextRange;
            setRange(nextRange);
        }
    }, [totalRows, computeRange, element, containerRef]);

    // Reset scroll when requested dependencies change
    useEffect(() => {
        if (resetDeps && resetDeps.length > 0) {
            resetScroll();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, resetDeps ?? []);

    // Attach scroll and resize listeners to element
    useEffect(() => {
        const el = element || containerRef?.current;
        if (!el) return;

        if (el.clientHeight > 0) {
            viewportHeightRef.current = el.clientHeight;
        }

        const handleScroll = () => {
            if (rafIdRef.current !== null) return;
            rafIdRef.current = window.requestAnimationFrame(() => {
                rafIdRef.current = null;
                const nextRange = computeRange(el.scrollTop, viewportHeightRef.current, totalRowsRef.current);
                if (
                    nextRange.startIndex !== rangeRef.current.startIndex ||
                    nextRange.endIndex !== rangeRef.current.endIndex
                ) {
                    rangeRef.current = nextRange;
                    setRange(nextRange);
                }
            });
        };

        const updateHeight = () => {
            if (el.clientHeight > 0 && el.clientHeight !== viewportHeightRef.current) {
                viewportHeightRef.current = el.clientHeight;
                const nextRange = computeRange(el.scrollTop, el.clientHeight, totalRowsRef.current);
                if (
                    nextRange.startIndex !== rangeRef.current.startIndex ||
                    nextRange.endIndex !== rangeRef.current.endIndex
                ) {
                    rangeRef.current = nextRange;
                    setRange(nextRange);
                }
            }
        };

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
    }, [element, containerRef, computeRange]);

    const safeTotalRows = Math.max(0, totalRows);
    const totalHeight = safeTotalRows * rowHeight;
    const topSpacerHeight = range.startIndex * rowHeight;
    const bottomSpacerHeight = Math.max(0, (safeTotalRows - range.endIndex) * rowHeight);

    return {
        startIndex: range.startIndex,
        endIndex: range.endIndex,
        topSpacerHeight,
        bottomSpacerHeight,
        totalHeight,
        resetScroll,
        containerRef: attachRef,
    };
}
