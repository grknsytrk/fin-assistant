import { renderHook, act } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { useTableVirtualizer } from './useTableVirtualizer';

describe('useTableVirtualizer', () => {
    it('returns 0 startIndex and 0 endIndex when totalRows is 0', () => {
        const containerRef = { current: document.createElement('div') };
        const { result } = renderHook(() =>
            useTableVirtualizer({
                totalRows: 0,
                rowHeight: 38,
                overscan: 10,
                containerRef,
            })
        );

        expect(result.current.startIndex).toBe(0);
        expect(result.current.endIndex).toBe(0);
        expect(result.current.topSpacerHeight).toBe(0);
        expect(result.current.bottomSpacerHeight).toBe(0);
        expect(result.current.totalHeight).toBe(0);
    });

    it('calculates correct initial slice and spacer heights for 580 rows', () => {
        const div = document.createElement('div');
        Object.defineProperty(div, 'clientHeight', { value: 760, configurable: true });
        Object.defineProperty(div, 'scrollTop', { value: 0, writable: true, configurable: true });
        const containerRef = { current: div };

        const { result } = renderHook(() =>
            useTableVirtualizer({
                totalRows: 580,
                rowHeight: 38,
                overscan: 10,
                containerRef,
            })
        );

        // At scrollTop 0:
        // startRow = 0
        // visibleRowCount = ceil(600 / 38) = 16 (default viewportHeight 600 before resize)
        // startIndex = 0
        // endIndex = 0 + 16 + 10 = 26
        // topSpacerHeight = 0
        // bottomSpacerHeight = (580 - 26) * 38 = 554 * 38 = 21052
        // totalHeight = 580 * 38 = 22040
        expect(result.current.startIndex).toBe(0);
        expect(result.current.endIndex).toBeGreaterThan(15);
        expect(result.current.topSpacerHeight).toBe(0);
        expect(result.current.topSpacerHeight + (result.current.endIndex - result.current.startIndex) * 38 + result.current.bottomSpacerHeight).toBe(22040);
        expect(result.current.totalHeight).toBe(22040);
    });

    it('updates range and spacers upon scroll', async () => {
        const div = document.createElement('div');
        Object.defineProperty(div, 'clientHeight', { value: 600, configurable: true });
        let currentScrollTop = 0;
        Object.defineProperty(div, 'scrollTop', {
            get: () => currentScrollTop,
            set: (val: number) => { currentScrollTop = val; },
            configurable: true,
        });
        const containerRef = { current: div };

        const { result } = renderHook(() =>
            useTableVirtualizer({
                totalRows: 580,
                rowHeight: 38,
                overscan: 10,
                containerRef,
            })
        );

        // Simulate scrolling down to row 100 (100 * 38 = 3800px)
        await act(async () => {
            currentScrollTop = 3800;
            div.dispatchEvent(new Event('scroll'));
            // Wait for RAF
            await new Promise((resolve) => requestAnimationFrame(resolve));
        });

        // startRow = 100
        // startIndex = 100 - 10 = 90
        // endIndex = 100 + 16 + 10 = 126
        expect(result.current.startIndex).toBe(90);
        expect(result.current.endIndex).toBe(126);
        expect(result.current.topSpacerHeight).toBe(90 * 38);
        expect(result.current.bottomSpacerHeight).toBe((580 - 126) * 38);
        expect(result.current.topSpacerHeight + (result.current.endIndex - result.current.startIndex) * 38 + result.current.bottomSpacerHeight).toBe(22040);
    });

    it('resets scroll on resetDeps change', async () => {
        const div = document.createElement('div');
        let currentScrollTop = 5000;
        Object.defineProperty(div, 'scrollTop', {
            get: () => currentScrollTop,
            set: (val: number) => { currentScrollTop = val; },
            configurable: true,
        });
        const containerRef = { current: div };

        let filter = 'all';
        const { result, rerender } = renderHook(() =>
            useTableVirtualizer({
                totalRows: 580,
                rowHeight: 38,
                overscan: 10,
                containerRef,
                resetDeps: [filter],
            })
        );

        expect(div.scrollTop).toBe(0);

        // Change filter
        filter = 'tech';
        currentScrollTop = 1000;
        rerender();

        expect(div.scrollTop).toBe(0);
        expect(result.current.startIndex).toBe(0);
    });
});
