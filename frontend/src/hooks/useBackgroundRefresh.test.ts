import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { useBackgroundRefresh } from './useBackgroundRefresh';

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.useRealTimers(); });

it('stops polling and aborts the active consumer when fresh data arrives', async () => {
    vi.useFakeTimers();
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible');
    const refresh = vi.fn().mockResolvedValue(undefined);
    const { rerender } = renderHook(({ enabled }) => useBackgroundRefresh(enabled, refresh), {
        initialProps: { enabled: true },
    });
    await act(() => vi.advanceTimersByTimeAsync(2000));
    expect(refresh).toHaveBeenCalledTimes(1);
    const signal = refresh.mock.calls[0][0] as AbortSignal;
    rerender({ enabled: false });
    expect(signal.aborted).toBe(true);
    await act(() => vi.advanceTimersByTimeAsync(60_000));
    expect(refresh).toHaveBeenCalledTimes(1);
});

it('pauses hidden pages and retries a failed refresh without overlapping requests', async () => {
    vi.useFakeTimers();
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden');
    let finish: () => void = () => {};
    const refresh = vi.fn()
        .mockRejectedValueOnce(new Error('temporary outage'))
        .mockImplementation(() => new Promise<void>((resolve) => { finish = resolve; }));
    const { unmount } = renderHook(() => useBackgroundRefresh(true, refresh));
    await act(() => vi.advanceTimersByTimeAsync(6000));
    expect(refresh).not.toHaveBeenCalled();
    visibility.mockReturnValue('visible');
    await act(() => vi.advanceTimersByTimeAsync(2000));
    expect(refresh).toHaveBeenCalledTimes(1);
    await act(() => vi.advanceTimersByTimeAsync(3000));
    expect(refresh).toHaveBeenCalledTimes(2);
    await act(() => vi.advanceTimersByTimeAsync(60_000));
    expect(refresh).toHaveBeenCalledTimes(2);
    unmount();
    await act(async () => finish());
});
