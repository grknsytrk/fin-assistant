// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { useBackgroundRefresh } from './useBackgroundRefresh';

afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.useRealTimers();
});

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

    // Hidden tab: zero timers should fire
    await act(() => vi.advanceTimersByTimeAsync(6000));
    expect(refresh).not.toHaveBeenCalled();

    // Tab returns to visible state
    visibility.mockReturnValue('visible');
    await act(async () => {
        document.dispatchEvent(new Event('visibilitychange'));
    });

    // Immediate tick runs on tab focus (call 1 fails with 'temporary outage')
    expect(refresh).toHaveBeenCalledTimes(1);

    // Exponential backoff to 3000ms: at 2000ms call 2 hasn't fired yet
    await act(() => vi.advanceTimersByTimeAsync(2000));
    expect(refresh).toHaveBeenCalledTimes(1);

    // At 3000ms total backoff elapsed, call 2 fires
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(refresh).toHaveBeenCalledTimes(2);

    // Call 2 is still in-flight (waiting for finish), so no overlapping call 3 fires
    await act(() => vi.advanceTimersByTimeAsync(60_000));
    expect(refresh).toHaveBeenCalledTimes(2);

    unmount();
    await act(async () => finish());
});

it('pauses polling when tab transitions from visible to hidden', async () => {
    vi.useFakeTimers();
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible');
    const refresh = vi.fn().mockResolvedValue(undefined);

    renderHook(() => useBackgroundRefresh(true, refresh));

    // First refresh after baseDelay (2000ms)
    await act(() => vi.advanceTimersByTimeAsync(2000));
    expect(refresh).toHaveBeenCalledTimes(1);

    // Tab is hidden
    visibility.mockReturnValue('hidden');
    await act(async () => {
        document.dispatchEvent(new Event('visibilitychange'));
    });

    // Advance 60 seconds while hidden; no further calls should occur
    await act(() => vi.advanceTimersByTimeAsync(60_000));
    expect(refresh).toHaveBeenCalledTimes(1);

    // Tab becomes visible again; immediately triggers refresh
    visibility.mockReturnValue('visible');
    await act(async () => {
        document.dispatchEvent(new Event('visibilitychange'));
    });

    expect(refresh).toHaveBeenCalledTimes(2);
});
