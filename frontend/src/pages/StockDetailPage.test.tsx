import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen } from '@testing-library/react';
import StockDetailPage from './StockDetailPage';
const mocks = vi.hoisted(() => ({ snapshot: vi.fn() }));
vi.mock('../api/client', () => ({ cachedKapSnapshot: () => null, apiClient: {
    kapSnapshot: mocks.snapshot,
    kapPrice: () => Promise.resolve({ ok: false }),
    marketStockCards: () => Promise.resolve({ items: [] }),
    marketStockCardChart: () => Promise.resolve({ line_points: [], error: 'No chart' }),
} }));
vi.mock('../components/MarketsNavigation', () => ({ default: () => null }));
vi.mock('../components/SymbolLogo', () => ({ default: () => null }));
vi.mock('./stock/sections/StockOverview', () => ({ default: ({ quarters }: { quarters: unknown[] }) => <div data-testid="overview">{quarters.length} quarters</div> }));
vi.mock('./stock/sections/StockFinancials', () => ({ default: () => null }));
vi.mock('./stock/sections/StockKAP', () => ({ default: () => null }));
const props = { ticker: 'TEST', onBack: vi.fn(), onNavigateSection: vi.fn(), onOpenTicker: vi.fn(), onOpenFund: vi.fn() };
const summary = { ok: true, company: 'TEST', quarters: Array.from({ length: 5 }, (_, i) => ({ year: 2025 + Math.floor(i / 4), period: 1 + i % 4 })) };
beforeEach(() => { vi.useFakeTimers(); mocks.snapshot.mockReset(); });
afterEach(() => { cleanup(); vi.useRealTimers(); });
it('shows five quarters before starting the slow full-history request', async () => {
    mocks.snapshot.mockResolvedValueOnce(summary).mockImplementation(() => new Promise(() => {}));
    const view = render(<StockDetailPage {...props} />);
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(screen.getByTestId('overview').textContent).toBe('5 quarters');
    expect(mocks.snapshot.mock.calls[0][2]).toBe(5);
    await act(async () => { await vi.advanceTimersByTimeAsync(300); });
    expect(mocks.snapshot.mock.calls[1][2]).toBe(20);
    expect(screen.getByTestId('overview').textContent).toBe('5 quarters');
    const signal = mocks.snapshot.mock.calls[1][3] as AbortSignal;
    view.unmount();
    expect(signal.aborted).toBe(true);
});
it('shows neutral loading and automatically completes after a long pending job', async () => {
    let calls = 0;
    mocks.snapshot.mockImplementation(() => Promise.resolve(++calls <= 10
        ? { ok: false, pending: true, refresh_pending: true, quarters: [], error: 'Finansal veriler hazırlanıyor. Kısa süre sonra yeniden deneyebilirsiniz.' }
        : summary));
    const view = render(<StockDetailPage {...props} />);
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(screen.getByRole('status').textContent).toContain('Finansal veriler yükleniyor');
    expect(view.container.querySelector('.alert-error')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Yeniden dene' })).toBeNull();
    await act(async () => { await vi.advanceTimersByTimeAsync(70000); });
    expect(calls).toBeGreaterThan(10);
    expect(screen.getByTestId('overview').textContent).toBe('5 quarters');
    expect(screen.queryByText('Finansal veriler yükleniyor…')).toBeNull();
});
it('stops automatic polling when leaving the page', async () => {
    mocks.snapshot.mockResolvedValue({ ok: false, pending: true, quarters: [] });
    const view = render(<StockDetailPage {...props} />);
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    view.unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
    expect(mocks.snapshot).toHaveBeenCalledTimes(1);
});
