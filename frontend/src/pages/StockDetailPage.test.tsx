import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import StockDetailPage from './StockDetailPage';

const mocks = vi.hoisted(() => ({
    snapshot: vi.fn(),
    marketStockCards: vi.fn(),
}));

vi.mock('../api/client', () => ({
    cachedKapSnapshot: () => null,
    apiClient: {
        kapSnapshot: mocks.snapshot,
        marketStockCards: mocks.marketStockCards,
        marketStockCardChart: () => Promise.resolve({ line_points: [], error: 'No chart' }),
    },
}));

vi.mock('../components/MarketsNavigation', () => ({ default: () => null }));
vi.mock('../components/SymbolLogo', () => ({ default: () => null }));
vi.mock('./stock/sections/StockOverview', () => ({
    default: ({ quarters }: { quarters: unknown[] }) => <div data-testid="overview">{quarters.length} quarters</div>,
}));
vi.mock('./stock/sections/StockFinancials', () => ({ default: () => null }));
vi.mock('./stock/sections/StockKAP', () => ({ default: () => null }));

const props = {
    ticker: 'TEST',
    onBack: vi.fn(),
    onNavigateSection: vi.fn(),
    onOpenTicker: vi.fn(),
    onOpenFund: vi.fn(),
};

const summary = {
    ok: true,
    company: 'TEST',
    quarters: Array.from({ length: 5 }, (_, i) => ({
        year: 2025 + Math.floor(i / 4),
        period: 1 + (i % 4),
    })),
};

const renderComponent = (customProps = {}) =>
    render(
        <MemoryRouter>
            <StockDetailPage {...props} {...customProps} />
        </MemoryRouter>
    );

beforeEach(() => {
    vi.useFakeTimers();
    mocks.snapshot.mockReset();
    mocks.marketStockCards.mockReset();
    mocks.marketStockCards.mockResolvedValue({ items: [] });
});

afterEach(() => {
    cleanup();
    vi.useRealTimers();
});

it('shows five quarters before starting the slow full-history request', async () => {
    mocks.snapshot.mockResolvedValueOnce(summary).mockImplementation(() => new Promise(() => {}));
    const view = renderComponent();
    await act(async () => {
        await vi.advanceTimersByTimeAsync(1);
    });
    expect(screen.getByTestId('overview').textContent).toBe('5 quarters');
    expect(mocks.snapshot.mock.calls[0][2]).toBe(5);
    await act(async () => {
        await vi.advanceTimersByTimeAsync(300);
    });
    expect(mocks.snapshot.mock.calls[1][2]).toBe(20);
    expect(screen.getByTestId('overview').textContent).toBe('5 quarters');
    const signal = mocks.snapshot.mock.calls[1][3] as AbortSignal;
    view.unmount();
    expect(signal.aborted).toBe(true);
});

it('shows neutral loading and automatically completes after a long pending job', async () => {
    let calls = 0;
    mocks.snapshot.mockImplementation(() =>
        Promise.resolve(
            ++calls <= 10
                ? {
                      ok: false,
                      pending: true,
                      refresh_pending: true,
                      quarters: [],
                      error: 'Finansal veriler hazırlanıyor. Kısa süre sonra yeniden deneyebilirsiniz.',
                  }
                : summary
        )
    );
    const view = renderComponent();
    await act(async () => {
        await vi.advanceTimersByTimeAsync(1);
    });
    expect(screen.getByRole('status').textContent).toContain('Finansal veriler yükleniyor');
    expect(view.container.querySelector('.alert-error')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Yeniden dene' })).toBeNull();
    await act(async () => {
        await vi.advanceTimersByTimeAsync(70000);
    });
    expect(calls).toBeGreaterThan(10);
    expect(screen.getByTestId('overview').textContent).toBe('5 quarters');
    expect(screen.queryByText('Finansal veriler yükleniyor…')).toBeNull();
});

it('stops automatic polling when leaving the page', async () => {
    mocks.snapshot.mockResolvedValue({ ok: false, pending: true, quarters: [] });
    const view = renderComponent();
    await act(async () => {
        await vi.advanceTimersByTimeAsync(1);
    });
    view.unmount();
    await act(async () => {
        await vi.advanceTimersByTimeAsync(60000);
    });
    expect(mocks.snapshot).toHaveBeenCalledTimes(1);
});

it('displays live hero price, change %, and multiples from marketStockCards', async () => {
    mocks.snapshot.mockResolvedValue(summary);
    mocks.marketStockCards.mockResolvedValue({
        items: [
            {
                symbol: 'TEST',
                price: 312.5,
                change: 7.25,
                change_pct: 2.37,
                currency: 'TRY',
                as_of: '2026-11-07T15:30:00Z',
                market_cap: 15600000000,
                fk: 5.42,
                pd_dd: 1.88,
            },
        ],
    });

    const view = renderComponent();
    await act(async () => {
        await vi.advanceTimersByTimeAsync(1);
    });

    // Verify hero price is rendered from marketCard
    expect(screen.getByText('₺312,50')).toBeDefined();
    // Verify change percentage is rendered from marketCard
    expect(screen.getByText(/%\s*\+2,37/)).toBeDefined();
    // Verify F/K and PD/DD multiples are rendered from marketCard
    expect(screen.getByText('5,42x')).toBeDefined();
    expect(screen.getByText('1,88x')).toBeDefined();
    // Verify marketStockCards was queried with normalized ticker
    expect(mocks.marketStockCards).toHaveBeenCalledWith(
        expect.objectContaining({ symbols: ['TEST'] })
    );

    view.unmount();
});

it('falls back to snapshot valuation when market card is unavailable', async () => {
    const summaryWithValuation = {
        ...summary,
        valuation: {
            price: 250.0,
            price_currency: 'TRY',
            price_as_of: '2026-11-06T18:00:00Z',
            market_cap: 10000000000,
            fk: 4.5,
            pd_dd: 1.2,
        },
    };
    mocks.snapshot.mockResolvedValue(summaryWithValuation);
    mocks.marketStockCards.mockResolvedValue({ items: [] });

    const view = renderComponent();
    await act(async () => {
        await vi.advanceTimersByTimeAsync(1);
    });

    expect(screen.getByText('₺250,00')).toBeDefined();
    expect(screen.getByText('4,50x')).toBeDefined();
    expect(screen.getByText('1,20x')).toBeDefined();

    view.unmount();
});
