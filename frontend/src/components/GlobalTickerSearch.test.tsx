import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { FundSummary, MarketUniverseRow } from '../api/types';
import GlobalTickerSearch from './GlobalTickerSearch';

const mocks = vi.hoisted(() => ({
  marketStockSearch: vi.fn(),
  fundSearch: vi.fn(),
}));

vi.mock('../api/client', () => ({
  apiClient: {
    marketStockSearch: mocks.marketStockSearch,
    fundSearch: mocks.fundSearch,
  },
}));

vi.mock('./SymbolLogo', () => ({ default: () => null }));

function fundRow(fundCode = 'THF'): FundSummary {
  return {
    fund_code: fundCode,
    name: `${fundCode} Test Fonu`,
    fund_type: 'SEC',
    founder_company: 'Test Portföy',
    manager_company: 'Test Portföy',
    price: 1.2345,
    daily_return: 1.2,
    period_returns: {},
    risk_value: null,
    currency: 'TRY',
    as_of: '2026-09-12',
    source: 'test',
  };
}

function stockRow(symbol: string): MarketUniverseRow {
  return {
    symbol,
    company: symbol,
    name: `${symbol} Test Hissesi`,
    latest_quarter: null,
    has_kap_cache: false,
    price: 100,
    price_currency: 'TRY',
    change: 1,
    change_pct: 1,
    price_as_of: null,
    market_cap: null,
  };
}

function renderSearch() {
  const onSelectTicker = vi.fn();
  const onSelectFund = vi.fn();
  const view = render(
    <GlobalTickerSearch onSelectTicker={onSelectTicker} onSelectFund={onSelectFund} />,
  );
  fireEvent.click(screen.getByRole('button', { name: 'Ara' }));
  return {
    view,
    input: screen.getByRole('textbox', { name: 'Hisse veya fon ara' }) as HTMLInputElement,
    onSelectTicker,
    onSelectFund,
  };
}

async function flushPromises() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

beforeEach(() => {
  window.localStorage.clear();
  mocks.marketStockSearch.mockReset();
  mocks.fundSearch.mockReset();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe('GlobalTickerSearch', () => {
  it('opens a typed three-letter fund code before the debounce finishes', () => {
    vi.useFakeTimers();

    const { input, onSelectTicker, onSelectFund } = renderSearch();
    fireEvent.change(input, { target: { value: 'THF' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    expect(mocks.fundSearch).not.toHaveBeenCalled();
    expect(mocks.marketStockSearch).not.toHaveBeenCalled();
    expect(onSelectFund).toHaveBeenCalledWith('THF');
    expect(onSelectTicker).not.toHaveBeenCalled();
  });

  it('waits for the fund when a faster stock result is already available', async () => {
    vi.useFakeTimers();
    let resolveFund!: (payload: { rows: FundSummary[] }) => void;
    const pendingFund = new Promise<{ rows: FundSummary[] }>((resolve) => {
      resolveFund = resolve;
    });
    mocks.fundSearch
      .mockReturnValueOnce(pendingFund)
      .mockResolvedValue({ rows: [] });
    mocks.marketStockSearch.mockResolvedValue({ rows: [stockRow('THF')] });

    const { view, input, onSelectTicker, onSelectFund } = renderSearch();
    fireEvent.change(input, { target: { value: 'THFX' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    fireEvent.keyDown(input, { key: 'Enter' });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    expect(mocks.marketStockSearch).toHaveBeenCalledTimes(2);
    expect(onSelectTicker).not.toHaveBeenCalled();
    expect(onSelectFund).not.toHaveBeenCalled();

    resolveFund({ rows: [fundRow('THFX')] });
    await flushPromises();

    expect(onSelectFund).toHaveBeenCalledWith('THFX');
    expect(onSelectFund).toHaveBeenCalledTimes(1);
    expect(onSelectTicker).not.toHaveBeenCalled();
    view.unmount();
  });

  it('opens a verified stock for a non-fund-code query', async () => {
    mocks.marketStockSearch.mockResolvedValue({ rows: [stockRow('BIMAS')] });
    mocks.fundSearch.mockResolvedValue({ rows: [] });

    const { input, onSelectTicker, onSelectFund } = renderSearch();
    fireEvent.change(input, { target: { value: 'BIMAS' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => expect(onSelectTicker).toHaveBeenCalledWith('BIMAS'));
    expect(onSelectFund).not.toHaveBeenCalled();
  });

  it('keeps the modal open without navigating when no result is found', async () => {
    mocks.marketStockSearch.mockResolvedValue({ rows: [] });
    mocks.fundSearch.mockResolvedValue({ rows: [] });

    const { input, onSelectTicker, onSelectFund } = renderSearch();
    fireEvent.change(input, { target: { value: 'UNKNOWN' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => expect(screen.getByText('Sonuç bulunamadı.')).toBeTruthy());
    expect(screen.getByRole('dialog')).toBeTruthy();
    expect(onSelectTicker).not.toHaveBeenCalled();
    expect(onSelectFund).not.toHaveBeenCalled();
  });

  it('keeps the modal open without navigating when search fails', async () => {
    mocks.marketStockSearch.mockRejectedValue(new Error('stock unavailable'));
    mocks.fundSearch.mockRejectedValue(new Error('fund unavailable'));

    const { input, onSelectTicker, onSelectFund } = renderSearch();
    fireEvent.change(input, { target: { value: 'UNKNOWN' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => expect(screen.getByText(/Arama yapılamadı/)).toBeTruthy());
    expect(screen.getByRole('dialog')).toBeTruthy();
    expect(onSelectTicker).not.toHaveBeenCalled();
    expect(onSelectFund).not.toHaveBeenCalled();
  });

  it('routes clicked funds and arrow-selected stocks according to their type', async () => {
    mocks.marketStockSearch.mockImplementation(({ q }: { q: string }) => Promise.resolve({
      rows: q === 'B' ? [stockRow('BIMAS'), stockRow('BRISA')] : [],
    }));
    mocks.fundSearch.mockImplementation((q: string) => Promise.resolve({
      rows: q === 'THF' ? [fundRow()] : [],
    }));

    const { input, onSelectTicker, onSelectFund } = renderSearch();
    fireEvent.change(input, { target: { value: 'THF' } });
    const fundOption = await screen.findByRole('option', { name: /THF/ });
    fireEvent.click(fundOption);
    expect(onSelectFund).toHaveBeenCalledWith('THF');

    fireEvent.click(screen.getByRole('button', { name: 'Ara' }));
    const reopenedInput = screen.getByRole('textbox', { name: 'Hisse veya fon ara' });
    fireEvent.change(reopenedInput, { target: { value: 'B' } });
    await screen.findAllByRole('option');
    fireEvent.keyDown(reopenedInput, { key: 'ArrowDown' });
    fireEvent.keyDown(reopenedInput, { key: 'Enter' });

    await waitFor(() => expect(onSelectTicker).toHaveBeenCalledWith('BRISA'));
    expect(onSelectFund).toHaveBeenCalledTimes(1);
  });

  it('does not navigate when a response arrives after unmount', async () => {
    let resolveFund!: (payload: { rows: FundSummary[] }) => void;
    const pendingFund = new Promise<{ rows: FundSummary[] }>((resolve) => {
      resolveFund = resolve;
    });
    mocks.fundSearch.mockReturnValue(pendingFund);
    mocks.marketStockSearch.mockResolvedValue({ rows: [] });

    const { view, input, onSelectTicker, onSelectFund } = renderSearch();
    fireEvent.change(input, { target: { value: 'THFX' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    view.unmount();

    resolveFund({ rows: [fundRow()] });
    await flushPromises();

    expect(onSelectTicker).not.toHaveBeenCalled();
    expect(onSelectFund).not.toHaveBeenCalled();
  });
});
