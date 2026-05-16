import { useEffect, useMemo, useState } from 'react';
import { Sun, Moon } from 'lucide-react';
import { Navigate, Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom';
import MarketPage from './pages/MarketPage';
import MarketsView from './pages/MarketsView';
import StockDetailPage from './pages/StockDetailPage';
import FundsPage from './pages/FundsPage';
import GlobalTickerSearch from './components/GlobalTickerSearch';
import type { MarketsNavigationSection } from './components/MarketsNavigation';
import {
  DEFAULT_FUND_TAB,
  DEFAULT_MARKET_INDEX,
  DEFAULT_STOCK_RETURN_MODE,
  DEFAULT_STOCK_TAB,
  ROUTE_PATHS,
  canonicalizeStocksReturnModeSearch,
  legacySearchToCanonical,
  normalizeMarketIndexCode,
  normalizeFundCode,
  normalizeFundTab,
  normalizeStockReturnMode,
  normalizeStockTab,
  normalizeTicker,
  toFundDetail,
  toFunds,
  toMarketsIndexDetail,
  toMarketsIndices,
  toMarketsOverview,
  toMarketsStocks,
  toStocksReturnModeSearch,
  toStockDetail,
} from './routing/routes';
import './index.css';

function LandingRoute() {
  const location = useLocation();
  const navigate = useNavigate();
  const legacyTarget = legacySearchToCanonical(location.search);

  if (legacyTarget) {
    return <Navigate to={legacyTarget} replace />;
  }

  return (
    <MarketPage
      onOpenMarkets={() => navigate(toMarketsOverview())}
      onOpenTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
    />
  );
}

function MarketsOverviewRoute() {
  const navigate = useNavigate();

  const handleSectionNavigate = (section: MarketsNavigationSection) => {
    if (section === 'markets') {
      navigate(toMarketsOverview());
      return;
    }
    if (section === 'funds') {
      navigate(toFunds());
      return;
    }
    if (section === 'indices') {
      navigate(toMarketsIndices());
      return;
    }
    navigate(toMarketsStocks(DEFAULT_MARKET_INDEX));
  };

  return (
    <MarketsView
      routeSection="markets"
      routeStockIndex={DEFAULT_MARKET_INDEX}
      routeSelectedIndex={null}
      onNavigateSection={handleSectionNavigate}
      onNavigateStockIndex={(nextIndex) => navigate(toMarketsStocks(nextIndex))}
      onNavigateIndexDetail={(nextIndex) => {
        if (!nextIndex) {
          navigate(toMarketsIndices());
          return;
        }
        navigate(toMarketsIndexDetail(nextIndex));
      }}
      onOpenTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
      onOpenFund={(fundCode) => navigate(toFundDetail(fundCode, DEFAULT_FUND_TAB))}
    />
  );
}

function MarketsStocksRoute() {
  const navigate = useNavigate();
  const location = useLocation();
  const params = useParams<{ indexCode: string }>();
  const rawIndexCode = String(params.indexCode || '');
  const normalizedIndexCode = normalizeMarketIndexCode(rawIndexCode);
  const { mode: returnMode, canonicalSearch } = canonicalizeStocksReturnModeSearch(location.search);
  const currentSearch = location.search.startsWith('?') ? location.search.slice(1) : location.search;
  const needsCanonicalIndex = !rawIndexCode || rawIndexCode.trim().toUpperCase() !== normalizedIndexCode;
  const needsCanonicalSearch = canonicalSearch !== currentSearch;

  const stocksPathWithMode = (
    indexCode: string | null | undefined,
    nextMode: string | null | undefined = returnMode,
  ) => `${toMarketsStocks(indexCode)}${toStocksReturnModeSearch(nextMode)}`;

  if (needsCanonicalIndex || needsCanonicalSearch) {
    const canonicalPath = toMarketsStocks(normalizedIndexCode);
    const canonicalTarget = canonicalSearch ? `${canonicalPath}?${canonicalSearch}` : canonicalPath;
    return <Navigate to={canonicalTarget} replace />;
  }

  const handleSectionNavigate = (section: MarketsNavigationSection) => {
    if (section === 'markets') {
      navigate(toMarketsOverview());
      return;
    }
    if (section === 'funds') {
      navigate(toFunds());
      return;
    }
    if (section === 'indices') {
      navigate(toMarketsIndices());
      return;
    }
    navigate(stocksPathWithMode(normalizedIndexCode));
  };

  return (
    <MarketsView
      routeSection="stocks"
      routeStockIndex={normalizedIndexCode}
      routeReturnMode={returnMode}
      routeSelectedIndex={null}
      onNavigateSection={handleSectionNavigate}
      onNavigateStockIndex={(nextIndex) => navigate(stocksPathWithMode(nextIndex))}
      onNavigateReturnMode={(nextMode) => {
        const normalizedMode = normalizeStockReturnMode(nextMode, DEFAULT_STOCK_RETURN_MODE);
        navigate(stocksPathWithMode(normalizedIndexCode, normalizedMode));
      }}
      onNavigateIndexDetail={(nextIndex) => {
        if (!nextIndex) {
          navigate(toMarketsIndices());
          return;
        }
        navigate(toMarketsIndexDetail(nextIndex));
      }}
      onOpenTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
      onOpenFund={(fundCode) => navigate(toFundDetail(fundCode, DEFAULT_FUND_TAB))}
    />
  );
}

function MarketsIndicesListRoute() {
  const navigate = useNavigate();

  return (
    <MarketsView
      routeSection="indices"
      routeStockIndex={DEFAULT_MARKET_INDEX}
      routeSelectedIndex={null}
      onNavigateSection={(section) => {
        if (section === 'markets') {
          navigate(toMarketsOverview());
          return;
        }
        if (section === 'funds') {
          navigate(toFunds());
          return;
        }
        if (section === 'indices') {
          navigate(toMarketsIndices());
          return;
        }
        navigate(toMarketsStocks(DEFAULT_MARKET_INDEX));
      }}
      onNavigateStockIndex={(nextIndex) => navigate(toMarketsStocks(nextIndex))}
      onNavigateIndexDetail={(nextIndex) => {
        if (!nextIndex) {
          navigate(toMarketsIndices());
          return;
        }
        navigate(toMarketsIndexDetail(nextIndex));
      }}
      onOpenTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
      onOpenFund={(fundCode) => navigate(toFundDetail(fundCode, DEFAULT_FUND_TAB))}
    />
  );
}

function MarketsIndicesDetailRoute() {
  const navigate = useNavigate();
  const params = useParams<{ indexCode: string }>();
  const rawIndexCode = String(params.indexCode || '');
  const normalizedIndexCode = normalizeMarketIndexCode(rawIndexCode);

  if (!rawIndexCode || rawIndexCode.trim().toUpperCase() !== normalizedIndexCode) {
    return <Navigate to={toMarketsIndexDetail(normalizedIndexCode)} replace />;
  }

  return (
    <MarketsView
      routeSection="indices"
      routeStockIndex={DEFAULT_MARKET_INDEX}
      routeSelectedIndex={normalizedIndexCode}
      onNavigateSection={(section) => {
        if (section === 'markets') {
          navigate(toMarketsOverview());
          return;
        }
        if (section === 'funds') {
          navigate(toFunds());
          return;
        }
        if (section === 'indices') {
          navigate(toMarketsIndices());
          return;
        }
        navigate(toMarketsStocks(DEFAULT_MARKET_INDEX));
      }}
      onNavigateStockIndex={(nextIndex) => navigate(toMarketsStocks(nextIndex))}
      onNavigateIndexDetail={(nextIndex) => {
        if (!nextIndex) {
          navigate(toMarketsIndices());
          return;
        }
        navigate(toMarketsIndexDetail(nextIndex));
      }}
      onOpenTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
      onOpenFund={(fundCode) => navigate(toFundDetail(fundCode, DEFAULT_FUND_TAB))}
    />
  );
}

function StockTickerDefaultTabRedirect() {
  const params = useParams<{ ticker: string }>();
  return <Navigate to={toStockDetail(params.ticker, DEFAULT_STOCK_TAB)} replace />;
}

function StockDetailRoute() {
  const navigate = useNavigate();
  const params = useParams<{ ticker: string; tab: string }>();

  const rawTicker = String(params.ticker || '');
  const rawTab = String(params.tab || '');
  const normalizedTicker = normalizeTicker(rawTicker);
  const normalizedTab = normalizeStockTab(rawTab);

  if (!normalizedTicker) {
    return <Navigate to={toMarketsStocks(DEFAULT_MARKET_INDEX)} replace />;
  }

  const needsCanonicalTicker = rawTicker.trim().toUpperCase() !== normalizedTicker;
  const normalizedRawTab = normalizeStockTab(rawTab, DEFAULT_STOCK_TAB);
  const needsCanonicalTab = rawTab.trim().toLowerCase() !== normalizedRawTab;

  if (needsCanonicalTicker || needsCanonicalTab) {
    return <Navigate to={toStockDetail(normalizedTicker, normalizedTab)} replace />;
  }

  const handleSectionNavigate = (section: MarketsNavigationSection) => {
    if (section === 'markets') {
      navigate(toMarketsOverview());
      return;
    }
    if (section === 'funds') {
      navigate(toFunds());
      return;
    }
    if (section === 'indices') {
      navigate(toMarketsIndices());
      return;
    }
    navigate(toMarketsStocks(DEFAULT_MARKET_INDEX));
  };

  return (
    <StockDetailPage
      key={normalizedTicker}
      ticker={normalizedTicker}
      activeTab={normalizedTab}
      onTabChange={(nextTab) => navigate(toStockDetail(normalizedTicker, nextTab))}
      onBack={() => navigate(toMarketsStocks(DEFAULT_MARKET_INDEX))}
      onNavigateSection={handleSectionNavigate}
      onOpenTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
      onOpenFund={(fundCode) => navigate(toFundDetail(fundCode, DEFAULT_FUND_TAB))}
    />
  );
}

function FundsListRoute() {
  const navigate = useNavigate();

  const handleSectionNavigate = (section: MarketsNavigationSection) => {
    if (section === 'markets') {
      navigate(toMarketsOverview());
      return;
    }
    if (section === 'stocks') {
      navigate(toMarketsStocks(DEFAULT_MARKET_INDEX));
      return;
    }
    if (section === 'indices') {
      navigate(toMarketsIndices());
      return;
    }
    navigate(toFunds());
  };

  return (
    <FundsPage
      onOpenFund={(nextFundCode, nextTab = DEFAULT_FUND_TAB) => navigate(toFundDetail(nextFundCode, nextTab))}
      onTabChange={() => undefined}
      onBack={() => navigate(toFunds())}
      onNavigateSection={handleSectionNavigate}
      onOpenTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
    />
  );
}

function FundCodeDefaultTabRedirect() {
  const params = useParams<{ fundCode: string }>();
  return <Navigate to={toFundDetail(params.fundCode, DEFAULT_FUND_TAB)} replace />;
}

function FundDetailRoute() {
  const navigate = useNavigate();
  const params = useParams<{ fundCode: string; tab: string }>();
  const rawFundCode = String(params.fundCode || '');
  const rawTab = String(params.tab || '');
  const normalizedFundCode = normalizeFundCode(rawFundCode);
  const normalizedTab = normalizeFundTab(rawTab);

  if (!normalizedFundCode) {
    return <Navigate to={toFunds()} replace />;
  }

  const needsCanonicalFundCode = rawFundCode.trim().toUpperCase() !== normalizedFundCode;
  const needsCanonicalTab = rawTab.trim().toLowerCase() !== normalizedTab;
  if (needsCanonicalFundCode || needsCanonicalTab) {
    return <Navigate to={toFundDetail(normalizedFundCode, normalizedTab)} replace />;
  }

  const handleSectionNavigate = (section: MarketsNavigationSection) => {
    if (section === 'markets') {
      navigate(toMarketsOverview());
      return;
    }
    if (section === 'stocks') {
      navigate(toMarketsStocks(DEFAULT_MARKET_INDEX));
      return;
    }
    if (section === 'indices') {
      navigate(toMarketsIndices());
      return;
    }
    navigate(toFunds());
  };

  return (
    <FundsPage
      key={`${normalizedFundCode}-${normalizedTab}`}
      fundCode={normalizedFundCode}
      activeTab={normalizedTab}
      onOpenFund={(nextFundCode, nextTab = DEFAULT_FUND_TAB) => navigate(toFundDetail(nextFundCode, nextTab))}
      onTabChange={(nextTab) => navigate(toFundDetail(normalizedFundCode, nextTab))}
      onBack={() => navigate(toFunds())}
      onNavigateSection={handleSectionNavigate}
      onOpenTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
    />
  );
}

function App() {
  const location = useLocation();
  const navigate = useNavigate();

  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    return (localStorage.getItem('rag_fin_theme') as 'light' | 'dark') || 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('rag_fin_theme', theme);
  }, [theme]);

  const currentTicker = useMemo(() => {
    const match = location.pathname.match(/^\/stocks\/([^/]+)/i);
    if (!match?.[1]) return null;
    try {
      return normalizeTicker(decodeURIComponent(match[1]));
    } catch {
      return normalizeTicker(match[1]);
    }
  }, [location.pathname]);

  const isLandingPage = location.pathname === ROUTE_PATHS.landing;
  const isMarketsView = location.pathname.startsWith(ROUTE_PATHS.markets);
  const isFundsView = location.pathname.startsWith(ROUTE_PATHS.funds);
  const isStocksView = location.pathname.startsWith(ROUTE_PATHS.stocks);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  return (
    <div className="app-layout">
      {!isLandingPage && !isMarketsView && !isFundsView && !isStocksView && (
        <header className="app-header">
          <div className="header-content">
            <div
              className="header-brand"
              onClick={() => navigate(toMarketsOverview())}
              style={{ cursor: 'pointer' }}
            >
              <div className="logo-area">
                {/* Removed RAG-Fin Terminal text as requested */}
              </div>
            </div>

            <div className="header-actions">
              <button
                className="nav-tab"
                onClick={() => navigate(toMarketsOverview())}
                style={{
                  marginRight: '0.5rem',
                  padding: '0.4rem 1.2rem',
                  background: 'color-mix(in srgb, var(--surface-color) 80%, var(--bg-color))',
                  border: '1px solid var(--surface-border)',
                  borderRadius: '99px',
                  cursor: 'pointer',
                }}
              >
                Piyasalar
              </button>
              <GlobalTickerSearch
                currentTicker={currentTicker}
                onSelectTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
                onSelectFund={(fundCode) => navigate(toFundDetail(fundCode, DEFAULT_FUND_TAB))}
              />

              <button
                className="theme-toggle-btn"
                onClick={toggleTheme}
                title={theme === 'light' ? 'Dark Mode' : 'Light Mode'}
              >
                {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
              </button>
            </div>
          </div>
        </header>
      )}

      <main className="app-main-full">
        <Routes>
          <Route path={ROUTE_PATHS.landing} element={<LandingRoute />} />

          <Route path={ROUTE_PATHS.markets} element={<Navigate to={toMarketsStocks(DEFAULT_MARKET_INDEX)} replace />} />
          <Route path={ROUTE_PATHS.marketsOverview} element={<MarketsOverviewRoute />} />
          <Route path={ROUTE_PATHS.marketsStocks} element={<Navigate to={toMarketsStocks(DEFAULT_MARKET_INDEX)} replace />} />
          <Route path={ROUTE_PATHS.marketsStocksIndex} element={<MarketsStocksRoute />} />
          <Route path={ROUTE_PATHS.marketsIndices} element={<MarketsIndicesListRoute />} />
          <Route path={ROUTE_PATHS.marketsIndicesDetail} element={<MarketsIndicesDetailRoute />} />

          <Route path={ROUTE_PATHS.funds} element={<FundsListRoute />} />
          <Route path={ROUTE_PATHS.fundDetailNoTab} element={<FundCodeDefaultTabRedirect />} />
          <Route path={ROUTE_PATHS.fundDetail} element={<FundDetailRoute />} />

          <Route path={ROUTE_PATHS.stocks} element={<Navigate to={toMarketsStocks(DEFAULT_MARKET_INDEX)} replace />} />
          <Route path={ROUTE_PATHS.stockDetailNoTab} element={<StockTickerDefaultTabRedirect />} />
          <Route path={ROUTE_PATHS.stockDetail} element={<StockDetailRoute />} />

          <Route path="*" element={<Navigate to={ROUTE_PATHS.landing} replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
