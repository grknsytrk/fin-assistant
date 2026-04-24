import { useState, useEffect } from 'react';
import { Sun, Moon } from 'lucide-react';
import MarketPage from './pages/MarketPage';
import MarketsView from './pages/MarketsView';
import StockDetailPage from './pages/StockDetailPage';
import GlobalTickerSearch from './components/GlobalTickerSearch';
import './index.css';

function App() {
  const [ticker, setTicker] = useState<string | null>(null);
  const [page, setPage] = useState<string | null>(null);

  // Theme state wrapper
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    return (localStorage.getItem('rag_fin_theme') as 'light' | 'dark') || 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('rag_fin_theme', theme);
  }, [theme]);

  useEffect(() => {
    // URL'den ticker okuma
    const params = new URLSearchParams(window.location.search);
    const urlTicker = params.get('ticker');
    if (urlTicker) {
      setTicker(urlTicker.toUpperCase());
    } else {
      setTicker(null);
    }
    setPage(params.get('page'));

    const handlePopState = () => {
      const p = new URLSearchParams(window.location.search);
      setTicker(p.get('ticker')?.toUpperCase() || null);
      setPage(p.get('page'));
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  const handleBackToMarket = () => {
    const url = new URL(window.location.href);
    url.searchParams.delete('ticker');
    url.searchParams.delete('section');
    url.searchParams.delete('page');
    window.history.pushState({}, '', url.toString());
    setTicker(null);
    setPage(null);
  };

  const handleGoToMarkets = () => {
    const url = new URL(window.location.href);
    url.searchParams.delete('ticker');
    url.searchParams.delete('section');
    url.searchParams.set('page', 'markets');
    window.history.pushState({}, '', url.toString());
    setTicker(null);
    setPage('markets');
  };

  const handleSelectTicker = (nextTicker: string) => {
    const normalizedTicker = nextTicker.trim().toUpperCase();
    if (!normalizedTicker) {
      return;
    }

    const url = new URL(window.location.href);
    url.searchParams.set('ticker', normalizedTicker);
    if (!url.searchParams.get('section')) {
      url.searchParams.set('section', 'overview');
    }
    window.history.pushState({}, '', url.toString());
    setTicker(normalizedTicker);
  };

  const isMarketsView = !ticker && page === 'markets';

  return (
    <div className="app-layout">
      <header className="app-header">
        <div className="header-content">
          <div className="header-brand" onClick={handleBackToMarket} style={{ cursor: 'pointer' }}>
            <div className="logo-area">
              {/* Removed RAG-Fin Terminal text as requested */}
            </div>
          </div>
          
          <div className="header-actions">
            <button 
              className="nav-tab" 
              onClick={handleGoToMarkets}
              style={{ 
                  marginRight: '0.5rem', 
                  padding: '0.4rem 1.2rem', 
                  background: 'color-mix(in srgb, var(--surface-color) 80%, var(--bg-color))',
                  border: '1px solid var(--surface-border)', 
                  borderRadius: '99px',
                  cursor: 'pointer'
              }}
            >
              Piyasalar
            </button>
            {!isMarketsView && (
              <GlobalTickerSearch currentTicker={ticker} onSelectTicker={handleSelectTicker} />
            )}
            
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

      <main className="app-main-full">
        {ticker ? (
          <StockDetailPage key={ticker} ticker={ticker} onBack={handleBackToMarket} />
        ) : page === 'markets' ? (
          <MarketsView />
        ) : (
          <MarketPage />
        )}
      </main>
    </div>
  );
}

export default App;
