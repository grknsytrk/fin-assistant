import { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import type { MarketStocksResponse } from '../api/types';
import MarketsNavigation, {
    type MarketsNavigationFundSection,
    type MarketsNavigationSection,
    type MarketsNavigationStockSection,
} from '../components/MarketsNavigation';
import StockComparisonSection from './markets/sections/StockComparisonSection';
import '../styles/market-shared.css';

export interface StockComparisonPageProps {
    onNavigateSection: (section: MarketsNavigationSection) => void;
    onNavigateStockSection?: (section: MarketsNavigationStockSection) => void;
    onNavigateFundSection?: (section: MarketsNavigationFundSection) => void;
    onOpenTicker: (ticker: string) => void;
    onOpenFund: (fundCode: string) => void;
}

export default function StockComparisonPage({
    onNavigateSection,
    onNavigateStockSection,
    onNavigateFundSection,
    onOpenTicker,
    onOpenFund,
}: StockComparisonPageProps) {
    const [navCollapsed, setNavCollapsed] = useState(false);
    const [stocks, setStocks] = useState<MarketStocksResponse | null>(() => apiClient.getCachedMarketStocks('XUTUM'));
    const [loading, setLoading] = useState(() => !apiClient.getCachedMarketStocks('XUTUM'));
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let alive = true;
        const cached = apiClient.getCachedMarketStocks('XUTUM');
        if (cached) {
            setStocks(cached);
            setLoading(false);
        } else {
            setLoading(true);
        }
        apiClient
            .marketStocks({ index: 'XUTUM' })
            .then((payload) => {
                if (!alive) return;
                setStocks(payload);
                setError(null);
            })
            .catch((err) => {
                if (!alive) return;
                if (!cached) {
                    setError(err instanceof Error ? err.message : 'Hisse listesi alınamadı.');
                }
            })
            .finally(() => {
                if (alive) setLoading(false);
            });
        return () => {
            alive = false;
        };
    }, []);

    return (
        <div className={`mn-layout${navCollapsed ? ' mn-nav-collapsed' : ''}`}>
            <MarketsNavigation
                collapsed={navCollapsed}
                activeSection="stocks"
                activeStockSection="compare"
                onCollapsedChange={setNavCollapsed}
                onSectionChange={onNavigateSection}
                onStockSectionChange={onNavigateStockSection}
                onFundSectionChange={onNavigateFundSection}
                onSelectTicker={onOpenTicker}
                onSelectFund={onOpenFund}
            />
            <div className="funds-workspace">
                <div className="funds-page">
                    <div className="mobile-subsection-switcher" role="navigation" aria-label="Hisse görünümleri">
                        <button
                            type="button"
                            className="mobile-subsection-btn"
                            aria-pressed={false}
                            onClick={() => {
                                if (onNavigateStockSection) {
                                    onNavigateStockSection('list');
                                } else {
                                    onNavigateSection('stocks');
                                }
                            }}
                        >
                            Hisse Listesi
                        </button>
                        <button
                            type="button"
                            className="mobile-subsection-btn active"
                            aria-pressed={true}
                            onClick={() => onNavigateStockSection?.('compare')}
                        >
                            Hisse Karşılaştırma
                        </button>
                    </div>
                    <StockComparisonSection stocks={stocks} loading={loading} error={error} />
                </div>
            </div>
        </div>
    );
}
