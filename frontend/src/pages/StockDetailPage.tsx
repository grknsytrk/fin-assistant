import { useCallback, useEffect, useState } from 'react';
import { ArrowLeft, BarChart3, BookOpen, ChevronRight, FileText, Info, Star } from 'lucide-react';
import './StockDetailPage.css';
import { apiClient } from '../api/client';
import type { KapSnapshotResponse, KapQuarter } from '../api/types';
import { prepareOrderedQuarters } from '../utils/chartBuilders';
import SymbolLogo from '../components/SymbolLogo';
import MarketsNavigation, { type MarketsNavigationFundSection, type MarketsNavigationSection } from '../components/MarketsNavigation';
import { buildDocumentTitle, formatTitleCurrency, formatTitlePct, useDocumentTitle } from '../hooks/useDocumentTitle';
import { useWatchlist } from '../hooks/useWatchlist';
import type { StockTab } from '../routing/routes';

import StockOverview from './stock/sections/StockOverview';
import StockFinancials from './stock/sections/StockFinancials';
import StockKAP from './stock/sections/StockKAP';

interface StockDetailPageProps {
    ticker: string;
    activeTab?: StockTab;
    onTabChange?: (tab: StockTab) => void;
    onBack: () => void;
    onNavigateSection: (section: MarketsNavigationSection) => void;
    onNavigateFundSection?: (section: MarketsNavigationFundSection) => void;
    onOpenTicker: (ticker: string) => void;
    onOpenFund: (fundCode: string) => void;
}

type TabType = StockTab;

type StockPriceData = {
    ok: boolean;
    symbol: string;
    price: number | null;
    change: number | null;
    change_pct: number | null;
    currency: string;
    market_state: string;
    as_of?: string | null;
    error?: string;
};

const STOCK_DETAIL_TABS: Array<{ key: StockTab; label: string; icon: typeof Info }> = [
    { key: 'overview', label: 'Genel Bakış', icon: Info },
    { key: 'financials', label: 'Finansal Tablolar', icon: FileText },
    { key: 'kap', label: 'KAP Bildirimleri', icon: BookOpen },
];
const FULL_KAP_QUARTER_COUNT = 20;

function formatAsOf(value?: string | null): string {
    if (!value) return 'Veri güncelleniyor';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return 'Veri güncelleniyor';

    return new Intl.DateTimeFormat('tr-TR', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
    }).format(parsed);
}

function formatQuotePrice(value: number | null | undefined, currency?: string | null): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const currencyLabel = !currency || currency === 'TRY' || currency === 'TL' ? '₺' : currency;
    return `${currencyLabel}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatCompactCurrency(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return new Intl.NumberFormat('tr-TR', {
        style: 'currency',
        currency: 'TRY',
        notation: 'compact',
        maximumFractionDigits: 2,
    }).format(value);
}

function formatRatio(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return `${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}x`;
}

function formatPct(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '';
    const sign = value > 0 ? '+' : '';
    return `% ${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function pctClass(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '';
    if (value > 0) return 'positive';
    if (value < 0) return 'negative';
    return 'neutral';
}

export default function StockDetailPage({
    ticker,
    activeTab = 'overview',
    onTabChange,
    onBack,
    onNavigateSection,
    onNavigateFundSection,
    onOpenTicker,
    onOpenFund,
}: StockDetailPageProps) {
    const [selectedTab, setSelectedTab] = useState<TabType>(activeTab);
    const [snapshot, setSnapshot] = useState<KapSnapshotResponse | null>(null);
    const [quarters, setQuarters] = useState<KapQuarter[]>([]);
    const [loading, setLoading] = useState(false);
    const [snapshotQuarterDepth, setSnapshotQuarterDepth] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [navCollapsed, setNavCollapsed] = useState(false);
    const [priceData, setPriceData] = useState<StockPriceData | null>(null);
    const watchlist = useWatchlist();
    const normalizedTicker = ticker.trim().toUpperCase();

    useEffect(() => {
        setSelectedTab(activeTab);
    }, [activeTab]);

    const handleTabChange = (tab: TabType) => {
        setSelectedTab(tab);
        onTabChange?.(tab);
    };

    useEffect(() => {
        let mounted = true;
        setSnapshot(null);
        setQuarters([]);
        setSnapshotQuarterDepth(0);
        setError(null);
        setLoading(true);
        // Genel Bakış tab'ında 5/10/15/20 çeyrek seçenekleri var; ilk istekte
        // de tam 20 çeyrek çekiyoruz ki 15c/20c butonları boş görünmesin. KAP
        // cache'i taze ise tek istekte gelir, soğuk ise tek seferde uzun ama
        // sonraki tab geçişlerinde yeniden istek atılmaz.
        const initialQuarterCount = FULL_KAP_QUARTER_COUNT;
        apiClient.kapSnapshot(ticker, false, initialQuarterCount)
            .then(data => {
                if (mounted) {
                    setSnapshot(data);
                    setQuarters(prepareOrderedQuarters(data));
                    setSnapshotQuarterDepth(initialQuarterCount);
                    if (!data.ok && data.error) setError(data.error);
                }
            })
            .catch(err => {
                if (mounted) setError(err.message || 'Veri alınamadı');
            })
            .finally(() => {
                if (mounted) setLoading(false);
            });
        
        return () => { mounted = false; };
    }, [ticker]);

    useEffect(() => {
        if (selectedTab !== 'financials' && selectedTab !== 'kap') return;
        if (!snapshot) return;
        if (snapshotQuarterDepth >= FULL_KAP_QUARTER_COUNT) return;
        let mounted = true;
        apiClient.kapSnapshot(ticker, false, FULL_KAP_QUARTER_COUNT)
            .then(data => {
                if (!mounted) return;
                setSnapshot(data);
                setQuarters(prepareOrderedQuarters(data));
                setSnapshotQuarterDepth(FULL_KAP_QUARTER_COUNT);
                if (!data.ok && data.error) setError(data.error);
            })
            .catch(() => {
                // İlk özet snapshot ekranda kalır; derin veri yüklenemese de fiyat akışı bloklanmaz.
            });
        return () => { mounted = false; };
    }, [selectedTab, snapshot, snapshotQuarterDepth, ticker]);

    useEffect(() => {
        let cancelled = false;
        setPriceData(null);
        apiClient.kapPrice(ticker)
            .then((response) => {
                if (!cancelled) {
                    setPriceData(response as StockPriceData);
                }
            })
            .catch(() => {
                if (!cancelled) {
                    setPriceData(null);
                }
            });

        return () => {
            cancelled = true;
        };
    }, [ticker]);

    const renderContent = () => {
        if (loading && !snapshot) {
            return <div className="sd-loading"><div className="spinner" /> Veriler yükleniyor...</div>;
        }
        
        if (!snapshot) return null;

        switch (selectedTab) {
            case 'overview':
                return <StockOverview snapshot={snapshot} quarters={quarters} />;
            case 'financials':
                return <StockFinancials quarters={quarters} analysisNote={snapshot.analysis_note} />;
            case 'kap':
                return <StockKAP ticker={ticker} quarters={quarters} />;
            default:
                return null;
        }
    };

    const valuation = snapshot?.valuation;
    const displayPrice = priceData?.ok && priceData.price != null ? priceData.price : valuation?.price;
    const displayCurrency = priceData?.currency || valuation?.price_currency;
    const displayAsOf = priceData?.as_of || valuation?.price_as_of || snapshot?.fetched_at;
    const displayChangePct = priceData?.ok ? priceData.change_pct : null;
    const selectedTabLabel = STOCK_DETAIL_TABS.find((tab) => tab.key === selectedTab)?.label;
    const quoteTitle = [
        formatTitleCurrency(displayPrice, displayCurrency),
        formatTitlePct(displayChangePct),
    ].filter(Boolean).join(' ');
    useDocumentTitle(buildDocumentTitle(normalizedTicker, quoteTitle, selectedTabLabel));
    const isStarred = watchlist.hasItem('stock', normalizedTicker);
    const toggleStarredStock = useCallback(() => {
        if (!normalizedTicker) return;
        watchlist.toggleItem({
            kind: 'stock',
            symbol: normalizedTicker,
            label: snapshot?.company_title || normalizedTicker,
        });
    }, [normalizedTicker, snapshot?.company_title, watchlist]);

    return (
        <div className={`mn-layout stock-detail-shell${navCollapsed ? ' mn-nav-collapsed' : ''}`}>
            <MarketsNavigation
                collapsed={navCollapsed}
                activeSection="stocks"
                onCollapsedChange={setNavCollapsed}
                onSectionChange={onNavigateSection}
                onFundSectionChange={onNavigateFundSection}
                onSelectTicker={onOpenTicker}
                onSelectFund={onOpenFund}
            />
            <div className="stock-workspace">
                <div className="stock-detail-page">
                    <header className="stock-market-shell">
                        <div className="stock-market-breadcrumb" aria-label="Hisse konumu">
                            <button type="button" className="stock-breadcrumb-back" onClick={onBack}>
                                <ArrowLeft size={15} aria-hidden="true" />
                                Hisseler
                            </button>
                            <ChevronRight size={14} aria-hidden="true" />
                            <span className="stock-breadcrumb-group">
                                <BarChart3 size={17} aria-hidden="true" />
                                BIST Hisseleri
                            </span>
                            <ChevronRight size={14} aria-hidden="true" />
                            <span className="stock-breadcrumb-code">
                                <SymbolLogo
                                    symbol={ticker}
                                    name={snapshot?.company_title || ticker}
                                    kind="stock"
                                    size="xs"
                                />
                                {normalizedTicker}
                            </span>
                        </div>

                        <div className="stock-market-hero">
                            <div className="stock-market-title">
                                <SymbolLogo
                                    symbol={ticker}
                                    name={snapshot?.company_title || ticker}
                                    kind="stock"
                                    size="lg"
                                    className="stock-market-logo"
                                />
                                <div className="stock-market-copy">
                                    <div className="stock-market-code-row">
                                        <h1>{normalizedTicker}</h1>
                                        <button
                                            type="button"
                                            className={`stock-icon-action${isStarred ? ' active' : ''}`}
                                            onClick={toggleStarredStock}
                                            aria-pressed={isStarred}
                                            aria-label={isStarred ? `${normalizedTicker} izleme listesinden çıkar` : `${normalizedTicker} izleme listesine ekle`}
                                            title={isStarred ? 'İzleme listesinden çıkar' : 'İzleme listesine ekle'}
                                        >
                                            <Star size={19} aria-hidden="true" />
                                        </button>
                                    </div>
                                    <p>{snapshot?.company_title || (snapshot?.error ? ticker : 'Şirket bilgisi yükleniyor...')}</p>
                                    <small>
                                        {snapshot?.error
                                            ? snapshot.error
                                            : snapshot?.latest_quarter
                                                ? `Son dönem · ${snapshot.latest_quarter}`
                                                : 'KAP finansal verileri yükleniyor'}
                                    </small>
                                </div>
                            </div>

                            <div className="stock-market-quote">
                                <div>
                                    <strong>{formatQuotePrice(displayPrice, displayCurrency)}</strong>
                                    {displayChangePct != null && (
                                        <span className={pctClass(displayChangePct)}>{formatPct(displayChangePct)}</span>
                                    )}
                                </div>
                                <small>{formatAsOf(displayAsOf)}</small>
                            </div>
                        </div>

                        <div className="stock-market-stats">
                            <div><span>Piyasa Değeri</span><strong>{formatCompactCurrency(valuation?.market_cap)}</strong></div>
                            <div><span>F/K</span><strong>{formatRatio(valuation?.fk)}</strong></div>
                            <div><span>PD/DD</span><strong>{formatRatio(valuation?.pd_dd)}</strong></div>
                            <div><span>Son Dönem</span><strong>{snapshot?.latest_quarter || '-'}</strong></div>
                        </div>

                        <nav className="stock-tabs" aria-label="Hisse detay sekmeleri">
                            {STOCK_DETAIL_TABS.map((tab) => {
                                const Icon = tab.icon;
                                return (
                                    <button
                                        key={tab.key}
                                        type="button"
                                        className={selectedTab === tab.key ? 'active' : ''}
                                        onClick={() => handleTabChange(tab.key)}
                                    >
                                        <Icon size={16} aria-hidden="true" />
                                        {tab.label}
                                    </button>
                                );
                            })}
                        </nav>
                    </header>

                    <div className="sd-content-area">
                        {error && <div className="alert-error">{error}</div>}
                        {renderContent()}
                    </div>
                </div>
            </div>
        </div>
    );
}
