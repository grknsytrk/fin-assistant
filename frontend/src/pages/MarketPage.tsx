import { useDeferredValue, useEffect, useMemo, useState, type KeyboardEvent } from 'react';
import { Search, BarChart2, TrendingUp, PieChart, ShieldAlert, LineChart, Activity, ChevronRight, LayoutGrid } from 'lucide-react';
import { apiClient } from '../api/client';
import type { FundSummary, MarketUniverseRow } from '../api/types';
import AssetSearchResult from '../components/AssetSearchResult';
import { buildDocumentTitle, useDocumentTitle } from '../hooks/useDocumentTitle';
import { toFundDetail, toStockDetail } from '../routing/routes';
import './MarketPage.css';

type HeroSearchResult = {
    kind: 'stock' | 'fund';
    symbol: string;
    name: string;
    logoUrl?: string | null;
    price: number | null;
    changePct: number | null;
};

function normalizeHeroSearchText(value: string | null | undefined): string {
    return String(value || '')
        .toLocaleUpperCase('tr-TR')
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .replace(/[^A-Z0-9]+/g, ' ')
        .trim();
}

interface MarketPageProps {
    onOpenTicker?: (ticker: string) => void;
    onOpenFund?: (fundCode: string) => void;
    onOpenMarkets?: () => void;
}

export default function MarketPage({ onOpenTicker, onOpenFund, onOpenMarkets }: MarketPageProps) {
    const [searchTerm, setSearchTerm] = useState('');
    const [searchOpen, setSearchOpen] = useState(false);
    const [stockCatalogRows, setStockCatalogRows] = useState<MarketUniverseRow[]>(() => (
        apiClient.getCachedMarketUniverse()?.rows || []
    ));
    const [fundCatalogRows, setFundCatalogRows] = useState<FundSummary[]>(() => (
        apiClient.getCachedFundsCatalog()?.payload.rows || []
    ));
    const [stockCatalogLoading, setStockCatalogLoading] = useState(() => !apiClient.getCachedMarketUniverse());
    const [fundCatalogLoading, setFundCatalogLoading] = useState(() => !apiClient.getCachedFundsCatalog());
    const [stockCatalogError, setStockCatalogError] = useState(false);
    const [fundCatalogError, setFundCatalogError] = useState(false);
    const [highlightedSearchIndex, setHighlightedSearchIndex] = useState(0);
    const [activeTab, setActiveTab] = useState('Piyasa ve Tablolar');
    const deferredSearchTerm = useDeferredValue(searchTerm);
    useDocumentTitle(buildDocumentTitle('Piyasa ve Finansal Analiz'));

    useEffect(() => {
        let active = true;
        const cachedUniverse = apiClient.getCachedMarketUniverse();
        if (cachedUniverse) setStockCatalogRows(cachedUniverse.rows || []);
        else setStockCatalogLoading(true);
        void apiClient.marketUniverse()
            .then((payload) => {
                if (!active) return;
                setStockCatalogRows(payload.rows || []);
                setStockCatalogError(false);
            })
            .catch(() => {
                if (active) setStockCatalogError(true);
            })
            .finally(() => {
                if (active) setStockCatalogLoading(false);
            });

        const applyFundCatalog = (payload: Awaited<ReturnType<typeof apiClient.funds>>) => {
            if (!active || payload.status === 'pending') return;
            setFundCatalogRows(payload.rows || []);
            setFundCatalogError(false);
            setFundCatalogLoading(false);
        };
        const unsubscribeFunds = apiClient.subscribeFundsCatalog(applyFundCatalog);
        const cachedFundsCatalog = apiClient.getCachedFundsCatalog();
        if (cachedFundsCatalog) {
            setFundCatalogRows(cachedFundsCatalog.payload.rows || []);
        }
        if (!cachedFundsCatalog || Date.now() - cachedFundsCatalog.fetchedAt >= 5 * 60 * 1000) {
            setFundCatalogLoading(!cachedFundsCatalog);
            void apiClient.funds()
                .then(applyFundCatalog)
                .catch(() => {
                    if (active) setFundCatalogError(true);
                })
                .finally(() => {
                    if (active) setFundCatalogLoading(false);
                });
        } else {
            setFundCatalogLoading(false);
        }

        return () => {
            active = false;
            unsubscribeFunds();
        };
    }, []);

    const stockSearchIndex = useMemo(() => stockCatalogRows.map((row, index) => {
        const symbol = normalizeHeroSearchText(row.symbol || row.company);
        const name = normalizeHeroSearchText(row.name || row.company);
        return {
            row,
            symbol,
            name,
            haystack: normalizeHeroSearchText(`${row.symbol || ''} ${row.name || ''} ${row.company}`),
            index,
        };
    }), [stockCatalogRows]);
    const fundSearchIndex = useMemo(() => fundCatalogRows.map((fund, index) => {
        const code = normalizeHeroSearchText(fund.fund_code);
        const name = normalizeHeroSearchText(fund.name);
        return {
            fund,
            code,
            name,
            haystack: normalizeHeroSearchText(
                `${fund.fund_code} ${fund.name} ${fund.fund_type || ''} ${fund.founder_company || ''} ${fund.manager_company || ''}`,
            ),
            index,
        };
    }), [fundCatalogRows]);
    const searchResults: HeroSearchResult[] = useMemo(() => {
        const normalizedQuery = normalizeHeroSearchText(deferredSearchTerm);
        if (!normalizedQuery) return [];
        const stockMatches = stockSearchIndex
            .filter(({ haystack }) => haystack.includes(normalizedQuery))
            .map((entry) => ({
                ...entry,
                rank: entry.symbol === normalizedQuery
                    ? 0
                    : entry.symbol.startsWith(normalizedQuery)
                        ? 1
                        : entry.name.startsWith(normalizedQuery)
                            ? 2
                            : entry.symbol.includes(normalizedQuery)
                                ? 3
                                : entry.name.includes(normalizedQuery)
                                    ? 4
                                    : 5,
            }))
            .sort((left, right) => left.rank - right.rank || left.index - right.index)
            .slice(0, 6)
            .map(({ row }) => ({
                kind: 'stock' as const,
                symbol: String(row.symbol || row.company).trim().toUpperCase(),
                name: row.name || row.company,
                logoUrl: row.logo_url,
                price: row.price,
                changePct: row.change_pct,
            }));
        const fundMatches = fundSearchIndex
            .filter(({ haystack }) => haystack.includes(normalizedQuery))
            .map((entry) => ({
                ...entry,
                rank: entry.code === normalizedQuery
                    ? 0
                    : entry.code.startsWith(normalizedQuery)
                        ? 1
                        : entry.name.startsWith(normalizedQuery)
                            ? 2
                            : entry.code.includes(normalizedQuery)
                                ? 3
                                : entry.name.includes(normalizedQuery)
                                    ? 4
                                    : 5,
            }))
            .sort((left, right) => left.rank - right.rank || left.index - right.index)
            .slice(0, 6)
            .map(({ fund }) => ({
                kind: 'fund' as const,
                symbol: fund.fund_code.trim().toUpperCase(),
                name: fund.name,
                price: fund.price,
                changePct: fund.daily_return,
            }));
        return [...stockMatches, ...fundMatches];
    }, [deferredSearchTerm, fundSearchIndex, stockSearchIndex]);
    const searchResultsAreCurrent = normalizeHeroSearchText(deferredSearchTerm) === normalizeHeroSearchText(searchTerm);
    const searchLoading = Boolean(searchOpen && searchTerm.trim())
        && (stockCatalogLoading || fundCatalogLoading || !searchResultsAreCurrent);
    const searchError = stockCatalogError || fundCatalogError;

    const openSearchResult = (result: HeroSearchResult) => {
        setSearchOpen(false);
        if (result.kind === 'fund') {
            if (onOpenFund) onOpenFund(result.symbol);
            else window.location.href = `/?fund=${encodeURIComponent(result.symbol)}`;
            return;
        }
        if (onOpenTicker) onOpenTicker(result.symbol);
        else window.location.href = `/?ticker=${encodeURIComponent(result.symbol)}`;
    };

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        if (!searchTerm.trim()) return;
        if (!searchResultsAreCurrent) return;
        const selected = searchResults[highlightedSearchIndex] || searchResults[0];
        if (selected) {
            openSearchResult(selected);
            return;
        }
        if (searchLoading) return;
        const nextTicker = searchTerm.trim().toUpperCase();
        if (onOpenTicker) {
            onOpenTicker(nextTicker);
        } else {
            window.location.href = `/?ticker=${nextTicker}`;
        }
    };

    const handleSearchKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Escape') {
            setSearchOpen(false);
            return;
        }
        if (event.key === 'ArrowDown' && searchResultsAreCurrent && searchResults.length) {
            event.preventDefault();
            setHighlightedSearchIndex((current) => (current + 1) % searchResults.length);
        } else if (event.key === 'ArrowUp' && searchResultsAreCurrent && searchResults.length) {
            event.preventDefault();
            setHighlightedSearchIndex((current) => (current <= 0 ? searchResults.length - 1 : current - 1));
        }
    };

    const getTabImage = () => {
        switch(activeTab) {
            case 'Piyasa ve Tablolar': return '/piyasa_tablolar.webp';
            case 'Hisse Tarama': return '/hisse_tarama.webp';
            case 'Model Portföy': return '/model_portfoy.webp';
            default: return '/piyasa_tablolar.webp';
        }
    };

    return (
        <div className="landing-page">
            {/* HERO SECTION */}
            <section className="hero-section">
                <div className="hero-glow"></div>
                <div className="hero-content">
                    <h1 className="hero-title">Doğru yatırımı keşfedin</h1>
                    <p className="hero-subtitle">Türkiye'nin en kapsamlı halka açık hisse senedi temel ve teknik analiz platformu.</p>
                    
                    <div
                        className="hero-search-container"
                        onBlur={(event) => {
                            const nextTarget = event.relatedTarget as Node | null;
                            if (!nextTarget || !event.currentTarget.contains(nextTarget)) setSearchOpen(false);
                        }}
                    >
                        <form className="hero-search" onSubmit={handleSearch}>
                            <Search className="hero-search-icon" size={16} aria-hidden="true" />
                            <input
                                type="text"
                                placeholder="Hisse veya fon kodu, ad ya da kurucu ara"
                                value={searchTerm}
                                onFocus={() => setSearchOpen(true)}
                                onKeyDown={handleSearchKeyDown}
                                onChange={(event) => {
                                    setSearchTerm(event.target.value);
                                    setSearchOpen(true);
                                    setHighlightedSearchIndex(0);
                                }}
                                role="combobox"
                                aria-autocomplete="list"
                                aria-expanded={searchOpen && Boolean(searchTerm.trim())}
                                aria-controls="hero-search-results"
                                aria-activedescendant={searchResults[highlightedSearchIndex]
                                    ? `hero-search-option-${searchResults[highlightedSearchIndex].kind}-${searchResults[highlightedSearchIndex].symbol}`
                                    : undefined}
                            />
                        </form>
                        {searchOpen && searchTerm.trim() && (
                            <div className="hero-search-results" id="hero-search-results" role="listbox" aria-label="Arama sonuçları">
                                {searchLoading && searchResults.length === 0 ? (
                                    <div className="hero-search-status">Aranıyor...</div>
                                ) : searchResults.length === 0 ? (
                                    <div className="hero-search-status">
                                        {searchError ? 'Arama şu anda yapılamıyor.' : 'Sonuç bulunamadı.'}
                                    </div>
                                ) : (
                                    (['stock', 'fund'] as const).map((kind) => {
                                        const groupResults = searchResults
                                            .map((result, index) => ({ result, index }))
                                            .filter(({ result }) => result.kind === kind);
                                        if (!groupResults.length) return null;
                                        return (
                                            <div className="hero-search-group" role="group" aria-label={kind === 'stock' ? 'Hisseler' : 'Fonlar'} key={kind}>
                                                <div className="hero-search-group-title">{kind === 'stock' ? 'HİSSELER' : 'FONLAR'}</div>
                                                {groupResults.map(({ result, index }) => (
                                                    <AssetSearchResult
                                                        id={`hero-search-option-${result.kind}-${result.symbol}`}
                                                        key={`${result.kind}-${result.symbol}`}
                                                        href={result.kind === 'fund' ? toFundDetail(result.symbol) : toStockDetail(result.symbol)}
                                                        kind={result.kind}
                                                        symbol={result.symbol}
                                                        name={result.name}
                                                        price={result.price}
                                                        changePct={result.changePct}
                                                        logoUrl={result.logoUrl}
                                                        active={index === highlightedSearchIndex}
                                                        onMouseDown={(event) => {
                                                            if (event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey) {
                                                                event.preventDefault();
                                                            }
                                                        }}
                                                        onMouseEnter={() => setHighlightedSearchIndex(index)}
                                                        onClick={(event) => {
                                                            if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                                                            event.preventDefault();
                                                            openSearchResult(result);
                                                        }}
                                                    />
                                                ))}
                                            </div>
                                        );
                                    })
                                )}
                            </div>
                        )}
                    </div>
                </div>

                <div 
                    className="hero-mockup-wrapper" 
                    onClick={() => {
                        if (onOpenMarkets) {
                            onOpenMarkets();
                        } else {
                            window.location.href = '/?page=markets';
                        }
                    }}
                    style={{ cursor: 'pointer' }}
                >
                    <img
                        src="/hero_dashboard.png"
                        alt="Platform Ekranı"
                        className="mockup-full-image"
                        loading="eager"
                        decoding="async"
                        fetchPriority="high"
                    />
                </div>
            </section>

            {/* TABS SECTION */}
            <section className="features-tabs-section">
                <div className="tabs-container">
                    <div className="tabs-sidebar">
                        <button 
                            className={`tab-btn ${activeTab === 'Piyasa ve Tablolar' ? 'active' : ''}`}
                            onClick={() => setActiveTab('Piyasa ve Tablolar')}
                        >
                            <div className="tab-icon"><LayoutGrid size={18} /></div>
                            <div className="tab-text">
                                <h3>Piyasa ve Tablolar</h3>
                                <p>Bist ve dünya piyasalarındaki durumu izleyin, şirketleri analiz edin.</p>
                            </div>
                        </button>
                        <button 
                            className={`tab-btn ${activeTab === 'Hisse Tarama' ? 'active' : ''}`}
                            onClick={() => setActiveTab('Hisse Tarama')}
                        >
                            <div className="tab-icon"><Search size={18} /></div>
                            <div className="tab-text">
                                <h3>Hisse Tarama</h3>
                                <p>Onlarca temel ve teknik kritere göre filtreleme ve karşılaştırma yapın.</p>
                            </div>
                        </button>
                        <button 
                            className={`tab-btn ${activeTab === 'Model Portföy' ? 'active' : ''}`}
                            onClick={() => setActiveTab('Model Portföy')}
                        >
                            <div className="tab-icon"><PieChart size={18} /></div>
                            <div className="tab-text">
                                <h3>Model Portföy</h3>
                                <p>Uzman stratejileri ile hazırlanan portföyleri inceleyin ve uygulayın.</p>
                            </div>
                        </button>
                    </div>
                    <div className="tabs-content">
                        <div className="tab-visual-mockup">
                            <img
                                src={getTabImage()}
                                alt={activeTab}
                                className="tab-visual-image"
                                loading="lazy"
                                decoding="async"
                            />
                        </div>
                    </div>
                </div>
            </section>

            {/* DETAILED FEATURES GRID */}
            <section className="details-section">
                <div className="details-header">
                    <h2>Aradığın hisse senedi ve inceleme detayı sana sunar</h2>
                    <p>Finansal okuryazarlığı güçlendiren özelliklere göz atın.</p>
                </div>
                
                <div className="details-grid">
                    <div className="detail-card">
                        <div className="dc-icon blue-bg"><BarChart2 size={20} /></div>
                        <h3>Finansal Model</h3>
                        <p>Geçmiş ve gelecek finansal hedefleri kıyaslayarak detaylı görünüm elde edin.</p>
                    </div>
                    <div className="detail-card">
                        <div className="dc-icon green-bg"><TrendingUp size={20} /></div>
                        <h3>Temel Analiz</h3>
                        <p>Şirketleri çarpanlara ve temel göstergelere göre hızla değerlendirip konumlayın.</p>
                    </div>
                    <div className="detail-card">
                        <div className="dc-icon purple-bg"><PieChart size={20} /></div>
                        <h3>Rasyo Analizi</h3>
                        <p>Karşılaştırmalı rasyo oranlarıyla şirketin geçmişten bugüne performansını ölçün.</p>
                    </div>
                    <div className="detail-card">
                        <div className="dc-icon orange-bg"><ShieldAlert size={20} /></div>
                        <h3>Risk Metrikleri</h3>
                        <p>Volatilite ve benzeri metrikler ile risk algınızı belirleyip portföyü dengede tutun.</p>
                    </div>
                    <div className="detail-card">
                        <div className="dc-icon cyan-bg"><LineChart size={20} /></div>
                        <h3>Gelişmiş Grafikler</h3>
                        <p>Hisse senedi verilerini interaktif grafiklerle zamana bazlı detaylıca analiz edin.</p>
                    </div>
                    <div className="detail-card">
                        <div className="dc-icon red-bg"><Activity size={20} /></div>
                        <h3>Teknik Analiz</h3>
                        <p>Hacim, destek-direnç ve indikatör incelemesiyle yatırımlarınızı yönlendirin.</p>
                    </div>
                </div>
            </section>

            {/* BOTTOM CTA */}
            <section className="bottom-cta-section">
                <div className="cta-content">
                    <h2>Doğru yatırımı keşfedin!</h2>
                    <p>Siz de profesyoneller gibi yatırım kararları alabilirsiniz.</p>
                    <button
                        className="cta-btn"
                        onClick={() => {
                            if (onOpenMarkets) {
                                onOpenMarkets();
                            } else {
                                window.location.href = '/?page=markets';
                            }
                        }}
                    >
                        Ücretsiz Başla <ChevronRight size={16} />
                    </button>
                </div>
                <div className="cta-footer-links">
                    <span>Hakkımızda</span>
                    <span>Gizlilik Sözleşmesi</span>
                    <span>Kullanım Koşulları</span>
                    <a href="https://github.com/grknsytrk" target="_blank" rel="noopener noreferrer" style={{color: 'inherit', textDecoration: 'none'}}>İletişim</a>
                </div>
                <div className="cta-copyright">
                    © 2026 RAG-Fin. Tüm hakları saklıdır.
                </div>
            </section>
        </div>
    );
}
