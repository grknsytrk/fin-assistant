import { useState } from 'react';
import { Search, BarChart2, TrendingUp, PieChart, ShieldAlert, LineChart, Activity, ChevronRight, LayoutGrid } from 'lucide-react';
import GlobalTickerSearch from '../components/GlobalTickerSearch';
import { buildDocumentTitle, useDocumentTitle } from '../hooks/useDocumentTitle';
import './MarketPage.css';

interface MarketPageProps {
    onOpenTicker?: (ticker: string) => void;
    onOpenFund?: (fundCode: string) => void;
    onOpenMarkets?: () => void;
}

export default function MarketPage({ onOpenTicker, onOpenFund, onOpenMarkets }: MarketPageProps) {
    const [activeTab, setActiveTab] = useState('Piyasa ve Tablolar');
    useDocumentTitle(buildDocumentTitle('Piyasa ve Finansal Analiz'));

    const handleOpenMarkets = () => {
        if (onOpenMarkets) {
            onOpenMarkets();
        } else {
            window.location.href = '/?page=markets';
        }
    };

    const handleSelectTicker = (ticker: string) => {
        if (onOpenTicker) onOpenTicker(ticker);
        else window.location.href = `/?ticker=${encodeURIComponent(ticker)}`;
    };

    const handleSelectFund = (fundCode: string) => {
        if (onOpenFund) onOpenFund(fundCode);
        else window.location.href = `/?fund=${encodeURIComponent(fundCode)}`;
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
                <div className="hero-content">
                    <h1 className="hero-title">Doğru yatırımı keşfedin</h1>
                    <p className="hero-subtitle">Türkiye'nin en kapsamlı halka açık hisse senedi temel ve teknik analiz platformu.</p>
                    
                    <div className="hero-actions">
                        <GlobalTickerSearch
                            variant="hero"
                            onSelectTicker={handleSelectTicker}
                            onSelectFund={handleSelectFund}
                        />
                        <button
                            type="button"
                            className="cta-btn hero-primary-btn"
                            onClick={handleOpenMarkets}
                        >
                            Piyasalar Terminalini Aç <ChevronRight size={16} aria-hidden="true" />
                        </button>
                    </div>
                </div>

                <div 
                    className="hero-mockup-wrapper" 
                    onClick={handleOpenMarkets}
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
                        <div className="dc-icon"><BarChart2 size={18} /></div>
                        <h3>Finansal Model</h3>
                        <p>Geçmiş ve gelecek finansal hedefleri kıyaslayarak detaylı görünüm elde edin.</p>
                    </div>
                    <div className="detail-card">
                        <div className="dc-icon"><TrendingUp size={18} /></div>
                        <h3>Temel Analiz</h3>
                        <p>Şirketleri çarpanlara ve temel göstergelere göre hızla değerlendirip konumlayın.</p>
                    </div>
                    <div className="detail-card">
                        <div className="dc-icon"><PieChart size={18} /></div>
                        <h3>Rasyo Analizi</h3>
                        <p>Karşılaştırmalı rasyo oranlarıyla şirketin geçmişten bugüne performansını ölçün.</p>
                    </div>
                    <div className="detail-card">
                        <div className="dc-icon"><ShieldAlert size={18} /></div>
                        <h3>Risk Metrikleri</h3>
                        <p>Volatilite ve benzeri metrikler ile risk algınızı belirleyip portföyü dengede tutun.</p>
                    </div>
                    <div className="detail-card">
                        <div className="dc-icon"><LineChart size={18} /></div>
                        <h3>Gelişmiş Grafikler</h3>
                        <p>Hisse senedi verilerini interaktif grafiklerle zamana bazlı detaylıca analiz edin.</p>
                    </div>
                    <div className="detail-card">
                        <div className="dc-icon"><Activity size={18} /></div>
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
                        type="button"
                        className="cta-btn"
                        onClick={handleOpenMarkets}
                    >
                        Ücretsiz Başla <ChevronRight size={16} aria-hidden="true" />
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
