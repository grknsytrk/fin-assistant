import { useState, useEffect } from 'react';
import { ArrowLeft, FileText, Info, MessageSquare, BookOpen } from 'lucide-react';
import './StockDetailPage.css';
import { apiClient } from '../api/client';
import type { KapSnapshotResponse, KapQuarter } from '../api/types';
import { prepareOrderedQuarters } from '../utils/chartBuilders';
import { PriceTicker } from '../components/stock/PriceTicker';

import StockOverview from './stock/sections/StockOverview';
import StockFinancials from './stock/sections/StockFinancials';
import StockKAP from './stock/sections/StockKAP';
import StockAsk from './stock/sections/StockAsk';

interface StockDetailPageProps {
    ticker: string;
    onBack: () => void;
}

type TabType = 'overview' | 'financials' | 'kap' | 'ask';

export default function StockDetailPage({ ticker, onBack }: StockDetailPageProps) {
    const [activeTab, setActiveTab] = useState<TabType>('overview');
    const [snapshot, setSnapshot] = useState<KapSnapshotResponse | null>(null);
    const [quarters, setQuarters] = useState<KapQuarter[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const section = params.get('section');
        const normalizedSection = section === 'charts' ? 'overview' : section;
        if (normalizedSection && ['overview', 'financials', 'kap', 'ask'].includes(normalizedSection)) {
            setActiveTab(normalizedSection as TabType);
        }
    }, []);

    useEffect(() => {
        const url = new URL(window.location.href);
        url.searchParams.set('section', activeTab);
        window.history.replaceState({}, '', url.toString());
    }, [activeTab]);

    useEffect(() => {
        let mounted = true;
        setSnapshot(null);
        setQuarters([]);
        setError(null);
        setLoading(true);
        apiClient.kapSnapshot(ticker, false, 10)
            .then(data => {
                if (mounted) {
                    setSnapshot(data);
                    setQuarters(prepareOrderedQuarters(data));
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

    const renderContent = () => {
        if (loading && !snapshot) {
            return <div className="sd-loading"><div className="spinner" /> Veriler yükleniyor...</div>;
        }
        
        if (!snapshot) return null;

        switch (activeTab) {
            case 'overview':
                return <StockOverview snapshot={snapshot} quarters={quarters} />;
            case 'financials':
                return <StockFinancials quarters={quarters} analysisNote={snapshot.analysis_note} />;
            case 'kap':
                return <StockKAP ticker={ticker} quarters={quarters} />;
            case 'ask':
                return <StockAsk ticker={ticker} />;
            default:
                return null;
        }
    };

    return (
        <div className="stock-detail-page">
            <div className="sd-sidebar">
                <div className="sd-back" onClick={onBack}>
                    <ArrowLeft size={16} /> Piyasa Görünümü
                </div>
                
                <nav className="sd-nav">
                    <button 
                        className={`sd-nav-item ${activeTab === 'overview' ? 'active' : ''}`}
                        onClick={() => setActiveTab('overview')}
                    >
                        <Info size={16} /> Genel Bakış
                    </button>
                    <button 
                        className={`sd-nav-item ${activeTab === 'financials' ? 'active' : ''}`}
                        onClick={() => setActiveTab('financials')}
                    >
                        <FileText size={16} /> Finansal Tablolar
                    </button>
                    <button 
                        className={`sd-nav-item ${activeTab === 'kap' ? 'active' : ''}`}
                        onClick={() => setActiveTab('kap')}
                    >
                        <BookOpen size={16} /> KAP Bildirimleri
                    </button>
                    <button 
                        className={`sd-nav-item ${activeTab === 'ask' ? 'active' : ''}`}
                        onClick={() => setActiveTab('ask')}
                    >
                        <MessageSquare size={16} /> RAG Asistanı
                    </button>
                </nav>
            </div>

            <div className="sd-main">
                <header className="sd-topbar">
                    <PriceTicker
                        symbol={ticker}
                        companyName={snapshot?.company_title}
                        priceAsOf={snapshot?.valuation?.price_as_of}
                    />
                </header>
                
                <div className="sd-content-area">
                    {error && <div className="alert-error">{error}</div>}
                    {renderContent()}
                </div>
            </div>
        </div>
    );
}
