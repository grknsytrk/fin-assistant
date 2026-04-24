import AskScreen from '../../../components/AskScreen';

export default function StockAsk({ ticker }: { ticker: string }) {
    return (
        <div className="section-ask fade-in">
            <div className="panel">
                <div className="panel-header" style={{ marginBottom: '0.5rem' }}>
                    <h3>{ticker} RAG Asistanı</h3>
                    <p className="subtext">Bu şirketin KAP raporları, faaliyet raporları ve mali tabloları hakkında sorular sorabilirsiniz.</p>
                </div>
                <div style={{ marginTop: '1rem' }}>
                    <AskScreen initialCompany={ticker} disableCompanySelect={true} />
                </div>
            </div>
        </div>
    );
}
