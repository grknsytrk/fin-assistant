import { useState } from 'react';
import { apiClient } from '../api/client';
import type { StatsResponse } from '../api/types';

export default function ReportsPage() {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<any>(null);
    const [stats, setStats] = useState<StatsResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    async function refreshStats() {
        try {
            setStats(await apiClient.stats());
        } catch {
            // Keep current UI response; stats are supplementary.
        }
    }

    async function handleIngest() {
        setLoading(true); setError(null); setResult(null);
        try {
            const res = await apiClient.ingest();
            setResult({ type: 'ingest', data: res });
            await refreshStats();
        } catch (err: any) {
            setError(err.message || 'Ingest hatası');
        } finally {
            setLoading(false);
        }
    }

    async function handleIndex(version: 'v1' | 'v2') {
        setLoading(true); setError(null); setResult(null);
        try {
            const res = await apiClient.index(version);
            setResult({ type: 'index', data: res });
            await refreshStats();
        } catch (err: any) {
            setError(err.message || 'Index hatası');
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="page-container">
            <header className="page-header">
                <h1>Raporlar ve İndeksleme</h1>
                <p>PDF ingest ve ChromaDB indeksleme işlemleri</p>
            </header>

            <div className="controls-panel">
                <button className="btn-primary" onClick={handleIngest} disabled={loading}>
                    1. PDF İngest
                </button>
                <button className="btn-secondary" onClick={() => handleIndex('v1')} disabled={loading}>
                    2. Index (v1)
                </button>
                <button className="btn-secondary" onClick={() => handleIndex('v2')} disabled={loading}>
                    3. Index (v2)
                </button>
            </div>

            {loading && <div className="loading-state">İşlem devam ediyor…</div>}
            {error && <div className="error-message">Hata: {error}</div>}

            {result && (
                <div className="panel result-panel" style={{ marginTop: '1.5rem' }}>
                    <h3>Sonuç ({result.type})</h3>
                    {result.type === 'ingest' && (
                        <div className="stats-grid" style={{ marginBottom: '1rem' }}>
                            <div className="stat-card">
                                <div className="stat-value">{result.data?.pages_written ?? 0}</div>
                                <div className="stat-label">Yazılan Sayfa</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{result.data?.summary?.num_pdfs ?? 0}</div>
                                <div className="stat-label">İşlenen PDF</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{result.data?.summary?.total_pages ?? 0}</div>
                                <div className="stat-label">Toplam Sayfa</div>
                            </div>
                        </div>
                    )}
                    {result.type === 'index' && (
                        <div className="stats-grid" style={{ marginBottom: '1rem' }}>
                            <div className="stat-card">
                                <div className="stat-value">{result.data?.summary?.indexed_chunks ?? 0}</div>
                                <div className="stat-label">İndekslenen Chunk</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{result.data?.summary?.collection_count ?? 0}</div>
                                <div className="stat-label">Collection Count</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{result.data?.summary?.indexing_success ? 'OK' : 'NOK'}</div>
                                <div className="stat-label">Durum</div>
                            </div>
                        </div>
                    )}
                    <pre>{JSON.stringify(result.data, null, 2)}</pre>
                </div>
            )}

            {stats && (
                <div className="panel" style={{ marginTop: '1rem' }}>
                    <h3>Güncel Veri Durumu</h3>
                    <div className="stats-grid" style={{ marginTop: '0.75rem' }}>
                        <div className="stat-card">
                            <div className="stat-value">{stats.pdf_count}</div>
                            <div className="stat-label">PDF</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{stats.page_count}</div>
                            <div className="stat-label">Sayfa</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{stats.chunk_count_v2}</div>
                            <div className="stat-label">Chunk v2</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{stats.companies.length}</div>
                            <div className="stat-label">Şirket</div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
