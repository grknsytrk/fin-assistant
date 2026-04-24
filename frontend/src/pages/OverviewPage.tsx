import { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import type { CompanyBreakdownRow, StatsResponse } from '../api/types';

export default function OverviewPage() {
    const [stats, setStats] = useState<StatsResponse | null>(null);
    const [breakdown, setBreakdown] = useState<CompanyBreakdownRow[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => { loadStats(); }, []);

    async function loadStats() {
        setLoading(true);
        setError(null);
        try {
            const [statsPayload, breakdownPayload] = await Promise.all([
                apiClient.stats(),
                apiClient.companyBreakdown(),
            ]);
            setStats(statsPayload);
            setBreakdown(breakdownPayload.rows || []);
        } catch (err: any) {
            setError(err.message || 'İstatistikler alınamadı.');
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="page-container">
            <header className="page-header">
                <h1>Genel Bakış</h1>
                <p>Sistem durumu ve istatistikler</p>
            </header>

            {loading && <div className="loading-state">Yükleniyor…</div>}

            {error && (
                <div className="error-message">
                    <strong>Hata:</strong> {error}
                    <button onClick={loadStats} className="btn-secondary" style={{ marginLeft: '1rem' }}>Tekrar Dene</button>
                </div>
            )}

            {stats && !loading && (
                <>
                    <div className="stats-grid">
                        <div className="stat-card">
                            <div className="stat-value">{stats.pdf_count}</div>
                            <div className="stat-label">PDF Sayısı</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{stats.page_count}</div>
                            <div className="stat-label">Sayfa Sayısı</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{stats.chunk_count_v1}</div>
                            <div className="stat-label">Chunk (v1)</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{stats.chunk_count_v2}</div>
                            <div className="stat-label">Chunk (v2)</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{stats.collection_count_v2 ?? '—'}</div>
                            <div className="stat-label">ChromaDB Kayıt</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value">{stats.companies.length}</div>
                            <div className="stat-label">Şirket Sayısı</div>
                        </div>
                    </div>

                    {breakdown.length > 0 && (
                        <div className="panel" style={{ marginTop: '1.5rem' }}>
                            <h3>Şirket Bazlı İndekslenen Chunk Dağılımı</h3>
                            <p className="subtext" style={{ marginBottom: '1rem' }}>
                                Ingest/Index sonrası hangi şirkette ne kadar veri bulunduğunu gösterir.
                            </p>
                            <div className="mini-bars">
                                {breakdown.slice(0, 10).map((row) => {
                                    const maxValue = breakdown[0]?.chunks || 1;
                                    const width = Math.max(6, Math.round((row.chunks / maxValue) * 100));
                                    return (
                                        <div key={row.company} className="mini-bar-row">
                                            <span className="mini-bar-label">{row.company}</span>
                                            <div className="mini-bar-track">
                                                <div className="mini-bar-fill" style={{ width: `${width}%` }} />
                                            </div>
                                            <span className="mini-bar-value">{row.chunks}</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
