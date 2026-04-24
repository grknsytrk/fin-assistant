import { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import type { CompanyBreakdownRow, StatsResponse } from '../api/types';

export default function CompaniesPage() {
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
            setError(err.message || 'Şirket listesi yüklenemedi.');
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="page-container">
            <header className="page-header">
                <h1>Şirketler</h1>
                <p>Sistemde indexlenmiş şirketler</p>
            </header>

            {loading && <div className="loading-state">Yükleniyor…</div>}
            {error && <div className="error-message">{error}</div>}

            {stats && !loading && (
                <div className="panel">
                    <h3>Mevcut Şirketler ({stats.companies?.length || 0})</h3>
                    {stats.companies && stats.companies.length > 0 ? (
                        <ul style={{ listStyle: 'none', padding: 0, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '0.75rem', marginTop: '1rem' }}>
                            {stats.companies.map(c => (
                                <li key={c} className="stat-card" style={{ textAlign: 'center', padding: '0.75rem' }}>
                                    <strong>{c}</strong>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>Henüz şirket bulunmamaktadır.</p>
                    )}
                </div>
            )}

            {breakdown.length > 0 && !loading && (
                <div className="panel" style={{ marginTop: '1.25rem' }}>
                    <h3>Şirket Veri Yoğunluğu</h3>
                    <p className="subtext" style={{ marginBottom: '1rem' }}>
                        Her şirket için indexlenen chunk sayısı ve tespit edilen çeyrek sayısı.
                    </p>
                    <div className="mini-bars">
                        {breakdown.map((row) => {
                            const maxValue = breakdown[0]?.chunks || 1;
                            const width = Math.max(6, Math.round((row.chunks / maxValue) * 100));
                            return (
                                <div key={row.company} className="mini-bar-row">
                                    <span className="mini-bar-label">{row.company}</span>
                                    <div className="mini-bar-track">
                                        <div className="mini-bar-fill" style={{ width: `${width}%` }} />
                                    </div>
                                    <span className="mini-bar-value">{row.chunks} chunk / {row.quarter_count} çeyrek</span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
