import { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import type { StatsResponse } from '../api/types';

export function StatsCards() {
    const [stats, setStats] = useState<StatsResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        apiClient.stats()
            .then(data => {
                setStats(data);
                setLoading(false);
            })
            .catch(err => {
                setError(err.message || 'Stats yüklenemedi');
                setLoading(false);
            });
    }, []);

    if (loading) {
        return <div className="stats-container loading"><span className="spinner"></span> İstatistikler yükleniyor...</div>;
    }

    if (error || !stats) {
        return <div className="stats-container error">{error || 'Veri yok'}</div>;
    }

    return (
        <div className="stats-container card-glass">
            <h3>Sistem İstatistikleri</h3>
            <div className="stats-grid">
                <div className="stat-card">
                    <span className="stat-value">{stats.pdf_count}</span>
                    <span className="stat-label">PDF Dosyası</span>
                </div>
                <div className="stat-card">
                    <span className="stat-value">{stats.page_count}</span>
                    <span className="stat-label">Toplam Sayfa</span>
                </div>
                <div className="stat-card">
                    <span className="stat-value">{stats.chunk_count_v2}</span>
                    <span className="stat-label">V2 Parça (Chunk)</span>
                </div>
                <div className="stat-card">
                    <span className="stat-value">{stats.companies.length}</span>
                    <span className="stat-label">Şirket</span>
                </div>
            </div>
            {stats.companies.length > 0 && (
                <div className="companies-list mt-3">
                    <strong>Aktif Şirketler: </strong>
                    <span className="subtext">{stats.companies.join(', ')}</span>
                </div>
            )}
        </div>
    );
}
