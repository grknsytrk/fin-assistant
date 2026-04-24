import { useState, useEffect } from 'react';
import type { KapQuarter } from '../../../api/types';
import { _periodLabel, intSafe } from '../../../utils/formatters';

export default function StockKAP({ ticker, quarters }: { ticker: string, quarters: KapQuarter[] }) {
    // Gelecekte buraya KAP haber akışını ekleyebiliriz.
    // Şimdilik sadece yayınlanmış bilançoların özet tablosunu sunuyoruz.
    
    const [sortedQuarters, setSortedQuarters] = useState<KapQuarter[]>([]);

    useEffect(() => {
        const sorted = [...quarters].sort((a, b) => {
            const dateA = a.publish_date ? new Date(a.publish_date).getTime() : 0;
            const dateB = b.publish_date ? new Date(b.publish_date).getTime() : 0;
            return dateB - dateA;
        });
        setSortedQuarters(sorted);
    }, [quarters]);

    return (
        <div className="section-kap fade-in">
            <div className="panel">
                <div className="panel-header">
                    <h3>{ticker} KAP Finansal Rapor Bildirimleri</h3>
                </div>
                {sortedQuarters.length > 0 ? (
                    <div className="coverage-list">
                        {sortedQuarters.map((q, i) => (
                            <div key={i} className="coverage-item" style={{ cursor: 'default' }}>
                                <span className="ci-company" style={{ width: '80px' }}>
                                    {_periodLabel(intSafe(q.year), intSafe(q.period), q.quarter)}
                                </span>
                                <div className="ci-bar-container" style={{ background: 'transparent' }}>
                                    <span style={{ fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                                        Finansal Rapor Yayınlandı
                                    </span>
                                </div>
                                <span className="ci-value" style={{ minWidth: '120px' }}>
                                    {q.publish_date ? new Date(q.publish_date).toLocaleDateString('tr-TR') : '-'}
                                </span>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="kap-empty">Bildirim bulunamadı.</div>
                )}
            </div>
        </div>
    );
}
