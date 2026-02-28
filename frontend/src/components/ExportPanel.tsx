import { useState } from 'react';
import { apiClient } from '../api/client';

export function ExportPanel() {
    const [company, setCompany] = useState('');

    const handleExport = (type: 'trend' | 'ratio') => {
        const url = apiClient.exportUrl(type, company.trim() || undefined);
        // Trigger download by creating a temporary link
        const a = document.createElement('a');
        a.href = url;
        a.download = `${type}_${company || 'ALL'}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    return (
        <div className="export-panel card-glass">
            <h3>Dışa Aktar (CSV)</h3>
            <p className="subtext">Trend ve oran analizlerini CSV formatında indirin.</p>

            <div className="form-group">
                <label>Şirket Kodu (Opsiyonel)</label>
                <input
                    type="text"
                    placeholder="Örn: THYAO"
                    value={company}
                    onChange={(e) => setCompany(e.target.value)}
                    className="input-field"
                />
            </div>

            <div className="export-actions">
                <button
                    onClick={() => handleExport('trend')}
                    className="secondary-btn"
                >
                    Trend İndir (CSV)
                </button>
                <button
                    onClick={() => handleExport('ratio')}
                    className="secondary-btn"
                >
                    Oran İndir (CSV)
                </button>
            </div>
        </div>
    );
}
