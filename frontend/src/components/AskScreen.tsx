import { useState } from 'react';
import type { AskRequest, AskResponse, ComparisonRow, EvidenceChunk } from '../api/types';
import { apiClient } from '../api/client';

interface AskScreenProps {
    settings?: {
        retriever: AskRequest['retriever'];
        mode: AskRequest['mode'];
    };
    initialCompany?: string;
    disableCompanySelect?: boolean;
}

function formatComparisonValue(row: ComparisonRow): string {
    if (typeof row.value !== 'number' || Number.isNaN(row.value)) {
        return '-';
    }

    const looksLikeRatio = String(row.target || '').toLowerCase().includes('margin');
    return looksLikeRatio
        ? `${row.value.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`
        : row.value.toLocaleString('tr-TR', { maximumFractionDigits: 2 });
}

function formatConfidence(confidence: number | null): string {
    if (typeof confidence !== 'number' || Number.isNaN(confidence)) {
        return 'M/A';
    }
    return `${(confidence * 100).toFixed(1)}%`;
}

function formatEvidencePeriod(chunk: EvidenceChunk): string {
    const parts = [chunk.year, chunk.quarter].filter((part) => part !== null && part !== undefined && String(part).trim() !== '');
    return parts.length > 0 ? parts.join(' ') : 'Dönem bilgisi yok';
}

export function AskScreen({ 
    settings, 
    initialCompany = '', 
    disableCompanySelect = false 
}: AskScreenProps) {
    const [question, setQuestion] = useState('');
    const [company, setCompany] = useState(initialCompany);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<AskResponse | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!question.trim()) return;

        setLoading(true);
        setError(null);
        try {
            const payload: AskRequest = {
                question,
                retriever: settings?.retriever || 'v5',
                mode: settings?.mode || 'single',
            };
            if (company.trim()) {
                payload.company = company.trim();
            }

            const response = await apiClient.ask(payload);
            setResult(response);
        } catch (err: any) {
            setError(err.message || 'Bir hata oluştu');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="ask-box card-glass">
            <h2>Soru Sor</h2>
            <form onSubmit={handleSubmit} className="ask-form">
                {!disableCompanySelect && (
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
                )}
                <div className="form-group">
                    <label>Sorunuz</label>
                    <textarea
                        placeholder="Sorunuzu buraya yazın..."
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        className="input-field textarea-field"
                        required
                        rows={3}
                    />
                </div>
                <button type="submit" className="primary-btn" disabled={loading}>
                    {loading ? <span className="spinner"></span> : 'Gönder'}
                </button>
            </form>

            {error && <div className="alert-error">{error}</div>}

            {result && (
                <div className="result-container fade-in">
                    <div className="answer-section">
                        <h3>Cevap</h3>

                        {result.answer.bullets.filter((bullet) => bullet.trim()).length > 0 ? (
                            <div className="answer-structured">
                                <ul className="bullets-list">
                                    {result.answer.bullets
                                        .filter((bullet) => bullet.trim())
                                        .map((bullet, idx) => (
                                            <li key={idx}>{bullet}</li>
                                        ))}
                                </ul>
                            </div>
                        ) : (
                            <p className="answer-empty">
                                {result.answer.found ? 'Yanıt üretilemedi.' : 'Dokümanda bulunamadı.'}
                            </p>
                        )}

                        <div className="meta-info">
                            <span className={`badge ${result.answer.verify_status.toLowerCase()}`}>
                                Guven: {(result.answer.confidence * 100).toFixed(1)}% | {result.answer.verify_status}
                            </span>
                            <span className="debug-badge">
                                Latans: {result.debug.latency_ms}ms | Metod: {result.debug.retriever}
                            </span>
                        </div>
                    </div>

                    {result.comparison && result.comparison.rows.length > 0 && (
                        <div className="trend-section mt-4">
                            <h3>Şirket Karşılaştırması</h3>
                            <p className="answer-main-text">
                                Hedef metrik: <strong>{result.comparison.target}</strong>
                                {result.comparison.best_company && (
                                    <>
                                        {' '}| Öne çıkan şirket: <strong>{result.comparison.best_company}</strong>
                                    </>
                                )}
                            </p>
                            <div className="table-responsive">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Şirket</th>
                                            <th>Çeyrek</th>
                                            <th>Değer</th>
                                            <th>Güven</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {result.comparison.rows.map((row, idx) => {
                                            const isBest = row.company === result.comparison?.best_company;
                                            return (
                                                <tr key={`${row.company}-${row.quarter || idx}`}>
                                                    <td>{isBest ? <strong>{row.company}</strong> : row.company}</td>
                                                    <td>{row.quarter || '-'}</td>
                                                    <td>{formatComparisonValue(row)}</td>
                                                    <td>{formatConfidence(row.confidence)}</td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {result.trend && result.trend.rows.length > 0 && (
                        <div className="trend-section mt-4">
                            <h3>Trend Verisi</h3>
                            <div className="table-responsive">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Çeyrek</th>
                                            <th>Değer</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {result.trend.rows.map((row, idx) => (
                                            <tr key={idx}>
                                                <td>{row.quarter}</td>
                                                <td>{row.value_display || row.value}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {result.evidence && result.evidence.length > 0 && (
                        <div className="evidence-section mt-4">
                            <h3>Kanıtlar</h3>
                            <div className="evidence-grid">
                                {result.evidence.map((chunk, idx) => (
                                    <div key={idx} className="evidence-card">
                                        <div className="evidence-header">
                                            <strong>{chunk.company || chunk.doc_id}</strong> - {formatEvidencePeriod(chunk)} (Sayfa: {chunk.page})
                                        </div>
                                        <div className="evidence-body">
                                            {chunk.excerpt}
                                        </div>
                                        <div className="evidence-footer">
                                            Güven: {formatConfidence(chunk.confidence)} | Bölüm: {chunk.section_title || '-'}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default AskScreen;
