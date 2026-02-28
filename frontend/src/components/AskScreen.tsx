import React, { useState } from 'react';
import type { AskRequest, AskResponse } from '../api/types';
import { apiClient } from '../api/client';

interface AskScreenProps {
    settings: {
        retriever: AskRequest['retriever'];
        mode: AskRequest['mode'];
    };
}

export function AskScreen({ settings }: AskScreenProps) {
    const [question, setQuestion] = useState('');
    const [company, setCompany] = useState('');
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
                retriever: settings.retriever,
                mode: settings.mode,
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
                        {result.answer.bullets.length > 0 ? (
                            <ul className="bullets-list">
                                {result.answer.bullets.map((bullet, idx) => (
                                    <li key={idx}>{bullet}</li>
                                ))}
                            </ul>
                        ) : (
                            <p>Bulunamadı.</p>
                        )}

                        <div className="meta-info">
                            <span className={`badge ${result.answer.verify_status.toLowerCase()}`}>
                                Güven: {(result.answer.confidence * 100).toFixed(1)}% | {result.answer.verify_status}
                            </span>
                            <span className="debug-badge">
                                Latans: {result.debug.latency_ms}ms | Metod: {result.debug.retriever}
                            </span>
                        </div>
                    </div>

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
                                            <strong>{chunk.company}</strong> - {chunk.year} {chunk.quarter} (Sayfa: {chunk.page})
                                        </div>
                                        <div className="evidence-body">
                                            {chunk.excerpt}
                                        </div>
                                        <div className="evidence-footer">
                                            Güven: {chunk.confidence ? (chunk.confidence * 100).toFixed(1) : 'M/A'}%
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
