import { useEffect, useMemo, useRef, useState } from 'react';
import type { KapSnapshotResponse, KapQuarter } from '../../../api/types';
import type { KapOverviewCommentaryResponse } from '../../../api/types';
import { apiClient } from '../../../api/client';
import { MultiplesRow } from '../../../components/stock/MultiplesRow';
import {
    _resolveMetricValueByPriority, _resolveMetricValue, _resolveMetricDisplayByPriority,
    _resolveMetricDisplay, _calcPctChange, _pctClass, _pctText, intSafe, _periodLabel,
} from '../../../utils/formatters';
import {
    buildOverviewAiPayload,
    buildOverviewHistoryContext,
    getOverviewSummaryRows,
    latestOverviewPeriod,
    overviewPayloadHash,
} from '../../../utils/overviewPayload';
import StockCharts from './StockCharts';
import './StockOverview.css';

const OVERVIEW_AI_MODEL_STORAGE_KEY = 'kapOverviewAi::selectedModel';
const OVERVIEW_AI_MODELS = [
    { id: 'minimaxai/minimax-m2.7', label: 'MiniMax M2.7' },
    { id: 'meta/llama-4-maverick-17b-128e-instruct', label: 'Llama 4 Maverick 17B' },
];
const DEFAULT_OVERVIEW_AI_MODEL = OVERVIEW_AI_MODELS[0].id;
const OVERVIEW_AI_LOADING_STEPS = [
    {
        title: 'Tarihsel baz kuruluyor',
        detail: 'Son 12 çeyrek içinden mevsimsel desen ve yakın trend eşleştiriliyor.',
    },
    {
        title: 'Base skor hesaplanıyor',
        detail: 'Deterministic büyüme, karlılık, bilanço ve nakit akışı puanları çıkarılıyor.',
    },
    {
        title: 'AI ayarlamaları yazılıyor',
        detail: 'Model yalnız sınırlı adjustment ve kısa finansal gerekçeler üretiyor.',
    },
];

function scoreSourceLabel(source: KapOverviewCommentaryResponse['scorecard']['score_source']) {
    if (source === 'ai_adjusted') return 'AI düzeltmeli skor';
    if (source === 'ai_failed_fallback') return 'Deterministic fallback';
    return 'Deterministic skor';
}

export default function StockOverview({ snapshot, quarters }: { snapshot: KapSnapshotResponse, quarters: KapQuarter[] }) {
    const latestQuarterIdx = quarters.length ? quarters.length - 1 : -1;
    const latestQuarter = latestQuarterIdx >= 0 ? quarters[latestQuarterIdx] : null;
    const prevQuarterIdx = latestQuarterIdx > 0 ? latestQuarterIdx - 1 : -1;
    const prevQuarter = prevQuarterIdx >= 0 ? quarters[prevQuarterIdx] : null;

    const prevYearSameQuarterIdx = latestQuarter
        ? quarters.findIndex(
            (q) =>
                intSafe(q.year) === intSafe(latestQuarter.year) - 1 &&
                intSafe(q.period) === intSafe(latestQuarter.period),
        )
        : -1;
    const prevYearSameQuarter = prevYearSameQuarterIdx >= 0 ? quarters[prevYearSameQuarterIdx] : null;

    const { incomeSummaryRows, balanceSummaryRows } = useMemo(
        () => getOverviewSummaryRows(snapshot, quarters),
        [snapshot, quarters],
    );
    const overviewAiPayload = useMemo(() => buildOverviewAiPayload(snapshot, quarters), [snapshot, quarters]);
    const overviewHistoryContext = useMemo(() => buildOverviewHistoryContext(snapshot, quarters), [snapshot, quarters]);
    const latestPeriod = useMemo(() => latestOverviewPeriod(quarters), [quarters]);
    const payloadHash = useMemo(() => overviewPayloadHash(overviewAiPayload), [overviewAiPayload]);
    const historyHash = useMemo(() => overviewPayloadHash(overviewHistoryContext), [overviewHistoryContext]);
    const [selectedModel, setSelectedModel] = useState(() => {
        if (typeof window === 'undefined') {
            return DEFAULT_OVERVIEW_AI_MODEL;
        }
        const saved = window.localStorage.getItem(OVERVIEW_AI_MODEL_STORAGE_KEY) || '';
        return OVERVIEW_AI_MODELS.some((item) => item.id === saved) ? saved : DEFAULT_OVERVIEW_AI_MODEL;
    });
    const commentaryCacheKey = useMemo(
        () => `kapOverviewAi::${snapshot.stock_code || snapshot.company}::${latestPeriod || 'unknown'}::${selectedModel}::${payloadHash}::${historyHash}`,
        [snapshot.stock_code, snapshot.company, latestPeriod, selectedModel, payloadHash, historyHash],
    );
    const [aiLoading, setAiLoading] = useState(false);
    const [aiLoadingStep, setAiLoadingStep] = useState(0);
    const [aiError, setAiError] = useState('');
    const [aiCommentary, setAiCommentary] = useState<KapOverviewCommentaryResponse | null>(null);
    const aiAbortRef = useRef<AbortController | null>(null);
    const aiRequestIdRef = useRef(0);
    const canRequestAi =
        Boolean(latestQuarter) &&
        overviewHistoryContext.quarters.length > 0 &&
        (overviewAiPayload.income_summary.length > 0 ||
            overviewAiPayload.balance_summary.length > 0 ||
            overviewAiPayload.charts.length > 0);
    const selectedModelLabel =
        OVERVIEW_AI_MODELS.find((item) => item.id === selectedModel)?.label || selectedModel;

    useEffect(() => {
        aiRequestIdRef.current += 1;
        aiAbortRef.current?.abort();
        aiAbortRef.current = null;
        setAiLoading(false);
        setAiError('');
        setAiCommentary(null);
        try {
            const cached = window.sessionStorage.getItem(commentaryCacheKey);
            if (cached) {
                setAiCommentary(JSON.parse(cached) as KapOverviewCommentaryResponse);
            }
        } catch {
            setAiCommentary(null);
        }
    }, [commentaryCacheKey]);

    useEffect(() => {
        return () => {
            aiAbortRef.current?.abort();
            aiAbortRef.current = null;
        };
    }, []);

    useEffect(() => {
        window.localStorage.setItem(OVERVIEW_AI_MODEL_STORAGE_KEY, selectedModel);
    }, [selectedModel]);

    useEffect(() => {
        if (!aiLoading) {
            setAiLoadingStep(0);
            return;
        }
        const timer = window.setInterval(() => {
            setAiLoadingStep((previous) => (previous + 1) % OVERVIEW_AI_LOADING_STEPS.length);
        }, 2200);
        return () => window.clearInterval(timer);
    }, [aiLoading]);

    const requestAiCommentary = async (forceRefresh = false) => {
        if (!canRequestAi || aiLoading) return;
        let requestController: AbortController | null = null;
        let requestId = aiRequestIdRef.current;
        setAiLoading(true);
        setAiError('');
        console.debug('[overview-ai] request started', {
            company: snapshot.stock_code || snapshot.company,
            latestPeriod,
            model: selectedModel,
            forceRefresh,
            incomeRows: overviewAiPayload.income_summary.length,
            balanceRows: overviewAiPayload.balance_summary.length,
            charts: overviewAiPayload.charts.length,
            historyQuarters: overviewHistoryContext.quarters.length,
            cacheKey: commentaryCacheKey,
        });
        try {
            if (!forceRefresh) {
                const cached = window.sessionStorage.getItem(commentaryCacheKey);
                if (cached) {
                    console.debug('[overview-ai] cached response used', {
                        company: snapshot.stock_code || snapshot.company,
                        cacheKey: commentaryCacheKey,
                    });
                    setAiCommentary(JSON.parse(cached) as KapOverviewCommentaryResponse);
                    return;
                }
            }
            aiAbortRef.current?.abort();
            const controller = new AbortController();
            requestController = controller;
            aiAbortRef.current = controller;
            requestId = aiRequestIdRef.current + 1;
            aiRequestIdRef.current = requestId;
            const response = await apiClient.kapOverviewCommentary({
                company: snapshot.stock_code || snapshot.company,
                company_title: snapshot.company_title || snapshot.company,
                latest_period: latestPeriod,
                model: selectedModel,
                history_context: overviewHistoryContext,
                overview_payload: overviewAiPayload,
            }, {
                signal: controller.signal,
            });
            if (controller.signal.aborted || requestId !== aiRequestIdRef.current) {
                return;
            }
            console.debug('[overview-ai] response received', {
                company: snapshot.stock_code || snapshot.company,
                ok: response.ok,
                model: response.model_used,
                error: response.error,
                scoreSource: response.scorecard?.score_source,
                debugTraceCount: response.debug_trace?.length ?? 0,
            });
            setAiCommentary(response);
            if (response.ok) {
                window.sessionStorage.setItem(commentaryCacheKey, JSON.stringify(response));
            }
        } catch (error) {
            if ((error as Error)?.name === 'AbortError') {
                return;
            }
            console.error('[overview-ai] request failed', {
                company: snapshot.stock_code || snapshot.company,
                error,
            });
            setAiError(error instanceof Error ? error.message : 'AI yorumu üretilemedi.');
        } finally {
            if (!requestController) {
                setAiLoading(false);
            } else if (aiAbortRef.current === requestController) {
                aiAbortRef.current = null;
                if (!requestController.signal.aborted && requestId === aiRequestIdRef.current) {
                    setAiLoading(false);
                }
            }
        }
    };

    return (
        <div className="section-overview fade-in">
            <MultiplesRow snapshot={snapshot} quarters={quarters} />

            {latestQuarter && (
                <div className="kap-summary-panel panel">
                    <div className="kap-summary-head">
                        <h3>Özet Finansallar</h3>
                        {snapshot.analysis_note ? (
                            <p className="kap-analysis-note">{snapshot.analysis_note}</p>
                        ) : null}
                    </div>

                    <div className="kap-summary-grid">
                        <div className="kap-summary-table-wrap">
                            <h4>
                                Özet Gelir Tablosu <small>{_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)}</small>
                            </h4>
                            <table className="kap-summary-table">
                                <thead>
                                    <tr>
                                        <th>Kalem</th>
                                        <th>{_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)}</th>
                                        <th>
                                            {prevYearSameQuarter
                                                ? _periodLabel(intSafe(prevYearSameQuarter.year), intSafe(prevYearSameQuarter.period), prevYearSameQuarter.quarter)
                                                : '-'}
                                        </th>
                                        <th>%</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {incomeSummaryRows.map((row) => {
                                        const currentValue = _resolveMetricValueByPriority(
                                            latestQuarter,
                                            row.key,
                                            ['metrics_ytd', 'metrics', 'metrics_quarterly'],
                                        );
                                        const baseValue =
                                            prevYearSameQuarter
                                                ? _resolveMetricValueByPriority(
                                                    prevYearSameQuarter,
                                                    row.key,
                                                    ['metrics_ytd', 'metrics', 'metrics_quarterly'],
                                                )
                                                : null;
                                        if (currentValue === null && baseValue === null) return null;
                                        const pct = _calcPctChange(currentValue, baseValue);
                                        return (
                                            <tr key={`income-${row.key}`}>
                                                <td>{row.label}</td>
                                                <td>
                                                    {_resolveMetricDisplayByPriority(
                                                        latestQuarter,
                                                        row.key,
                                                        ['metrics_ytd', 'metrics', 'metrics_quarterly'],
                                                    )}
                                                </td>
                                                <td>
                                                    {prevYearSameQuarter
                                                        ? _resolveMetricDisplayByPriority(
                                                            prevYearSameQuarter,
                                                            row.key,
                                                            ['metrics_ytd', 'metrics', 'metrics_quarterly'],
                                                        )
                                                        : '-'}
                                                </td>
                                                <td className={`kap-summary-pct-cell ${_pctClass(pct, row.key)}`}>{_pctText(pct)}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>

                        <div className="kap-summary-table-wrap">
                            <h4>
                                Özet Bilanço <small>{_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)}</small>
                            </h4>
                            <table className="kap-summary-table">
                                <thead>
                                    <tr>
                                        <th>Kalem</th>
                                        <th>{_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)}</th>
                                        <th>
                                            {prevQuarter
                                                ? _periodLabel(intSafe(prevQuarter.year), intSafe(prevQuarter.period), prevQuarter.quarter)
                                                : '-'}
                                        </th>
                                        <th>%</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {balanceSummaryRows.map((row) => {
                                        const currentValue = _resolveMetricValue(quarters, latestQuarterIdx, row.key, false);
                                        const baseValue =
                                            prevQuarterIdx >= 0
                                                ? _resolveMetricValue(quarters, prevQuarterIdx, row.key, false)
                                                : null;
                                        if (currentValue === null && baseValue === null) return null;
                                        const pct = _calcPctChange(currentValue, baseValue);
                                        return (
                                            <tr key={`balance-${row.key}`}>
                                                <td>{row.label}</td>
                                                <td>{_resolveMetricDisplay(quarters, latestQuarterIdx, row.key, false)}</td>
                                                <td>
                                                    {prevQuarterIdx >= 0
                                                        ? _resolveMetricDisplay(quarters, prevQuarterIdx, row.key, false)
                                                        : '-'}
                                                </td>
                                                <td className={`kap-summary-pct-cell ${_pctClass(pct, row.key)}`}>{_pctText(pct)}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {quarters.length > 0 && (
                <section className="overview-charts-shell">
                    <div className="overview-charts-head">
                        <h3>Grafikler ve Analiz</h3>
                        <p>İlk olarak çeyreklik bar grafikler, ardından marj ve oran trendleri.</p>
                    </div>
                    <StockCharts snapshot={snapshot} quarters={quarters} embedded />
                </section>
            )}

            {canRequestAi && (
                <section className="overview-ai-panel panel">
                    <div className="overview-ai-head">
                        <div>
                            <h3>AI Finansal Yorum</h3>
                            <p>Bu yorum yalnızca ekranda görülen finansal verilerden otomatik üretilmiştir.</p>
                        </div>
                        <div className="overview-ai-controls">
                            <label className="overview-ai-model-picker">
                                <span>Model</span>
                                <select
                                    value={selectedModel}
                                    onChange={(event) => setSelectedModel(event.target.value)}
                                    disabled={aiLoading}
                                >
                                    {OVERVIEW_AI_MODELS.map((option) => (
                                        <option key={option.id} value={option.id}>
                                            {option.label}
                                        </option>
                                    ))}
                                </select>
                            </label>
                            <button
                                className="overview-ai-button"
                                type="button"
                                onClick={() => void requestAiCommentary(Boolean(aiCommentary?.ok))}
                                disabled={aiLoading}
                            >
                                {aiLoading ? 'Analiz hazırlanıyor...' : aiCommentary?.ok ? 'Analizi Yenile' : 'AI Analiz ve Puan'}
                            </button>
                        </div>
                    </div>

                    {aiLoading ? (
                        <div className={`overview-ai-loading ${aiCommentary?.ok ? 'is-refreshing' : ''}`} aria-live="polite">
                            <div className="overview-ai-loading-orb" aria-hidden="true">
                                <span />
                                <span />
                                <span />
                            </div>
                            <div className="overview-ai-loading-copy">
                                <strong>{OVERVIEW_AI_LOADING_STEPS[aiLoadingStep].title}</strong>
                                <p>{OVERVIEW_AI_LOADING_STEPS[aiLoadingStep].detail}</p>
                                <small>{selectedModelLabel} ile skor ve yorum birlikte hazırlanıyor.</small>
                            </div>
                            {!aiCommentary?.ok ? (
                                <div className="overview-ai-loading-skeleton" aria-hidden="true">
                                    <span className="overview-ai-loading-line overview-ai-loading-line-wide" />
                                    <span className="overview-ai-loading-line" />
                                    <span className="overview-ai-loading-line overview-ai-loading-line-soft" />
                                </div>
                            ) : null}
                        </div>
                    ) : null}

                    {aiError ? (
                        <>
                            <div className="overview-ai-error">
                                <span>{aiError}</span>
                                <button type="button" onClick={() => void requestAiCommentary(true)} disabled={aiLoading}>
                                    Tekrar dene
                                </button>
                            </div>
                            {aiCommentary?.debug_trace?.length ? (
                                <details className="overview-ai-debug">
                                    <summary>Debug detayi</summary>
                                    <pre>{aiCommentary.debug_trace.join('\n')}</pre>
                                </details>
                            ) : null}
                        </>
                    ) : null}

                    {aiCommentary?.ok ? (
                        <div className="overview-ai-body">
                            <div className="overview-ai-scorecard">
                                <div className="overview-ai-scorecard-main">
                                    <div className="overview-ai-score-badge">
                                        <strong>{aiCommentary.scorecard.overall_score.toFixed(1)}</strong>
                                        <span>/10</span>
                                    </div>
                                    <div className="overview-ai-score-copy">
                                        <div className="overview-ai-score-meta">
                                            <h4>{aiCommentary.scorecard.overall_label}</h4>
                                            <small>{scoreSourceLabel(aiCommentary.scorecard.score_source)}</small>
                                        </div>
                                        <p>{aiCommentary.scorecard.summary}</p>
                                        <p className="overview-ai-seasonality">{aiCommentary.scorecard.seasonality_note}</p>
                                    </div>
                                </div>
                                <div className="overview-ai-score-grid">
                                    {aiCommentary.scorecard.subscores.map((item) => (
                                        <article key={item.key} className="overview-ai-score-item">
                                            <div className="overview-ai-score-item-head">
                                                <span>{item.label}</span>
                                                <strong>{item.score.toFixed(1)}</strong>
                                            </div>
                                            <p>{item.summary}</p>
                                        </article>
                                    ))}
                                </div>
                            </div>
                            {aiCommentary.error ? (
                                <div className="overview-ai-warning">
                                    <span>{aiCommentary.error}</span>
                                </div>
                            ) : null}
                            {aiCommentary.headline ? <h4>{aiCommentary.headline}</h4> : null}
                            {aiCommentary.bullets.length > 0 ? (
                                <ul>
                                    {aiCommentary.bullets.map((item, idx) => (
                                        <li key={`${idx}-${item}`}>{item}</li>
                                    ))}
                                </ul>
                            ) : null}
                            {aiCommentary.risk_note ? (
                                <p className="overview-ai-risk">
                                    <strong>Risk notu:</strong> {aiCommentary.risk_note}
                                </p>
                            ) : null}
                            {aiCommentary.watch_metrics.length > 0 ? (
                                <div className="overview-ai-watch">
                                    <span>İzlenecek metrikler</span>
                                    <div>
                                        {aiCommentary.watch_metrics.map((metric) => (
                                            <small key={metric}>{metric}</small>
                                        ))}
                                    </div>
                                </div>
                            ) : null}
                            {aiCommentary.model_used ? (
                                <p className="overview-ai-model">Model: {aiCommentary.model_used}</p>
                            ) : null}
                        </div>
                    ) : null}
                </section>
            )}
        </div>
    );
}
