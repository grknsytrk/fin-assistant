import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import type {
    KapMetricValue,
    KapQuarter,
    KapSnapshotResponse,
    KapValuationHistoryRequest,
    KapValuationHistoryResponse,
} from '../../api/types';
import { apiClient } from '../../api/client';
import { LineChartCard } from '../charts/BarChartCard';
import { _quarterKey, type SeriesPoint } from '../../utils/chartBuilders';
import './HistoricalMultiplesChart.css';

type MultipleKey = 'fk' | 'pd_satis' | 'pd_dd' | 'fd_favok';
type HistoryWindow = '5y' | '10y' | 'all';

const METRICS_TO_KEEP = [
    'net_kar',
    'satis_gelirleri',
    'favok',
    'ozkaynaklar',
    'net_borc',
    'odenmis_sermaye',
    'cikarilmis_sermaye',
] as const;

const MULTIPLE_OPTIONS: Array<{ key: MultipleKey; title: string }> = [
    { key: 'fk', title: 'F/K' },
    { key: 'pd_satis', title: 'PD/Satış' },
    { key: 'pd_dd', title: 'PD/DD' },
    { key: 'fd_favok', title: 'FD/FAVÖK' },
];

function keepMetrics(source: Record<string, KapMetricValue> | undefined): Record<string, KapMetricValue> {
    const output: Record<string, KapMetricValue> = {};
    for (const key of METRICS_TO_KEEP) {
        const metric = source?.[key];
        if (metric) output[key] = metric;
    }
    return output;
}

function compactQuarter(quarter: KapQuarter): KapQuarter {
    return {
        quarter: quarter.quarter,
        year: quarter.year,
        period: quarter.period,
        currency: quarter.currency,
        publish_date: quarter.publish_date,
        metrics: {},
        metrics_quarterly: {},
        metrics_ytd: {},
        metrics_original: keepMetrics(quarter.metrics_original || quarter.metrics),
        metrics_quarterly_original: keepMetrics(quarter.metrics_quarterly_original || quarter.metrics_quarterly),
        metrics_ytd_original: keepMetrics(quarter.metrics_ytd_original || quarter.metrics_ytd),
        metrics_comparative: {},
        metrics_quarterly_comparative: {},
        metrics_ytd_comparative: keepMetrics(quarter.metrics_ytd_comparative),
        analysis_multiplier: quarter.analysis_multiplier,
    };
}

function formatMultiple(value: number): string {
    return value.toLocaleString('tr-TR', { maximumFractionDigits: 2 }) + 'x';
}

export default function HistoricalMultiplesChart({
    snapshot,
    historyLoading = false,
    onHistoryChange,
    highlightedQuarterKey,
    onHighlightQuarterKey,
}: {
    snapshot: KapSnapshotResponse;
    historyLoading?: boolean;
    onHistoryChange?: (history: KapValuationHistoryResponse | null) => void;
    highlightedQuarterKey?: string | null;
    onHighlightQuarterKey?: (key: string | null) => void;
}) {
    const company = snapshot.stock_code || snapshot.company;
    const compactQuarters = useMemo(
        () => snapshot.quarters.map(compactQuarter),
        [snapshot.quarters],
    );
    const requestBody = useMemo(
        () => JSON.stringify({
            company,
            company_kind: snapshot.company_kind,
            quarters: compactQuarters,
        } satisfies KapValuationHistoryRequest),
        [company, compactQuarters, snapshot.company_kind],
    );
    const quarterCount = compactQuarters.length;
    const shouldRequestHistory = quarterCount >= 20 || (!historyLoading && quarterCount > 0);
    const [history, setHistory] = useState<KapValuationHistoryResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [requestError, setRequestError] = useState('');
    const [selectedWindow, setSelectedWindow] = useState<HistoryWindow>('5y');
    const [expandedMultiple, setExpandedMultiple] = useState<MultipleKey | null>(null);
    const [localHighlightedQuarterKey, setLocalHighlightedQuarterKey] = useState<string | null>(null);
    const activeHighlightedQuarterKey = highlightedQuarterKey === undefined
        ? localHighlightedQuarterKey
        : highlightedQuarterKey;
    const updateHighlightedQuarterKey = useCallback((key: string | null) => {
        setLocalHighlightedQuarterKey(key);
        onHighlightQuarterKey?.(key);
    }, [onHighlightQuarterKey]);
    const currentCompany = useRef(company);
    const closeExpandedChart = useCallback(() => {
        setExpandedMultiple(null);
        updateHighlightedQuarterKey(null);
    }, [updateHighlightedQuarterKey]);

    useEffect(() => {
        onHistoryChange?.(history);
    }, [history, onHistoryChange]);

    useEffect(() => {
        const controller = new AbortController();
        let timer: number | undefined;
        let attempts = 0;
        if (currentCompany.current !== company) {
            currentCompany.current = company;
            setHistory(null);
            setSelectedWindow('5y');
            setExpandedMultiple(null);
            updateHighlightedQuarterKey(null);
        }
        setRequestError('');
        setLoading(shouldRequestHistory);

        const loadHistory = async () => {
            if (controller.signal.aborted) return;
            try {
                const response = await apiClient.kapValuationHistory(
                    JSON.parse(requestBody) as KapValuationHistoryRequest,
                    controller.signal,
                );
                if (controller.signal.aborted) return;
                if (response.ok) {
                    setHistory(response);
                    setRequestError('');
                    setLoading(false);
                    return;
                }

                setHistory(response);
                setRequestError(response.error || 'Tarihsel çarpanlar alınamadı.');
                attempts += 1;
                setLoading(Boolean(response.pending));
                const delay = response.pending ? Math.min(15000, 1500 + attempts * 1500) : 60000;
                timer = window.setTimeout(() => {
                    if (controller.signal.aborted) return;
                    setLoading(true);
                    void loadHistory();
                }, delay);
            } catch (error) {
                if (controller.signal.aborted) return;
                setRequestError(error instanceof Error ? error.message : 'Tarihsel çarpanlar alınamadı.');
                setLoading(true);
                attempts += 1;
                timer = window.setTimeout(loadHistory, Math.min(15000, 1500 + attempts * 1500));
            }
        };

        if (shouldRequestHistory) void loadHistory();
        else setLoading(false);

        return () => {
            controller.abort();
            if (timer !== undefined) window.clearTimeout(timer);
        };
    }, [company, quarterCount, requestBody, shouldRequestHistory, updateHighlightedQuarterKey]);

    useEffect(() => {
        if (!expandedMultiple) return;
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') closeExpandedChart();
        };
        document.addEventListener('keydown', onKeyDown);
        return () => document.removeEventListener('keydown', onKeyDown);
    }, [closeExpandedChart, expandedMultiple]);

    const periods = history?.periods || [];
    const windowedPeriods = useMemo(() => {
        if (selectedWindow === '5y') return periods.slice(-20);
        if (selectedWindow === '10y') return periods.slice(-40);
        return periods;
    }, [periods, selectedWindow]);
    const seriesByMultiple = useMemo(() => {
        const groups = {} as Record<MultipleKey, SeriesPoint[]>;
        for (const { key } of MULTIPLE_OPTIONS) {
            groups[key] = windowedPeriods.flatMap((point) => {
                const value = point[key];
                if (value == null || !Number.isFinite(value) || value <= 0) return [];
                return [{
                    key: String(point.year) + '-Q' + String(point.period),
                    quarterKey: _quarterKey(point.year, point.period),
                    label: String(point.year) + 'Ç' + String(point.period),
                    value,
                    display: formatMultiple(value),
                }];
            });
        }
        return groups;
    }, [windowedPeriods]);
    const averageSeriesByMultiple = useMemo(() => {
        const averages = {} as Record<MultipleKey, SeriesPoint[]>;
        for (const { key } of MULTIPLE_OPTIONS) {
            let sum = 0;
            averages[key] = seriesByMultiple[key].map((point, index) => {
                sum += point.value;
                const value = sum / (index + 1);
                return {
                    ...point,
                    value,
                    display: formatMultiple(value),
                };
            });
        }
        return averages;
    }, [seriesByMultiple]);
    const averageByMultiple = useMemo(() => {
        const averages = {} as Record<MultipleKey, number | null>;
        for (const { key } of MULTIPLE_OPTIONS) {
            const points = averageSeriesByMultiple[key];
            averages[key] = points.length ? points[points.length - 1].value : null;
        }
        return averages;
    }, [averageSeriesByMultiple]);

    const rangeOptions: Array<{ key: HistoryWindow; label: string; available: boolean }> = [
        { key: '5y', label: '5Y', available: periods.length > 0 },
        { key: '10y', label: '10Y', available: periods.length >= 40 },
        { key: 'all', label: 'Tümü', available: periods.length > 20 },
    ];
    const hasChartData = MULTIPLE_OPTIONS.some((option) => seriesByMultiple[option.key].length > 0);
    const expandedOption = MULTIPLE_OPTIONS.find((option) => option.key === expandedMultiple);
    const expandedSeries = expandedMultiple ? seriesByMultiple[expandedMultiple] : [];
    const expandedAverageSeries = expandedMultiple ? averageSeriesByMultiple[expandedMultiple] : [];
    const expandedAverage = expandedMultiple ? averageByMultiple[expandedMultiple] : null;
    const pendingNote = requestError || (
        loading
            ? 'Tarihsel kapanış fiyatları hazırlanıyor; grafik hazır olduğunda kendiliğinden güncellenecek.'
            : ''
    );
    const backfillNote = historyLoading
        ? periods.length >= 20
            ? 'Son 5 yıl hazır; daha eski çeyrekler arka planda yükleniyor.'
            : 'Son 5 yıllık çeyrekler hazırlanıyor.'
        : '';

    const expandedChartModal = expandedOption && expandedSeries.length ? (
        <div className="kap-chart-modal-backdrop" role="presentation" onClick={closeExpandedChart}>
            <div
                className="kap-chart-modal"
                role="dialog"
                aria-modal="true"
                aria-label={`${expandedOption.title} grafiği`}
                onClick={(event) => event.stopPropagation()}
            >
                <button className="kap-chart-modal-close" type="button" onClick={closeExpandedChart} aria-label="Grafiği kapat">
                    <X size={18} strokeWidth={2.4} aria-hidden="true" />
                </button>
                <LineChartCard
                    title={expandedOption.title}
                    series={expandedSeries}
                    averageSeries={expandedAverageSeries}
                    averageLabel={expandedAverage == null ? undefined : 'Ort. ' + formatMultiple(expandedAverage)}
                    highlightedQuarterKey={activeHighlightedQuarterKey}
                    onHighlightQuarterKey={updateHighlightedQuarterKey}
                    className="kap-chart-card-expanded historical-multiples-chart"
                />
            </div>
        </div>
    ) : null;

    if (!history && !loading && !compactQuarters.length) return null;

    return (
        <section className="historical-multiples-panel">
            <div className="historical-multiples-head">
                <div>
                    <h3>Tarihsel Çarpanlar</h3>
                    <p>{history?.method_note || 'Çeyreklik F/K, PD/Satış, PD/DD ve FD/FAVÖK; rapor tarihindeki fiyat ve finansallarla hesaplanır.'}</p>
                </div>
                <div className="historical-multiples-ranges" role="group" aria-label="Çarpan geçmişi aralığı">
                    {rangeOptions.map((option) => (
                        <button
                            key={option.key}
                            type="button"
                            className={selectedWindow === option.key ? 'is-active' : ''}
                            aria-pressed={selectedWindow === option.key}
                            disabled={!option.available}
                            onClick={() => setSelectedWindow(option.key)}
                        >
                            {option.label}
                        </button>
                    ))}
                </div>
            </div>

            {backfillNote ? <p className="historical-multiples-status" role="status">{backfillNote}</p> : null}
            {pendingNote ? <p className="historical-multiples-status" role="status">{pendingNote}</p> : null}

            {hasChartData ? (
                <div className="kap-charts-grid kap-charts-grid-3 historical-multiples-grid">
                    {MULTIPLE_OPTIONS.map((option) => {
                        const series = seriesByMultiple[option.key];
                        const averageSeries = averageSeriesByMultiple[option.key];
                        const average = averageByMultiple[option.key];
                        return series.length ? (
                            <LineChartCard
                                key={option.key}
                                title={option.title}
                                series={series}
                                averageSeries={averageSeries}
                                averageLabel={average == null ? undefined : 'Ort. ' + formatMultiple(average)}
                                highlightedQuarterKey={activeHighlightedQuarterKey}
                                onHighlightQuarterKey={updateHighlightedQuarterKey}
                                onOpen={() => setExpandedMultiple(option.key)}
                                className="historical-multiples-chart"
                            />
                        ) : (
                            <div className="historical-multiples-empty" key={option.key}>
                                <h4>{option.title}</h4>
                                <p>Bu dönem için yeterli veri bulunamadı.</p>
                            </div>
                        );
                    })}
                </div>
            ) : (
                <div className="historical-multiples-empty">
                    <h4>Çarpan grafikleri hazırlanıyor</h4>
                    <p>{pendingNote || backfillNote || 'Yayımlanmış finansal dönem ve fiyat verisi bekleniyor.'}</p>
                </div>
            )}

            {hasChartData ? (
                <p className="historical-multiples-footnote">
                    Ortalama çizgisi her çeyrekte seçili aralığın başlangıcından o döneme kadar hesaplanır; son noktası seçili aralığın tamamının ortalamasıdır.
                </p>
            ) : null}

            {history?.price_cache_stale ? (
                <p className="historical-multiples-footnote">Önbellekteki tarihsel kapanışlar gösteriliyor; fiyat serisi arka planda yenileniyor.</p>
            ) : history?.price_as_of ? (
                <p className="historical-multiples-footnote">
                    Fiyat verisi {new Date(history.price_as_of + 'T12:00:00').toLocaleDateString('tr-TR')} tarihine kadar.
                    {periods.length ? ' ' + windowedPeriods.length + ' dönem gösteriliyor.' : ''}
                </p>
            ) : null}
            {expandedChartModal ? createPortal(expandedChartModal, document.body) : null}
        </section>
    );
}
