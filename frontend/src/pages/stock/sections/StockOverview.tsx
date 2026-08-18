import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import type { KapSnapshotResponse, KapQuarter, KapInsurancePremiumDisclosure } from '../../../api/types';
import type { KapOverviewCommentaryResponse } from '../../../api/types';
import { apiClient } from '../../../api/client';
import { BarChartCard } from '../../../components/charts/BarChartCard';
import { MultiplesRow } from '../../../components/stock/MultiplesRow';
import {
    _resolveMetricValue, _resolveMetricDisplay, _resolveMetricYtdValue, _resolveMetricYtdDisplay,
    _calcPctChange, _pctClass, _pctText, intSafe, _periodLabel, _formatMetric,
} from '../../../utils/formatters';
import type { SeriesPoint } from '../../../utils/chartBuilders';
import {
    buildOverviewAiPayload,
    buildOverviewHistoryContext,
    getOverviewSummaryRows,
    latestOverviewPeriod,
    overviewPayloadHash,
    type OverviewChartGroup,
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

const CHART_QUARTER_WINDOW_STORAGE_KEY = 'ragfin_chart_quarter_window';
const QUARTER_WINDOW_OPTIONS = [5, 10, 15, 20] as const;
const DEFAULT_CHART_QUARTER_WINDOW = 20;
const PREMIUM_TABLE_ROW_LIMIT = 12;
const MONTH_LABELS_TR = [
    'Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran',
    'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık',
];
const MONTH_SHORT_LABELS_TR = [
    'Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz',
    'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara',
];
const PREMIUM_YEAR_COLORS = ['#63b39a', '#e4835d', '#8190b4', '#a86bd9'] as const;

type PremiumSeasonalObservation = {
    year: number;
    month: number;
    value: number;
    display: string;
    priority: number;
};

type PremiumSeasonalGrowthPoint = {
    year: number;
    month: number;
    value: number;
    display: string;
    periodLabel: string;
};

type PremiumSeasonalChartData = {
    months: number[];
    years: number[];
    valuesByKey: Record<string, PremiumSeasonalObservation>;
    rollingAverageByMonth: Record<number, { value: number; display: string }>;
    yoyByMonth: Record<number, PremiumSeasonalGrowthPoint>;
};

type PremiumSeasonalTooltip = {
    x: number;
    y: number;
    label: string;
    value: string;
};

function readStoredChartQuarterWindow(): number {
    if (typeof window === 'undefined') return DEFAULT_CHART_QUARTER_WINDOW;
    try {
        const raw = window.localStorage.getItem(CHART_QUARTER_WINDOW_STORAGE_KEY);
        if (raw) {
            const parsed = Number(raw);
            if ((QUARTER_WINDOW_OPTIONS as readonly number[]).includes(parsed)) return parsed;
        }
    } catch { /* ignore */ }
    return DEFAULT_CHART_QUARTER_WINDOW;
}

function scoreSourceLabel(source: KapOverviewCommentaryResponse['scorecard']['score_source']) {
    if (source === 'ai_adjusted') return 'AI düzeltmeli skor';
    if (source === 'ai_failed_fallback') return 'Deterministic fallback';
    return 'Deterministic skor';
}

function previousQuarterWindowOption(option: number): number {
    const idx = (QUARTER_WINDOW_OPTIONS as readonly number[]).indexOf(option);
    if (idx <= 0) return 0;
    return QUARTER_WINDOW_OPTIONS[idx - 1];
}

function isQuarterWindowOptionUseful(option: number, availableQuarterCount: number): boolean {
    return availableQuarterCount > previousQuarterWindowOption(option);
}

function displayQuarterWindowOption(selectedWindow: number, availableQuarterCount: number): number {
    const effectiveWindow = Math.max(1, Math.min(selectedWindow, availableQuarterCount || selectedWindow));
    return (
        QUARTER_WINDOW_OPTIONS.find(
            (option) => effectiveWindow <= option && isQuarterWindowOptionUseful(option, availableQuarterCount),
        ) || QUARTER_WINDOW_OPTIONS[0]
    );
}

function premiumPeriodLabel(row: KapInsurancePremiumDisclosure): string {
    const month = intSafe(row.month);
    const year = intSafe(row.year);
    if (month >= 1 && month <= 12 && year > 0) return `${MONTH_LABELS_TR[month - 1]} ${year}`;
    return row.period_label || '-';
}

function premiumChartPeriodLabel(row: KapInsurancePremiumDisclosure): string {
    const month = intSafe(row.month);
    const year = intSafe(row.year);
    if (month >= 1 && month <= 12 && year > 0) return `${MONTH_SHORT_LABELS_TR[month - 1]} ${String(year).slice(-2)}`;
    return row.period_label || '-';
}

function premiumDisplay(display: string | null | undefined, value: number | null | undefined): string {
    if (display && display !== '-') return display;
    if (typeof value === 'number' && Number.isFinite(value)) return _formatMetric(value, 'TL');
    return '-';
}

function premiumPctDisplay(display: string | null | undefined, value: number | null | undefined): string {
    if (display && display !== '-') return display;
    if (typeof value === 'number' && Number.isFinite(value)) return _pctText(value);
    return '-';
}

function buildMonthlyPremiumSeries(rows: KapInsurancePremiumDisclosure[]): SeriesPoint[] {
    return rows
        .map((row) => {
            const value = row.monthly_gross_premium;
            if (typeof value !== 'number' || Number.isNaN(value) || !Number.isFinite(value)) return null;
            return {
                key: `${row.year}-${row.month}-${row.disclosure_index || 0}-monthly-premium`,
                label: premiumChartPeriodLabel(row),
                value,
                display: premiumDisplay(row.monthly_gross_premium_display, value),
            };
        })
        .filter((row): row is SeriesPoint => row !== null);
}

function buildQuarterlyPremiumSeries(rows: KapInsurancePremiumDisclosure[]): SeriesPoint[] {
    const byQuarter = new Map<string, { year: number; quarter: number; value: number }>();
    rows.forEach((row) => {
        const year = intSafe(row.year);
        const month = intSafe(row.month);
        const value = row.monthly_gross_premium;
        if (year <= 0 || month < 1 || month > 12 || typeof value !== 'number' || !Number.isFinite(value)) {
            return;
        }
        const quarter = Math.ceil(month / 3);
        const key = `${year}-${quarter}`;
        const current = byQuarter.get(key);
        if (current) {
            current.value += value;
        } else {
            byQuarter.set(key, { year, quarter, value });
        }
    });

    return [...byQuarter.values()]
        .sort((a, b) => (a.year * 10 + a.quarter) - (b.year * 10 + b.quarter))
        .map((item) => ({
            key: `${item.year}-Q${item.quarter}-quarterly-premium`,
            label: `${item.year} Ç${item.quarter}`,
            value: item.value,
            display: _formatMetric(item.value, 'TL'),
        }));
}

function premiumObservationKey(year: number, month: number): string {
    return `${year}-${month}`;
}

function addPremiumSeasonalObservation(
    observations: Map<string, PremiumSeasonalObservation>,
    observation: PremiumSeasonalObservation,
): void {
    if (observation.month < 1 || observation.month > 12 || observation.year <= 0) return;
    const key = premiumObservationKey(observation.year, observation.month);
    const existing = observations.get(key);
    if (!existing || observation.priority >= existing.priority) {
        observations.set(key, observation);
    }
}

function buildPremiumSeasonalChartData(rows: KapInsurancePremiumDisclosure[]): PremiumSeasonalChartData | null {
    const observations = new Map<string, PremiumSeasonalObservation>();
    const rollingValues: number[] = [];
    const rollingAverageByMonth: PremiumSeasonalChartData['rollingAverageByMonth'] = {};
    const yoyByMonth: PremiumSeasonalChartData['yoyByMonth'] = {};

    rows.forEach((row) => {
        const year = intSafe(row.year);
        const month = intSafe(row.month);
        const currentValue = row.monthly_gross_premium;
        if (typeof currentValue === 'number' && Number.isFinite(currentValue)) {
            addPremiumSeasonalObservation(observations, {
                year,
                month,
                value: currentValue,
                display: premiumDisplay(row.monthly_gross_premium_display, currentValue),
                priority: 2,
            });
            rollingValues.push(currentValue);
            const lastThree = rollingValues.slice(-3);
            const rollingAverage = lastThree.reduce((sum, value) => sum + value, 0) / lastThree.length;
            rollingAverageByMonth[month] = {
                value: rollingAverage,
                display: _formatMetric(rollingAverage, 'TL'),
            };
        }

        const yoyValue = row.monthly_yoy_pct;
        if (typeof yoyValue === 'number' && Number.isFinite(yoyValue) && month >= 1 && month <= 12 && year > 0) {
            const existing = yoyByMonth[month];
            if (!existing || year >= existing.year) {
                yoyByMonth[month] = {
                    year,
                    month,
                    value: yoyValue,
                    display: premiumPctDisplay(row.monthly_yoy_pct_display, yoyValue),
                    periodLabel: premiumPeriodLabel(row),
                };
            }
        }

        const previousValue = row.previous_year_monthly_gross_premium;
        if (typeof previousValue === 'number' && Number.isFinite(previousValue)) {
            addPremiumSeasonalObservation(observations, {
                year: year - 1,
                month,
                value: previousValue,
                display: premiumDisplay(row.previous_year_monthly_gross_premium_display, previousValue),
                priority: 1,
            });
        }
    });

    const allObservations = [...observations.values()];
    const latestYear = rows.reduce((maxYear, row) => Math.max(maxYear, intSafe(row.year)), 0);
    const years = latestYear > 0
        ? [latestYear - 2, latestYear - 1, latestYear]
        : [...new Set(allObservations.map((item) => item.year))]
            .filter((year) => year > 0)
            .sort((a, b) => a - b)
            .slice(-3);
    const yearSet = new Set(years);
    const months = Array.from({ length: 12 }, (_, idx) => idx + 1);
    const valuesByKey: PremiumSeasonalChartData['valuesByKey'] = {};
    observations.forEach((observation) => {
        if (yearSet.has(observation.year)) {
            valuesByKey[premiumObservationKey(observation.year, observation.month)] = observation;
        }
    });

    if (!months.length || !years.length) return null;
    return { months, years, valuesByKey, rollingAverageByMonth, yoyByMonth };
}

function formatPremiumAxis(value: number): string {
    if (Math.abs(value) >= 1_000_000_000) {
        return `${(value / 1_000_000_000).toLocaleString('tr-TR', { maximumFractionDigits: 1 })} Mr`;
    }
    return `${(value / 1_000_000).toLocaleString('tr-TR', { maximumFractionDigits: 0 })} Mn`;
}

function formatPremiumPctAxis(value: number): string {
    return `% ${Math.round(value)}`;
}

function buildPremiumPctTicks(values: number[]): { min: number; max: number; ticks: number[] } | null {
    if (!values.length) return null;
    const rawMin = Math.min(0, ...values);
    const rawMax = Math.max(0, ...values);
    const padding = Math.max(5, (rawMax - rawMin) * 0.12);
    let min = Math.floor((rawMin - padding) / 10) * 10;
    let max = Math.ceil((rawMax + padding) / 10) * 10;
    if (min === max) {
        min -= 10;
        max += 10;
    }
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => min + (max - min) * ratio);
    return { min, max, ticks };
}

function premiumPctClass(value: number | null | undefined): string {
    if (typeof value !== 'number' || Number.isNaN(value)) return 'pct-neutral';
    if (value > 0) return 'pct-positive';
    if (value < 0) return 'pct-negative';
    return 'pct-neutral';
}

function PremiumSeasonalMonthlyChart({ data }: { data: PremiumSeasonalChartData }) {
    const cardRef = useRef<HTMLDivElement | null>(null);
    const [tooltip, setTooltip] = useState<PremiumSeasonalTooltip | null>(null);

    const width = 760;
    const height = 320;
    const padLeft = 52;
    const padRight = 62;
    const padTop = 28;
    const padBottom = 64;
    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;
    const allValues = [
        ...Object.values(data.valuesByKey).map((item) => item.value),
        ...Object.values(data.rollingAverageByMonth).map((item) => item.value),
    ];
    const maxValue = Math.max(...allValues, 1);
    const axisMax = Math.ceil((maxValue * 1.12) / 1_000_000_000) * 1_000_000_000;
    const y = (value: number) => padTop + (1 - value / axisMax) * plotHeight;
    const slotWidth = plotWidth / data.months.length;
    const barGap = 3;
    const groupWidth = Math.min(slotWidth * 0.76, 58);
    const barWidth = Math.max(5, (groupWidth - barGap * Math.max(0, data.years.length - 1)) / data.years.length);
    const yTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => axisMax * ratio);
    const averagePoints = data.months
        .map((month, idx) => {
            const average = data.rollingAverageByMonth[month];
            if (!average) return null;
            return {
                month,
                x: padLeft + slotWidth * idx + slotWidth / 2,
                y: y(average.value),
                value: average.value,
                display: average.display,
            };
        })
        .filter((point): point is { month: number; x: number; y: number; value: number; display: string } => point !== null);
    const averagePath = averagePoints
        .map((point, idx) => `${idx === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
        .join(' ');
    const yoyPoints = data.months
        .map((month, idx) => {
            const point = data.yoyByMonth[month];
            if (!point) return null;
            return {
                ...point,
                x: padLeft + slotWidth * idx + slotWidth / 2,
            };
        })
        .filter((point): point is PremiumSeasonalGrowthPoint & { x: number } => point !== null);
    const pctAxis = buildPremiumPctTicks(yoyPoints.map((point) => point.value));
    const pctY = (value: number) => {
        if (!pctAxis) return padTop + plotHeight / 2;
        return padTop + (1 - (value - pctAxis.min) / (pctAxis.max - pctAxis.min)) * plotHeight;
    };

    const onHover = (event: MouseEvent<SVGElement>, label: string, value: string) => {
        const cardRect = cardRef.current?.getBoundingClientRect();
        if (!cardRect) return;
        setTooltip({
            x: event.clientX - cardRect.left,
            y: event.clientY - cardRect.top,
            label,
            value,
        });
    };

    return (
        <div className="kap-premium-seasonal-card" ref={cardRef}>
            <div className="kap-premium-seasonal-head">
                <h4>Aylık Brüt Prim</h4>
                <div className="kap-premium-seasonal-legend">
                    {data.years.map((year, idx) => (
                        <span key={year}>
                            <i style={{ backgroundColor: PREMIUM_YEAR_COLORS[idx % PREMIUM_YEAR_COLORS.length] }} />
                            {year}
                        </span>
                    ))}
                    <span>
                        <i className="kap-premium-average-legend" />
                        3A Ort.
                    </span>
                    {yoyPoints.length > 0 ? (
                        <span>
                            <i className="kap-premium-yoy-legend" />
                            Yıllık Değişim
                        </span>
                    ) : null}
                </div>
            </div>
            <svg
                className="kap-premium-seasonal-svg"
                viewBox={`0 0 ${width} ${height}`}
                role="img"
                aria-label="Aylık brüt prim yıllara göre karşılaştırma"
                onMouseLeave={() => setTooltip(null)}
            >
                {yTicks.map((tick) => {
                    const yy = y(tick);
                    return (
                        <g key={tick}>
                            <line x1={padLeft} y1={yy} x2={width - padRight} y2={yy} className="kap-premium-seasonal-grid" />
                            <text x={padLeft - 10} y={yy + 4} textAnchor="end" className="kap-premium-seasonal-axis">
                                {formatPremiumAxis(tick)}
                            </text>
                        </g>
                    );
                })}
                {pctAxis ? (
                    <g>
                        {pctAxis.ticks.map((tick) => {
                            const yy = pctY(tick);
                            return (
                                <text key={`pct-${tick}`} x={width - padRight + 12} y={yy + 4} className="kap-premium-seasonal-axis">
                                    {formatPremiumPctAxis(tick)}
                                </text>
                            );
                        })}
                    </g>
                ) : null}
                {data.months.map((month, monthIdx) => {
                    const groupStart = padLeft + slotWidth * monthIdx + slotWidth / 2 - groupWidth / 2;
                    return (
                        <g key={month}>
                            {data.years.map((year, yearIdx) => {
                                const observation = data.valuesByKey[premiumObservationKey(year, month)];
                                if (!observation) return null;
                                const barHeight = Math.max(2, plotHeight - (y(observation.value) - padTop));
                                const x = groupStart + yearIdx * (barWidth + barGap);
                                const yy = padTop + plotHeight - barHeight;
                                return (
                                    <rect
                                        key={`${year}-${month}`}
                                        x={x}
                                        y={yy}
                                        width={barWidth}
                                        height={barHeight}
                                        rx={1}
                                        className="kap-premium-seasonal-bar"
                                        fill={PREMIUM_YEAR_COLORS[yearIdx % PREMIUM_YEAR_COLORS.length]}
                                        onMouseEnter={(event) => onHover(
                                            event,
                                            `${MONTH_LABELS_TR[month - 1]} ${year}`,
                                            observation.display,
                                        )}
                                        onMouseLeave={() => setTooltip(null)}
                                    />
                                );
                            })}
                            <text
                                x={padLeft + slotWidth * monthIdx + slotWidth / 2}
                                y={height - 17}
                                textAnchor="end"
                                transform={`rotate(-45 ${padLeft + slotWidth * monthIdx + slotWidth / 2} ${height - 17})`}
                                className="kap-premium-seasonal-x-label"
                            >
                                {MONTH_LABELS_TR[month - 1]}
                            </text>
                        </g>
                    );
                })}
                {averagePath ? <path d={averagePath} className="kap-premium-average-line" /> : null}
                {yoyPoints.map((point) => {
                    const yy = pctY(point.value);
                    const label = point.display;
                    const labelWidth = Math.max(28, label.length * 6.4 + 10);
                    const labelX = Math.min(width - padRight - labelWidth - 2, point.x + 7);
                    const labelY = Math.max(padTop + 11, Math.min(padTop + plotHeight - 9, yy - 8));
                    const toneClass = point.value < 0 ? 'is-negative' : 'is-positive';
                    return (
                        <g key={`yoy-${point.year}-${point.month}`}>
                            <circle
                                cx={point.x}
                                cy={yy}
                                r={3.7}
                                className="kap-premium-yoy-point"
                                onMouseEnter={(event) => onHover(event, `${point.periodLabel} yıllık değişim`, point.display)}
                                onMouseLeave={() => setTooltip(null)}
                            />
                            <rect
                                x={labelX}
                                y={labelY - 11}
                                width={labelWidth}
                                height={18}
                                rx={2}
                                className={`kap-premium-yoy-label-bg ${toneClass}`}
                            />
                            <text x={labelX + labelWidth / 2} y={labelY + 2.5} textAnchor="middle" className="kap-premium-yoy-label">
                                {label}
                            </text>
                        </g>
                    );
                })}
            </svg>
            {tooltip && (
                <div className="kap-chart-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
                    <div className="kap-chart-tooltip-label">{tooltip.label}</div>
                    <div className="kap-chart-tooltip-value">{tooltip.value}</div>
                </div>
            )}
        </div>
    );
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
        () => getOverviewSummaryRows(snapshot),
        [snapshot],
    );
    const premiumRows = useMemo(
        () => [...(snapshot.insurance_premium_disclosures || [])].sort(
            (a, b) => (intSafe(a.year) * 100 + intSafe(a.month)) - (intSafe(b.year) * 100 + intSafe(b.month)),
        ),
        [snapshot.insurance_premium_disclosures],
    );
    const visiblePremiumRows = useMemo(
        () => premiumRows.slice(-PREMIUM_TABLE_ROW_LIMIT).reverse(),
        [premiumRows],
    );
    const latestPremium = premiumRows.length > 0 ? premiumRows[premiumRows.length - 1] : null;
    const premiumFallbackRows = useMemo(() => premiumRows.slice(-PREMIUM_TABLE_ROW_LIMIT), [premiumRows]);
    const premiumMonthlySeries = useMemo(() => buildMonthlyPremiumSeries(premiumFallbackRows), [premiumFallbackRows]);
    const premiumSeasonalData = useMemo(() => buildPremiumSeasonalChartData(premiumRows), [premiumRows]);
    const premiumQuarterlySeries = useMemo(() => buildQuarterlyPremiumSeries(premiumRows), [premiumRows]);
    const latestPremiumIsKapSource = Boolean(latestPremium?.source_url?.includes('kap.org.tr'));
    const latestPremiumSourceLabel = latestPremiumIsKapSource ? 'Son KAP' : 'Kaynak';
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
    const [premiumHighlightedIndex, setPremiumHighlightedIndex] = useState<number | null>(null);
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

    const [chartQuarterWindow, setChartQuarterWindow] = useState(readStoredChartQuarterWindow);
    const displayedChartQuarterWindow = useMemo(
        () => displayQuarterWindowOption(chartQuarterWindow, quarters.length),
        [chartQuarterWindow, quarters.length],
    );
    const handleChartQuarterWindowChange = (nextWindow: number) => {
        setChartQuarterWindow(nextWindow);
        try {
            window.localStorage.setItem(CHART_QUARTER_WINDOW_STORAGE_KEY, String(nextWindow));
        } catch { /* ignore */ }
    };
    const chartQuarterButtons = QUARTER_WINDOW_OPTIONS.map((opt) => {
        const disabled = !isQuarterWindowOptionUseful(opt, quarters.length);
        return (
            <button
                key={opt}
                type="button"
                className={`overview-charts-quarter-btn${displayedChartQuarterWindow === opt ? ' is-active' : ''}`}
                onClick={() => handleChartQuarterWindowChange(opt)}
                disabled={disabled}
                title={disabled ? `${quarters.length} dönem veri var` : undefined}
            >
                {opt}Ç
            </button>
        );
    });
    const premiumOverviewCharts = useMemo<OverviewChartGroup[]>(
        () => (
            premiumQuarterlySeries.length > 0
                ? [{
                    title: 'Çeyreksel Brüt Prim Üretimi',
                    kind: 'bar' as const,
                    series: premiumQuarterlySeries.slice(-displayedChartQuarterWindow),
                }]
                : []
        ),
        [displayedChartQuarterWindow, premiumQuarterlySeries],
    );

    const summaryWarnings = useMemo(() => {
        const warnings: string[] = [];
        if (snapshot.cache_stale) {
            warnings.push('Canlı KAP yenilemesi tamamlanamadı; yerel cache gösteriliyor.');
        }
        if (latestQuarter && quarters.length < 2) {
            warnings.push(
                `${_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)} dışında geçmiş dönem bulunamadığı için karşılaştırma kolonları boş.`,
            );
        }
        return warnings;
    }, [latestQuarter, quarters.length, snapshot.cache_stale, snapshot.error]);

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
                        {summaryWarnings.length > 0 ? (
                            <div className="kap-summary-warnings" role="status">
                                {summaryWarnings.map((warning) => (
                                    <p key={warning}>{warning}</p>
                                ))}
                            </div>
                        ) : null}
                    </div>

                    <div className="kap-summary-grid">
                        <div className="kap-summary-table-wrap">
                            <h4>
                                Özet Gelir Tablosu <small>{_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)} YTD</small>
                            </h4>
                            <table className="kap-summary-table">
                                <thead>
                                    <tr>
                                        <th>Kalem</th>
                                        <th>{_periodLabel(intSafe(latestQuarter.year), intSafe(latestQuarter.period), latestQuarter.quarter)} YTD</th>
                                        <th>
                                            {prevYearSameQuarter
                                                        ? `${_periodLabel(intSafe(prevYearSameQuarter.year), intSafe(prevYearSameQuarter.period), prevYearSameQuarter.quarter)} YTD`
                                                : '-'}
                                        </th>
                                        <th>%</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {incomeSummaryRows.map((row) => {
                                        const currentValue = _resolveMetricYtdValue(latestQuarter, row.key);
                                        const baseValue =
                                            prevYearSameQuarterIdx >= 0
                                                ? _resolveMetricYtdValue(prevYearSameQuarter, row.key)
                                                : null;
                                        if (currentValue === null && baseValue === null) return null;
                                        const pct = _calcPctChange(currentValue, baseValue);
                                        return (
                                            <tr key={`income-${row.key}`}>
                                                <td>{row.label}</td>
                                                <td>
                                                    {_resolveMetricYtdDisplay(latestQuarter, row.key)}
                                                </td>
                                                <td>
                                                    {prevYearSameQuarterIdx >= 0
                                                        ? _resolveMetricYtdDisplay(prevYearSameQuarter, row.key)
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

            {latestPremium && (
                <section className="kap-premium-panel panel">
                    <div className="kap-premium-head">
                        <div>
                            <h3>Prim Üretimi</h3>
                            <p>Brüt yazılan prim; KAP özel durumları ve TSB serileriyle tamamlanır.</p>
                        </div>
                        {latestPremium.source_url ? (
                            <a href={latestPremium.source_url} target="_blank" rel="noreferrer">
                                {latestPremiumSourceLabel}
                            </a>
                        ) : null}
                    </div>

                    <div className="kap-premium-latest">
                        <div>
                            <span>Son dönem</span>
                            <strong>{premiumPeriodLabel(latestPremium)}</strong>
                        </div>
                        <div>
                            <span>Aylık prim</span>
                            <strong>{latestPremium.monthly_gross_premium_display}</strong>
                            <small className={premiumPctClass(latestPremium.monthly_yoy_pct)}>
                                {latestPremium.monthly_yoy_pct_display}
                            </small>
                        </div>
                    </div>

                    {(premiumSeasonalData || premiumMonthlySeries.length > 0) ? (
                        <div className="kap-premium-chart-block">
                            <div className="kap-premium-charts">
                                {premiumSeasonalData ? (
                                    <PremiumSeasonalMonthlyChart data={premiumSeasonalData} />
                                ) : premiumMonthlySeries.length > 0 ? (
                                    <BarChartCard
                                        title="Aylık Brüt Prim"
                                        series={premiumMonthlySeries}
                                        highlightedIndex={premiumHighlightedIndex}
                                        onHighlight={setPremiumHighlightedIndex}
                                        className="kap-premium-chart-card"
                                    />
                                ) : null}
                            </div>
                        </div>
                    ) : null}

                    <div className="kap-premium-table-scroll">
                        <table className="kap-premium-table">
                            <thead>
                                <tr>
                                    <th>Dönem</th>
                                    <th>Aylık</th>
                                    <th>Yıllık %</th>
                                </tr>
                            </thead>
                            <tbody>
                                {visiblePremiumRows.map((row) => (
                                    <tr key={`${row.year}-${row.month}-${row.disclosure_index || ''}`}>
                                        <td>
                                            {row.source_url ? (
                                                <a href={row.source_url} target="_blank" rel="noreferrer">
                                                    {premiumPeriodLabel(row)}
                                                </a>
                                            ) : (
                                                premiumPeriodLabel(row)
                                            )}
                                        </td>
                                        <td>{row.monthly_gross_premium_display}</td>
                                        <td className={premiumPctClass(row.monthly_yoy_pct)}>{row.monthly_yoy_pct_display}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </section>
            )}

            {quarters.length > 0 && (
                <section className="overview-charts-shell">
                    <div className="overview-charts-head">
                        <div>
                            <h3>Grafikler ve Analiz</h3>
                            <p>İlk olarak çeyreklik bar grafikler, ardından marj ve oran trendleri.</p>
                        </div>
                        <div className="overview-charts-quarter-picker">
                            <span className="overview-charts-quarter-label">Çeyrek:</span>
                            {chartQuarterButtons}
                        </div>
                    </div>
                    <StockCharts
                        snapshot={snapshot}
                        quarters={quarters}
                        embedded
                        chartWindowQuarters={displayedChartQuarterWindow}
                        extraCharts={premiumOverviewCharts}
                    />
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
