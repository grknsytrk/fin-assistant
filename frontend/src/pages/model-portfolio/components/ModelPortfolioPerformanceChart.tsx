import { useEffect, useRef, useState } from 'react';
import type {
    CSSProperties,
    KeyboardEvent as ReactKeyboardEvent,
    PointerEvent as ReactPointerEvent,
} from 'react';
import { RefreshCw, X } from 'lucide-react';
import type { ModelPortfolioPerformancePoint } from '../../../api/types';
import { useMobileChartLayout } from '../../../hooks/useMobileChartLayout';
import type { PerformanceSeries } from '../constants';
import {
    compoundAlpha,
    formatDateTR,
    formatPct,
    niceAxisStep,
    smoothLinePath,
    toneClass,
} from '../utils';
import './ModelPortfolioPerformanceChart.css';

export function ModelPortfolioPerformanceChart({
    points,
    loading,
    error,
    onRetry,
    hoverIndex,
    onHoverIndexChange,
}: {
    points: ModelPortfolioPerformancePoint[];
    loading: boolean;
    error: string | null;
    onRetry: () => void;
    hoverIndex: number | null;
    onHoverIndexChange: (idx: number | null) => void;
}) {
    const isMobile = useMobileChartLayout();
    const [measureAnchorIndex, setMeasureAnchorIndex] = useState<number | null>(null);
    const [selectedMeasureRange, setSelectedMeasureRange] = useState<{ start: number; end: number } | null>(null);
    const [visibleSeries, setVisibleSeries] = useState<Record<PerformanceSeries, boolean>>({
        portfolio: true,
        xu100: true,
        xu030: true,
    });
    const lastPointerTypeRef = useRef<string>('mouse');
    const chartStageRef = useRef<HTMLDivElement>(null);
    const pointWindowKey = `${points.length}:${points[0]?.date || ''}:${points[points.length - 1]?.date || ''}`;

    useEffect(() => {
        setMeasureAnchorIndex(null);
        setSelectedMeasureRange(null);
    }, [pointWindowKey]);

    useEffect(() => {
        if (!selectedMeasureRange && hoverIndex == null) return;
        const handleOutsidePointerDown = (e: PointerEvent) => {
            if (chartStageRef.current && !chartStageRef.current.contains(e.target as Node)) {
                setSelectedMeasureRange(null);
                setMeasureAnchorIndex(null);
                onHoverIndexChange(null);
            }
        };
        document.addEventListener('pointerdown', handleOutsidePointerDown);
        return () => document.removeEventListener('pointerdown', handleOutsidePointerDown);
    }, [selectedMeasureRange, hoverIndex, onHoverIndexChange]);

    const width = isMobile ? 420 : 900;
    const height = isMobile ? 245 : 340;
    const padLeft = isMobile ? 42 : 56;
    const padRight = isMobile ? 46 : 68;
    const padTop = isMobile ? 14 : 18;
    const padBottom = isMobile ? 28 : 34;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    if (loading && points.length === 0) {
        return <div className="mp-chart-empty">Kümülatif performans serisi hazırlanıyor...</div>;
    }
    if (error) {
        return (
            <div className="mp-chart-empty mp-chart-error-state" role="alert">
                <span>{error}</span>
                <button type="button" className="mp-btn mp-btn-subtle" onClick={onRetry}>
                    <RefreshCw size={13} /> Grafiği Yenile
                </button>
            </div>
        );
    }
    if (points.length < 2) {
        return (
            <div className="mp-chart-empty">
                Seçili dönemde grafik için en az 2 işlem günü verisi gerekir. Bu durum portföy işlemlerini etkilemez.
            </div>
        );
    }

    const allVals = points.flatMap((p) => [
        ...(visibleSeries.portfolio ? [p.portfolio_value] : []),
        ...(visibleSeries.xu100 ? [p.xu100_value] : []),
        ...(visibleSeries.xu030 ? [p.xu030_value] : []),
    ]);
    if (allVals.length === 0) allVals.push(100);
    const rawMin = Math.min(...allVals, 100);
    const rawMax = Math.max(...allVals, 100);
    const niceStep = niceAxisStep(Math.max(2, rawMax - rawMin) / (isMobile ? 3 : 4));
    const minVal = Math.floor((rawMin - niceStep * 0.15) / niceStep) * niceStep;
    const maxVal = Math.ceil((rawMax + niceStep * 0.15) / niceStep) * niceStep;
    const tickDecimals = niceStep < 1 ? 1 : 0;

    const toX = (i: number) => padLeft + (i / Math.max(1, points.length - 1)) * plotW;
    const toY = (v: number) => padTop + ((maxVal - v) / Math.max(1e-6, maxVal - minVal)) * plotH;

    const portCoords = points.map((p, i): [number, number] => [toX(i), toY(p.portfolio_value)]);
    const portPath = smoothLinePath(portCoords);
    const portArea = `${portPath} L${toX(points.length - 1).toFixed(1)},${(padTop + plotH).toFixed(1)} L${padLeft.toFixed(1)},${(padTop + plotH).toFixed(1)} Z`;
    const xuPath = smoothLinePath(points.map((p, i): [number, number] => [toX(i), toY(p.xu100_value)]));
    const xu030Path = smoothLinePath(points.map((p, i): [number, number] => [toX(i), toY(p.xu030_value)]));

    const ticks: Array<{ y: number; val: number }> = [];
    for (let v = minVal; v <= maxVal + niceStep / 2; v += niceStep) {
        ticks.push({ y: toY(v), val: v });
    }

    // Evenly spaced date labels; drop any label that would collide with the next one.
    const minLabelGap = isMobile ? 95 : 120;
    const xLabelIndexes: number[] = [];
    {
        const wanted = Math.min(points.length, isMobile ? 3 : 4);
        const candidates = Array.from(
            new Set(
                Array.from({ length: wanted }, (_, k) =>
                    Math.round((k / Math.max(1, wanted - 1)) * (points.length - 1)),
                ),
            ),
        );
        candidates.forEach((idx) => {
            const prev = xLabelIndexes[xLabelIndexes.length - 1];
            if (prev != null && toX(idx) - toX(prev) < minLabelGap) {
                if (idx === points.length - 1) xLabelIndexes.pop();
                else return;
            }
            xLabelIndexes.push(idx);
        });
    }
    const spansMultipleYears = points[0].date.slice(0, 4) !== points[points.length - 1].date.slice(0, 4);
    const formatAxisDate = (iso: string) => {
        const d = new Date(`${iso.slice(0, 10)}T00:00:00`);
        if (Number.isNaN(d.getTime())) return iso;
        return d.toLocaleDateString('tr-TR', spansMultipleYears
            ? { day: '2-digit', month: 'short', year: 'numeric' }
            : { day: '2-digit', month: 'short' });
    };
    const formatLegendValue = (v: number) =>
        `${v.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ₺`;
    const activeIdx = hoverIndex != null && hoverIndex >= 0 && hoverIndex < points.length
        ? hoverIndex
        : points.length - 1;
    const activePt = points[activeIdx];
    const hasPointHover = hoverIndex != null && hoverIndex >= 0 && hoverIndex < points.length;

    const dragMeasureRange = measureAnchorIndex != null && activeIdx !== measureAnchorIndex
        ? { start: Math.min(measureAnchorIndex, activeIdx), end: Math.max(measureAnchorIndex, activeIdx) }
        : null;
    const visibleMeasureRange = dragMeasureRange || selectedMeasureRange;
    const measureStartPt = visibleMeasureRange ? points[visibleMeasureRange.start] : null;
    const measureEndPt = visibleMeasureRange ? points[visibleMeasureRange.end] : null;
    const measurePortfolioReturn = measureStartPt && measureEndPt && measureStartPt.portfolio_value !== 0
        ? ((measureEndPt.portfolio_value / measureStartPt.portfolio_value) - 1) * 100
        : null;
    const measureXu100Return = measureStartPt && measureEndPt && measureStartPt.xu100_value !== 0
        ? ((measureEndPt.xu100_value / measureStartPt.xu100_value) - 1) * 100
        : null;
    const measureXu030Return = measureStartPt && measureEndPt && measureStartPt.xu030_value !== 0
        ? ((measureEndPt.xu030_value / measureStartPt.xu030_value) - 1) * 100
        : null;
    const measureRelativeXu100Return = compoundAlpha(measurePortfolioReturn, measureXu100Return);
    const measureRelativeXu030Return = compoundAlpha(measurePortfolioReturn, measureXu030Return);

    const measureTooltipItems: Array<{ label: string; color?: string; change: number | null }> = [];
    const measureAlphaItems: Array<{ label: string; change: number | null }> = [];
    if (measureStartPt && measureEndPt) {
        if (visibleSeries.portfolio) {
            measureTooltipItems.push({ label: 'Model Portföy', color: '#3b82f6', change: measurePortfolioReturn });
        }
        if (visibleSeries.xu100) {
            measureTooltipItems.push({ label: 'BIST 100', color: '#10b981', change: measureXu100Return });
        }
        if (visibleSeries.xu030) {
            measureTooltipItems.push({ label: 'BIST 30', color: '#f59e0b', change: measureXu030Return });
        }
        if (visibleSeries.portfolio && visibleSeries.xu100) {
            measureAlphaItems.push({ label: 'Alfa · BIST 100', change: measureRelativeXu100Return });
        }
        if (visibleSeries.portfolio && visibleSeries.xu030) {
            measureAlphaItems.push({ label: 'Alfa · BIST 30', change: measureRelativeXu030Return });
        }
    }

    const pointTooltipItems: Array<{ label: string; color: string; value: number; change: number | null }> = [];
    const pointAlphaItems: Array<{ label: string; change: number | null }> = [];
    if (hasPointHover && !visibleMeasureRange) {
        if (visibleSeries.portfolio) {
            pointTooltipItems.push({
                label: 'Model Portföy',
                color: '#3b82f6',
                value: activePt.portfolio_value,
                change: activePt.portfolio_return_pct,
            });
        }
        if (visibleSeries.xu100) {
            pointTooltipItems.push({
                label: 'BIST 100',
                color: '#10b981',
                value: activePt.xu100_value,
                change: activePt.xu100_return_pct,
            });
        }
        if (visibleSeries.xu030) {
            pointTooltipItems.push({
                label: 'BIST 30',
                color: '#f59e0b',
                value: activePt.xu030_value,
                change: activePt.xu030_return_pct,
            });
        }
        if (visibleSeries.portfolio && visibleSeries.xu100) {
            pointAlphaItems.push({ label: 'Alfa · BIST 100', change: activePt.relative_return_pct });
        }
        if (visibleSeries.portfolio && visibleSeries.xu030) {
            pointAlphaItems.push({ label: 'Alfa · BIST 30', change: activePt.relative_xu030_return_pct });
        }
    }

    let measureLineTop = Infinity;
    let measureLineBottom = -Infinity;
    if (visibleMeasureRange) {
        for (let i = visibleMeasureRange.start; i <= visibleMeasureRange.end; i++) {
            const p = points[i];
            const ys = [
                ...(visibleSeries.portfolio ? [toY(p.portfolio_value)] : []),
                ...(visibleSeries.xu100 ? [toY(p.xu100_value)] : []),
                ...(visibleSeries.xu030 ? [toY(p.xu030_value)] : []),
            ];
            ys.forEach((y) => {
                measureLineTop = Math.min(measureLineTop, y);
                measureLineBottom = Math.max(measureLineBottom, y);
            });
        }
    }
    const measureCenterX = visibleMeasureRange
        ? (toX(visibleMeasureRange.start) + toX(visibleMeasureRange.end)) / 2
        : 0;
    const roomAbove = Number.isFinite(measureLineTop) ? Math.max(0, measureLineTop - padTop) : 0;
    const roomBelow = Number.isFinite(measureLineBottom) ? Math.max(0, padTop + plotH - measureLineBottom) : 0;
    const tooltipAbove = roomAbove >= roomBelow || roomAbove >= 110;
    const measureTooltipStyle: CSSProperties = isMobile
        ? {}
        : {
            left: `clamp(8px, calc(${(measureCenterX / width) * 100}% - 112px), calc(100% - 232px))`,
            ...(tooltipAbove
                ? roomAbove >= 145
                    ? { bottom: `calc(${Math.max(8, 100 - (measureLineTop / height) * 100)}% + 12px)` }
                    : { top: '8px' }
                : roomBelow >= 145
                  ? { top: `calc(${(Math.min(measureLineBottom, padTop + plotH) / height) * 100}% + 12px)` }
                  : { bottom: '8px' }),
        };

    // Desktop floating tooltip positioning:
    // Point hover tooltip flips cleanly to the left or right of the crosshair so it never clips or covers the dot.
    const hoverXPct = (toX(activeIdx) / width) * 100;
    const placeHoverLeft = activeIdx > points.length * 0.58;
    const hoverYs = [
        ...(visibleSeries.portfolio ? [toY(activePt.portfolio_value)] : []),
        ...(visibleSeries.xu100 ? [toY(activePt.xu100_value)] : []),
        ...(visibleSeries.xu030 ? [toY(activePt.xu030_value)] : []),
    ];
    const anchorY = hoverYs.length ? Math.min(...hoverYs) : padTop + plotH * 0.3;
    const hoverTopPct = Math.max(4, Math.min(48, (anchorY / height) * 100));
    const pointTooltipStyle: CSSProperties = isMobile
        ? {}
        : {
            left: placeHoverLeft
                ? `clamp(8px, calc(${hoverXPct}% - 238px), calc(100% - 232px))`
                : `clamp(8px, calc(${hoverXPct}% + 14px), calc(100% - 232px))`,
            top: `${hoverTopPct}%`,
        };

    const toggleSeries = (series: PerformanceSeries) => {
        setVisibleSeries((current) => ({ ...current, [series]: !current[series] }));
    };
    const hasVisibleSeries = visibleSeries.portfolio || visibleSeries.xu100 || visibleSeries.xu030;

    const resolveIndexFromClientX = (svgEl: SVGSVGElement, clientX: number): number | null => {
        const rect = svgEl.getBoundingClientRect();
        if (rect.width <= 0) return null;
        const relX = ((clientX - rect.left) / rect.width) * width;
        const ratio = Math.max(0, Math.min(1, (relX - padLeft) / plotW));
        return Math.round(ratio * (points.length - 1));
    };

    const handlePointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
        lastPointerTypeRef.current = e.pointerType;
        const idx = resolveIndexFromClientX(e.currentTarget, e.clientX);
        if (idx == null) return;
        onHoverIndexChange(idx);
    };

    const handlePointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        lastPointerTypeRef.current = e.pointerType;
        const idx = resolveIndexFromClientX(e.currentTarget, e.clientX);
        if (idx == null) return;
        if (e.pointerType !== 'mouse') {
            setSelectedMeasureRange(null);
            onHoverIndexChange(idx);
            return;
        }
        e.preventDefault();
        setSelectedMeasureRange(null);
        setMeasureAnchorIndex(idx);
        onHoverIndexChange(idx);
        e.currentTarget.setPointerCapture?.(e.pointerId);
    };

    const finishPointerMeasure = (e: ReactPointerEvent<SVGSVGElement>, commit: boolean) => {
        if (e.pointerType !== 'mouse') return;
        if (commit && measureAnchorIndex != null) {
            const endIndex = resolveIndexFromClientX(e.currentTarget, e.clientX);
            if (endIndex != null) {
                onHoverIndexChange(endIndex);
                if (endIndex !== measureAnchorIndex) {
                    setSelectedMeasureRange({
                        start: Math.min(measureAnchorIndex, endIndex),
                        end: Math.max(measureAnchorIndex, endIndex),
                    });
                }
            }
        }
        setMeasureAnchorIndex(null);
        try {
            if (e.currentTarget.hasPointerCapture?.(e.pointerId)) {
                e.currentTarget.releasePointerCapture(e.pointerId);
            }
        } catch {
            // Ignore release errors when capture has already ended.
        }
    };

    const handleChartKeyDown = (e: ReactKeyboardEvent<SVGSVGElement>) => {
        if (e.key === 'Escape') {
            setMeasureAnchorIndex(null);
            setSelectedMeasureRange(null);
            onHoverIndexChange(null);
            return;
        }
        if (e.key === ' ' || e.key === 'Enter') {
            e.preventDefault();
            if (measureAnchorIndex == null) {
                setSelectedMeasureRange(null);
                setMeasureAnchorIndex(activeIdx);
            } else {
                if (activeIdx !== measureAnchorIndex) {
                    setSelectedMeasureRange({
                        start: Math.min(measureAnchorIndex, activeIdx),
                        end: Math.max(measureAnchorIndex, activeIdx),
                    });
                }
                setMeasureAnchorIndex(null);
            }
            return;
        }
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'Home' || e.key === 'End') {
            e.preventDefault();
            const current = hoverIndex ?? points.length - 1;
            const next = e.key === 'Home'
                ? 0
                : e.key === 'End'
                    ? points.length - 1
                    : Math.max(0, Math.min(points.length - 1, current + (e.key === 'ArrowRight' ? 1 : -1)));
            onHoverIndexChange(next);
        }
    };

    const showMeasureTooltip = Boolean(visibleMeasureRange && hasVisibleSeries && measureStartPt && measureEndPt);
    const showPointTooltip = Boolean(!showMeasureTooltip && hasPointHover && hasVisibleSeries);

    return (
        <div className="mp-chart-container">
            <div className="mp-chart-legend">
                <div className="mp-legend-toggles">
                    {([
                        { key: 'portfolio', label: 'Model Portföy', aria: 'Model Portföy', value: activePt.portfolio_value, pct: activePt.portfolio_return_pct },
                        { key: 'xu100', label: 'BIST 100', aria: 'BIST 100 XU100', value: activePt.xu100_value, pct: activePt.xu100_return_pct },
                        { key: 'xu030', label: 'BIST 30', aria: 'BIST 30 XU030', value: activePt.xu030_value, pct: activePt.xu030_return_pct },
                    ] as Array<{ key: PerformanceSeries; label: string; aria: string; value: number; pct: number | null }>).map((s) => (
                        <button
                            key={s.key}
                            type="button"
                            className={`mp-legend-toggle${visibleSeries[s.key] ? '' : ' is-hidden'}`}
                            aria-pressed={visibleSeries[s.key]}
                            aria-label={`${visibleSeries[s.key] ? 'Grafikten gizle' : 'Grafikte göster'}: ${s.aria}`}
                            onClick={() => toggleSeries(s.key)}
                        >
                            <span className="mp-legend-name">
                                <span className={`mp-legend-swatch is-${s.key}`} />
                                {s.label}
                            </span>
                            <span className="mp-legend-figures mp-mono">
                                <strong>{formatLegendValue(s.value)}</strong>
                                <span className={toneClass(s.pct)}>{formatPct(s.pct)}</span>
                            </span>
                        </button>
                    ))}
                </div>
                <div className="mp-legend-meta">
                    <span className="mp-legend-date mp-mono">{formatDateTR(activePt.date)}</span>
                    <span className="mp-legend-alphas">
                        {visibleSeries.portfolio && visibleSeries.xu100 && (
                            <span className="mp-legend-alpha">
                                Alfa · BIST 100
                                <strong className={`mp-mono ${toneClass(activePt.relative_return_pct)}`}>
                                    {formatPct(activePt.relative_return_pct)}
                                </strong>
                            </span>
                        )}
                        {visibleSeries.portfolio && visibleSeries.xu030 && (
                            <span className="mp-legend-alpha">
                                Alfa · BIST 30
                                <strong className={`mp-mono ${toneClass(activePt.relative_xu030_return_pct)}`}>
                                    {formatPct(activePt.relative_xu030_return_pct)}
                                </strong>
                            </span>
                        )}
                    </span>
                </div>
            </div>

            <div className="mp-chart-stage" ref={chartStageRef}>
                <svg
                    className="mp-perf-svg"
                    viewBox={`0 0 ${width} ${height}`}
                    role="group"
                    aria-label="Model portföyü, XU100 ve XU030 performans grafiği. Serileri göstermek veya gizlemek için lejantlarına basın. İki tarih arasındaki değişimi görmek için grafikte sürükleyin."
                    tabIndex={0}
                    onPointerMove={handlePointerMove}
                    onPointerDown={handlePointerDown}
                    onPointerUp={(e) => finishPointerMeasure(e, true)}
                    onPointerCancel={(e) => finishPointerMeasure(e, false)}
                    onPointerLeave={(e) => {
                        if (e.pointerType === 'mouse' && measureAnchorIndex == null) {
                            onHoverIndexChange(null);
                        }
                    }}
                    onKeyDown={handleChartKeyDown}
                >
                    <title>Model Portföy, BIST 100 ve BIST 30 performansı</title>
                    <defs>
                        <linearGradient id="mpPortFill" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.32" />
                            <stop offset="70%" stopColor="#3b82f6" stopOpacity="0.05" />
                            <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                        </linearGradient>
                        <clipPath id="mpPlotClip">
                            <rect x={padLeft} y={padTop - 2} width={plotW} height={plotH + 4} />
                        </clipPath>
                    </defs>

                    {ticks.map((t, i) => (
                        <g key={i}>
                            <line
                                x1={padLeft}
                                y1={t.y}
                                x2={width - padRight}
                                y2={t.y}
                                className="mp-chart-grid"
                            />
                            <text x={padLeft - 8} y={t.y + 4} textAnchor="end" className="mp-chart-axis-text">
                                {t.val.toLocaleString('tr-TR', { minimumFractionDigits: tickDecimals, maximumFractionDigits: tickDecimals })}
                            </text>
                        </g>
                    ))}

                    {/* 100 TL baseline */}
                    {minVal < 100 && maxVal > 100 && (
                        <g>
                            <line
                                x1={padLeft}
                                y1={toY(100)}
                                x2={width - padRight}
                                y2={toY(100)}
                                className="mp-chart-baseline"
                            />
                            <text x={width - padRight + 6} y={toY(100) + 4} className="mp-chart-baseline-text">
                                100 ₺
                            </text>
                        </g>
                    )}

                    {visibleMeasureRange && (
                        <rect
                            x={toX(visibleMeasureRange.start)}
                            y={padTop}
                            width={Math.max(1, toX(visibleMeasureRange.end) - toX(visibleMeasureRange.start))}
                            height={plotH}
                            className="mp-chart-selection-band"
                        />
                    )}

                    <g clipPath="url(#mpPlotClip)">
                        {visibleSeries.portfolio && <path d={portArea} fill="url(#mpPortFill)" />}
                        {visibleSeries.xu030 && <path d={xu030Path} fill="none" className="mp-line-xu030" />}
                        {visibleSeries.xu100 && <path d={xuPath} fill="none" className="mp-line-xu100" />}
                        {visibleSeries.portfolio && <path d={portPath} fill="none" className="mp-line-portfolio" />}
                    </g>
                    {!hasVisibleSeries && (
                        <text x={width / 2} y={padTop + plotH / 2} textAnchor="middle" className="mp-chart-no-series">
                            Görmek için üstteki serilerden birine basın
                        </text>
                    )}

                    {/* Active crosshair & dots */}
                    {(hasPointHover || !isMobile) && (
                        <>
                            <line
                                x1={toX(activeIdx)}
                                y1={padTop}
                                x2={toX(activeIdx)}
                                y2={padTop + plotH}
                                className="mp-chart-crosshair"
                            />
                            {visibleSeries.xu100 && (
                                <circle
                                    cx={toX(activeIdx)}
                                    cy={toY(activePt.xu100_value)}
                                    r={isMobile ? '3.5' : '4'}
                                    className="mp-dot-xu100"
                                />
                            )}
                            {visibleSeries.xu030 && (
                                <circle
                                    cx={toX(activeIdx)}
                                    cy={toY(activePt.xu030_value)}
                                    r={isMobile ? '3.5' : '4'}
                                    className="mp-dot-xu030"
                                />
                            )}
                            {visibleSeries.portfolio && (
                                <>
                                    <circle
                                        cx={toX(activeIdx)}
                                        cy={toY(activePt.portfolio_value)}
                                        r={isMobile ? '7.5' : '9'}
                                        className="mp-dot-portfolio-halo"
                                    />
                                    <circle
                                        cx={toX(activeIdx)}
                                        cy={toY(activePt.portfolio_value)}
                                        r={isMobile ? '4' : '4.5'}
                                        className="mp-dot-portfolio"
                                    />
                                </>
                            )}
                        </>
                    )}

                    {visibleMeasureRange && measureStartPt && measureEndPt && (
                        <g className="mp-chart-measure-markers">
                            <line x1={toX(visibleMeasureRange.start)} y1={padTop} x2={toX(visibleMeasureRange.start)} y2={padTop + plotH} />
                            <line x1={toX(visibleMeasureRange.end)} y1={padTop} x2={toX(visibleMeasureRange.end)} y2={padTop + plotH} />
                            {visibleSeries.portfolio && <circle cx={toX(visibleMeasureRange.start)} cy={toY(measureStartPt.portfolio_value)} r="4" className="mp-dot-portfolio" />}
                            {visibleSeries.portfolio && <circle cx={toX(visibleMeasureRange.end)} cy={toY(measureEndPt.portfolio_value)} r="4" className="mp-dot-portfolio" />}
                            {visibleSeries.xu100 && <circle cx={toX(visibleMeasureRange.start)} cy={toY(measureStartPt.xu100_value)} r="3.5" className="mp-dot-xu100" />}
                            {visibleSeries.xu100 && <circle cx={toX(visibleMeasureRange.end)} cy={toY(measureEndPt.xu100_value)} r="3.5" className="mp-dot-xu100" />}
                            {visibleSeries.xu030 && <circle cx={toX(visibleMeasureRange.start)} cy={toY(measureStartPt.xu030_value)} r="3.5" className="mp-dot-xu030" />}
                            {visibleSeries.xu030 && <circle cx={toX(visibleMeasureRange.end)} cy={toY(measureEndPt.xu030_value)} r="3.5" className="mp-dot-xu030" />}
                        </g>
                    )}

                    {/* X axis date labels */}
                    {xLabelIndexes.map((idx, k) => (
                        <text
                            key={idx}
                            x={toX(idx)}
                            y={height - 8}
                            textAnchor={k === 0 ? 'start' : k === xLabelIndexes.length - 1 ? 'end' : 'middle'}
                            className="mp-chart-axis-text"
                        >
                            {formatAxisDate(points[idx].date)}
                        </text>
                    ))}
                </svg>

                {showMeasureTooltip && measureStartPt && measureEndPt && (
                    <div className="mp-chart-tooltip mp-chart-measure-tooltip" style={measureTooltipStyle} role="status" aria-live="polite">
                        <div className="mp-chart-tooltip-header">
                            <span>{formatDateTR(measureStartPt.date)} → {formatDateTR(measureEndPt.date)}</span>
                            {selectedMeasureRange && (
                                <button
                                    type="button"
                                    aria-label="Ölçümü temizle"
                                    title="Ölçümü temizle"
                                    onClick={() => setSelectedMeasureRange(null)}
                                >
                                    <X size={12} />
                                </button>
                            )}
                        </div>
                        <ul className="mp-chart-tooltip-list">
                            {measureTooltipItems.map((item) => (
                                <li key={item.label}>
                                    <span className="mp-chart-tooltip-name">
                                        {item.color && (
                                            <span className="mp-chart-tooltip-dot" style={{ backgroundColor: item.color }} />
                                        )}
                                        {item.label}
                                    </span>
                                    <strong className={`mp-mono ${toneClass(item.change)}`}>{formatPct(item.change)}</strong>
                                </li>
                            ))}
                        </ul>
                        {measureAlphaItems.length > 0 && (
                            <div className="mp-chart-tooltip-alphas">
                                {measureAlphaItems.map((item) => (
                                    <div key={item.label} className="mp-chart-tooltip-alpha-row">
                                        <span>{item.label}</span>
                                        <strong className={`mp-mono ${toneClass(item.change)}`}>{formatPct(item.change)}</strong>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {showPointTooltip && (
                    <div className="mp-chart-tooltip" style={pointTooltipStyle} role="status" aria-live="polite">
                        <div className="mp-chart-tooltip-header">
                            <span>{formatDateTR(activePt.date)}</span>
                            {(isMobile || lastPointerTypeRef.current !== 'mouse') && (
                                <button
                                    type="button"
                                    aria-label="Seçili tarihi kapat"
                                    title="Kapat"
                                    onClick={() => onHoverIndexChange(null)}
                                >
                                    <X size={12} />
                                </button>
                            )}
                        </div>
                        <ul className="mp-chart-tooltip-list">
                            {pointTooltipItems.map((item) => (
                                <li key={item.label}>
                                    <span className="mp-chart-tooltip-name">
                                        <span className="mp-chart-tooltip-dot" style={{ backgroundColor: item.color }} />
                                        {item.label}
                                    </span>
                                    <span className="mp-chart-tooltip-val mp-mono">
                                        <span className="mp-chart-tooltip-price">{formatLegendValue(item.value)}</span>
                                        <strong className={toneClass(item.change)}>{formatPct(item.change)}</strong>
                                    </span>
                                </li>
                            ))}
                        </ul>
                        {pointAlphaItems.length > 0 && (
                            <div className="mp-chart-tooltip-alphas">
                                {pointAlphaItems.map((item) => (
                                    <div key={item.label} className="mp-chart-tooltip-alpha-row">
                                        <span>{item.label}</span>
                                        <strong className={`mp-mono ${toneClass(item.change)}`}>{formatPct(item.change)}</strong>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
