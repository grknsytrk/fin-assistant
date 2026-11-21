import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import {
    AlertTriangle,
    Award,
    BriefcaseBusiness,
    CheckCircle2,
    ChevronRight,
    Clock,
    Eye,
    EyeOff,
    History,
    KeyRound,
    Lock,
    LogOut,
    Pencil,
    Plus,
    RefreshCw,
    Trash2,
    X,
} from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { apiClient } from '../api/client';
import type {
    MarketStockRow,
    MarketStocksResponse,
    MarketUniverseRow,
    ModelPortfolioChangelogItem,
    ModelPortfolioPerformancePoint,
    ModelPortfolioPosition,
    ModelPortfolioResponse,
} from '../api/types';
import MarketsNavigation, { type MarketsNavigationSection } from '../components/MarketsNavigation';
import MarketWatchRail from '../components/MarketWatchRail';
import SymbolLogo from '../components/SymbolLogo';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { useMarketStore } from '../stores/marketStore';
import {
    DEFAULT_FUND_TAB,
    DEFAULT_MARKET_INDEX,
    DEFAULT_STOCK_TAB,
    toFundComparison,
    toFundDetail,
    toFunds,
    toMarketsIndices,
    toMarketsOverview,
    toMarketsStocks,
    toStockComparison,
    toStockDetail,
} from '../routing/routes';
import '../styles/market-shared.css';
import './ModelPortfolioPage.css';
import { ModelPortfolioDonut } from './model-portfolio/components/ModelPortfolioDonut';
import { ModelPortfolioPerformanceChart } from './model-portfolio/components/ModelPortfolioPerformanceChart';
import { ModelPortfolioThesisDrawer } from './model-portfolio/components/ModelPortfolioThesisDrawer';
import {
    DONUT_PALETTE,
    PERF_RANGES,
    type PerfRange,
    type PortfolioTab,
} from './model-portfolio/constants';
import { useAccessibleDialog } from './model-portfolio/hooks/useAccessibleDialog';
import {
    compoundAlpha,
    computeProgress,
    formatCompactDateTimeTR,
    formatCompactDateTR,
    formatDateTimeTR,
    formatDateTR,
    formatPct,
    formatPrice,
    formatSessionCountdown,
    mergeLiveMarketRows,
    quoteTimestampAtLeastAsRecent,
    sliceAndNormalizePerformance,
    toneClass,
} from './model-portfolio/utils';

const ModelPortfolioEditorModal = lazy(() => import('./model-portfolio/components/ModelPortfolioEditorModal'));
const ModelPortfolioCloseModal = lazy(() => import('./model-portfolio/components/ModelPortfolioCloseModal'));

export default function ModelPortfolioPage() {
    useDocumentTitle('Model Portföy | Fin-Rag');
    const navigate = useNavigate();
    const [navCollapsed, setNavCollapsed] = useState(false);

    // Shared rail universe & live quote state (reuses existing caches, 0 extra calls)
    const [xu100UniverseRows, setXu100UniverseRows] = useState<MarketUniverseRow[]>(
        () => apiClient.getCachedMarketUniverse('XUTUM')?.rows || [],
    );
    const [liveStockMap, setLiveStockMap] = useState<Map<string, MarketStockRow>>(() => {
        const cached = apiClient.getCachedMarketStocks('XUTUM');
        return new Map((cached?.rows || []).map((row) => {
            const symbol = String(row.symbol || row.company).toUpperCase();
            return [symbol, { ...row, price_as_of: row.price_as_of || cached?.as_of || null }];
        }));
    });

    // Model Portfolio state
    const [portfolioData, setPortfolioData] = useState<ModelPortfolioResponse | null>(
        () => apiClient.getCachedModelPortfolio(),
    );
    const [loading, setLoading] = useState<boolean>(() => !apiClient.getCachedModelPortfolio());
    const [error, setError] = useState<string | null>(null);

    // Live benchmark pipeline (M3)
    const [liveXu100, setLiveXu100] = useState<number | null>(null);
    const { itemsBySymbol } = useMarketStore();
    const watchXu100 = itemsBySymbol.get('XU100')?.price ?? null;
    const effectiveXu100 = liveXu100 ?? watchXu100 ?? portfolioData?.current_xu100 ?? null;

    // Performance chart state (loaded separately for instant table first-paint)
    const [perfPoints, setPerfPoints] = useState<ModelPortfolioPerformancePoint[]>([]);
    const [perfLoading, setPerfLoading] = useState(false);
    const [perfError, setPerfError] = useState<string | null>(null);
    const [selectedRange, setSelectedRange] = useState<PerfRange>('all');
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const [activeDonutIndex, setActiveDonutIndex] = useState<number | null>(null);

    // Tabs & Drawer state
    const [activeTab, setActiveTab] = useState<PortfolioTab>('active');
    const [selectedPositionId, setSelectedPositionId] = useState<string | null>(null);

    // Short-lived admin session stays in memory and is never persisted in browser storage.
    const [adminToken, setAdminToken] = useState<string>('');
    const [adminSessionExpiresAt, setAdminSessionExpiresAt] = useState<number | null>(null);
    const [adminRemainingSeconds, setAdminRemainingSeconds] = useState<number | null>(null);
    const [isAdminVerified, setIsAdminVerified] = useState(false);
    const [adminModalOpen, setAdminModalOpen] = useState(false);
    const [adminTokenInput, setAdminTokenInput] = useState('');
    const [showAdminToken, setShowAdminToken] = useState(false);
    const [adminAuthError, setAdminAuthError] = useState<string | null>(null);
    const [adminAuthLoading, setAdminAuthLoading] = useState(false);
    const adminDialogRef = useAccessibleDialog<HTMLDivElement>(adminModalOpen, () => setAdminModalOpen(false));

    useEffect(() => {
        if (!adminToken || adminSessionExpiresAt === null) {
            setAdminRemainingSeconds(null);
            return;
        }
        const updateRemaining = () => {
            const remaining = Math.max(0, Math.ceil((adminSessionExpiresAt - Date.now()) / 1000));
            setAdminRemainingSeconds(remaining);
            if (remaining <= 0) {
                setAdminToken('');
                setAdminSessionExpiresAt(null);
                setIsAdminVerified(false);
            }
        };
        updateRemaining();
        const intervalId = window.setInterval(updateRemaining, 1000);
        return () => window.clearInterval(intervalId);
    }, [adminSessionExpiresAt, adminToken]);

    const adminCountdown = formatSessionCountdown(adminRemainingSeconds);

    // Position Add / Edit / Close modal state
    const [editorModalOpen, setEditorModalOpen] = useState(false);
    const [editingPosition, setEditingPosition] = useState<ModelPortfolioPosition | null>(null);
    const [closingPosition, setClosingPosition] = useState<ModelPortfolioPosition | null>(null);

    useEffect(() => {
        if (xu100UniverseRows.length) return;
        let cancelled = false;
        apiClient
            .marketUniverse({ index: 'XUTUM' })
            .then((res) => {
                if (!cancelled) setXu100UniverseRows(res.rows || []);
            })
            .catch(() => {});
        return () => {
            cancelled = true;
        };
    }, [xu100UniverseRows.length]);

    // Reuse the shared XUTUM snapshot and keep its ref-counted live stream active on this page.
    useEffect(() => {
        let cancelled = false;
        const applyPayload = (payload: MarketStocksResponse) => {
            setLiveStockMap((prev) => mergeLiveMarketRows(prev, payload));
            const benchmarkPrice = (payload as any)?.benchmarks?.['XU100']?.price
                ?? (payload as any)?.benchmarks?.XU100?.price
                ?? (payload.index === 'XU100' ? ((payload as any).index_price ?? (payload as any).price) : null);
            if (typeof benchmarkPrice === 'number' && benchmarkPrice > 0) {
                setLiveXu100(benchmarkPrice);
            }
        };
        const unsubscribe = apiClient.subscribeMarketStocks((index, payload) => {
            if (index !== 'XUTUM' && index !== 'XU100') return;
            applyPayload(payload);
        });
        const releaseStream = apiClient.retainMarketStocksStream('XUTUM');

        apiClient
            .marketStocks({ index: 'XUTUM' })
            .then((payload) => {
                if (!cancelled) applyPayload(payload);
            })
            .catch(() => {});

        return () => {
            cancelled = true;
            unsubscribe();
            releaseStream();
        };
    }, []);

    const loadPerformance = useCallback((version: number, refresh = false) => {
        setPerfLoading(true);
        setPerfError(null);
        return apiClient
            .modelPortfolioPerformance({ version, refresh })
            .then((perf) => setPerfPoints(perf.points || []))
            .catch((err: unknown) => {
                setPerfPoints([]);
                setPerfError(err instanceof Error ? err.message : 'Performans grafiği verisi alınamadı.');
            })
            .finally(() => setPerfLoading(false));
    }, []);

    const loadPortfolio = useCallback((refresh = false) => {
        if (!portfolioData) setLoading(true);
        setError(null);
        apiClient
            .modelPortfolio({ refresh })
            .then((data) => {
                setPortfolioData(data);
                setLoading(false);
                void loadPerformance(data.version, refresh);
            })
            .catch((err: unknown) => {
                setError(err instanceof Error ? err.message : 'Model portföy verisi alınamadı.');
            })
            .finally(() => setLoading(false));
    }, [loadPerformance, portfolioData]);

    useEffect(() => {
        loadPortfolio(false);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Live-enriched active positions using `liveStockMap` from MarketWatchRail and `effectiveXu100`
    const liveActivePositions = useMemo<ModelPortfolioPosition[]>(() => {
        const rawList = portfolioData?.active_positions || [];
        const currentXu100 = effectiveXu100;
        return rawList.map((pos) => {
            const liveRow = liveStockMap.get(pos.symbol);
            const liveQuoteIsCurrent = Boolean(
                liveRow?.price != null
                && liveRow.price > 0
                && quoteTimestampAtLeastAsRecent(liveRow.price_as_of, pos.price_as_of),
            );
            const livePrice = liveQuoteIsCurrent ? liveRow!.price : pos.current_price;
            const liveChangePct = liveQuoteIsCurrent ? liveRow!.change_pct : pos.daily_change_pct;
            const livePriceAsOf = liveQuoteIsCurrent ? liveRow!.price_as_of : pos.price_as_of;
            const effectivePrice = livePrice != null ? livePrice : pos.current_price;

            const retPct = pos.entry_price > 0 && effectivePrice != null
                ? Number((((effectivePrice - pos.entry_price) / pos.entry_price) * 100).toFixed(2))
                : pos.return_pct;
            const xuRetPct = pos.entry_xu100 && currentXu100
                ? Number((((currentXu100 - pos.entry_xu100) / pos.entry_xu100) * 100).toFixed(2))
                : pos.xu100_return_pct;
            const relRetPct = compoundAlpha(retPct, xuRetPct);
            const upsidePct = effectivePrice != null && effectivePrice > 0
                ? Number((((pos.target_price - effectivePrice) / effectivePrice) * 100).toFixed(2))
                : pos.target_upside_pct;
            const progressPct = effectivePrice != null
                ? computeProgress(pos.entry_price, effectivePrice, pos.target_price)
                : pos.target_progress_pct;
            const attrPct = retPct != null ? Number(((pos.weight_pct * retPct) / 100).toFixed(2)) : 0;

            const priceChanged = livePrice != null && livePrice !== pos.current_price;
            const changePctChanged = liveChangePct !== pos.daily_change_pct;
            const asOfChanged = livePriceAsOf !== pos.price_as_of;
            const xuRetChanged = xuRetPct !== pos.xu100_return_pct;
            const relRetChanged = relRetPct !== pos.relative_return_pct;

            if (!priceChanged && !changePctChanged && !asOfChanged && !xuRetChanged && !relRetChanged) {
                return pos;
            }

            return {
                ...pos,
                current_price: effectivePrice,
                daily_change_pct: liveChangePct,
                price_as_of: livePriceAsOf,
                return_pct: retPct,
                xu100_return_pct: xuRetPct,
                relative_return_pct: relRetPct,
                target_upside_pct: upsidePct,
                target_progress_pct: progressPct,
                target_reached: effectivePrice != null && effectivePrice >= pos.target_price,
                attribution_pct: attrPct,
            };
        });
    }, [effectiveXu100, liveStockMap, portfolioData]);

    const liveSummary = useMemo(() => {
        const base = portfolioData?.summary;
        if (!base) {
            return {
                active_count: 0,
                closed_count: 0,
                total_weight_pct: 0,
                cash_weight_pct: 100,
                portfolio_return_pct: null,
                invested_return_pct: null,
                xu100_return_pct: null,
                relative_return_pct: null,
                weighted_target_upside_pct: null,
                closed_hit_rate_pct: null,
                closed_win_count: 0,
                closed_avg_return_pct: null,
                closed_avg_relative_pct: null,
                inception_date: null,
            };
        }
        if (!liveActivePositions.length) return base;
        const totalW = Number(liveActivePositions.reduce((acc, r) => acc + (r.weight_pct || 0), 0).toFixed(2));
        const cashW = Number(Math.max(0, 100 - totalW).toFixed(2));
        const weightedRetSum = liveActivePositions.reduce(
            (acc, r) => acc + (r.return_pct != null ? r.weight_pct * r.return_pct : 0),
            0,
        );
        const closedWeightedRetSum = (portfolioData?.closed_positions || []).reduce(
            (acc, r) => acc + (r.return_pct != null ? (r.weight_pct || 0) * r.return_pct : 0),
            0,
        );
        const portRet = Number(((weightedRetSum + closedWeightedRetSum) / 100).toFixed(2));
        const invRet = totalW > 0 ? Number((weightedRetSum / totalW).toFixed(2)) : null;
        const upsideSum = liveActivePositions.reduce(
            (acc, r) => acc + (r.target_upside_pct != null ? r.weight_pct * r.target_upside_pct : 0),
            0,
        );
        const avgUpside = totalW > 0 ? Number((upsideSum / totalW).toFixed(2)) : base.weighted_target_upside_pct;

        const xu100WeightedSum = liveActivePositions.reduce(
            (acc, r) => acc + (r.xu100_return_pct != null ? (r.weight_pct || 0) * r.xu100_return_pct : 0),
            0,
        );
        const xu100WeightDenom = liveActivePositions.reduce(
            (acc, r) => acc + (r.xu100_return_pct != null ? (r.weight_pct || 0) : 0),
            0,
        );
        const liveXu100RetPct = xu100WeightDenom > 0
            ? Number((xu100WeightedSum / xu100WeightDenom).toFixed(2))
            : base.xu100_return_pct;
        const effectiveXu100RetPct = liveXu100RetPct ?? base.xu100_return_pct;
        const relRet = compoundAlpha(portRet, effectiveXu100RetPct);

        return {
            ...base,
            total_weight_pct: totalW,
            cash_weight_pct: cashW,
            portfolio_return_pct: portRet,
            invested_return_pct: invRet,
            xu100_return_pct: effectiveXu100RetPct,
            relative_return_pct: relRet,
            weighted_target_upside_pct: avgUpside,
        };
    }, [liveActivePositions, portfolioData?.closed_positions, portfolioData?.summary]);

    const selectedPosition = useMemo(() => {
        if (!selectedPositionId) return null;
        return (
            liveActivePositions.find((p) => p.id === selectedPositionId)
            || (portfolioData?.closed_positions || []).find((p) => p.id === selectedPositionId)
            || null
        );
    }, [liveActivePositions, portfolioData?.closed_positions, selectedPositionId]);

    const slicedPerfPoints = useMemo(
        () => sliceAndNormalizePerformance(perfPoints, selectedRange, liveSummary.inception_date),
        [perfPoints, selectedRange, liveSummary.inception_date],
    );

    const handleSectionChange = (section: MarketsNavigationSection) => {
        if (section === 'markets') {
            navigate(toMarketsOverview());
        } else if (section === 'funds') {
            navigate(toFunds());
        } else if (section === 'indices') {
            navigate(toMarketsIndices());
        } else {
            navigate(toMarketsStocks(DEFAULT_MARKET_INDEX));
        }
    };

    const handleAdminVerifySubmit = async (e: FormEvent) => {
        e.preventDefault();
        let cleaned = adminTokenInput.trim();
        if (cleaned.toUpperCase().startsWith('RAGFIN_ADMIN_TOKEN=')) {
            cleaned = cleaned.slice('RAGFIN_ADMIN_TOKEN='.length).trim();
        }
        cleaned = cleaned.replace(/^['"]|['"]$/g, '').trim();
        if (!cleaned) return;
        setAdminAuthLoading(true);
        setAdminAuthError(null);
        try {
            const res = await apiClient.modelPortfolioVerifyAdmin(cleaned);
            if (res.ok && res.session_token) {
                setAdminToken(res.session_token);
                setAdminSessionExpiresAt(Date.now() + res.expires_in_seconds * 1000);
                setIsAdminVerified(true);
                setAdminModalOpen(false);
                setAdminTokenInput('');
            }
        } catch (err: unknown) {
            setAdminAuthError(err instanceof Error ? err.message : 'Yönetici anahtarı doğrulanamadı.');
        } finally {
            setAdminAuthLoading(false);
        }
    };

    const handleAdminLogout = () => {
        setAdminToken('');
        setAdminSessionExpiresAt(null);
        setIsAdminVerified(false);
    };

    const handlePortfolioMutated = (nextData: ModelPortfolioResponse) => {
        setPortfolioData(nextData);
        void loadPerformance(nextData.version, true);
    };

    const handleDeletePosition = async (position: ModelPortfolioPosition) => {
        if (!adminToken) return;
        if (!window.confirm(`${position.symbol} pozisyonunu tamamen silmek istediğinize emin misiniz?`)) return;
        try {
            const next = await apiClient.modelPortfolioDeletePosition(position.id, adminToken);
            if (selectedPositionId === position.id) setSelectedPositionId(null);
            handlePortfolioMutated(next);
        } catch (err: unknown) {
            window.alert(err instanceof Error ? err.message : 'Silme işlemi başarısız oldu.');
        }
    };

    // Donut segments (Active stocks + Cash)
    const donutItems = useMemo(() => {
        const items = liveActivePositions.map((pos, idx) => ({
            symbol: pos.symbol,
            label: pos.company_name,
            weight: pos.weight_pct,
            returnPct: pos.return_pct,
            attributionPct: pos.attribution_pct ?? 0,
            color: DONUT_PALETTE[idx % DONUT_PALETTE.length],
            isCash: false,
        }));
        if (liveSummary.cash_weight_pct > 0.05) {
            const realizedCashAttr = Number(
                ((portfolioData?.closed_positions || []).reduce(
                    (acc, r) => acc + (r.return_pct != null ? (r.weight_pct || 0) * r.return_pct : 0),
                    0,
                ) / 100).toFixed(2),
            );
            items.push({
                symbol: 'NAKİT',
                label: 'Nakit / Boşta Pay',
                weight: liveSummary.cash_weight_pct,
                returnPct: 0,
                attributionPct: realizedCashAttr,
                color: '#334155',
                isCash: true,
            });
        }
        return items;
    }, [liveActivePositions, liveSummary.cash_weight_pct, portfolioData?.closed_positions]);

    const hasPortfolioSetup = Boolean(
        portfolioData?.active_positions.length || portfolioData?.closed_positions.length,
    );

    return (
        <div className={`mn-layout${navCollapsed ? ' mn-nav-collapsed' : ''}`}>
            <MarketsNavigation
                collapsed={navCollapsed}
                activeSection="markets"
                onCollapsedChange={setNavCollapsed}
                onSectionChange={handleSectionChange}
                onStockSectionChange={(section) => navigate(section === 'compare' ? toStockComparison() : toMarketsStocks(DEFAULT_MARKET_INDEX))}
                onFundSectionChange={(section) => navigate(section === 'compare' ? toFundComparison() : toFunds())}
                onSelectTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
                onSelectFund={(fundCode) => navigate(toFundDetail(fundCode, DEFAULT_FUND_TAB))}
            />
            <div className="markets-workspace markets-workspace-with-rail">
                <main className="market-page model-portfolio-page">
                    {/* Header */}
                    <header className="market-header mp-header">
                        <div className="market-title">
                            <div className="mp-title-row">
                                <h1 tabIndex={-1} data-dialog-return-focus>Model Portföy</h1>
                                {hasPortfolioSetup && portfolioData?.updated_at && (
                                    <span className="mp-update-meta">
                                        <Clock size={12} /> {formatDateTimeTR(portfolioData.updated_at)}
                                    </span>
                                )}
                            </div>
                        </div>
                    </header>

                    {error && (
                        <div className="mp-alert mp-alert-error" role="alert">
                            <AlertTriangle size={16} />
                            <span>{error}</span>
                            <button type="button" className="mp-btn mp-btn-subtle" onClick={() => loadPortfolio(true)}>
                                <RefreshCw size={13} /> Tekrar Dene
                            </button>
                        </div>
                    )}

                    {/* Portfolio Summary Strip */}
                    <section className="mp-kpi-strip" aria-label="Model portföy özet göstergeleri">
                        {[
                            {
                                label: 'Getiri',
                                value: formatPct(liveSummary.portfolio_return_pct),
                                valueClass: toneClass(liveSummary.portfolio_return_pct),
                                valueTitle: 'Nakit payı (%0 getiri) dahil toplam model portföy getirisi.',
                                sub: <>Hisse bazlı <b>{formatPct(liveSummary.invested_return_pct)}</b></>,
                            },
                            {
                                label: 'XU100 Alfa',
                                value: formatPct(liveSummary.relative_return_pct),
                                valueClass: toneClass(liveSummary.relative_return_pct),
                                subTitle: 'Her pozisyon, kendi giriş günündeki XU100 seviyesinden itibaren kıyaslanır.',
                                sub: <>XU100 <b>{formatPct(liveSummary.xu100_return_pct)}</b></>,
                            },
                            {
                                label: 'Hedef Potansiyel',
                                value: formatPct(liveSummary.weighted_target_upside_pct),
                                valueClass: toneClass(liveSummary.weighted_target_upside_pct),
                                sub: 'Nakit hariç',
                            },
                            {
                                label: 'Dağılım',
                                value: !hasPortfolioSetup
                                    ? '—'
                                    : liveSummary.active_count > 0
                                        ? `${liveSummary.active_count} hisse`
                                        : 'Nakit',
                                valueClass: !hasPortfolioSetup ? 'mp-kpi-empty-value' : '',
                                sub: !hasPortfolioSetup
                                    ? 'Pozisyon yok'
                                    : liveSummary.active_count > 0
                                        ? <>Hisse <b>%{liveSummary.total_weight_pct.toFixed(0)}</b> · Nakit <b>%{liveSummary.cash_weight_pct.toFixed(1)}</b></>
                                        : 'Tümü kapatıldı',
                            },
                            {
                                label: 'Başarı Oranı',
                                value: liveSummary.closed_hit_rate_pct != null
                                    ? `%${liveSummary.closed_hit_rate_pct.toFixed(0)}`
                                    : '—',
                                valueClass: 'mp-tone-accent',
                                sub: liveSummary.closed_count > 0
                                    ? `${liveSummary.closed_win_count}/${liveSummary.closed_count} XU100 üstü`
                                    : 'Kapalı işlem yok',
                            },
                        ].map((kpi) => (
                            <div key={kpi.label} className="mp-kpi-item">
                                <span className="mp-kpi-label">{kpi.label}</span>
                                {loading && !portfolioData ? (
                                    <>
                                        <span className="mp-skel-block mp-skel-kpi-value" aria-hidden="true" />
                                        <span className="mp-skel-block mp-skel-kpi-sub" aria-hidden="true" />
                                    </>
                                ) : (
                                    <>
                                        <strong className={`mp-kpi-value ${kpi.valueClass}`.trim()} title={kpi.valueTitle}>
                                            {kpi.value}
                                        </strong>
                                        <span className="mp-kpi-sub" title={kpi.subTitle}>
                                            {kpi.sub}
                                        </span>
                                    </>
                                )}
                            </div>
                        ))}
                    </section>

                    {/* Analytics Grid: Left = Cumulative Performance Chart, Right = Allocation Donut & Attribution */}
                    {hasPortfolioSetup && (
                        <section className="mp-analytics-grid">
                            {/* Left: Cumulative Return vs XU100 SVG Chart */}
                            <div className="mp-panel mp-chart-panel">
                                <div className="mp-panel-header">
                                    <h2
                                        className="mp-panel-title"
                                        title="Kapanan pozisyonlar çıkış tarihine kadar katkı verir; ağırlıklar güncel dağılıma göredir."
                                    >
                                        Kümülatif Performans (100 ₺)
                                    </h2>
                                    <div className="mp-range-tabs" role="tablist" aria-label="Grafik periyodu">
                                        {PERF_RANGES.map((r) => (
                                            <button
                                                key={r.id}
                                                type="button"
                                                role="tab"
                                                aria-selected={selectedRange === r.id}
                                                className={`mp-range-btn${selectedRange === r.id ? ' active' : ''}`}
                                                onClick={() => {
                                                    setSelectedRange(r.id);
                                                    setHoverIndex(null);
                                                }}
                                            >
                                                {r.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <ModelPortfolioPerformanceChart
                                    points={slicedPerfPoints}
                                    loading={perfLoading}
                                    error={perfError}
                                    onRetry={() => {
                                        if (portfolioData) void loadPerformance(portfolioData.version, true);
                                    }}
                                    hoverIndex={hoverIndex}
                                    onHoverIndexChange={setHoverIndex}
                                />
                            </div>

                            {/* Right: Allocation Donut & Return Attribution */}
                            <div className="mp-panel mp-allocation-panel">
                                <div className="mp-panel-header">
                                    <h2 className="mp-panel-title">Dağılım ve Katkı</h2>
                                </div>

                                {liveActivePositions.length > 0 ? (
                                    <div className="mp-allocation-body">
                                        <ModelPortfolioDonut
                                            items={donutItems}
                                            activeIndex={activeDonutIndex}
                                            onActiveIndexChange={setActiveDonutIndex}
                                        />

                                        <div className="mp-attribution-list">
                                            <div className="mp-attribution-head">
                                                <span>Hisse / Ağırlık</span>
                                                <span>Portföye Katkı (Puan)</span>
                                            </div>
                                            {donutItems.map((item, idx) => (
                                                <div
                                                    key={item.symbol}
                                                    className={`mp-attribution-row${activeDonutIndex === idx ? ' is-highlighted' : ''}`}
                                                    onMouseEnter={() => setActiveDonutIndex(idx)}
                                                    onMouseLeave={() => setActiveDonutIndex(null)}
                                                >
                                                    <div className="mp-attribution-left">
                                                        <span
                                                            className="mp-color-dot"
                                                            style={{ backgroundColor: item.color }}
                                                        />
                                                        <span className="mp-attr-symbol">{item.symbol}</span>
                                                        <span className="mp-attr-weight">%{item.weight.toFixed(1)}</span>
                                                    </div>
                                                    <div className="mp-attribution-right">
                                                        {item.isCash && item.attributionPct === 0 ? (
                                                            <span className="mp-tone-flat">0,00 puan</span>
                                                        ) : (
                                                            <span className={toneClass(item.attributionPct)}>
                                                                {item.attributionPct > 0 ? '+' : ''}
                                                                {item.attributionPct.toFixed(2)} puan
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="mp-allocation-body mp-allocation-empty">
                                        <p className="mp-chart-note">Aktif pozisyon bulunmuyor.</p>
                                    </div>
                                )}
                            </div>
                        </section>
                    )}

                    {/* Main Content Section with 3 Tabs */}
                    <section className="mp-panel mp-main-section">
                        <div className="mp-tabs-header">
                            <div className="mp-tabs-nav" role="tablist">
                                <button
                                    type="button"
                                    role="tab"
                                    aria-selected={activeTab === 'active'}
                                    className={`mp-tab-btn${activeTab === 'active' ? ' active' : ''}`}
                                    onClick={() => setActiveTab('active')}
                                >
                                    <span>Aktif Pozisyonlar</span>
                                    <span className="mp-tab-count">{liveActivePositions.length}</span>
                                </button>
                                <button
                                    type="button"
                                    role="tab"
                                    aria-selected={activeTab === 'closed'}
                                    className={`mp-tab-btn${activeTab === 'closed' ? ' active' : ''}`}
                                    onClick={() => setActiveTab('closed')}
                                >
                                    <span>Kapatılan Pozisyonlar</span>
                                    <span className="mp-tab-count">{portfolioData?.closed_positions.length || 0}</span>
                                </button>
                                <button
                                    type="button"
                                    role="tab"
                                    aria-selected={activeTab === 'changelog'}
                                    className={`mp-tab-btn${activeTab === 'changelog' ? ' active' : ''}`}
                                    onClick={() => setActiveTab('changelog')}
                                >
                                    <span>Değişiklik Günlüğü</span>
                                    <span className="mp-tab-count">{portfolioData?.changelog.length || 0}</span>
                                </button>
                            </div>
                            <div className="mp-tabs-meta">
                                {effectiveXu100 && (
                                    <span className="mp-xu100-pill">
                                        XU100: <b>{effectiveXu100.toLocaleString('tr-TR', { maximumFractionDigits: 2 })}</b>
                                    </span>
                                )}
                                <div className="mp-toolbar-actions">
                                    {isAdminVerified ? (
                                        <>
                                            <span className="mp-admin-status" title="Kurucu oturumu aktif">
                                                <span>Kurucu</span>
                                                {adminCountdown && (
                                                    <span
                                                        className={`mp-admin-timer mp-mono${adminRemainingSeconds != null && adminRemainingSeconds <= 120 ? ' is-expiring' : ''}`}
                                                        title="Oturum kalan süresi"
                                                    >
                                                        {adminCountdown}
                                                    </span>
                                                )}
                                            </span>
                                            <button
                                                type="button"
                                                className="mp-admin-add-btn"
                                                onClick={() => {
                                                    setEditingPosition(null);
                                                    setEditorModalOpen(true);
                                                }}
                                            >
                                                <Plus size={12} />
                                                <span>Hisse Ekle</span>
                                            </button>
                                            <button
                                                type="button"
                                                className="mp-admin-logout-btn"
                                                onClick={handleAdminLogout}
                                                title="Kurucu oturumunu kapat"
                                                aria-label="Kurucu oturumunu kapat"
                                            >
                                                <LogOut size={12} />
                                                <span>Çıkış</span>
                                            </button>
                                        </>
                                    ) : (
                                        <button
                                            type="button"
                                            className="mp-founder-trigger"
                                            onClick={() => setAdminModalOpen(true)}
                                            title="Model portföyü yönetmek için kurucu girişi yapın"
                                        >
                                            <Lock size={12} />
                                            <span>Kurucu Girişi</span>
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* TAB 1: Active Positions */}
                        {activeTab === 'active' && (
                            loading && !portfolioData ? (
                                <div className="mp-table-wrap" role="status" aria-busy="true" aria-label="Model portföy verileri yükleniyor">
                                    <table className="mp-table mp-active-table mp-table-skeleton">
                                        <thead>
                                            <tr>
                                                <th>Hisse</th>
                                                <th className="num">Giriş</th>
                                                <th className="num">Güncel</th>
                                                <th className="mp-col-target">Hedef ve potansiyel</th>
                                                <th className="num">Ağırlık</th>
                                                <th className="num">Net getiri</th>
                                                <th className="num">XU100’e göre</th>
                                                <th className="act"><span className="mp-sr-only">İşlemler</span></th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {[0, 1, 2, 3, 4, 5].map((idx) => (
                                                <tr key={idx} className="mp-row mp-skeleton-row" aria-hidden="true">
                                                    <td>
                                                        <div className="mp-stock-cell">
                                                            <span className="mp-skel-block mp-skel-logo" />
                                                            <div className="mp-stock-meta">
                                                                <span className="mp-skel-block mp-skel-ticker" />
                                                                <span className="mp-skel-block mp-skel-company" />
                                                            </div>
                                                        </div>
                                                    </td>
                                                    <td className="num" data-label="Giriş">
                                                        <span className="mp-skel-block mp-skel-num" />
                                                        <span className="mp-skel-block mp-skel-sub" />
                                                    </td>
                                                    <td className="num" data-label="Güncel">
                                                        <span className="mp-skel-block mp-skel-num" />
                                                    </td>
                                                    <td className="mp-col-target" data-label="Hedef ve potansiyel">
                                                        <span className="mp-skel-block mp-skel-target" />
                                                    </td>
                                                    <td className="num" data-label="Ağırlık">
                                                        <span className="mp-skel-block mp-skel-num-sm" />
                                                    </td>
                                                    <td className="num" data-label="Net getiri">
                                                        <span className="mp-skel-block mp-skel-num-sm" />
                                                    </td>
                                                    <td className="num" data-label="XU100’e göre">
                                                        <span className="mp-skel-block mp-skel-num-sm" />
                                                    </td>
                                                    <td className="act">
                                                        <span className="mp-skel-block mp-skel-act" />
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            ) : liveActivePositions.length === 0 ? (
                                <div className="mp-empty-box">
                                    <BriefcaseBusiness size={26} className="mp-empty-icon" />
                                    <h3>{hasPortfolioSetup ? 'Aktif Pozisyon Yok' : 'Model Portföy Boş'}</h3>
                                    <p>
                                        {hasPortfolioSetup
                                            ? 'Tüm pozisyonlar kapatılmış. Geçmiş işlemler Kapatılanlar sekmesinde.'
                                            : isAdminVerified
                                                ? 'Hisse, hedef fiyat ve yatırım tezi ekleyerek başlayın.'
                                                : 'Henüz aktif hisse önerisi bulunmuyor.'}
                                    </p>
                                    {isAdminVerified && !hasPortfolioSetup && (
                                        <button
                                            type="button"
                                            className="mp-btn mp-btn-primary"
                                            onClick={() => {
                                                setEditingPosition(null);
                                                setEditorModalOpen(true);
                                            }}
                                        >
                                            <Plus size={15} /> İlk Hisseyi Ekle
                                        </button>
                                    )}
                                </div>
                            ) : (
                                <>
                                <div className="mp-table-wrap" role="region" aria-label="Aktif pozisyon tablosu; yatay kaydırılabilir" tabIndex={0}>
                                    <p className="mp-table-scroll-hint">Tabloyu yana kaydırarak tüm sütunları görebilirsiniz.</p>
                                    <table className="mp-table mp-active-table">
                                        <thead>
                                            <tr>
                                                <th>Hisse</th>
                                                <th className="num">Giriş</th>
                                                <th className="num">Güncel</th>
                                                <th className="mp-col-target">Hedef ve potansiyel</th>
                                                <th className="num">Ağırlık</th>
                                                <th className="num">Net getiri</th>
                                                <th className="num">XU100’e göre</th>
                                                <th className="act"><span className="mp-sr-only">İşlemler</span></th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {liveActivePositions.map((pos) => {
                                                const progress = pos.target_progress_pct ?? 0;
                                                const clampedBar = Math.max(0, Math.min(100, progress));
                                                const isSelected = selectedPositionId === pos.id;
                                                return (
                                                    <tr
                                                        key={pos.id}
                                                        className={`mp-row${isSelected ? ' is-selected' : ''}`}
                                                        tabIndex={0}
                                                        aria-label={`${pos.symbol} pozisyon ayrıntılarını aç`}
                                                        onClick={() => setSelectedPositionId(pos.id)}
                                                        onKeyDown={(e) => {
                                                            if ((e.key === 'Enter' || e.key === ' ') && e.target === e.currentTarget) {
                                                                e.preventDefault();
                                                                setSelectedPositionId(pos.id);
                                                            }
                                                        }}
                                                    >
                                                        <td>
                                                            <div className="mp-stock-cell">
                                                                <SymbolLogo
                                                                    symbol={pos.symbol}
                                                                    name={pos.company_name}
                                                                    kind="stock"
                                                                    logoUrl={pos.logo_url}
                                                                    size="sm"
                                                                />
                                                                <div className="mp-stock-meta">
                                                                    <Link
                                                                        to={toStockDetail(pos.symbol, DEFAULT_STOCK_TAB)}
                                                                        className="mp-ticker"
                                                                        onClick={(event) => event.stopPropagation()}
                                                                    >
                                                                        {pos.symbol}
                                                                    </Link>
                                                                    <span className="mp-company-name" title={pos.sector ? `${pos.company_name} · ${pos.sector}` : pos.company_name}>
                                                                        {pos.company_name}
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className="num mp-mono" data-label="Giriş">
                                                            <div>{formatPrice(pos.entry_price)}</div>
                                                            <span className="mp-sub-cell">
                                                                {formatCompactDateTR(pos.entry_date)} · {pos.holding_days} gün
                                                            </span>
                                                        </td>
                                                        <td
                                                            className="num mp-mono"
                                                            data-label="Güncel"
                                                            title={pos.price_as_of ? `Fiyat zamanı: ${formatCompactDateTimeTR(pos.price_as_of)}` : undefined}
                                                        >
                                                            <div className="mp-price-main">{formatPrice(pos.current_price)}</div>
                                                            {pos.daily_change_pct != null && (
                                                                <span className={`mp-sub-cell ${toneClass(pos.daily_change_pct)}`}>
                                                                    {formatPct(pos.daily_change_pct)} bugün
                                                                </span>
                                                            )}
                                                        </td>
                                                        <td className="mp-col-target" data-label="Hedef ve potansiyel">
                                                            <div className="mp-target-cell">
                                                                <div className="mp-target-top">
                                                                    <span className="mp-mono mp-target-price">
                                                                        {formatPrice(pos.target_price)}
                                                                    </span>
                                                                    {pos.target_reached ? (
                                                                        <span className="mp-reached-badge">
                                                                            <CheckCircle2 size={11} /> Hedefe ulaştı
                                                                        </span>
                                                                    ) : (
                                                                        <span className={`mp-target-upside mp-mono ${toneClass(pos.target_upside_pct)}`}>
                                                                            {formatPct(pos.target_upside_pct)}
                                                                        </span>
                                                                    )}
                                                                </div>
                                                                <div
                                                                    className="mp-progress-track"
                                                                    title={`Hedefe ilerleme: ${formatPct(progress, false)}`}
                                                                >
                                                                    <div
                                                                        className={`mp-progress-fill${pos.target_reached ? ' is-reached' : ''}`}
                                                                        style={{ width: `${clampedBar}%` }}
                                                                    />
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className="num mp-mono" data-label="Ağırlık">
                                                            %{pos.weight_pct.toLocaleString('tr-TR', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
                                                        </td>
                                                        <td className={`num mp-mono mp-bold ${toneClass(pos.return_pct)}`} data-label="Net getiri">
                                                            {formatPct(pos.return_pct)}
                                                        </td>
                                                        <td className={`num mp-mono mp-bold ${toneClass(pos.relative_return_pct)}`} data-label="XU100’e göre">
                                                            {formatPct(pos.relative_return_pct)}
                                                        </td>
                                                        <td className="act" onClick={(e) => e.stopPropagation()}>
                                                            <div className="mp-row-actions">
                                                                {isAdminVerified && (
                                                                    <>
                                                                        <button
                                                                            type="button"
                                                                            className="mp-icon-btn"
                                                                            title="Pozisyonu Düzenle / Hedef Revize Et"
                                                                            aria-label={`${pos.symbol} pozisyonunu düzenle`}
                                                                            onClick={() => {
                                                                                setEditingPosition(pos);
                                                                                setEditorModalOpen(true);
                                                                            }}
                                                                        >
                                                                            <Pencil size={13} />
                                                                        </button>
                                                                        <button
                                                                            type="button"
                                                                            className="mp-icon-btn mp-icon-btn-warn"
                                                                            title="Pozisyonu Kapat"
                                                                            aria-label={`${pos.symbol} pozisyonunu kapat`}
                                                                            onClick={() => setClosingPosition(pos)}
                                                                        >
                                                                            <CheckCircle2 size={13} />
                                                                        </button>
                                                                    </>
                                                                )}
                                                                <button
                                                                    type="button"
                                                                    className="mp-thesis-btn"
                                                                    onClick={() => setSelectedPositionId(pos.id)}
                                                                >
                                                                    <span>Tez</span>
                                                                    <ChevronRight size={14} />
                                                                </button>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                                {(() => {
                                    const latestPriceAsOf = liveActivePositions
                                        .map((p) => p.price_as_of)
                                        .filter((v): v is string => Boolean(v))
                                        .sort()
                                        .pop();
                                    return (
                                        <p className="mp-table-footnote">
                                            Tez detayı için satıra tıklayın
                                            {latestPriceAsOf ? ` · ${formatCompactDateTimeTR(latestPriceAsOf)}` : ''}
                                        </p>
                                    );
                                })()}
                                </>
                            )
                        )}

                        {/* TAB 2: Closed Positions */}
                        {activeTab === 'closed' && (
                            loading && !portfolioData ? (
                                <div className="mp-table-wrap" role="status" aria-busy="true" aria-label="Kapatılan pozisyonlar yükleniyor">
                                    <table className="mp-table mp-closed-table mp-table-skeleton">
                                        <thead>
                                            <tr>
                                                <th>Hisse</th>
                                                <th className="num">Tarih</th>
                                                <th className="num">Giriş</th>
                                                <th className="num">Çıkış</th>
                                                <th className="num">Net getiri</th>
                                                <th className="num">XU100’e göre</th>
                                                <th>Gerekçe</th>
                                                <th className="act"><span className="mp-sr-only">İşlemler</span></th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {[0, 1, 2, 3].map((idx) => (
                                                <tr key={idx} className="mp-row mp-skeleton-row" aria-hidden="true">
                                                    <td>
                                                        <div className="mp-stock-cell">
                                                            <span className="mp-skel-block mp-skel-logo" />
                                                            <div className="mp-stock-meta">
                                                                <span className="mp-skel-block mp-skel-ticker" />
                                                                <span className="mp-skel-block mp-skel-company" />
                                                            </div>
                                                        </div>
                                                    </td>
                                                    <td className="num" data-label="Tarih">
                                                        <span className="mp-skel-block mp-skel-num" />
                                                        <span className="mp-skel-block mp-skel-sub" />
                                                    </td>
                                                    <td className="num" data-label="Giriş"><span className="mp-skel-block mp-skel-num-sm" /></td>
                                                    <td className="num" data-label="Çıkış"><span className="mp-skel-block mp-skel-num-sm" /></td>
                                                    <td className="num" data-label="Net getiri"><span className="mp-skel-block mp-skel-num-sm" /></td>
                                                    <td className="num" data-label="XU100’e göre"><span className="mp-skel-block mp-skel-num-sm" /></td>
                                                    <td data-label="Gerekçe"><span className="mp-skel-block mp-skel-target" /></td>
                                                    <td className="act"><span className="mp-skel-block mp-skel-act" /></td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            ) : (portfolioData?.closed_positions.length || 0) === 0 ? (
                                <div className="mp-empty-box">
                                    <Award size={24} className="mp-empty-icon" />
                                    <h3>Kapatılan Pozisyon Yok</h3>
                                    <p>Kapatılan işlemler gerçekleşen getiri ve alfa verileriyle burada listelenir.</p>
                                </div>
                            ) : (
                                <div className="mp-closed-section">
                                    <div className="mp-closed-scorecard">
                                        <div className="mp-score-item">
                                            <span>Kapatılan öneri</span>
                                            <strong>{liveSummary.closed_count} işlem</strong>
                                        </div>
                                        <div className="mp-score-item">
                                            <span>BIST 100 üzeri başarı</span>
                                            <strong className="mp-tone-up">
                                                %{liveSummary.closed_hit_rate_pct?.toFixed(1) ?? '0.0'} ({liveSummary.closed_win_count}/{liveSummary.closed_count})
                                            </strong>
                                        </div>
                                        <div className="mp-score-item">
                                            <span>Ortalama net getiri</span>
                                            <strong className={toneClass(liveSummary.closed_avg_return_pct)}>
                                                {formatPct(liveSummary.closed_avg_return_pct)}
                                            </strong>
                                        </div>
                                        <div className="mp-score-item">
                                            <span>Ortalama alfa</span>
                                            <strong className={toneClass(liveSummary.closed_avg_relative_pct)}>
                                                {formatPct(liveSummary.closed_avg_relative_pct)}
                                            </strong>
                                        </div>
                                    </div>

                                    <p className="mp-table-scroll-hint">Tabloyu yana kaydırarak tüm sütunları görebilirsiniz.</p>
                                    <div className="mp-table-wrap" role="region" aria-label="Kapatılmış pozisyon tablosu; yatay kaydırılabilir" tabIndex={0}>
                                        <table className="mp-table mp-closed-table">
                                            <thead>
                                                <tr>
                                                    <th>Hisse</th>
                                                    <th className="num">Tarih</th>
                                                    <th className="num">Giriş</th>
                                                    <th className="num">Çıkış</th>
                                                    <th className="num">Net getiri</th>
                                                    <th className="num">XU100’e göre</th>
                                                    <th>Gerekçe</th>
                                                    <th className="act"><span className="mp-sr-only">İşlemler</span></th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {(portfolioData?.closed_positions || []).map((pos) => (
                                                    <tr
                                                        key={pos.id}
                                                        className="mp-row"
                                                        tabIndex={0}
                                                        aria-label={`${pos.symbol} kapanmış pozisyon ayrıntılarını aç`}
                                                        onClick={() => setSelectedPositionId(pos.id)}
                                                        onKeyDown={(e) => {
                                                            if ((e.key === 'Enter' || e.key === ' ') && e.target === e.currentTarget) {
                                                                e.preventDefault();
                                                                setSelectedPositionId(pos.id);
                                                            }
                                                        }}
                                                    >
                                                        <td>
                                                            <div className="mp-stock-cell">
                                                                <SymbolLogo
                                                                    symbol={pos.symbol}
                                                                    name={pos.company_name}
                                                                    kind="stock"
                                                                    logoUrl={pos.logo_url}
                                                                    size="sm"
                                                                />
                                                                <div className="mp-stock-meta">
                                                                    <Link
                                                                        to={toStockDetail(pos.symbol, DEFAULT_STOCK_TAB)}
                                                                        className="mp-ticker"
                                                                        onClick={(event) => event.stopPropagation()}
                                                                    >
                                                                        {pos.symbol}
                                                                    </Link>
                                                                    <span className="mp-company-name">{pos.company_name}</span>
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className="num mp-mono" data-label="Tarih">
                                                            <div>
                                                                {formatDateTR(pos.entry_date)} → {formatDateTR(pos.closed_date)}
                                                            </div>
                                                            <span className="mp-sub-cell">{pos.holding_days} gün tutuldu</span>
                                                        </td>
                                                        <td className="num mp-mono" data-label="Giriş">{formatPrice(pos.entry_price)}</td>
                                                        <td className="num mp-mono" data-label="Çıkış">{formatPrice(pos.exit_price)}</td>
                                                        <td className={`num mp-mono mp-bold ${toneClass(pos.return_pct)}`} data-label="Net getiri">
                                                            {formatPct(pos.return_pct)}
                                                        </td>
                                                        <td className={`num mp-mono mp-bold ${toneClass(pos.relative_return_pct)}`} data-label="XU100’e göre">
                                                            {formatPct(pos.relative_return_pct)}
                                                        </td>
                                                        <td data-label="Gerekçe">
                                                            <span className="mp-close-reason">
                                                                {pos.close_reason || 'Portföy Değişikliği'}
                                                            </span>
                                                        </td>
                                                        <td className="act" onClick={(e) => e.stopPropagation()}>
                                                            <div className="mp-row-actions">
                                                                {isAdminVerified && (
                                                                    <button
                                                                        type="button"
                                                                        className="mp-icon-btn mp-icon-btn-danger"
                                                                        title="Kaydı Sil"
                                                                        aria-label={`${pos.symbol} kaydını sil`}
                                                                        onClick={() => handleDeletePosition(pos)}
                                                                    >
                                                                        <Trash2 size={13} />
                                                                    </button>
                                                                )}
                                                                <button
                                                                    type="button"
                                                                    className="mp-thesis-btn"
                                                                    onClick={() => setSelectedPositionId(pos.id)}
                                                                >
                                                                    <span>Tez</span>
                                                                    <ChevronRight size={14} />
                                                                </button>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            )
                        )}

                        {/* TAB 3: Changelog Timeline */}
                        {activeTab === 'changelog' && (
                            loading && !portfolioData ? (
                                <div className="mp-changelog-list" role="status" aria-busy="true" aria-label="Değişiklik günlüğü yükleniyor">
                                    {[0, 1, 2, 3].map((idx) => (
                                        <div key={idx} className="mp-changelog-item" aria-hidden="true">
                                            <div className="mp-changelog-meta">
                                                <span className="mp-skel-block mp-skel-badge" />
                                                <span className="mp-skel-block mp-skel-num-sm" />
                                            </div>
                                            <div className="mp-changelog-body">
                                                <span className="mp-skel-block mp-skel-logo" />
                                                <div className="mp-changelog-content">
                                                    <div className="mp-changelog-top">
                                                        <span className="mp-skel-block mp-skel-ticker" />
                                                        <span className="mp-skel-block mp-skel-company" />
                                                    </div>
                                                    <span className="mp-skel-block mp-skel-summary" />
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (portfolioData?.changelog.length || 0) === 0 ? (
                                <div className="mp-empty-box">
                                    <History size={24} className="mp-empty-icon" />
                                    <h3>Değişiklik Kaydı Yok</h3>
                                    <p>Yeni giriş, hedef/ağırlık revizyonu ve kapanış hareketleri burada listelenir.</p>
                                </div>
                            ) : (
                                <div className="mp-changelog-list">
                                    {(portfolioData?.changelog || []).map((entry: ModelPortfolioChangelogItem) => {
                                        const cleanSummary = entry.summary.startsWith(`${entry.symbol} `)
                                            ? entry.summary.slice(entry.symbol.length + 1)
                                            : entry.summary;
                                        const formattedSummary = cleanSummary.replace(/(\d+)\.(\d+)/g, '$1,$2');
                                        return (
                                            <div key={entry.id} className="mp-changelog-item">
                                                <div className="mp-changelog-meta">
                                                    <span className={`mp-changelog-badge is-${entry.action}`}>
                                                        {entry.action === 'added' && 'Yeni Giriş'}
                                                        {entry.action === 'target_revised' && 'Hedef Revizyonu'}
                                                        {entry.action === 'weight_revised' && 'Ağırlık Revizyonu'}
                                                        {entry.action === 'updated' && 'Güncelleme'}
                                                        {entry.action === 'closed' && 'Pozisyon Kapanışı'}
                                                    </span>
                                                    <time className="mp-changelog-date mp-mono">
                                                        {formatDateTR(entry.date)}
                                                    </time>
                                                </div>
                                                <div className="mp-changelog-body">
                                                    <SymbolLogo
                                                        symbol={entry.symbol}
                                                        name={entry.company_name}
                                                        kind="stock"
                                                        size="sm"
                                                    />
                                                    <div className="mp-changelog-content">
                                                        <div className="mp-changelog-top">
                                                            <Link
                                                                to={toStockDetail(entry.symbol, DEFAULT_STOCK_TAB)}
                                                                className="mp-changelog-ticker mp-mono"
                                                            >
                                                                {entry.symbol}
                                                            </Link>
                                                            {entry.company_name && (
                                                                <span className="mp-changelog-company" title={entry.company_name}>
                                                                    {entry.company_name}
                                                                </span>
                                                            )}
                                                        </div>
                                                        <p className="mp-changelog-summary">{formattedSummary}</p>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            )
                        )}
                    </section>
                </main>

                {/* Right Rail (Includes MarketFlowPanel with the new 'Model Portföy' KAP Filter!) */}
                <MarketWatchRail
                    xu100Rows={xu100UniverseRows}
                    onSelectTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
                    onSelectFund={(fundCode) => navigate(toFundDetail(fundCode, DEFAULT_FUND_TAB))}
                    showMobileToggle={false}
                />
            </div>

            {/* Right-Side Investment Thesis & Multiples Comparison Drawer */}
            {selectedPosition && (
                <ModelPortfolioThesisDrawer
                    position={selectedPosition}
                    isAdmin={isAdminVerified}
                    onClose={() => setSelectedPositionId(null)}
                    onEdit={(pos) => {
                        document.querySelector<HTMLElement>('[data-dialog-return-focus]')?.focus();
                        setSelectedPositionId(null);
                        setEditingPosition(pos);
                        setEditorModalOpen(true);
                    }}
                    onClosePosition={(pos) => {
                        document.querySelector<HTMLElement>('[data-dialog-return-focus]')?.focus();
                        setSelectedPositionId(null);
                        setClosingPosition(pos);
                    }}
                    onDeletePosition={handleDeletePosition}
                />
            )}

            {/* Founder Login Modal */}
            {adminModalOpen && (
                <div className="mp-modal-backdrop" onClick={() => setAdminModalOpen(false)}>
                    <div
                        ref={adminDialogRef}
                        className="mp-modal mp-modal-sm"
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="mp-admin-dialog-title"
                        tabIndex={-1}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="mp-modal-header">
                            <div className="mp-modal-title-row">
                                <div className="mp-modal-icon-badge is-lock">
                                    <KeyRound size={15} />
                                </div>
                                <h3 id="mp-admin-dialog-title">Kurucu Girişi</h3>
                            </div>
                            <button type="button" className="mp-icon-btn" aria-label="Giriş penceresini kapat" onClick={() => setAdminModalOpen(false)}>
                                <X size={15} />
                            </button>
                        </div>
                        <form onSubmit={handleAdminVerifySubmit} className="mp-modal-body">
                            <label className="mp-field">
                                <span className="mp-field-label">Yönetici Anahtarı</span>
                                <div className="mp-password-input-wrap">
                                    <input
                                        type={showAdminToken ? 'text' : 'password'}
                                        data-dialog-autofocus
                                        value={adminTokenInput}
                                        onChange={(e) => setAdminTokenInput(e.target.value)}
                                        placeholder="Anahtarı girin"
                                        autoComplete="off"
                                        autoFocus
                                    />
                                    <button
                                        type="button"
                                        className="mp-password-toggle"
                                        aria-label={showAdminToken ? 'Anahtarı gizle' : 'Anahtarı göster'}
                                        onClick={() => setShowAdminToken((v) => !v)}
                                    >
                                        {showAdminToken ? <EyeOff size={15} /> : <Eye size={15} />}
                                    </button>
                                </div>
                            </label>
                            {adminAuthError && <div className="mp-form-error">{adminAuthError}</div>}
                            <div className="mp-modal-footer">
                                <button
                                    type="button"
                                    className="mp-btn mp-btn-ghost"
                                    onClick={() => setAdminModalOpen(false)}
                                >
                                    İptal
                                </button>
                                <button
                                    type="submit"
                                    className="mp-btn mp-btn-primary"
                                    disabled={adminAuthLoading || !adminTokenInput.trim()}
                                >
                                    {adminAuthLoading ? 'Doğrulanıyor...' : 'Giriş Yap'}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}

            {/* Add / Edit Position Modal */}
            {editorModalOpen && (
                <Suspense fallback={<div className="mp-modal-backdrop" role="status" aria-live="polite" aria-label="Pencere yükleniyor" />}>
                    <ModelPortfolioEditorModal
                        adminToken={adminToken}
                        editingPosition={editingPosition}
                        availableWeightPct={
                            editingPosition
                                ? Math.max(0, 100 - liveSummary.total_weight_pct + editingPosition.weight_pct)
                                : Math.max(0, 100 - liveSummary.total_weight_pct)
                        }
                        universeRows={xu100UniverseRows}
                        onClose={() => {
                            setEditorModalOpen(false);
                            setEditingPosition(null);
                        }}
                        onSaved={(nextData) => {
                            setEditorModalOpen(false);
                            setEditingPosition(null);
                            handlePortfolioMutated(nextData);
                        }}
                    />
                </Suspense>
            )}

            {/* Close Position Modal */}
            {closingPosition && (
                <Suspense fallback={<div className="mp-modal-backdrop" role="status" aria-live="polite" aria-label="Pencere yükleniyor" />}>
                    <ModelPortfolioCloseModal
                        adminToken={adminToken}
                        position={closingPosition}
                        currentXu100={effectiveXu100}
                        onClose={() => setClosingPosition(null)}
                        onClosed={(nextData) => {
                            setClosingPosition(null);
                            setSelectedPositionId(null);
                            handlePortfolioMutated(nextData);
                        }}
                    />
                </Suspense>
            )}
        </div>
    );
}
