import {
    Suspense,
    lazy,
    useCallback,
    useDeferredValue,
    useEffect,
    useLayoutEffect,
    useMemo,
    useRef,
    useState,
    type CSSProperties,
    type MouseEvent as ReactMouseEvent,
} from 'react';
import {
    AlertCircle,
    ChevronRight,
    Clock3,
    Database,
    RefreshCw,
    ShieldAlert,
    SlidersHorizontal,
    Star,
    X,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { apiClient } from '../api/client';
import { useBackgroundRefresh } from '../hooks/useBackgroundRefresh';
import type {
    FundAllocationsHistoryResponse,
    FundAllocationsResponse,
    FundCategoriesResponse,
    FundDetail,
    FundHoldingsResponse,
    FundPerformanceResponse,
    FundSummary,
    FundYieldSummaryResponse,
    FundsResponse,
    MarketComparisonHistoryResponse,
} from '../api/types';
import MarketsNavigation from '../components/MarketsNavigation';
import SymbolLogo from '../components/SymbolLogo';
import { FinLoader } from '../components/FinLoader';
import FundsListLoader from '../components/FundsListLoader';
import FundRequestState from '../components/FundRequestState';
import { buildDocumentTitle, formatTitleCurrency, formatTitlePct, useDocumentTitle } from '../hooks/useDocumentTitle';
import { useWatchlist } from '../hooks/useWatchlist';
import { toFundDetail } from '../routing/routes';
import {
    canonicalFundPrice,
    buildFundHistoryDiagnosticLines,
    isFintablesHistoryPending,
    mergeFundPerformancePayloads,
} from '../utils/fundPresentation';
import './FundsPage.css';
import {
    DETAIL_RETURN_PERIODS,
    EMPTY_FUND_SCREENER_FILTERS,
    EXTENDED_DETAIL_RETURN_PERIODS,
    FUND_HISTORY_TABS,
    FUND_HOLDINGS_LIVE_MAX_BACKOFF_MS,
    FUND_HOLDINGS_LIVE_REFRESH_MS,
    FUND_HOLDINGS_REPORT_REFRESH_MS,
    FUND_INITIAL_PERFORMANCE_MONTHS,
    FUND_SCREENER_PRESETS,
    FUND_SCREENER_RANGE_SPECS,
    FUND_TABS,
    type FundChartRange,
    type FundDetailReturnPeriod,
    type FundHistorySubtab,
    type FundHoldingCategoryFilter,
    type FundHoldingChangeFilter,
    type FundScreenerFilterState,
    type FundScreenerPresetId,
    type FundSortKey,
    type FundTableMode,
    type FundsPageProps,
} from './funds/types';
import {
    compareFundRows,
    fetchFundsCatalog,
    filterPointsForRange,
    formatAllocationWeight,
    formatCompactCurrency,
    formatDate,
    formatPct,
    formatQuotePrice,
    FUNDS_CATALOG_MEMORY_TTL_MS,
    getFundsCatalogMemoryCache,
    hasFullHistoryPerformance,
    hasUsableRangeCoverage,
    isBistMarketOpen,
    isoDateMonthsAgo,
    latestFundDate,
    mergeFundHoldingsLivePayload,
    monthlyReturnsFromPoints,
    normalizeCompareSearch,
    overviewMetricsFromPoints,
    parseFundScreenerNumber,
    pctClass,
    periodReturnsFromPerformancePoints,
    rangeStartDate,
    returnBetween,
    shiftIsoDateDays,
    shouldRefreshSnapshot,
    sortFundPoints,
    updateFundsCatalogSnapshot,
} from './funds/formatters';
import { FundsSearchInput, FundsTable } from './funds/components/FundsTable';
import { FundsScreenerModal } from './funds/components/FundsScreenerModal';
import { FundReturnComparison, useFundComparisonState } from './funds/components/FundReturnComparison';
import {
    FundCategoryRankingsPanel,
    FundHistoryTable,
    FundMonthlyHeatmap,
    FundOverviewMetrics,
    FundPerformanceChart,
    FundPeriodStatsStrip,
} from './funds/sections/FundPerformanceSection';
import {
    FundAllocationHistoryPanel,
    FundAllocationSummary,
    FundHoldingsPanel,
    FundOverviewHoldingsSection,
} from './funds/sections/FundHoldingsSection';

export type { FundsPageProps };

const FundComparisonSection = lazy(() => import('./funds/sections/FundComparisonSection'));

export default function FundsPage({
    view = 'list',
    fundCode,
    activeTab = 'overview',
    onOpenFund,
    onTabChange,
    onBack,
    onNavigateSection,
    onNavigateStockSection,
    onNavigateFundSection,
    onOpenTicker,
}: FundsPageProps) {
    const [navCollapsed, setNavCollapsed] = useState(false);
    const watchlist = useWatchlist();
    const [funds, setFunds] = useState<FundsResponse | null>(null);
    const [categories, setCategories] = useState<FundCategoriesResponse | null>(null);
    const [detail, setDetail] = useState<FundDetail | null>(null);
    const [performance, setPerformance] = useState<FundPerformanceResponse | null>(null);
    const [heatmapPerformance, setHeatmapPerformance] = useState<FundPerformanceResponse | null>(null);
    const [yieldSummary, setYieldSummary] = useState<FundYieldSummaryResponse | null>(null);
    const [allocations, setAllocations] = useState<FundAllocationsResponse | null>(null);
    const [allocationHistory, setAllocationHistory] = useState<FundAllocationsHistoryResponse | null>(null);
    const [holdings, setHoldings] = useState<FundHoldingsResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [detailLoading, setDetailLoading] = useState(false);
    const [performanceLoading, setPerformanceLoading] = useState(false);
    const [yieldLoading, setYieldLoading] = useState(false);
    const [allocationLoading, setAllocationLoading] = useState(false);
    const [holdingsLoading, setHoldingsLoading] = useState(() => Boolean(fundCode));
    const [error, setError] = useState<string | null>(null);
    const [refreshError, setRefreshError] = useState<string | null>(null);
    const [detailError, setDetailError] = useState<string | null>(null);
    const [performanceError, setPerformanceError] = useState<string | null>(null);
    const [historyBackfillError, setHistoryBackfillError] = useState<string | null>(null);
    const [yieldError, setYieldError] = useState<string | null>(null);
    const [holdingsError, setHoldingsError] = useState<string | null>(null);
    const [historySubtab, setHistorySubtab] = useState<FundHistorySubtab>('prices');
    const [allocationHistoryLoading, setAllocationHistoryLoading] = useState(false);
    const [allocationHistoryError, setAllocationHistoryError] = useState<string | null>(null);
    const [chartRange, setChartRange] = useState<FundChartRange>('1w');
    const [pendingChartRange, setPendingChartRange] = useState<FundChartRange | null>(null);
    const [chartRangeError, setChartRangeError] = useState<string | null>(null);
    const [customStartDate, setCustomStartDate] = useState(isoDateMonthsAgo(6));
    const [customEndDate, setCustomEndDate] = useState(new Date().toISOString().slice(0, 10));
    const [searchTerm, setSearchTerm] = useState('');
    const [fundTypeFilter, setFundTypeFilter] = useState('');
    const [riskFilter, setRiskFilter] = useState('');
    const [fundTableMode, setFundTableMode] = useState<FundTableMode>('summary');
    const [fundScreenerOpen, setFundScreenerOpen] = useState(false);
    const [fundScreenerFilters, setFundScreenerFilters] = useState<FundScreenerFilterState>(EMPTY_FUND_SCREENER_FILTERS);
    const [activeFundScreenerPreset, setActiveFundScreenerPreset] = useState<FundScreenerPresetId | null>(null);
    const [sortKey, setSortKey] = useState<FundSortKey>('fund_code');
    const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
    const deferredSearchTerm = useDeferredValue(searchTerm);
    const mountedRef = useRef(false);
    const holdingsRef = useRef<FundHoldingsResponse | null>(null);
    const activeFundCodeRef = useRef('');
    const fundDetailRequestIdRef = useRef(0);
    const comparisonPanelRef = useRef<HTMLDivElement | null>(null);
    const fundsPageRef = useRef<HTMLDivElement | null>(null);
    const [comparisonHistory, setComparisonHistory] = useState<MarketComparisonHistoryResponse | null>(null);
    const [comparisonHistoryLoading, setComparisonHistoryLoading] = useState(false);
    const [comparisonHistoryError, setComparisonHistoryError] = useState<string | null>(null);
    const [holdingCategoryFilter, setHoldingCategoryFilter] = useState<FundHoldingCategoryFilter>('all');
    const [holdingChangeFilter, setHoldingChangeFilter] = useState<FundHoldingChangeFilter>('all');
    const holdingsFetchAttemptedRef = useRef(new Set<string>());
    const holdingsRefreshInFlightRef = useRef(false);
    const holdingsLiveInFlightRef = useRef(false);
    const holdingsLiveFailureCountRef = useRef(0);
    const chartRequestIdRef = useRef(0);

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
        };
    }, []);

    useLayoutEffect(() => {
        const resetScroll = () => {
            fundsPageRef.current?.scrollTo({ top: 0, left: 0, behavior: 'auto' });
            window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
            document.documentElement.scrollTop = 0;
            document.body.scrollTop = 0;
        };
        resetScroll();
    }, [fundCode, activeTab, view]);

    useEffect(() => {
        holdingsRef.current = holdings;
    }, [holdings]);

    const refreshSnapshot = useCallback(async (options: { rethrow?: boolean } = {}) => {
        setRefreshing(true);
        setRefreshError(null);
        try {
            // Snapshot refresh remains an admin/workflow operation. Public
            // clients reread the current canonical list without a secret.
            const { funds: fundPayload, categories: categoryPayload } = await fetchFundsCatalog({ bypassMemoryCache: true });
            if (!mountedRef.current) return;
            setFunds(fundPayload);
            setCategories(categoryPayload);
            return { fundPayload, categoryPayload };
        } catch (err) {
            if (!mountedRef.current) return;
            const message = err instanceof Error ? err.message : 'Fon listesi yenilenemedi.';
            setRefreshError(message);
            if (options.rethrow) {
                throw err instanceof Error ? err : new Error(message);
            }
        } finally {
            if (mountedRef.current) setRefreshing(false);
        }
    }, []);

    const loadHoldings = useCallback(async (
        normalizedCode: string,
        options: { silent?: boolean; force?: boolean } = {},
    ) => {
        if (!normalizedCode) return;
        const silent = Boolean(options.silent);
        if (holdingsRefreshInFlightRef.current) {
            if (!silent) setHoldingsLoading(true);
            return;
        }
        holdingsRefreshInFlightRef.current = true;
        holdingsFetchAttemptedRef.current.add(normalizedCode);
        if (!silent) setHoldingsLoading(true);
        setHoldingsError(null);
        try {
            const payload = await apiClient.fundHoldings(normalizedCode, { force: options.force });
            if (!mountedRef.current || activeFundCodeRef.current !== normalizedCode) return;
            setHoldings(payload);
        } catch (err) {
            if (!mountedRef.current || activeFundCodeRef.current !== normalizedCode) return;
            if (!silent) setHoldings(null);
            setHoldingsError(err instanceof Error ? err.message : 'KAP fon içeriği alınamadı.');
        } finally {
            holdingsRefreshInFlightRef.current = false;
            if (mountedRef.current && activeFundCodeRef.current === normalizedCode && !silent) {
                setHoldingsLoading(false);
            }
        }
    }, []);

    const loadHoldingsLive = useCallback(async (normalizedCode: string): Promise<boolean> => {
        if (!normalizedCode || holdingsLiveInFlightRef.current) return false;
        holdingsLiveInFlightRef.current = true;
        try {
            const payload = await apiClient.fundHoldingsLive(normalizedCode);
            if (!mountedRef.current || activeFundCodeRef.current !== normalizedCode) return false;
            setHoldings((current) => {
                if (!current || current.fund_code !== normalizedCode) return current;
                return mergeFundHoldingsLivePayload(current, payload);
            });
            holdingsLiveFailureCountRef.current = 0;
            return true;
        } catch {
            // Live fiyat/KZ güncellemesi statik KAP tablosunu düşürmemeli.
            holdingsLiveFailureCountRef.current += 1;
            return false;
        } finally {
            holdingsLiveInFlightRef.current = false;
        }
    }, []);

    useEffect(() => {
        let alive = true;
        setError(null);
        setRefreshError(null);

        const cachedCatalog = getFundsCatalogMemoryCache();
        const hasUsableCachedCatalog = Boolean(cachedCatalog && !shouldRefreshSnapshot(cachedCatalog.funds));
        if (hasUsableCachedCatalog && cachedCatalog) {
            setFunds(cachedCatalog.funds);
            setCategories(cachedCatalog.categories);
            setLoading(false);
        } else {
            setLoading(true);
        }

        const loadInitialCatalog = async () => {
            try {
                const { funds: fundPayload, categories: categoryPayload } = await fetchFundsCatalog();
                if (!alive) return;
                setFunds(fundPayload);
                setCategories(categoryPayload);
            } catch (err) {
                if (!alive) return;
                setError(err instanceof Error ? err.message : 'Fon listesi alınamadı.');
            } finally {
                if (alive && !hasUsableCachedCatalog) setLoading(false);
            }
        };

        const cacheIsFresh = Boolean(
            cachedCatalog && Date.now() - cachedCatalog.fetchedAt < FUNDS_CATALOG_MEMORY_TTL_MS,
        );
        if (!cacheIsFresh || !hasUsableCachedCatalog) {
            void loadInitialCatalog();
        }
        return () => {
            alive = false;
        };
    }, []);

    useEffect(() => {
        activeFundCodeRef.current = fundCode?.trim().toUpperCase() || '';
    }, [fundCode]);

    const loadFundDetail = useCallback(async (rawFundCode: string) => {
        const normalizedCode = rawFundCode.trim().toUpperCase();
        if (!normalizedCode) return;
        const requestId = ++fundDetailRequestIdRef.current;
        setDetailLoading(true);
        setDetailError(null);
        setDetail((current) => (
            current?.fund_code.trim().toUpperCase() === normalizedCode ? current : null
        ));

        try {
            const payload = await apiClient.fundDetail(normalizedCode);
            if (
                !mountedRef.current
                || activeFundCodeRef.current !== normalizedCode
                || requestId !== fundDetailRequestIdRef.current
            ) return;
            setDetail(payload);
        } catch (err) {
            if (
                !mountedRef.current
                || activeFundCodeRef.current !== normalizedCode
                || requestId !== fundDetailRequestIdRef.current
            ) return;
            setDetailError(err instanceof Error ? err.message : 'Fon detayı alınamadı.');
        } finally {
            if (
                mountedRef.current
                && activeFundCodeRef.current === normalizedCode
                && requestId === fundDetailRequestIdRef.current
            ) {
                setDetailLoading(false);
            }
        }
    }, []);

    useEffect(() => {
        if (!fundCode) {
            fundDetailRequestIdRef.current += 1;
            setDetail(null);
            setDetailError(null);
            setDetailLoading(false);
            return;
        }
        void loadFundDetail(fundCode);
        return () => {
            fundDetailRequestIdRef.current += 1;
        };
    }, [fundCode, loadFundDetail]);

    useEffect(() => {
        if (!fundCode) return;
        setChartRange('1w');
        setPendingChartRange(null);
        setChartRangeError(null);
        setYieldSummary(null);
        setYieldError(null);
        setAllocations(null);
        setAllocationHistory(null);
        setAllocationHistoryError(null);
        setHistorySubtab('prices');
        setCustomStartDate(isoDateMonthsAgo(6));
        setCustomEndDate(new Date().toISOString().slice(0, 10));
        setHeatmapPerformance(null);
        setHoldings(null);
        setHoldingsError(null);
        setHoldingsLoading(true);
        setHoldingCategoryFilter('all');
        setHoldingChangeFilter('all');
        holdingsFetchAttemptedRef.current.clear();
    }, [fundCode]);

    useEffect(() => {
        if (!fundCode) {
            setPerformance(null);
            setHeatmapPerformance(null);
            setPerformanceError(null);
            setHistoryBackfillError(null);
            return;
        }
        let alive = true;
        const normalizedCode = fundCode.trim().toUpperCase();
        setPerformance(null);
        setPerformanceLoading(true);
        setPerformanceError(null);
        setHistoryBackfillError(null);
            apiClient
            .fundPerformance(normalizedCode, { startDate: isoDateMonthsAgo(FUND_INITIAL_PERFORMANCE_MONTHS) })
            .then((payload) => {
                if (!alive) return;
                setPerformance(payload);
                const completedHistoryJob = payload.source_metadata?.history_job;
                const requestedStart = payload.source_metadata?.requested_start_date
                    || completedHistoryJob?.requested_start
                    || null;
                const refreshStart = completedHistoryJob?.effective_start || null;
                const refreshEnd = completedHistoryJob?.effective_end || null;
                if (
                    completedHistoryJob?.status === 'succeeded'
                    && requestedStart
                    && refreshStart
                    && refreshEnd
                    && refreshStart < requestedStart
                ) {
                    void apiClient
                        .fundPerformance(normalizedCode, {
                            startDate: refreshStart,
                            endDate: refreshEnd,
                            refresh: true,
                        })
                        .then((refreshed) => {
                            if (alive) setPerformance((current) => mergeFundPerformancePayloads(current, refreshed));
                        })
                        .catch(() => {
                            // The narrow initial series remains usable if the wider refresh fails.
                        });
                }
            })
            .catch((err) => {
                if (!alive) return;
                setPerformance(null);
                setPerformanceError(err instanceof Error ? err.message : 'Grafik verisi alınamadı.');
            })
            .finally(() => {
                if (alive) setPerformanceLoading(false);
            });
        return () => {
            alive = false;
        };
    }, [fundCode]);

    useEffect(() => {
        if (!fundCode) {
            setHeatmapPerformance(null);
            return;
        }
        // The performance endpoint now coalesces the requested range into one
        // per-fund background job. A second full-history request here would
        // only duplicate the old backfill race, so the heatmap follows the
        // same response until the job publishes a newer one.
        setHeatmapPerformance(performance);
    }, [fundCode, performance]);

    const historyJob = performance?.source_metadata?.history_job || null;
    const historyJobId = historyJob?.job_id || null;
    const historyJobStatus = historyJob?.status || null;
    const historyJobRequestedStart = historyJob?.requested_start || null;
    const historyJobRequestedEnd = historyJob?.requested_end || null;
    const historyJobEffectiveStart = historyJob?.effective_start || null;
    const historyJobEffectiveEnd = historyJob?.effective_end || null;

    useEffect(() => {
        if (!fundCode || !historyJobId || !historyJobStatus || !['queued', 'running'].includes(historyJobStatus)) {
            return;
        }
        const normalizedCode = fundCode.trim().toUpperCase();
        const requestedStart = historyJobRequestedStart || performance?.source_metadata?.requested_start_date || undefined;
        const requestedEnd = historyJobRequestedEnd || performance?.source_metadata?.requested_end_date || undefined;
        // Backend history jobs are retained for 30 minutes. Keep polling for
        // that same contract instead of declaring a cold TEFAS backfill dead
        // after two minutes.
        const deadline = Date.now() + 30 * 60 * 1000;
        let alive = true;
        let timer: number | null = null;
        let inFlight = false;
        let continuePolling = true;
        let pollAttempt = 0;

        const schedule = () => {
            if (timer !== null) {
                window.clearTimeout(timer);
                timer = null;
            }
            if (alive && continuePolling) {
                const delay = pollAttempt < 5 ? 800 : pollAttempt < 12 ? 2000 : 5000;
                pollAttempt += 1;
                timer = window.setTimeout(poll, delay);
            }
        };
        const poll = async () => {
            if (!alive || inFlight) return;
            if (Date.now() >= deadline) {
                setHistoryBackfillError('Günlük geçmiş hazırlaması zaman aşımına uğradı; mevcut veriler gösteriliyor.');
                continuePolling = false;
                return;
            }
            if (document.visibilityState !== 'visible') {
                schedule();
                return;
            }
            inFlight = true;
            try {
                const status = await apiClient.fundPerformanceStatus(normalizedCode, {
                    startDate: requestedStart,
                    endDate: requestedEnd,
                });
                const statusJob = status.history_job;
                if (!alive) return;
                if (!statusJob) {
                    continuePolling = false;
                    return;
                }
                if (!['queued', 'running'].includes(statusJob.status)) {
                    continuePolling = false;
                    if (statusJob.status === 'succeeded') {
                        // The worker may widen the range beyond the original
                        // six-month request. Read that effective range and
                        // merge it so a narrower late response cannot erase
                        // an already complete series in the UI.
                        const refreshStart = statusJob.effective_start
                            || historyJobEffectiveStart
                            || requestedStart;
                        const refreshEnd = statusJob.effective_end
                            || historyJobEffectiveEnd
                            || requestedEnd;
                        const refreshed = await apiClient.fundPerformance(normalizedCode, {
                            startDate: refreshStart,
                            endDate: refreshEnd,
                            refresh: true,
                        });
                        if (!alive) return;
                        setPerformance((current) => mergeFundPerformancePayloads(current, refreshed));
                        setHistoryBackfillError(null);
                    } else if (statusJob.status === 'failed') {
                        setHistoryBackfillError('Günlük geçmiş alınamadı; mevcut veriler gösteriliyor.');
                    }
                    return;
                }
            } catch {
                // A transient status failure should not erase the chart or
                // turn a healthy partial series into an error state.
            } finally {
                inFlight = false;
                schedule();
            }
        };

        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible') void poll();
        };
        document.addEventListener('visibilitychange', onVisibilityChange);
        void poll();
        return () => {
            alive = false;
            if (timer !== null) window.clearTimeout(timer);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [
        fundCode,
        historyJobEffectiveEnd,
        historyJobEffectiveStart,
        historyJobId,
        historyJobRequestedEnd,
        historyJobRequestedStart,
        historyJobStatus,
        performance?.source_metadata?.requested_end_date,
        performance?.source_metadata?.requested_start_date,
    ]);

    useEffect(() => {
        if (!fundCode) {
            setYieldSummary(null);
            setYieldError(null);
            return;
        }
        if (activeTab !== 'overview' || yieldSummary?.fund_code === fundCode.trim().toUpperCase()) return;
        let alive = true;
        const controller = new AbortController();
        const normalizedCode = fundCode.trim().toUpperCase();
        setYieldLoading(true);
        setYieldError(null);
        apiClient
            .fundYieldSummary(normalizedCode, { signal: controller.signal })
            .then((payload) => {
                if (alive) setYieldSummary(payload);
            })
            .catch((err) => {
                if (!alive) return;
                setYieldSummary(null);
                setYieldError(err instanceof Error ? err.message : 'Getiri özeti alınamadı.');
            })
            .finally(() => {
                if (alive) setYieldLoading(false);
            });
        return () => {
            alive = false;
            controller.abort();
        };
    }, [fundCode, activeTab, yieldSummary?.fund_code]);

    useBackgroundRefresh(Boolean(funds?.refresh_pending || funds?.refresh_job?.status === 'queued' || funds?.refresh_job?.status === 'running'), async (signal) => {
        const payload = await apiClient.funds({ signal });
        if (signal.aborted) return;
        setFunds(payload);
        updateFundsCatalogSnapshot(payload);
    });
    useBackgroundRefresh(Boolean(fundCode && activeTab === 'overview' && yieldSummary?.refresh_pending), async (signal) => {
        const code = fundCode!.trim().toUpperCase();
        const payload = await apiClient.fundYieldSummary(code, { signal });
        if (!signal.aborted && activeFundCodeRef.current === code) setYieldSummary(payload);
    });
    useBackgroundRefresh(Boolean(fundCode && holdings?.refresh_pending), async (signal) => {
        const code = fundCode!.trim().toUpperCase();
        const payload = await apiClient.fundHoldings(code, { force: true });
        if (!signal.aborted && activeFundCodeRef.current === code) setHoldings(payload);
    });
    useBackgroundRefresh(
        Boolean(
            fundCode
            && (activeTab === 'overview' || activeTab === 'allocation')
            && allocations?.refresh_pending,
        ),
        async (signal) => {
            const code = fundCode!.trim().toUpperCase();
            const payload = await apiClient.fundAllocations(code);
            if (!signal.aborted && activeFundCodeRef.current === code) setAllocations(payload);
        },
    );

    useEffect(() => {
        if (!fundCode) {
            setAllocations(null);
            setAllocationHistory(null);
            setAllocationHistoryError(null);
            setAllocationHistoryLoading(false);
            return;
        }
        if (activeTab !== 'overview' && activeTab !== 'allocation') return;
        if (allocations?.fund_code === fundCode.trim().toUpperCase()) return;
        let alive = true;
        const normalizedCode = fundCode.trim().toUpperCase();
        setAllocationLoading(true);
        apiClient
            .fundAllocations(normalizedCode)
            .then((payload) => {
                if (!alive) return;
                setAllocations(payload);
            })
            .catch(() => {
                if (alive) setAllocations(null);
            })
            .finally(() => {
                if (alive) setAllocationLoading(false);
            });
        return () => {
            alive = false;
        };
    }, [fundCode, activeTab, allocations?.fund_code]);

    useEffect(() => {
        if (!fundCode) {
            setHoldings(null);
            setHoldingsError(null);
            setHoldingsLoading(false);
            return;
        }
        if (activeTab !== 'overview' && activeTab !== 'allocation') return;
        const normalizedCode = fundCode.trim().toUpperCase();
        if (holdings?.fund_code === normalizedCode) {
            return;
        }
        if (holdingsFetchAttemptedRef.current.has(normalizedCode)) {
            if (holdingsRefreshInFlightRef.current) {
                setHoldingsLoading(true);
            }
            return;
        }
        void loadHoldings(normalizedCode);
    }, [fundCode, activeTab, holdings?.fund_code, loadHoldings]);

    useEffect(() => {
        if (!fundCode || (activeTab !== 'overview' && activeTab !== 'allocation')) return;
        const normalizedCode = fundCode.trim().toUpperCase();
        let alive = true;
        let timer: number | null = null;

        const revalidate = () => {
            if (!alive || document.visibilityState !== 'visible') return;
            void loadHoldings(normalizedCode, { silent: true });
        };
        const onFocus = () => revalidate();
        const onVisibilityChange = () => revalidate();

        timer = window.setInterval(revalidate, FUND_HOLDINGS_REPORT_REFRESH_MS);
        window.addEventListener('focus', onFocus);
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            alive = false;
            if (timer !== null) window.clearInterval(timer);
            window.removeEventListener('focus', onFocus);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [fundCode, activeTab, loadHoldings]);

    useEffect(() => {
        if (!fundCode) return;
        if (activeTab !== 'overview' && activeTab !== 'allocation') return;
        const normalizedCode = fundCode.trim().toUpperCase();
        let timer: number | null = null;
        let alive = true;
        const schedule = (delay: number) => {
            if (timer !== null) {
                window.clearTimeout(timer);
                timer = null;
            }
            if (alive) timer = window.setTimeout(refreshVisibleHoldings, delay);
        };
        const refreshVisibleHoldings = async () => {
            if (document.visibilityState !== 'visible' || !isBistMarketOpen()) {
                schedule(FUND_HOLDINGS_LIVE_MAX_BACKOFF_MS);
                return;
            }
            if (!holdingsRef.current || holdingsRef.current.fund_code !== normalizedCode) return;
            const succeeded = await loadHoldingsLive(normalizedCode);
            const failures = holdingsLiveFailureCountRef.current;
            const delay = succeeded
                ? FUND_HOLDINGS_LIVE_REFRESH_MS
                : Math.min(FUND_HOLDINGS_LIVE_MAX_BACKOFF_MS, FUND_HOLDINGS_LIVE_REFRESH_MS * (2 ** Math.max(0, failures)));
            schedule(delay);
        };
        const onFocus = () => { void refreshVisibleHoldings(); };
        const onVisibilityChange = () => { void refreshVisibleHoldings(); };
        void refreshVisibleHoldings();
        window.addEventListener('focus', onFocus);
        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => {
            alive = false;
            if (timer !== null) window.clearTimeout(timer);
            window.removeEventListener('focus', onFocus);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        };
    }, [fundCode, activeTab, loadHoldingsLive]);

    const fundSearchIndex = useMemo(() => (funds?.rows || []).map((row) => ({
        row,
        haystack: normalizeCompareSearch(
            `${row.fund_code} ${row.name} ${row.fund_type || ''} ${row.founder_company || ''} ${row.manager_company || ''}`,
        ),
    })), [funds]);

    const parsedFundScreenerBounds = useMemo(() => ({
        minDaily: parseFundScreenerNumber(fundScreenerFilters.min_daily),
        maxDaily: parseFundScreenerNumber(fundScreenerFilters.max_daily),
        min1w: parseFundScreenerNumber(fundScreenerFilters.min_1w),
        max1w: parseFundScreenerNumber(fundScreenerFilters.max_1w),
        min1m: parseFundScreenerNumber(fundScreenerFilters.min_1m),
        max1m: parseFundScreenerNumber(fundScreenerFilters.max_1m),
        min3m: parseFundScreenerNumber(fundScreenerFilters.min_3m),
        max3m: parseFundScreenerNumber(fundScreenerFilters.max_3m),
        min6m: parseFundScreenerNumber(fundScreenerFilters.min_6m),
        max6m: parseFundScreenerNumber(fundScreenerFilters.max_6m),
        minYtd: parseFundScreenerNumber(fundScreenerFilters.min_ytd),
        maxYtd: parseFundScreenerNumber(fundScreenerFilters.max_ytd),
        min1y: parseFundScreenerNumber(fundScreenerFilters.min_1y),
        max1y: parseFundScreenerNumber(fundScreenerFilters.max_1y),
        minRisk: parseFundScreenerNumber(fundScreenerFilters.min_risk),
        maxRisk: parseFundScreenerNumber(fundScreenerFilters.max_risk),
        minAumM: parseFundScreenerNumber(fundScreenerFilters.min_aum_m),
        maxAumM: parseFundScreenerNumber(fundScreenerFilters.max_aum_m),
        minInvestors: parseFundScreenerNumber(fundScreenerFilters.min_investors),
        maxInvestors: parseFundScreenerNumber(fundScreenerFilters.max_investors),
    }), [fundScreenerFilters]);

    const filteredFunds = useMemo(() => {
        const needle = normalizeCompareSearch(deferredSearchTerm);
        const matchesRange = (val: number | null | undefined, min?: number, max?: number) => {
            if (min !== undefined && (val == null || !Number.isFinite(val) || val < min)) return false;
            if (max !== undefined && (val == null || !Number.isFinite(val) || val > max)) return false;
            return true;
        };
        const b = parsedFundScreenerBounds;
        const filtered = fundSearchIndex
            .filter(({ row, haystack }) => {
                if (needle && !haystack.includes(needle)) return false;
                if (fundTypeFilter && row.fund_type !== fundTypeFilter) return false;
                if (riskFilter && String(row.risk_value ?? '') !== riskFilter) return false;
                if (fundScreenerFilters.founder) {
                    const founder = row.founder_company || row.manager_company || '';
                    if (founder !== fundScreenerFilters.founder) return false;
                }
                if (!matchesRange(row.daily_return, b.minDaily, b.maxDaily)) return false;
                if (!matchesRange(row.period_returns?.['1w'], b.min1w, b.max1w)) return false;
                if (!matchesRange(row.period_returns?.['1m'], b.min1m, b.max1m)) return false;
                if (!matchesRange(row.period_returns?.['3m'], b.min3m, b.max3m)) return false;
                if (!matchesRange(row.period_returns?.['6m'], b.min6m, b.max6m)) return false;
                if (!matchesRange(row.period_returns?.ytd, b.minYtd, b.maxYtd)) return false;
                if (!matchesRange(row.period_returns?.['1y'], b.min1y, b.max1y)) return false;
                if (!matchesRange(row.risk_value, b.minRisk, b.maxRisk)) return false;
                const aumMillions = row.aum != null && Number.isFinite(row.aum) ? row.aum / 1_000_000 : null;
                if (!matchesRange(aumMillions, b.minAumM, b.maxAumM)) return false;
                if (!matchesRange(row.investor_count, b.minInvestors, b.maxInvestors)) return false;
                return true;
            })
            .map(({ row }) => row);
        filtered.sort((a, bRow) => compareFundRows(a, bRow, sortKey, sortOrder));
        return filtered;
    }, [
        deferredSearchTerm,
        fundScreenerFilters.founder,
        fundSearchIndex,
        fundTypeFilter,
        parsedFundScreenerBounds,
        riskFilter,
        sortKey,
        sortOrder,
    ]);

    const tableResetDeps = useMemo(
        () => [searchTerm, deferredSearchTerm, fundTypeFilter, riskFilter, fundScreenerFilters, fundTableMode, sortKey, sortOrder],
        [searchTerm, deferredSearchTerm, fundTypeFilter, riskFilter, fundScreenerFilters, fundTableMode, sortKey, sortOrder],
    );

    const normalizedRouteFundCode = fundCode?.trim().toUpperCase() || '';
    const routeFundSummary = normalizedRouteFundCode
        ? funds?.rows.find((row) => row.fund_code.trim().toUpperCase() === normalizedRouteFundCode) || null
        : null;
    const selectedFund = (
        detail && (!normalizedRouteFundCode || detail.fund_code.trim().toUpperCase() === normalizedRouteFundCode)
            ? detail
            : routeFundSummary || filteredFunds.find((row) => row.fund_code.trim().toUpperCase() === normalizedRouteFundCode) || null
    );
    const performancePoints = useMemo(() => sortFundPoints(performance?.points), [performance]);
    const fintablesHistoryPending = performanceLoading && !performance
        || isFintablesHistoryPending(performance, historyJob);
    const visiblePerformancePoints = fintablesHistoryPending ? [] : performancePoints;
    const heatmapSourcePerformance = hasFullHistoryPerformance(performance)
        ? performance
        : (hasFullHistoryPerformance(heatmapPerformance) ? heatmapPerformance : performance);
    const heatmapPerformancePoints = useMemo(
        () => sortFundPoints(heatmapSourcePerformance?.points),
        [heatmapSourcePerformance],
    );
    const visibleHeatmapPerformancePoints = fintablesHistoryPending ? [] : heatmapPerformancePoints;
    const latestPerformanceDate = visiblePerformancePoints[visiblePerformancePoints.length - 1]?.date || null;
    const latestAvailableDate = latestFundDate(
        selectedFund?.as_of,
        performance?.as_of,
        performance?.source_metadata?.available_end_date,
        performance?.source_metadata?.date_max,
        latestPerformanceDate,
    ) || new Date().toISOString().slice(0, 10);
    const chartEndDate = chartRange === 'custom'
        ? (customEndDate || latestPerformanceDate || latestAvailableDate)
        : (latestPerformanceDate || latestAvailableDate);
    const chartPoints = useMemo(
        () => filterPointsForRange(visiblePerformancePoints, chartRange, chartEndDate, customStartDate, customEndDate),
        [visiblePerformancePoints, chartRange, chartEndDate, customStartDate, customEndDate],
    );
    const monthlyReturns = useMemo(() => monthlyReturnsFromPoints(visibleHeatmapPerformancePoints), [visibleHeatmapPerformancePoints]);
    const overviewMetricSeries = useMemo(() => overviewMetricsFromPoints(visiblePerformancePoints), [visiblePerformancePoints]);
    const allocationRows = useMemo(
        () => [...(allocations?.allocations || [])]
            .filter((item) => Number.isFinite(Number(item.weight)))
            .sort((a, b) => Math.abs(Number(b.weight || 0)) - Math.abs(Number(a.weight || 0))),
        [allocations],
    );
    const selectedFundPrice = Number(selectedFund?.price);
    // Price, daily return and as-of in the hero must describe the same
    // canonical snapshot. Performance points remain chart-only data.
    const detailLatestPrice = canonicalFundPrice(selectedFundPrice);
    const detailPeriodReturns = useMemo(
        () => {
            if (fintablesHistoryPending) {
                return DETAIL_RETURN_PERIODS.reduce<FundSummary['period_returns']>((returns, period) => {
                    returns[period.key] = null;
                    return returns;
                }, {});
            }
            const computed = periodReturnsFromPerformancePoints(
                visiblePerformancePoints,
                performance?.source_metadata?.coverage_state === 'range_incomplete',
            );
            for (const period of [...DETAIL_RETURN_PERIODS, ...EXTENDED_DETAIL_RETURN_PERIODS]) {
                if (computed[period.key] != null) continue;
                const yieldPeriod = yieldSummary?.periods?.[period.key];
                const yieldReturnPct = yieldPeriod?.return_pct;
                const yieldComputed =
                    typeof yieldReturnPct === 'number' && Number.isFinite(yieldReturnPct)
                        ? yieldReturnPct
                        : returnBetween(detailLatestPrice, yieldPeriod?.prev_close ?? null);
                const detailVal = selectedFund?.period_returns?.[period.key];
                const resolved =
                    yieldComputed ?? (typeof detailVal === 'number' && Number.isFinite(detailVal) ? detailVal : null);
                if (resolved != null) {
                    computed[period.key] = resolved;
                }
            }
            return computed;
        },
        [
            detailLatestPrice,
            fintablesHistoryPending,
            performance?.source_metadata?.coverage_state,
            selectedFund?.period_returns,
            visiblePerformancePoints,
            yieldSummary?.periods,
        ],
    );
    const earliestKnownFundDate = useMemo(() => {
        const candidates = [
            visiblePerformancePoints[0]?.date,
            detail?.history_start_date,
            performance?.source_metadata?.available_start_date,
            performance?.source_metadata?.date_min,
            yieldSummary?.periods?.oldest?.prev_close_date,
        ].filter((val): val is string => typeof val === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(val));
        return candidates.sort()[0] || null;
    }, [
        detail?.history_start_date,
        performance?.source_metadata?.available_start_date,
        performance?.source_metadata?.date_min,
        visiblePerformancePoints,
        yieldSummary?.periods?.oldest?.prev_close_date,
    ]);
    const visibleDetailPeriods = useMemo<FundDetailReturnPeriod[]>(() => {
        const base: FundDetailReturnPeriod[] = [...DETAIL_RETURN_PERIODS];
        for (const period of EXTENDED_DETAIL_RETURN_PERIODS) {
            const hasReturn = detailPeriodReturns?.[period.key] != null;
            const targetCutoff = shiftIsoDateDays(rangeStartDate(period.key, latestAvailableDate, ''), 7);
            const hasDateCoverage = Boolean(earliestKnownFundDate && earliestKnownFundDate <= targetCutoff);
            if (hasReturn || hasDateCoverage) {
                base.push({ key: period.key, label: period.label });
            }
        }
        return base;
    }, [detailPeriodReturns, earliestKnownFundDate, latestAvailableDate]);
    const activeRangeCovered = useMemo(() => {
        if (chartRange === 'all') {
            return hasFullHistoryPerformance(performance) && performancePoints.length >= 2;
        }
        if (chartRange !== '3y' && chartRange !== '5y' && chartRange !== 'custom') return true;
        if (chartPoints.length < 2) return false;
        const startIso = rangeStartDate(chartRange, chartEndDate, customStartDate);
        const targetCutoff = shiftIsoDateDays(startIso, 14);
        return chartPoints[0].date <= targetCutoff;
    }, [chartEndDate, chartPoints, chartRange, customStartDate, performance, performancePoints.length]);
    const activeRangeJobPending = Boolean(
        (chartRange === '3y' || chartRange === '5y' || chartRange === 'custom' || chartRange === 'all')
        && !activeRangeCovered
        && historyJobStatus
        && ['queued', 'running'].includes(historyJobStatus),
    );
    const fundsDocumentTitle = useMemo(() => {
        if (fundCode || selectedFund) {
            const normalizedCode = selectedFund?.fund_code || fundCode?.trim().toUpperCase();
            const quoteTitle = [
                formatTitleCurrency(detailLatestPrice, selectedFund?.currency, {
                    minimumFractionDigits: 4,
                    maximumFractionDigits: 6,
                }),
                formatTitlePct(selectedFund?.daily_return),
            ].filter(Boolean).join(' ');
            return buildDocumentTitle(normalizedCode, quoteTitle, selectedFund?.name || 'Fon Detayı');
        }
        const fundCount = filteredFunds.length || funds?.rows.length;
        return buildDocumentTitle('Fonlar', fundCount ? `${fundCount} fon` : null);
    }, [
        detailLatestPrice,
        filteredFunds.length,
        fundCode,
        funds?.rows.length,
        selectedFund,
    ]);
    useDocumentTitle(fundsDocumentTitle, view !== 'compare');
    const comparisonState = useFundComparisonState({
        selectedFund,
        baseReturns: detailPeriodReturns,
        baseReturnsReady: !fintablesHistoryPending,
        funds: funds?.rows || [],
        openSearchSignal: 0,
    });
    const comparisonChartStartDate = chartPoints[0]?.date || null;
    const comparisonChartEndDate = chartPoints[chartPoints.length - 1]?.date || null;
    const comparisonReturnStartDate = comparisonChartEndDate
        ? rangeStartDate(comparisonState.period, comparisonChartEndDate, '')
        : null;
    const comparisonHistoryStartDate = [comparisonChartStartDate, comparisonReturnStartDate]
        .filter((value): value is string => Boolean(value))
        .sort()
        .at(0) || null;
    const comparisonHistoryFetchStartDate = comparisonHistoryStartDate
        ? shiftIsoDateDays(comparisonHistoryStartDate, -10)
        : null;
    const baseFundCurrency = (selectedFund?.currency || 'TRY').trim().toUpperCase() || 'TRY';
    const comparisonTargetCurrency: 'TRY' | undefined =
        comparisonState.fxAdjusted && baseFundCurrency === 'TRY' ? 'TRY' : undefined;
    const comparisonAssets = comparisonState.selectedAssets;
    const comparisonAssetRequests = useMemo(
        () => comparisonAssets.map((asset) => ({
            id: asset.id,
            kind: asset.kind,
            symbol: asset.symbol,
            label: asset.displaySymbol || asset.label,
        })),
        [comparisonAssets],
    );
    const comparisonAssetRequestKey = comparisonAssetRequests
        .map((asset) => `${asset.id}:${asset.kind}:${asset.symbol}:${asset.label || ''}`)
        .join('|');
    useEffect(() => {
        if (!comparisonAssetRequests.length || !comparisonHistoryFetchStartDate || !comparisonChartEndDate) {
            setComparisonHistory(null);
            setComparisonHistoryError(null);
            setComparisonHistoryLoading(false);
            return;
        }
        if (pendingChartRange || activeRangeJobPending) {
            setComparisonHistory(null);
            setComparisonHistoryError(null);
            setComparisonHistoryLoading(true);
            return;
        }
        const controller = new AbortController();
        let retryTimer: number | null = null;
        const maxRetries = 12;

        const runFetch = (attempt: number) => {
            if (controller.signal.aborted) return;
            if (attempt === 0) {
                setComparisonHistory(null);
                setComparisonHistoryError(null);
            }
            setComparisonHistoryLoading(true);
            apiClient
                .marketComparisonHistory(
                    {
                        start_date: comparisonHistoryFetchStartDate,
                        end_date: comparisonChartEndDate,
                        target_currency: comparisonTargetCurrency,
                        assets: comparisonAssetRequests,
                    },
                    { signal: controller.signal },
                )
                .then((payload) => {
                    if (controller.signal.aborted) return;
                    setComparisonHistory(payload);
                    const hasPendingAsset = (payload.assets || []).some(
                        (item) => Boolean(item.refresh_pending || item.error === 'refresh_pending'),
                    );
                    if (hasPendingAsset && attempt < maxRetries) {
                        retryTimer = window.setTimeout(() => {
                            runFetch(attempt + 1);
                        }, 1500);
                    } else {
                        setComparisonHistoryLoading(false);
                    }
                })
                .catch((err) => {
                    if ((err as Error)?.name === 'AbortError' || controller.signal.aborted) return;
                    setComparisonHistory(null);
                    setComparisonHistoryError(err instanceof Error ? err.message : 'Karşılaştırma geçmişi alınamadı.');
                    setComparisonHistoryLoading(false);
                });
        };

        runFetch(0);
        return () => {
            controller.abort();
            if (retryTimer != null) {
                window.clearTimeout(retryTimer);
            }
        };
    }, [
        activeRangeJobPending,
        comparisonAssetRequestKey,
        comparisonChartEndDate,
        comparisonHistoryFetchStartDate,
        comparisonTargetCurrency,
        pendingChartRange,
    ]);
    const detailSourceWarnings = useMemo(
        () => fundCode
            ? buildFundHistoryDiagnosticLines({
                fundCode: fundCode.trim().toUpperCase(),
                selectedFund,
                performance,
                points: performancePoints,
                periodReturns: detailPeriodReturns,
                historyJob,
                performanceLoading,
                performanceError,
                historyBackfillError,
                yieldSummary,
                yieldLoading,
                yieldError,
            })
            : [],
        [
            detailPeriodReturns,
            fundCode,
            historyBackfillError,
            historyJob,
            performance,
            performanceError,
            performanceLoading,
            performancePoints,
            selectedFund,
            yieldError,
            yieldLoading,
            yieldSummary,
        ],
    );
    const visibleFundTypes = categories?.fund_types.length ? categories.fund_types : Array.from(new Set((funds?.rows || []).map((row) => row.fund_type).filter(Boolean))) as string[];
    const visibleRiskValues = categories?.risk_values.length ? categories.risk_values : Array.from(new Set((funds?.rows || []).map((row) => row.risk_value).filter((value): value is number => typeof value === 'number'))).sort((a, b) => a - b);
    const visibleFounders = useMemo(
        () => Array.from(new Set((funds?.rows || []).map((row) => row.founder_company || row.manager_company).filter(Boolean) as string[])).sort((a, b) => a.localeCompare(b, 'tr')),
        [funds],
    );
    const activeFundFilterCount = useMemo(() => {
        let count = 0;
        if (fundTypeFilter) count += 1;
        if (riskFilter) count += 1;
        if (fundScreenerFilters.founder) count += 1;
        for (const spec of FUND_SCREENER_RANGE_SPECS) {
            if (fundScreenerFilters[spec.minKey].trim() || fundScreenerFilters[spec.maxKey].trim()) {
                count += 1;
            }
        }
        return count;
    }, [fundScreenerFilters, fundTypeFilter, riskFilter]);
    const isFundScreenerActive = activeFundFilterCount > 0;
    const isFundsFiltered = isFundScreenerActive || Boolean(deferredSearchTerm.trim());
    const activeFundChips = useMemo(() => {
        const chips: Array<{ id: string; label: string; onClear: () => void }> = [];
        if (fundTypeFilter) {
            chips.push({ id: 'fund_type', label: `Tür: ${fundTypeFilter}`, onClear: () => setFundTypeFilter('') });
        }
        if (riskFilter) {
            chips.push({ id: 'risk', label: `Risk: ${riskFilter}`, onClear: () => setRiskFilter('') });
        }
        if (fundScreenerFilters.founder) {
            chips.push({
                id: 'founder',
                label: `Kurucu: ${fundScreenerFilters.founder}`,
                onClear: () => {
                    setActiveFundScreenerPreset(null);
                    setFundScreenerFilters((prev) => ({ ...prev, founder: '' }));
                },
            });
        }
        for (const spec of FUND_SCREENER_RANGE_SPECS) {
            const minVal = fundScreenerFilters[spec.minKey].trim();
            const maxVal = fundScreenerFilters[spec.maxKey].trim();
            if (!minVal && !maxVal) continue;
            const unit = spec.unit ? ` ${spec.unit}` : '';
            const rangeText = minVal && maxVal
                ? `${minVal}–${maxVal}${unit}`
                : minVal
                    ? `≥ ${minVal}${unit}`
                    : `≤ ${maxVal}${unit}`;
            chips.push({
                id: spec.minKey,
                label: `${spec.label}: ${rangeText}`,
                onClear: () => {
                    setActiveFundScreenerPreset(null);
                    setFundScreenerFilters((prev) => ({ ...prev, [spec.minKey]: '', [spec.maxKey]: '' }));
                },
            });
        }
        return chips;
    }, [fundScreenerFilters, fundTypeFilter, riskFilter]);

    useEffect(() => {
        if (!fundScreenerOpen) return;
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') setFundScreenerOpen(false);
        };
        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [fundScreenerOpen]);

    const handleFundScreenerFilterChange = useCallback(<K extends keyof FundScreenerFilterState>(
        key: K,
        value: FundScreenerFilterState[K],
    ) => {
        setActiveFundScreenerPreset(null);
        setFundScreenerFilters((prev) => ({ ...prev, [key]: value }));
    }, []);

    const applyFundScreenerPreset = useCallback((presetId: FundScreenerPresetId) => {
        if (activeFundScreenerPreset === presetId) {
            setActiveFundScreenerPreset(null);
            setFundScreenerFilters(EMPTY_FUND_SCREENER_FILTERS);
            return;
        }
        const preset = FUND_SCREENER_PRESETS.find((item) => item.id === presetId);
        if (!preset) return;
        setActiveFundScreenerPreset(presetId);
        setFundScreenerFilters({ ...EMPTY_FUND_SCREENER_FILTERS, ...preset.filters });
        if (preset.fundType !== undefined) setFundTypeFilter(preset.fundType);
        if (preset.risk !== undefined) setRiskFilter(preset.risk);
    }, [activeFundScreenerPreset]);

    const clearFundScreenerFilters = useCallback(() => {
        setActiveFundScreenerPreset(null);
        setFundScreenerFilters(EMPTY_FUND_SCREENER_FILTERS);
        setFundTypeFilter('');
        setRiskFilter('');
    }, []);

    const handleFundTableModeChange = useCallback((mode: FundTableMode) => {
        setFundTableMode(mode);
        if (mode === 'summary' && sortKey.startsWith('return_')) {
            setSortKey('daily_return');
            setSortOrder('desc');
        } else if (mode === 'performance' && ['fund_type', 'founder_company', 'risk_value', 'investor_count', 'as_of'].includes(sortKey)) {
            setSortKey('return_1y');
            setSortOrder('desc');
        }
    }, [sortKey]);

    const setTableSort = useCallback((key: FundSortKey) => {
        if (sortKey === key) {
            setSortOrder((currentOrder) => (currentOrder === 'asc' ? 'desc' : 'asc'));
            return;
        }
        const defaultDescKeys = new Set<FundSortKey>([
            'price',
            'daily_return',
            'return_1w',
            'return_1m',
            'return_3m',
            'return_6m',
            'return_ytd',
            'return_1y',
            'risk_value',
            'aum',
            'investor_count',
            'as_of',
        ]);
        setSortKey(key);
        setSortOrder(defaultDescKeys.has(key) ? 'desc' : 'asc');
    }, [sortKey]);
    const selectedFundCode = selectedFund?.fund_code || fundCode?.trim().toUpperCase() || '';
    const isStarred = selectedFundCode ? watchlist.hasItem('fund', selectedFundCode) : false;
    const toggleStarredFund = useCallback(() => {
        if (!selectedFundCode) return;
        watchlist.toggleItem({
            kind: 'fund',
            symbol: selectedFundCode,
            label: selectedFund?.name || selectedFundCode,
        });
    }, [selectedFund?.name, selectedFundCode, watchlist]);

    const loadAllocationHistory = useCallback((force = false) => {
        if (!fundCode) return;
        const normalizedCode = fundCode.trim().toUpperCase();
        setAllocationHistoryError(null);
        if (!force && allocationHistory?.fund_code === normalizedCode && allocationHistory.lookback_days === 30) {
            return;
        }
        if (allocationHistoryLoading) {
            return;
        }
        setAllocationHistoryLoading(true);
        apiClient
            .fundAllocationsHistory(normalizedCode, 30)
            .then((payload) => {
                if (!mountedRef.current || activeFundCodeRef.current !== normalizedCode) return;
                setAllocationHistory(payload);
            })
            .catch((err) => {
                if (!mountedRef.current || activeFundCodeRef.current !== normalizedCode) return;
                setAllocationHistory(null);
                setAllocationHistoryError(err instanceof Error ? err.message : 'Dağılım geçmişi alınamadı.');
            })
            .finally(() => {
                if (!mountedRef.current || activeFundCodeRef.current !== normalizedCode) return;
                setAllocationHistoryLoading(false);
            });
    }, [allocationHistory, allocationHistoryLoading, fundCode]);

    const openAllocationHistory = useCallback(() => {
        setHistorySubtab('allocation');
        onTabChange('history');
        loadAllocationHistory();
    }, [loadAllocationHistory, onTabChange]);

    useEffect(() => {
        if (
            activeTab === 'history'
            && historySubtab === 'allocation'
            && !allocationHistory
            && !allocationHistoryError
            && !allocationHistoryLoading
        ) {
            loadAllocationHistory();
        }
    }, [
        activeTab,
        allocationHistory,
        allocationHistoryError,
        allocationHistoryLoading,
        historySubtab,
        loadAllocationHistory,
    ]);

    useEffect(() => {
        if (
            activeTab !== 'history'
            || historySubtab !== 'allocation'
            || !allocationHistory?.refresh_pending
            || allocationHistoryLoading
        ) {
            return;
        }
        const timer = window.setTimeout(() => {
            void loadAllocationHistory(true);
        }, 5000);
        return () => window.clearTimeout(timer);
    }, [activeTab, allocationHistory?.refresh_pending, allocationHistoryLoading, historySubtab, loadAllocationHistory]);

    const refreshChartRange = useCallback((range: FundChartRange, nextCustomStart = customStartDate, nextCustomEnd = customEndDate) => {
        if (!fundCode) return;
        const normalizedCode = fundCode.trim().toUpperCase();
        const requestId = chartRequestIdRef.current + 1;
        chartRequestIdRef.current = requestId;
        if (range === 'all') {
            setChartRange(range);
            setChartRangeError(null);
            if (hasFullHistoryPerformance(performance) && performancePoints.length >= 2) return;
            setPendingChartRange(range);
            setPerformanceError(null);
            setPerformanceLoading(true);
            apiClient
                .fundPerformance(normalizedCode)
                .then((payload) => {
                    if (chartRequestIdRef.current !== requestId) return;
                    setPerformance((current) => mergeFundPerformancePayloads(current, payload));
                })
                .catch((err) => {
                    if (chartRequestIdRef.current !== requestId) return;
                    const message = err instanceof Error ? err.message : 'Grafik verisi alınamadı.';
                    setChartRangeError(message);
                    setPerformanceError(message);
                })
                .finally(() => {
                    if (chartRequestIdRef.current !== requestId) return;
                    setPendingChartRange(null);
                    setPerformanceLoading(false);
                });
            return;
        }
        const endIso = range === 'custom'
            ? (nextCustomEnd || chartEndDate)
            : chartEndDate;
        const startIso = rangeStartDate(range, endIso, nextCustomStart);
        if (hasUsableRangeCoverage(performancePoints, startIso, endIso)) {
            setChartRange(range);
            setChartRangeError(null);
            return;
        }
        setChartRange(range);
        setPendingChartRange(range);
        setChartRangeError(null);
        setPerformanceError(null);
        setPerformanceLoading(true);
        apiClient
            .fundPerformance(normalizedCode, { startDate: startIso, endDate: endIso })
            .then((payload) => {
                if (chartRequestIdRef.current !== requestId) return;
                setPerformance((current) => mergeFundPerformancePayloads(current, payload));
            })
            .catch((err) => {
                if (chartRequestIdRef.current !== requestId) return;
                const message = err instanceof Error ? err.message : 'Grafik verisi alınamadı.';
                setChartRangeError(message);
                setPerformanceError(message);
            })
            .finally(() => {
                if (chartRequestIdRef.current !== requestId) return;
                setPendingChartRange(null);
                setPerformanceLoading(false);
            });
    }, [chartEndDate, customEndDate, customStartDate, fundCode, performance, performancePoints]);

    const handleChartRangeSelect = useCallback((range: FundChartRange) => {
        refreshChartRange(range);
    }, [refreshChartRange]);

    const handleCustomStartDateChange = useCallback((value: string) => {
        const nextEnd = value && customEndDate && value > customEndDate ? value : customEndDate;
        setCustomStartDate(value);
        if (nextEnd !== customEndDate) {
            setCustomEndDate(nextEnd);
        }
        refreshChartRange('custom', value, nextEnd);
    }, [customEndDate, refreshChartRange]);

    const handleCustomEndDateChange = useCallback((value: string) => {
        const nextStart = value && customStartDate && value < customStartDate ? value : customStartDate;
        setCustomEndDate(value);
        if (nextStart !== customStartDate) {
            setCustomStartDate(nextStart);
        }
        refreshChartRange('custom', nextStart, value);
    }, [customStartDate, refreshChartRange]);

    return (
        <div className={`mn-layout${navCollapsed ? ' mn-nav-collapsed' : ''}`}>
            <MarketsNavigation
                collapsed={navCollapsed}
                activeSection="funds"
                activeFundSection={view === 'compare' ? 'compare' : 'investment'}
                onCollapsedChange={setNavCollapsed}
                onSectionChange={onNavigateSection}
                onStockSectionChange={onNavigateStockSection}
                onFundSectionChange={onNavigateFundSection}
                onSelectTicker={onOpenTicker}
                onSelectFund={(nextFundCode) => onOpenFund(nextFundCode, 'overview')}
            />
            <div className="funds-workspace">
                <div className="funds-page" ref={fundsPageRef}>
                    {view === 'compare' ? (
                        <Suspense fallback={<FinLoader message="Fon karşılaştırma yükleniyor" />}>
                            <FundComparisonSection
                                funds={funds}
                                loading={loading}
                                error={error}
                            />
                        </Suspense>
                    ) : !fundCode ? (
                        <section className="funds-panel funds-table-panel">
                            <div className="funds-panel-header">
                                <div className="funds-panel-header-top">
                                    <div className="funds-panel-title-group">
                                        <h2>Fonlar</h2>
                                        <span className="funds-count-badge">
                                            {loading && !funds
                                                ? 'Yükleniyor…'
                                                : isFundsFiltered
                                                    ? `${filteredFunds.length} / ${funds?.total_count ?? funds?.count ?? 0} fon`
                                                    : `${filteredFunds.length} fon`}
                                        </span>
                                    </div>
                                    <div className="funds-header-actions">
                                        <FundsSearchInput value={searchTerm} onChange={setSearchTerm} />
                                        <button
                                            type="button"
                                            className={`funds-screener-trigger-btn${isFundScreenerActive ? ' is-active' : ''}`}
                                            onClick={() => setFundScreenerOpen(true)}
                                            aria-label="Fon Tarayıcısı ve Filtreler"
                                        >
                                            <SlidersHorizontal size={13} aria-hidden="true" />
                                            <span>Fon Tarayıcısı</span>
                                            {activeFundFilterCount > 0 && (
                                                <span className="funds-screener-count-badge">{activeFundFilterCount}</span>
                                            )}
                                        </button>
                                    </div>
                                </div>
                                <div className="funds-panel-toolbar">
                                    <div className="funds-toolbar-left">
                                        <div className="funds-control-group">
                                            <span className="funds-control-label">Görünüm</span>
                                            <div
                                                className="funds-segment"
                                                aria-label="Fon tablo görünümü"
                                                style={{
                                                    '--item-count': 2,
                                                    '--active-index': fundTableMode === 'summary' ? 0 : 1,
                                                } as CSSProperties}
                                            >
                                                {([['summary', 'Özet'], ['performance', 'Getiri']] as const).map(([mode, label]) => (
                                                    <button
                                                        key={mode}
                                                        type="button"
                                                        className={fundTableMode === mode ? 'active' : ''}
                                                        aria-pressed={fundTableMode === mode}
                                                        onClick={() => handleFundTableModeChange(mode)}
                                                    >
                                                        {label}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                        <div className="funds-control-group">
                                            <span className="funds-control-label">Tür</span>
                                            <select
                                                className={`funds-toolbar-select${fundTypeFilter ? ' is-active' : ''}`}
                                                value={fundTypeFilter}
                                                onChange={(event) => setFundTypeFilter(event.target.value)}
                                                aria-label="Fon türü filtresi"
                                            >
                                                <option value="">Tümü</option>
                                                {visibleFundTypes.map((item) => (
                                                    <option key={item} value={item}>{item}</option>
                                                ))}
                                            </select>
                                        </div>
                                        <div className="funds-control-group">
                                            <span className="funds-control-label">Risk</span>
                                            <select
                                                className={`funds-toolbar-select${riskFilter ? ' is-active' : ''}`}
                                                value={riskFilter}
                                                onChange={(event) => setRiskFilter(event.target.value)}
                                                aria-label="Risk filtresi"
                                            >
                                                <option value="">Tümü</option>
                                                {visibleRiskValues.map((item) => (
                                                    <option key={item} value={String(item)}>Risk {item}</option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                    <div className="funds-toolbar-meta">
                                        <span className="funds-meta-chip">
                                            Kaynak: <strong>{funds?.source?.toUpperCase() || 'TEFAS'}</strong>
                                        </span>
                                        {funds?.refresh_pending && (
                                            <span className="funds-status-pill is-info" title="Fon kataloğu arka planda yenileniyor.">
                                                <span>Yenileniyor</span>
                                            </span>
                                        )}
                                        {(funds?.stale || funds?.degraded) && (
                                            <span
                                                className="funds-status-pill is-warning"
                                                title={funds?.degraded ? 'Fon snapshot cache boş veya kullanılamıyor.' : 'Son güvenilir veri gösteriliyor.'}
                                            >
                                                <AlertCircle size={12} aria-hidden="true" />
                                                <span>{funds?.degraded ? 'Cache uyarısı' : 'Son sağlıklı veri'}</span>
                                            </span>
                                        )}
                                        {refreshError && (
                                            <span className="funds-status-pill is-danger" title={refreshError}>
                                                <AlertCircle size={12} aria-hidden="true" />
                                                <span>Yenileme başarısız</span>
                                            </span>
                                        )}
                                        <span className="funds-meta-time" title="Son fon veri tarihi">
                                            <Clock3 size={12} aria-hidden="true" />
                                            <span>{formatDate(funds?.as_of)}</span>
                                        </span>
                                        <button
                                            type="button"
                                            className={`funds-refresh-btn${refreshing ? ' is-spinning' : ''}`}
                                            onClick={() => { void refreshSnapshot(); }}
                                            disabled={refreshing}
                                            title="Fon listesini yenile"
                                            aria-label="Fon listesini yenile"
                                        >
                                            <RefreshCw size={12} aria-hidden="true" />
                                        </button>
                                    </div>
                                </div>
                            </div>

                            {(Boolean(refreshError) || funds?.stale || funds?.degraded || funds?.warnings?.length) && (
                                <div className="funds-source-warning">
                                    <ShieldAlert size={14} aria-hidden="true" />
                                    <span>
                                        {refreshError
                                            ? `Fon listesi yenilenemedi: ${refreshError}`
                                            : refreshing
                                              ? 'Fon listesi yeniden yükleniyor.'
                                              : funds?.refresh_pending
                                                ? 'Fon kataloğu arka planda otomatik yenileniyor.'
                                                : funds?.degraded
                                                  ? 'Fon snapshot cache boş veya kullanılamıyor.'
                                                  : funds?.stale && funds?.source_metadata?.snapshot_as_of && funds?.source_metadata?.price_history_as_of
                                                    ? `Fon kataloğu ${formatDate(funds.source_metadata.snapshot_as_of)} tarihli; fiyat verileri ${formatDate(funds.source_metadata.price_history_as_of)} tarihine kadar güncel.`
                                                    : funds?.stale
                                                      ? 'Fon kataloğu güncel snapshot alınamadığı için son güvenilir veriyle gösteriliyor.'
                                                      : funds?.warnings?.length
                                                        ? `Fon verisi TEFAS birincil kaynağından alınamadı; fallback sonucu gösteriliyor. ${funds.warnings[0]}`
                                                        : 'Fon kaynak durumu uyarı verdi.'}
                                    </span>
                                    <button
                                        type="button"
                                        className="funds-banner-btn"
                                        onClick={() => { void refreshSnapshot(); }}
                                        disabled={refreshing}
                                    >
                                        <RefreshCw size={12} aria-hidden="true" className={refreshing ? 'funds-spin' : undefined} />
                                        {refreshing ? 'Yükleniyor' : 'Yenile'}
                                    </button>
                                </div>
                            )}

                            {isFundScreenerActive && (
                                <div className="funds-screener-active-banner">
                                    <div className="funds-screener-active-banner-left">
                                        <SlidersHorizontal size={13} aria-hidden="true" />
                                        <span>
                                            <strong>{activeFundFilterCount} filtre devrede:</strong> {filteredFunds.length} / {funds?.total_count ?? funds?.count ?? 0} fon gösteriliyor
                                        </span>
                                    </div>
                                    <div className="funds-screener-active-banner-actions">
                                        <button
                                            type="button"
                                            className="funds-screener-banner-btn"
                                            onClick={() => setFundScreenerOpen(true)}
                                        >
                                            Filtreleri Düzenle
                                        </button>
                                        <button
                                            type="button"
                                            className="funds-screener-banner-btn funds-screener-banner-clear"
                                            onClick={clearFundScreenerFilters}
                                        >
                                            <X size={12} aria-hidden="true" /> Temizle
                                        </button>
                                    </div>
                                </div>
                            )}

                            <FundsScreenerModal
                                open={fundScreenerOpen}
                                onClose={() => setFundScreenerOpen(false)}
                                totalFundsCount={funds?.total_count ?? funds?.count ?? 0}
                                filteredFundsCount={filteredFunds.length}
                                activePreset={activeFundScreenerPreset}
                                onApplyPreset={applyFundScreenerPreset}
                                activeChips={activeFundChips}
                                fundTypeFilter={fundTypeFilter}
                                onFundTypeFilterChange={(value) => {
                                    setActiveFundScreenerPreset(null);
                                    setFundTypeFilter(value);
                                }}
                                riskFilter={riskFilter}
                                onRiskFilterChange={(value) => {
                                    setActiveFundScreenerPreset(null);
                                    setRiskFilter(value);
                                }}
                                visibleFundTypes={visibleFundTypes}
                                visibleRiskValues={visibleRiskValues}
                                visibleFounders={visibleFounders}
                                filters={fundScreenerFilters}
                                onFilterChange={handleFundScreenerFilterChange}
                                isScreenerActive={isFundScreenerActive}
                                onClearFilters={clearFundScreenerFilters}
                            />
                            {loading ? (
                                <FundsListLoader />
                            ) : error ? (
                                <div className="funds-state funds-state-error">
                                    <span>{error}</span>
                                    <button
                                        type="button"
                                        className="funds-retry-btn"
                                        onClick={() => void refreshSnapshot()}
                                        disabled={refreshing}
                                    >
                                        {refreshing ? 'Yenileniyor…' : 'Tekrar Dene'}
                                    </button>
                                </div>
                            ) : filteredFunds.length === 0 ? (
                                <div className="funds-state">
                                    Aranan kritere uyan fon bulunamadı.
                                </div>
                            ) : (
                                <FundsTable
                                    rows={filteredFunds}
                                    mode={fundTableMode}
                                    sortKey={sortKey}
                                    sortOrder={sortOrder}
                                    onSort={setTableSort}
                                    onOpenFund={onOpenFund}
                                    resetDeps={tableResetDeps}
                                />
                            )}
                        </section>
                    ) : (
                        <section className="fund-detail">
                            {detailLoading && !selectedFund ? (
                                <FundRequestState mode="loading" symbol={fundCode} />
                            ) : selectedFund ? (
                                <>
                                    <header className="fund-market-shell">
                                        <div className="fund-market-breadcrumb" aria-label="Fon konumu">
                                            <button
                                                type="button"
                                                className="fund-breadcrumb-back fund-breadcrumb-group"
                                                onClick={onBack}
                                                aria-label="Yatırım fonları listesine dön"
                                            >
                                                <Database size={17} aria-hidden="true" />
                                                Yatırım Fonları
                                            </button>
                                            <ChevronRight size={14} aria-hidden="true" />
                                            <span className="fund-breadcrumb-code">
                                                <SymbolLogo
                                                    symbol={selectedFund.fund_code}
                                                    name={selectedFund.founder_company || selectedFund.manager_company || selectedFund.name}
                                                    kind="fund"
                                                    size="xs"
                                                />
                                                {selectedFund.fund_code}
                                            </span>
                                        </div>

                                        <div className="fund-market-hero">
                                            <div className="fund-market-title">
                                                <SymbolLogo
                                                    symbol={selectedFund.fund_code}
                                                    name={selectedFund.founder_company || selectedFund.manager_company || selectedFund.name}
                                                    kind="fund"
                                                    size="lg"
                                                    className="fund-market-logo"
                                                />
                                                <div className="fund-market-copy">
                                                    <div className="fund-market-code-row">
                                                        <h1>{selectedFund.fund_code}</h1>
                                                        <button
                                                            type="button"
                                                            className={`fund-icon-action${isStarred ? ' active' : ''}`}
                                                            onClick={toggleStarredFund}
                                                            aria-pressed={isStarred}
                                                            aria-label={isStarred ? `${selectedFund.fund_code} izleme listesinden çıkar` : `${selectedFund.fund_code} izleme listesine ekle`}
                                                            title={isStarred ? 'İzleme listesinden çıkar' : 'İzleme listesine ekle'}
                                                        >
                                                            <Star size={19} aria-hidden="true" />
                                                        </button>
                                                    </div>
                                                    <p>{selectedFund.name}</p>
                                                    <small>
                                                        {selectedFund.fund_type || 'Fon'}
                                                        {selectedFund.founder_company || selectedFund.manager_company
                                                            ? ` · ${selectedFund.founder_company || selectedFund.manager_company}`
                                                            : ''}
                                                    </small>
                                                </div>
                                            </div>

                                            <div className="fund-market-quote">
                                                <div>
                                                    <strong>{formatQuotePrice(detailLatestPrice, selectedFund.currency)}</strong>
                                                    <span className={pctClass(selectedFund.daily_return)}>{formatPct(selectedFund.daily_return)}</span>
                                                </div>
                                                <small>{formatDate(selectedFund.as_of)} · {selectedFund.source || 'TEFAS'}</small>
                                            </div>
                                        </div>

                                        <div className="fund-market-stats">
                                            <div><span>Portföy</span><strong>{formatCompactCurrency(selectedFund.aum)}</strong></div>
                                            <div><span>Yatırımcı</span><strong>{selectedFund.investor_count?.toLocaleString('tr-TR') || '-'}</strong></div>
                                            <div><span>Risk</span><strong>{selectedFund.risk_value ?? '-'}</strong></div>
                                            <div><span>Kaynak</span><strong>{selectedFund.source?.toUpperCase() || 'TEFAS'}</strong></div>
                                        </div>

                                        <nav className="fund-tabs">
                                            {FUND_TABS.map((tab) => {
                                                const Icon = tab.icon;
                                                return (
                                                    <Link
                                                        key={tab.key}
                                                        to={toFundDetail(selectedFund.fund_code, tab.key)}
                                                        className={activeTab === tab.key ? 'active' : ''}
                                                        onClick={(event: ReactMouseEvent<HTMLAnchorElement>) => {
                                                            if (
                                                                event.button !== 0
                                                                || event.metaKey
                                                                || event.ctrlKey
                                                                || event.shiftKey
                                                                || event.altKey
                                                            ) {
                                                                return;
                                                            }
                                                            event.preventDefault();
                                                            onTabChange(tab.key);
                                                        }}
                                                    >
                                                        <Icon size={16} aria-hidden="true" />
                                                        {tab.label}
                                                    </Link>
                                                );
                                            })}
                                        </nav>
                                    </header>

                                    {detailLoading ? (
                                        <FundRequestState
                                            mode="loading"
                                            symbol={selectedFund.fund_code}
                                            fundName={selectedFund.name}
                                            compact
                                        />
                                    ) : detailError ? (
                                        <FundRequestState
                                            mode="error"
                                            symbol={selectedFund.fund_code}
                                            fundName={selectedFund.name}
                                            compact
                                            onRetry={() => {
                                                if (fundCode) void loadFundDetail(fundCode);
                                            }}
                                        />
                                    ) : null}

                                    {detailSourceWarnings.length > 0 && (
                                        <div className="funds-source-warning funds-source-warning-detail" role="status" aria-live="polite">
                                            <ShieldAlert size={17} aria-hidden="true" />
                                            <div className="funds-source-warning-details">
                                                {detailSourceWarnings.map((warning) => (
                                                    <div className="funds-source-warning-line" key={warning}>{warning}</div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {activeTab === 'overview' && (
                                        <div className="fund-overview-stack">
                                            <div className="fund-overview-grid">
                                                <FundPerformanceChart
                                                    fundCode={selectedFund.fund_code}
                                                    points={chartPoints}
                                                    comparisonAssets={comparisonState.selectedAssets}
                                                    comparisonHistory={comparisonHistory}
                                                    comparisonLoading={comparisonHistoryLoading}
                                                    comparisonError={comparisonHistoryError}
                                                    comparison={comparisonState}
                                                    selectedRange={chartRange}
                                                    pendingRange={pendingChartRange || (activeRangeJobPending ? chartRange : null)}
                                                    historyPending={fintablesHistoryPending}
                                                    rangeError={chartRangeError}
                                                    customStartDate={customStartDate}
                                                    customEndDate={customEndDate}
                                                    maxDate={latestAvailableDate}
                                                    visiblePeriods={visibleDetailPeriods}
                                                    activeRangeCovered={activeRangeCovered}
                                                    periodReturns={detailPeriodReturns}
                                                    yieldPeriods={yieldSummary?.periods}
                                                    currency={selectedFund.currency}
                                                    onRangeSelect={handleChartRangeSelect}
                                                    onCustomStartDateChange={handleCustomStartDateChange}
                                                    onCustomEndDateChange={handleCustomEndDateChange}
                                                />
                                                <FundAllocationSummary
                                                    allocations={allocationRows}
                                                    loading={allocationLoading && !allocations}
                                                    onOpenHistory={openAllocationHistory}
                                                    historyLoading={allocationHistoryLoading}
                                                />
                                            </div>
                                            <FundOverviewMetrics series={overviewMetricSeries} />
                                            <FundCategoryRankingsPanel rankings={detail?.category_rankings} />
                                            <FundOverviewHoldingsSection
                                                holdings={holdings}
                                                loading={holdingsLoading}
                                                error={holdingsError}
                                            />
                                            <FundReturnComparison
                                                comparison={comparisonState}
                                                containerRef={comparisonPanelRef}
                                                comparisonHistory={comparisonHistory}
                                                basePerformancePoints={visiblePerformancePoints}
                                            />
                                            {yieldLoading && !yieldSummary && (
                                                <FinLoader message="Getiri kartları yükleniyor" />
                                            )}
                                            <FundMonthlyHeatmap monthlyReturns={monthlyReturns} />
                                        </div>
                                    )}

                                    {activeTab === 'allocation' && (
                                        <div className="fund-detail-panel">
                                            <h2>Portföy Dağılımı</h2>
                                            <div className="fund-allocation-stack">
                                                {allocationRows.length ? (
                                                    <div className="fund-allocation-detail">
                                                        <FundAllocationSummary
                                                            allocations={allocationRows}
                                                            loading={allocationLoading && !allocations}
                                                            onOpenHistory={openAllocationHistory}
                                                            historyLoading={allocationHistoryLoading}
                                                        />
                                                        <div className="fund-allocation-table-wrap">
                                                            <table className="fund-history-table">
                                                                <thead>
                                                                    <tr>
                                                                        <th>Varlık</th>
                                                                        <th>Kod</th>
                                                                        <th>Ağırlık</th>
                                                                        <th>Tarih</th>
                                                                    </tr>
                                                                </thead>
                                                                <tbody>
                                                                    {allocationRows.map((item) => (
                                                                        <tr key={item.allocation_type}>
                                                                            <td>{item.label}</td>
                                                                            <td>{item.allocation_type.toUpperCase()}</td>
                                                                            <td className={pctClass(item.weight)}>{formatAllocationWeight(item.weight)}</td>
                                                                            <td>{formatDate(item.report_date)}</td>
                                                                        </tr>
                                                                    ))}
                                                                </tbody>
                                                            </table>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    allocationLoading ? (
                                                        <FinLoader message="Dağılım verisi yükleniyor" />
                                                    ) : (
                                                        <div className="funds-state">TEFAS dağılım verisi bu fon için yok.</div>
                                                    )
                                                )}
                                                <FundHoldingsPanel
                                                    holdings={holdings}
                                                    loading={holdingsLoading}
                                                    error={holdingsError}
                                                    categoryFilter={holdingCategoryFilter}
                                                    changeFilter={holdingChangeFilter}
                                                    onCategoryFilterChange={setHoldingCategoryFilter}
                                                    onChangeFilterChange={setHoldingChangeFilter}
                                                    onOpenTicker={onOpenTicker}
                                                    onOpenFund={onOpenFund}
                                                />
                                            </div>
                                        </div>
                                    )}

                                            {activeTab === 'history' && (
                                        <div className="fund-detail-panel fund-history-panel">
                                            <div className="fund-history-head">
                                                <h2>Geçmiş Veriler</h2>
                                                <div className="fund-history-tabs" role="tablist" aria-label="Geçmiş veri türü">
                                                    {FUND_HISTORY_TABS.map((tab) => {
                                                        const Icon = tab.icon;
                                                        return (
                                                            <button
                                                                key={tab.key}
                                                                type="button"
                                                                role="tab"
                                                                aria-selected={historySubtab === tab.key}
                                                                className={historySubtab === tab.key ? 'active' : ''}
                                                                onClick={() => setHistorySubtab(tab.key)}
                                                            >
                                                                <Icon size={15} aria-hidden="true" />
                                                                {tab.label}
                                                            </button>
                                                        );
                                                    })}
                                                </div>
                                            </div>
                                            {historySubtab === 'prices' ? (
                                                fintablesHistoryPending ? (
                                                    <FinLoader message="Getiri geçmişi hazırlanıyor" />
                                                ) : performanceError ? (
                                                    <div className="funds-state funds-state-error">{performanceError}</div>
                                                ) : visiblePerformancePoints.length ? (
                                                    <>
                                                        <FundPeriodStatsStrip stats={performance?.period_stats} />
                                                        <FundHistoryTable points={visiblePerformancePoints} />
                                                    </>
                                                ) : (
                                                    <div className="funds-state">Geçmiş fiyat verisi henüz hazır değil.</div>
                                                )
                                            ) : (
                                                <FundAllocationHistoryPanel
                                                    history={allocationHistory}
                                                    loading={allocationHistoryLoading}
                                                    error={allocationHistoryError}
                                                />
                                            )}
                                        </div>
                                    )}
                                </>
                            ) : detailError ? (
                                <FundRequestState
                                    mode="error"
                                    symbol={fundCode}
                                    onRetry={() => {
                                        if (fundCode) void loadFundDetail(fundCode);
                                    }}
                                />
                            ) : (
                                <div className="funds-state">Fon bulunamadı.</div>
                            )}
                        </section>
                    )}
                </div>
            </div>
        </div>
    );
}
