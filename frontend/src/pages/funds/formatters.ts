import { apiClient } from '../../api/client';
import type {
    FundHoldingsLiveResponse,
    FundHoldingsResponse,
    FundPerformanceResponse,
    FundPortfolioPosition,
    FundPricePoint,
    FundSummary,
    FundYieldSummaryResponse,
    FundsResponse,
    FxQuote,
    MarketIndexListRow,
    MarketUniverseRow,
} from '../../api/types';
import type { SymbolLogoKind } from '../../components/SymbolLogo';
import { STOCK_LOGO_DOMAIN_MAP } from '../../components/symbolLogoMaps';
import { isBistTradingHours } from '../../utils/marketHours';
import { formatFundQuotePrice, formatFundReportDate } from '../../utils/fundPresentation';
import {
    DETAIL_RETURN_PERIODS,
    EXTENDED_DETAIL_RETURN_PERIODS,
    STOCK_SEARCH_LABELS,
    type ComparisonAsset,
    type ComparisonAssetKind,
    type ComparisonPeriod,
    type ComparisonReturnSource,
    type FundChartRange,
    type FundHoldingAssetCategory,
    type FundHoldingCategoryFilter,
    type FundHoldingChangeFilter,
    type FundHoldingSortDirection,
    type FundHoldingSortKey,
    type FundHoldingsReportMeta,
    type FundOverviewMetricPoint,
    type FundOverviewMetricSeries,
    type FundSortKey,
    type FundsCatalogMemoryCache,
    type FundsCatalogPayload,
    type MonthlyReturn,
} from './types';

export function parseFundScreenerNumber(raw: string): number | undefined {
    const text = String(raw || '').trim().replace(',', '.');
    if (!text) return undefined;
    const value = Number(text);
    return Number.isFinite(value) ? value : undefined;
}

export function formatDate(value: string | null | undefined): string {
    return formatFundReportDate(value);
}

export function isoDateDaysAgo(days: number): string {
    const date = new Date();
    date.setDate(date.getDate() - days);
    return date.toISOString().slice(0, 10);
}

export function isoDateMonthsAgo(months: number): string {
    const date = new Date();
    date.setMonth(date.getMonth() - months);
    return date.toISOString().slice(0, 10);
}

export function formatCurrency(value: number | null | undefined, currency = 'TRY'): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const prefix = currency === 'TRY' ? '₺' : `${currency} `;
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 4,
    })}`;
}

export function formatCompactCurrency(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const abs = Math.abs(value);
    if (abs >= 1_000_000_000) {
        return `₺${(value / 1_000_000_000).toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })} mr`;
    }
    if (abs >= 1_000_000) {
        return `₺${(value / 1_000_000).toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })} mn`;
    }
    return formatCurrency(value);
}

export function formatSignedCompactCurrency(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    if (value > 0) return `+${formatCompactCurrency(value)}`;
    if (value < 0) return `-${formatCompactCurrency(Math.abs(value))}`;
    return formatCompactCurrency(value);
}

export function formatQuotePrice(value: number | null | undefined, currency = 'TRY'): string {
    return formatFundQuotePrice(value, currency);
}

export function isBistMarketOpen(now = new Date()): boolean {
    return isBistTradingHours(now);
}

export function formatHoldingPrice(value: number | null | undefined, currency: string | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const abs = Math.abs(value);
    const prefix = currency === 'TRY' ? '₺' : currency ? `${currency} ` : '';
    if (abs >= 100) {
        return `${prefix}${value.toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })}`;
    }
    if (abs >= 10) {
        return `${prefix}${value.toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 4,
        })}`;
    }
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 4,
        maximumFractionDigits: 6,
    })}`;
}

export function formatYieldBounds(
    period: FundYieldSummaryResponse['periods'][typeof DETAIL_RETURN_PERIODS[number]['key']] | undefined,
    currency = 'TRY',
): string {
    if (!period) return '';
    const low = formatCurrency(period.low, currency);
    const high = formatCurrency(period.high, currency);
    if (low === '-' && high === '-') return '';
    return `D ${low} / Y ${high}`;
}

export function formatPct(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const sign = value > 0 ? '+' : '';
    return `% ${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

export function formatSignedInteger(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '';
    const sign = value > 0 ? '+' : '';
    return `${sign}${Math.trunc(value).toLocaleString('tr-TR')}`;
}

export function pctClass(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return 'funds-flat';
    if (value > 0) return 'funds-up';
    if (value < 0) return 'funds-down';
    return 'funds-flat';
}

export function normalizeCompareSymbol(value: string | null | undefined): string {
    return String(value || '').trim().toUpperCase().replace(/\s+/g, '');
}

export function normalizeCompareSearch(value: string | null | undefined): string {
    return String(value || '')
        .toLocaleUpperCase('tr-TR')
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .replace(/[^A-Z0-9]+/g, ' ')
        .trim();
}

export function compactCompareSearch(value: string | null | undefined): string {
    return normalizeCompareSearch(value).replace(/\s+/g, '');
}

export function comparisonAssetId(kind: ComparisonAssetKind, symbol: string): string {
    return `${kind}:${normalizeCompareSymbol(symbol)}`;
}

export function periodReturnFromAsset(asset: Pick<ComparisonAsset, 'returns'>, period: ComparisonPeriod): number | null {
    const value = asset.returns[period];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function marketReturnForPeriod(
    row: ComparisonReturnSource,
    period: ComparisonPeriod,
): number | null {
    const map: Record<ComparisonPeriod, keyof typeof row> = {
        '1w': 'return_1w_pct',
        '1m': 'return_1m_pct',
        '3m': 'return_3m_pct',
        '6m': 'return_6m_pct',
        ytd: 'return_ytd_pct',
        '1y': 'return_1y_pct',
    };
    const value = row[map[period]];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function returnsFromMarketRow(
    row: ComparisonReturnSource,
): Partial<Record<ComparisonPeriod, number | null>> {
    return {
        '1w': marketReturnForPeriod(row, '1w'),
        '1m': marketReturnForPeriod(row, '1m'),
        '3m': marketReturnForPeriod(row, '3m'),
        '6m': marketReturnForPeriod(row, '6m'),
        ytd: marketReturnForPeriod(row, 'ytd'),
        '1y': marketReturnForPeriod(row, '1y'),
    };
}

export function fundToComparisonAsset(row: FundSummary): ComparisonAsset {
    const symbol = normalizeCompareSymbol(row.fund_code);
    return {
        id: comparisonAssetId('fund', symbol),
        kind: 'fund',
        symbol,
        label: row.name,
        logoName: row.founder_company || row.manager_company || row.name,
        returns: {
            '1w': row.period_returns?.['1w'] ?? null,
            '1m': row.period_returns?.['1m'] ?? null,
            '3m': row.period_returns?.['3m'] ?? null,
            '6m': row.period_returns?.['6m'] ?? null,
            ytd: row.period_returns?.ytd ?? null,
            '1y': row.period_returns?.['1y'] ?? null,
        },
    };
}

export function stockToComparisonAsset(row: MarketUniverseRow): ComparisonAsset {
    const symbol = normalizeCompareSymbol(row.symbol || row.company);
    const label = row.name || STOCK_SEARCH_LABELS[symbol]?.[0] || 'Hisse';
    return {
        id: comparisonAssetId('stock', symbol),
        kind: 'stock',
        symbol,
        label,
        logoName: label,
        logoUrl: row.logo_url,
        returns: returnsFromMarketRow(row),
    };
}

export function indexToComparisonAsset(row: MarketIndexListRow): ComparisonAsset {
    const symbol = normalizeCompareSymbol(row.symbol);
    return {
        id: comparisonAssetId('index', symbol),
        kind: 'index',
        symbol,
        label: row.label,
        logoName: row.label,
        returns: returnsFromMarketRow({
            change_pct: row.change_pct,
            return_1w_pct: row.return_1w_pct,
            return_1m_pct: row.return_1m_pct,
            return_3m_pct: row.return_3m_pct,
            return_6m_pct: row.return_6m_pct,
            return_ytd_pct: row.return_ytd_pct,
            return_1y_pct: row.return_1y_pct,
        }),
    };
}

export function fxToComparisonAsset(row: FxQuote): ComparisonAsset | null {
    const rawSymbol = String(row.symbol || '').trim().toUpperCase();
    if (!rawSymbol.endsWith('/TRY')) return null;
    const returns = returnsFromMarketRow(row);
    const hasReturn = Object.values(returns).some((value) => typeof value === 'number' && Number.isFinite(value));
    if (!hasReturn) return null;
    const displaySymbol = rawSymbol.replace('/', '');
    return {
        id: comparisonAssetId('fx', rawSymbol),
        kind: 'fx',
        symbol: rawSymbol,
        displaySymbol,
        label: row.label,
        logoName: row.label,
        logoUrl: row.logo_url,
        returns,
    };
}

export function comparisonAssetKindLabel(kind: ComparisonAssetKind): string {
    if (kind === 'fund') return 'Fon';
    if (kind === 'stock') return 'Hisse';
    if (kind === 'fx') return 'Döviz';
    return 'Endeks';
}

export function comparisonLogoKind(kind: ComparisonAssetKind): 'fund' | 'stock' | 'index' | 'fx' {
    if (kind === 'fund') return 'fund';
    if (kind === 'stock') return 'stock';
    if (kind === 'fx') return 'fx';
    return 'index';
}

export function shouldRefreshSnapshot(payload: FundsResponse | null): boolean {
    if (!payload) return false;
    const visibleCount = payload.count ?? payload.rows?.length ?? 0;
    const snapshotCount = payload.total_count ?? 0;
    return visibleCount === 0 && (
        snapshotCount === 0 ||
        payload.degraded ||
        payload.stale ||
        payload.status === 'unavailable'
    );
}

export const FUNDS_CATALOG_MEMORY_TTL_MS = 5 * 60 * 1000;
let fundsCatalogMemoryCache: FundsCatalogMemoryCache | null = null;
let fundsCatalogInFlight: Promise<FundsCatalogPayload> | null = null;

export async function fetchFundsCatalog(options: { bypassMemoryCache?: boolean } = {}): Promise<FundsCatalogPayload> {
    const bypassMemoryCache = Boolean(options.bypassMemoryCache);
    const now = Date.now();
    if (!bypassMemoryCache && fundsCatalogMemoryCache && now - fundsCatalogMemoryCache.fetchedAt < FUNDS_CATALOG_MEMORY_TTL_MS) {
        return {
            funds: fundsCatalogMemoryCache.funds,
            categories: fundsCatalogMemoryCache.categories,
        };
    }
    if (!bypassMemoryCache && fundsCatalogInFlight) return fundsCatalogInFlight;

    const task = (async () => {
        // Refresh is an admin/workflow operation. Public clients only reread
        // the current canonical snapshot and never carry an admin token.
        const [fundPayload, categoryPayload] = await Promise.all([apiClient.funds(), apiClient.fundCategories()]);
        const result = { funds: fundPayload, categories: categoryPayload };
        fundsCatalogMemoryCache = { ...result, fetchedAt: Date.now() };
        return result;
    })();

    if (!bypassMemoryCache) {
        fundsCatalogInFlight = task;
        task.finally(() => {
            if (fundsCatalogInFlight === task) fundsCatalogInFlight = null;
        }).catch(() => undefined);
    }
    return task;
}

export function searchCachedFundCatalog(queryText: string, limit: number): FundSummary[] | null {
    const rows = fundsCatalogMemoryCache?.funds?.rows;
    if (!rows || rows.length === 0) return null;
    const needle = normalizeCompareSearch(queryText.trim());
    if (!needle) return [];
    const exactCode: FundSummary[] = [];
    const prefixCode: FundSummary[] = [];
    const otherMatches: FundSummary[] = [];
    for (const row of rows) {
        const code = normalizeCompareSearch(row.fund_code || '');
        const haystack = normalizeCompareSearch(
            `${row.fund_code || ''} ${row.name || ''} ${row.fund_type || ''} ${row.founder_company || ''} ${row.manager_company || ''}`,
        );
        if (!haystack.includes(needle)) continue;
        if (code === needle) exactCode.push(row);
        else if (code.startsWith(needle)) prefixCode.push(row);
        else otherMatches.push(row);
    }
    return [...exactCode, ...prefixCode, ...otherMatches].slice(0, limit);
}

export function fundSortValue(row: FundSummary, key: FundSortKey): string | number | null {
    switch (key) {
        case 'return_1w':
            return row.period_returns?.['1w'] ?? null;
        case 'return_1m':
            return row.period_returns?.['1m'] ?? null;
        case 'return_3m':
            return row.period_returns?.['3m'] ?? null;
        case 'return_6m':
            return row.period_returns?.['6m'] ?? null;
        case 'return_ytd':
            return row.period_returns?.ytd ?? null;
        case 'return_1y':
            return row.period_returns?.['1y'] ?? null;
        case 'investor_count':
            return row.investor_count ?? null;
        case 'founder_company':
            return row.founder_company || row.manager_company || null;
        default:
            return row[key] ?? null;
    }
}

export function compareFundRows(a: FundSummary, b: FundSummary, key: FundSortKey, order: 'asc' | 'desc'): number {
    const av = fundSortValue(a, key);
    const bv = fundSortValue(b, key);
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    const direction = order === 'desc' ? -1 : 1;
    if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * direction;
    return String(av).localeCompare(String(bv), 'tr', { numeric: true }) * direction;
}

export function getFundsCatalogMemoryCache(): FundsCatalogMemoryCache | null {
    return fundsCatalogMemoryCache;
}

export function updateFundsCatalogSnapshot(funds: FundsResponse): void {
    if (fundsCatalogMemoryCache) {
        fundsCatalogMemoryCache = { ...fundsCatalogMemoryCache, funds, fetchedAt: Date.now() };
    }
}

export function returnBetween(latestPrice: number | null | undefined, basePrice: number | null | undefined): number | null {
    if (latestPrice == null || basePrice == null || !Number.isFinite(latestPrice) || !Number.isFinite(basePrice) || basePrice <= 0) {
        return null;
    }
    return ((latestPrice / basePrice) - 1) * 100;
}

export function sortFundPoints(points: FundPerformanceResponse['points'] | undefined): FundPricePoint[] {
    return [...(points || [])]
        .filter((point) => Number.isFinite(Number(point.price)) && Number(point.price) > 0 && point.date)
        .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
}

export function latestFundDate(...values: Array<string | null | undefined>): string | null {
    return values
        .filter((value): value is string => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value))
        .sort((left, right) => left.localeCompare(right))
        .at(-1) || null;
}

export function shiftIsoDateDays(isoDate: string, deltaDays: number): string {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(isoDate || '').trim());
    if (!match) return isoDate;
    const utcMs = Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]) + deltaDays);
    return new Date(utcMs).toISOString().slice(0, 10);
}

export function rangeStartDate(range: FundChartRange, endDateIso: string, customStartDate: string): string {
    const end = new Date(endDateIso);
    if (Number.isNaN(end.getTime())) return isoDateDaysAgo(30);
    const start = new Date(end);
    if (range === '1w') start.setDate(start.getDate() - 7);
    if (range === '1m') start.setMonth(start.getMonth() - 1);
    if (range === '3m') start.setMonth(start.getMonth() - 3);
    if (range === '6m') start.setMonth(start.getMonth() - 6);
    if (range === 'ytd') {
        start.setMonth(0, 1);
    }
    if (range === '1y') start.setFullYear(start.getFullYear() - 1);
    if (range === '3y') start.setFullYear(start.getFullYear() - 3);
    if (range === '5y') start.setFullYear(start.getFullYear() - 5);
    if (range === 'custom') return customStartDate || isoDateMonthsAgo(6);
    return start.toISOString().slice(0, 10);
}

export function isoWeekBucketKey(isoDate: string): string {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(isoDate || '').trim());
    if (!match) return isoDate;
    const d = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])));
    const dayNum = d.getUTCDay() || 7;
    d.setUTCDate(d.getUTCDate() + 4 - dayNum);
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
    const weekNo = Math.ceil((((d.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
    return `${d.getUTCFullYear()}-W${String(weekNo).padStart(2, '0')}`;
}

export function downsamplePointsWeekly<T extends { date: string }>(points: T[]): T[] {
    if (points.length <= 2) return points;
    const first = points[0];
    const last = points[points.length - 1];
    const byWeek = new Map<string, T>();
    for (const point of points) {
        byWeek.set(isoWeekBucketKey(point.date), point);
    }
    const sampled = Array.from(byWeek.values()).sort((a, b) => a.date.localeCompare(b.date));
    if (sampled[0]?.date !== first.date) {
        sampled.unshift(first);
    }
    if (sampled[sampled.length - 1]?.date !== last.date) {
        sampled.push(last);
    }
    return sampled;
}

export function filterPointsForRange(
    points: FundPricePoint[],
    range: FundChartRange,
    endDateIso: string,
    customStartDate: string,
    customEndDate: string,
): FundPricePoint[] {
    if (range === 'all') return points;
    const startIso = rangeStartDate(range, endDateIso, customStartDate);
    const effectiveEndIso = range === 'custom' ? (customEndDate || endDateIso) : endDateIso;
    const startTs = new Date(startIso).getTime();
    const endTs = new Date(effectiveEndIso).getTime();
    if (!Number.isFinite(startTs) || !Number.isFinite(endTs)) return points;
    const inRange = points.filter((point) => {
        const ts = new Date(point.date).getTime();
        return Number.isFinite(ts) && ts >= startTs && ts <= endTs;
    });
    const isWeeklyRange = range === '3y' || range === '5y';
    if (!inRange.length || points[0]?.date >= startIso) {
        return isWeeklyRange ? downsamplePointsWeekly(inRange) : inRange;
    }

    let anchor: FundPricePoint | null = null;
    for (const point of points) {
        const ts = new Date(point.date).getTime();
        if (!Number.isFinite(ts) || ts > startTs) break;
        anchor = point;
    }
    const sliced = !anchor || anchor.date === inRange[0]?.date ? inRange : [anchor, ...inRange];
    return isWeeklyRange ? downsamplePointsWeekly(sliced) : sliced;
}

export function hasUsableRangeCoverage(points: FundPricePoint[], startIso: string, endIso: string): boolean {
    const filtered = points.filter((point) => point.date >= startIso && point.date <= endIso);
    if (filtered.length < 2) return false;
    const earliest = filtered[0];
    const earliestTs = new Date(earliest.date).getTime();
    const startTs = new Date(startIso).getTime();
    const startToleranceMs = 10 * 24 * 60 * 60 * 1000;
    if (!Number.isFinite(earliestTs) || !Number.isFinite(startTs) || earliestTs > startTs + startToleranceMs) {
        return false;
    }
    const latest = filtered[filtered.length - 1];
    const latestTs = new Date(latest.date).getTime();
    const endTs = new Date(endIso).getTime();
    return Number.isFinite(latestTs) && Number.isFinite(endTs) && latestTs >= endTs - 10 * 24 * 60 * 60 * 1000;
}

export function hasFullHistoryPerformance(performance: FundPerformanceResponse | null): boolean {
    return performance?.source_metadata?.full_history_requested === true;
}

export function formatChartDate(value: string | null | undefined, range: FundChartRange): string {
    if (!value) return '-';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return new Intl.DateTimeFormat('tr-TR', {
        day: '2-digit',
        month: 'short',
        ...(range === '1w' ? {} : { year: '2-digit' }),
    }).format(date);
}

export function monthKey(value: string): string {
    return value.slice(0, 7);
}

export function formatMonthLabel(value: string): string {
    const date = new Date(`${value}-01T00:00:00`);
    if (Number.isNaN(date.getTime())) return value;
    return new Intl.DateTimeFormat('tr-TR', { month: 'short', year: 'numeric' }).format(date);
}

export function formatMonthShortLabel(value: string): string {
    const date = new Date(`${value}-01T00:00:00`);
    if (Number.isNaN(date.getTime())) return value;
    return new Intl.DateTimeFormat('tr-TR', { month: 'short' }).format(date);
}

export function formatMonthLegendLabel(value: string | null | undefined, fallback: string): string {
    if (!value) return fallback;
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return fallback;
    return new Intl.DateTimeFormat('tr-TR', { month: 'long' })
        .format(date)
        .toLocaleUpperCase('tr-TR');
}

export function monthlyReturnsFromPoints(points: FundPricePoint[]): MonthlyReturn[] {
    const monthEndPoints = new Map<string, FundPricePoint>();
    for (const point of points) {
        const key = monthKey(point.date);
        const current = monthEndPoints.get(key);
        if (!current || point.date > current.date) {
            monthEndPoints.set(key, point);
        }
    }
    const ordered = [...monthEndPoints.entries()].sort(([a], [b]) => a.localeCompare(b));
    const monthly: MonthlyReturn[] = [];
    for (let index = 1; index < ordered.length; index += 1) {
        const [month, endPoint] = ordered[index];
        const [, startPoint] = ordered[index - 1];
        const startPrice = Number(startPoint.price);
        const endPrice = Number(endPoint.price);
        const returnPct = returnBetween(endPrice, startPrice);
        if (returnPct == null) continue;
        const date = new Date(`${month}-01T00:00:00`);
        monthly.push({
            month,
            label: formatMonthLabel(month),
            year: date.getFullYear(),
            monthIndex: date.getMonth(),
            returnPct,
            startPrice,
            endPrice,
        });
    }
    return monthly;
}

export function monthlyEndPointsFromFundPoints(points: FundPricePoint[]): Array<[string, FundPricePoint]> {
    const monthEndPoints = new Map<string, FundPricePoint>();
    for (const point of points) {
        const key = monthKey(point.date);
        const current = monthEndPoints.get(key);
        if (!current || point.date > current.date) {
            monthEndPoints.set(key, point);
        }
    }
    return [...monthEndPoints.entries()].sort(([a], [b]) => a.localeCompare(b));
}

export function monthlyMetricPointsFromFundPoints(
    points: FundPricePoint[],
    valueForPoint: (point: FundPricePoint) => number | null | undefined,
): Map<string, FundPricePoint> {
    const monthEndPoints = new Map<string, FundPricePoint>();
    for (const point of points) {
        const value = valueForPoint(point);
        if (value == null || !Number.isFinite(value)) continue;
        const key = monthKey(point.date);
        const current = monthEndPoints.get(key);
        if (!current || point.date > current.date) {
            monthEndPoints.set(key, point);
        }
    }
    return monthEndPoints;
}

export function estimateCashFlow(current: FundPricePoint, previous: FundPricePoint): number | null {
    const currentAum = finiteNumber(current.aum);
    const previousAum = finiteNumber(previous.aum);
    if (currentAum == null || previousAum == null) return null;
    const currentPrice = finiteNumber(current.price);
    const previousPrice = finiteNumber(previous.price);
    if (currentPrice != null && currentPrice > 0 && previousPrice != null && previousPrice > 0) {
        return currentAum - previousAum * (currentPrice / previousPrice);
    }
    return currentAum - previousAum;
}

export function overviewMetricsFromPoints(points: FundPricePoint[]): FundOverviewMetricSeries {
    const ordered = monthlyEndPointsFromFundPoints(points);
    const visible = ordered.slice(-6);
    const aumPointsByMonth = monthlyMetricPointsFromFundPoints(points, (point) => point.aum);
    const investorPointsByMonth = monthlyMetricPointsFromFundPoints(points, (point) => point.investor_count);
    const cashFlowByMonth = new Map<string, number | null>();
    const aumOrdered = [...aumPointsByMonth.entries()].sort(([a], [b]) => a.localeCompare(b));
    for (let index = 1; index < aumOrdered.length; index += 1) {
        cashFlowByMonth.set(aumOrdered[index][0], estimateCashFlow(aumOrdered[index][1], aumOrdered[index - 1][1]));
    }
    const metricPoint = (month: string, value: number | null): FundOverviewMetricPoint => ({
        month,
        label: formatMonthShortLabel(month),
        value: value != null && Number.isFinite(value) ? value : null,
    });
    const numericOrNull = (value: number | null | undefined): number | null => (
        value != null && Number.isFinite(value) ? value : null
    );

    return {
        aum: visible.map(([month]) => metricPoint(month, numericOrNull(aumPointsByMonth.get(month)?.aum))),
        investors: visible.map(([month]) => metricPoint(month, numericOrNull(investorPointsByMonth.get(month)?.investor_count))),
        cashFlow: visible.map(([month]) => metricPoint(month, cashFlowByMonth.get(month) ?? null)),
    };
}

export function latestFiniteValue(points: FundOverviewMetricPoint[]): number | null {
    for (let index = points.length - 1; index >= 0; index -= 1) {
        const value = points[index]?.value;
        if (value != null && Number.isFinite(value)) return value;
    }
    return null;
}

export function formatWholeNumber(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return Math.round(value).toLocaleString('tr-TR');
}

export function formatCompactOverviewNumber(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const abs = Math.abs(value);
    const sign = value < 0 ? '-' : '';
    const formatScaled = (divider: number, suffix: string) =>
        `${sign}${(abs / divider).toLocaleString('tr-TR', { maximumFractionDigits: 1 })} ${suffix}`;
    if (abs >= 1_000_000_000_000) return formatScaled(1_000_000_000_000, 'tr');
    if (abs >= 1_000_000_000) return formatScaled(1_000_000_000, 'mr');
    if (abs >= 1_000_000) return formatScaled(1_000_000, 'mn');
    if (abs >= 1_000) return formatScaled(1_000, 'bin');
    return formatWholeNumber(value);
}

export function formatOverviewMetricFullValue(value: number | null | undefined, suffix?: string): string {
    const formatted = formatWholeNumber(value);
    if (formatted === '-') return formatted;
    return suffix ? `${formatted} ${suffix}` : formatted;
}

export function pointOnOrBefore(points: FundPricePoint[], targetDate: string): FundPricePoint | null {
    let match: FundPricePoint | null = null;
    for (const point of points) {
        if (point.date > targetDate) break;
        match = point;
    }
    return match;
}

export function periodReturnsFromPerformancePoints(
    points: FundPricePoint[],
    allowPartialPeriodFallback = false,
): FundSummary['period_returns'] {
    const ordered = sortFundPoints(points);
    const latest = ordered[ordered.length - 1];
    const first = ordered[0];
    const latestPrice = Number(latest?.price);
    if (!latest || !first || !Number.isFinite(latestPrice) || latestPrice <= 0) {
        return {};
    }
    const returns = DETAIL_RETURN_PERIODS.reduce<FundSummary['period_returns']>((acc, period) => {
        const startDate = rangeStartDate(period.key, latest.date, '');
        const basePoint = pointOnOrBefore(ordered, startDate)
            || (allowPartialPeriodFallback && first.date < latest.date ? first : null);
        acc[period.key] = returnBetween(latestPrice, Number(basePoint?.price));
        return acc;
    }, {});
    for (const period of EXTENDED_DETAIL_RETURN_PERIODS) {
        const targetDate = rangeStartDate(period.key, latest.date, '');
        const minTolerance = shiftIsoDateDays(targetDate, 7);
        const basePoint = pointOnOrBefore(ordered, targetDate)
            || (first.date <= minTolerance && first.date < latest.date ? first : null);
        const val = returnBetween(latestPrice, Number(basePoint?.price));
        if (val != null) {
            returns[period.key] = val;
        }
    }
    return returns;
}

export function formatAllocationWeight(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return `% ${value.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function holdingWeightValue(position: FundPortfolioPosition, key: 'weight' | 'previous_weight' = 'weight'): number | null {
    const value = Number(key === 'weight' ? position.weight : position.previous_weight);
    return Number.isFinite(value) ? value : null;
}

export function formatHoldingWeight(position: FundPortfolioPosition, key: 'weight' | 'previous_weight' = 'weight'): string {
    return formatAllocationWeight(holdingWeightValue(position, key));
}

export function holdingWeightWarning(
    position: FundPortfolioPosition,
    key: 'weight' | 'previous_weight' = 'weight',
): string | null {
    const warning = key === 'weight' ? position.weight_warning : position.previous_weight_warning;
    return warning ? String(warning) : null;
}

export function formatCompactWeightDelta(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const sign = value > 0 ? '+' : '';
    return `% ${sign}${value.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function normalizeHoldingAssetType(
    value: string | null | undefined,
    assetCode?: string | null | undefined,
    assetName?: string | null | undefined,
    assetRegion?: string | null | undefined,
): FundHoldingAssetCategory | null {
    const normalized = String(value || '').trim().toLowerCase();
    const region = String(assetRegion || '').trim().toLowerCase();
    const symbol = normalizeCompareSymbol(assetCode);
    const name = normalizeCompareSearch(assetName);
    if (normalized === 'foreign_fund' || (region === 'foreign' && (normalized.includes('fund') || normalized.includes('fon')))) return 'etf';
    if (normalized === 'foreign_equity' || region === 'foreign' || normalized.includes('yabanci') || normalized.includes('yabancı')) return 'foreign_equity';
    if (symbol && STOCK_LOGO_DOMAIN_MAP[symbol]) return 'local_equity';
    if (/\b(YATIRIM FONU|BORSA YATIRIM FONU|BYF|FONU)\b/.test(name)) return 'fund';
    if (normalized === 'local_equity' || normalized.includes('hisse')) return 'local_equity';
    if (normalized === 'fund' || normalized.includes('fon') || normalized.includes('byf')) return 'fund';
    return null;
}

export function holdingAssetCategory(position: FundPortfolioPosition): FundHoldingAssetCategory | null {
    return normalizeHoldingAssetType(position.asset_type, position.asset_code, position.asset_name, position.asset_region);
}

export function holdingCategoryLabel(position: FundPortfolioPosition): string {
    const category = holdingAssetCategory(position);
    if (category === 'fund') return 'Fonlar/BYF';
    if (category === 'etf') return 'ETF';
    if (category === 'foreign_equity') return 'Yabancı Hisseler';
    return 'Hisseler';
}

export function holdingLogoSymbol(position: FundPortfolioPosition): string {
    return String(position.logo_symbol || position.provider_symbol || position.asset_code || position.asset_name || '');
}

export function holdingLogoName(position: FundPortfolioPosition): string {
    const category = holdingAssetCategory(position);
    if (category === 'fund') {
        return String(position.provider_name || position.logo_symbol || position.asset_name || '');
    }
    return String(position.provider_name || position.asset_name || '');
}

export function holdingDisplayCode(position: FundPortfolioPosition): string {
    const category = holdingAssetCategory(position);
    const providerSymbol = String(position.provider_symbol || '').trim().toUpperCase();
    if ((category === 'foreign_equity' || category === 'etf') && providerSymbol) return providerSymbol;
    return String(position.asset_code || '').trim() || '-';
}

export function holdingLogoKind(category: FundHoldingAssetCategory): SymbolLogoKind {
    return category === 'fund' ? 'fund' : 'stock';
}

export function holdingChangeStatus(position: FundPortfolioPosition): FundHoldingChangeFilter | 'unchanged' {
    const raw = String(position.change_status || '').trim().toLowerCase();
    if (raw === 'new' || raw === 'increased' || raw === 'decreased' || raw === 'removed') return raw;
    const delta = Number(position.weight_change);
    if (Number.isFinite(delta)) {
        if (delta > 0.005) return 'increased';
        if (delta < -0.005) return 'decreased';
    }
    return 'unchanged';
}

export function filterFundHoldings(
    positions: FundPortfolioPosition[],
    categoryFilter: FundHoldingCategoryFilter,
    changeFilter: FundHoldingChangeFilter,
): FundPortfolioPosition[] {
    return positions.filter((position) => {
        const category = holdingAssetCategory(position);
        if (!category) return false;
        if (categoryFilter !== 'all' && category !== categoryFilter) return false;
        if (changeFilter !== 'all' && holdingChangeStatus(position) !== changeFilter) return false;
        return true;
    });
}

export function holdingPositionKey(position: FundPortfolioPosition): string {
    return normalizeCompareSymbol(position.asset_code || position.asset_name);
}

export function mergeFundHoldingsLivePayload(
    current: FundHoldingsResponse,
    live: FundHoldingsLiveResponse,
): FundHoldingsResponse {
    const liveByCode = new Map(
        (live.positions || [])
            .map((position) => [normalizeCompareSymbol(position.asset_code), position] as const)
            .filter(([key]) => Boolean(key)),
    );
    const positions = current.positions.map((position) => {
        const livePosition = liveByCode.get(holdingPositionKey(position));
        if (!livePosition) return position;
        return {
            ...position,
            price: livePosition.price,
            price_currency: livePosition.price_currency,
            return_pct: livePosition.return_pct,
            return_source: livePosition.return_source,
            return_as_of: livePosition.return_as_of,
            estimated_exposure_value: livePosition.estimated_exposure_value,
            estimated_pnl_value: livePosition.estimated_pnl_value,
            estimated_fund_return_contribution_pct: livePosition.estimated_fund_return_contribution_pct,
        };
    });
    const currentMeta = current.source_metadata as unknown as Record<string, unknown>;
    const liveMeta = live.source_metadata as unknown as Record<string, unknown>;
    return {
        ...current,
        status: live.status || current.status,
        positions,
        portfolio_effect: live.portfolio_effect ?? current.portfolio_effect,
        source_metadata: {
            ...(current.source_metadata || {}),
            live_market_enrichment: live.source_metadata,
            daily_market_enrichment:
                liveMeta.daily_market_enrichment
                ?? currentMeta.daily_market_enrichment,
            market_enrichment:
                liveMeta.market_enrichment
                ?? currentMeta.market_enrichment,
        } as FundHoldingsResponse['source_metadata'],
    };
}

export function latestHoldingReportMeta(holdings: FundHoldingsResponse | null): FundHoldingsReportMeta | null {
    const metadata = holdings?.source_metadata as { latest_report?: FundHoldingsReportMeta | null } | undefined;
    return metadata?.latest_report || null;
}

export function previousHoldingReportMeta(holdings: FundHoldingsResponse | null): FundHoldingsReportMeta | null {
    const metadata = holdings?.source_metadata as { previous_report?: FundHoldingsReportMeta | null } | undefined;
    return metadata?.previous_report || null;
}

export function holdingSortLabel(position: FundPortfolioPosition): string {
    return `${position.asset_code || ''} ${position.asset_name || ''}`.trim();
}

export function holdingNumberValue(position: FundPortfolioPosition, key: Exclude<FundHoldingSortKey, 'asset'>): number | null {
    if (key === 'weight') return holdingWeightValue(position, 'weight');
    if (key === 'previous_weight') return holdingWeightValue(position, 'previous_weight');
    const value = Number(position[key]);
    return Number.isFinite(value) ? value : null;
}

export function compareHoldingNumbers(
    a: number | null,
    b: number | null,
    direction: FundHoldingSortDirection,
): number {
    if (a == null && b == null) return 0;
    if (a == null) return 1;
    if (b == null) return -1;
    const result = a - b;
    return direction === 'asc' ? result : -result;
}

export function sortHoldingPositions(
    positions: FundPortfolioPosition[],
    key: FundHoldingSortKey,
    direction: FundHoldingSortDirection,
): FundPortfolioPosition[] {
    const sorted = [...positions];
    sorted.sort((a, b) => {
        const fallback = holdingSortLabel(a).localeCompare(holdingSortLabel(b), 'tr', { numeric: true });
        if (key === 'asset') {
            return direction === 'asc' ? fallback : -fallback;
        }
        const result = compareHoldingNumbers(holdingNumberValue(a, key), holdingNumberValue(b, key), direction);
        return result || fallback;
    });
    return sorted;
}

export function sortHoldingPositionsByWeight(
    positions: FundPortfolioPosition[],
    direction: FundHoldingSortDirection = 'desc',
): FundPortfolioPosition[] {
    return sortHoldingPositions(positions, 'weight', direction);
}

export function polarPoint(cx: number, cy: number, radius: number, angleDeg: number): { x: number; y: number } {
    const radians = ((angleDeg - 90) * Math.PI) / 180;
    return {
        x: cx + radius * Math.cos(radians),
        y: cy + radius * Math.sin(radians),
    };
}

export function donutSegmentPath(startAngle: number, endAngle: number, outerRadius = 92, innerRadius = 44): string {
    const cx = 100;
    const cy = 100;
    const safeEndAngle = endAngle - startAngle >= 359.99 ? startAngle + 359.99 : endAngle;
    const outerStart = polarPoint(cx, cy, outerRadius, startAngle);
    const outerEnd = polarPoint(cx, cy, outerRadius, safeEndAngle);
    const innerEnd = polarPoint(cx, cy, innerRadius, safeEndAngle);
    const innerStart = polarPoint(cx, cy, innerRadius, startAngle);
    const largeArc = safeEndAngle - startAngle > 180 ? 1 : 0;
    return [
        `M ${outerStart.x} ${outerStart.y}`,
        `A ${outerRadius} ${outerRadius} 0 ${largeArc} 1 ${outerEnd.x} ${outerEnd.y}`,
        `L ${innerEnd.x} ${innerEnd.y}`,
        `A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${innerStart.x} ${innerStart.y}`,
        'Z',
    ].join(' ');
}

export function chartDateMs(value: string): number {
    const ts = new Date(`${value}T00:00:00`).getTime();
    return Number.isFinite(ts) ? ts : 0;
}

export function nearestChartPoint<T extends { date: string }>(points: T[], targetDate: string): T | null {
    if (!points.length) return null;
    const targetTs = chartDateMs(targetDate);
    let closest = points[0];
    let closestDiff = Math.abs(chartDateMs(points[0].date) - targetTs);
    for (const point of points.slice(1)) {
        const diff = Math.abs(chartDateMs(point.date) - targetTs);
        if (diff < closestDiff) {
            closest = point;
            closestDiff = diff;
        }
    }
    return closest;
}

export function finiteNumber(value: number | null | undefined): number | null {
    return value == null || !Number.isFinite(value) ? null : value;
}

export function formatPlainPct(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return '-';
    return `% ${value.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function returnFromDatedValues(
    points: Array<{ date: string; value: number }> | undefined,
    period: ComparisonPeriod,
): number | null {
    const ordered = [...(points || [])]
        .filter((point) => point.date && Number.isFinite(Number(point.value)) && Number(point.value) > 0)
        .sort((a, b) => a.date.localeCompare(b.date));
    if (ordered.length < 2) return null;
    const latest = ordered[ordered.length - 1];
    const startDate = rangeStartDate(period, latest.date, '');
    let base: (typeof ordered)[number] | null = null;
    for (const point of ordered) {
        if (point.date > startDate) break;
        base = point;
    }
    if (!base) return null;
    return returnBetween(latest.value, base.value);
}

export function returnFromFundPoints(points: FundPricePoint[], period: ComparisonPeriod): number | null {
    return returnFromDatedValues(
        sortFundPoints(points).map((point) => ({ date: point.date, value: Number(point.price) })),
        period,
    );
}
