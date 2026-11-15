import type {
    MarketStockRow,
    MarketStocksResponse,
    ModelPortfolioPerformancePoint,
} from '../../api/types';
import type { PerfRange } from './constants';

export function quoteTimestampAtLeastAsRecent(
    candidate: string | null | undefined,
    current: string | null | undefined,
): boolean {
    if (!candidate) return false;
    if (!current) return true;
    const candidateTime = Date.parse(candidate);
    const currentTime = Date.parse(current);
    if (Number.isFinite(candidateTime) && Number.isFinite(currentTime)) {
        return candidateTime >= currentTime;
    }
    return candidate >= current;
}

export function mergeLiveMarketRows(
    current: Map<string, MarketStockRow>,
    payload: MarketStocksResponse,
): Map<string, MarketStockRow> {
    const next = new Map(current);
    for (const row of payload.rows || []) {
        const symbol = String(row.symbol || row.company || '').trim().toUpperCase();
        if (!symbol || row.price == null || row.price <= 0) continue;
        const candidate = { ...row, price_as_of: row.price_as_of || payload.as_of };
        const existing = next.get(symbol);
        if (!existing || quoteTimestampAtLeastAsRecent(candidate.price_as_of, existing.price_as_of)) {
            next.set(symbol, candidate);
        }
    }
    return next;
}

export function formatPrice(val: number | null | undefined): string {
    if (val == null || !Number.isFinite(val)) return '—';
    return `${val.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ₺`;
}

export function formatPct(val: number | null | undefined, signed = true): string {
    if (val == null || !Number.isFinite(val)) return '—';
    const sign = signed && val > 0 ? '+' : '';
    return `%${sign}${val.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function formatMultiple(val: number | null | undefined): string {
    if (val == null || !Number.isFinite(val)) return '—';
    return `${val.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}x`;
}

export function formatDateTR(iso: string | null | undefined): string {
    if (!iso) return '—';
    const d = new Date(iso.length === 10 ? `${iso}T00:00:00` : iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleDateString('tr-TR', { day: '2-digit', month: 'short', year: 'numeric' });
}

export function formatDateTimeTR(iso: string | null | undefined): string {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString('tr-TR', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

export function formatCompactDateTR(iso: string | null | undefined): string {
    if (!iso) return '—';
    const d = new Date(iso.length === 10 ? `${iso}T00:00:00` : iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit', year: '2-digit' });
}

export function formatCompactDateTimeTR(iso: string | null | undefined): string {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const time = d.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
    return `${formatCompactDateTR(iso)} ${time}`;
}

export function toneClass(val: number | null | undefined): string {
    if (val == null || !Number.isFinite(val) || val === 0) return 'mp-tone-flat';
    return val > 0 ? 'mp-tone-up' : 'mp-tone-down';
}

export function compoundAlpha(
    assetRet: number | null | undefined,
    benchRet: number | null | undefined,
): number | null {
    if (assetRet == null || benchRet == null || !Number.isFinite(assetRet) || !Number.isFinite(benchRet)) {
        return null;
    }
    const denom = 1 + benchRet / 100;
    if (denom <= 0) return null;
    return Number((((1 + assetRet / 100) / denom - 1) * 100).toFixed(2));
}

export function computeProgress(entry: number, current: number | null, target: number): number {
    if (current == null || !Number.isFinite(current)) return 0;
    const span = target - entry;
    if (Math.abs(span) < 1e-6) return current >= target ? 100 : 0;
    return Math.max(-100, Math.min(150, Number((((current - entry) / span) * 100).toFixed(1))));
}

export function todayIsoDate(): string {
    const d = new Date();
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

export function isWeekdayIsoDate(iso: string): boolean {
    const d = new Date(`${iso.slice(0, 10)}T00:00:00Z`);
    if (Number.isNaN(d.getTime())) return true;
    const day = d.getUTCDay();
    return day !== 0 && day !== 6;
}

export function sliceAndNormalizePerformance(
    points: ModelPortfolioPerformancePoint[],
    range: PerfRange,
    inceptionDate: string | null,
): ModelPortfolioPerformancePoint[] {
    const weekdayPoints = points.filter((p) => isWeekdayIsoDate(p.date));
    if (!weekdayPoints.length) return [];
    const lastDateStr = weekdayPoints[weekdayPoints.length - 1].date;
    const lastDate = new Date(`${lastDateStr}T00:00:00Z`);
    let cutoffIso = weekdayPoints[0].date;

    if (range === '1w') {
        const d = new Date(lastDate);
        d.setUTCDate(d.getUTCDate() - 7);
        cutoffIso = d.toISOString().slice(0, 10);
    } else if (range === '1m') {
        const d = new Date(lastDate);
        d.setUTCDate(d.getUTCDate() - 30);
        cutoffIso = d.toISOString().slice(0, 10);
    } else if (range === '3m') {
        const d = new Date(lastDate);
        d.setUTCDate(d.getUTCDate() - 91);
        cutoffIso = d.toISOString().slice(0, 10);
    } else if (range === '6m') {
        const d = new Date(lastDate);
        d.setUTCDate(d.getUTCDate() - 182);
        cutoffIso = d.toISOString().slice(0, 10);
    } else if (range === 'ytd') {
        cutoffIso = `${lastDate.getUTCFullYear()}-01-01`;
    } else if (range === '1y') {
        const d = new Date(lastDate);
        d.setUTCDate(d.getUTCDate() - 365);
        cutoffIso = d.toISOString().slice(0, 10);
    } else if (range === 'all' && inceptionDate) {
        cutoffIso = inceptionDate;
    }

    if (inceptionDate && inceptionDate > cutoffIso) cutoffIso = inceptionDate;
    const working = weekdayPoints.filter((p) => p.date >= cutoffIso);
    if (!working.length) return [];
    const basePort = working[0].portfolio_value || 100;
    const baseXu = working[0].xu100_value || 100;
    const baseXu030 = working[0].xu030_value || 100;

    return working.map((pt) => {
        const normPort = Number(((pt.portfolio_value / basePort) * 100).toFixed(2));
        const normXu = Number(((pt.xu100_value / baseXu) * 100).toFixed(2));
        const normXu030 = Number(((pt.xu030_value / baseXu030) * 100).toFixed(2));
        const portRet = Number((normPort - 100).toFixed(2));
        const xuRet = Number((normXu - 100).toFixed(2));
        const xu030Ret = Number((normXu030 - 100).toFixed(2));
        const relRet = compoundAlpha(portRet, xuRet) ?? 0;
        const relXu030Ret = compoundAlpha(portRet, xu030Ret) ?? 0;
        return {
            date: pt.date,
            portfolio_value: normPort,
            xu100_value: normXu,
            xu030_value: normXu030,
            portfolio_return_pct: portRet,
            xu100_return_pct: xuRet,
            xu030_return_pct: xu030Ret,
            relative_return_pct: relRet,
            relative_xu030_return_pct: relXu030Ret,
        };
    });
}

/** Picks a 1/2/5 × 10^n step so axis ticks land on round numbers. */
export function niceAxisStep(rawStep: number): number {
    const exp = Math.floor(Math.log10(rawStep));
    const base = 10 ** exp;
    const frac = rawStep / base;
    const nice = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10;
    return nice * base;
}

/** Monotone cubic (Fritsch–Carlson) path: smooth, but never overshoots the data points. */
export function smoothLinePath(coords: Array<[number, number]>): string {
    const n = coords.length;
    if (n === 0) return '';
    const fmt = (v: number) => v.toFixed(1);
    if (n < 3) return coords.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${fmt(x)},${fmt(y)}`).join(' ');

    const dx: number[] = [];
    const slope: number[] = [];
    for (let i = 0; i < n - 1; i++) {
        dx.push(coords[i + 1][0] - coords[i][0]);
        slope.push((coords[i + 1][1] - coords[i][1]) / (dx[i] || 1));
    }
    const tangent: number[] = [slope[0]];
    for (let i = 1; i < n - 1; i++) {
        tangent.push(slope[i - 1] * slope[i] <= 0 ? 0 : (slope[i - 1] + slope[i]) / 2);
    }
    tangent.push(slope[n - 2]);
    for (let i = 0; i < n - 1; i++) {
        if (slope[i] === 0) {
            tangent[i] = 0;
            tangent[i + 1] = 0;
            continue;
        }
        const a = tangent[i] / slope[i];
        const b = tangent[i + 1] / slope[i];
        const h = Math.hypot(a, b);
        if (h > 3) {
            tangent[i] = (3 * a * slope[i]) / h;
            tangent[i + 1] = (3 * b * slope[i]) / h;
        }
    }

    let d = `M${fmt(coords[0][0])},${fmt(coords[0][1])}`;
    for (let i = 0; i < n - 1; i++) {
        const [x0, y0] = coords[i];
        const [x1, y1] = coords[i + 1];
        const c1x = x0 + dx[i] / 3;
        const c2x = x1 - dx[i] / 3;
        d += ` C${fmt(c1x)},${fmt(y0 + (tangent[i] * dx[i]) / 3)} ${fmt(c2x)},${fmt(y1 - (tangent[i + 1] * dx[i]) / 3)} ${fmt(x1)},${fmt(y1)}`;
    }
    return d;
}
