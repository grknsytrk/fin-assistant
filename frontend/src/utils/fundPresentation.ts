/** Pure fund quote helpers shared by the detail view and its tests. */

export function formatFundReportDate(value: string | null | undefined): string {
    if (!value) return '-';
    const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
    if (dateOnly) {
        const [, year, month, day] = dateOnly;
        return new Date(Number(year), Number(month) - 1, Number(day)).toLocaleDateString('tr-TR', {
            day: '2-digit', month: 'short', year: 'numeric',
        });
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleDateString('tr-TR', { day: '2-digit', month: 'short', year: 'numeric' });
}

export function formatFundQuotePrice(value: number | null | undefined, currency = 'TRY'): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const prefix = currency === 'TRY' ? '₺' : `${currency} `;
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 4,
        maximumFractionDigits: 6,
    })}`;
}

export function canonicalFundPrice(value: number | null | undefined): number | null {
    return value != null && Number.isFinite(value) && value > 0 ? value : null;
}

export function hasFundRangeStartCoverage(startIso: string, actualStartIso: string, maximumGapDays = 7): boolean {
    const requested = Date.parse(`${startIso}T00:00:00Z`);
    const actual = Date.parse(`${actualStartIso}T00:00:00Z`);
    return Number.isFinite(requested)
        && Number.isFinite(actual)
        && actual - requested <= maximumGapDays * 24 * 60 * 60 * 1000;
}
