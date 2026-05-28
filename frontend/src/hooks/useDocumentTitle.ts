import { useEffect } from 'react';

export const APP_TITLE = 'RAG-FIN';

type TitlePart = string | number | null | undefined | false;

function cleanTitlePart(value: TitlePart): string {
    return String(value ?? '').replace(/\s+/g, ' ').trim();
}

export function buildDocumentTitle(...parts: TitlePart[]): string {
    const titleParts = parts.map(cleanTitlePart).filter(Boolean);
    return titleParts.length ? `${titleParts.join(' · ')} | ${APP_TITLE}` : APP_TITLE;
}

export function useDocumentTitle(title: string | null | undefined, enabled = true): void {
    const nextTitle = cleanTitlePart(title) || APP_TITLE;

    useEffect(() => {
        if (!enabled || typeof document === 'undefined') return;
        document.title = nextTitle;
    }, [enabled, nextTitle]);
}

export function formatTitleNumber(
    value: number | null | undefined,
    options: Intl.NumberFormatOptions = { minimumFractionDigits: 2, maximumFractionDigits: 2 },
): string | null {
    if (value == null || !Number.isFinite(value)) return null;
    return value.toLocaleString('tr-TR', options);
}

export function formatTitleCurrency(
    value: number | null | undefined,
    currency?: string | null,
    options: Intl.NumberFormatOptions = { minimumFractionDigits: 2, maximumFractionDigits: 2 },
): string | null {
    const formatted = formatTitleNumber(value, options);
    if (!formatted) return null;
    const currencyLabel = !currency || currency === 'TRY' || currency === 'TL' ? '₺' : currency;
    return `${currencyLabel}${formatted}`;
}

export function formatTitlePct(value: number | null | undefined): string | null {
    if (value == null || !Number.isFinite(value)) return null;
    const sign = value > 0 ? '+' : '';
    return `%${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

