import { memo, useEffect, useMemo, useState } from 'react';
import {
    FX_COUNTRY_FLAGS,
    explicitLogoUrlsForSymbol,
    fintablesFundManagerLogoUrls,
    type SymbolLogoKind,
    fintablesLogoUrlsForSymbol,
    localIconForSymbol,
    logoDevDomainUrl,
    normalizeLogoSymbol,
    stockLogoDevDomains,
    tradingViewCountryFlagUrl,
    tradingViewLogoUrl,
} from './symbolLogoMaps';
import './SymbolLogo.css';

export type { SymbolLogoKind };

const LOGO_SUCCESS_CACHE_KEY = 'ragfin.logo.success.v4';
const LOGO_FAILED_CACHE_KEY = 'ragfin.logo.failed.v4';
const LOGO_SUCCESS_TTL_MS = 24 * 60 * 60 * 1000;
const LOGO_FAILED_TTL_MS = 60 * 60 * 1000;
const logoSuccessMemory = new Map<string, { url: string; ts: number }>();
const logoFailedMemory = new Map<string, number>();
let logoSuccessSessionLoaded = false;
let logoFailedSessionLoaded = false;

export type SymbolLogoProps = {
    symbol: string;
    name?: string;
    kind: SymbolLogoKind;
    logoUrl?: string | null;
    size?: 'xs' | 'sm' | 'md' | 'lg';
    className?: string;
};

function monogramForSymbol(symbol: string): string {
    const normalized = normalizeLogoSymbol(symbol).replace(/[^A-Z0-9]/g, '');
    if (!normalized) return '?';
    return normalized.slice(0, 2);
}

function nowMs(): number {
    return Date.now();
}

function readSessionObject<T>(key: string): Record<string, T> {
    if (typeof window === 'undefined') return {};
    try {
        const raw = window.sessionStorage.getItem(key);
        if (!raw) return {};
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === 'object' ? parsed : {};
    } catch {
        return {};
    }
}

function writeSessionObject<T>(key: string, payload: Record<string, T>): void {
    if (typeof window === 'undefined') return;
    try {
        window.sessionStorage.setItem(key, JSON.stringify(payload));
    } catch {
        // Session storage can be unavailable in privacy modes; memory cache still applies.
    }
}

function logoCacheKey(kind: SymbolLogoKind, symbol: string, name?: string, logoUrl?: string | null): string {
    if (kind === 'fund') {
        // Fund logos are derived from the manager name / logoUrl, so keep those in the key.
        return `${kind}:${String(logoUrl || name || '').trim()}`;
    }
    // For everything else (stocks, indices, fx) the symbol uniquely identifies the logo,
    // regardless of whether the surrounding label is "Hisse" or the full company name.
    return `${kind}:${normalizeLogoSymbol(symbol)}`;
}

function ensureLogoSuccessCacheLoaded(): void {
    if (logoSuccessSessionLoaded) return;
    const stored = readSessionObject<{ url: string; ts: number }>(LOGO_SUCCESS_CACHE_KEY);
    Object.entries(stored).forEach(([key, value]) => {
        if (value?.url && value?.ts) {
            logoSuccessMemory.set(key, value);
        }
    });
    logoSuccessSessionLoaded = true;
}

function ensureLogoFailedCacheLoaded(): void {
    if (logoFailedSessionLoaded) return;
    const stored = readSessionObject<number>(LOGO_FAILED_CACHE_KEY);
    Object.entries(stored).forEach(([url, ts]) => {
        if (ts) {
            logoFailedMemory.set(url, ts);
        }
    });
    logoFailedSessionLoaded = true;
}

function persistLogoSuccessCache(): void {
    writeSessionObject(LOGO_SUCCESS_CACHE_KEY, Object.fromEntries(logoSuccessMemory));
}

function persistLogoFailedCache(): void {
    writeSessionObject(LOGO_FAILED_CACHE_KEY, Object.fromEntries(logoFailedMemory));
}

function cachedSuccessUrl(key: string, candidates: string[]): string | null {
    ensureLogoSuccessCacheLoaded();
    const current = logoSuccessMemory.get(key);
    const now = nowMs();
    if (current && now - current.ts < LOGO_SUCCESS_TTL_MS && candidates.includes(current.url)) {
        return current.url;
    }
    return null;
}

function isFailedLogoUrl(url: string): boolean {
    ensureLogoFailedCacheLoaded();
    const failedAt = logoFailedMemory.get(url);
    const now = nowMs();
    if (failedAt && now - failedAt < LOGO_FAILED_TTL_MS) return true;
    return false;
}

function markLogoSuccess(key: string, url: string): void {
    const now = nowMs();
    const existing = logoSuccessMemory.get(key);
    if (existing?.url === url && now - existing.ts < LOGO_SUCCESS_TTL_MS) return;
    const item = { url, ts: now };
    logoSuccessMemory.set(key, item);
    persistLogoSuccessCache();
}

function markLogoFailed(key: string, url: string): void {
    const ts = nowMs();
    const existing = logoFailedMemory.get(url);
    if (existing && ts - existing < LOGO_FAILED_TTL_MS) return;
    logoFailedMemory.set(url, ts);
    persistLogoFailedCache();

    const success = logoSuccessMemory.get(key);
    if (success?.url === url) {
        logoSuccessMemory.delete(key);
        persistLogoSuccessCache();
    }
}

function cacheAwareCandidates(key: string, candidates: string[]): string[] {
    const success = cachedSuccessUrl(key, candidates);
    const remaining = candidates.filter((item) => item === success || !isFailedLogoUrl(item));
    if (remaining.length > 0) return remaining;
    // All candidates are currently marked as failed; fall back to the original list so
    // a transient network failure doesn't permanently lock the symbol to its monogram.
    return candidates;
}

function buildCandidates({
    symbol,
    name,
    kind,
    logoUrl,
}: {
    symbol: string;
    name?: string;
    kind: SymbolLogoKind;
    logoUrl?: string | null;
}): string[] {
    const normalized = normalizeLogoSymbol(symbol);
    const candidates: string[] = [];

    if (kind === 'fund') {
        candidates.push(...fintablesFundManagerLogoUrls(logoUrl || name));
        return candidates;
    }

    if (kind === 'stock') {
        candidates.push(...fintablesLogoUrlsForSymbol(normalized));
        candidates.push(...explicitLogoUrlsForSymbol(normalized));

        const mappedTradingViewUrl = tradingViewLogoUrl(normalized, { force: true, requireMappedSlug: true });
        if (mappedTradingViewUrl) {
            candidates.push(mappedTradingViewUrl);
        }

        const tradingViewUrl = tradingViewLogoUrl(normalized, { force: true });
        if (tradingViewUrl) {
            candidates.push(tradingViewUrl);
        }

        const domains = stockLogoDevDomains(normalized, name);
        for (const domain of domains) {
            const logoDevUrl = logoDevDomainUrl(domain);
            if (logoDevUrl) {
                candidates.push(logoDevUrl);
            }
        }

        if (logoUrl) {
            candidates.push(logoUrl);
        }
    } else {
        const tradingViewUrl = tradingViewLogoUrl(normalized, { force: true, requireMappedSlug: true });
        if (tradingViewUrl) {
            candidates.push(tradingViewUrl);
        }

        const localIcon = localIconForSymbol(normalized);
        if (localIcon) {
            candidates.push(localIcon);
        }
    }

    return Array.from(new Set(candidates.filter((item) => Boolean(item))));
}

function SymbolLogo({
    symbol,
    name,
    kind,
    logoUrl,
    size = 'sm',
    className,
}: SymbolLogoProps) {
    const normalizedSymbol = normalizeLogoSymbol(symbol);
    const fxFlags = kind === 'fx' ? FX_COUNTRY_FLAGS[normalizedSymbol] : undefined;
    const fallbackText = monogramForSymbol(normalizedSymbol);
    const rawCandidates = useMemo(
        () => buildCandidates({ symbol: normalizedSymbol, name, kind, logoUrl }),
        [normalizedSymbol, name, kind, logoUrl],
    );
    const cacheKey = useMemo(
        () => logoCacheKey(kind, normalizedSymbol, name, logoUrl),
        [kind, normalizedSymbol, name, logoUrl],
    );
    const candidates = useMemo(
        () => cacheAwareCandidates(cacheKey, rawCandidates),
        [cacheKey, rawCandidates],
    );
    const candidateKey = candidates.join('|');
    const [candidateIndex, setCandidateIndex] = useState(0);
    const [fxFlagFailed, setFxFlagFailed] = useState(false);

    useEffect(() => {
        setCandidateIndex(0);
    }, [candidateKey]);

    useEffect(() => {
        setFxFlagFailed(false);
    }, [normalizedSymbol, kind, size]);

    const currentSrc = candidates[candidateIndex] || null;
    const rootClassName = [
        'symbol-logo',
        `symbol-logo-${size}`,
        className || '',
    ]
        .filter(Boolean)
        .join(' ');

    if (kind === 'fx' && fxFlags && !fxFlagFailed) {
        return (
            <span
                className={`${rootClassName} symbol-logo-fx-flags`}
                data-kind={kind}
                aria-hidden="true"
            >
                {fxFlags.quote ? (
                    <img
                        src={tradingViewCountryFlagUrl(fxFlags.quote)}
                        alt={`${fxFlags.quote} flag`}
                        className="symbol-logo-fx-flag symbol-logo-fx-flag-back"
                        decoding="async"
                        onError={() => {
                            setFxFlagFailed(true);
                        }}
                    />
                ) : null}
                <img
                    src={tradingViewCountryFlagUrl(fxFlags.base)}
                    alt={`${fxFlags.base} flag`}
                    className="symbol-logo-fx-flag symbol-logo-fx-flag-front"
                    decoding="async"
                    onError={() => {
                        setFxFlagFailed(true);
                    }}
                />
            </span>
        );
    }

    return (
        <span
            className={rootClassName}
            data-kind={kind}
            aria-hidden="true"
        >
            {currentSrc ? (
                <img
                    src={currentSrc}
                    alt={name || normalizedSymbol}
                    className="symbol-logo-image"
                    decoding="async"
                    onLoad={() => {
                        markLogoSuccess(cacheKey, currentSrc);
                    }}
                    onError={() => {
                        markLogoFailed(cacheKey, currentSrc);
                        setCandidateIndex((prev) => prev + 1);
                    }}
                />
            ) : (
                <span className="symbol-logo-fallback">{fallbackText}</span>
            )}
        </span>
    );
}

export default memo(SymbolLogo);
