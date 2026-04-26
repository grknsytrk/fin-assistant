import { useEffect, useMemo, useState } from 'react';
import {
    FX_COUNTRY_FLAGS,
    type SymbolLogoKind,
    localIconForSymbol,
    logoDevDomainUrl,
    normalizeLogoSymbol,
    stockLogoDevDomains,
    tradingViewCountryFlagUrl,
    tradingViewLogoUrl,
} from './symbolLogoMaps';
import './SymbolLogo.css';

export type { SymbolLogoKind };

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

    if (kind === 'stock') {
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

        const tradingViewUrl = tradingViewLogoUrl(normalized);
        if (tradingViewUrl) {
            candidates.push(tradingViewUrl);
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

export default function SymbolLogo({
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
    const candidates = useMemo(
        () => buildCandidates({ symbol: normalizedSymbol, name, kind, logoUrl }),
        [normalizedSymbol, name, kind, logoUrl],
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
                        loading="lazy"
                        onError={() => {
                            setFxFlagFailed(true);
                        }}
                    />
                ) : null}
                <img
                    src={tradingViewCountryFlagUrl(fxFlags.base)}
                    alt={`${fxFlags.base} flag`}
                    className="symbol-logo-fx-flag symbol-logo-fx-flag-front"
                    loading="lazy"
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
                    loading="lazy"
                    onError={() => {
                        setCandidateIndex((prev) => prev + 1);
                    }}
                />
            ) : (
                <span className="symbol-logo-fallback">{fallbackText}</span>
            )}
        </span>
    );
}
