import type { MouseEvent as ReactMouseEvent } from 'react';
import SymbolLogo from './SymbolLogo';
import './AssetSearchResult.css';

export type AssetSearchResultKind = 'stock' | 'fund';

interface AssetSearchResultProps {
    id?: string;
    href: string;
    kind: AssetSearchResultKind;
    symbol: string;
    name: string;
    price: number | null;
    changePct: number | null;
    logoUrl?: string | null;
    logoName?: string | null;
    active?: boolean;
    onMouseDown?: (event: ReactMouseEvent<HTMLAnchorElement>) => void;
    onMouseEnter?: () => void;
    onClick?: (event: ReactMouseEvent<HTMLAnchorElement>) => void;
}

function formatSearchPrice(value: number | null, kind: AssetSearchResultKind): string | null {
    if (value == null || !Number.isFinite(value)) return null;
    return value.toLocaleString('tr-TR', {
        minimumFractionDigits: kind === 'fund' ? 4 : 2,
        maximumFractionDigits: kind === 'fund' ? 6 : 2,
    });
}

function formatSearchChange(value: number | null): string | null {
    if (value == null || !Number.isFinite(value)) return null;
    const sign = value > 0 ? '+' : '';
    return `% ${sign}${value.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export default function AssetSearchResult({
    id,
    href,
    kind,
    symbol,
    name,
    price,
    changePct,
    logoUrl,
    logoName,
    active = false,
    onMouseDown,
    onMouseEnter,
    onClick,
}: AssetSearchResultProps) {
    const formattedPrice = formatSearchPrice(price, kind);
    const formattedChange = formatSearchChange(changePct);

    return (
        <a
            id={id}
            href={href}
            className={`asset-search-result${active ? ' active' : ''}${formattedPrice || formattedChange ? ' has-values' : ''}`}
            role="option"
            aria-selected={active}
            onMouseDown={onMouseDown}
            onMouseEnter={onMouseEnter}
            onClick={onClick}
        >
            <SymbolLogo
                symbol={symbol}
                name={logoName || name}
                kind={kind}
                logoUrl={logoUrl}
                size="md"
            />
            <span className="asset-search-result-main">
                <strong>{symbol}</strong>
                <span>{name}</span>
            </span>
            <span className="asset-search-result-values">
                {formattedPrice ? (
                    <span className="asset-search-result-price">
                        {kind === 'stock' ? <small>G</small> : null}
                        {formattedPrice}
                    </span>
                ) : null}
                {formattedChange ? (
                    <span className={`asset-search-result-change ${changePct! < 0 ? 'down' : changePct! > 0 ? 'up' : 'flat'}`}>
                        {formattedChange}
                    </span>
                ) : null}
            </span>
            <span className={`asset-search-result-kind ${kind}`}>
                {kind === 'fund' ? 'Fon' : 'Hisse'}
            </span>
        </a>
    );
}
