import { memo, useCallback, useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';
import { Search, X } from 'lucide-react';
import type { FundSummary } from '../../../api/types';
import SymbolLogo from '../../../components/SymbolLogo';
import { useTableVirtualizer } from '../../../hooks/useTableVirtualizer';
import { toFundDetail, type FundTab } from '../../../routing/routes';
import {
    formatCompactCurrency,
    formatCurrency,
    formatDate,
    formatPct,
    pctClass,
} from '../formatters';
import {
    FUND_TABLE_ROW_HEIGHT,
    type FundSortKey,
    type FundTableMode,
} from '../types';
import { useDebouncedValue } from '../useDebouncedValue';
import './FundsTable.css';

export const FundsSearchInput = memo(function FundsSearchInput({
    value,
    onChange,
}: {
    value: string;
    onChange: (value: string) => void;
}) {
    const [draft, setDraft] = useState(value);
    const debouncedDraft = useDebouncedValue(draft, 160);

    useEffect(() => {
        setDraft(value);
    }, [value]);

    useEffect(() => {
        if (debouncedDraft !== value) onChange(debouncedDraft);
    }, [debouncedDraft, onChange, value]);

    return (
        <div className="funds-search-wrap">
            <Search size={14} className="funds-search-icon" aria-hidden="true" />
            <input
                type="text"
                className="funds-search-input"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="Fon kodu, ad veya kurucu ara..."
            />
            {draft && (
                <button
                    type="button"
                    className="funds-search-clear"
                    onClick={() => {
                        setDraft('');
                        onChange('');
                    }}
                    aria-label="Aramayı temizle"
                >
                    <X size={12} aria-hidden="true" />
                </button>
            )}
        </div>
    );
});

export type FundsTableProps = {
    rows: FundSummary[];
    mode: FundTableMode;
    sortKey: FundSortKey;
    sortOrder: 'asc' | 'desc';
    onSort: (key: FundSortKey) => void;
    onOpenFund: (fundCode: string, tab?: FundTab) => void;
    resetDeps?: unknown[];
};

export const FundTableRow = memo(function FundTableRow({
    row,
    rank,
    rowIndex,
    mode,
    onOpenFund,
}: {
    row: FundSummary;
    rank: number;
    rowIndex: number;
    mode: FundTableMode;
    onOpenFund: (fundCode: string, tab?: FundTab) => void;
}) {
    const handleOpen = useCallback(() => {
        onOpenFund(row.fund_code, 'overview');
    }, [onOpenFund, row.fund_code]);

    const renderPctCell = (val: number | null | undefined) => (
        <td className={`funds-cell-right ${val == null || !Number.isFinite(val) ? 'funds-cell-muted' : pctClass(val)}`}>
            {formatPct(val)}
        </td>
    );

    return (
        <tr
            aria-rowindex={rowIndex}
            onClick={(event: ReactMouseEvent<HTMLTableRowElement>) => {
                if (event.button !== 0) return;
                handleOpen();
            }}
            onAuxClick={(event: ReactMouseEvent<HTMLTableRowElement>) => {
                if (event.button !== 1) return;
                if ((event.target as Element).closest('a[href]')) return;
                event.preventDefault();
                window.open(toFundDetail(row.fund_code, 'overview'), '_blank', 'noopener,noreferrer');
            }}
        >
            <td className="funds-rank">{rank}</td>
            <td className="funds-symbol-cell">
                <a
                    className="funds-symbol-link"
                    href={toFundDetail(row.fund_code, 'overview')}
                    onClick={(event) => {
                        event.stopPropagation();
                        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                        event.preventDefault();
                        handleOpen();
                    }}
                >
                    <div className="funds-symbol">
                        <SymbolLogo
                            symbol={row.fund_code}
                            name={row.founder_company || row.manager_company || row.name}
                            kind="fund"
                            size="xs"
                            className="funds-symbol-logo"
                        />
                        <span>
                            <strong>{row.fund_code}</strong>
                            <small title={row.name}>{row.name}</small>
                        </span>
                    </div>
                </a>
            </td>
            {mode === 'performance' ? (
                <>
                    <td className={`funds-cell-right ${row.price == null ? 'funds-cell-muted' : 'funds-price-cell'}`}>
                        {formatCurrency(row.price, row.currency)}
                    </td>
                    {renderPctCell(row.daily_return)}
                    {renderPctCell(row.period_returns?.['1w'])}
                    {renderPctCell(row.period_returns?.['1m'])}
                    {renderPctCell(row.period_returns?.['3m'])}
                    {renderPctCell(row.period_returns?.['6m'])}
                    {renderPctCell(row.period_returns?.ytd)}
                    {renderPctCell(row.period_returns?.['1y'])}
                    <td className={`funds-cell-right ${row.aum == null ? 'funds-cell-muted' : 'funds-aum-cell'}`}>
                        {formatCompactCurrency(row.aum)}
                    </td>
                </>
            ) : (
                <>
                    <td className={`funds-cell-right ${row.price == null ? 'funds-cell-muted' : 'funds-price-cell'}`}>
                        {formatCurrency(row.price, row.currency)}
                    </td>
                    {renderPctCell(row.daily_return)}
                    <td className={row.fund_type ? 'funds-text-muted' : 'funds-cell-muted'}>{row.fund_type || '—'}</td>
                    <td className={row.founder_company || row.manager_company ? 'funds-text-muted' : 'funds-cell-muted'} title={row.founder_company || row.manager_company || undefined}>
                        {row.founder_company || row.manager_company || '—'}
                    </td>
                    <td className={`funds-cell-right ${row.risk_value == null ? 'funds-cell-muted' : ''}`}>
                        {row.risk_value ?? '—'}
                    </td>
                    <td className={`funds-cell-right ${row.aum == null ? 'funds-cell-muted' : 'funds-aum-cell'}`}>
                        {formatCompactCurrency(row.aum)}
                    </td>
                    <td className={`funds-cell-right ${row.investor_count == null ? 'funds-cell-muted' : 'funds-aum-cell'}`}>
                        {row.investor_count != null ? row.investor_count.toLocaleString('tr-TR') : '—'}
                    </td>
                    <td className={`funds-cell-right ${row.as_of ? 'funds-aum-cell' : 'funds-cell-muted'}`}>
                        {formatDate(row.as_of)}
                    </td>
                </>
            )}
        </tr>
    );
});

export const FundsTable = memo(function FundsTable({
    rows,
    mode,
    sortKey,
    sortOrder,
    onSort,
    onOpenFund,
    resetDeps,
}: FundsTableProps) {
    const tableWrapRef = useRef<HTMLDivElement>(null);
    const {
        startIndex,
        endIndex,
        topSpacerHeight,
        bottomSpacerHeight,
        containerRef: tableWrapCallbackRef,
    } = useTableVirtualizer({
        totalRows: rows.length,
        rowHeight: FUND_TABLE_ROW_HEIGHT,
        overscan: 12,
        containerRef: tableWrapRef,
        resetDeps: resetDeps ?? [mode, sortKey, sortOrder],
    });

    const visibleRows = useMemo(() => {
        return rows.slice(startIndex, endIndex);
    }, [rows, startIndex, endIndex]);

    const sortLabel = sortOrder === 'asc' ? 'artan' : 'azalan';
    const renderSortHeader = (key: FundSortKey, label: string, align: 'left' | 'right' = 'left') => {
        const isActive = sortKey === key;
        return (
            <th className={align === 'right' ? 'funds-cell-right' : undefined} aria-sort={isActive ? (sortOrder === 'asc' ? 'ascending' : 'descending') : 'none'}>
                <button
                    type="button"
                    className={`funds-th-sort${align === 'right' ? ' funds-th-sort-right' : ''}${isActive ? ' active' : ''}`}
                    onClick={() => onSort(key)}
                    title={`${label} sütununa göre sırala`}
                >
                    <span>{label}</span>
                    {isActive && (
                        <span className="funds-sort-indicator" aria-hidden="true">
                            {sortOrder === 'asc' ? '↑' : '↓'}
                        </span>
                    )}
                    <span className="sr-only">{isActive ? `Şu an ${sortLabel}` : ''}</span>
                </button>
            </th>
        );
    };

    const columnCount = mode === 'performance' ? 11 : 10;

    return (
        <div className="funds-table-wrap" ref={tableWrapCallbackRef}>
            <table className={`funds-table${mode === 'performance' ? ' funds-performance-table' : ''}`} aria-rowcount={rows.length + 1}>
                <thead>
                    <tr>
                        <th className="funds-rank">#</th>
                        {renderSortHeader('fund_code', 'Fon')}
                        {mode === 'performance' ? (
                            <>
                                {renderSortHeader('price', 'Fiyat', 'right')}
                                {renderSortHeader('daily_return', 'Gün %', 'right')}
                                {renderSortHeader('return_1w', '1 Hafta', 'right')}
                                {renderSortHeader('return_1m', '1 Ay', 'right')}
                                {renderSortHeader('return_3m', '3 Ay', 'right')}
                                {renderSortHeader('return_6m', '6 Ay', 'right')}
                                {renderSortHeader('return_ytd', 'Yılbaşı', 'right')}
                                {renderSortHeader('return_1y', '1 Yıl', 'right')}
                                {renderSortHeader('aum', 'Portföy', 'right')}
                            </>
                        ) : (
                            <>
                                {renderSortHeader('price', 'Fiyat', 'right')}
                                {renderSortHeader('daily_return', 'Gün %', 'right')}
                                {renderSortHeader('fund_type', 'Tür')}
                                {renderSortHeader('founder_company', 'Kurucu')}
                                {renderSortHeader('risk_value', 'Risk', 'right')}
                                {renderSortHeader('aum', 'Portföy', 'right')}
                                {renderSortHeader('investor_count', 'Yatırımcı', 'right')}
                                {renderSortHeader('as_of', 'Tarih', 'right')}
                            </>
                        )}
                    </tr>
                </thead>
                <tbody>
                    {topSpacerHeight > 0 && (
                        <tr key="funds-top-spacer" className="funds-spacer-row" style={{ height: `${topSpacerHeight}px` }} aria-hidden="true">
                            <td colSpan={columnCount} style={{ height: `${topSpacerHeight}px` }} />
                        </tr>
                    )}
                    {visibleRows.map((row, index) => (
                        <FundTableRow
                            key={row.fund_code}
                            row={row}
                            rank={startIndex + index + 1}
                            rowIndex={startIndex + index + 2}
                            mode={mode}
                            onOpenFund={onOpenFund}
                        />
                    ))}
                    {bottomSpacerHeight > 0 && (
                        <tr key="funds-bottom-spacer" className="funds-spacer-row" style={{ height: `${bottomSpacerHeight}px` }} aria-hidden="true">
                            <td colSpan={columnCount} style={{ height: `${bottomSpacerHeight}px` }} />
                        </tr>
                    )}
                </tbody>
            </table>
        </div>
    );
});
