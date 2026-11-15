import { memo, useEffect, useRef, useState } from 'react';
import type { MouseEvent as ReactMouseEvent } from 'react';
import type { MarketStockRow, MarketStocksResponse } from '../../../api/types';
import { toStockDetail } from '../../../routing/routes';
import type { StockReturnMode } from '../../../routing/routes';
import SymbolLogo from '../../../components/SymbolLogo';
import {
    RETURN_KEYS,
    formatFundamentalPercent,
    formatFundamentalRatio,
    formatMarketCap,
    formatStockPrice,
    formatTablePct,
    formatVolume,
    getReturnValue,
    getTableChangeClass,
} from '../stockScreenerConfig';
import type { StockSectorDiffFields, StockTableMode } from '../types';

export interface StockTableRowProps {
    row: MarketStockRow;
    rank: number;
    mode: StockTableMode;
    returnMode: StockReturnMode;
    benchmarks?: MarketStocksResponse['benchmarks'];
    sectorDiff?: StockSectorDiffFields;
    onCompanyClick: (symbol: string) => void;
}

export const StockTableRow = memo(function StockTableRow({
    row,
    rank,
    mode,
    returnMode,
    benchmarks,
    sectorDiff,
    onCompanyClick,
}: StockTableRowProps) {
    const prevPriceRef = useRef(row.price);
    const [flashClass, setFlashClass] = useState('');

    useEffect(() => {
        if (row.price != null && prevPriceRef.current != null && row.price !== prevPriceRef.current) {
            setFlashClass(row.price > prevPriceRef.current ? 'stocks-flash-up' : 'stocks-flash-down');
            const timer = window.setTimeout(() => setFlashClass(''), 1100);
            prevPriceRef.current = row.price;
            return () => window.clearTimeout(timer);
        }
        prevPriceRef.current = row.price;
    }, [row.price]);

    const symbol = row.symbol || row.company;

    return (
        <tr
            aria-rowindex={rank + 1}
            className={flashClass}
            onClick={(event: ReactMouseEvent<HTMLTableRowElement>) => {
                if (event.button !== 0) return;
                onCompanyClick(row.company);
            }}
            onAuxClick={(event: ReactMouseEvent<HTMLTableRowElement>) => {
                if (event.button !== 1) return;
                if ((event.target as Element).closest('a[href]')) return;
                event.preventDefault();
                window.open(toStockDetail(symbol), '_blank', 'noopener,noreferrer');
            }}
        >
            <td className="stocks-rank">{rank}</td>
            <td className="stocks-symbol-cell">
                <a
                    className="stocks-symbol-link"
                    href={toStockDetail(row.company)}
                    onClick={(event) => {
                        event.stopPropagation();
                        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                        event.preventDefault();
                        onCompanyClick(row.company);
                    }}
                >
                    <span className="stocks-symbol-main">
                        <SymbolLogo
                            symbol={row.company}
                            name={row.company}
                            kind="stock"
                            logoUrl={row.logo_url}
                            size="xs"
                            className="stocks-symbol-logo"
                        />
                        <span>{row.company}</span>
                    </span>
                </a>
            </td>
            {mode === 'fundamentals' ? (
                <>
                    <td title={row.sector || undefined}>{row.sector || '—'}</td>
                    <td>{row.fundamental_period || row.latest_quarter || '—'}</td>
                    <td className="stocks-cell-right stocks-price-cell">{formatStockPrice(row)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.market_cap)}</td>
                    <td className="stocks-cell-right">{formatFundamentalRatio(row.fk)}</td>
                    <td className="stocks-cell-right">{formatFundamentalRatio(row.pd_dd)}</td>
                    <td className="stocks-cell-right">{formatFundamentalRatio(row.fd_favok)}</td>
                    <td className="stocks-cell-right">{formatFundamentalRatio(row.fd_satis)}</td>
                    <td className="stocks-cell-right" title={(row.hist_fk_quarters ?? row.hist_quarters) != null ? `Son ${row.hist_fk_quarters ?? row.hist_quarters} çeyrek KAP ortalaması` : undefined}>
                        {formatFundamentalRatio(row.hist_fk_avg)}
                        {row.hist_fk_avg != null && (row.hist_fk_quarters ?? row.hist_quarters) != null ? (
                            <span className="stocks-cell-muted"> ({row.hist_fk_quarters ?? row.hist_quarters}Ç)</span>
                        ) : null}
                    </td>
                    <td className={`stocks-cell-right ${row.fk_hist_diff_pct == null ? 'stocks-cell-muted' : row.fk_hist_diff_pct < 0 ? 'stocks-up' : row.fk_hist_diff_pct > 0 ? 'stocks-down' : 'stocks-flat'}`}>
                        {formatTablePct(row.fk_hist_diff_pct ?? null)}
                    </td>
                    <td className="stocks-cell-right" title={(row.hist_pd_dd_quarters ?? row.hist_quarters) != null ? `Son ${row.hist_pd_dd_quarters ?? row.hist_quarters} çeyrek KAP ortalaması` : undefined}>
                        {formatFundamentalRatio(row.hist_pd_dd_avg)}
                        {row.hist_pd_dd_avg != null && (row.hist_pd_dd_quarters ?? row.hist_quarters) != null ? (
                            <span className="stocks-cell-muted"> ({row.hist_pd_dd_quarters ?? row.hist_quarters}Ç)</span>
                        ) : null}
                    </td>
                    <td className={`stocks-cell-right ${row.pd_dd_hist_diff_pct == null ? 'stocks-cell-muted' : row.pd_dd_hist_diff_pct < 0 ? 'stocks-up' : row.pd_dd_hist_diff_pct > 0 ? 'stocks-down' : 'stocks-flat'}`}>
                        {formatTablePct(row.pd_dd_hist_diff_pct ?? null)}
                    </td>
                    <td className="stocks-cell-right" title={(row.hist_fd_favok_quarters ?? row.hist_quarters) != null ? `Son ${row.hist_fd_favok_quarters ?? row.hist_quarters} çeyrek KAP ortalaması` : undefined}>
                        {formatFundamentalRatio(row.hist_fd_favok_avg)}
                        {row.hist_fd_favok_avg != null && (row.hist_fd_favok_quarters ?? row.hist_quarters) != null ? (
                            <span className="stocks-cell-muted"> ({row.hist_fd_favok_quarters ?? row.hist_quarters}Ç)</span>
                        ) : null}
                    </td>
                    <td className={`stocks-cell-right ${sectorDiff?.fk_sector_diff_pct == null ? 'stocks-cell-muted' : sectorDiff.fk_sector_diff_pct < 0 ? 'stocks-up' : sectorDiff.fk_sector_diff_pct > 0 ? 'stocks-down' : 'stocks-flat'}`}>
                        {formatTablePct(sectorDiff?.fk_sector_diff_pct ?? null)}
                    </td>
                    <td className={`stocks-cell-right ${row.sma_50_diff_pct == null ? 'stocks-cell-muted' : getTableChangeClass(row.sma_50_diff_pct)}`}>
                        {formatTablePct(row.sma_50_diff_pct ?? null)}
                    </td>
                    <td className={`stocks-cell-right ${row.sma_200_diff_pct == null ? 'stocks-cell-muted' : getTableChangeClass(row.sma_200_diff_pct)}`}>
                        {formatTablePct(row.sma_200_diff_pct ?? null)}
                    </td>
                    <td className={`stocks-cell-right ${row.high_52w_dist_pct == null ? 'stocks-cell-muted' : getTableChangeClass(row.high_52w_dist_pct)}`}>
                        {formatTablePct(row.high_52w_dist_pct ?? null)}
                    </td>
                    <td className="stocks-cell-right">{formatFundamentalRatio(row.net_borc_favok)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.ttm_net_kar ?? null)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.ttm_fcf ?? null)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.ttm_satis_gelirleri ?? null)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.ttm_favok ?? null)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.net_borc ?? null)}</td>
                    <td className="stocks-cell-right">{formatMarketCap(row.ozkaynaklar ?? null)}</td>
                    <td className="stocks-cell-right">{formatFundamentalPercent(row.roe_pct)}</td>
                    <td className="stocks-cell-right">{formatFundamentalPercent(row.free_float_pct)}</td>
                </>
            ) : (
                <>
                    <td className="stocks-cell-right stocks-price-cell">{formatStockPrice(row)}</td>
                    <td className={`stocks-cell-right ${row.change_pct == null ? 'stocks-cell-muted' : getTableChangeClass(row.change_pct)}`}>
                        {formatTablePct(row.change_pct)}
                    </td>
                    <td className="stocks-cell-right stocks-volume-cell">{formatVolume(row.volume)}</td>
                    <td className="stocks-cell-right stocks-market-cap-cell">
                        {formatMarketCap(row.market_cap)}
                    </td>
                    {RETURN_KEYS.map((k) => {
                        const val = getReturnValue(row, k, benchmarks, returnMode);
                        const isNull = val == null || !Number.isFinite(val);
                        return (
                            <td
                                key={k}
                                className={`stocks-cell-right ${isNull ? 'stocks-cell-muted' : getTableChangeClass(val)}`}
                            >
                                {formatTablePct(val)}
                            </td>
                        );
                    })}
                </>
            )}
        </tr>
    );
});
