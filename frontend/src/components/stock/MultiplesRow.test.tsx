import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { MultiplesRow } from './MultiplesRow';
import type { KapQuarter, KapSnapshotResponse } from '../../api/types';
const rows = [1, 2, 3, 4].map(period => ({ year: 2025, period, metrics: {
    net_borc: { value: 2 }, ozkaynaklar: { value: 10 }, donen_varliklar: { value: 8 }, kisa_vadeli_yukumlulukler: { value: 2 },
}, metrics_quarterly: { net_kar: { value: 1 }, favok: { value: 2 }, satis_gelirleri: { value: 4 } } })) as unknown as KapQuarter[];
describe('valuation presentation', () => {
    it.each(['bank', 'insurance'] as const)('hides industrial ratios for %s even when provided', company_kind => {
        const snapshot = { company_kind, valuation: { ttm_net_kar: 4, ttm_favok: 8, fd_favok: 3 } } as KapSnapshotResponse;
        const html = renderToStaticMarkup(<MultiplesRow snapshot={snapshot} quarters={rows} />);
        expect(html).not.toContain('FD/FAVÖK');
        expect(html).not.toContain('Cari Oran');
        expect(html).not.toContain('Borç/Özkaynak');
        expect(html).toContain('ROE');
    });
    it('does not replace a backend null TTM with a local sum', () => {
        const snapshot = { company_kind: 'generic', valuation: { ttm_net_kar: null, ttm_favok: null } } as KapSnapshotResponse;
        const html = renderToStaticMarkup(<MultiplesRow snapshot={snapshot} quarters={rows} />);
        expect(html).toContain('eksik dönem');
        expect(html).not.toContain('Net Kâr Marjı');
    });
    it('uses the stock card multiples and includes net debt to EBITDA', () => {
        const snapshot = { company_kind: 'generic', valuation: { fk: 99, pd_dd: 99, fd_favok: 99 } } as KapSnapshotResponse;
        const html = renderToStaticMarkup(
            <MultiplesRow
                snapshot={snapshot}
                quarters={rows}
                marketCard={{ fk: 11.1, pd_dd: 0.7, fd_favok: 2.11, net_borc_favok: 1 }}
            />,
        );
        expect(html).toContain('11.10x');
        expect(html).toContain('0.70x');
        expect(html).toContain('2.11x');
        expect(html).toContain('1.00x');
        expect(html).not.toContain('99.00x');
    });
});
