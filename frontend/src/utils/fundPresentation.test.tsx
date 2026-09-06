import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import {
    canonicalFundPrice,
    formatFundQuotePrice,
    formatFundReportDate,
    hasFundRangeStartCoverage,
} from './fundPresentation';

describe('fund presentation helpers', () => {
    it('uses the canonical snapshot price and its currency', () => {
        expect(canonicalFundPrice(12.345678)).toBe(12.345678);
        expect(canonicalFundPrice(0)).toBeNull();
        expect(formatFundQuotePrice(12.345678, 'USD')).toMatch(/^USD /);
    });

    it('renders date-only values as reporting dates without UTC rollover', () => {
        render(<output>{formatFundReportDate('2026-09-06')}</output>);

        expect(screen.getByText(/6 Eyl 2026/i)).toBeTruthy();
    });

    it('rejects comparison ranges with a material history gap', () => {
        expect(hasFundRangeStartCoverage('2025-09-06', '2025-09-10')).toBe(true);
        expect(hasFundRangeStartCoverage('2025-09-06', '2025-09-20')).toBe(false);
    });
});
