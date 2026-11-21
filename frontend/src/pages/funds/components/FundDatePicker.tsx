import { useEffect, useMemo, useRef, useState } from 'react';
import { Calendar, ChevronLeft, ChevronRight } from 'lucide-react';
import './FundDatePicker.css';

const MONTH_NAMES = [
    'Ocak',
    'Şubat',
    'Mart',
    'Nisan',
    'Mayıs',
    'Haziran',
    'Temmuz',
    'Ağustos',
    'Eylül',
    'Ekim',
    'Kasım',
    'Aralık',
] as const;

const WEEKDAY_LABELS = ['Pt', 'Sa', 'Ça', 'Pe', 'Cu', 'Ct', 'Pa'] as const;

function parseIsoParts(value: string | undefined): { year: number; month: number; day: number } | null {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value || '').trim());
    if (!match) return null;
    const year = Number(match[1]);
    const month = Number(match[2]) - 1;
    const day = Number(match[3]);
    if (month < 0 || month > 11 || day < 1 || day > 31) return null;
    return { year, month, day };
}

function toIsoDate(year: number, month: number, day: number): string {
    const d = new Date(Date.UTC(year, month, day));
    return d.toISOString().slice(0, 10);
}

function formatDisplayDate(value: string): string {
    const parts = parseIsoParts(value);
    if (!parts) return value || '--.--.----';
    return `${String(parts.day).padStart(2, '0')}.${String(parts.month + 1).padStart(2, '0')}.${parts.year}`;
}

export interface FundDatePickerProps {
    label: string;
    value: string;
    onChange: (nextDate: string) => void;
    min?: string;
    max?: string;
    rangeStart?: string;
    rangeEnd?: string;
    align?: 'left' | 'right';
}

export function FundDatePicker({
    label,
    value,
    onChange,
    min = '2014-01-01',
    max,
    rangeStart,
    rangeEnd,
    align = 'left',
}: FundDatePickerProps) {
    const todayIso = useMemo(() => new Date().toISOString().slice(0, 10), []);
    const effectiveMax = max && /^\d{4}-\d{2}-\d{2}$/.test(max) ? max : todayIso;
    const [open, setOpen] = useState(false);
    const wrapRef = useRef<HTMLDivElement | null>(null);

    const initialParts = parseIsoParts(value) || parseIsoParts(effectiveMax) || parseIsoParts(todayIso)!;
    const [viewYear, setViewYear] = useState(initialParts.year);
    const [viewMonth, setViewMonth] = useState(initialParts.month);

    useEffect(() => {
        if (!open) return;
        const parsed = parseIsoParts(value) || parseIsoParts(effectiveMax) || parseIsoParts(todayIso);
        if (parsed) {
            setViewYear(parsed.year);
            setViewMonth(parsed.month);
        }
    }, [effectiveMax, open, todayIso, value]);

    useEffect(() => {
        if (!open) return;
        const onDocClick = (event: MouseEvent) => {
            if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) {
                setOpen(false);
            }
        };
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                setOpen(false);
            }
        };
        document.addEventListener('mousedown', onDocClick);
        window.addEventListener('keydown', onKeyDown);
        return () => {
            document.removeEventListener('mousedown', onDocClick);
            window.removeEventListener('keydown', onKeyDown);
        };
    }, [open]);

    const minYear = parseIsoParts(min)?.year ?? 2014;
    const maxYear = parseIsoParts(effectiveMax)?.year ?? initialParts.year;
    const yearOptions = useMemo(() => {
        const list: number[] = [];
        for (let y = Math.min(minYear, viewYear); y <= Math.max(maxYear, viewYear); y += 1) {
            list.push(y);
        }
        return list;
    }, [maxYear, minYear, viewYear]);

    const calendarCells = useMemo(() => {
        const firstDayUtc = new Date(Date.UTC(viewYear, viewMonth, 1));
        const mondayOffset = (firstDayUtc.getUTCDay() + 6) % 7;
        const gridStartUtcMs = Date.UTC(viewYear, viewMonth, 1 - mondayOffset);
        const cells: Array<{
            iso: string;
            day: number;
            outsideMonth: boolean;
            disabled: boolean;
            isToday: boolean;
            isSelected: boolean;
            inRange: boolean;
        }> = [];
        for (let i = 0; i < 42; i += 1) {
            const cellDate = new Date(gridStartUtcMs + i * 86_400_000);
            const cellYear = cellDate.getUTCFullYear();
            const cellMonth = cellDate.getUTCMonth();
            const cellDay = cellDate.getUTCDate();
            const iso = toIsoDate(cellYear, cellMonth, cellDay);
            const outsideMonth = cellYear !== viewYear || cellMonth !== viewMonth;
            const disabled = Boolean((min && iso < min) || (effectiveMax && iso > effectiveMax));
            const isSelected = iso === value;
            const inRange = Boolean(
                rangeStart
                && rangeEnd
                && rangeStart <= rangeEnd
                && iso > rangeStart
                && iso < rangeEnd,
            );
            cells.push({
                iso,
                day: cellDay,
                outsideMonth,
                disabled,
                isToday: iso === todayIso,
                isSelected,
                inRange,
            });
        }
        return cells;
    }, [effectiveMax, min, rangeEnd, rangeStart, todayIso, value, viewMonth, viewYear]);

    const stepMonth = (delta: number) => {
        const next = new Date(Date.UTC(viewYear, viewMonth + delta, 1));
        setViewYear(next.getUTCFullYear());
        setViewMonth(next.getUTCMonth());
    };

    const selectDate = (iso: string) => {
        onChange(iso);
        setOpen(false);
    };

    const clampedToday = effectiveMax && todayIso > effectiveMax ? effectiveMax : todayIso;

    return (
        <div className="fund-datepicker" ref={wrapRef}>
            <button
                type="button"
                className={`fund-datepicker-trigger${open ? ' is-open' : ''}`}
                aria-label={`${label}: ${formatDisplayDate(value)}`}
                aria-expanded={open}
                onClick={() => setOpen((prev) => !prev)}
            >
                <Calendar size={13} aria-hidden="true" className="fund-datepicker-icon" />
                <span className="fund-datepicker-label">{label}</span>
                <span className="fund-datepicker-value">{formatDisplayDate(value)}</span>
            </button>

            {open && (
                <div
                    className={`fund-datepicker-popover${align === 'right' ? ' is-align-right' : ''}`}
                    role="dialog"
                    aria-label={`${label} takvim seçici`}
                >
                    <div className="fund-datepicker-header">
                        <button
                            type="button"
                            className="fund-datepicker-nav"
                            onClick={() => stepMonth(-1)}
                            aria-label="Önceki ay"
                        >
                            <ChevronLeft size={14} />
                        </button>
                        <div className="fund-datepicker-selects">
                            <select
                                aria-label="Ay seç"
                                value={viewMonth}
                                onChange={(event) => setViewMonth(Number(event.target.value))}
                            >
                                {MONTH_NAMES.map((name, idx) => (
                                    <option key={name} value={idx}>{name}</option>
                                ))}
                            </select>
                            <select
                                aria-label="Yıl seç"
                                value={viewYear}
                                onChange={(event) => setViewYear(Number(event.target.value))}
                            >
                                {yearOptions.map((y) => (
                                    <option key={y} value={y}>{y}</option>
                                ))}
                            </select>
                        </div>
                        <button
                            type="button"
                            className="fund-datepicker-nav"
                            onClick={() => stepMonth(1)}
                            aria-label="Sonraki ay"
                        >
                            <ChevronRight size={14} />
                        </button>
                    </div>

                    <div className="fund-datepicker-weekdays" aria-hidden="true">
                        {WEEKDAY_LABELS.map((dayName) => (
                            <span key={dayName}>{dayName}</span>
                        ))}
                    </div>

                    <div className="fund-datepicker-grid">
                        {calendarCells.map((cell) => (
                            <button
                                key={cell.iso}
                                type="button"
                                disabled={cell.disabled}
                                className={[
                                    'fund-datepicker-day',
                                    cell.outsideMonth ? 'is-outside' : '',
                                    cell.isToday ? 'is-today' : '',
                                    cell.inRange ? 'is-in-range' : '',
                                    cell.isSelected ? 'is-selected' : '',
                                ].filter(Boolean).join(' ')}
                                onClick={() => selectDate(cell.iso)}
                                aria-pressed={cell.isSelected}
                                title={formatDisplayDate(cell.iso)}
                            >
                                {cell.day}
                            </button>
                        ))}
                    </div>

                    <div className="fund-datepicker-footer">
                        <button
                            type="button"
                            className="fund-datepicker-foot-btn"
                            onClick={() => selectDate(clampedToday)}
                        >
                            Bugün
                        </button>
                        <button
                            type="button"
                            className="fund-datepicker-foot-btn"
                            onClick={() => setOpen(false)}
                        >
                            Kapat
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}

export default FundDatePicker;
