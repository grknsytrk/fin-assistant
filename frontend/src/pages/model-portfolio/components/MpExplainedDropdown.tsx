import { useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown } from 'lucide-react';
import type { ExplainedOption } from '../constants';

export function MpExplainedDropdown({
    label,
    value,
    options,
    onChange,
}: {
    label: string;
    value: string;
    options: ExplainedOption[];
    onChange: (nextVal: string) => void;
}) {
    const [open, setOpen] = useState(false);
    const wrapRef = useRef<HTMLDivElement | null>(null);

    const selected = useMemo(
        () => options.find((o) => o.value === value) || options[0],
        [options, value],
    );

    useEffect(() => {
        if (!open) return;
        const onDocClick = (e: MouseEvent) => {
            if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
                setOpen(false);
            }
        };
        const onKey = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setOpen(false);
        };
        document.addEventListener('mousedown', onDocClick);
        window.addEventListener('keydown', onKey);
        return () => {
            document.removeEventListener('mousedown', onDocClick);
            window.removeEventListener('keydown', onKey);
        };
    }, [open]);

    return (
        <div className="mp-field mp-explained-select-wrap" ref={wrapRef}>
            <span className="mp-field-label">{label}</span>
            <button
                type="button"
                className={`mp-explained-trigger${open ? ' is-open' : ''}`}
                aria-label={`${label}: ${selected.label}`}
                aria-expanded={open}
                onClick={() => setOpen((p) => !p)}
            >
                <div className="mp-explained-trigger-main">
                    <span className="mp-explained-trigger-title">{selected.label}</span>
                </div>
                <ChevronDown size={15} className={`mp-explained-chevron${open ? ' is-open' : ''}`} />
            </button>

            {open && (
                <div className="mp-explained-menu" role="listbox">
                    {options.map((opt) => {
                        const isSelected = opt.value === selected.value;
                        return (
                            <button
                                key={opt.value}
                                type="button"
                                role="option"
                                aria-selected={isSelected}
                                title={opt.description}
                                className={`mp-explained-option${isSelected ? ' is-selected' : ''}`}
                                onClick={() => {
                                    onChange(opt.value);
                                    setOpen(false);
                                }}
                            >
                                <span className="mp-explained-option-label">{opt.label}</span>
                                <span className="mp-explained-check">
                                    {isSelected && <Check size={14} />}
                                </span>
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
