import { useEffect, useRef } from 'react';

export function useAccessibleDialog<T extends HTMLElement>(active: boolean, onClose: () => void) {
    const dialogRef = useRef<T | null>(null);
    const onCloseRef = useRef(onClose);
    onCloseRef.current = onClose;

    useEffect(() => {
        if (!active) return;
        const dialog = dialogRef.current;
        if (!dialog) return;

        const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
        const previousBodyOverflow = document.body.style.overflow;
        const focusableSelector = [
            'a[href]',
            'button:not([disabled])',
            'input:not([disabled])',
            'select:not([disabled])',
            'textarea:not([disabled])',
            '[tabindex]:not([tabindex="-1"])',
        ].join(',');
        const getFocusable = () => Array.from(dialog.querySelectorAll<HTMLElement>(focusableSelector));

        document.body.style.overflow = 'hidden';
        (dialog.querySelector<HTMLElement>('[data-dialog-autofocus]') || getFocusable()[0] || dialog).focus();

        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                event.preventDefault();
                onCloseRef.current();
                return;
            }
            if (event.key !== 'Tab') return;

            const focusable = getFocusable();
            if (!focusable.length) {
                event.preventDefault();
                dialog.focus();
                return;
            }

            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.body.style.overflow = previousBodyOverflow;
            if (previousFocus?.isConnected) previousFocus.focus();
            else document.querySelector<HTMLElement>('[data-dialog-return-focus]')?.focus();
        };
    }, [active]);

    return dialogRef;
}
