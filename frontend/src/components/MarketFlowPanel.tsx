import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ExternalLink, RefreshCw, SlidersHorizontal } from 'lucide-react';
import { apiClient } from '../api/client';
import type { MarketFlowItem } from '../api/types';

type FlowFilter = 'all' | 'ozel_durum' | 'finansal_rapor' | 'kar_payi' | 'genel_kurul' | 'diger';

const FLOW_FILTERS: Array<{ value: FlowFilter; label: string }> = [
    { value: 'all', label: 'Tümü' },
    { value: 'ozel_durum', label: 'Özel Durum' },
    { value: 'finansal_rapor', label: 'Finansal Rapor' },
    { value: 'kar_payi', label: 'Kâr Payı' },
    { value: 'genel_kurul', label: 'Genel Kurul' },
    { value: 'diger', label: 'Diğer' },
];

const FLOW_SIZE_OPTIONS = [25, 50, 100];
const FLOW_SIZE_STORAGE_KEY = 'ragfin.flow.size';
const FLOW_SIZE_DEFAULT = 50;

function readInitialFlowSize(): number {
    if (typeof window === 'undefined') return FLOW_SIZE_DEFAULT;
    try {
        const saved = Number.parseInt(window.localStorage.getItem(FLOW_SIZE_STORAGE_KEY) || '', 10);
        return FLOW_SIZE_OPTIONS.includes(saved) ? saved : FLOW_SIZE_DEFAULT;
    } catch {
        return FLOW_SIZE_DEFAULT;
    }
}

function matchesFlowFilter(item: MarketFlowItem, filter: FlowFilter): boolean {
    if (filter === 'all') return true;
    if (filter === 'ozel_durum') return item.category === 'ozel_durum' || item.source === 'Özel Durum';
    if (filter === 'finansal_rapor') return item.category === 'finansal_rapor';
    if (filter === 'kar_payi') return item.category === 'kar_payi';
    if (filter === 'genel_kurul') return item.category === 'genel_kurul';
    return !['ozel_durum', 'finansal_rapor', 'kar_payi', 'genel_kurul'].includes(item.category);
}

function formatFlowTime(iso: string): string {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return '--:--';
    return date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
}

function formatFlowDate(iso: string): string {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return '';
    return date.toLocaleDateString('tr-TR', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
    });
}

function formatFlowCodes(item: MarketFlowItem): string {
    const codes = item.stock_codes?.filter(Boolean) || [];
    if (codes.length === 0) return item.symbol || 'KAP';
    if (codes.length <= 2) return codes.join(' ');
    return `${codes.slice(0, 2).join(' ')} +${codes.length - 2} şirket`;
}

function flowFilterLabel(filter: FlowFilter): string {
    return FLOW_FILTERS.find((option) => option.value === filter)?.label || 'Tümü';
}

export default function MarketFlowPanel({
    onSelectTicker,
}: {
    onSelectTicker?: (ticker: string) => void;
}) {
    const [items, setItems] = useState<MarketFlowItem[] | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [filter, setFilter] = useState<FlowFilter>('all');
    const [size, setSize] = useState(readInitialFlowSize);
    const sizeRef = useRef(size);
    const [showOptions, setShowOptions] = useState(false);
    const [warning, setWarning] = useState<string | null>(null);

    const load = useCallback((refresh = false, requestedSize?: number) => {
        const requestSize = requestedSize ?? sizeRef.current;
        setLoading(true);
        setError(null);
        apiClient
            .marketFlow(requestSize, undefined, { refresh })
            .then((payload) => {
                setItems(payload.items || []);
                setWarning(payload.warning || null);
            })
            .catch((requestError: unknown) => {
                setError(requestError instanceof Error ? requestError.message : 'Akış verisi alınamadı.');
            })
            .finally(() => setLoading(false));
    }, []);

    useEffect(() => {
        load();
    }, [load]);

    useEffect(() => {
        const timer = window.setInterval(() => {
            if (document.visibilityState === 'visible') load(true);
        }, 30000);
        return () => window.clearInterval(timer);
    }, [load]);

    const filteredItems = useMemo(
        () => (items || []).filter((item) => matchesFlowFilter(item, filter)),
        [filter, items],
    );

    const handleSizeChange = (next: number) => {
        sizeRef.current = next;
        setSize(next);
        try {
            window.localStorage.setItem(FLOW_SIZE_STORAGE_KEY, String(next));
        } catch {
            // localStorage unavailable; the current selection still works.
        }
        load(false, next);
    };

    const handleItemClick = (item: MarketFlowItem) => {
        if (item.kap_url) {
            window.open(item.kap_url, '_blank', 'noopener,noreferrer');
            return;
        }
        if (item.symbol) onSelectTicker?.(item.symbol);
    };

    return (
        <div className="mwr-flow-panel">
            <div className="mwr-flow-toolbar">
                <label className="mwr-flow-select-wrap">
                    <span className="sr-only">Akış filtresi</span>
                    <select
                        className="mwr-flow-select"
                        value={filter}
                        onChange={(event) => setFilter(event.target.value as FlowFilter)}
                        aria-label="Akış filtresi"
                    >
                        {FLOW_FILTERS.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                    </select>
                </label>
                <button
                    type="button"
                    className={`mwr-flow-options-button${showOptions ? ' is-active' : ''}`}
                    onClick={() => setShowOptions((current) => !current)}
                    aria-expanded={showOptions}
                    aria-label="Akış seçenekleri"
                    title="Akış seçenekleri"
                >
                    <SlidersHorizontal size={16} aria-hidden="true" />
                </button>
                <button
                    type="button"
                    className="mwr-flow-refresh-button"
                    onClick={() => load(true)}
                    disabled={loading}
                    aria-label="Akışı yenile"
                    title="Akışı yenile"
                >
                    <RefreshCw size={15} className={loading ? 'is-spinning' : ''} aria-hidden="true" />
                </button>
            </div>

            {showOptions && (
                <div className="mwr-flow-options" role="group" aria-label="Akış kayıt sayısı">
                    <span>Gösterilecek kayıt</span>
                    <div className="mwr-flow-size-options">
                        {FLOW_SIZE_OPTIONS.map((option) => (
                            <button
                                key={option}
                                type="button"
                                className={size === option ? 'is-active' : ''}
                                onClick={() => handleSizeChange(option)}
                                disabled={loading && size === option}
                            >
                                {option}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            <div className="mwr-flow-status-row">
                <span>Resmi KAP akışı</span>
                <span>{flowFilterLabel(filter)}</span>
            </div>

            {warning && <div className="mwr-flow-warning" role="status">{warning}</div>}
            {loading && !items && <div className="mwr-flow-state">Akış yükleniyor…</div>}
            {error && <div className="mwr-flow-state mwr-flow-error">{error}</div>}
            {!loading && !error && items && filteredItems.length === 0 && (
                <div className="mwr-flow-state">Bu filtreye uygun bildirim yok.</div>
            )}

            <div className="mwr-flow-list">
                {filteredItems.map((item) => (
                    <button
                        key={item.id}
                        type="button"
                        className="mwr-flow-item"
                        onClick={() => handleItemClick(item)}
                        title={`${item.title} · ${formatFlowDate(item.published_at)}`}
                    >
                        <div className="mwr-flow-item-meta">
                            <span className="mwr-flow-source">KAP</span>
                            <span className="mwr-flow-dot" aria-hidden="true">·</span>
                            <span className="mwr-flow-codes">{formatFlowCodes(item)}</span>
                            <time dateTime={item.published_at}>{formatFlowTime(item.published_at)}</time>
                        </div>
                        <div className="mwr-flow-item-title">{item.title}</div>
                        {item.source && item.source !== 'KAP' && (
                            <div className="mwr-flow-item-type">
                                <span>{item.source}</span>
                                <ExternalLink size={12} aria-hidden="true" />
                            </div>
                        )}
                    </button>
                ))}
            </div>
        </div>
    );
}
