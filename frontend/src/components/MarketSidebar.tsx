import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChevronRight, ChevronLeft, Activity } from 'lucide-react';
import { apiClient } from '../api/client';
import type {
    CommodityQuote,
    FxQuote,
    MarketFlowItem,
    MarketUniverseRow,
} from '../api/types';
import './MarketSidebar.css';

type TabId = 'XU100' | 'XU030' | 'ENDEKSLER' | 'DOVIZ' | 'EMTIA';

type PanelMode = 'markets' | 'flow';

const PANEL_MODES: { id: PanelMode; label: string }[] = [
    { id: 'markets', label: 'Piyasalar' },
    { id: 'flow', label: 'Akış' },
];

const TABS: { id: TabId; label: string }[] = [
    { id: 'XU100', label: 'XU100' },
    { id: 'XU030', label: 'XU030' },
    { id: 'ENDEKSLER', label: 'Endeksler' },
    { id: 'DOVIZ', label: 'Döviz' },
    { id: 'EMTIA', label: 'Emtia' },
];

type FlowFilter = 'all' | 'ozel_durum' | 'finansal_rapor' | 'kar_payi' | 'genel_kurul' | 'diger';

const FLOW_FILTERS: { id: FlowFilter; label: string }[] = [
    { id: 'all', label: 'Tümü' },
    { id: 'ozel_durum', label: 'Özel Durum' },
    { id: 'finansal_rapor', label: 'Finansal' },
    { id: 'kar_payi', label: 'Kâr Payı' },
    { id: 'genel_kurul', label: 'Genel Kurul' },
    { id: 'diger', label: 'Diğer' },
];

const FLOW_SIZE_OPTIONS: Array<{ value: number; label: string }> = [
    { value: 25, label: '25' },
    { value: 50, label: '50' },
    { value: 100, label: '100' },
    { value: 500, label: 'Tümü' },
];
const FLOW_SIZE_STORAGE_KEY = 'ragfin.flow.size';
const FLOW_SIZE_DEFAULT = 50;

function readInitialFlowSize(): number {
    if (typeof window === 'undefined') return FLOW_SIZE_DEFAULT;
    try {
        const raw = window.localStorage.getItem(FLOW_SIZE_STORAGE_KEY);
        const parsed = raw ? Number.parseInt(raw, 10) : NaN;
        if (FLOW_SIZE_OPTIONS.some((option) => option.value === parsed)) {
            return parsed;
        }
    } catch {
        // localStorage yasak/erisilemez; sessizce varsayilana don.
    }
    return FLOW_SIZE_DEFAULT;
}

function matchesFlowFilter(item: MarketFlowItem, filter: FlowFilter): boolean {
    if (filter === 'all') return true;
    if (filter === 'ozel_durum') return item.category === 'ozel_durum' || item.source === 'Özel Durum';
    if (filter === 'finansal_rapor') return item.category === 'finansal_rapor';
    if (filter === 'kar_payi') return item.category === 'kar_payi';
    if (filter === 'genel_kurul') return item.category === 'genel_kurul';
    if (filter === 'diger') {
        return ![
            'finansal_rapor',
            'kar_payi',
            'genel_kurul',
            'ozel_durum',
        ].includes(item.category);
    }
    return true;
}

function formatPrice(value: number | null, currency?: string | null, decimals = 2): string {
    if (value == null) return '-';
    let prefix = '';
    if (currency === 'TRY') prefix = '₺';
    else if (currency === 'USD') prefix = '$';
    else if (currency === 'EUR') prefix = '€';
    else if (currency) prefix = `${currency} `;
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    })}`;
}

function pickDecimals(value: number | null): number {
    if (value == null) return 2;
    const abs = Math.abs(value);
    if (abs < 10) return 4;
    if (abs < 100) return 3;
    return 2;
}

function FlashRow({
    onClick,
    price,
    className = "",
    title,
    children
}: React.PropsWithChildren<{ onClick?: () => void; price: number | null | undefined; className?: string; title?: string; }>) {
    const prevRef = useRef(price);
    const [flash, setFlash] = useState('');

    useEffect(() => {
        if (price != null && prevRef.current != null && price !== prevRef.current) {
            setFlash(price > prevRef.current ? 'ms-flash-up' : 'ms-flash-down');
            const t = setTimeout(() => setFlash(''), 1000);
            prevRef.current = price;
            return () => clearTimeout(t);
        }
        prevRef.current = price;
    }, [price]);

    return onClick ? (
        <button type="button" className={`ms-row ${className} ${flash}`} onClick={onClick} title={title}>
            {children}
        </button>
    ) : (
        <div className={`ms-row ${className} ${flash}`} title={title}>
            {children}
        </div>
    );
}

function formatPct(value: number | null): string {
    if (value == null) return '-';
    const sign = value > 0 ? '+' : '';
    return `%${sign}${value.toFixed(2)}`;
}

function changeClass(value: number | null): string {
    if (value == null || value === 0) return 'ms-change-flat';
    return value > 0 ? 'ms-change-up' : 'ms-change-down';
}

function formatTimeHm(iso: string): string {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    const hh = d.getHours().toString().padStart(2, '0');
    const mm = d.getMinutes().toString().padStart(2, '0');
    return `${hh}:${mm}`;
}

function startOfDay(d: Date): number {
    const c = new Date(d);
    c.setHours(0, 0, 0, 0);
    return c.getTime();
}

function flowGroupLabel(iso: string): { key: string; label: string; order: number } {
    const fallback = { key: 'unknown', label: 'Tarihsiz', order: -Infinity };
    if (!iso) return fallback;
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return fallback;
    const today = startOfDay(new Date());
    const itemDay = startOfDay(d);
    const diffDays = Math.round((today - itemDay) / 86400000);
    if (diffDays === 0) return { key: 'today', label: 'Bugün', order: today };
    if (diffDays === 1) return { key: 'yesterday', label: 'Dün', order: today - 86400000 };
    const label = d.toLocaleDateString('tr-TR', { day: '2-digit', month: 'long' });
    return { key: `d-${itemDay}`, label, order: itemDay };
}

export default function MarketSidebar({
    rows,
    onSelectTicker,
}: {
    rows: MarketUniverseRow[];
    onSelectTicker: (ticker: string) => void;
}) {
    const [open, setOpen] = useState(false);
    const [panelMode, setPanelMode] = useState<PanelMode>('markets');
    const [tab, setTab] = useState<TabId>('XU100');

    const [flow, setFlow] = useState<MarketFlowItem[] | null>(null);
    const [flowLoading, setFlowLoading] = useState(false);
    const [flowError, setFlowError] = useState<string | null>(null);
    const [flowFilter, setFlowFilter] = useState<FlowFilter>('all');
    const [flowDegraded, setFlowDegraded] = useState<boolean>(false);
    const [flowWarning, setFlowWarning] = useState<string | null>(null);
    const [flowSize, setFlowSize] = useState<number>(() => readInitialFlowSize());

    const [xu030, setXu030] = useState<MarketUniverseRow[] | null>(null);
    const [xu030Loading, setXu030Loading] = useState(false);
    const [xu030Error, setXu030Error] = useState<string | null>(null);

    const [commodities, setCommodities] = useState<CommodityQuote[] | null>(null);
    const [commoditiesLoading, setCommoditiesLoading] = useState(false);
    const [commoditiesError, setCommoditiesError] = useState<string | null>(null);
    const [commoditiesDelayNote, setCommoditiesDelayNote] = useState<string>('');

    const [fx, setFx] = useState<FxQuote[] | null>(null);
    const [fxLoading, setFxLoading] = useState(false);
    const [fxError, setFxError] = useState<string | null>(null);
    const [fxDelayNote, setFxDelayNote] = useState<string>('');

    const [indicesData, setIndicesData] = useState<any[] | null>(null);
    const [indicesLoading, setIndicesLoading] = useState(false);
    const [indicesError, setIndicesError] = useState<string | null>(null);

    const flowInFlight = useRef(false);
    const xu030InFlight = useRef(false);
    const commoditiesInFlight = useRef(false);
    const fxInFlight = useRef(false);
    const indicesInFlight = useRef(false);

    const loadFlow = useCallback((options?: { refresh?: boolean; size?: number }) => {
        if (flowInFlight.current) return;
        flowInFlight.current = true;
        setFlowLoading(true);
        setFlowError(null);
        const size = options?.size ?? flowSize;
        apiClient
            .marketFlow(size, undefined, { refresh: options?.refresh })
            .then((res) => {
                setFlow(res.items || []);
                setFlowDegraded(Boolean(res.degraded_mode));
                setFlowWarning(res.warning || null);
            })
            .catch((err: any) => {
                setFlowError(err?.message || 'Akış verisi alınamadı.');
            })
            .finally(() => {
                flowInFlight.current = false;
                setFlowLoading(false);
            });
    }, [flowSize]);

    const handleFlowSizeChange = useCallback(
        (next: number) => {
            if (next === flowSize) return;
            setFlowSize(next);
            try {
                window.localStorage.setItem(FLOW_SIZE_STORAGE_KEY, String(next));
            } catch {
                // localStorage erisilemez; sessiz gec.
            }
            // Cache key server tarafinda limit bazli kovalara ayriliyor; bu
            // nedenle kullanicinin secimine guvenerek tekrar yukluyoruz.
            loadFlow({ size: next });
        },
        [flowSize, loadFlow],
    );

    const loadXu030 = useCallback((force = false) => {
        if (xu030InFlight.current) return;
        if (!force && xu030) return;
        xu030InFlight.current = true;
        setXu030Loading(true);
        setXu030Error(null);
        apiClient
            .marketXu030()
            .then((res) => {
                setXu030(res.rows || []);
            })
            .catch((err: any) => {
                setXu030Error(err?.message || 'XU030 verisi alınamadı.');
            })
            .finally(() => {
                xu030InFlight.current = false;
                setXu030Loading(false);
            });
    }, [xu030]);

    const loadCommodities = useCallback((force = false) => {
        if (commoditiesInFlight.current) return;
        if (!force && commodities) return;
        commoditiesInFlight.current = true;
        setCommoditiesLoading(true);
        setCommoditiesError(null);
        apiClient
            .marketCommodities()
            .then((res) => {
                setCommodities(res.items || []);
                setCommoditiesDelayNote(res.delay_note || '');
            })
            .catch((err: any) => {
                setCommoditiesError(err?.message || 'Emtia verisi alınamadı.');
            })
            .finally(() => {
                commoditiesInFlight.current = false;
                setCommoditiesLoading(false);
            });
    }, [commodities]);

    const loadFx = useCallback((force = false) => {
        if (fxInFlight.current) return;
        if (!force && fx) return;
        fxInFlight.current = true;
        setFxLoading(true);
        setFxError(null);
        apiClient
            .marketFx()
            .then((res) => {
                setFx(res.items || []);
                setFxDelayNote(res.delay_note || '');
            })
            .catch((err: any) => {
                setFxError(err?.message || 'Döviz verisi alınamadı.');
            })
            .finally(() => {
                fxInFlight.current = false;
                setFxLoading(false);
            });
    }, [fx]);

    const loadIndices = useCallback((force = false) => {
        if (indicesInFlight.current) return;
        if (!force && indicesData) return;
        indicesInFlight.current = true;
        setIndicesLoading(true);
        setIndicesError(null);
        apiClient
            .marketIndices()
            .then((res) => {
                setIndicesData(res.items || []);
            })
            .catch((err: any) => {
                setIndicesError(err?.message || 'Endeks verisi alınamadı.');
            })
            .finally(() => {
                indicesInFlight.current = false;
                setIndicesLoading(false);
            });
    }, [indicesData]);

    useEffect(() => {
        if (!open) return;
        loadFlow();
        const xu030Timer = window.setTimeout(() => loadXu030(), 150);
        const indicesTimer = window.setTimeout(() => loadIndices(), 300);
        const fxTimer = window.setTimeout(() => loadFx(), 450);
        const commoditiesTimer = window.setTimeout(() => loadCommodities(), 700);
        return () => {
            window.clearTimeout(xu030Timer);
            window.clearTimeout(indicesTimer);
            window.clearTimeout(fxTimer);
            window.clearTimeout(commoditiesTimer);
        };
    }, [open, loadFlow, loadXu030, loadIndices, loadFx, loadCommodities]);

    useEffect(() => {
        if (!open) return;
        if (tab === 'XU030') loadXu030();
        else if (tab === 'ENDEKSLER') loadIndices();
        else if (tab === 'DOVIZ') loadFx();
        else if (tab === 'EMTIA') loadCommodities();
    }, [open, tab, loadXu030, loadIndices, loadFx, loadCommodities]);

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.key === 'Escape' && open) setOpen(false);
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [open]);

    // OTO YENİLEME: Panel açıkken ve Akış sekmesindeyken her 30 saniyede bir KAP'ı canlı çek
    useEffect(() => {
        if (!open || panelMode !== 'flow') return;
        const intervalId = window.setInterval(() => {
            loadFlow({ refresh: true });
        }, 30000);
        return () => window.clearInterval(intervalId);
    }, [open, panelMode, loadFlow]);

    // OTO YENİLEME (PİYASALAR): Panel açıkken her 10 saniyede bir diğer sekmeleri arka planda tazeleyelim
    useEffect(() => {
        if (!open || panelMode !== 'markets') return;
        const intervalId = window.setInterval(() => {
            if (tab === 'XU030') loadXu030(true);
            else if (tab === 'ENDEKSLER') loadIndices(true);
            else if (tab === 'DOVIZ') loadFx(true);
            else if (tab === 'EMTIA') loadCommodities(true);
            // XU100 parent'tan (MarketsView) zaten setInterval üzerinden yenilenip geliyor.
        }, 3000);
        return () => window.clearInterval(intervalId);
    }, [open, panelMode, tab, loadXu030, loadFx, loadCommodities]);

    const sortedXu100 = useMemo(() => {
        const arr = [...rows];
        arr.sort((a, b) => (b.change_pct ?? -Infinity) - (a.change_pct ?? -Infinity));
        return arr;
    }, [rows]);

    const sortedXu030 = useMemo(() => {
        if (!xu030) return [];
        const arr = [...xu030];
        arr.sort((a, b) => (b.change_pct ?? -Infinity) - (a.change_pct ?? -Infinity));
        return arr;
    }, [xu030]);

    const filteredFlow = useMemo(() => {
        if (!flow) return [] as MarketFlowItem[];
        return flow.filter((item) => matchesFlowFilter(item, flowFilter));
    }, [flow, flowFilter]);

    const groupedFlow = useMemo(() => {
        const groups = new Map<string, { key: string; label: string; order: number; items: MarketFlowItem[] }>();
        for (const item of filteredFlow) {
            const g = flowGroupLabel(item.published_at);
            const bucket = groups.get(g.key) ?? { ...g, items: [] };
            bucket.items.push(item);
            groups.set(g.key, bucket);
        }
        return Array.from(groups.values())
            .sort((a, b) => b.order - a.order)
            .map((g) => ({ key: g.key, label: g.label, items: g.items }));
    }, [filteredFlow]);

    const renderFlowItem = (item: MarketFlowItem) => {
        const codes = item.stock_codes && item.stock_codes.length > 1
            ? item.stock_codes.slice(0, 3).join(', ')
            : item.symbol;
        return (
            <button
                key={item.id}
                type="button"
                className="ms-flow-item"
                onClick={() => {
                    onSelectTicker(item.symbol);
                    setOpen(false);
                }}
            >
                <div className="ms-flow-meta">
                    <span className="ms-flow-source">{item.source}</span>
                    <span className="ms-flow-dot">•</span>
                    <span className="ms-flow-symbol">{codes}</span>
                    <span className="ms-flow-time">{formatTimeHm(item.published_at)}</span>
                </div>
                <div className="ms-flow-title">{item.title}</div>
            </button>
        );
    };

    return (
        <>
            <button
                type="button"
                className={`ms-toggle${open ? ' is-open' : ''}`}
                onClick={() => setOpen((v) => !v)}
                aria-label={open ? 'Paneli Kapat' : 'Piyasa Panelini Aç'}
                title={open ? 'Paneli Kapat' : 'Piyasa Paneli'}
            >
                {open ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
                {!open && <Activity size={16} />}
            </button>

            {open && <div className="ms-backdrop" onClick={() => setOpen(false)} />}

            <aside className={`ms-panel${open ? ' is-open' : ''}`} aria-hidden={!open}>
                <div className="ms-panel-modes" role="tablist" aria-label="Panel görünümü">
                    {PANEL_MODES.map((m) => (
                        <button
                            key={m.id}
                            type="button"
                            role="tab"
                            aria-selected={panelMode === m.id}
                            className={`ms-mode-tab${panelMode === m.id ? ' active' : ''}`}
                            onClick={() => setPanelMode(m.id)}
                        >
                            {m.label}
                        </button>
                    ))}
                </div>

                {panelMode === 'markets' && (
                <section className="ms-section ms-markets ms-panel-single">
                    <div className="ms-tabs" role="tablist" aria-label="Piyasa sekmeleri">
                        {TABS.map((t) => (
                            <button
                                key={t.id}
                                type="button"
                                role="tab"
                                aria-selected={tab === t.id}
                                className={`ms-tab${tab === t.id ? ' active' : ''}`}
                                onClick={() => setTab(t.id)}
                            >
                                {t.label}
                            </button>
                        ))}
                    </div>

                    <div className="ms-table-head">
                        <span>Sembol</span>
                        <span className="ms-th-right">Fiyat</span>
                        <span className="ms-th-right">Değişim</span>
                    </div>

                    <div className="ms-list">
                        {tab === 'XU100' && (
                            sortedXu100.length === 0 ? (
                                <div className="ms-empty">Veri bulunamadı.</div>
                            ) : (
                                sortedXu100.map((row) => (
                                    <FlashRow
                                        key={row.company}
                                        price={row.price}
                                        onClick={() => {
                                            onSelectTicker(row.company);
                                            setOpen(false);
                                        }}
                                    >
                                        <span className="ms-sym">{row.company}</span>
                                        <span className="ms-price">{formatPrice(row.price, row.price_currency)}</span>
                                        <span className={`ms-change ${changeClass(row.change_pct)}`}>
                                            {formatPct(row.change_pct)}
                                        </span>
                                    </FlashRow>
                                ))
                            )
                        )}

                        {tab === 'XU030' && (
                            xu030Loading && !xu030 ? (
                                <div className="ms-empty">Yükleniyor…</div>
                            ) : xu030Error ? (
                                <div className="ms-empty ms-empty-error">{xu030Error}</div>
                            ) : sortedXu030.length === 0 ? (
                                <div className="ms-empty">Veri bulunamadı.</div>
                            ) : (
                                sortedXu030.map((row) => (
                                    <FlashRow
                                        key={row.company}
                                        price={row.price}
                                        onClick={() => {
                                            onSelectTicker(row.company);
                                            setOpen(false);
                                        }}
                                    >
                                        <span className="ms-sym">{row.company}</span>
                                        <span className="ms-price">{formatPrice(row.price, row.price_currency)}</span>
                                        <span className={`ms-change ${changeClass(row.change_pct)}`}>
                                            {formatPct(row.change_pct)}
                                        </span>
                                    </FlashRow>
                                ))
                            )
                        )}
                        
                        {tab === 'ENDEKSLER' && (
                            indicesLoading && !indicesData ? (
                                <div className="ms-empty">Yükleniyor…</div>
                            ) : indicesError ? (
                                <div className="ms-empty ms-empty-error">{indicesError}</div>
                            ) : !indicesData || indicesData.length === 0 ? (
                                <div className="ms-empty">Veri bulunamadı.</div>
                            ) : (
                                indicesData.map((row) => (
                                    <FlashRow
                                        key={row.symbol}
                                        price={row.price}
                                        className="ms-row-static"
                                        title={row.label}
                                    >
                                        <span className="ms-sym">{row.symbol}</span>
                                        <span className="ms-price">
                                            {formatPrice(row.price, row.currency, pickDecimals(row.price))}
                                        </span>
                                        <span className={`ms-change ${changeClass(row.change_pct)}`}>
                                            {formatPct(row.change_pct)}
                                        </span>
                                    </FlashRow>
                                ))
                            )
                        )}

                        {tab === 'DOVIZ' && (
                            fxLoading && !fx ? (
                                <div className="ms-empty">Yükleniyor…</div>
                            ) : fxError ? (
                                <div className="ms-empty ms-empty-error">{fxError}</div>
                            ) : !fx || fx.length === 0 ? (
                                <div className="ms-empty">Veri bulunamadı.</div>
                            ) : (
                                <>
                                    {fx.map((row) => (
                                        <FlashRow
                                            key={row.symbol}
                                            price={row.price}
                                            className="ms-row-static"
                                            title={row.label}
                                        >
                                            <span className="ms-sym">{row.symbol}</span>
                                            <span className="ms-price">
                                                {formatPrice(row.price, row.currency, pickDecimals(row.price))}
                                            </span>
                                            <span className={`ms-change ${changeClass(row.change_pct)}`}>
                                                {formatPct(row.change_pct)}
                                            </span>
                                        </FlashRow>
                                    ))}
                                    {fxDelayNote && <div className="ms-delay-note">{fxDelayNote}</div>}
                                </>
                            )
                        )}

                        {tab === 'EMTIA' && (
                            commoditiesLoading && !commodities ? (
                                <div className="ms-empty">Yükleniyor…</div>
                            ) : commoditiesError ? (
                                <div className="ms-empty ms-empty-error">{commoditiesError}</div>
                            ) : !commodities || commodities.length === 0 ? (
                                <div className="ms-empty">Veri bulunamadı.</div>
                            ) : (
                                <>
                                    {commodities.map((row) => (
                                        <FlashRow
                                            key={row.symbol}
                                            price={row.price}
                                            className="ms-row-static"
                                            title={row.label}
                                        >
                                            <span className="ms-sym">{row.symbol}</span>
                                            <span className="ms-price">
                                                {formatPrice(row.price, row.currency, pickDecimals(row.price))}
                                            </span>
                                            <span className={`ms-change ${changeClass(row.change_pct)}`}>
                                                {formatPct(row.change_pct)}
                                            </span>
                                        </FlashRow>
                                    ))}
                                    {commoditiesDelayNote && (
                                        <div className="ms-delay-note">{commoditiesDelayNote}</div>
                                    )}
                                </>
                            )
                        )}
                    </div>
                </section>
                )}

                {panelMode === 'flow' && (
                <section className="ms-section ms-flow ms-panel-single">
                    <div className="ms-flow-head">
                        <h3>Akış</h3>
                        <button
                            type="button"
                            className="ms-flow-refresh"
                            onClick={() => loadFlow({ refresh: true })}
                            disabled={flowLoading}
                            title="Akışı yenile"
                        >
                            {flowLoading ? 'Yükleniyor…' : 'Yenile'}
                        </button>
                    </div>

                    <div className="ms-flow-filters" role="tablist" aria-label="Akış filtresi">
                        {FLOW_FILTERS.map((f) => (
                            <button
                                key={f.id}
                                type="button"
                                role="tab"
                                aria-selected={flowFilter === f.id}
                                className={`ms-flow-filter${flowFilter === f.id ? ' active' : ''}`}
                                onClick={() => setFlowFilter(f.id)}
                            >
                                {f.label}
                            </button>
                        ))}
                    </div>

                    <div className="ms-flow-size" role="group" aria-label="Akış boyutu">
                        <span className="ms-flow-size-label">Kayıt</span>
                        {FLOW_SIZE_OPTIONS.map((option) => (
                            <button
                                key={option.value}
                                type="button"
                                aria-pressed={flowSize === option.value}
                                className={`ms-flow-size-btn${flowSize === option.value ? ' active' : ''}`}
                                onClick={() => handleFlowSizeChange(option.value)}
                                disabled={flowLoading && flowSize === option.value}
                            >
                                {option.label}
                            </button>
                        ))}
                    </div>

                    {flowDegraded && flowWarning && (
                        <div className="ms-flow-warning" role="status">
                            {flowWarning}
                        </div>
                    )}

                    {flowLoading && !flow && <div className="ms-empty">Yükleniyor…</div>}
                    {flowError && <div className="ms-empty ms-empty-error">{flowError}</div>}
                    {!flowLoading && !flowError && flow && filteredFlow.length === 0 && (
                        <div className="ms-empty">Bu filtreye uygun kayıt yok.</div>
                    )}
                    <div className="ms-flow-list">
                        {groupedFlow.map((group) => (
                            <div key={group.key} className="ms-flow-group">
                                <div className="ms-flow-group-head">
                                    <span className="ms-flow-group-label">{group.label}</span>
                                    <span className="ms-flow-group-count">{group.items.length}</span>
                                </div>
                                {group.items.map(renderFlowItem)}
                            </div>
                        ))}
                    </div>
                </section>
                )}
            </aside>
        </>
    );
}
