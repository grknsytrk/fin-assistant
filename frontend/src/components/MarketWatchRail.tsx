import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDown, ArrowUp, ArrowUpDown, Clock, BarChart3, FileText, Wallet, Star, FolderPlus, Search, X, Pencil, Trash2, GripVertical } from 'lucide-react';
import { apiClient } from '../api/client';
import SymbolLogo, { type SymbolLogoKind } from './SymbolLogo';
import type {
    CommodityQuote,
    FxQuote,
    MarketStockIndex,
    MarketStockRow,
    MarketStocksResponse,
    MarketUniverseRow,
    MarketWatchGlobalResponse,
    MarketWatchItem,
} from '../api/types';
import './MarketWatchRail.css';

type RailTab = 'global' | 'commodities' | 'fx' | 'xu100' | 'xu030';
type StockRailTab = Extract<RailTab, 'xu100' | 'xu030'>;
type RailSortKey = 'symbol' | 'price' | 'changePct';
type RailSortDirection = 'asc' | 'desc';

const LIVE_RAIL_REFRESH_MS = 3000;
const STOCK_TAB_TO_INDEX: Record<StockRailTab, MarketStockIndex> = {
    xu100: 'XU100',
    xu030: 'XU030',
};

const DEFAULT_ORDER: Partial<Record<RailTab, string[]>> = {
    global: ['SP500', 'NASDAQ', 'DOW', 'DAX', 'FTSE', 'CAC40', 'NIKKEI', 'HANGSENG'],
    commodities: ['BRENT', 'WTI', 'USOIL', 'ALTIN', 'GUMUS', 'PLATIN', 'PALADYUM', 'BAKIR', 'DOGALGAZ', 'KAHVE', 'SEKER', 'BUGDAY', 'MISIR', 'PAMUK', 'KAKAO', 'SOYA'],
    fx: ['USD/TRY', 'EUR/TRY', 'GBP/TRY', 'CHF/TRY', 'AUD/TRY', 'CAD/TRY', 'JPY/TRY', 'EUR/USD', 'GBP/USD', 'USD/JPY']
};

interface RailRow {
    symbol: string;
    label: string;
    kind: SymbolLogoKind;
    logoUrl?: string | null;
    price: number | null;
    changePct: number | null;
    currency?: string | null;
    error?: string | null;
    clickable?: boolean;
}

interface MarketWatchRailProps {
    xu100Rows: MarketUniverseRow[];
    onSelectTicker: (ticker: string) => void;
}

const RAIL_TABS: Array<{ id: RailTab; label: string }> = [
    { id: 'global', label: 'Endeks' },
    { id: 'commodities', label: 'Emtia' },
    { id: 'fx', label: 'Döviz' },
    { id: 'xu100', label: 'XU100' },
    { id: 'xu030', label: 'XU030' },
];

const RAIL_SORT_COLUMNS: Array<{ key: RailSortKey; label: string; align?: 'left' | 'right' }> = [
    { key: 'symbol', label: 'Sembol' },
    { key: 'price', label: 'Fiyat', align: 'right' },
    { key: 'changePct', label: 'Değişim', align: 'right' },
];

function formatRailPrice(value: number | null): string {
    if (value == null || !Number.isFinite(value)) return '-';
    const abs = Math.abs(value);
    const decimals = abs < 10 ? 4 : abs < 100 ? 3 : 2;
    return value.toLocaleString('tr-TR', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    });
}

function formatRailPct(value: number | null): string {
    if (value == null || !Number.isFinite(value)) return '%0,00';
    const sign = value > 0 ? '+' : '';
    return `%${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function changeClass(value: number | null): string {
    if (value == null || value === 0) return 'mwr-change-flat';
    return value > 0 ? 'mwr-change-up' : 'mwr-change-down';
}

function railSortValue(row: RailRow, key: RailSortKey): string | number | null {
    if (key === 'symbol') return row.symbol;
    if (key === 'price') return row.price;
    return row.changePct;
}

function compareRailRows(a: RailRow, b: RailRow, key: RailSortKey, direction: RailSortDirection): number {
    const av = railSortValue(a, key);
    const bv = railSortValue(b, key);
    const aMissing = av == null || av === '' || (typeof av === 'number' && !Number.isFinite(av));
    const bMissing = bv == null || bv === '' || (typeof bv === 'number' && !Number.isFinite(bv));

    if (aMissing && bMissing) return a.symbol.localeCompare(b.symbol, 'tr');
    if (aMissing) return 1;
    if (bMissing) return -1;

    let result = 0;
    if (key === 'symbol') {
        result = String(av).localeCompare(String(bv), 'tr');
    } else {
        result = Number(av) - Number(bv);
    }
    if (result === 0) result = a.symbol.localeCompare(b.symbol, 'tr');
    return direction === 'asc' ? result : -result;
}

function fromWatchItem(item: MarketWatchItem | CommodityQuote | FxQuote, kind: SymbolLogoKind): RailRow {
    return {
        symbol: item.symbol,
        label: item.label,
        kind,
        logoUrl: item.logo_url,
        price: item.price,
        changePct: item.change_pct,
        currency: item.currency,
        error: item.error,
    };
}

function fromStockRow(row: MarketUniverseRow | MarketStockRow): RailRow {
    const ticker = 'symbol' in row && typeof row.symbol === 'string' ? row.symbol : row.company;
    return {
        symbol: ticker,
        label: row.company,
        kind: 'stock',
        logoUrl: row.logo_url,
        price: row.price,
        changePct: row.change_pct,
        currency: row.price_currency,
        clickable: true,
    };
}

function FlashRailRow({
    row,
    onClick,
    children,
}: React.PropsWithChildren<{ row: RailRow; onClick?: () => void }>) {
    const previousRef = useRef<{ price: number | null; changePct: number | null }>({
        price: row.price,
        changePct: row.changePct,
    });
    const [flashClass, setFlashClass] = useState('');

    useEffect(() => {
        const previous = previousRef.current;
        const priceChanged =
            row.price != null && previous.price != null && row.price !== previous.price;
        const changeChanged =
            !priceChanged
            && row.changePct != null
            && previous.changePct != null
            && row.changePct !== previous.changePct;

        if (priceChanged || changeChanged) {
            const currentValue = priceChanged ? row.price : row.changePct;
            const previousValue = priceChanged ? previous.price : previous.changePct;
            setFlashClass((currentValue ?? 0) > (previousValue ?? 0) ? 'mwr-flash-up' : 'mwr-flash-down');
            const timer = window.setTimeout(() => setFlashClass(''), 1000);
            previousRef.current = { price: row.price, changePct: row.changePct };
            return () => window.clearTimeout(timer);
        }

        previousRef.current = { price: row.price, changePct: row.changePct };
    }, [row.price, row.changePct]);

    const className = [
        'mwr-row',
        onClick ? 'mwr-row-clickable' : '',
        row.error ? 'mwr-row-muted' : '',
        flashClass,
    ]
        .filter(Boolean)
        .join(' ');

    if (onClick) {
        return (
            <button type="button" className={className} onClick={onClick}>
                {children}
            </button>
        );
    }

    return <div className={className}>{children}</div>;
}

export default function MarketWatchRail({ xu100Rows, onSelectTicker }: MarketWatchRailProps) {
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [activeTool, setActiveTool] = useState<'markets' | 'watchlist' | 'news' | 'history' | 'portfolio'>('markets');
    const [activeTab, setActiveTab] = useState<RailTab>('global');
    const [sort, setSort] = useState<{ key: RailSortKey | null; direction: RailSortDirection | null }>({
        key: null,
        direction: null,
    });
    const [globalPayload, setGlobalPayload] = useState<MarketWatchGlobalResponse | null>(null);
    const [commodityRows, setCommodityRows] = useState<CommodityQuote[] | null>(null);
    const [fxRows, setFxRows] = useState<FxQuote[] | null>(null);
    const [xu100Payload, setXu100Payload] = useState<MarketStocksResponse | null>(null);
    const [xu030Payload, setXu030Payload] = useState<MarketStocksResponse | null>(null);
    const [loading, setLoading] = useState<Partial<Record<RailTab, boolean>>>({});
    const [errors, setErrors] = useState<Partial<Record<RailTab, string | null>>>({});
    const inFlightRef = useRef<Partial<Record<RailTab, boolean>>>({});

    // İzleme listesi state
    const [watchlistSearch, setWatchlistSearch] = useState('');
    const [watchlistSymbols, setWatchlistSymbols] = useState<string[]>(() => {
        try {
            const saved = localStorage.getItem('mwr_watchlist');
            return saved ? JSON.parse(saved) : [];
        } catch { return []; }
    });

    // İzleme listesi: arama sonuçları
    const watchlistCandidates = useMemo(() => {
        const term = watchlistSearch.trim().toLowerCase();
        return (xu100Rows || [])
            .filter((row) => !watchlistSymbols.includes(row.company))
            .filter((row) => {
                if (!term) return true;
                return row.company.toLowerCase().includes(term);
            })
            .slice(0, 20);
    }, [xu100Rows, watchlistSearch, watchlistSymbols]);

    // İzleme listesindeki hisselerin satırları
    const watchlistRows = useMemo(() => {
        if (!watchlistSymbols.length) return [];
        return watchlistSymbols
            .map((sym) => (xu100Rows || []).find((r) => r.company === sym))
            .filter(Boolean) as typeof xu100Rows;
    }, [xu100Rows, watchlistSymbols]);

    const handleAddToWatchlist = (symbol: string) => {
        setWatchlistSymbols((prev) => {
            if (prev.includes(symbol)) return prev;
            const next = [...prev, symbol];
            localStorage.setItem('mwr_watchlist', JSON.stringify(next));
            return next;
        });
    };

    const handleRemoveFromWatchlist = (symbol: string) => {
        setWatchlistSymbols((prev) => {
            const next = prev.filter((s) => s !== symbol);
            localStorage.setItem('mwr_watchlist', JSON.stringify(next));
            return next;
        });
    };

    // İzleme listesi düzenleme modu ve Drag-Drop
    const [watchlistEditMode, setWatchlistEditMode] = useState(false);
    const [draggedSymbol, setDraggedSymbol] = useState<string | null>(null);

    const handleDragStart = (e: React.DragEvent<HTMLDivElement>, symbol: string) => {
        setDraggedSymbol(symbol);
        e.dataTransfer.effectAllowed = 'move';
        // Şeffaf bir drag imajı için
        const el = e.currentTarget;
        setTimeout(() => el.classList.add('is-dragging'), 0);
    };

    const handleDragEnter = (e: React.DragEvent<HTMLDivElement>, targetSymbol: string) => {
        e.preventDefault();
        if (draggedSymbol && draggedSymbol !== targetSymbol) {
            setWatchlistSymbols((prev) => {
                const dragIndex = prev.indexOf(draggedSymbol);
                const dropIndex = prev.indexOf(targetSymbol);
                if (dragIndex < 0 || dropIndex < 0 || dragIndex === dropIndex) return prev;

                const next = [...prev];
                next.splice(dragIndex, 1);
                next.splice(dropIndex, 0, draggedSymbol);
                return next;
            });
        }
    };

    const handleDragEnd = (e: React.DragEvent<HTMLDivElement>) => {
        setDraggedSymbol(null);
        e.currentTarget.classList.remove('is-dragging');
        // Sürükleme bittiğinde son hali kaydet
        setWatchlistSymbols((prev) => {
            localStorage.setItem('mwr_watchlist', JSON.stringify(prev));
            return prev;
        });
    };

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        // İşlemler artık DragEnter anında gerçek zamanlı yapıldığı için burada bir şeye gerek yok
    };

    // İzleme listesi sıralaması
    const [watchlistSort, setWatchlistSort] = useState<{ key: RailSortKey | null; direction: RailSortDirection | null }>({
        key: null,
        direction: null,
    });

    const handleWatchlistSort = (key: RailSortKey) => {
        setWatchlistSort((prev) => {
            if (prev.key === key) {
                const initialDir = key === 'symbol' ? 'asc' : 'desc';
                if (prev.direction === initialDir) {
                    return { key, direction: initialDir === 'asc' ? 'desc' : 'asc' };
                }
                // Eğer 2. state'teyse, 3. state olan default (sıralamasız) hale getir
                return { key: null, direction: null };
            }
            // İlk tıklamada kolonun varsayılan yönüne göre sırala
            return { key, direction: key === 'symbol' ? 'asc' : 'desc' };
        });
    };

    const sortedWatchlistRailRows = useMemo<RailRow[]>(() => {
        const railRows: RailRow[] = watchlistRows.map((row) => ({
            symbol: row.company,
            label: row.company,
            kind: 'stock' as SymbolLogoKind,
            logoUrl: row.logo_url,
            price: row.price,
            changePct: row.change_pct,
            currency: row.price_currency,
            clickable: true,
        }));

        if (!watchlistSort.key || !watchlistSort.direction) {
            return railRows;
        }

        return [...railRows].sort((a, b) => compareRailRows(a, b, watchlistSort.key as RailSortKey, watchlistSort.direction as RailSortDirection));
    }, [watchlistRows, watchlistSort.key, watchlistSort.direction]);

    const setTabLoading = (tab: RailTab, value: boolean) => {
        setLoading((previous) => ({ ...previous, [tab]: value }));
    };

    const setTabError = (tab: RailTab, value: string | null) => {
        setErrors((previous) => ({ ...previous, [tab]: value }));
    };

    const handleSort = (key: RailSortKey) => {
        setSort((prev) => {
            if (prev.key === key) {
                const initialDir = key === 'symbol' ? 'asc' : 'desc';
                if (prev.direction === initialDir) {
                    return { key, direction: initialDir === 'asc' ? 'desc' : 'asc' };
                }
                return { key: null, direction: null };
            }
            return { key, direction: key === 'symbol' ? 'asc' : 'desc' };
        });
    };

    // Endeks (global) verilerini çeker; döviz ile aynı kalıpta canlı polling destekler.
    const loadGlobalRows = useCallback((options?: { silent?: boolean }) => {
        if (inFlightRef.current.global) return;
        inFlightRef.current.global = true;
        if (!options?.silent) {
            setTabLoading('global', true);
            setTabError('global', null);
        }
        apiClient
            .marketWatchGlobal()
            .then((payload) => {
                setGlobalPayload(payload);
                setTabError('global', null);
            })
            .catch((error) => {
                if (!options?.silent || !globalPayload?.items?.length) {
                    setTabError('global', error instanceof Error ? error.message : 'Endeks verisi alınamadı.');
                }
            })
            .finally(() => {
                inFlightRef.current.global = false;
                if (!options?.silent) setTabLoading('global', false);
            });
    }, [globalPayload?.items?.length]);

    // Emtia verilerini çeker; döviz ile aynı kalıpta canlı polling destekler.
    const loadCommodityRows = useCallback((options?: { silent?: boolean }) => {
        if (inFlightRef.current.commodities) return;
        inFlightRef.current.commodities = true;
        if (!options?.silent) {
            setTabLoading('commodities', true);
            setTabError('commodities', null);
        }
        apiClient
            .marketCommodities()
            .then((payload) => {
                setCommodityRows(payload.items || []);
                setTabError('commodities', null);
            })
            .catch((error) => {
                if (!options?.silent || !commodityRows?.length) {
                    setTabError('commodities', error instanceof Error ? error.message : 'Emtia verisi alınamadı.');
                }
            })
            .finally(() => {
                inFlightRef.current.commodities = false;
                if (!options?.silent) setTabLoading('commodities', false);
            });
    }, [commodityRows?.length]);

    const loadFxRows = useCallback((options?: { silent?: boolean }) => {
        if (inFlightRef.current.fx) return;
        inFlightRef.current.fx = true;
        if (!options?.silent) {
            setTabLoading('fx', true);
            setTabError('fx', null);
        }
        apiClient
            .marketFx()
            .then((payload) => {
                setFxRows(payload.items || []);
                setTabError('fx', null);
            })
            .catch((error) => {
                if (!options?.silent || !fxRows?.length) {
                    setTabError('fx', error instanceof Error ? error.message : 'Döviz verisi alınamadı.');
                }
            })
            .finally(() => {
                inFlightRef.current.fx = false;
                if (!options?.silent) setTabLoading('fx', false);
            });
    }, [fxRows?.length]);

    const loadStockTab = useCallback((tab: StockRailTab, options?: { silent?: boolean; refresh?: boolean }) => {
        if (inFlightRef.current[tab]) return;
        inFlightRef.current[tab] = true;
        if (!options?.silent) {
            setTabLoading(tab, true);
            setTabError(tab, null);
        }
        apiClient
            .marketStocks({ index: STOCK_TAB_TO_INDEX[tab], refresh: options?.refresh })
            .then((payload) => {
                if (tab === 'xu100') setXu100Payload(payload);
                else setXu030Payload(payload);
                setTabError(tab, null);
            })
            .catch((error) => {
                const fallback = tab === 'xu100' ? 'XU100 verisi alınamadı.' : 'XU030 verisi alınamadı.';
                const hasRows =
                    tab === 'xu100'
                        ? Boolean((xu100Payload?.rows?.length || xu100Rows.length) > 0)
                        : Boolean(xu030Payload?.rows?.length);
                if (!options?.silent || !hasRows) {
                    setTabError(tab, error instanceof Error ? error.message : fallback);
                }
            })
            .finally(() => {
                inFlightRef.current[tab] = false;
                if (!options?.silent) setTabLoading(tab, false);
            });
    }, [xu030Payload?.rows?.length, xu100Payload?.rows?.length, xu100Rows.length]);

    useEffect(() => {
        if (isCollapsed || activeTool !== 'markets') return;

        if (activeTab === 'global' && !globalPayload) {
            loadGlobalRows();
        }

        if (activeTab === 'commodities' && !commodityRows) {
            loadCommodityRows();
        }

        if (activeTab === 'fx' && !fxRows) {
            loadFxRows();
        }

        if ((activeTab === 'xu100' || activeTab === 'xu030')) {
            const hasCurrentRows =
                activeTab === 'xu100'
                    ? Boolean((xu100Payload?.rows?.length || xu100Rows.length) > 0)
                    : Boolean(xu030Payload?.rows?.length);
            loadStockTab(activeTab, { silent: hasCurrentRows, refresh: true });
        }
    }, [
        activeTab,
        activeTool,
        commodityRows,
        fxRows,
        globalPayload,
        isCollapsed,
        loadCommodityRows,
        loadFxRows,
        loadGlobalRows,
        loadStockTab,
        xu030Payload?.rows?.length,
        xu100Payload?.rows?.length,
        xu100Rows.length,
    ]);

    // Tüm sekmeler (endeks, emtia, döviz, xu100, xu030) için 3 saniyelik canlı polling döngüsü.
    useEffect(() => {
        if (isCollapsed || activeTool !== 'markets') return;

        const timer = window.setInterval(() => {
            if (activeTab === 'global') {
                loadGlobalRows({ silent: true });
            } else if (activeTab === 'commodities') {
                loadCommodityRows({ silent: true });
            } else if (activeTab === 'fx') {
                loadFxRows({ silent: true });
            } else if (activeTab === 'xu100' || activeTab === 'xu030') {
                loadStockTab(activeTab, { silent: true, refresh: true });
            }
        }, LIVE_RAIL_REFRESH_MS);
        return () => window.clearInterval(timer);
    }, [activeTab, activeTool, isCollapsed, loadCommodityRows, loadFxRows, loadGlobalRows, loadStockTab]);

    const rawRows = useMemo<RailRow[]>(() => {
        let mapped: RailRow[] = [];
        if (activeTab === 'global') mapped = (globalPayload?.items || []).map((item) => fromWatchItem(item, 'index'));
        else if (activeTab === 'commodities') mapped = (commodityRows || []).map((item) => fromWatchItem(item, 'commodity'));
        else if (activeTab === 'fx') mapped = (fxRows || []).map((item) => fromWatchItem(item, 'fx'));
        else if (activeTab === 'xu030') mapped = (xu030Payload?.rows || []).map(fromStockRow);
        else mapped = (xu100Payload?.rows || xu100Rows).map(fromStockRow);

        const order = DEFAULT_ORDER[activeTab];
        if (order) {
            mapped.sort((a, b) => {
                const idxA = order.indexOf(a.symbol);
                const idxB = order.indexOf(b.symbol);
                if (idxA !== -1 && idxB !== -1) return idxA - idxB;
                if (idxA !== -1) return -1;
                if (idxB !== -1) return 1;

                if (activeTab === 'fx') {
                    const isTryA = a.symbol.endsWith('TRY');
                    const isTryB = b.symbol.endsWith('TRY');
                    if (isTryA && !isTryB) return -1;
                    if (!isTryA && isTryB) return 1;
                }

                return a.symbol.localeCompare(b.symbol);
            });
        }
        return mapped;
    }, [activeTab, commodityRows, fxRows, globalPayload?.items, xu030Payload?.rows, xu100Payload?.rows, xu100Rows]);

    const rows = useMemo<RailRow[]>(() => {
        const sorted = [...rawRows];
        if (!sort.key || !sort.direction) return sorted;
        sorted.sort((a, b) => compareRailRows(a, b, sort.key as RailSortKey, sort.direction as RailSortDirection));
        return sorted;
    }, [rawRows, sort.direction, sort.key]);

    const tabLoading = Boolean(loading[activeTab]);
    const tabError = errors[activeTab];
    return (
        <div className={`mwr-shell${isCollapsed ? ' mwr-shell-collapsed' : ''}`}>
            <aside
                className={`mwr-panel${isCollapsed ? ' is-collapsed' : ''}`}
                aria-label="İzleme listesi"
                aria-hidden={isCollapsed}
            >
                <div className="mwr-head">
                    <div className="mwr-head-row">
                        <h2 className="mwr-panel-title">
                            {activeTool === 'watchlist' ? 'İzleme listesi' : 'Piyasalar'}
                        </h2>
                        {activeTool === 'watchlist' && watchlistSymbols.length > 0 && (
                            <button
                                type="button"
                                className={`mwr-watchlist-edit-btn${watchlistEditMode ? ' is-active' : ''}`}
                                onClick={() => setWatchlistEditMode((p) => !p)}
                                aria-label={watchlistEditMode ? 'Düzenlemeyi bitir' : 'İzleme listesini düzenle'}
                                title={watchlistEditMode ? 'Bitti' : 'Düzenle'}
                            >
                                <Pencil size={16} aria-hidden="true" />
                            </button>
                        )}
                    </div>
                    {activeTool === 'markets' && (
                        <div className="mwr-tabs" role="tablist" aria-label="İzleme sekmeleri">
                            {RAIL_TABS.map((tab) => (
                                <button
                                    key={tab.id}
                                    type="button"
                                    role="tab"
                                    aria-selected={activeTab === tab.id}
                                    className={`mwr-tab${activeTab === tab.id ? ' is-active' : ''}`}
                                    onClick={() => setActiveTab(tab.id)}
                                >
                                    {tab.label}
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                {activeTool === 'watchlist' ? (
                    <div className="mwr-watchlist">
                        <div className="mwr-watchlist-search">
                            <Search size={16} aria-hidden="true" />
                            <input
                                type="text"
                                placeholder="Hisse ara..."
                                value={watchlistSearch}
                                onChange={(e) => setWatchlistSearch(e.target.value)}
                                autoFocus
                            />
                            {watchlistSearch && (
                                <button
                                    type="button"
                                    className="mwr-watchlist-search-clear"
                                    onClick={() => setWatchlistSearch('')}
                                    aria-label="Aramayı temizle"
                                >
                                    <X size={14} aria-hidden="true" />
                                </button>
                            )}
                        </div>

                        {watchlistSearch ? (
                            <>
                                <div className="mwr-watchlist-heading">Hisse Senetleri</div>
                                <div className="mwr-watchlist-list">
                                    {watchlistCandidates.length > 0 ? (
                                        watchlistCandidates.map((row) => (
                                            <button
                                                key={row.company}
                                                type="button"
                                                className="mwr-watchlist-item"
                                                onClick={() => handleAddToWatchlist(row.company)}
                                            >
                                                <SymbolLogo
                                                    symbol={row.company}
                                                    name={row.company}
                                                    kind="stock"
                                                    logoUrl={row.logo_url}
                                                    size="sm"
                                                    className="mwr-watchlist-logo"
                                                />
                                                <span className="mwr-watchlist-copy">
                                                    <strong>{row.company}</strong>
                                                </span>
                                            </button>
                                        ))
                                    ) : (
                                        <div className="mwr-watchlist-empty-search">Sonuç bulunamadı.</div>
                                    )}
                                </div>
                            </>
                        ) : watchlistRows.length > 0 ? (
                            watchlistEditMode ? (
                                <div className="mwr-watchlist-edit-list">
                                    {watchlistSymbols.map((sym) => {
                                        const row = watchlistRows.find((r) => r.company === sym);
                                        if (!row) return null;
                                        return (
                                            <div
                                                key={sym}
                                                className="mwr-watchlist-edit-row"
                                                draggable
                                                onDragStart={(e) => handleDragStart(e, sym)}
                                                onDragEnter={(e) => handleDragEnter(e, sym)}
                                                onDragOver={(e) => e.preventDefault()}
                                                onDragEnd={handleDragEnd}
                                                onDrop={handleDrop}
                                            >
                                                <div className="mwr-watchlist-edit-reorder mwr-drag-handle">
                                                    <GripVertical size={16} />
                                                </div>
                                                <SymbolLogo
                                                    symbol={row.company}
                                                    name={row.company}
                                                    kind="stock"
                                                    logoUrl={row.logo_url}
                                                    size="sm"
                                                    className="mwr-watchlist-logo"
                                                />
                                                <span className="mwr-watchlist-edit-name">
                                                    <strong>{row.company}</strong>
                                                </span>
                                                <button
                                                    type="button"
                                                    className="mwr-watchlist-delete-btn"
                                                    onClick={() => handleRemoveFromWatchlist(sym)}
                                                    aria-label={`${sym} sil`}
                                                    title="Sil"
                                                >
                                                    <Trash2 size={16} aria-hidden="true" />
                                                </button>
                                            </div>
                                        );
                                    })}
                                </div>
                            ) : (
                                <>
                                    <div className="mwr-table-head">
                                        {RAIL_SORT_COLUMNS.map((column) => {
                                            const isActiveSort = watchlistSort.key === column.key && watchlistSort.direction !== null;
                                            const WlSortIcon = isActiveSort
                                                ? (watchlistSort.direction === 'asc' ? ArrowUp : ArrowDown)
                                                : ArrowUpDown;
                                            return (
                                                <button
                                                    key={column.key}
                                                    type="button"
                                                    className={[
                                                        'mwr-sort-button',
                                                        column.align === 'right' ? 'mwr-sort-button-right' : '',
                                                        isActiveSort ? 'is-active' : '',
                                                    ].filter(Boolean).join(' ')}
                                                    onClick={() => handleWatchlistSort(column.key)}
                                                >
                                                    <span>{column.label}</span>
                                                    <WlSortIcon size={12} aria-hidden="true" />
                                                </button>
                                            );
                                        })}
                                    </div>
                                    <div className="mwr-list">
                                        {sortedWatchlistRailRows.map((row) => (
                                            <FlashRailRow
                                                key={`wl-${row.symbol}`}
                                                row={row}
                                                onClick={() => onSelectTicker(row.symbol)}
                                            >
                                                <span className="mwr-symbol-cell">
                                                    <SymbolLogo
                                                        symbol={row.symbol}
                                                        name={row.label}
                                                        kind={row.kind}
                                                        logoUrl={row.logoUrl}
                                                        size="xs"
                                                        className="mwr-logo"
                                                    />
                                                    <span className="mwr-symbol-copy">
                                                        <strong>{row.symbol}</strong>
                                                        <small>{row.label}</small>
                                                    </span>
                                                </span>
                                                <span className="mwr-price">{formatRailPrice(row.price)}</span>
                                                <span className={`mwr-change ${changeClass(row.changePct)}`}>
                                                    {formatRailPct(row.changePct)}
                                                </span>
                                            </FlashRailRow>
                                        ))}
                                    </div>
                                </>
                            )
                        ) : (
                            <div className="mwr-empty-state">
                                <div className="mwr-empty-icon">
                                    <FolderPlus size={48} />
                                </div>
                                <h3>Henüz izleme listenize ekleme yapmadınız</h3>
                                <p>
                                    Yukarıdaki arama kutusundan hisse arayıp izleme listenize ekleyebilirsiniz.
                                </p>
                            </div>
                        )}
                    </div>
                ) : (
                    <>
                        <div className="mwr-table-head">
                            {RAIL_SORT_COLUMNS.map((column) => {
                                const isActiveSort = sort.key === column.key && sort.direction !== null;
                                const Icon = isActiveSort
                                    ? (sort.direction === 'asc' ? ArrowUp : ArrowDown)
                                    : ArrowUpDown;
                                return (
                                    <button
                                        key={column.key}
                                        type="button"
                                        className={[
                                            'mwr-sort-button',
                                            column.align === 'right' ? 'mwr-sort-button-right' : '',
                                            isActiveSort ? 'is-active' : '',
                                        ]
                                            .filter(Boolean)
                                            .join(' ')}
                                        onClick={() => handleSort(column.key)}
                                        aria-sort={
                                            isActiveSort
                                                ? sort.direction === 'asc'
                                                    ? 'ascending'
                                                    : 'descending'
                                                : 'none'
                                        }
                                    >
                                        <span>{column.label}</span>
                                        <Icon size={12} aria-hidden="true" />
                                    </button>
                                );
                            })}
                        </div>

                        {tabLoading && rows.length === 0 && (
                            <div className="mwr-skeleton">
                                {[...Array(10)].map((_, i) => (
                                    <div key={i} className="mwr-skeleton-row">
                                        <div className="mwr-skeleton-symbol-cell">
                                            <div className="mwr-skeleton-logo mwr-skeleton-pulse" />
                                            <div className="mwr-skeleton-copy">
                                                <div className="mwr-skeleton-title mwr-skeleton-pulse" />
                                                <div className="mwr-skeleton-desc mwr-skeleton-pulse" />
                                            </div>
                                        </div>
                                        <div className="mwr-skeleton-price mwr-skeleton-pulse" />
                                        <div className="mwr-skeleton-change mwr-skeleton-pulse" />
                                    </div>
                                ))}
                            </div>
                        )}
                        {tabError && rows.length === 0 && <div className="mwr-state mwr-error">{tabError}</div>}
                        {!tabLoading && !tabError && rows.length === 0 && <div className="mwr-state">Veri yok</div>}

                        <div className="mwr-list">
                            {rows.map((row) => {
                                const content = (
                                    <>
                                        <span className="mwr-symbol-cell">
                                            <SymbolLogo
                                                symbol={row.symbol}
                                                name={row.label}
                                                kind={row.kind}
                                                logoUrl={row.logoUrl}
                                                size="xs"
                                                className="mwr-logo"
                                            />
                                            <span className="mwr-symbol-copy">
                                                <strong>{row.symbol}</strong>
                                                <small>{row.label}</small>
                                            </span>
                                        </span>
                                        <span className="mwr-price">{formatRailPrice(row.price)}</span>
                                        <span className={`mwr-change ${changeClass(row.changePct)}`}>
                                            {formatRailPct(row.changePct)}
                                        </span>
                                    </>
                                );

                                return (
                                    <FlashRailRow
                                        key={`${activeTab}-${row.symbol}`}
                                        row={row}
                                        onClick={
                                            row.clickable
                                                ? () => onSelectTicker(row.symbol)
                                                : undefined
                                        }
                                    >
                                        {content}
                                    </FlashRailRow>
                                );
                            })}
                        </div>

                        {tabLoading && rows.length > 0 && <div className="mwr-sync">Güncelleniyor</div>}
                        {tabError && rows.length > 0 && <div className="mwr-sync mwr-error-text">Son yenileme başarısız</div>}
                    </>
                )}
            </aside>

            <nav className="mwr-dock" aria-label="Market araç menüsü">
                <button
                    type="button"
                    className={`mwr-dock-button${activeTool === 'watchlist' && !isCollapsed ? ' is-active' : ''}`}
                    onClick={() => {
                        if (activeTool === 'watchlist') {
                            setIsCollapsed((v) => !v);
                        } else {
                            setActiveTool('watchlist');
                            setIsCollapsed(false);
                        }
                    }}
                    aria-label="İzleme Listem"
                    title="İzleme Listem"
                >
                    <Star size={20} aria-hidden="true" />
                </button>

                <button
                    type="button"
                    className={`mwr-dock-button${activeTool === 'markets' && !isCollapsed ? ' is-active' : ''}`}
                    onClick={() => {
                        if (activeTool === 'markets') {
                            setIsCollapsed((v) => !v);
                        } else {
                            setActiveTool('markets');
                            setIsCollapsed(false);
                        }
                    }}
                    aria-label="Piyasalar"
                    title="Piyasalar"
                >
                    <BarChart3 size={20} aria-hidden="true" />
                </button>

                <span className="mwr-dock-divider" aria-hidden="true" />

                <button
                    type="button"
                    className={`mwr-dock-button${activeTool === 'news' ? ' is-active' : ''}`}
                    onClick={() => {
                        setActiveTool('news');
                        setIsCollapsed(true); // Close watchlist if we move to another tool
                    }}
                    aria-label="Haber Akışı"
                    title="Haber Akışı"
                >
                    <FileText size={18} aria-hidden="true" />
                </button>
                <button
                    type="button"
                    className={`mwr-dock-button${activeTool === 'history' ? ' is-active' : ''}`}
                    onClick={() => {
                        setActiveTool('history');
                        setIsCollapsed(true);
                    }}
                    aria-label="Geçmiş/Alarm"
                    title="Geçmiş/Alarm"
                >
                    <Clock size={18} aria-hidden="true" />
                </button>
                <button
                    type="button"
                    className={`mwr-dock-button${activeTool === 'portfolio' ? ' is-active' : ''}`}
                    onClick={() => {
                        setActiveTool('portfolio');
                        setIsCollapsed(true);
                    }}
                    aria-label="Portföyüm"
                    title="Portföyüm"
                >
                    <Wallet size={18} aria-hidden="true" />
                </button>
            </nav>
        </div>
    );
}
