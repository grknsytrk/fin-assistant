import { useEffect, useMemo, useRef, useState } from 'react';
import { Clock, BarChart3, FileText, Wallet, Star, FolderPlus } from 'lucide-react';
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

export default function MarketWatchRail({ xu100Rows, onSelectTicker }: MarketWatchRailProps) {
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [activeTool, setActiveTool] = useState<'markets' | 'watchlist' | 'news' | 'history' | 'portfolio'>('markets');
    const [activeTab, setActiveTab] = useState<RailTab>('global');
    const [globalPayload, setGlobalPayload] = useState<MarketWatchGlobalResponse | null>(null);
    const [commodityRows, setCommodityRows] = useState<CommodityQuote[] | null>(null);
    const [fxRows, setFxRows] = useState<FxQuote[] | null>(null);
    const [xu030Payload, setXu030Payload] = useState<MarketStocksResponse | null>(null);
    const [loading, setLoading] = useState<Partial<Record<RailTab, boolean>>>({});
    const [errors, setErrors] = useState<Partial<Record<RailTab, string | null>>>({});
    const inFlightRef = useRef<Partial<Record<RailTab, boolean>>>({});

    const setTabLoading = (tab: RailTab, value: boolean) => {
        setLoading((previous) => ({ ...previous, [tab]: value }));
    };

    const setTabError = (tab: RailTab, value: string | null) => {
        setErrors((previous) => ({ ...previous, [tab]: value }));
    };

    useEffect(() => {
        if (isCollapsed) return;

        if (activeTab === 'global' && !globalPayload && !inFlightRef.current.global) {
            inFlightRef.current.global = true;
            setTabLoading('global', true);
            setTabError('global', null);
            apiClient
                .marketWatchGlobal()
                .then((payload) => {
                    setGlobalPayload(payload);
                })
                .catch((error) => {
                    setTabError('global', error instanceof Error ? error.message : 'Endeks verisi alınamadı.');
                })
                .finally(() => {
                    inFlightRef.current.global = false;
                    setTabLoading('global', false);
                });
        }

        if (activeTab === 'commodities' && !commodityRows && !inFlightRef.current.commodities) {
            inFlightRef.current.commodities = true;
            setTabLoading('commodities', true);
            setTabError('commodities', null);
            apiClient
                .marketCommodities()
                .then((payload) => {
                    setCommodityRows(payload.items || []);
                })
                .catch((error) => {
                    setTabError('commodities', error instanceof Error ? error.message : 'Emtia verisi alınamadı.');
                })
                .finally(() => {
                    inFlightRef.current.commodities = false;
                    setTabLoading('commodities', false);
                });
        }

        if (activeTab === 'fx' && !fxRows && !inFlightRef.current.fx) {
            inFlightRef.current.fx = true;
            setTabLoading('fx', true);
            setTabError('fx', null);
            apiClient
                .marketFx()
                .then((payload) => {
                    setFxRows(payload.items || []);
                })
                .catch((error) => {
                    setTabError('fx', error instanceof Error ? error.message : 'Döviz verisi alınamadı.');
                })
                .finally(() => {
                    inFlightRef.current.fx = false;
                    setTabLoading('fx', false);
                });
        }

        if (activeTab === 'xu030' && !xu030Payload && !inFlightRef.current.xu030) {
            inFlightRef.current.xu030 = true;
            setTabLoading('xu030', true);
            setTabError('xu030', null);
            apiClient
                .marketStocks({ index: 'XU030' as MarketStockIndex })
                .then((payload) => {
                    setXu030Payload(payload);
                })
                .catch((error) => {
                    setTabError('xu030', error instanceof Error ? error.message : 'XU030 verisi alınamadı.');
                })
                .finally(() => {
                    inFlightRef.current.xu030 = false;
                    setTabLoading('xu030', false);
                });
        }
    }, [activeTab, commodityRows, fxRows, globalPayload, isCollapsed, xu030Payload]);

    const rows = useMemo<RailRow[]>(() => {
        if (activeTab === 'global') return (globalPayload?.items || []).map((item) => fromWatchItem(item, 'index'));
        if (activeTab === 'commodities') return (commodityRows || []).map((item) => fromWatchItem(item, 'commodity'));
        if (activeTab === 'fx') return (fxRows || []).map((item) => fromWatchItem(item, 'fx'));
        if (activeTab === 'xu030') return (xu030Payload?.rows || []).map(fromStockRow);
        return xu100Rows.map(fromStockRow);
    }, [activeTab, commodityRows, fxRows, globalPayload?.items, xu030Payload?.rows, xu100Rows]);

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
                    <h2 className="mwr-panel-title">
                        {activeTool === 'watchlist' ? 'İzleme listesi' : 'Piyasalar'}
                    </h2>
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
                    <div className="mwr-empty-state">
                        <div className="mwr-empty-icon">
                            <FolderPlus size={48} />
                        </div>
                        <h3>Henüz izleme listenize ekleme yapmadınız</h3>
                        <p>
                            İzlediğiniz hisseleri, fonları, endeksleri vs. izleme listenize ekleyip takip edebilirsiniz.
                        </p>
                        <button type="button" className="mwr-empty-action">
                            İzleme Listesini Düzenle
                        </button>
                    </div>
                ) : (
                    <>
                        <div className="mwr-table-head">
                            <span>Sembol</span>
                            <span>Fiyat</span>
                            <span>Değişim</span>
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

                                if (row.clickable) {
                                    return (
                                        <button
                                            key={`${activeTab}-${row.symbol}`}
                                            type="button"
                                            className="mwr-row mwr-row-clickable"
                                            onClick={() => onSelectTicker(row.symbol)}
                                        >
                                            {content}
                                        </button>
                                    );
                                }

                                return (
                                    <div key={`${activeTab}-${row.symbol}`} className={`mwr-row${row.error ? ' mwr-row-muted' : ''}`}>
                                        {content}
                                    </div>
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
