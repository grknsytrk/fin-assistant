import { useEffect, useMemo, useRef, useState } from 'react';
import { apiClient } from '../api/client';
import type {
    MarketIndexCode,
    MarketIndexConstituent,
    MarketIndexDetailResponse,
    MarketIndexListRow,
    MarketIndicesResponse,
    MarketReturnBenchmark,
    MarketStockIndex,
    MarketStockRow,
    MarketStocksResponse,
    MarketUniverseResponse,
    MarketUniverseRow,
} from '../api/types';
import MarketSidebar from '../components/MarketSidebar';
import MarketWatchStrip from '../components/MarketWatchStrip';
import MarketsNavigation from '../components/MarketsNavigation';
import './MarketsView.css';

type MarketSection = 'markets' | 'stocks' | 'indices';
type SortDirection = 'asc' | 'desc';
type StockReturnMode = 'absolute' | 'relative_xu100' | 'relative_xu030';
type StockSortKey =
    | 'company'
    | 'price'
    | 'change_pct'
    | 'volume'
    | 'return_1w_pct'
    | 'return_1m_pct'
    | 'return_3m_pct'
    | 'return_6m_pct'
    | 'return_ytd_pct'
    | 'return_1y_pct';
type IndexSortKey = keyof MarketIndexListRow;

type StockReturnKey = Extract<
    StockSortKey,
    'return_1w_pct' | 'return_1m_pct' | 'return_3m_pct' | 'return_6m_pct' | 'return_ytd_pct' | 'return_1y_pct'
>;

const STOCK_COLUMNS: Array<{ key: StockSortKey; label: string; sublabel?: string; align?: 'left' | 'right' }> = [
    { key: 'company', label: 'Hisse', align: 'left' },
    { key: 'price', label: 'Fiyat', align: 'right' },
    { key: 'change_pct', label: 'Gün %', align: 'right' },
    { key: 'volume', label: 'Hacim', align: 'right' },
    { key: 'return_1w_pct', label: 'Getiri %', sublabel: 'Son 1 hafta', align: 'right' },
    { key: 'return_1m_pct', label: 'Getiri %', sublabel: 'Son 1 ay', align: 'right' },
    { key: 'return_3m_pct', label: 'Getiri %', sublabel: 'Son 3 ay', align: 'right' },
    { key: 'return_6m_pct', label: 'Getiri %', sublabel: 'Son 6 ay', align: 'right' },
    { key: 'return_ytd_pct', label: 'Getiri %', sublabel: 'YTA', align: 'right' },
    { key: 'return_1y_pct', label: 'Getiri %', sublabel: 'Son 1 yıl', align: 'right' },
];
const STOCK_INDEX_OPTIONS: MarketStockIndex[] = ['XU100', 'XU030'];
const RETURN_MODE_OPTIONS: Array<{ id: StockReturnMode; label: string }> = [
    { id: 'absolute', label: 'Mutlak' },
    { id: 'relative_xu100', label: "XU100'a göre" },
    { id: 'relative_xu030', label: "XU030'a göre" },
];
const RETURN_KEYS: StockReturnKey[] = [
    'return_1w_pct',
    'return_1m_pct',
    'return_3m_pct',
    'return_6m_pct',
    'return_ytd_pct',
    'return_1y_pct',
];
const INDEX_COLUMNS: Array<{ key: keyof MarketIndexListRow; label: string; align?: 'left' | 'right' }> = [
    { key: 'symbol', label: 'Endeks', align: 'left' },
    { key: 'price', label: 'Son Fiyat', align: 'right' },
    { key: 'change_pct', label: 'Gün %', align: 'right' },
    { key: 'volume', label: 'Hacim', align: 'right' },
    { key: 'return_1w_pct', label: '1 Hafta %', align: 'right' },
    { key: 'return_1m_pct', label: '1 Ay %', align: 'right' },
    { key: 'return_3m_pct', label: '3 Ay %', align: 'right' },
    { key: 'return_6m_pct', label: '6 Ay %', align: 'right' },
    { key: 'return_ytd_pct', label: 'YTA %', align: 'right' },
    { key: 'return_1y_pct', label: '1 Yıl %', align: 'right' },
];
const DETAIL_RETURN_KEYS: Array<{ key: keyof MarketIndexListRow; label: string }> = [
    { key: 'change_pct', label: 'Gün içi' },
    { key: 'return_1w_pct', label: '1 Hafta' },
    { key: 'return_1m_pct', label: '1 Ay' },
    { key: 'return_ytd_pct', label: 'YTA' },
    { key: 'return_6m_pct', label: '6 Ay' },
    { key: 'return_1y_pct', label: '1 Yıl' },
    { key: 'return_5y_pct', label: '5 Yıl' },
];

function getStatusMeta(row: MarketUniverseRow): { label: string; className: string; hint: string } {
    if (row.has_rag) {
        return {
            label: 'Analiz Hazır',
            className: 'cc-status-rag',
            hint: 'Detay ekranı ve soru-cevap bölümü kullanılabilir.',
        };
    }
    return {
        label: 'Finansal Görünüm',
        className: 'cc-status-kap',
        hint: 'Detay ekranı açılır, soru-cevap kapsamı daha sonra genişler.',
    };
}

function formatPrice(row: MarketUniverseRow): string {
    if (row.price == null) {
        return '-';
    }
    const currencyPrefix = row.price_currency && row.price_currency !== 'TRY' ? `${row.price_currency} ` : '₺';
    return `${currencyPrefix}${row.price.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatStockPrice(row: MarketStockRow): string {
    if (row.price == null) return '-';
    const currencyPrefix = row.price_currency && row.price_currency !== 'TRY' ? `${row.price_currency} ` : '₺';
    return `${currencyPrefix}${row.price.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatIndexPrice(value: number | null): string {
    if (value == null) return '-';
    return value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

function formatMaybeCurrency(value: number | null, currency?: string | null): string {
    if (value == null) return '-';
    const prefix = currency && currency !== 'TRY' ? `${currency} ` : '₺';
    return `${prefix}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatChangePct(changePct: number | null): string {
    if (changePct == null) {
        return 'Veri bekleniyor';
    }
    const sign = changePct > 0 ? '+' : '';
    return `% ${sign}${changePct.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatTablePct(value: number | null): string {
    if (value == null) return 'N/A';
    const sign = value > 0 ? '+' : '';
    return `% ${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatVolume(value: number | null): string {
    if (value == null) return '-';
    const abs = Math.abs(value);
    if (abs >= 1_000_000_000) {
        return `${(value / 1_000_000_000).toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })} mr`;
    }
    if (abs >= 1_000_000) {
        return `${(value / 1_000_000).toLocaleString('tr-TR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })} mn`;
    }
    return value.toLocaleString('tr-TR', { maximumFractionDigits: 0 });
}

function formatUpdateTime(iso: string | null | undefined): string {
    if (!iso) return '--:--';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '--:--';
    return dt.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatDateTime(iso: string | null | undefined): string {
    if (!iso) return '-';
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '-';
    return dt.toLocaleString('tr-TR', {
        day: '2-digit',
        month: 'long',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
}

function getPriceChangeClass(changePct: number | null): string {
    if (changePct == null || changePct === 0) {
        return 'cc-change-flat';
    }
    return changePct > 0 ? 'cc-change-up' : 'cc-change-down';
}

function getTableChangeClass(value: number | null): string {
    if (value == null || value === 0) return 'stocks-flat';
    return value > 0 ? 'stocks-up' : 'stocks-down';
}

function numericOrNull(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function indexCellValue(row: MarketIndexListRow, key: keyof MarketIndexListRow): string {
    if (key === 'symbol') return row.symbol;
    if (key === 'price') return formatIndexPrice(row.price);
    if (key === 'volume') return formatVolume(row.volume);
    if (String(key).includes('return') || key === 'change_pct') {
        return formatTablePct(numericOrNull(row[key]));
    }
    const value = row[key];
    return value == null ? '-' : String(value);
}

function constituentPrice(row: MarketIndexConstituent): string {
    return formatMaybeCurrency(row.price, row.price_currency);
}

function formatWeight(value: number | null): string {
    if (value == null) return '-';
    return `% ${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function formatPointEffect(value: number | null): string {
    if (value == null) return '-';
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toLocaleString('tr-TR', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function isReturnKey(key: StockSortKey): key is StockReturnKey {
    return RETURN_KEYS.includes(key as StockReturnKey);
}

function getBenchmarkIndex(returnMode: StockReturnMode): MarketStockIndex | null {
    if (returnMode === 'relative_xu100') return 'XU100';
    if (returnMode === 'relative_xu030') return 'XU030';
    return null;
}

function getReturnValue(
    row: MarketStockRow,
    key: StockReturnKey,
    benchmarks: Record<MarketStockIndex, MarketReturnBenchmark> | undefined,
    returnMode: StockReturnMode,
): number | null {
    const rawValue = row[key];
    const benchmarkIndex = getBenchmarkIndex(returnMode);
    if (!benchmarkIndex) return rawValue;

    const benchmarkValue = benchmarks?.[benchmarkIndex]?.[key];
    if (rawValue == null || benchmarkValue == null) return null;
    return Math.round((rawValue - benchmarkValue) * 100) / 100;
}

function getColumnLabel(key: StockSortKey, returnMode: StockReturnMode): string {
    if (!isReturnKey(key)) {
        return STOCK_COLUMNS.find((column) => column.key === key)?.label || '';
    }
    return returnMode === 'absolute' ? 'Getiri %' : 'Relatif %';
}

function stockSortValue(
    row: MarketStockRow,
    key: StockSortKey,
    benchmarks: Record<MarketStockIndex, MarketReturnBenchmark> | undefined,
    returnMode: StockReturnMode,
): string | number | null {
    if (key === 'company') return row.company;
    if (isReturnKey(key)) return getReturnValue(row, key, benchmarks, returnMode);
    return row[key];
}

function getLatestQuarterText(row: MarketUniverseRow): string {
    return row.latest_quarter || 'Detay ekranında yüklenecek';
}

function getFinancialReadinessText(row: MarketUniverseRow): string {
    return row.has_kap_cache ? 'Hazır' : 'İlk açılışta yüklenir';
}

function FlashCompanyCard({
    row,
    children,
    onClick,
}: React.PropsWithChildren<{ row: MarketUniverseRow; onClick: () => void }>) {
    const prevPriceRef = useRef(row.price);
    const [flashClass, setFlashClass] = useState('');

    useEffect(() => {
        if (row.price != null && prevPriceRef.current != null && row.price !== prevPriceRef.current) {
            setFlashClass(row.price > prevPriceRef.current ? 'cc-flash-up' : 'cc-flash-down');
            const timer = window.setTimeout(() => setFlashClass(''), 1100);
            prevPriceRef.current = row.price;
            return () => window.clearTimeout(timer);
        }
        prevPriceRef.current = row.price;
    }, [row.price]);

    return (
        <div className={`company-card ${flashClass}`} onClick={onClick}>
            {children}
        </div>
    );
}

function FlashStockRow({
    row,
    rank,
    children,
    onClick,
}: React.PropsWithChildren<{ row: MarketStockRow; rank: number; onClick: () => void }>) {
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

    return (
        <tr className={flashClass} onClick={onClick}>
            <td className="stocks-rank">{rank}</td>
            {children}
        </tr>
    );
}

function IndexLineChart({ points, prevClose }: { points: MarketIndexDetailResponse['line_points'], prevClose: number | null }) {
    const width = 1120;
    const height = 400;
    const padding = { top: 30, right: 65, bottom: 40, left: 16 };
    const validPoints = points.filter((point) => Number.isFinite(point.close));

    if (validPoints.length < 2) {
        return <div className="indices-chart-empty">Grafik verisi bekleniyor.</div>;
    }

    const values = validPoints.map((point) => point.close);
    if (prevClose != null) values.push(prevClose);
    
    // Add small visual padding around min/max values
    let minValue = Math.min(...values);
    let maxValue = Math.max(...values);
    const spanRaw = Math.max(1, maxValue - minValue);
    minValue -= spanRaw * 0.05;
    maxValue += spanRaw * 0.05;
    const span = maxValue - minValue;

    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const xFor = (index: number) =>
        padding.left + (validPoints.length === 1 ? 0 : (index / (validPoints.length - 1)) * plotWidth);
    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;

    const last = validPoints[validPoints.length - 1].close;

    // Y ekseni çizgileri (5 adet)
    const tickCount = 6;
    const tickValues = Array.from({ length: tickCount }).map((_, i) => minValue + (span * i) / (tickCount - 1));

    // X ekseni (saat başları için tahmini çizim veya eşit dağılımlı)
    const timeTickCount = 8;
    const timeTicks = Array.from({ length: timeTickCount }).map((_, i) => Math.floor((validPoints.length - 1) * (i / (timeTickCount - 1))));

    // Çizgi ve Gradient Rengi (Referans resmindeki gibi soldan sağa Mavi -> Mor geçişi)
    const strokeLeft = '#3b82f6';  // Parlak mavi
    const strokeRight = '#c084fc'; // Parlak mor

    const pathData = validPoints
        .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xFor(index)} ${yFor(point.close)}`)
        .join(' ');
        
    const areaData = `${pathData} L ${xFor(validPoints.length - 1)} ${height - padding.bottom} L ${padding.left} ${height - padding.bottom} Z`;

    return (
        <div style={{ backgroundColor: '#0f1214', borderRadius: '8px', border: '1px solid #1e2327', position: 'relative', overflow: 'hidden' }}>
        <svg className="indices-line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Endeks çizgi grafiği" style={{ display: 'block', width: '100%', height: 'auto', borderBottom: 'none' }}>
            <defs>
                <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor={strokeLeft} />
                    <stop offset="100%" stopColor={strokeRight} />
                </linearGradient>
                <linearGradient id="areaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor={strokeRight} stopOpacity="0.25" />
                    <stop offset="100%" stopColor={strokeLeft} stopOpacity="0.0" />
                </linearGradient>
            </defs>
            {/* Koyu Arkaplan (Uygulamanın karanlık temasına uygun) */}
            <rect x="0" y="0" width={width} height={height} fill="#0d1113" rx="4" />
            
            {/* Yatay Grid ve Değerler */}
            {tickValues.map((value) => (
                <g key={value}>
                    <line
                        x1={padding.left}
                        x2={width - padding.right}
                        y1={yFor(value)}
                        y2={yFor(value)}
                        stroke="rgba(255,255,255,0.05)"
                        strokeWidth="1"
                    />
                    <text x={width - padding.right + 10} y={yFor(value) + 4} fill="rgba(255,255,255,0.4)" fontSize="11" fontFamily="monospace">
                        {formatIndexPrice(value)}
                    </text>
                </g>
            ))}

            {/* Önceki Kapanış Referans Çizgisi */}
            {prevClose != null && (
                <line
                    x1={padding.left}
                    x2={width - padding.right}
                    y1={yFor(prevClose)}
                    y2={yFor(prevClose)}
                    stroke="rgba(180, 180, 180, 0.3)"
                    strokeWidth="1"
                    strokeDasharray="4 4"
                />
            )}

            {/* Dikey Grid Zaman Etiketleri */}
            {timeTicks.map((index) => {
                const dt = new Date(validPoints[index].time);
                const label = Number.isNaN(dt.getTime())
                    ? ''
                    : dt.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
                const x = xFor(index);
                return (
                    <g key={index}>
                        <line
                            x1={x}
                            x2={x}
                            y1={padding.top}
                            y2={height - padding.bottom}
                            stroke="rgba(255,255,255,0.03)"
                            strokeWidth="1"
                        />
                        <text x={x} y={height - 15} fill="rgba(255,255,255,0.5)" fontSize="11" fontFamily="monospace" textAnchor="middle">
                            {label}
                        </text>
                    </g>
                );
            })}

            {/* Alan ve Çizgi (Ağ/Line) (Açılış, yüksek, düşük kullanılmıyor, SADECE CLOSE) */}
            <path d={areaData} fill="url(#areaGrad)" />
            <path d={pathData} fill="none" stroke="url(#lineGrad)" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
            
            {/* Uç Noktası (Kapanış) Dot */}
            <circle cx={xFor(validPoints.length - 1)} cy={yFor(last)} r="4" fill={strokeRight} />

            {/* Anlık Fiyat İşareti */}
            <g transform={`translate(${width - padding.right}, ${yFor(last)})`}>
                <rect x="0" y="-10" width="65" height="20" fill={strokeRight} rx="2" />
                <path d="M 0 0 L 6 -6 L 6 6 Z" fill={strokeRight} transform="translate(-5, 0)" />
                <text x="32" y="3" fill="#ffffff" fontSize="11" fontFamily="monospace" textAnchor="middle" fontWeight="bold">
                    {formatIndexPrice(last)}
                </text>
            </g>
        </svg>
        </div>
    );
}

export default function MarketsView() {
    const [market, setMarket] = useState<MarketUniverseResponse | null>(null);
    const [stocks, setStocks] = useState<MarketStocksResponse | null>(null);
    const [indices, setIndices] = useState<MarketIndicesResponse | null>(null);
    const [indexDetail, setIndexDetail] = useState<MarketIndexDetailResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [stocksLoading, setStocksLoading] = useState(false);
    const [stocksError, setStocksError] = useState<string | null>(null);
    const [indicesLoading, setIndicesLoading] = useState(false);
    const [indicesError, setIndicesError] = useState<string | null>(null);
    const [indexDetailLoading, setIndexDetailLoading] = useState(false);
    const [indexDetailError, setIndexDetailError] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [navCollapsed, setNavCollapsed] = useState(false);
    const [activeSection, setActiveSection] = useState<MarketSection>('markets');
    const [selectedIndex, setSelectedIndex] = useState<MarketIndexCode | null>(null);
    const [stockIndex, setStockIndex] = useState<MarketStockIndex>('XU100');
    const [returnMode, setReturnMode] = useState<StockReturnMode>('absolute');
    const [stockSort, setStockSort] = useState<{ key: StockSortKey; direction: SortDirection }>({
        key: 'company',
        direction: 'asc',
    });
    const [indexSort, setIndexSort] = useState<{ key: IndexSortKey; direction: SortDirection }>({
        key: 'symbol',
        direction: 'asc',
    });
    const stocksInFlightRef = useRef(false);
    const indicesInFlightRef = useRef(false);
    const indexDetailInFlightRef = useRef(false);
    const latestStockIndexRef = useRef<MarketStockIndex>(stockIndex);
    const latestSelectedIndexRef = useRef<MarketIndexCode | null>(selectedIndex);

    useEffect(() => {
        latestStockIndexRef.current = stockIndex;
    }, [stockIndex]);

    useEffect(() => {
        latestSelectedIndexRef.current = selectedIndex;
    }, [selectedIndex]);

    useEffect(() => { 
        loadStats(); 
        // XU100 ve pazar özetleri için her 10 saniyede bir arka planda sessizce çek
        const intervalId = window.setInterval(() => {
            loadStats(true);
        }, 10000);
        return () => window.clearInterval(intervalId);
    }, []);

    useEffect(() => {
        if (activeSection !== 'stocks') return;
        setStocks(null);
        setStocksError(null);
        loadStocks(false, true, stockIndex);
        const intervalId = window.setInterval(() => {
            loadStocks(true, true, stockIndex);
        }, 3000);
        return () => window.clearInterval(intervalId);
    }, [activeSection, stockIndex]);

    useEffect(() => {
        if (activeSection !== 'indices') return;
        loadIndices(false, true);
        const intervalId = window.setInterval(() => {
            loadIndices(true, true);
        }, 60000);
        return () => window.clearInterval(intervalId);
    }, [activeSection]);

    useEffect(() => {
        if (activeSection !== 'indices' || !selectedIndex) return;
        setIndexDetail(null);
        setIndexDetailError(null);
        loadIndexDetail(false, true, selectedIndex);
        const intervalId = window.setInterval(() => {
            loadIndexDetail(true, true, selectedIndex);
        }, 60000);
        return () => window.clearInterval(intervalId);
    }, [activeSection, selectedIndex]);

    async function loadStats(silent = false) {
        if (!silent) setLoading(true);
        if (!silent) setError(null);
        try {
            const marketPayload = await apiClient.marketUniverse();
            setMarket(marketPayload);
        } catch (err: any) {
            if (!silent) setError(err.message || 'Veriler yüklenemedi.');
        } finally {
            if (!silent) setLoading(false);
        }
    }

    async function loadStocks(silent = false, refresh = false, requestedIndex: MarketStockIndex = stockIndex) {
        if (stocksInFlightRef.current) return;
        stocksInFlightRef.current = true;
        if (!silent) setStocksLoading(true);
        if (!silent) setStocksError(null);
        try {
            const stocksPayload = await apiClient.marketStocks({ index: requestedIndex, refresh });
            if (latestStockIndexRef.current !== requestedIndex) return;
            setStocks(stocksPayload);
            setStocksError(null);
        } catch (err: any) {
            if (!silent || !stocks) {
                setStocksError(err.message || 'Hisse verileri yüklenemedi.');
            }
        } finally {
            stocksInFlightRef.current = false;
            if (!silent) setStocksLoading(false);
        }
    }

    async function loadIndices(silent = false, refresh = false) {
        if (indicesInFlightRef.current) return;
        indicesInFlightRef.current = true;
        if (!silent) setIndicesLoading(true);
        if (!silent) setIndicesError(null);
        try {
            const payload = await apiClient.marketIndices({ refresh });
            setIndices(payload);
            setIndicesError(null);
        } catch (err: any) {
            if (!silent || !indices) {
                setIndicesError(err.message || 'Endeks verileri yüklenemedi.');
            }
        } finally {
            indicesInFlightRef.current = false;
            if (!silent) setIndicesLoading(false);
        }
    }

    async function loadIndexDetail(
        silent = false,
        refresh = false,
        requestedIndex: MarketIndexCode = selectedIndex || 'XU100',
    ) {
        if (indexDetailInFlightRef.current) return;
        indexDetailInFlightRef.current = true;
        if (!silent) setIndexDetailLoading(true);
        if (!silent) setIndexDetailError(null);
        try {
            const payload = await apiClient.marketIndexDetail(requestedIndex, { refresh });
            if (latestSelectedIndexRef.current !== requestedIndex) return;
            setIndexDetail(payload);
            setIndexDetailError(null);
        } catch (err: any) {
            if (!silent || !indexDetail) {
                setIndexDetailError(err.message || 'Endeks detayı yüklenemedi.');
            }
        } finally {
            indexDetailInFlightRef.current = false;
            if (!silent) setIndexDetailLoading(false);
        }
    }

    const normalizedSearch = searchTerm.trim().toLowerCase();
    const filteredCompanies = useMemo(
        () =>
            (market?.rows || []).filter((row) =>
                row.company.toLowerCase().includes(normalizedSearch),
            ),
        [market?.rows, normalizedSearch],
    );

    const filteredStocks = useMemo(
        () =>
            (stocks?.rows || []).filter((row) =>
                row.company.toLowerCase().includes(normalizedSearch),
            ),
        [stocks?.rows, normalizedSearch],
    );

    const filteredIndices = useMemo(
        () =>
            (indices?.rows || []).filter((row) => {
                const query = normalizedSearch;
                if (!query) return true;
                return (
                    row.symbol.toLowerCase().includes(query) ||
                    row.label.toLowerCase().includes(query)
                );
            }),
        [indices?.rows, normalizedSearch],
    );

    const sortedIndices = useMemo(() => {
        const arr = [...filteredIndices];
        arr.sort((a, b) => {
            const av = a[indexSort.key];
            const bv = b[indexSort.key];
            const aMissing = av == null || av === '';
            const bMissing = bv == null || bv === '';

            if (aMissing && bMissing) return String(a.symbol).localeCompare(String(b.symbol), 'tr');
            if (aMissing) return 1;
            if (bMissing) return -1;

            let result = 0;
            if (typeof av === 'string' || typeof bv === 'string') {
                result = String(av).localeCompare(String(bv), 'tr');
            } else {
                result = Number(av) - Number(bv);
            }
            if (result === 0) result = String(a.symbol).localeCompare(String(b.symbol), 'tr');
            return indexSort.direction === 'asc' ? result : -result;
        });
        return arr;
    }, [filteredIndices, indexSort]);

    const sortedStocks = useMemo(() => {
        const arr = [...filteredStocks];
        arr.sort((a, b) => {
            const av = stockSortValue(a, stockSort.key, stocks?.benchmarks, returnMode);
            const bv = stockSortValue(b, stockSort.key, stocks?.benchmarks, returnMode);
            const aMissing = av == null || av === '';
            const bMissing = bv == null || bv === '';
            if (aMissing && bMissing) return a.company.localeCompare(b.company, 'tr');
            if (aMissing) return 1;
            if (bMissing) return -1;

            let result = 0;
            if (typeof av === 'string' || typeof bv === 'string') {
                result = String(av).localeCompare(String(bv), 'tr');
            } else {
                result = Number(av) - Number(bv);
            }
            if (result === 0) result = a.company.localeCompare(b.company, 'tr');
            return stockSort.direction === 'asc' ? result : -result;
        });
        return arr;
    }, [filteredStocks, stockSort, stocks?.benchmarks, returnMode]);

    const coverageRows = market?.coverage_rows || [];
    const maxCoverageQuarterCount = Math.max(1, ...coverageRows.map((row) => row.quarter_count));
    const activeBenchmarkIndex = getBenchmarkIndex(returnMode);
    const activeReturnModeLabel = RETURN_MODE_OPTIONS.find((option) => option.id === returnMode)?.label || 'Mutlak';
    const indexConstituents = indexDetail?.constituents || [];
    const positiveConstituents = indexConstituents.filter((row) => (row.change_pct || 0) > 0).length;
    const negativeConstituents = indexConstituents.filter((row) => (row.change_pct || 0) < 0).length;
    const neutralConstituents = indexConstituents.length - positiveConstituents - negativeConstituents;
    const weightedConstituents = indexConstituents.filter((row) => row.weight_pct != null && row.weight_pct > 0);
    const pageTitle = activeSection === 'indices' ? 'Borsa İstanbul Endeksleri' : 'Piyasa Görünümü';
    const pageDescription =
        activeSection === 'indices'
            ? 'XU100 ve XU030 endekslerini, getirileri ve endeks içi şirket hareketlerini takip edin.'
            : 'Güncel fiyatlar, finansal görünüm ve analiz erişimi tek ekranda.';

    const onCompanyClick = (ticker: string) => {
        window.location.href = `/?ticker=${ticker}`;
    };

    const handleSectionChange = (section: MarketSection) => {
        setActiveSection(section);
        setSelectedIndex(null);
    };

    const handleStockSort = (key: StockSortKey) => {
        setStockSort((prev) => {
            if (prev.key === key) {
                return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
            }
            return { key, direction: key === 'company' ? 'asc' : 'desc' };
        });
    };

    const handleIndexSort = (key: IndexSortKey) => {
        setIndexSort((prev) => {
            if (prev.key === key) {
                return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
            }
            return { key, direction: key === 'symbol' ? 'asc' : 'desc' };
        });
    };

    const renderReturnCell = (row: MarketStockRow, key: StockReturnKey) => {
        const value = getReturnValue(row, key, stocks?.benchmarks, returnMode);
        return (
            <td className={`stocks-cell-right ${getTableChangeClass(value)}`}>
                {formatTablePct(value)}
            </td>
        );
    };

    return (
        <div className={`mn-layout${navCollapsed ? ' mn-nav-collapsed' : ''}`}>
            <MarketsNavigation
                collapsed={navCollapsed}
                activeSection={activeSection}
                onCollapsedChange={setNavCollapsed}
                onSectionChange={handleSectionChange}
            />
            <div className="market-page">
                {market && (
                    <MarketSidebar rows={market.rows} onSelectTicker={onCompanyClick} />
                )}
            <header className="market-header">
                <div className="market-title">
                    <h1>{pageTitle}</h1>
                    <p>{pageDescription}</p>
                </div>
                
                {activeSection === 'markets' && market && (
                    <div className="market-quick-stats">
                        <div className="quick-stat">
                            <span className="qs-label">BIST100</span>
                            <span className="qs-value">{market.stats.bist100_count}</span>
                        </div>
                        <div className="quick-stat">
                            <span className="qs-label">Analiz Hazır</span>
                            <span className="qs-value">{market.stats.rag_ready_count}</span>
                        </div>
                        <div className="quick-stat">
                            <span className="qs-label">Finansal Görünüm</span>
                            <span className="qs-value">{market.stats.kap_only_count}</span>
                        </div>
                        <div className="quick-stat">
                            <span className="qs-label">Rapor</span>
                            <span className="qs-value">{market.stats.pdf_count}</span>
                        </div>
                        <div className="quick-stat">
                            <span className="qs-label">Sayfa</span>
                            <span className="qs-value">{market.stats.page_count}</span>
                        </div>
                    </div>
                )}
            </header>

            {activeSection === 'markets' && (
                <div className="market-watch-slot">
                    <MarketWatchStrip />
                </div>
            )}

            {activeSection === 'markets' && loading && <div className="loading-state"><div className="spinner"/> Yükleniyor...</div>}
            
            {activeSection === 'markets' && error && (
                <div className="error-message">
                    <strong>Hata:</strong> {error}
                    <button onClick={() => loadStats()} className="btn-secondary" style={{ marginLeft: '1rem' }}>Tekrar Dene</button>
                </div>
            )}

            {activeSection === 'markets' && !loading && !error && market && (
                <div className="market-content">
                    <div className="market-main-column">
                        <div className="panel company-search-panel">
                            <div className="panel-header">
                                <h2>Hisse Seçimi</h2>
                                <div className="search-box">
                                    <input 
                                        type="text" 
                                        placeholder="Hisse kodu ile ara..." 
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                        className="input-field"
                                    />
                                </div>
                            </div>
                            
                            <div className="company-grid">
                                {filteredCompanies.length > 0 ? (
                                    filteredCompanies.map((row) => {
                                        const status = getStatusMeta(row);
                                        return (
                                            <FlashCompanyCard
                                                key={row.company} 
                                                row={row}
                                                onClick={() => onCompanyClick(row.company)}
                                            >
                                                <div className="cc-header">
                                                    <div className="cc-symbol-row">
                                                        <h3>{row.company}</h3>
                                                        <span className={`cc-status ${status.className}`}>{status.label}</span>
                                                    </div>
                                                    <div className="cc-price-row">
                                                        <span className="cc-price">{formatPrice(row)}</span>
                                                        <span className={`cc-change ${getPriceChangeClass(row.change_pct)}`}>
                                                            {formatChangePct(row.change_pct)}
                                                        </span>
                                                    </div>
                                                </div>
                                                <div className="cc-body">
                                                    <div className="cc-metric">
                                                        <span className="cc-label">Son Rapor</span>
                                                        <span className="cc-value">
                                                            {getLatestQuarterText(row)}
                                                        </span>
                                                    </div>
                                                    <div className="cc-metric">
                                                        <span className="cc-label">Analiz Kapsamı</span>
                                                        <span className={`cc-value ${row.has_rag ? '' : 'empty'}`}>
                                                            {row.has_rag ? `${row.quarter_count} çeyrek` : 'Finansal görünüm'}
                                                        </span>
                                                    </div>
                                                    <div className="cc-metric">
                                                        <span className="cc-label">Finansal Veri</span>
                                                        <span className={`cc-value ${row.has_kap_cache ? '' : 'empty'}`}>
                                                            {getFinancialReadinessText(row)}
                                                        </span>
                                                    </div>
                                                </div>
                                                <div className="cc-foot">{status.hint}</div>
                                            </FlashCompanyCard>
                                        );
                                    })
                                ) : (
                                    <div className="no-results">Aranan kritere uyan BIST100 hissesi bulunamadı.</div>
                                )}
                            </div>
                        </div>
                    </div>
                    
                    <div className="market-side-column">
                        {coverageRows.length > 0 && (
                            <div className="panel coverage-panel">
                                <div className="panel-header">
                                    <h3>Analiz Kapsamı</h3>
                                    <span className="panel-kicker">Belge kapsamı en güçlü hisseler</span>
                                </div>
                                <div className="coverage-list">
                                    {coverageRows.map((row) => (
                                        <div key={row.company} className="coverage-item" onClick={() => onCompanyClick(row.company)}>
                                            <span className="ci-company">{row.company}</span>
                                            <div className="ci-bar-container">
                                                <div
                                                    className="ci-bar"
                                                    style={{ width: `${Math.max(12, (row.quarter_count / maxCoverageQuarterCount) * 100)}%` }}
                                                />
                                            </div>
                                            <span className="ci-value">{row.quarter_count} Çyrk</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {activeSection === 'stocks' && (
                <div className="stocks-view">
                    <div className="panel stocks-table-panel">
                        <div className="panel-header stocks-panel-header">
                            <div>
                                <h2>Hisseler</h2>
                                <span className="panel-kicker">
                                    {stocks?.index || stockIndex} ·{' '}
                                    {sortedStocks.length || stocks?.rows.length || 0} hisse
                                    {stocksLoading && stocks ? ' · güncelleniyor' : ''}
                                </span>
                            </div>
                            <div className="stocks-panel-actions">
                                <div className="stocks-segment" aria-label="Endeks seçimi">
                                    {STOCK_INDEX_OPTIONS.map((option) => (
                                        <button
                                            key={option}
                                            type="button"
                                            className={stockIndex === option ? 'active' : ''}
                                            aria-pressed={stockIndex === option}
                                            onClick={() => setStockIndex(option)}
                                        >
                                            {option}
                                        </button>
                                    ))}
                                </div>
                                <div className="stocks-segment stocks-return-segment" aria-label="Getiri modu">
                                    {RETURN_MODE_OPTIONS.map((option) => (
                                        <button
                                            key={option.id}
                                            type="button"
                                            className={returnMode === option.id ? 'active' : ''}
                                            aria-pressed={returnMode === option.id}
                                            onClick={() => setReturnMode(option.id)}
                                        >
                                            {option.label}
                                        </button>
                                    ))}
                                </div>
                                <div className="search-box">
                                    <input
                                        type="text"
                                        placeholder="Hisse kodu ile ara..."
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                        className="input-field"
                                    />
                                </div>
                            </div>
                        </div>

                        {stocksLoading && !stocks && (
                            <div className="loading-state"><div className="spinner" /> Hisseler yükleniyor...</div>
                        )}

                        {stocksError && !stocks && (
                            <div className="error-message">
                                <strong>Hata:</strong> {stocksError}
                                <button
                                    onClick={() => loadStocks(false, true, stockIndex)}
                                    className="btn-secondary"
                                    style={{ marginLeft: '1rem' }}
                                >
                                    Tekrar Dene
                                </button>
                            </div>
                        )}

                        {stocks && (
                            <>
                                <div className="stocks-table-meta">
                                    <span>Son güncelleme: {formatUpdateTime(stocks.as_of)}</span>
                                    <span>Getiri: {activeReturnModeLabel}</span>
                                    {activeBenchmarkIndex && (
                                        <span>
                                            Benchmark: {activeBenchmarkIndex} ·{' '}
                                            {formatUpdateTime(stocks.benchmarks?.[activeBenchmarkIndex]?.as_of)}
                                        </span>
                                    )}
                                    {stocksError && <span className="stocks-soft-error">Son yenileme başarısız oldu.</span>}
                                </div>

                                {sortedStocks.length === 0 ? (
                                    <div className="no-results">Aranan kritere uyan hisse bulunamadı.</div>
                                ) : (
                                    <div className="stocks-table-wrap">
                                        <table className="stocks-table">
                                            <thead>
                                                <tr>
                                                    <th className="stocks-rank">#</th>
                                                    {STOCK_COLUMNS.map((column) => (
                                                        <th
                                                            key={column.key}
                                                            className={column.align === 'right' ? 'stocks-cell-right' : undefined}
                                                        >
                                                            <button
                                                                type="button"
                                                                className="stocks-sort-button"
                                                                onClick={() => handleStockSort(column.key)}
                                                                aria-sort={
                                                                    stockSort.key === column.key
                                                                        ? stockSort.direction === 'asc'
                                                                            ? 'ascending'
                                                                            : 'descending'
                                                                        : 'none'
                                                                }
                                                            >
                                                                <span>{getColumnLabel(column.key, returnMode)}</span>
                                                                {column.sublabel && <small>{column.sublabel}</small>}
                                                                {stockSort.key === column.key && (
                                                                    <span className="stocks-sort-indicator">
                                                                        {stockSort.direction === 'asc' ? '↑' : '↓'}
                                                                    </span>
                                                                )}
                                                            </button>
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {sortedStocks.map((row, index) => (
                                                    <FlashStockRow
                                                        key={row.company}
                                                        row={row}
                                                        rank={index + 1}
                                                        onClick={() => onCompanyClick(row.company)}
                                                    >
                                                        <td className="stocks-symbol-cell">{row.company}</td>
                                                        <td className="stocks-cell-right stocks-price-cell">{formatStockPrice(row)}</td>
                                                        <td className={`stocks-cell-right ${getTableChangeClass(row.change_pct)}`}>
                                                            {formatTablePct(row.change_pct)}
                                                        </td>
                                                        <td className="stocks-cell-right stocks-volume-cell">{formatVolume(row.volume)}</td>
                                                        {renderReturnCell(row, 'return_1w_pct')}
                                                        {renderReturnCell(row, 'return_1m_pct')}
                                                        {renderReturnCell(row, 'return_3m_pct')}
                                                        {renderReturnCell(row, 'return_6m_pct')}
                                                        {renderReturnCell(row, 'return_ytd_pct')}
                                                        {renderReturnCell(row, 'return_1y_pct')}
                                                    </FlashStockRow>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>
            )}

            {activeSection === 'indices' && !selectedIndex && (
                <div className="indices-view">
                    <div className="panel stocks-table-panel">
                        <div className="panel-header stocks-panel-header">
                            <div>
                                <h2>Endeksler</h2>
                                <span className="panel-kicker">
                                    {filteredIndices.length || indices?.rows.length || 0} endeks
                                    {indicesLoading && indices ? ' · güncelleniyor' : ''}
                                </span>
                            </div>
                            <div className="stocks-panel-actions">
                                <div className="search-box">
                                    <input
                                        type="text"
                                        placeholder="Endeks kodu ile ara..."
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                        className="input-field"
                                    />
                                </div>
                            </div>
                        </div>

                        {indicesLoading && !indices && (
                            <div className="loading-state"><div className="spinner" /> Endeksler yükleniyor...</div>
                        )}

                        {indicesError && !indices && (
                            <div className="error-message">
                                <strong>Hata:</strong> {indicesError}
                                <button
                                    onClick={() => loadIndices(false, true)}
                                    className="btn-secondary"
                                    style={{ marginLeft: '1rem' }}
                                >
                                    Tekrar Dene
                                </button>
                            </div>
                        )}

                        {indices && (
                            <>
                                <div className="stocks-table-meta">
                                    <span>Son güncelleme: {formatUpdateTime(indices.as_of)}</span>
                                    {indicesError && <span className="stocks-soft-error">Son yenileme başarısız oldu.</span>}
                                </div>
                                {filteredIndices.length === 0 ? (
                                    <div className="no-results">Aranan kritere uyan endeks bulunamadı.</div>
                                ) : (
                                    <div className="stocks-table-wrap indices-table-wrap">
                                        <table className="stocks-table indices-table">
                                            <thead>
                                                <tr>
                                                    <th className="stocks-rank">#</th>
                                                    {INDEX_COLUMNS.map((column) => (
                                                        <th
                                                            key={column.key}
                                                            className={column.align === 'right' ? 'stocks-cell-right' : undefined}
                                                        >
                                                            <button
                                                                type="button"
                                                                className="stocks-sort-button"
                                                                onClick={() => handleIndexSort(column.key)}
                                                                aria-sort={
                                                                    indexSort.key === column.key
                                                                        ? indexSort.direction === 'asc'
                                                                            ? 'ascending'
                                                                            : 'descending'
                                                                        : 'none'
                                                                }
                                                            >
                                                                <span>{column.label}</span>
                                                                {indexSort.key === column.key && (
                                                                    <span className="stocks-sort-indicator">
                                                                        {indexSort.direction === 'asc' ? '↑' : '↓'}
                                                                    </span>
                                                                )}
                                                            </button>
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {sortedIndices.map((row, index) => (
                                                    <tr
                                                        key={row.symbol}
                                                        onClick={() => setSelectedIndex(row.symbol)}
                                                    >
                                                        <td className="stocks-rank">{index + 1}</td>
                                                        {INDEX_COLUMNS.map((column) => {
                                                            const value = numericOrNull(row[column.key]);
                                                            const isPct =
                                                                column.key === 'change_pct' ||
                                                                String(column.key).startsWith('return_');
                                                            return (
                                                                <td
                                                                    key={column.key}
                                                                    className={[
                                                                        column.align === 'right' ? 'stocks-cell-right' : '',
                                                                        isPct ? getTableChangeClass(value) : '',
                                                                        column.key === 'symbol' ? 'stocks-symbol-cell' : '',
                                                                        column.key === 'price' ? 'stocks-price-cell' : '',
                                                                    ].join(' ')}
                                                                >
                                                                    {column.key === 'symbol' ? (
                                                                        <span className="indices-symbol-main">
                                                                            <span>{row.symbol}</span>
                                                                            <small>{row.label}</small>
                                                                        </span>
                                                                    ) : (
                                                                        indexCellValue(row, column.key)
                                                                    )}
                                                                </td>
                                                            );
                                                        })}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>
            )}

            {activeSection === 'indices' && selectedIndex && (
                <div className="indices-detail-view">
                    <div className="indices-breadcrumb">
                        <button type="button" onClick={() => setSelectedIndex(null)}>Endeksler</button>
                        <span>/</span>
                        <strong>{selectedIndex}</strong>
                    </div>

                    {indexDetailLoading && !indexDetail && (
                        <div className="loading-state"><div className="spinner" /> Endeks detayı yükleniyor...</div>
                    )}

                    {indexDetailError && !indexDetail && (
                        <div className="error-message">
                            <strong>Hata:</strong> {indexDetailError}
                            <button
                                onClick={() => loadIndexDetail(false, true, selectedIndex)}
                                className="btn-secondary"
                                style={{ marginLeft: '1rem' }}
                            >
                                Tekrar Dene
                            </button>
                        </div>
                    )}

                    {indexDetail && (
                        <>
                            <section className="indices-hero">
                                <div className="indices-hero-main">
                                    <div className="indices-logo">{indexDetail.symbol.slice(-3)}</div>
                                    <div>
                                        <h2>{indexDetail.symbol}</h2>
                                        <p>{indexDetail.label}</p>
                                    </div>
                                </div>
                                <div className="indices-hero-price">
                                    <strong>{formatIndexPrice(indexDetail.price)}</strong>
                                    <span className={getTableChangeClass(indexDetail.change_pct)}>
                                        {formatTablePct(indexDetail.change_pct)}
                                    </span>
                                    <small>{formatDateTime(indexDetail.as_of)}</small>
                                </div>
                            </section>

                            <div className="indices-stat-row">
                                <span>Yüksek: <strong>{formatIndexPrice(indexDetail.high)}</strong></span>
                                <span>Düşük: <strong>{formatIndexPrice(indexDetail.low)}</strong></span>
                                <span>Hacim: <strong>{formatVolume(indexDetail.volume)}</strong></span>
                                <span>Önc.Kap.: <strong>{formatIndexPrice(indexDetail.prev_close)}</strong></span>
                            </div>

                            <div className="indices-return-strip">
                                {DETAIL_RETURN_KEYS.map((item) => {
                                    const value = numericOrNull(indexDetail[item.key]);
                                    return (
                                        <div key={item.key} className="indices-return-card">
                                            <span>{item.label}</span>
                                            <strong className={getTableChangeClass(value)}>{formatTablePct(value)}</strong>
                                        </div>
                                    );
                                })}
                            </div>

                            <section className="indices-chart-panel">
                                <IndexLineChart points={indexDetail.line_points} prevClose={indexDetail.prev_close} />
                            </section>

                            <section className="indices-impact-panel">
                                <div className="indices-impact-head">
                                    <div>
                                        <h3>{indexDetail.symbol} Endeksindeki Şirketler ve Etkileri</h3>
                                        <span className="panel-kicker">
                                            {indexDetail.constituents.length} şirket · {indexDetail.weight_status === 'available' ? 'ağırlık hesaplandı' : 'ağırlık bekleniyor'}
                                        </span>
                                    </div>
                                    {indexDetailError && <span className="stocks-soft-error">Son yenileme başarısız oldu.</span>}
                                </div>

                                {indexDetail.weight_status !== 'available' && (
                                    <div className="indices-weight-note">{indexDetail.weight_note}</div>
                                )}

                                {indexDetail.weight_status === 'available' && (
                                    <div className="indices-treemap">
                                        {weightedConstituents.map((row) => (
                                            <div
                                                key={row.symbol}
                                                className={`indices-tree-tile ${getTableChangeClass(row.change_pct)}`}
                                                style={{ flexBasis: `${Math.max(7, Math.min(38, row.weight_pct || 0))}%` }}
                                                title={`${row.symbol} · ${formatWeight(row.weight_pct)} · ${formatPointEffect(row.point_effect)} puan`}
                                            >
                                                <strong>{row.symbol}</strong>
                                                <span>{formatPointEffect(row.point_effect)}</span>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                <div className="indices-impact-summary">
                                    <span className="stocks-up">{positiveConstituents} pozitif</span>
                                    <span className="stocks-flat">{neutralConstituents} nötr</span>
                                    <span className="stocks-down">{negativeConstituents} negatif</span>
                                </div>

                                <div className="stocks-table-wrap indices-constituent-wrap">
                                    <table className="stocks-table indices-constituent-table">
                                        <thead>
                                            <tr>
                                                <th>Şirket</th>
                                                <th className="stocks-cell-right">Son Fiyat</th>
                                                <th className="stocks-cell-right">%</th>
                                                <th className="stocks-cell-right">Hacim</th>
                                                <th className="stocks-cell-right">Endeks Ağırlığı</th>
                                                <th className="stocks-cell-right">Puan Etkisi</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {indexConstituents.map((row) => (
                                                <tr key={row.symbol} onClick={() => onCompanyClick(row.symbol)}>
                                                    <td className="stocks-symbol-cell">{row.symbol}</td>
                                                    <td className="stocks-cell-right stocks-price-cell">{constituentPrice(row)}</td>
                                                    <td className={`stocks-cell-right ${getTableChangeClass(row.change_pct)}`}>
                                                        {formatTablePct(row.change_pct)}
                                                    </td>
                                                    <td className="stocks-cell-right stocks-volume-cell">{formatVolume(row.volume)}</td>
                                                    <td className="stocks-cell-right">{formatWeight(row.weight_pct)}</td>
                                                    <td className={`stocks-cell-right ${getTableChangeClass(row.point_effect)}`}>
                                                        {formatPointEffect(row.point_effect)}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </section>
                        </>
                    )}
                </div>
            )}
            </div>
        </div>
    );
}
