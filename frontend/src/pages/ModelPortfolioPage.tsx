import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';
import {
    AlertTriangle,
    ArrowUpRight,
    Award,
    BriefcaseBusiness,
    Check,
    CheckCircle2,
    ChevronDown,
    ChevronRight,
    Clock,
    ExternalLink,
    History,
    KeyRound,
    Layers,
    Lock,
    LogOut,
    Pencil,
    Plus,
    RefreshCw,
    Search,
    SlidersHorizontal,
    Sparkles,
    Target,
    Trash2,
    TrendingUp,
    X,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../api/client';
import type {
    MarketStockRow,
    MarketStocksResponse,
    MarketUniverseRow,
    ModelPortfolioChangelogItem,
    ModelPortfolioPerformancePoint,
    ModelPortfolioPosition,
    ModelPortfolioResponse,
} from '../api/types';
import MarketsNavigation, { type MarketsNavigationSection } from '../components/MarketsNavigation';
import MarketWatchRail from '../components/MarketWatchRail';
import SymbolLogo from '../components/SymbolLogo';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import {
    DEFAULT_FUND_TAB,
    DEFAULT_MARKET_INDEX,
    DEFAULT_STOCK_TAB,
    toFundComparison,
    toFundDetail,
    toFunds,
    toMarketsIndices,
    toMarketsOverview,
    toMarketsStocks,
    toStockDetail,
} from '../routing/routes';
import './MarketsView.css';
import './ModelPortfolioPage.css';

type PortfolioTab = 'active' | 'closed' | 'changelog';
type PerfRange = '1m' | '3m' | '6m' | 'ytd' | '1y' | 'all';
type PerformanceSeries = 'portfolio' | 'xu100' | 'xu030';

type ExplainedOption = {
    value: string;
    label: string;
    badge: string;
    tone: 'emerald' | 'blue' | 'amber' | 'purple';
    description: string;
};

const PERF_RANGES: Array<{ id: PerfRange; label: string }> = [
    { id: '1m', label: '1A' },
    { id: '3m', label: '3A' },
    { id: '6m', label: '6A' },
    { id: 'ytd', label: 'YTD' },
    { id: '1y', label: '1Y' },
    { id: 'all', label: 'Tüm Veri' },
];

function quoteTimestampAtLeastAsRecent(candidate: string | null | undefined, current: string | null | undefined): boolean {
    if (!candidate) return false;
    if (!current) return true;
    const candidateTime = Date.parse(candidate);
    const currentTime = Date.parse(current);
    if (Number.isFinite(candidateTime) && Number.isFinite(currentTime)) {
        return candidateTime >= currentTime;
    }
    return candidate >= current;
}

function mergeLiveMarketRows(
    current: Map<string, MarketStockRow>,
    payload: MarketStocksResponse,
): Map<string, MarketStockRow> {
    const next = new Map(current);
    for (const row of payload.rows || []) {
        const symbol = String(row.symbol || row.company || '').trim().toUpperCase();
        if (!symbol || row.price == null || row.price <= 0) continue;
        const candidate = { ...row, price_as_of: row.price_as_of || payload.as_of };
        const existing = next.get(symbol);
        if (!existing || quoteTimestampAtLeastAsRecent(candidate.price_as_of, existing.price_as_of)) {
            next.set(symbol, candidate);
        }
    }
    return next;
}

const RATING_OPTIONS: ExplainedOption[] = [
    {
        value: 'Endeks Üstü Getiri',
        label: 'Endeks Üstü Getiri (AL)',
        badge: 'AL · ANA ÖNERİ',
        tone: 'emerald',
        description: 'BIST 100 endeksinden belirgin şekilde pozitif ayrışması beklenen standart model portföy hissesi.',
    },
    {
        value: 'Güçlü AL / Model Portföy',
        label: 'Güçlü AL / En Çok Beğenilen',
        badge: 'GÜÇLÜ AL · FAVORİ',
        tone: 'blue',
        description: 'Derin çarpan iskontosu ve yakın vadeli güçlü bilanço/haber katalizörü barındıran yüksek beklentili seçim.',
    },
    {
        value: 'Uzun Vadeli Biriktir',
        label: 'Uzun Vadeli Biriktir',
        badge: 'BİRİKTİR · DEFANSİF',
        tone: 'amber',
        description: 'Kademeli alıma uygun, düzenli nakit akışı, temettü verimi ve bileşik büyüme odaklı çekirdek pozisyon.',
    },
    {
        value: 'Taktiksel / Kısa-Orta Vade',
        label: 'Taktiksel / Dönemsel Fırsat',
        badge: 'TAKTİKSEL · DÖNEMSEL',
        tone: 'purple',
        description: 'Çeyreklik bilanço beklentisi, mevsimsellik veya sektörel haber akışı odaklı taktiksel getiri fırsatı.',
    },
];

const HORIZON_OPTIONS: ExplainedOption[] = [
    {
        value: '1-3 Ay',
        label: '1-3 Ay (Çeyreklik / Kısa Vade)',
        badge: 'KISA VADE',
        tone: 'purple',
        description: 'Yaklaşan finansal sonuçlar veya kısa vadeli katalizörlerin fiyatlanması beklenen vade.',
    },
    {
        value: '3-6 Ay',
        label: '3-6 Ay (Orta Vade)',
        badge: 'ORTA VADE',
        tone: 'amber',
        description: '1-2 çeyreklik marj iyileşmesi ve hedef çarpanlara yakınsama beklenen dönem.',
    },
    {
        value: '6-12 Ay',
        label: '6-12 Ay (Standart Model Portföy)',
        badge: '12 AY HEDEF',
        tone: 'blue',
        description: 'Aracı kurum araştırma raporları ve model portföyler için standart 1 yıllık hedef fiyat ufku.',
    },
    {
        value: '12+ Ay',
        label: '12+ Ay (Uzun Vadeli Yatırım)',
        badge: 'UZUN VADE',
        tone: 'emerald',
        description: 'Yeni kapasite devreye girişi, ihracat pazarı genişlemesi veya çok yıllık büyüme hikayesi.',
    },
];

const SECTOR_OPTIONS: Array<{ value: string; badge: string; description: string }> = [
    { value: 'Havacılık & Ulaştırma', badge: ' döviz gelirli', description: 'Havayolu, yer hizmetleri ve lojistik şirketleri' },
    { value: 'Bankacılık & Finans', badge: 'faiz duyarlı', description: 'Mevduat bankaları, katılım, sigorta ve finansal kiralama' },
    { value: 'Holding & Yatırım', badge: 'NAD iskontosu', description: 'Çeşitlendirilmiş sanayi, enerji ve finans iştirakli holdingler' },
    { value: 'Perakende & Gıda', badge: 'defansif nakit', description: 'Gıda perakendesi, içecek, tarım ve hızlı tüketim ürünleri' },
    { value: 'Enerji & Rafineri', badge: 'nakit akışı', description: 'Elektrik üretimi, yenilenebilir enerji, doğalgaz ve rafineri' },
    { value: 'Otomotiv & Yan Sanayi', badge: 'ihracat odaklı', description: 'Otomotiv ana üreticileri, ticari araç ve yedek parça' },
    { value: 'Savunma & Teknoloji', badge: 'bakiye sipariş', description: 'Savunma sanayi, yazılım, bilişim ve elektronik sistemler' },
    { value: 'Demir-Çelik & Madencilik', badge: 'emtia döngüsel', description: 'Yassı/uzun çelik üreticileri, altın ve endüstriyel madencilik' },
    { value: 'GYO & İnşaat / Çimento', badge: 'varlık bazlı', description: 'Gayrimenkul yatırım ortaklıkları, taahhüt, çimento ve cam' },
    { value: 'İletişim & Telekom', badge: 'ARPU büyümesi', description: 'Mobil operatörler, fiber altyapı ve dijital servis sağlayıcılar' },
    { value: 'Kimya, Gübre & Petrokimya', badge: 'sanayi üretimi', description: 'Özel kimyasallar, tarımsal gübre ve plastik/kauçuk' },
    { value: 'Sağlık, İlaç & Biyoteknoloji', badge: 'defansif büyüme', description: 'Hastane zincirleri, ilaç üretimi ve medikal ekipman' },
];

const CLOSE_REASON_OPTIONS: ExplainedOption[] = [
    {
        value: 'Hedef Fiyata Ulaştı / Kâr Realizasyonu',
        label: 'Hedef Fiyata Ulaştı / Kâr Realizasyonu',
        badge: 'HEDEF TAMAMLANDI',
        tone: 'emerald',
        description: 'Hisse hedef fiyatımıza ulaştığı veya iskonto potansiyelini tamamladığı için kâr realize edildi.',
    },
    {
        value: 'Portföy Ağırlık Değişikliği / Yeni Fırsat',
        label: 'Portföy Değişikliği / Daha Yüksek Potansiyelli Fırsat',
        badge: 'ROTASYON',
        tone: 'blue',
        description: 'Nakit yaratmak veya daha yüksek alfa potansiyeli sunan başka bir hisseye yer açmak için kapatıldı.',
    },
    {
        value: 'Bilanço & Yatırım Tezi Zayıflaması',
        label: 'Bilanço & Yatırım Tezi Zayıflaması',
        badge: 'TEZ REVİZYONU',
        tone: 'amber',
        description: 'Beklenen operasyonel marjların veya katalizörlerin gerçekleşmemesi nedeniyle portföyden çıkarıldı.',
    },
    {
        value: 'Zarar Kes / Risk Yönetimi',
        label: 'Zarar Kes (Stop-Loss) / Risk Yönetimi',
        badge: 'RİSK KONTROLÜ',
        tone: 'purple',
        description: 'Piyasa/sektör koşullarındaki bozulma sonrası portföy riskini sınırlamak amacıyla kapatıldı.',
    },
];

const CATALYST_OPTIONS = [
    { value: 'Çarpan İskontosu', label: 'Çarpan iskontosu' },
    { value: 'Güçlü Bilanço Beklentisi', label: 'Güçlü bilanço' },
    { value: 'İhracat & Döviz Geliri', label: 'İhracat geliri' },
    { value: 'Yüksek Nakit Akışı', label: 'Nakit akışı' },
    { value: 'Yeni Kapasite Yatırımı', label: 'Yeni yatırım' },
    { value: 'Temettü Verimi', label: 'Temettü' },
    { value: 'Marj Genişlemesi', label: 'Marj artışı' },
    { value: 'Borçluluk Azalışı', label: 'Borç azalması' },
];

const DONUT_PALETTE = [
    '#3b82f6',
    '#10b981',
    '#f59e0b',
    '#8b5cf6',
    '#ec4899',
    '#06b6d4',
    '#eab308',
    '#f97316',
    '#14b8a6',
    '#6366f1',
];

function formatPrice(val: number | null | undefined): string {
    if (val == null || !Number.isFinite(val)) return '—';
    return `${val.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ₺`;
}

function formatPct(val: number | null | undefined, signed = true): string {
    if (val == null || !Number.isFinite(val)) return '—';
    const sign = signed && val > 0 ? '+' : '';
    return `%${sign}${val.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatMultiple(val: number | null | undefined): string {
    if (val == null || !Number.isFinite(val)) return '—';
    return `${val.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}x`;
}

function formatDateTR(iso: string | null | undefined): string {
    if (!iso) return '—';
    const d = new Date(iso.length === 10 ? `${iso}T00:00:00` : iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleDateString('tr-TR', { day: '2-digit', month: 'short', year: 'numeric' });
}

function formatDateTimeTR(iso: string | null | undefined): string {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString('tr-TR', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

function formatCompactDateTR(iso: string | null | undefined): string {
    if (!iso) return '—';
    const d = new Date(iso.length === 10 ? `${iso}T00:00:00` : iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit', year: '2-digit' });
}

function formatCompactDateTimeTR(iso: string | null | undefined): string {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const time = d.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
    return `${formatCompactDateTR(iso)} ${time}`;
}

function useAccessibleDialog<T extends HTMLElement>(active: boolean, onClose: () => void) {
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

function toneClass(val: number | null | undefined): string {
    if (val == null || !Number.isFinite(val) || val === 0) return 'mp-tone-flat';
    return val > 0 ? 'mp-tone-up' : 'mp-tone-down';
}

function compoundAlpha(assetRet: number | null | undefined, benchRet: number | null | undefined): number | null {
    if (assetRet == null || benchRet == null || !Number.isFinite(assetRet) || !Number.isFinite(benchRet)) {
        return null;
    }
    const denom = 1 + benchRet / 100;
    if (denom <= 0) return null;
    return Number((((1 + assetRet / 100) / denom - 1) * 100).toFixed(2));
}

function computeProgress(entry: number, current: number | null, target: number): number {
    if (current == null || !Number.isFinite(current)) return 0;
    const span = target - entry;
    if (Math.abs(span) < 1e-6) return current >= target ? 100 : 0;
    return Math.max(-100, Math.min(150, Number((((current - entry) / span) * 100).toFixed(1))));
}

function todayIsoDate(): string {
    const d = new Date();
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

function sliceAndNormalizePerformance(
    points: ModelPortfolioPerformancePoint[],
    range: PerfRange,
    inceptionDate: string | null,
): ModelPortfolioPerformancePoint[] {
    if (!points.length) return [];
    const lastDateStr = points[points.length - 1].date;
    const lastDate = new Date(`${lastDateStr}T00:00:00Z`);
    let cutoffIso = points[0].date;

    if (range === '1m') {
        const d = new Date(lastDate);
        d.setUTCDate(d.getUTCDate() - 30);
        cutoffIso = d.toISOString().slice(0, 10);
    } else if (range === '3m') {
        const d = new Date(lastDate);
        d.setUTCDate(d.getUTCDate() - 91);
        cutoffIso = d.toISOString().slice(0, 10);
    } else if (range === '6m') {
        const d = new Date(lastDate);
        d.setUTCDate(d.getUTCDate() - 182);
        cutoffIso = d.toISOString().slice(0, 10);
    } else if (range === 'ytd') {
        cutoffIso = `${lastDate.getUTCFullYear()}-01-01`;
    } else if (range === '1y') {
        const d = new Date(lastDate);
        d.setUTCDate(d.getUTCDate() - 365);
        cutoffIso = d.toISOString().slice(0, 10);
    } else if (range === 'all' && inceptionDate) {
        cutoffIso = inceptionDate;
    }

    if (inceptionDate && inceptionDate > cutoffIso) cutoffIso = inceptionDate;
    const working = points.filter((p) => p.date >= cutoffIso);
    if (!working.length) return [];
    const basePort = working[0].portfolio_value || 100;
    const baseXu = working[0].xu100_value || 100;
    const baseXu030 = working[0].xu030_value || 100;

    return working.map((pt) => {
        const normPort = Number(((pt.portfolio_value / basePort) * 100).toFixed(2));
        const normXu = Number(((pt.xu100_value / baseXu) * 100).toFixed(2));
        const normXu030 = Number(((pt.xu030_value / baseXu030) * 100).toFixed(2));
        const portRet = Number((normPort - 100).toFixed(2));
        const xuRet = Number((normXu - 100).toFixed(2));
        const xu030Ret = Number((normXu030 - 100).toFixed(2));
        const relRet = compoundAlpha(portRet, xuRet) ?? 0;
        const relXu030Ret = compoundAlpha(portRet, xu030Ret) ?? 0;
        return {
            date: pt.date,
            portfolio_value: normPort,
            xu100_value: normXu,
            xu030_value: normXu030,
            portfolio_return_pct: portRet,
            xu100_return_pct: xuRet,
            xu030_return_pct: xu030Ret,
            relative_return_pct: relRet,
            relative_xu030_return_pct: relXu030Ret,
        };
    });
}

export default function ModelPortfolioPage() {
    useDocumentTitle('Model Portföy | Fin-Rag');
    const navigate = useNavigate();
    const [navCollapsed, setNavCollapsed] = useState(false);

    // Shared rail universe & live quote state (reuses existing caches, 0 extra calls)
    const [xu100UniverseRows, setXu100UniverseRows] = useState<MarketUniverseRow[]>(
        () => apiClient.getCachedMarketUniverse('XUTUM')?.rows || [],
    );
    const [liveStockMap, setLiveStockMap] = useState<Map<string, MarketStockRow>>(() => {
        const cached = apiClient.getCachedMarketStocks('XUTUM');
        return new Map((cached?.rows || []).map((row) => {
            const symbol = String(row.symbol || row.company).toUpperCase();
            return [symbol, { ...row, price_as_of: row.price_as_of || cached?.as_of || null }];
        }));
    });

    // Model Portfolio state
    const [portfolioData, setPortfolioData] = useState<ModelPortfolioResponse | null>(
        () => apiClient.getCachedModelPortfolio(),
    );
    const [loading, setLoading] = useState<boolean>(() => !apiClient.getCachedModelPortfolio());
    const [error, setError] = useState<string | null>(null);

    // Performance chart state (loaded separately for instant table first-paint)
    const [perfPoints, setPerfPoints] = useState<ModelPortfolioPerformancePoint[]>([]);
    const [perfLoading, setPerfLoading] = useState(false);
    const [perfError, setPerfError] = useState<string | null>(null);
    const [selectedRange, setSelectedRange] = useState<PerfRange>('all');
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const [activeDonutIndex, setActiveDonutIndex] = useState<number | null>(null);

    // Tabs & Drawer state
    const [activeTab, setActiveTab] = useState<PortfolioTab>('active');
    const [selectedPositionId, setSelectedPositionId] = useState<string | null>(null);

    // Short-lived admin session stays in memory and is never persisted in browser storage.
    const [adminToken, setAdminToken] = useState<string>('');
    const [adminSessionExpiresAt, setAdminSessionExpiresAt] = useState<number | null>(null);
    const [isAdminVerified, setIsAdminVerified] = useState(false);
    const [adminModalOpen, setAdminModalOpen] = useState(false);
    const [adminTokenInput, setAdminTokenInput] = useState('');
    const [adminAuthError, setAdminAuthError] = useState<string | null>(null);
    const [adminAuthLoading, setAdminAuthLoading] = useState(false);
    const adminDialogRef = useAccessibleDialog<HTMLDivElement>(adminModalOpen, () => setAdminModalOpen(false));

    useEffect(() => {
        if (!adminToken || adminSessionExpiresAt === null) return;
        const timeoutId = window.setTimeout(() => {
            setAdminToken('');
            setAdminSessionExpiresAt(null);
            setIsAdminVerified(false);
        }, Math.max(0, adminSessionExpiresAt - Date.now()));
        return () => window.clearTimeout(timeoutId);
    }, [adminSessionExpiresAt, adminToken]);

    // Position Add / Edit / Close modal state
    const [editorModalOpen, setEditorModalOpen] = useState(false);
    const [editingPosition, setEditingPosition] = useState<ModelPortfolioPosition | null>(null);
    const [closingPosition, setClosingPosition] = useState<ModelPortfolioPosition | null>(null);

    useEffect(() => {
        if (xu100UniverseRows.length) return;
        let cancelled = false;
        apiClient
            .marketUniverse({ index: 'XUTUM' })
            .then((res) => {
                if (!cancelled) setXu100UniverseRows(res.rows || []);
            })
            .catch(() => {});
        return () => {
            cancelled = true;
        };
    }, [xu100UniverseRows.length]);

    // Reuse the shared XUTUM snapshot and keep its ref-counted live stream active on this page.
    useEffect(() => {
        let cancelled = false;
        const applyPayload = (payload: MarketStocksResponse) => {
            setLiveStockMap((prev) => mergeLiveMarketRows(prev, payload));
        };
        const unsubscribe = apiClient.subscribeMarketStocks((index, payload) => {
            if (index !== 'XUTUM' && index !== 'XU100') return;
            applyPayload(payload);
        });
        const releaseStream = apiClient.retainMarketStocksStream('XUTUM');

        apiClient
            .marketStocks({ index: 'XUTUM' })
            .then((payload) => {
                if (!cancelled) applyPayload(payload);
            })
            .catch(() => {});

        return () => {
            cancelled = true;
            unsubscribe();
            releaseStream();
        };
    }, []);

    const loadPerformance = useCallback((version: number, refresh = false) => {
        setPerfLoading(true);
        setPerfError(null);
        return apiClient
            .modelPortfolioPerformance({ version, refresh })
            .then((perf) => setPerfPoints(perf.points || []))
            .catch((err: unknown) => {
                setPerfPoints([]);
                setPerfError(err instanceof Error ? err.message : 'Performans grafiği verisi alınamadı.');
            })
            .finally(() => setPerfLoading(false));
    }, []);

    const loadPortfolio = useCallback((refresh = false) => {
        if (!portfolioData) setLoading(true);
        setError(null);
        apiClient
            .modelPortfolio({ refresh })
            .then((data) => {
                setPortfolioData(data);
                setLoading(false);
                void loadPerformance(data.version, refresh);
            })
            .catch((err: unknown) => {
                setError(err instanceof Error ? err.message : 'Model portföy verisi alınamadı.');
            })
            .finally(() => setLoading(false));
    }, [loadPerformance, portfolioData]);

    useEffect(() => {
        loadPortfolio(false);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Live-enriched active positions using `liveStockMap` from MarketWatchRail
    const liveActivePositions = useMemo<ModelPortfolioPosition[]>(() => {
        const rawList = portfolioData?.active_positions || [];
        const currentXu100 = portfolioData?.current_xu100 ?? null;
        return rawList.map((pos) => {
            const liveRow = liveStockMap.get(pos.symbol);
            const liveQuoteIsCurrent = Boolean(
                liveRow?.price != null
                && liveRow.price > 0
                && quoteTimestampAtLeastAsRecent(liveRow.price_as_of, pos.price_as_of),
            );
            const livePrice = liveQuoteIsCurrent ? liveRow!.price : pos.current_price;
            const liveChangePct = liveQuoteIsCurrent ? liveRow!.change_pct : pos.daily_change_pct;
            const livePriceAsOf = liveQuoteIsCurrent ? liveRow!.price_as_of : pos.price_as_of;
            if (livePrice == null || livePrice === pos.current_price) {
                if (liveChangePct === pos.daily_change_pct && livePriceAsOf === pos.price_as_of) return pos;
                return { ...pos, daily_change_pct: liveChangePct, price_as_of: livePriceAsOf };
            }
            const retPct = pos.entry_price > 0
                ? Number((((livePrice - pos.entry_price) / pos.entry_price) * 100).toFixed(2))
                : pos.return_pct;
            const xuRetPct = pos.entry_xu100 && currentXu100
                ? Number((((currentXu100 - pos.entry_xu100) / pos.entry_xu100) * 100).toFixed(2))
                : pos.xu100_return_pct;
            const relRetPct = compoundAlpha(retPct, xuRetPct);
            const upsidePct = livePrice > 0
                ? Number((((pos.target_price - livePrice) / livePrice) * 100).toFixed(2))
                : pos.target_upside_pct;
            const progressPct = computeProgress(pos.entry_price, livePrice, pos.target_price);
            const attrPct = retPct != null ? Number(((pos.weight_pct * retPct) / 100).toFixed(2)) : 0;

            return {
                ...pos,
                current_price: livePrice,
                daily_change_pct: liveChangePct,
                price_as_of: livePriceAsOf,
                return_pct: retPct,
                xu100_return_pct: xuRetPct,
                relative_return_pct: relRetPct,
                target_upside_pct: upsidePct,
                target_progress_pct: progressPct,
                target_reached: livePrice >= pos.target_price,
                attribution_pct: attrPct,
            };
        });
    }, [liveStockMap, portfolioData]);

    const liveSummary = useMemo(() => {
        const base = portfolioData?.summary;
        if (!base) {
            return {
                active_count: 0,
                closed_count: 0,
                total_weight_pct: 0,
                cash_weight_pct: 100,
                portfolio_return_pct: null,
                invested_return_pct: null,
                xu100_return_pct: null,
                relative_return_pct: null,
                weighted_target_upside_pct: null,
                closed_hit_rate_pct: null,
                closed_win_count: 0,
                closed_avg_return_pct: null,
                closed_avg_relative_pct: null,
                inception_date: null,
            };
        }
        if (!liveActivePositions.length) return base;
        const totalW = Number(liveActivePositions.reduce((acc, r) => acc + (r.weight_pct || 0), 0).toFixed(2));
        const cashW = Number(Math.max(0, 100 - totalW).toFixed(2));
        const weightedRetSum = liveActivePositions.reduce(
            (acc, r) => acc + (r.return_pct != null ? r.weight_pct * r.return_pct : 0),
            0,
        );
        const portRet = Number((weightedRetSum / 100).toFixed(2));
        const invRet = totalW > 0 ? Number((weightedRetSum / totalW).toFixed(2)) : null;
        const upsideSum = liveActivePositions.reduce(
            (acc, r) => acc + (r.target_upside_pct != null ? r.weight_pct * r.target_upside_pct : 0),
            0,
        );
        const avgUpside = totalW > 0 ? Number((upsideSum / totalW).toFixed(2)) : base.weighted_target_upside_pct;
        const relRet = compoundAlpha(invRet, base.xu100_return_pct);

        return {
            ...base,
            total_weight_pct: totalW,
            cash_weight_pct: cashW,
            portfolio_return_pct: portRet,
            invested_return_pct: invRet,
            relative_return_pct: relRet,
            weighted_target_upside_pct: avgUpside,
        };
    }, [liveActivePositions, portfolioData?.summary]);

    const selectedPosition = useMemo(() => {
        if (!selectedPositionId) return null;
        return (
            liveActivePositions.find((p) => p.id === selectedPositionId)
            || (portfolioData?.closed_positions || []).find((p) => p.id === selectedPositionId)
            || null
        );
    }, [liveActivePositions, portfolioData?.closed_positions, selectedPositionId]);

    const slicedPerfPoints = useMemo(
        () => sliceAndNormalizePerformance(perfPoints, selectedRange, liveSummary.inception_date),
        [perfPoints, selectedRange, liveSummary.inception_date],
    );

    const handleSectionChange = (section: MarketsNavigationSection) => {
        if (section === 'markets') {
            navigate(toMarketsOverview());
        } else if (section === 'funds') {
            navigate(toFunds());
        } else if (section === 'indices') {
            navigate(toMarketsIndices());
        } else {
            navigate(toMarketsStocks(DEFAULT_MARKET_INDEX));
        }
    };

    const handleAdminVerifySubmit = async (e: FormEvent) => {
        e.preventDefault();
        let cleaned = adminTokenInput.trim();
        if (cleaned.toUpperCase().startsWith('RAGFIN_ADMIN_TOKEN=')) {
            cleaned = cleaned.slice('RAGFIN_ADMIN_TOKEN='.length).trim();
        }
        cleaned = cleaned.replace(/^['"]|['"]$/g, '').trim();
        if (!cleaned) return;
        setAdminAuthLoading(true);
        setAdminAuthError(null);
        try {
            const res = await apiClient.modelPortfolioVerifyAdmin(cleaned);
            if (res.ok && res.session_token) {
                setAdminToken(res.session_token);
                setAdminSessionExpiresAt(Date.now() + res.expires_in_seconds * 1000);
                setIsAdminVerified(true);
                setAdminModalOpen(false);
                setAdminTokenInput('');
            }
        } catch (err: unknown) {
            setAdminAuthError(err instanceof Error ? err.message : 'Yönetici anahtarı doğrulanamadı.');
        } finally {
            setAdminAuthLoading(false);
        }
    };

    const handleAdminLogout = () => {
        setAdminToken('');
        setAdminSessionExpiresAt(null);
        setIsAdminVerified(false);
    };

    const handlePortfolioMutated = (nextData: ModelPortfolioResponse) => {
        setPortfolioData(nextData);
        void loadPerformance(nextData.version, true);
    };

    const handleDeletePosition = async (position: ModelPortfolioPosition) => {
        if (!adminToken) return;
        if (!window.confirm(`${position.symbol} pozisyonunu tamamen silmek istediğinize emin misiniz?`)) return;
        try {
            const next = await apiClient.modelPortfolioDeletePosition(position.id, adminToken);
            if (selectedPositionId === position.id) setSelectedPositionId(null);
            handlePortfolioMutated(next);
        } catch (err: unknown) {
            window.alert(err instanceof Error ? err.message : 'Silme işlemi başarısız oldu.');
        }
    };

    // Donut segments (Active stocks + Cash)
    const donutItems = useMemo(() => {
        const items = liveActivePositions.map((pos, idx) => ({
            symbol: pos.symbol,
            label: pos.company_name,
            weight: pos.weight_pct,
            returnPct: pos.return_pct,
            attributionPct: pos.attribution_pct ?? 0,
            color: DONUT_PALETTE[idx % DONUT_PALETTE.length],
            isCash: false,
        }));
        if (liveSummary.cash_weight_pct > 0.05) {
            items.push({
                symbol: 'NAKİT',
                label: 'Nakit / Boşta Pay',
                weight: liveSummary.cash_weight_pct,
                returnPct: 0,
                attributionPct: 0,
                color: '#334155',
                isCash: true,
            });
        }
        return items;
    }, [liveActivePositions, liveSummary.cash_weight_pct]);

    const hasPortfolioSetup = Boolean(
        portfolioData?.active_positions.length || portfolioData?.closed_positions.length,
    );

    return (
        <div className={`mn-layout${navCollapsed ? ' mn-nav-collapsed' : ''}`}>
            <MarketsNavigation
                collapsed={navCollapsed}
                activeSection="markets"
                onCollapsedChange={setNavCollapsed}
                onSectionChange={handleSectionChange}
                onFundSectionChange={(section) => navigate(section === 'compare' ? toFundComparison() : toFunds())}
                onSelectTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
                onSelectFund={(fundCode) => navigate(toFundDetail(fundCode, DEFAULT_FUND_TAB))}
            />
            <div className="markets-workspace markets-workspace-with-rail">
                <main className="market-page model-portfolio-page">
                    {/* Header */}
                    <header className="market-header mp-header">
                        <div className="market-title">
                            <span className="model-portfolio-kicker">ARAŞTIRMA & YATIRIM STRATEJİLERİ</span>
                            <h1 tabIndex={-1} data-dialog-return-focus>Model Portföy</h1>
                            <p>
                                Temel analizle seçilen hisseleri canlı fiyatlarla ve yatırım tezleriyle takip edin. Geçmiş grafik, bugünkü aktif sepetin XU100 karşısındaki simülasyonunu gösterir.
                            </p>
                            {hasPortfolioSetup && portfolioData?.updated_at && (
                                <span className="mp-update-meta">
                                    <Clock size={12} /> Son portföy düzenlemesi: {formatDateTimeTR(portfolioData.updated_at)}
                                </span>
                            )}
                        </div>
                    </header>

                    {error && (
                        <div className="mp-alert mp-alert-error" role="alert">
                            <AlertTriangle size={16} />
                            <span>{error}</span>
                            <button type="button" className="mp-btn mp-btn-subtle" onClick={() => loadPortfolio(true)}>
                                <RefreshCw size={13} /> Tekrar Dene
                            </button>
                        </div>
                    )}

                    {/* 5 KPI Summary Strip */}
                    <section className="market-quick-stats mp-kpi-strip" aria-label="Model portföy özet göstergeleri">
                        <div className="quick-stat mp-kpi-card">
                            <span className="qs-label">AKTİF HİSSELERİN AĞIRLIKLI GETİRİSİ</span>
                            <strong
                                className={`qs-value ${toneClass(liveSummary.invested_return_pct)}`}
                                title="Her pozisyonun getirisi kendi giriş gününden bugüne hesaplanır ve ağırlıkla birleştirilir."
                            >
                                {formatPct(liveSummary.invested_return_pct)}
                            </strong>
                            <span className="mp-kpi-sub">
                                Nakit dâhil portföy: <b>{formatPct(liveSummary.portfolio_return_pct)}</b>
                            </span>
                        </div>

                        <div className="quick-stat mp-kpi-card">
                            <span className="qs-label">XU100’E GÖRE AĞIRLIKLI GETİRİ (ALFA)</span>
                            <strong className={`qs-value ${toneClass(liveSummary.relative_return_pct)}`}>
                                {formatPct(liveSummary.relative_return_pct)}
                            </strong>
                            <span
                                className="mp-kpi-sub"
                                title="Her pozisyon, kendi giriş günündeki XU100 seviyesinden itibaren kıyaslanır."
                            >
                                XU100 baz getirisi: <b>{formatPct(liveSummary.xu100_return_pct)}</b>
                            </span>
                        </div>

                        <div className="quick-stat mp-kpi-card">
                            <span className="qs-label">AĞIRLIKLI HEDEF POTANSİYELİ</span>
                            <strong className={`qs-value ${toneClass(liveSummary.weighted_target_upside_pct)}`}>
                                {formatPct(liveSummary.weighted_target_upside_pct)}
                            </strong>
                            <span className="mp-kpi-sub">
                                Model hedeflerine göre · nakit hariç · garanti değildir
                            </span>
                        </div>

                        <div className="quick-stat mp-kpi-card">
                            <span className="qs-label">AKTİF POZİSYON & NAKİT DAĞILIMI</span>
                            <strong className={`qs-value${!hasPortfolioSetup ? ' mp-kpi-empty-value' : ''}`}>
                                {!hasPortfolioSetup
                                    ? 'Kurulmadı'
                                    : liveSummary.active_count > 0
                                        ? `${liveSummary.active_count} Hisse · %${liveSummary.total_weight_pct.toFixed(0)}`
                                        : 'Aktif pozisyon yok'}
                            </strong>
                            <span className="mp-kpi-sub">
                                {!hasPortfolioSetup
                                    ? 'Portföy verisi henüz eklenmedi'
                                    : liveSummary.active_count > 0
                                        ? <>Nakit / Boşta Pay: <b>%{liveSummary.cash_weight_pct.toFixed(1)}</b></>
                                        : 'Kapanmış kayıtlar Geçmiş sekmesinde'}
                            </span>
                        </div>

                        <div className="quick-stat mp-kpi-card">
                            <span className="qs-label">XU100’Ü GEÇEN KAPALI İŞLEM ORANI</span>
                            <strong className="qs-value mp-tone-accent">
                                {liveSummary.closed_hit_rate_pct != null
                                    ? `%${liveSummary.closed_hit_rate_pct.toFixed(0)}`
                                    : '—'}
                            </strong>
                            <span className="mp-kpi-sub">
                                {liveSummary.closed_count > 0
                                    ? `${liveSummary.closed_win_count}/${liveSummary.closed_count} işlemde XU100 geçildi`
                                    : 'Henüz kapatılan pozisyon yok'}
                            </span>
                        </div>
                    </section>

                    {/* Analytics Grid: Left = Cumulative Performance Chart, Right = Allocation Donut & Attribution */}
                    {liveActivePositions.length > 0 && (
                        <section className="mp-analytics-grid">
                            {/* Left: Cumulative Return vs XU100 SVG Chart */}
                            <div className="mp-panel mp-chart-panel">
                                <div className="mp-panel-header">
                                    <div>
                                        <span className="mp-panel-kicker">AKTİF MODEL PORTFÖYÜN GEÇMİŞ PERFORMANSI</span>
                                        <h2 className="mp-panel-title">Aktif Sepet vs. BIST 100 / BIST 30 (100 ₺ Bazlı)</h2>
                                        <p className="mp-chart-note">
                                            Bugünkü aktif hisseler ve ağırlıklarıyla hesaplanır; kapanmış pozisyonları ve geçmiş ağırlık değişikliklerini içermez. Grafik ilk giriş tarihinden başlar; XU100/XU030 serilerini lejanttan açıp kapatabilirsiniz.
                                        </p>
                                    </div>
                                    <div className="mp-range-tabs" role="tablist" aria-label="Grafik periyodu">
                                        {PERF_RANGES.map((r) => (
                                            <button
                                                key={r.id}
                                                type="button"
                                                role="tab"
                                                aria-selected={selectedRange === r.id}
                                                className={`mp-range-btn${selectedRange === r.id ? ' active' : ''}`}
                                                onClick={() => {
                                                    setSelectedRange(r.id);
                                                    setHoverIndex(null);
                                                }}
                                            >
                                                {r.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <ModelPortfolioPerformanceChart
                                    points={slicedPerfPoints}
                                    loading={perfLoading}
                                    error={perfError}
                                    onRetry={() => {
                                        if (portfolioData) void loadPerformance(portfolioData.version, true);
                                    }}
                                    hoverIndex={hoverIndex}
                                    onHoverIndexChange={setHoverIndex}
                                />
                            </div>

                            {/* Right: Allocation Donut & Return Attribution */}
                            <div className="mp-panel mp-allocation-panel">
                                <div className="mp-panel-header">
                                    <div>
                                        <span className="mp-panel-kicker">VARLIK DAĞILIMI & KATKI ANALİZİ</span>
                                        <h2 className="mp-panel-title">Ağırlık & Getiri Katkısı</h2>
                                    </div>
                                </div>

                                <div className="mp-allocation-body">
                                    <ModelPortfolioDonut
                                        items={donutItems}
                                        activeIndex={activeDonutIndex}
                                        onActiveIndexChange={setActiveDonutIndex}
                                    />

                                    <div className="mp-attribution-list">
                                        <div className="mp-attribution-head">
                                            <span>Hisse / Ağırlık</span>
                                            <span>Portföye Katkı (Puan)</span>
                                        </div>
                                        {donutItems.map((item, idx) => (
                                            <div
                                                key={item.symbol}
                                                className={`mp-attribution-row${activeDonutIndex === idx ? ' is-highlighted' : ''}`}
                                                onMouseEnter={() => setActiveDonutIndex(idx)}
                                                onMouseLeave={() => setActiveDonutIndex(null)}
                                            >
                                                <div className="mp-attribution-left">
                                                    <span
                                                        className="mp-color-dot"
                                                        style={{ backgroundColor: item.color }}
                                                    />
                                                    <span className="mp-attr-symbol">{item.symbol}</span>
                                                    <span className="mp-attr-weight">%{item.weight.toFixed(1)}</span>
                                                </div>
                                                <div className="mp-attribution-right">
                                                    {item.isCash ? (
                                                        <span className="mp-tone-flat">0,00 puan</span>
                                                    ) : (
                                                        <span className={toneClass(item.attributionPct)}>
                                                            {item.attributionPct > 0 ? '+' : ''}
                                                            {item.attributionPct.toFixed(2)} puan
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </section>
                    )}

                    {/* Main Content Section with 3 Tabs */}
                    <section className="mp-panel mp-main-section">
                        <div className="mp-tabs-header">
                            <div className="mp-tabs-nav" role="tablist">
                                <button
                                    type="button"
                                    role="tab"
                                    aria-selected={activeTab === 'active'}
                                    className={`mp-tab-btn${activeTab === 'active' ? ' active' : ''}`}
                                    onClick={() => setActiveTab('active')}
                                >
                                    <BriefcaseBusiness size={15} />
                                    <span>Aktif Pozisyonlar</span>
                                    <span className="mp-tab-count">{liveActivePositions.length}</span>
                                </button>
                                <button
                                    type="button"
                                    role="tab"
                                    aria-selected={activeTab === 'closed'}
                                    className={`mp-tab-btn${activeTab === 'closed' ? ' active' : ''}`}
                                    onClick={() => setActiveTab('closed')}
                                >
                                    <Award size={15} />
                                    <span>Kapatılan Pozisyonlar (Geçmiş)</span>
                                    <span className="mp-tab-count">{portfolioData?.closed_positions.length || 0}</span>
                                </button>
                                <button
                                    type="button"
                                    role="tab"
                                    aria-selected={activeTab === 'changelog'}
                                    className={`mp-tab-btn${activeTab === 'changelog' ? ' active' : ''}`}
                                    onClick={() => setActiveTab('changelog')}
                                >
                                    <History size={15} />
                                    <span>Değişiklik Günlüğü</span>
                                    <span className="mp-tab-count">{portfolioData?.changelog.length || 0}</span>
                                </button>
                            </div>
                            <div className="mp-tabs-meta">
                                {portfolioData?.current_xu100 && (
                                    <span className="mp-xu100-pill">
                                        XU100: <b>{portfolioData.current_xu100.toLocaleString('tr-TR', { maximumFractionDigits: 2 })}</b>
                                    </span>
                                )}
                                <div className="mp-toolbar-actions">
                                    {isAdminVerified ? (
                                        <>
                                            <span className="mp-admin-badge" title="Kurucu düzenleme modu açık">
                                                <Sparkles size={12} />
                                                Kurucu Modu
                                            </span>
                                            <button
                                                type="button"
                                                className="mp-btn mp-btn-primary mp-btn-sm"
                                                onClick={() => {
                                                    setEditingPosition(null);
                                                    setEditorModalOpen(true);
                                                }}
                                            >
                                                <Plus size={14} />
                                                Hisse Ekle
                                            </button>
                                            <button
                                                type="button"
                                                className="mp-btn mp-btn-subtle mp-btn-sm"
                                                onClick={handleAdminLogout}
                                                title="Kurucu modundan çık"
                                            >
                                                <LogOut size={13} />
                                                <span>Çıkış</span>
                                            </button>
                                        </>
                                    ) : (
                                        <button
                                            type="button"
                                            className="mp-founder-trigger"
                                            onClick={() => setAdminModalOpen(true)}
                                            title="Model portföyü yönetmek için kurucu girişi yapın"
                                        >
                                            <Lock size={12} />
                                            <span>Kurucu Girişi</span>
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* TAB 1: Active Positions */}
                        {activeTab === 'active' && (
                            loading && !portfolioData ? (
                                <div className="mp-empty-box">Model portföy verileri yükleniyor...</div>
                            ) : liveActivePositions.length === 0 ? (
                                <div className="mp-empty-box">
                                    <BriefcaseBusiness size={26} className="mp-empty-icon" />
                                    <h3>{hasPortfolioSetup ? 'Aktif Pozisyon Bulunmuyor' : 'Model Portföy Henüz Oluşturulmadı'}</h3>
                                    <p>
                                        {hasPortfolioSetup
                                            ? 'Aktif hisseler kapatıldı. Geçmiş sekmesinden gerçekleşen işlemleri, Değişiklik Günlüğü sekmesinden portföy hareketlerini inceleyebilirsiniz.'
                                            : isAdminVerified
                                                ? 'Kurucu modu açık. Hisse, hedef fiyat ve yatırım tezi ekleyerek portföyü oluşturabilirsiniz.'
                                                : 'Model portföye hisse eklendiğinde aktif öneriler, hedef fiyatlar ve yatırım tezleri burada listelenecektir.'}
                                    </p>
                                    {isAdminVerified && !hasPortfolioSetup && (
                                        <button
                                            type="button"
                                            className="mp-btn mp-btn-primary"
                                            onClick={() => {
                                                setEditingPosition(null);
                                                setEditorModalOpen(true);
                                            }}
                                        >
                                            <Plus size={15} /> İlk Hisseyi Ekle
                                        </button>
                                    )}
                                </div>
                            ) : (
                                <div className="mp-table-wrap" role="region" aria-label="Aktif pozisyon tablosu; yatay kaydırılabilir" tabIndex={0}>
                                    <p className="mp-table-scroll-hint">Tabloyu yana kaydırarak tüm sütunları görebilirsiniz.</p>
                                    <table className="mp-table mp-active-table">
                                        <thead>
                                            <tr>
                                                <th>Hisse / Strateji</th>
                                                <th className="num">Giriş Tarihi</th>
                                                <th className="num">Giriş Fiyatı</th>
                                                <th className="num">Güncel Fiyat</th>
                                                <th className="mp-col-target">Hedef Fiyat & İlerleme</th>
                                                <th className="num">Potansiyel</th>
                                                <th className="num">Ağırlık</th>
                                                <th className="num">Net Getiri</th>
                                                <th className="num">XU100 Rölatif</th>
                                                <th className="act">Yatırım Tezi</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {liveActivePositions.map((pos) => {
                                                const progress = pos.target_progress_pct ?? 0;
                                                const clampedBar = Math.max(0, Math.min(100, progress));
                                                const isSelected = selectedPositionId === pos.id;
                                                return (
                                                    <tr
                                                        key={pos.id}
                                                        className={`mp-row${isSelected ? ' is-selected' : ''}`}
                                                        tabIndex={0}
                                                        aria-label={`${pos.symbol} pozisyon ayrıntılarını aç`}
                                                        onClick={() => setSelectedPositionId(pos.id)}
                                                        onKeyDown={(e) => {
                                                            if ((e.key === 'Enter' || e.key === ' ') && e.target === e.currentTarget) {
                                                                e.preventDefault();
                                                                setSelectedPositionId(pos.id);
                                                            }
                                                        }}
                                                    >
                                                        <td>
                                                            <div className="mp-stock-cell">
                                                                <SymbolLogo
                                                                    symbol={pos.symbol}
                                                                    name={pos.company_name}
                                                                    kind="stock"
                                                                    logoUrl={pos.logo_url}
                                                                    size="sm"
                                                                />
                                                                <div className="mp-stock-meta">
                                                                    <div className="mp-stock-line">
                                                                        <span className="mp-ticker">{pos.symbol}</span>
                                                                        {pos.sector && (
                                                                            <span className="mp-sector-tag">{pos.sector}</span>
                                                                        )}
                                                                    </div>
                                                                    <span className="mp-company-name" title={pos.company_name}>
                                                                        {pos.company_name}
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className="num mp-mono mp-entry-date">
                                                            <div>{formatCompactDateTR(pos.entry_date)}</div>
                                                            <span className="mp-sub-cell">{pos.holding_days} gün</span>
                                                        </td>
                                                        <td className="num mp-mono">{formatPrice(pos.entry_price)}</td>
                                                        <td className="num mp-mono mp-current-price">
                                                            <div className="mp-price-main">{formatPrice(pos.current_price)}</div>
                                                            {pos.daily_change_pct != null && (
                                                                <span className={`mp-sub-cell ${toneClass(pos.daily_change_pct)}`}>
                                                                    Günlük {formatPct(pos.daily_change_pct)}
                                                                </span>
                                                            )}
                                                            {pos.price_as_of && (
                                                                <span className="mp-sub-cell">Fiyat: {formatCompactDateTimeTR(pos.price_as_of)}</span>
                                                            )}
                                                        </td>
                                                        <td className="mp-col-target">
                                                            <div className="mp-target-cell">
                                                                <div className="mp-target-top">
                                                                    <span className="mp-mono mp-target-price">
                                                                        {formatPrice(pos.target_price)}
                                                                    </span>
                                                                    {pos.target_reached ? (
                                                                        <span className="mp-reached-badge">
                                                                            <CheckCircle2 size={11} /> Hedefe Ulaştı
                                                                        </span>
                                                                    ) : (
                                                                        <span className="mp-progress-label mp-mono">
                                                                            %{progress.toFixed(0)}
                                                                        </span>
                                                                    )}
                                                                </div>
                                                                <div className="mp-progress-track">
                                                                    <div
                                                                        className={`mp-progress-fill${pos.target_reached ? ' is-reached' : progress < 0 ? ' is-negative' : ''}`}
                                                                        style={{ width: `${clampedBar}%` }}
                                                                    />
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className={`num mp-mono ${toneClass(pos.target_upside_pct)}`}>
                                                            {formatPct(pos.target_upside_pct)}
                                                        </td>
                                                        <td className="num mp-mono">
                                                            <span className="mp-weight-pill">%{pos.weight_pct.toFixed(1)}</span>
                                                        </td>
                                                        <td className={`num mp-mono mp-bold ${toneClass(pos.return_pct)}`}>
                                                            {formatPct(pos.return_pct)}
                                                        </td>
                                                        <td className={`num mp-mono mp-bold ${toneClass(pos.relative_return_pct)}`}>
                                                            {formatPct(pos.relative_return_pct)}
                                                        </td>
                                                        <td className="act" onClick={(e) => e.stopPropagation()}>
                                                            <div className="mp-row-actions">
                                                                <button
                                                                    type="button"
                                                                    className="mp-thesis-btn"
                                                                    onClick={() => setSelectedPositionId(pos.id)}
                                                                >
                                                                    <span>Tezi İncele</span>
                                                                    <ChevronRight size={14} />
                                                                </button>
                                                                {isAdminVerified && (
                                                                    <>
                                                                        <button
                                                                            type="button"
                                                                            className="mp-icon-btn"
                                                                            title="Pozisyonu Düzenle / Hedef Revize Et"
                                                                            aria-label={`${pos.symbol} pozisyonunu düzenle`}
                                                                            onClick={() => {
                                                                                setEditingPosition(pos);
                                                                                setEditorModalOpen(true);
                                                                            }}
                                                                        >
                                                                            <Pencil size={13} />
                                                                        </button>
                                                                        <button
                                                                            type="button"
                                                                            className="mp-icon-btn mp-icon-btn-warn"
                                                                            title="Pozisyonu Kapat"
                                                                            aria-label={`${pos.symbol} pozisyonunu kapat`}
                                                                            onClick={() => setClosingPosition(pos)}
                                                                        >
                                                                            <CheckCircle2 size={13} />
                                                                        </button>
                                                                    </>
                                                                )}
                                                            </div>
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            )
                        )}

                        {/* TAB 2: Closed Positions */}
                        {activeTab === 'closed' && (
                            (portfolioData?.closed_positions.length || 0) === 0 ? (
                                <div className="mp-empty-box">
                                    <Award size={26} className="mp-empty-icon" />
                                    <h3>Henüz Kapatılan Pozisyon Bulunmuyor</h3>
                                    <p>
                                        Hedef fiyatına ulaşan veya portföyden çıkarılan hisseler gerçekleşen getiri ve
                                        BIST 100 rölatif performans karnesiyle bu sekmede arşivlenir.
                                    </p>
                                </div>
                            ) : (
                                <div className="mp-closed-section">
                                    <div className="mp-closed-scorecard">
                                        <div className="mp-score-item">
                                            <span>Kapatılan Öneri Sayısı</span>
                                            <strong>{liveSummary.closed_count} İşlem</strong>
                                        </div>
                                        <div className="mp-score-item">
                                            <span>BIST 100 Üzeri Başarı (Hit Rate)</span>
                                            <strong className="mp-tone-up">
                                                %{liveSummary.closed_hit_rate_pct?.toFixed(1) ?? '0.0'} ({liveSummary.closed_win_count}/{liveSummary.closed_count})
                                            </strong>
                                        </div>
                                        <div className="mp-score-item">
                                            <span>Ortalama Gerçekleşen Getiri</span>
                                            <strong className={toneClass(liveSummary.closed_avg_return_pct)}>
                                                {formatPct(liveSummary.closed_avg_return_pct)}
                                            </strong>
                                        </div>
                                        <div className="mp-score-item">
                                            <span>Ortalama Rölatif Alfa</span>
                                            <strong className={toneClass(liveSummary.closed_avg_relative_pct)}>
                                                {formatPct(liveSummary.closed_avg_relative_pct)}
                                            </strong>
                                        </div>
                                    </div>

                                    <p className="mp-table-scroll-hint">Tabloyu yana kaydırarak tüm sütunları görebilirsiniz.</p>
                                    <div className="mp-table-wrap" role="region" aria-label="Kapatılmış pozisyon tablosu; yatay kaydırılabilir" tabIndex={0}>
                                        <table className="mp-table mp-closed-table">
                                            <thead>
                                                <tr>
                                                    <th>Hisse</th>
                                                    <th className="num">Giriş → Çıkış Tarihi</th>
                                                    <th className="num">Giriş Fiyatı</th>
                                                    <th className="num">Çıkış Fiyatı</th>
                                                    <th className="num">Gerçekleşen Getiri</th>
                                                    <th className="num">XU100 Rölatif (Alfa)</th>
                                                    <th>Kapanış Gerekçesi</th>
                                                    <th className="act">Detay</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {(portfolioData?.closed_positions || []).map((pos) => (
                                                    <tr
                                                        key={pos.id}
                                                        className="mp-row"
                                                        tabIndex={0}
                                                        aria-label={`${pos.symbol} kapanmış pozisyon ayrıntılarını aç`}
                                                        onClick={() => setSelectedPositionId(pos.id)}
                                                        onKeyDown={(e) => {
                                                            if ((e.key === 'Enter' || e.key === ' ') && e.target === e.currentTarget) {
                                                                e.preventDefault();
                                                                setSelectedPositionId(pos.id);
                                                            }
                                                        }}
                                                    >
                                                        <td>
                                                            <div className="mp-stock-cell">
                                                                <SymbolLogo
                                                                    symbol={pos.symbol}
                                                                    name={pos.company_name}
                                                                    kind="stock"
                                                                    logoUrl={pos.logo_url}
                                                                    size="sm"
                                                                />
                                                                <div className="mp-stock-meta">
                                                                    <span className="mp-ticker">{pos.symbol}</span>
                                                                    <span className="mp-company-name">{pos.company_name}</span>
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className="num mp-mono">
                                                            <div>
                                                                {formatDateTR(pos.entry_date)} → {formatDateTR(pos.closed_date)}
                                                            </div>
                                                            <span className="mp-sub-cell">{pos.holding_days} gün tutuldu</span>
                                                        </td>
                                                        <td className="num mp-mono">{formatPrice(pos.entry_price)}</td>
                                                        <td className="num mp-mono">{formatPrice(pos.exit_price)}</td>
                                                        <td className={`num mp-mono mp-bold ${toneClass(pos.return_pct)}`}>
                                                            {formatPct(pos.return_pct)}
                                                        </td>
                                                        <td className={`num mp-mono mp-bold ${toneClass(pos.relative_return_pct)}`}>
                                                            {formatPct(pos.relative_return_pct)}
                                                        </td>
                                                        <td>
                                                            <span className="mp-close-reason">
                                                                {pos.close_reason || 'Portföy Değişikliği'}
                                                            </span>
                                                        </td>
                                                        <td className="act" onClick={(e) => e.stopPropagation()}>
                                                            <div className="mp-row-actions">
                                                                <button
                                                                    type="button"
                                                                    className="mp-thesis-btn"
                                                                    onClick={() => setSelectedPositionId(pos.id)}
                                                                >
                                                                    <span>Tezi Gör</span>
                                                                    <ChevronRight size={14} />
                                                                </button>
                                                                {isAdminVerified && (
                                                                    <button
                                                                        type="button"
                                                                        className="mp-icon-btn mp-icon-btn-danger"
                                                                        title="Kaydı Sil"
                                                                        aria-label={`${pos.symbol} kaydını sil`}
                                                                        onClick={() => handleDeletePosition(pos)}
                                                                    >
                                                                        <Trash2 size={13} />
                                                                    </button>
                                                                )}
                                                            </div>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            )
                        )}

                        {/* TAB 3: Changelog Timeline */}
                        {activeTab === 'changelog' && (
                            (portfolioData?.changelog.length || 0) === 0 ? (
                                <div className="mp-empty-box">
                                    <History size={26} className="mp-empty-icon" />
                                    <h3>Henüz Portföy Hareketi Kaydedilmedi</h3>
                                    <p>
                                        Model portföye hisse eklendiğinde, hedef fiyat/ağırlık revizyonu yapıldığında veya
                                        pozisyon kapatıldığında tüm değişiklikler tarih sırasıyla burada listelenir.
                                    </p>
                                </div>
                            ) : (
                                <div className="mp-changelog-list">
                                    {(portfolioData?.changelog || []).map((entry: ModelPortfolioChangelogItem) => (
                                        <div key={entry.id} className="mp-changelog-item">
                                            <div className={`mp-changelog-badge is-${entry.action}`}>
                                                {entry.action === 'added' && 'YENİ GİRİŞ'}
                                                {entry.action === 'target_revised' && 'HEDEF REVİZYONU'}
                                                {entry.action === 'weight_revised' && 'AĞIRLIK REVİZYONU'}
                                                {entry.action === 'updated' && 'GÜNCELLEME'}
                                                {entry.action === 'closed' && 'POZİSYON KAPANIŞI'}
                                            </div>
                                            <div className="mp-changelog-content">
                                                <div className="mp-changelog-top">
                                                    <button
                                                        type="button"
                                                        className="mp-changelog-ticker"
                                                        onClick={() => navigate(toStockDetail(entry.symbol, DEFAULT_STOCK_TAB))}
                                                    >
                                                        {entry.symbol}
                                                    </button>
                                                    <span className="mp-changelog-company">{entry.company_name}</span>
                                                    <time className="mp-changelog-date mp-mono">
                                                        {formatDateTR(entry.date)}
                                                    </time>
                                                </div>
                                                <p className="mp-changelog-summary">{entry.summary}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )
                        )}
                    </section>
                </main>

                {/* Right Rail (Includes MarketFlowPanel with the new 'Model Portföy' KAP Filter!) */}
                <MarketWatchRail
                    xu100Rows={xu100UniverseRows}
                    onSelectTicker={(ticker) => navigate(toStockDetail(ticker, DEFAULT_STOCK_TAB))}
                    onSelectFund={(fundCode) => navigate(toFundDetail(fundCode, DEFAULT_FUND_TAB))}
                />
            </div>

            {/* Right-Side Investment Thesis & Multiples Comparison Drawer */}
            {selectedPosition && (
                <ModelPortfolioThesisDrawer
                    position={selectedPosition}
                    isAdmin={isAdminVerified}
                    onClose={() => setSelectedPositionId(null)}
                    onOpenStockDetail={(sym) => navigate(toStockDetail(sym, DEFAULT_STOCK_TAB))}
                    onEdit={(pos) => {
                        document.querySelector<HTMLElement>('[data-dialog-return-focus]')?.focus();
                        setSelectedPositionId(null);
                        setEditingPosition(pos);
                        setEditorModalOpen(true);
                    }}
                    onClosePosition={(pos) => {
                        document.querySelector<HTMLElement>('[data-dialog-return-focus]')?.focus();
                        setSelectedPositionId(null);
                        setClosingPosition(pos);
                    }}
                    onDeletePosition={handleDeletePosition}
                />
            )}

            {/* Founder Login Modal */}
            {adminModalOpen && (
                <div className="mp-modal-backdrop" onClick={() => setAdminModalOpen(false)}>
                    <div
                        ref={adminDialogRef}
                        className="mp-modal mp-modal-sm"
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="mp-admin-dialog-title"
                        tabIndex={-1}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="mp-modal-header">
                            <div className="mp-modal-title-row">
                                <KeyRound size={17} className="mp-accent-icon" />
                                <h3 id="mp-admin-dialog-title">Kurucu / Yönetici Girişi</h3>
                            </div>
                            <button type="button" className="mp-icon-btn" aria-label="Giriş penceresini kapat" onClick={() => setAdminModalOpen(false)}>
                                <X size={16} />
                            </button>
                        </div>
                        <form onSubmit={handleAdminVerifySubmit} className="mp-modal-body">
                            <p className="mp-modal-desc">
                                Model portföye hisse eklemek, hedef fiyat revize etmek ve yatırım tezi yazmak için
                                yönetici anahtarınızı girin.
                            </p>
                            <label className="mp-field">
                                <span>Yönetici Anahtarı</span>
                                <input
                                    type="password"
                                    data-dialog-autofocus
                                    value={adminTokenInput}
                                    onChange={(e) => setAdminTokenInput(e.target.value)}
                                    placeholder="••••••••••••••••"
                                    autoComplete="off"
                                    autoFocus
                                />
                            </label>
                            {adminAuthError && <div className="mp-form-error">{adminAuthError}</div>}
                            <div className="mp-modal-footer">
                                <button
                                    type="button"
                                    className="mp-btn mp-btn-ghost"
                                    onClick={() => setAdminModalOpen(false)}
                                >
                                    İptal
                                </button>
                                <button
                                    type="submit"
                                    className="mp-btn mp-btn-primary"
                                    disabled={adminAuthLoading || !adminTokenInput.trim()}
                                >
                                    {adminAuthLoading ? 'Doğrulanıyor...' : 'Giriş Yap'}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}

            {/* Add / Edit Position Modal */}
            {editorModalOpen && (
                <ModelPortfolioEditorModal
                    adminToken={adminToken}
                    editingPosition={editingPosition}
                    availableWeightPct={
                        editingPosition
                            ? Math.max(0, 100 - liveSummary.total_weight_pct + editingPosition.weight_pct)
                            : Math.max(0, 100 - liveSummary.total_weight_pct)
                    }
                    universeRows={xu100UniverseRows}
                    onClose={() => {
                        setEditorModalOpen(false);
                        setEditingPosition(null);
                    }}
                    onSaved={(nextData) => {
                        setEditorModalOpen(false);
                        setEditingPosition(null);
                        handlePortfolioMutated(nextData);
                    }}
                />
            )}

            {/* Close Position Modal */}
            {closingPosition && (
                <ModelPortfolioCloseModal
                    adminToken={adminToken}
                    position={closingPosition}
                    currentXu100={portfolioData?.current_xu100 ?? null}
                    onClose={() => setClosingPosition(null)}
                    onClosed={(nextData) => {
                        setClosingPosition(null);
                        setSelectedPositionId(null);
                        handlePortfolioMutated(nextData);
                    }}
                />
            )}
        </div>
    );
}

/* ==========================================================================
   Subcomponent 1: Cumulative Performance vs XU100 SVG Chart
   ========================================================================== */

function ModelPortfolioPerformanceChart({
    points,
    loading,
    error,
    onRetry,
    hoverIndex,
    onHoverIndexChange,
}: {
    points: ModelPortfolioPerformancePoint[];
    loading: boolean;
    error: string | null;
    onRetry: () => void;
    hoverIndex: number | null;
    onHoverIndexChange: (idx: number | null) => void;
}) {
    const [measureAnchorIndex, setMeasureAnchorIndex] = useState<number | null>(null);
    const [selectedMeasureRange, setSelectedMeasureRange] = useState<{ start: number; end: number } | null>(null);
    const [visibleSeries, setVisibleSeries] = useState<Record<PerformanceSeries, boolean>>({
        portfolio: true,
        xu100: true,
        xu030: true,
    });
    const pointWindowKey = `${points.length}:${points[0]?.date || ''}:${points[points.length - 1]?.date || ''}`;

    useEffect(() => {
        setMeasureAnchorIndex(null);
        setSelectedMeasureRange(null);
    }, [pointWindowKey]);

    const width = 760;
    const height = 270;
    const padLeft = 52;
    const padRight = 18;
    const padTop = 20;
    const padBottom = 32;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    if (loading && points.length === 0) {
        return <div className="mp-chart-empty">Kümülatif performans serisi hazırlanıyor...</div>;
    }
    if (error) {
        return (
            <div className="mp-chart-empty mp-chart-error-state" role="alert">
                <span>{error}</span>
                <button type="button" className="mp-btn mp-btn-subtle" onClick={onRetry}>
                    <RefreshCw size={13} /> Grafiği Yenile
                </button>
            </div>
        );
    }
    if (points.length < 2) {
        return (
            <div className="mp-chart-empty">
                Seçili dönemde grafik için en az 2 işlem günü verisi gerekir. Bu durum portföy işlemlerini etkilemez.
            </div>
        );
    }

    const allVals = points.flatMap((p) => [
        ...(visibleSeries.portfolio ? [p.portfolio_value] : []),
        ...(visibleSeries.xu100 ? [p.xu100_value] : []),
        ...(visibleSeries.xu030 ? [p.xu030_value] : []),
    ]);
    if (allVals.length === 0) allVals.push(100);
    const rawMin = Math.min(...allVals, 100);
    const rawMax = Math.max(...allVals, 100);
    const span = Math.max(4, rawMax - rawMin);
    const minVal = rawMin - span * 0.08;
    const maxVal = rawMax + span * 0.08;

    const toX = (i: number) => padLeft + (i / Math.max(1, points.length - 1)) * plotW;
    const toY = (v: number) => padTop + ((maxVal - v) / Math.max(1e-6, maxVal - minVal)) * plotH;

    const portPath = points
        .map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(p.portfolio_value).toFixed(1)}`)
        .join(' ');
    const portArea = `${portPath} L${toX(points.length - 1).toFixed(1)},${(padTop + plotH).toFixed(1)} L${padLeft.toFixed(1)},${(padTop + plotH).toFixed(1)} Z`;

    const xuPath = points
        .map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(p.xu100_value).toFixed(1)}`)
        .join(' ');
    const xu030Path = points
        .map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(p.xu030_value).toFixed(1)}`)
        .join(' ');

    const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => {
        const val = maxVal - t * (maxVal - minVal);
        return { y: padTop + t * plotH, val };
    });

    const activeIdx = hoverIndex != null && hoverIndex >= 0 && hoverIndex < points.length
        ? hoverIndex
        : points.length - 1;
    const activePt = points[activeIdx];

    const dragMeasureRange = measureAnchorIndex != null && activeIdx !== measureAnchorIndex
        ? { start: Math.min(measureAnchorIndex, activeIdx), end: Math.max(measureAnchorIndex, activeIdx) }
        : null;
    const visibleMeasureRange = dragMeasureRange || selectedMeasureRange;
    const measureStartPt = visibleMeasureRange ? points[visibleMeasureRange.start] : null;
    const measureEndPt = visibleMeasureRange ? points[visibleMeasureRange.end] : null;
    const measurePortfolioReturn = measureStartPt && measureEndPt && measureStartPt.portfolio_value !== 0
        ? ((measureEndPt.portfolio_value / measureStartPt.portfolio_value) - 1) * 100
        : null;
    const measureXu100Return = measureStartPt && measureEndPt && measureStartPt.xu100_value !== 0
        ? ((measureEndPt.xu100_value / measureStartPt.xu100_value) - 1) * 100
        : null;
    const measureXu030Return = measureStartPt && measureEndPt && measureStartPt.xu030_value !== 0
        ? ((measureEndPt.xu030_value / measureStartPt.xu030_value) - 1) * 100
        : null;
    const measureRelativeXu100Return = compoundAlpha(measurePortfolioReturn, measureXu100Return);
    const measureRelativeXu030Return = compoundAlpha(measurePortfolioReturn, measureXu030Return);
    const formatMeasureDelta = (end: number, start: number) => {
        const delta = end - start;
        const sign = delta > 0 ? '+' : '';
        return `${sign}${delta.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ₺`;
    };
    const measureTooltipItems: Array<{ label: string; value: string; change: number | null }> = [];
    if (measureStartPt && measureEndPt) {
        if (visibleSeries.portfolio) {
            measureTooltipItems.push({
                label: 'Sepet',
                value: `${formatMeasureDelta(measureEndPt.portfolio_value, measureStartPt.portfolio_value)} · ${formatPct(measurePortfolioReturn)}`,
                change: measurePortfolioReturn,
            });
        }
        if (visibleSeries.xu100) {
            measureTooltipItems.push({
                label: 'XU100',
                value: `${formatMeasureDelta(measureEndPt.xu100_value, measureStartPt.xu100_value)} · ${formatPct(measureXu100Return)}`,
                change: measureXu100Return,
            });
        }
        if (visibleSeries.xu030) {
            measureTooltipItems.push({
                label: 'XU030',
                value: `${formatMeasureDelta(measureEndPt.xu030_value, measureStartPt.xu030_value)} · ${formatPct(measureXu030Return)}`,
                change: measureXu030Return,
            });
        }
        if (visibleSeries.portfolio && visibleSeries.xu100) {
            measureTooltipItems.push({
                label: 'Alfa / XU100',
                value: formatPct(measureRelativeXu100Return),
                change: measureRelativeXu100Return,
            });
        }
        if (visibleSeries.portfolio && visibleSeries.xu030) {
            measureTooltipItems.push({
                label: 'Alfa / XU030',
                value: formatPct(measureRelativeXu030Return),
                change: measureRelativeXu030Return,
            });
        }
    }
    const measureFocusX = visibleMeasureRange ? toX(visibleMeasureRange.end) : 0;
    const measureFocusY = measureEndPt
        ? Math.min(
            visibleSeries.portfolio ? toY(measureEndPt.portfolio_value) : Infinity,
            visibleSeries.xu100 ? toY(measureEndPt.xu100_value) : Infinity,
            visibleSeries.xu030 ? toY(measureEndPt.xu030_value) : Infinity,
        )
        : padTop;
    const measureTooltipStyle = {
        left: `clamp(8px, calc(${(measureFocusX / width) * 100}% - 140px), calc(100% - 288px))`,
        top: `clamp(8px, calc(${(measureFocusY / height) * 100}% - 78px), calc(100% - 190px))`,
    };
    const toggleSeries = (series: PerformanceSeries) => {
        setVisibleSeries((current) => ({ ...current, [series]: !current[series] }));
    };
    const hasVisibleSeries = visibleSeries.portfolio || visibleSeries.xu100 || visibleSeries.xu030;

    const handlePointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        if (rect.width <= 0) return;
        const relX = ((e.clientX - rect.left) / rect.width) * width;
        const ratio = Math.max(0, Math.min(1, (relX - padLeft) / plotW));
        const idx = Math.round(ratio * (points.length - 1));
        onHoverIndexChange(idx);
    };

    const handlePointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        e.preventDefault();
        const rect = e.currentTarget.getBoundingClientRect();
        if (rect.width <= 0) return;
        const relX = ((e.clientX - rect.left) / rect.width) * width;
        const ratio = Math.max(0, Math.min(1, (relX - padLeft) / plotW));
        const idx = Math.round(ratio * (points.length - 1));
        setSelectedMeasureRange(null);
        setMeasureAnchorIndex(idx);
        onHoverIndexChange(idx);
        e.currentTarget.setPointerCapture?.(e.pointerId);
    };

    const finishPointerMeasure = (e: ReactPointerEvent<SVGSVGElement>, commit: boolean) => {
        if (commit && measureAnchorIndex != null) {
            const rect = e.currentTarget.getBoundingClientRect();
            if (rect.width > 0) {
                const relX = ((e.clientX - rect.left) / rect.width) * width;
                const ratio = Math.max(0, Math.min(1, (relX - padLeft) / plotW));
                const endIndex = Math.round(ratio * (points.length - 1));
                onHoverIndexChange(endIndex);
                if (endIndex !== measureAnchorIndex) {
                    setSelectedMeasureRange({
                        start: Math.min(measureAnchorIndex, endIndex),
                        end: Math.max(measureAnchorIndex, endIndex),
                    });
                }
            }
        }
        setMeasureAnchorIndex(null);
        try {
            if (e.currentTarget.hasPointerCapture?.(e.pointerId)) {
                e.currentTarget.releasePointerCapture(e.pointerId);
            }
        } catch {
            // Ignore release errors when capture has already ended.
        }
    };

    const handleChartKeyDown = (e: ReactKeyboardEvent<SVGSVGElement>) => {
        if (e.key === 'Escape') {
            setMeasureAnchorIndex(null);
            setSelectedMeasureRange(null);
            return;
        }
        if (e.key === ' ' || e.key === 'Enter') {
            e.preventDefault();
            if (measureAnchorIndex == null) {
                setSelectedMeasureRange(null);
                setMeasureAnchorIndex(activeIdx);
            } else {
                if (activeIdx !== measureAnchorIndex) {
                    setSelectedMeasureRange({
                        start: Math.min(measureAnchorIndex, activeIdx),
                        end: Math.max(measureAnchorIndex, activeIdx),
                    });
                }
                setMeasureAnchorIndex(null);
            }
            return;
        }
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'Home' || e.key === 'End') {
            e.preventDefault();
            const current = hoverIndex ?? points.length - 1;
            const next = e.key === 'Home'
                ? 0
                : e.key === 'End'
                    ? points.length - 1
                    : Math.max(0, Math.min(points.length - 1, current + (e.key === 'ArrowRight' ? 1 : -1)));
            onHoverIndexChange(next);
        }
    };

    return (
        <div className="mp-chart-container">
            <div className="mp-chart-legend">
                <button
                    type="button"
                    className={`mp-legend-item mp-legend-toggle${visibleSeries.portfolio ? '' : ' is-hidden'}`}
                    aria-pressed={visibleSeries.portfolio}
                    aria-label={`${visibleSeries.portfolio ? 'Grafikten gizle' : 'Grafikte göster'}: Model Portföy`}
                    onClick={() => toggleSeries('portfolio')}
                >
                    <span className="mp-legend-swatch is-portfolio" />
                    <span className="mp-legend-label">Model Portföy:</span>
                    <strong className="mp-mono">{activePt.portfolio_value.toFixed(2)} ₺</strong>
                    <span className={`mp-mono ${toneClass(activePt.portfolio_return_pct)}`}>
                        ({formatPct(activePt.portfolio_return_pct)})
                    </span>
                </button>
                <button
                    type="button"
                    className={`mp-legend-item mp-legend-toggle${visibleSeries.xu100 ? '' : ' is-hidden'}`}
                    aria-pressed={visibleSeries.xu100}
                    aria-label={`${visibleSeries.xu100 ? 'Grafikten gizle' : 'Grafikte göster'}: BIST 100 XU100`}
                    onClick={() => toggleSeries('xu100')}
                >
                    <span className="mp-legend-swatch is-xu100" />
                    <span className="mp-legend-label">BIST 100 (XU100):</span>
                    <strong className="mp-mono">{activePt.xu100_value.toFixed(2)} ₺</strong>
                    <span className={`mp-mono ${toneClass(activePt.xu100_return_pct)}`}>
                        ({formatPct(activePt.xu100_return_pct)})
                    </span>
                </button>
                <button
                    type="button"
                    className={`mp-legend-item mp-legend-toggle${visibleSeries.xu030 ? '' : ' is-hidden'}`}
                    aria-pressed={visibleSeries.xu030}
                    aria-label={`${visibleSeries.xu030 ? 'Grafikten gizle' : 'Grafikte göster'}: BIST 30 XU030`}
                    onClick={() => toggleSeries('xu030')}
                >
                    <span className="mp-legend-swatch is-xu030" />
                    <span className="mp-legend-label">BIST 30 (XU030):</span>
                    <strong className="mp-mono">{activePt.xu030_value.toFixed(2)} ₺</strong>
                    <span className={`mp-mono ${toneClass(activePt.xu030_return_pct)}`}>
                        ({formatPct(activePt.xu030_return_pct)})
                    </span>
                </button>
                <div className="mp-legend-alpha-group">
                    <div className="mp-legend-item mp-legend-alpha">
                        <span className="mp-legend-label">Alfa / XU100:</span>
                        <strong className={`mp-mono ${toneClass(activePt.relative_return_pct)}`}>
                            {formatPct(activePt.relative_return_pct)}
                        </strong>
                    </div>
                    <div className="mp-legend-item mp-legend-alpha">
                        <span className="mp-legend-label">Alfa / XU030:</span>
                        <strong className={`mp-mono ${toneClass(activePt.relative_xu030_return_pct)}`}>
                            {formatPct(activePt.relative_xu030_return_pct)}
                        </strong>
                    </div>
                    <span className="mp-legend-date mp-mono">{formatDateTR(activePt.date)}</span>
                </div>
            </div>

            {!visibleMeasureRange && (
                <div className="mp-chart-measure-hint">
                    İki nokta arasındaki değişimi görmek için grafikte sürükleyin.
                </div>
            )}

            <div className="mp-chart-stage">
            <svg
                className="mp-perf-svg"
                viewBox={`0 0 ${width} ${height}`}
                role="group"
                aria-label="Model portföyü, XU100 ve XU030 performans grafiği. Serileri göstermek veya gizlemek için lejantlarına basın. İki tarih arasındaki değişimi görmek için grafikte sürükleyin; klavyede ok tuşlarıyla ilerleyip boşluk veya Enter ile başlangıç ve bitiş noktalarını seçin. Escape ölçümü temizler."
                tabIndex={0}
                onPointerMove={handlePointerMove}
                onPointerDown={handlePointerDown}
                onPointerUp={(e) => finishPointerMeasure(e, true)}
                onPointerCancel={(e) => finishPointerMeasure(e, false)}
                onPointerLeave={() => {
                    if (measureAnchorIndex == null) onHoverIndexChange(null);
                }}
                onKeyDown={handleChartKeyDown}
            >
                <title>Model Portföy, BIST 100 ve BIST 30 performansı</title>
                <defs>
                    <linearGradient id="mpPortFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.26" />
                        <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.0" />
                    </linearGradient>
                </defs>

                {ticks.map((t, i) => (
                    <g key={i}>
                        <line
                            x1={padLeft}
                            y1={t.y}
                            x2={width - padRight}
                            y2={t.y}
                            className="mp-chart-grid"
                        />
                        <text x={padLeft - 8} y={t.y + 4} textAnchor="end" className="mp-chart-axis-text">
                            {t.val.toFixed(0)}₺
                        </text>
                    </g>
                ))}

                {/* 100 TL baseline */}
                {minVal < 100 && maxVal > 100 && (
                    <line
                        x1={padLeft}
                        y1={toY(100)}
                        x2={width - padRight}
                        y2={toY(100)}
                        className="mp-chart-baseline"
                    />
                )}

                {visibleMeasureRange && (
                    <rect
                        x={toX(visibleMeasureRange.start)}
                        y={padTop}
                        width={Math.max(1, toX(visibleMeasureRange.end) - toX(visibleMeasureRange.start))}
                        height={plotH}
                        className="mp-chart-selection-band"
                    />
                )}

                {visibleSeries.portfolio && <path d={portArea} fill="url(#mpPortFill)" />}
                {visibleSeries.xu100 && <path d={xuPath} fill="none" className="mp-line-xu100" />}
                {visibleSeries.xu030 && <path d={xu030Path} fill="none" className="mp-line-xu030" />}
                {!hasVisibleSeries && (
                    <text x={width / 2} y={padTop + plotH / 2} textAnchor="middle" className="mp-chart-no-series">
                        Görmek için üstteki serilerden birine basın
                    </text>
                )}

                {/* Active crosshair & dots */}
                <line
                    x1={toX(activeIdx)}
                    y1={padTop}
                    x2={toX(activeIdx)}
                    y2={padTop + plotH}
                    className="mp-chart-crosshair"
                />
                {visibleSeries.xu100 && (
                    <circle
                        cx={toX(activeIdx)}
                        cy={toY(activePt.xu100_value)}
                        r="4"
                        className="mp-dot-xu100"
                    />
                )}
                {visibleSeries.xu030 && (
                    <circle
                        cx={toX(activeIdx)}
                        cy={toY(activePt.xu030_value)}
                        r="4"
                        className="mp-dot-xu030"
                    />
                )}
                {visibleSeries.portfolio && (
                    <circle
                        cx={toX(activeIdx)}
                        cy={toY(activePt.portfolio_value)}
                        r="4.5"
                        className="mp-dot-portfolio"
                    />
                )}

                {visibleMeasureRange && measureStartPt && measureEndPt && (
                    <g className="mp-chart-measure-markers">
                        <line x1={toX(visibleMeasureRange.start)} y1={padTop} x2={toX(visibleMeasureRange.start)} y2={padTop + plotH} />
                        <line x1={toX(visibleMeasureRange.end)} y1={padTop} x2={toX(visibleMeasureRange.end)} y2={padTop + plotH} />
                        {visibleSeries.portfolio && <circle cx={toX(visibleMeasureRange.start)} cy={toY(measureStartPt.portfolio_value)} r="4" className="mp-dot-portfolio" />}
                        {visibleSeries.portfolio && <circle cx={toX(visibleMeasureRange.end)} cy={toY(measureEndPt.portfolio_value)} r="4" className="mp-dot-portfolio" />}
                        {visibleSeries.xu100 && <circle cx={toX(visibleMeasureRange.start)} cy={toY(measureStartPt.xu100_value)} r="3.5" className="mp-dot-xu100" />}
                        {visibleSeries.xu100 && <circle cx={toX(visibleMeasureRange.end)} cy={toY(measureEndPt.xu100_value)} r="3.5" className="mp-dot-xu100" />}
                        {visibleSeries.xu030 && <circle cx={toX(visibleMeasureRange.start)} cy={toY(measureStartPt.xu030_value)} r="3.5" className="mp-dot-xu030" />}
                        {visibleSeries.xu030 && <circle cx={toX(visibleMeasureRange.end)} cy={toY(measureEndPt.xu030_value)} r="3.5" className="mp-dot-xu030" />}
                    </g>
                )}

                {/* Start / Middle / End X labels */}
                <text x={padLeft} y={height - 8} textAnchor="start" className="mp-chart-axis-text">
                    {formatDateTR(points[0].date)}
                </text>
                <text x={padLeft + plotW / 2} y={height - 8} textAnchor="middle" className="mp-chart-axis-text">
                    {formatDateTR(points[Math.floor(points.length / 2)].date)}
                </text>
                <text x={width - padRight} y={height - 8} textAnchor="end" className="mp-chart-axis-text">
                    {formatDateTR(points[points.length - 1].date)}
                </text>
            </svg>
                {visibleMeasureRange && hasVisibleSeries && measureStartPt && measureEndPt && (
                    <div className="mp-chart-measure-tooltip" style={measureTooltipStyle} role="status" aria-live="polite">
                        <div className="mp-chart-measure-tooltip-header">
                            <strong>{formatDateTR(measureStartPt.date)} → {formatDateTR(measureEndPt.date)}</strong>
                            {selectedMeasureRange && (
                                <button
                                    type="button"
                                    aria-label="Ölçümü temizle"
                                    title="Ölçümü temizle"
                                    onClick={() => setSelectedMeasureRange(null)}
                                >
                                    <X size={13} />
                                </button>
                            )}
                        </div>
                        <div className="mp-chart-measure-tooltip-grid">
                            {measureTooltipItems.map((item) => (
                                <div className="mp-chart-measure-tooltip-item" key={item.label}>
                                    <span>{item.label}</span>
                                    <strong className={toneClass(item.change)}>{item.value}</strong>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

/* ==========================================================================
   Subcomponent 2: Portfolio Weight Donut Chart
   ========================================================================== */

function ModelPortfolioDonut({
    items,
    activeIndex,
    onActiveIndexChange,
}: {
    items: Array<{ symbol: string; label: string; weight: number; color: string; isCash: boolean }>;
    activeIndex: number | null;
    onActiveIndexChange: (idx: number | null) => void;
}) {
    const radius = 64;
    const circumference = 2 * Math.PI * radius;
    const total = Math.max(1e-6, items.reduce((acc, item) => acc + Math.max(0, item.weight), 0));

    let accumulated = 0;
    const segments = items.map((item, idx) => {
        const fraction = Math.max(0, item.weight) / total;
        const length = fraction * circumference;
        const gap = Math.max(0, circumference - length);
        const offset = -accumulated;
        accumulated += length;
        return { item, idx, length, gap, offset };
    });

    const activeItem = activeIndex != null ? items[activeIndex] : null;

    return (
        <div className="mp-donut-wrap">
            <svg viewBox="0 0 160 160" className="mp-donut-svg">
                <circle cx="80" cy="80" r={radius} className="mp-donut-track" />
                {segments.map((seg) => (
                    <circle
                        key={seg.item.symbol}
                        cx="80"
                        cy="80"
                        r={radius}
                        fill="none"
                        stroke={seg.item.color}
                        strokeWidth={activeIndex === seg.idx ? 18 : 14}
                        strokeDasharray={`${seg.length} ${seg.gap}`}
                        strokeDashoffset={seg.offset}
                        transform="rotate(-90 80 80)"
                        className="mp-donut-segment"
                        style={{
                            opacity: activeIndex != null && activeIndex !== seg.idx ? 0.38 : 1,
                        }}
                        onMouseEnter={() => onActiveIndexChange(seg.idx)}
                        onMouseLeave={() => onActiveIndexChange(null)}
                    />
                ))}
            </svg>
            <div className="mp-donut-center">
                <span>{activeItem ? activeItem.symbol : 'HİSSE PAYI'}</span>
                <strong className="mp-mono">
                    %{activeItem ? activeItem.weight.toFixed(1) : items.filter((i) => !i.isCash).reduce((a, b) => a + b.weight, 0).toFixed(0)}
                </strong>
            </div>
        </div>
    );
}

/* ==========================================================================
   Subcomponent 3: Right-Side Investment Thesis & Multiples Drawer
   ========================================================================== */

function ModelPortfolioThesisDrawer({
    position,
    isAdmin,
    onClose,
    onOpenStockDetail,
    onEdit,
    onClosePosition,
    onDeletePosition,
}: {
    position: ModelPortfolioPosition;
    isAdmin: boolean;
    onClose: () => void;
    onOpenStockDetail: (symbol: string) => void;
    onEdit: (pos: ModelPortfolioPosition) => void;
    onClosePosition: (pos: ModelPortfolioPosition) => void;
    onDeletePosition: (pos: ModelPortfolioPosition) => void;
}) {
    const progress = position.target_progress_pct ?? 0;
    const clampedBar = Math.max(0, Math.min(100, progress));
    const dialogRef = useAccessibleDialog<HTMLElement>(true, onClose);

    const multipleRows = [
        {
            label: 'F/K (Fiyat / Kazanç)',
            entry: position.entry_multiples?.fk,
            current: position.current_multiples?.fk,
        },
        {
            label: 'FD/FAVÖK',
            entry: position.entry_multiples?.fd_favok,
            current: position.current_multiples?.fd_favok,
        },
        {
            label: 'PD/DD (Piyasa Değeri / Defter Değeri)',
            entry: position.entry_multiples?.pd_dd,
            current: position.current_multiples?.pd_dd,
        },
        {
            label: 'Net Borç / FAVÖK',
            entry: position.entry_multiples?.net_borc_favok,
            current: position.current_multiples?.net_borc_favok,
        },
    ];

    return (
        <div className="mp-drawer-backdrop" onClick={onClose}>
            <aside
                ref={dialogRef}
                className="mp-drawer"
                role="dialog"
                aria-modal="true"
                aria-labelledby="mp-thesis-dialog-title"
                tabIndex={-1}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Drawer Header */}
                <div className="mp-drawer-header">
                    <div className="mp-drawer-identity">
                        <SymbolLogo
                            symbol={position.symbol}
                            name={position.company_name}
                            kind="stock"
                            logoUrl={position.logo_url}
                            size="md"
                        />
                        <div>
                            <div className="mp-drawer-title-row">
                                <h2 id="mp-thesis-dialog-title">{position.symbol}</h2>
                                <span className="mp-rating-badge">{position.rating || 'Endeks Üstü Getiri'}</span>
                                {position.status === 'closed' && (
                                    <span className="mp-closed-badge">KAPATILDI</span>
                                )}
                            </div>
                            <p className="mp-drawer-company">{position.company_name}</p>
                        </div>
                    </div>
                    <div className="mp-drawer-header-btns">
                        <button
                            type="button"
                            className="mp-btn mp-btn-subtle"
                            onClick={() => onOpenStockDetail(position.symbol)}
                        >
                            <span>Hisse Detayı</span>
                            <ExternalLink size={13} />
                        </button>
                        <button type="button" className="mp-icon-btn" onClick={onClose} aria-label="Detay penceresini kapat">
                            <X size={18} />
                        </button>
                    </div>
                </div>

                {/* Drawer Body */}
                <div className="mp-drawer-body">
                    {/* Price & Target Progress Summary */}
                    <div className="mp-drawer-card mp-drawer-price-card">
                        <div className="mp-drawer-metrics-grid">
                            <div>
                                <span>Giriş Fiyatı ({formatDateTR(position.entry_date)})</span>
                                <strong className="mp-mono">{formatPrice(position.entry_price)}</strong>
                            </div>
                            <div>
                                <span>{position.status === 'closed' ? 'Çıkış Fiyatı' : 'Güncel Fiyat'}</span>
                                <strong className="mp-mono">{formatPrice(position.current_price)}</strong>
                            </div>
                            <div>
                                <span>Hedef Fiyat ({position.horizon})</span>
                                <strong className="mp-mono mp-tone-accent">{formatPrice(position.target_price)}</strong>
                            </div>
                            <div>
                                <span>Net Getiri / XU100 Alfa</span>
                                <strong className={`mp-mono ${toneClass(position.return_pct)}`}>
                                    {formatPct(position.return_pct)} / {formatPct(position.relative_return_pct)}
                                </strong>
                            </div>
                        </div>

                        {position.status === 'active' && (
                            <div className="mp-drawer-progress-box">
                                <div className="mp-drawer-progress-labels">
                                    <span>
                                        Hedef Fiyat İlerlemesi: <b>%{progress.toFixed(1)}</b>
                                    </span>
                                    <span className={toneClass(position.target_upside_pct)}>
                                        Kalan Potansiyel: <b>{formatPct(position.target_upside_pct)}</b>
                                    </span>
                                </div>
                                <div className="mp-progress-track mp-progress-track-lg">
                                    <div
                                        className={`mp-progress-fill${position.target_reached ? ' is-reached' : ''}`}
                                        style={{ width: `${clampedBar}%` }}
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Investment Thesis Section */}
                    <div className="mp-drawer-section">
                        <div className="mp-section-heading">
                            <TrendingUp size={15} className="mp-accent-icon" />
                            <h3>Model Portföye Giriş Gerekçesi & Yatırım Tezi</h3>
                        </div>
                        <div className="mp-thesis-box">
                            {position.thesis ? (
                                position.thesis.split('\n').map((para, i) => (
                                    <p key={i}>{para}</p>
                                ))
                            ) : (
                                <p className="mp-muted">Bu pozisyon için henüz detaylı yatırım tezi metni girilmemiş.</p>
                            )}
                        </div>
                    </div>

                    {/* Catalysts / Expectations */}
                    {position.catalysts && position.catalysts.length > 0 && (
                        <div className="mp-drawer-section">
                            <div className="mp-section-heading">
                                <Target size={15} className="mp-accent-icon" />
                                <h3>Beklentiler & Katalizörler</h3>
                            </div>
                            <div className="mp-catalyst-tags">
                                {position.catalysts.map((cat) => (
                                    <span key={cat} className="mp-catalyst-chip">
                                        <ArrowUpRight size={13} />
                                        {cat}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Entry Day vs Current Valuation Multiples */}
                    <div className="mp-drawer-section">
                        <div className="mp-section-heading">
                            <Layers size={15} className="mp-accent-icon" />
                            <h3>Giriş Günü vs. Güncel Çarpan Kıyaslaması</h3>
                        </div>
                        <div className="mp-multiples-compare-table-wrap">
                            <table className="mp-multiples-table">
                                <thead>
                                    <tr>
                                        <th>Çarpan</th>
                                        <th className="num">Portföye Girişte</th>
                                        <th className="num">Güncel</th>
                                        <th className="num">Değişim</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {multipleRows.map((row) => {
                                        const diff =
                                            row.entry != null && row.current != null
                                                ? row.current - row.entry
                                                : null;
                                        return (
                                            <tr key={row.label}>
                                                <td>{row.label}</td>
                                                <td className="num mp-mono">{formatMultiple(row.entry)}</td>
                                                <td className="num mp-mono mp-bold">{formatMultiple(row.current)}</td>
                                                <td className="num mp-mono">
                                                    {diff != null ? (
                                                        <span className={diff <= 0 ? 'mp-tone-up' : 'mp-tone-flat'}>
                                                            {diff > 0 ? '+' : ''}
                                                            {diff.toFixed(2)}x
                                                        </span>
                                                    ) : (
                                                        '—'
                                                    )}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Risk Note */}
                    {position.risk_note && (
                        <div className="mp-drawer-section">
                            <div className="mp-section-heading">
                                <AlertTriangle size={15} className="mp-warn-icon" />
                                <h3>Riskler & İzlenecek Metrikler</h3>
                            </div>
                            <div className="mp-risk-box">
                                <p>{position.risk_note}</p>
                            </div>
                        </div>
                    )}

                    {/* Target Price & Weight Revision History */}
                    {position.revisions && position.revisions.length > 0 && (
                        <div className="mp-drawer-section">
                            <div className="mp-section-heading">
                                <Clock size={15} className="mp-accent-icon" />
                                <h3>Hedef Fiyat & Ağırlık Revizyon Geçmişi</h3>
                            </div>
                            <div className="mp-revisions-list">
                                {position.revisions.map((rev) => (
                                    <div key={rev.id} className="mp-revision-item">
                                        <div className="mp-revision-top">
                                            <span className="mp-mono mp-revision-date">{formatDateTR(rev.date)}</span>
                                            <span className="mp-mono">
                                                Hedef: <s>{formatPrice(rev.prev_target_price)}</s> →{' '}
                                                <b>{formatPrice(rev.new_target_price)}</b>
                                            </span>
                                            {rev.prev_weight_pct !== rev.new_weight_pct && (
                                                <span className="mp-mono">
                                                    Ağırlık: %{rev.prev_weight_pct} → <b>%{rev.new_weight_pct}</b>
                                                </span>
                                            )}
                                        </div>
                                        {rev.note && <p className="mp-revision-note">{rev.note}</p>}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Drawer Admin Footer */}
                {isAdmin && (
                    <div className="mp-drawer-footer">
                        {position.status === 'active' && (
                            <>
                                <button
                                    type="button"
                                    className="mp-btn mp-btn-subtle"
                                    onClick={() => onEdit(position)}
                                >
                                    <Pencil size={14} /> Düzenle / Revize Et
                                </button>
                                <button
                                    type="button"
                                    className="mp-btn mp-btn-primary"
                                    onClick={() => onClosePosition(position)}
                                >
                                    <CheckCircle2 size={14} /> Pozisyonu Kapat
                                </button>
                            </>
                        )}
                        <button
                            type="button"
                            className="mp-btn mp-btn-danger"
                            onClick={() => onDeletePosition(position)}
                        >
                            <Trash2 size={14} /> Sil
                        </button>
                    </div>
                )}
            </aside>
        </div>
    );
}

/* ==========================================================================
   Helper Subcomponents: Custom Explained Dropdowns (No Native <select>)
   ========================================================================== */

function MpExplainedDropdown({
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
                    <div className="mp-explained-trigger-top">
                        <span className="mp-explained-trigger-title">{selected.label}</span>
                        <span className={`mp-explained-badge is-${selected.tone}`}>{selected.badge}</span>
                    </div>
                    <span className="mp-explained-trigger-desc">{selected.description}</span>
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
                                className={`mp-explained-option${isSelected ? ' is-selected' : ''}`}
                                onClick={() => {
                                    onChange(opt.value);
                                    setOpen(false);
                                }}
                            >
                                <div className="mp-explained-option-body">
                                    <div className="mp-explained-option-top">
                                        <strong>{opt.label}</strong>
                                        <span className={`mp-explained-badge is-${opt.tone}`}>{opt.badge}</span>
                                    </div>
                                    <p>{opt.description}</p>
                                </div>
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

function MpSectorDropdown({
    value,
    onChange,
}: {
    value: string;
    onChange: (nextSector: string) => void;
}) {
    const [open, setOpen] = useState(false);
    const wrapRef = useRef<HTMLDivElement | null>(null);

    const matchedPreset = useMemo(
        () => SECTOR_OPTIONS.find((s) => s.value.toLowerCase() === value.trim().toLowerCase()),
        [value],
    );

    useEffect(() => {
        if (!open) return;
        const onDocClick = (e: MouseEvent) => {
            if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
                setOpen(false);
            }
        };
        document.addEventListener('mousedown', onDocClick);
        return () => document.removeEventListener('mousedown', onDocClick);
    }, [open]);

    return (
        <div className="mp-field mp-explained-select-wrap" ref={wrapRef}>
            <span className="mp-field-label">Sektör & Tema Etiketi</span>
            <div className="mp-sector-input-box">
                <input
                    type="text"
                    aria-label="Sektör ve tema etiketi"
                    value={value}
                    onFocus={() => setOpen(true)}
                    onChange={(e) => {
                        onChange(e.target.value);
                        setOpen(true);
                    }}
                    placeholder="Sektör seçin veya yazın..."
                />
                <button
                    type="button"
                    className="mp-sector-toggle-btn"
                    onClick={() => setOpen((p) => !p)}
                    aria-label="Sektör listesini aç"
                >
                    <ChevronDown size={14} className={`mp-explained-chevron${open ? ' is-open' : ''}`} />
                </button>
            </div>
            {matchedPreset && (
                <span className="mp-field-micro-hint">{matchedPreset.description}</span>
            )}

            {open && (
                <div className="mp-explained-menu mp-sector-menu" role="listbox">
                    {SECTOR_OPTIONS.map((sec) => {
                        const isSelected = sec.value === value;
                        return (
                            <button
                                key={sec.value}
                                type="button"
                                className={`mp-explained-option${isSelected ? ' is-selected' : ''}`}
                                onClick={() => {
                                    onChange(sec.value);
                                    setOpen(false);
                                }}
                            >
                                <div className="mp-explained-option-body">
                                    <div className="mp-explained-option-top">
                                        <strong>{sec.value}</strong>
                                        <span className="mp-explained-badge is-blue">{sec.badge}</span>
                                    </div>
                                    <p>{sec.description}</p>
                                </div>
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

/* ==========================================================================
   Subcomponent 4: Founder Add / Edit Position Modal with Guided UX
   ========================================================================== */

function ModelPortfolioEditorModal({
    adminToken,
    editingPosition,
    availableWeightPct,
    universeRows,
    onClose,
    onSaved,
}: {
    adminToken: string;
    editingPosition: ModelPortfolioPosition | null;
    availableWeightPct: number;
    universeRows: MarketUniverseRow[];
    onClose: () => void;
    onSaved: (nextData: ModelPortfolioResponse) => void;
}) {
    const isEdit = Boolean(editingPosition);
    const [symbol, setSymbol] = useState(editingPosition?.symbol || '');
    const [symbolQuery, setSymbolQuery] = useState(editingPosition?.symbol || '');
    const [companyName, setCompanyName] = useState(editingPosition?.company_name || '');
    const [sector, setSector] = useState(editingPosition?.sector || '');
    const [entryDate, setEntryDate] = useState(editingPosition?.entry_date || todayIsoDate());
    const [entryPrice, setEntryPrice] = useState(
        editingPosition ? String(editingPosition.entry_price) : '',
    );
    const [entryXu100, setEntryXu100] = useState(
        editingPosition?.entry_xu100 != null ? String(editingPosition.entry_xu100) : '',
    );
    const [weightPct, setWeightPct] = useState(
        editingPosition ? String(editingPosition.weight_pct) : String(Math.min(15, availableWeightPct)),
    );
    const [targetPrice, setTargetPrice] = useState(
        editingPosition ? String(editingPosition.target_price) : '',
    );
    const [horizon, setHorizon] = useState(editingPosition?.horizon || '6-12 Ay');
    const [rating, setRating] = useState(editingPosition?.rating || 'Endeks Üstü Getiri');
    const [thesis, setThesis] = useState(editingPosition?.thesis || '');
    const [catalysts, setCatalysts] = useState<string[]>(editingPosition?.catalysts || []);
    const [customCatalyst, setCustomCatalyst] = useState('');
    const [riskNote, setRiskNote] = useState(editingPosition?.risk_note || '');
    const [revisionNote, setRevisionNote] = useState('');

    // Entry multiples (auto-filled on symbol selection, collapsible manual override)
    const [fk, setFk] = useState(
        editingPosition?.entry_multiples?.fk != null ? String(editingPosition.entry_multiples.fk) : '',
    );
    const [fdFavok, setFdFavok] = useState(
        editingPosition?.entry_multiples?.fd_favok != null ? String(editingPosition.entry_multiples.fd_favok) : '',
    );
    const [pdDd, setPdDd] = useState(
        editingPosition?.entry_multiples?.pd_dd != null ? String(editingPosition.entry_multiples.pd_dd) : '',
    );
    const [showMultiplesOverride, setShowMultiplesOverride] = useState(false);

    const [prefillLoading, setPrefillLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [formError, setFormError] = useState<string | null>(null);
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const prefillAbortRef = useRef<AbortController | null>(null);
    const selectedSymbolRef = useRef((editingPosition?.symbol || '').toUpperCase());
    const sectorEditVersionRef = useRef(0);
    const dialogRef = useAccessibleDialog<HTMLDivElement>(true, onClose);

    const matchingUniverse = useMemo(() => {
        const q = symbolQuery.trim().toUpperCase();
        if (!q) return universeRows.slice(0, 10);
        return universeRows
            .filter(
                (r) =>
                    String(r.symbol || r.company).toUpperCase().includes(q)
                    || String(r.name || '').toUpperCase().includes(q),
            )
            .slice(0, 10);
    }, [symbolQuery, universeRows]);

    const triggerPrefill = useCallback((targetSym: string, targetDate: string) => {
        const cleanSym = targetSym.trim().toUpperCase();
        if (!cleanSym) return;
        prefillAbortRef.current?.abort();
        const controller = new AbortController();
        prefillAbortRef.current = controller;
        setPrefillLoading(true);
        apiClient
            .modelPortfolioPrefill(cleanSym, targetDate, controller.signal)
            .then((res) => {
                setCompanyName(res.company_name || cleanSym);
                if (res.entry_price != null) {
                    setEntryPrice(String(res.entry_price));
                }
                if (res.entry_xu100 != null) setEntryXu100(String(res.entry_xu100));
                if (res.entry_multiples) {
                    setFk(res.entry_multiples.fk != null ? String(res.entry_multiples.fk) : '');
                    setFdFavok(res.entry_multiples.fd_favok != null ? String(res.entry_multiples.fd_favok) : '');
                    setPdDd(res.entry_multiples.pd_dd != null ? String(res.entry_multiples.pd_dd) : '');
                }
            })
            .catch(() => {})
            .finally(() => setPrefillLoading(false));
    }, []);

    const autoSelectSector = useCallback(async (targetSym: string) => {
        const normalizedSymbol = targetSym.trim().toUpperCase();
        if (!normalizedSymbol) return;
        const editVersion = sectorEditVersionRef.current;
        const findSector = (rows: MarketStockRow[]) => rows.find(
            (row) => String(row.symbol || row.company || '').toUpperCase() === normalizedSymbol,
        )?.sector?.trim();

        let resolvedSector = findSector(apiClient.getCachedMarketStocks('XUTUM')?.rows || []);
        if (!resolvedSector) {
            try {
                const payload = await apiClient.marketStocks({ index: 'XUTUM' });
                resolvedSector = findSector(payload.rows || []);
            } catch {
                return;
            }
        }

        if (
            resolvedSector
            && selectedSymbolRef.current === normalizedSymbol
            && sectorEditVersionRef.current === editVersion
        ) {
            setSector(resolvedSector);
        }
    }, []);

    const selectStock = useCallback((targetSym: string, targetName: string) => {
        const normalizedSymbol = targetSym.trim().toUpperCase();
        selectedSymbolRef.current = normalizedSymbol;
        sectorEditVersionRef.current += 1;
        setSymbol(normalizedSymbol);
        setSymbolQuery(normalizedSymbol);
        setCompanyName(targetName || normalizedSymbol);
        setSector('');
        setDropdownOpen(false);
        void autoSelectSector(normalizedSymbol);
        triggerPrefill(normalizedSymbol, entryDate);
    }, [autoSelectSector, entryDate, triggerPrefill]);

    const impliedUpsidePct = useMemo(() => {
        const ep = Number(entryPrice);
        const tp = Number(targetPrice);
        if (!Number.isFinite(ep) || !Number.isFinite(tp) || ep <= 0 || tp <= 0) return null;
        return Number((((tp - ep) / ep) * 100).toFixed(1));
    }, [entryPrice, targetPrice]);

    const applyTargetUpsidePreset = (pct: number) => {
        const ep = Number(entryPrice);
        if (!Number.isFinite(ep) || ep <= 0) return;
        const computed = Number((ep * (1 + pct / 100)).toFixed(2));
        setTargetPrice(String(computed));
    };

    const remainingCashAfterWeight = useMemo(() => {
        const wp = Number(weightPct);
        if (!Number.isFinite(wp)) return availableWeightPct;
        return Math.max(0, Number((availableWeightPct - wp).toFixed(1)));
    }, [availableWeightPct, weightPct]);

    const toggleCatalyst = (label: string) => {
        setCatalysts((prev) =>
            prev.includes(label) ? prev.filter((c) => c !== label) : [...prev, label],
        );
    };

    const addCustomCatalyst = () => {
        const clean = customCatalyst.trim();
        if (!clean) return;
        if (!catalysts.includes(clean)) {
            setCatalysts((prev) => [...prev, clean]);
        }
        setCustomCatalyst('');
    };

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setFormError(null);
        const ep = Number(entryPrice);
        const tp = Number(targetPrice);
        const wp = Number(weightPct);
        if (!symbol.trim()) {
            setFormError('Lütfen bir hisse kodu seçin.');
            return;
        }
        if (!Number.isFinite(ep) || ep <= 0) {
            setFormError('Geçerli bir giriş fiyatı girin.');
            return;
        }
        if (!Number.isFinite(tp) || tp <= 0) {
            setFormError('Geçerli bir hedef fiyat girin.');
            return;
        }
        if (!isEdit && availableWeightPct <= 0) {
            setFormError('Portföy ağırlıkları zaten %100. Yeni hisse eklemek için önce mevcut ağırlıkları azaltın.');
            return;
        }
        if (!Number.isFinite(wp) || wp <= 0 || wp > 100) {
            setFormError('Ağırlık %0 ile %100 arasında olmalıdır.');
            return;
        }
        if (wp > availableWeightPct + 1e-6) {
            setFormError(
                `Kullanılabilir portföy payı %${availableWeightPct.toFixed(1)}. Yeni hisse eklemek için önce mevcut ağırlıkları azaltın.`,
            );
            return;
        }

        const multiplesPayload = {
            fk: fk.trim() ? Number(fk) : null,
            fd_favok: fdFavok.trim() ? Number(fdFavok) : null,
            pd_dd: pdDd.trim() ? Number(pdDd) : null,
        };

        setSaving(true);
        try {
            if (isEdit && editingPosition) {
                const updated = await apiClient.modelPortfolioUpdatePosition(
                    editingPosition.id,
                    {
                        entry_date: entryDate,
                        entry_price: ep,
                        entry_xu100: entryXu100.trim() ? Number(entryXu100) : null,
                        weight_pct: wp,
                        target_price: tp,
                        sector: sector.trim() || null,
                        horizon,
                        rating,
                        thesis,
                        catalysts,
                        risk_note: riskNote,
                        revision_note: revisionNote.trim() || undefined,
                        entry_multiples: multiplesPayload,
                    },
                    adminToken,
                );
                onSaved(updated);
            } else {
                const created = await apiClient.modelPortfolioCreatePosition(
                    {
                        symbol: symbol.trim().toUpperCase(),
                        company_name: companyName.trim() || symbol.trim().toUpperCase(),
                        sector: sector.trim() || null,
                        entry_date: entryDate,
                        entry_price: ep,
                        entry_xu100: entryXu100.trim() ? Number(entryXu100) : null,
                        weight_pct: wp,
                        target_price: tp,
                        horizon,
                        rating,
                        thesis,
                        catalysts,
                        risk_note: riskNote,
                        entry_multiples: multiplesPayload,
                    },
                    adminToken,
                );
                onSaved(created);
            }
        } catch (err: unknown) {
            setFormError(err instanceof Error ? err.message : 'Kayıt sırasında hata oluştu.');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="mp-modal-backdrop" onClick={onClose}>
            <div
                ref={dialogRef}
                className="mp-modal mp-modal-lg"
                role="dialog"
                aria-modal="true"
                aria-labelledby="mp-editor-dialog-title"
                tabIndex={-1}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Sticky Header */}
                <div className="mp-modal-header">
                    <div className="mp-modal-title-row">
                        <BriefcaseBusiness size={18} className="mp-accent-icon" />
                        <div>
                            <h3 id="mp-editor-dialog-title">
                                {isEdit
                                    ? `${editingPosition?.symbol} Pozisyonunu Düzenle / Revize Et`
                                    : 'Model Portföye Yeni Hisse Ekle'}
                            </h3>
                            <span className="mp-modal-subtitle">
                                Hisse ve tarih seçtiğinizde fiyat, XU100 seviyesi ve çarpanlar otomatik gelir. Hedef fiyatı araştırma varsayımınıza göre belirleyin.
                            </span>
                        </div>
                    </div>
                    <button type="button" className="mp-icon-btn" aria-label="Pozisyon formunu kapat" onClick={onClose}>
                        <X size={17} />
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="mp-editor-shell">
                    <div className="mp-modal-body mp-editor-form">
                        {/* SECTION 1: Stock Selection, Date & Sector */}
                        <div className="mp-editor-section-card">
                            <div className="mp-editor-step-header">
                                <span className="mp-step-pill">1</span>
                                <div>
                                    <h4>Hisse, Giriş Tarihi ve Sektör</h4>
                                    <p>BIST hissesini aratıp seçin; veriler mevcut önbellekten anında çekilir.</p>
                                </div>
                            </div>

                            <div className="mp-form-grid-3">
                                <div className="mp-field mp-symbol-picker">
                                    <span className="mp-field-label">Hisse Kodu *</span>
                                    <div className="mp-search-input-wrap">
                                        <Search size={14} className="mp-search-icon" />
                                        <input
                                            type="text"
                                            aria-label="Hisse kodu"
                                            data-dialog-autofocus={!isEdit ? 'true' : undefined}
                                            role="combobox"
                                            aria-autocomplete="list"
                                            aria-expanded={dropdownOpen}
                                            aria-controls="mp-symbol-options"
                                            value={symbolQuery}
                                            disabled={isEdit}
                                            onFocus={() => setDropdownOpen(true)}
                                            onChange={(e) => {
                                                const val = e.target.value.toUpperCase();
                                                const normalizedSymbol = val.trim();
                                                if (normalizedSymbol !== selectedSymbolRef.current) {
                                                    selectedSymbolRef.current = normalizedSymbol;
                                                    sectorEditVersionRef.current += 1;
                                                    setSector('');
                                                }
                                                setSymbolQuery(val);
                                                setSymbol(val);
                                                setDropdownOpen(true);
                                            }}
                                            onBlur={() => {
                                                window.setTimeout(() => setDropdownOpen(false), 180);
                                                const normalizedSymbol = selectedSymbolRef.current;
                                                if (!isEdit && normalizedSymbol) {
                                                    void autoSelectSector(normalizedSymbol);
                                                    triggerPrefill(normalizedSymbol, entryDate);
                                                }
                                            }}
                                            placeholder="THYAO, BIMAS, TUPRS..."
                                        />
                                    </div>
                                    {!isEdit && dropdownOpen && matchingUniverse.length > 0 && (
                                        <div className="mp-symbol-dropdown" id="mp-symbol-options" role="listbox" aria-label="Hisse seçenekleri">
                                            {matchingUniverse.map((row) => {
                                                const sym = String(row.symbol || row.company).toUpperCase();
                                                return (
                                                    <button
                                                        key={sym}
                                                        type="button"
                                                        className="mp-symbol-option"
                                                        role="option"
                                                        aria-selected={symbol === sym}
                                                        onMouseDown={(e) => {
                                                            e.preventDefault();
                                                            selectStock(sym, row.name || sym);
                                                        }}
                                                    >
                                                        <div className="mp-symbol-option-left">
                                                            <SymbolLogo
                                                                symbol={sym}
                                                                name={row.name || sym}
                                                                kind="stock"
                                                                size="xs"
                                                            />
                                                            <strong>{sym}</strong>
                                                        </div>
                                                        <span>{row.name}</span>
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>

                                <div className="mp-field">
                                    <div className="mp-field-label-row">
                                        <span className="mp-field-label">Portföye Giriş Tarihi *</span>
                                        <div className="mp-mini-presets">
                                            <button
                                                type="button"
                                                className="mp-mini-preset-btn"
                                                onClick={() => {
                                                    const d = todayIsoDate();
                                                    setEntryDate(d);
                                                    if (symbol.trim()) triggerPrefill(symbol, d);
                                                }}
                                            >
                                                Bugün
                                            </button>
                                        </div>
                                    </div>
                                    <input
                                        type="date"
                                        aria-label="Portföye giriş tarihi"
                                        value={entryDate}
                                        onChange={(e) => {
                                            const nextDate = e.target.value;
                                            setEntryDate(nextDate);
                                            if (symbol.trim()) triggerPrefill(symbol, nextDate);
                                        }}
                                    />
                                </div>

                                <MpSectorDropdown
                                    value={sector}
                                    onChange={(nextSector) => {
                                        sectorEditVersionRef.current += 1;
                                        setSector(nextSector);
                                    }}
                                />
                            </div>

                            {/* Selected Stock & Auto-Prefilled Reference Strip */}
                            {symbol.trim() && (
                                <div className="mp-selected-stock-banner">
                                    <div className="mp-selected-stock-left">
                                        <SymbolLogo
                                            symbol={symbol.trim().toUpperCase()}
                                            name={companyName || symbol}
                                            kind="stock"
                                            size="sm"
                                        />
                                        <div>
                                            <div className="mp-selected-stock-title">
                                                <strong className="mp-mono">{symbol.trim().toUpperCase()}</strong>
                                                <span>{companyName || 'Şirket seçildi'}</span>
                                            </div>
                                            <div className="mp-selected-stock-multiples mp-mono">
                                                <span>
                                                    XU100: <b>{entryXu100 ? Number(entryXu100).toLocaleString('tr-TR') : '—'}</b>
                                                </span>
                                                <span>
                                                    F/K: <b>{fk ? `${Number(fk).toFixed(2)}x` : '—'}</b>
                                                </span>
                                                <span>
                                                    FD/FAVÖK: <b>{fdFavok ? `${Number(fdFavok).toFixed(2)}x` : '—'}</b>
                                                </span>
                                                <span>
                                                    PD/DD: <b>{pdDd ? `${Number(pdDd).toFixed(2)}x` : '—'}</b>
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                    <button
                                        type="button"
                                        className="mp-btn mp-btn-subtle mp-btn-sm"
                                        onClick={() => setShowMultiplesOverride((p) => !p)}
                                    >
                                        <SlidersHorizontal size={12} />
                                        <span>{showMultiplesOverride ? 'Çarpanları Gizle' : 'Çarpanları Düzenle'}</span>
                                    </button>
                                </div>
                            )}

                            {showMultiplesOverride && (
                                <div className="mp-form-grid-4 mp-override-grid">
                                    <label className="mp-field">
                                        <span className="mp-field-label">Giriş Günü XU100</span>
                                        <input
                                            type="number"
                                            step="0.01"
                                            value={entryXu100}
                                            onChange={(e) => setEntryXu100(e.target.value)}
                                            placeholder="Otomatik"
                                        />
                                    </label>
                                    <label className="mp-field">
                                        <span className="mp-field-label">Giriş F/K</span>
                                        <input
                                            type="number"
                                            step="0.01"
                                            value={fk}
                                            onChange={(e) => setFk(e.target.value)}
                                            placeholder="Otomatik"
                                        />
                                    </label>
                                    <label className="mp-field">
                                        <span className="mp-field-label">Giriş FD/FAVÖK</span>
                                        <input
                                            type="number"
                                            step="0.01"
                                            value={fdFavok}
                                            onChange={(e) => setFdFavok(e.target.value)}
                                            placeholder="Otomatik"
                                        />
                                    </label>
                                    <label className="mp-field">
                                        <span className="mp-field-label">Giriş PD/DD</span>
                                        <input
                                            type="number"
                                            step="0.01"
                                            value={pdDd}
                                            onChange={(e) => setPdDd(e.target.value)}
                                            placeholder="Otomatik"
                                        />
                                    </label>
                                </div>
                            )}
                        </div>

                        {/* SECTION 2: Price, Target Price & Weight with Quick Presets */}
                        <div className="mp-editor-section-card">
                            <div className="mp-editor-step-header">
                                <span className="mp-step-pill">2</span>
                                <div>
                                    <h4>Giriş Fiyatı, Hedef Fiyat & Portföy Ağırlığı</h4>
                                    <p>Hızlı yüzde butonlarıyla hedef fiyatınızı ve portföy ağırlığını tek tıkla belirleyin.</p>
                                </div>
                            </div>

                            <div className="mp-form-grid-3">
                                <div className="mp-field">
                                    <div className="mp-field-label-row">
                                        <span className="mp-field-label">Giriş Fiyatı (₺) *</span>
                                        {prefillLoading && <em className="mp-prefill-tag">Çekiliyor...</em>}
                                    </div>
                                    <input
                                        type="number"
                                        aria-label="Giriş fiyatı"
                                        step="0.01"
                                        value={entryPrice}
                                        onChange={(e) => setEntryPrice(e.target.value)}
                                        placeholder="0.00"
                                    />
                                    <span className="mp-field-micro-hint">
                                        Seçilen tarihteki kapanış/canlı fiyat otomatik gelir.
                                    </span>
                                </div>

                                <div className="mp-field">
                                    <div className="mp-field-label-row">
                                        <span className="mp-field-label">Hedef Fiyat (₺) *</span>
                                        {impliedUpsidePct != null && (
                                            <span className={`mp-upside-live-pill ${toneClass(impliedUpsidePct)}`}>
                                                Potansiyel: {formatPct(impliedUpsidePct)}
                                            </span>
                                        )}
                                    </div>
                                    <input
                                        type="number"
                                        aria-label="Hedef fiyat"
                                        step="0.01"
                                        value={targetPrice}
                                        onChange={(e) => setTargetPrice(e.target.value)}
                                        placeholder="0.00"
                                    />
                                    <div className="mp-quick-pills-row">
                                        {[25, 35, 50, 75].map((pct) => (
                                            <button
                                                key={pct}
                                                type="button"
                                                className="mp-quick-pill"
                                                disabled={!Number(entryPrice)}
                                                onClick={() => applyTargetUpsidePreset(pct)}
                                                title={`Giriş fiyatının %${pct} üzerini hesapla`}
                                            >
                                                +%{pct}
                                            </button>
                                        ))}
                                    </div>
                                    <span className="mp-field-micro-hint">
                                        Hızlı oranlar giriş fiyatına göre örnek hesaplamadır; seçilen hedefi ayrıca gözden geçirin.
                                    </span>
                                </div>

                                <div className="mp-field">
                                    <div className="mp-field-label-row">
                                        <span className="mp-field-label">Portföy Ağırlığı (%) *</span>
                                        <span className="mp-field-micro-hint mp-mono">
                                            Kalan Pay: %{remainingCashAfterWeight.toFixed(1)}
                                        </span>
                                    </div>
                                    <input
                                        type="number"
                                        aria-label="Portföy ağırlığı yüzde"
                                        step="0.5"
                                        min="0.5"
                                        max="100"
                                        value={weightPct}
                                        onChange={(e) => setWeightPct(e.target.value)}
                                    />
                                    <div className="mp-quick-pills-row">
                                        {[5, 10, 15, 20].map((w) => (
                                            <button
                                                key={w}
                                                type="button"
                                                className={`mp-quick-pill${Number(weightPct) === w ? ' active' : ''}`}
                                                onClick={() => setWeightPct(String(Math.min(w, availableWeightPct)))}
                                            >
                                                %{w}
                                            </button>
                                        ))}
                                        <button
                                            type="button"
                                            className="mp-quick-pill"
                                            onClick={() => setWeightPct(String(availableWeightPct))}
                                        >
                                            Maks (%{availableWeightPct.toFixed(0)})
                                        </button>
                                    </div>
                                    {!isEdit && availableWeightPct <= 0 && (
                                        <span className="mp-field-micro-hint">
                                            Portföy %100 dolu; yeni hisse eklemek için mevcut ağırlıklardan pay açın.
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* SECTION 3: Custom Explained Dropdowns for Rating & Horizon */}
                        <div className="mp-editor-section-card">
                            <div className="mp-editor-step-header">
                                <span className="mp-step-pill">3</span>
                                <div>
                                    <h4>Tavsiye Derecesi ve Beklenen Vade Ufku</h4>
                                    <p>Önerinizin stratejik sınıfını ve hedef fiyat vadesini açıklamalı menüden seçin.</p>
                                </div>
                            </div>

                            <div className="mp-form-grid-2">
                                <MpExplainedDropdown
                                    label="Tavsiye Etiketi"
                                    value={rating}
                                    options={RATING_OPTIONS}
                                    onChange={setRating}
                                />
                                <MpExplainedDropdown
                                    label="Beklenen Vade Ufku"
                                    value={horizon}
                                    options={HORIZON_OPTIONS}
                                    onChange={setHorizon}
                                />
                            </div>
                        </div>

                        {/* SECTION 4: Investment Thesis, Catalysts & Risk */}
                        <div className="mp-editor-section-card">
                            <div className="mp-editor-step-header">
                                <span className="mp-step-pill">4</span>
                                <div>
                                    <h4>Yatırım gerekçesi ve beklenti</h4>
                                    <p>Bu alanlar isteğe bağlı. Kısa bir not ve varsa beklentilerinizi ekleyin.</p>
                                </div>
                            </div>

                            <div className="mp-field">
                                <span className="mp-field-label">Beklenti etiketleri</span>
                                <small className="mp-field-micro-hint">Uygun olanları seçin; birden fazla seçebilirsiniz.</small>
                                <div className="mp-catalyst-choices" role="group" aria-label="Beklenti etiketleri">
                                    {CATALYST_OPTIONS.map((item) => {
                                        const active = catalysts.includes(item.value);
                                        return (
                                            <button
                                                key={item.value}
                                                type="button"
                                                className={`mp-catalyst-choice${active ? ' active' : ''}`}
                                                aria-pressed={active}
                                                onClick={() => toggleCatalyst(item.value)}
                                            >
                                                {active && <Check size={13} />}
                                                {item.label}
                                            </button>
                                        );
                                    })}
                                    {catalysts
                                        .filter((item) => !CATALYST_OPTIONS.some((option) => option.value === item))
                                        .map((item) => (
                                            <button
                                                key={item}
                                                type="button"
                                                className="mp-catalyst-choice active"
                                                aria-pressed="true"
                                                onClick={() => toggleCatalyst(item)}
                                            >
                                                <Check size={13} />
                                                {item}
                                            </button>
                                        ))}
                                </div>
                                <div className="mp-custom-catalyst-row">
                                    <input
                                        type="text"
                                        aria-label="Başka bir beklenti etiketi ekle"
                                        value={customCatalyst}
                                        onChange={(e) => setCustomCatalyst(e.target.value)}
                                        placeholder="Başka bir beklenti ekleyin (isteğe bağlı)"
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter') {
                                                e.preventDefault();
                                                addCustomCatalyst();
                                            }
                                        }}
                                    />
                                    <button
                                        type="button"
                                        className="mp-btn mp-btn-subtle"
                                        onClick={addCustomCatalyst}
                                        disabled={!customCatalyst.trim()}
                                    >
                                        Ekle
                                    </button>
                                </div>
                            </div>

                            <div className="mp-field">
                                <span className="mp-field-label">Yatırım gerekçesi</span>
                                <small className="mp-field-micro-hint">Bir iki cümleyle ana beklentinizi yazmanız yeterli.</small>
                                <textarea
                                    rows={3}
                                    aria-label="Yatırım gerekçesi"
                                    value={thesis}
                                    onChange={(e) => setThesis(e.target.value)}
                                    placeholder="Bu hisseyi neden seçtiniz? (isteğe bağlı)"
                                />
                            </div>

                            <label className="mp-field">
                                <span className="mp-field-label">Takip edeceğim risk (isteğe bağlı)</span>
                                <textarea
                                    rows={2}
                                    aria-label="Takip edeceğim risk"
                                    value={riskNote}
                                    onChange={(e) => setRiskNote(e.target.value)}
                                    placeholder="Örn. maliyet artışı veya talep düşüşü"
                                />
                            </label>

                            {isEdit && (
                                <label className="mp-field">
                                    <span className="mp-field-label">Revizyon Notu (Değişiklik Günlüğünde Görünür)</span>
                                    <input
                                        type="text"
                                        value={revisionNote}
                                        onChange={(e) => setRevisionNote(e.target.value)}
                                        placeholder="Örn: Güçlü 3Ç bilançosu sonrası hedef fiyat yukarı revize edildi."
                                    />
                                </label>
                            )}
                        </div>

                        {formError && <div className="mp-form-error">{formError}</div>}
                    </div>

                    {/* Sticky Footer */}
                    <div className="mp-modal-footer mp-modal-footer-sticky">
                        <div className="mp-footer-summary">
                            {symbol.trim() ? (
                                <span>
                                    <b>{symbol.trim().toUpperCase()}</b> · Ağırlık: <b>%{weightPct || 0}</b>
                                    {impliedUpsidePct != null && (
                                        <>
                                            {' '}
                                            · Hedef Potansiyel: <b>{formatPct(impliedUpsidePct)}</b>
                                        </>
                                    )}
                                </span>
                            ) : (
                                <span>Başlamak için bir BIST hisse kodu seçin.</span>
                            )}
                        </div>
                        <div className="mp-footer-btns">
                            <button type="button" className="mp-btn mp-btn-ghost" onClick={onClose}>
                                İptal
                            </button>
                            <button type="submit" className="mp-btn mp-btn-primary" disabled={saving}>
                                {saving ? 'Kaydediliyor...' : isEdit ? 'Değişiklikleri Kaydet' : 'Model Portföye Ekle'}
                            </button>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    );
}

/* ==========================================================================
   Subcomponent 5: Close Position Modal with Explained Reason Dropdown
   ========================================================================== */

function ModelPortfolioCloseModal({
    adminToken,
    position,
    currentXu100,
    onClose,
    onClosed,
}: {
    adminToken: string;
    position: ModelPortfolioPosition;
    currentXu100: number | null;
    onClose: () => void;
    onClosed: (nextData: ModelPortfolioResponse) => void;
}) {
    const [exitPrice, setExitPrice] = useState(
        String(position.current_price ?? position.entry_price),
    );
    const [closedDate, setClosedDate] = useState(todayIsoDate());
    const [closeReason, setCloseReason] = useState(CLOSE_REASON_OPTIONS[0].value);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const dialogRef = useAccessibleDialog<HTMLDivElement>(true, onClose);

    const previewReturnPct = useMemo(() => {
        const ep = Number(exitPrice);
        if (!Number.isFinite(ep) || ep <= 0 || position.entry_price <= 0) return null;
        return Number((((ep - position.entry_price) / position.entry_price) * 100).toFixed(2));
    }, [exitPrice, position.entry_price]);

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        const ep = Number(exitPrice);
        if (!Number.isFinite(ep) || ep <= 0) {
            setError('Geçerli bir çıkış fiyatı girin.');
            return;
        }
        setSaving(true);
        setError(null);
        try {
            const res = await apiClient.modelPortfolioClosePosition(
                position.id,
                {
                    exit_price: ep,
                    closed_date: closedDate,
                    exit_xu100: currentXu100,
                    close_reason: closeReason,
                },
                adminToken,
            );
            onClosed(res);
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'Pozisyon kapatılamadı.');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="mp-modal-backdrop" onClick={onClose}>
            <div
                ref={dialogRef}
                className="mp-modal mp-modal-md"
                role="dialog"
                aria-modal="true"
                aria-labelledby="mp-close-dialog-title"
                tabIndex={-1}
                onClick={(e) => e.stopPropagation()}
            >
                <div className="mp-modal-header">
                    <div className="mp-modal-title-row">
                        <CheckCircle2 size={18} className="mp-accent-icon" />
                        <div>
                            <h3 id="mp-close-dialog-title">{position.symbol} Pozisyonunu Kapat</h3>
                            <span className="mp-modal-subtitle">
                                Gerçekleşen getiri ve BIST 100 rölatif alfa karnesiyle geçmiş işlemlere arşivlenir.
                            </span>
                        </div>
                    </div>
                    <button type="button" className="mp-icon-btn" aria-label="Kapatma penceresini kapat" onClick={onClose}>
                        <X size={16} />
                    </button>
                </div>
                <form onSubmit={handleSubmit} className="mp-editor-shell">
                    <div className="mp-modal-body">
                        <div className="mp-form-grid-2">
                            <label className="mp-field">
                                <div className="mp-field-label-row">
                                    <span className="mp-field-label">Çıkış Fiyatı (₺) *</span>
                                    {previewReturnPct != null && (
                                        <span className={`mp-upside-live-pill ${toneClass(previewReturnPct)}`}>
                                            Net Getiri: {formatPct(previewReturnPct)}
                                        </span>
                                    )}
                                </div>
                                <input
                                    type="number"
                                    data-dialog-autofocus
                                    step="0.01"
                                    value={exitPrice}
                                    onChange={(e) => setExitPrice(e.target.value)}
                                />
                            </label>
                            <label className="mp-field">
                                <span className="mp-field-label">Kapanış Tarihi *</span>
                                <input
                                    type="date"
                                    value={closedDate}
                                    onChange={(e) => setClosedDate(e.target.value)}
                                />
                            </label>
                        </div>

                        <MpExplainedDropdown
                            label="Kapanış Gerekçesi"
                            value={closeReason}
                            options={CLOSE_REASON_OPTIONS}
                            onChange={setCloseReason}
                        />

                        {error && <div className="mp-form-error">{error}</div>}
                    </div>
                    <div className="mp-modal-footer mp-modal-footer-sticky">
                        <div className="mp-footer-summary">
                            <span>Giriş: <b>{formatPrice(position.entry_price)}</b></span>
                        </div>
                        <div className="mp-footer-btns">
                            <button type="button" className="mp-btn mp-btn-ghost" onClick={onClose}>
                                İptal
                            </button>
                            <button type="submit" className="mp-btn mp-btn-primary" disabled={saving}>
                                {saving ? 'Kapatılıyor...' : 'Pozisyonu Kapat'}
                            </button>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    );
}

