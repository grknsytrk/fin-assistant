export type PortfolioTab = 'active' | 'closed' | 'changelog';
export type PerfRange = '1w' | '1m' | '3m' | '6m' | 'ytd' | '1y' | 'all';
export type PerformanceSeries = 'portfolio' | 'xu100' | 'xu030';

export type ExplainedOption = {
    value: string;
    label: string;
    shortLabel?: string;
    badge: string;
    tone: 'emerald' | 'blue' | 'amber' | 'purple';
    description: string;
};

export type DonutItem = {
    symbol: string;
    label: string;
    weight: number;
    returnPct: number | null;
    attributionPct: number;
    color: string;
    isCash: boolean;
};

export const PERF_RANGES: Array<{ id: PerfRange; label: string }> = [
    { id: '1w', label: '1H' },
    { id: '1m', label: '1A' },
    { id: '3m', label: '3A' },
    { id: '6m', label: '6A' },
    { id: 'ytd', label: 'YTD' },
    { id: '1y', label: '1Y' },
    { id: 'all', label: 'Tüm Veri' },
];

export const RATING_OPTIONS: ExplainedOption[] = [
    {
        value: 'Endeks Üstü Getiri',
        label: 'Endeks Üstü Getiri (AL)',
        shortLabel: 'Endeks Üstü (AL)',
        badge: 'AL · ANA ÖNERİ',
        tone: 'emerald',
        description: 'BIST 100 endeksinden belirgin şekilde pozitif ayrışması beklenen standart model portföy hissesi.',
    },
    {
        value: 'Güçlü AL / Model Portföy',
        label: 'Güçlü AL / En Çok Beğenilen',
        shortLabel: 'Güçlü AL (Favori)',
        badge: 'GÜÇLÜ AL · FAVORİ',
        tone: 'blue',
        description: 'Derin çarpan iskontosu ve yakın vadeli güçlü bilanço/haber katalizörü barındıran yüksek beklentili seçim.',
    },
    {
        value: 'Uzun Vadeli Biriktir',
        label: 'Uzun Vadeli Biriktir',
        shortLabel: 'Uzun Vadeli Biriktir',
        badge: 'BİRİKTİR · DEFANSİF',
        tone: 'amber',
        description: 'Kademeli alıma uygun, düzenli nakit akışı, temettü verimi ve bileşik büyüme odaklı çekirdek pozisyon.',
    },
    {
        value: 'Taktiksel / Kısa-Orta Vade',
        label: 'Taktiksel / Dönemsel Fırsat',
        shortLabel: 'Taktiksel / Dönemsel',
        badge: 'TAKTİKSEL · DÖNEMSEL',
        tone: 'purple',
        description: 'Çeyreklik bilanço beklentisi, mevsimsellik veya sektörel haber akışı odaklı taktiksel getiri fırsatı.',
    },
];

export const HORIZON_OPTIONS: ExplainedOption[] = [
    {
        value: '1-3 Ay',
        label: '1-3 Ay (Çeyreklik / Kısa Vade)',
        shortLabel: '1-3 Ay (Kısa Vade)',
        badge: 'KISA VADE',
        tone: 'purple',
        description: 'Yaklaşan finansal sonuçlar veya kısa vadeli katalizörlerin fiyatlanması beklenen vade.',
    },
    {
        value: '3-6 Ay',
        label: '3-6 Ay (Orta Vade)',
        shortLabel: '3-6 Ay (Orta Vade)',
        badge: 'ORTA VADE',
        tone: 'amber',
        description: '1-2 çeyreklik marj iyileşmesi ve hedef çarpanlara yakınsama beklenen dönem.',
    },
    {
        value: '6-12 Ay',
        label: '6-12 Ay (Standart Model Portföy)',
        shortLabel: '6-12 Ay (Standart)',
        badge: '12 AY HEDEF',
        tone: 'blue',
        description: 'Aracı kurum araştırma raporları ve model portföyler için standart 1 yıllık hedef fiyat ufku.',
    },
    {
        value: '12+ Ay',
        label: '12+ Ay (Uzun Vadeli Yatırım)',
        shortLabel: '12+ Ay (Uzun Vade)',
        badge: 'UZUN VADE',
        tone: 'emerald',
        description: 'Yeni kapasite devreye girişi, ihracat pazarı genişlemesi veya çok yıllık büyüme hikayesi.',
    },
];

export const SECTOR_OPTIONS: Array<{ value: string; badge: string; description: string }> = [
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

export const CLOSE_REASON_OPTIONS: ExplainedOption[] = [
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

export const CATALYST_OPTIONS = [
    { value: 'Çarpan İskontosu', label: 'Çarpan iskontosu' },
    { value: 'Güçlü Bilanço Beklentisi', label: 'Güçlü bilanço' },
    { value: 'İhracat & Döviz Geliri', label: 'İhracat geliri' },
    { value: 'Yüksek Nakit Akışı', label: 'Nakit akışı' },
    { value: 'Yeni Kapasite Yatırımı', label: 'Yeni yatırım' },
    { value: 'Temettü Verimi', label: 'Temettü' },
    { value: 'Marj Genişlemesi', label: 'Marj artışı' },
    { value: 'Borçluluk Azalışı', label: 'Borç azalması' },
];

export const DONUT_PALETTE = [
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
