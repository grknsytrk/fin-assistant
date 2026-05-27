export type SymbolLogoKind = 'stock' | 'index' | 'commodity' | 'fx' | 'fund';

export const STOCK_LOGO_DOMAIN_MAP: Record<string, string | string[]> = {
    AEFES: 'anadoluefes.com',
    AGHOL: 'anadolugroup.com',
    AKBNK: 'akbank.com',
    AKSA: 'aksa.com',
    AKSEN: 'aksaenerji.com.tr',
    ALARK: 'alarko.com.tr',
    ARCLK: 'arcelikglobal.com',
    ASELS: 'aselsan.com',
    ANSGR: 'anadolusigorta.com',
    ASTOR: 'astoras.com.tr',
    ALTNY: 'altinaysavunma.com',
    BALSU: 'balsugida.com',
    BRSAN: 'borusan.com',
    BSOKE: 'baticim.com',
    BTCIM: 'baticim.com',
    CANTE: 'can2termik.com.tr',
    CWENE: 'cw-enerji.com',
    BRYAT: 'borusanyatirim.com',
    BIMAS: 'bim.com.tr',
    CCOLA: 'coca-cola.com.tr',
    DOHOL: 'doganholding.com.tr',
    DAPGM: 'dapgayrimenkulgelistirme.com.tr',
    DOAS:  'dogusotomotiv.com.tr',
    DSTKF: 'destekfaktoring.com.tr',
    ECILC: 'eczacibasi.com.tr',
    EGEEN: 'egeendustri.com.tr',
    EKGYO: 'emlakkonut.com.tr',
    ENJSA: 'enerjisa.com.tr',
    ENKAI: 'enka.com',
    EREGL: 'erdemir.com.tr',
    ENERY: 'enerya.com.tr',
    EBEBK: 'e-bebek.com',
    ECOGR: 'ecogreenenerji.com',
    EGEGY: 'egeyapigyo.com',
    EUPWR: 'europowerenerji.com.tr',
    FENER: 'fenerbahce.org/fbfutbol/',
    FROTO: 'ford.com.tr',
    GARAN: 'garanti.com.tr',
    GENIL: 'genilac.com.tr',
    GESAN: 'girisimelk.com.tr',
    GLRMK: 'gulermak.com.tr',
    GRSEL: 'gurseltur.com.tr',
    GRTHO: 'grainturk.com',
    GSRAY: 'sportif.galatasaray.org',
    GUBRF: 'gubretas.com.tr',
    GUNDG: 'gundogdugida.com.tr',
    HALKB: 'halkbank.com.tr',
    HEKTS: 'hektas.com.tr',
    ISCTR: 'isbank.com.tr',
    ISMEN: ['isyatirim.com.tr', 'isinvestment.com'],
    IZENR: 'izdemirenerji.com',
    KCAER: ['kocaercelik.com', 'kocaersteel.com'],
    KCHOL: 'koc.com.tr',
    KLRHO: 'kilerholding.com.tr',
    KONTR: 'kontrolmatik.com',
    KRDMD: 'kardemir.com',
    KTLEV: 'katilimevim.com.tr',
    MAGEN: 'margunenerji.com.tr',
    MAVI: ['mavi.com', 'maviyatirimciliskileri.com', 'mavicompany.com'],
    MGROS: 'migros.ch',
    MIATK: 'miateknoloji.com',
    MPARK: ['mlpcare.com', 'investor.mlpcare.com'],
    ODINE: 'odine.com',
    OBAMS: 'obamakarna.com.tr',
    OTKAR: 'otokar.com.tr',
    OYAKC: 'oyakcimento.com',
    PASEU: 'pasifikeurasia.com.tr',
    PATEK: 'pasifikteknoloji.com',
    PETKM: 'petkim.com.tr',
    PGSUS: 'flypgs.com',
    QUAGR: 'qua.com.tr',
    RALYH: 'ralyatirim.com',
    REEDR: 'reeder.com.tr',
    SAHOL: 'sabancidx.com',
    SASA: 'sasa.com.tr',
    SKBNK: 'sekerbank.com.tr',
    SISE: 'sisecam.com.tr',
    SOKM: 'sokmarket.com.tr',
    TABGD: 'tabgida.com.tr',
    TCELL: 'turkcell.com.tr',
    TERA: 'terayatirim.com',
    TEHOL: 'terayatirim.com',
    TAVHL: 'tavhavalimanlari.com.tr',
    THYAO: 'turkishairlines.com',
    TKFEN: 'tekfen.com.tr',
    TOASO: 'tofas.com.tr',
    TSPOR: 'trabzonspor.org.tr',
    TTKOM: 'turktelekom.com.tr',
    TTRAK: 'turktraktor.com.tr',
    TRALT: 'turkaltinisletmeleri.com',
    TRHOL: 'dagi.com.tr',
    TRENJ: 'turkaltinisletmeleri.com',
    TRMET: 'turkaltinisletmeleri.com',
    TUREX: 'turexturizm.com.tr',
    TURSG: 'turkiyesigorta.com.tr',
    TUPRS: 'tupras.com.tr',
    ULKER: 'ulker.com.tr',
    VAKBN: 'vakifbank.com.tr',
    VESTL: 'vestelinternational.com/tr/yatirimci-iliskileri',
    YEOTK: 'yeo.com.tr',
    YKBNK: 'yapikredi.com.tr',
    ZOREN: 'zorluenerji.com.tr',
    
};

const COMPANY_TOKEN_STOPWORDS = new Set([
    'a',
    'as',
    'anonim',
    'sirket',
    'sirketi',
    'sirketi',
    'holding',
    'holdings',
    'grup',
    'grubu',
    'group',
    've',
    'sanayi',
    'ticaret',
    'yatirim',
    'yatirimlari',
    'bankasi',
    'bank',
    'turkiye',
    'turk',
]);

function foldTurkishToAscii(raw: string): string {
    return raw
        .toLowerCase()
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .replace(/ç/g, 'c')
        .replace(/ğ/g, 'g')
        .replace(/ı/g, 'i')
        .replace(/ö/g, 'o')
        .replace(/ş/g, 's')
        .replace(/ü/g, 'u');
}

function uniqueLimited(values: Array<string | null | undefined>, limit = 10): string[] {
    const out: string[] = [];
    for (const value of values) {
        const normalized = String(value || '')
            .trim()
            .toLowerCase();
        if (!normalized || out.includes(normalized)) continue;
        out.push(normalized);
        if (out.length >= limit) break;
    }
    return out;
}

function uniquePreserveCase(values: Array<string | null | undefined>, limit = 10): string[] {
    const out: string[] = [];
    const seen = new Set<string>();
    for (const value of values) {
        const text = String(value || '').trim();
        const key = text.toLowerCase();
        if (!text || seen.has(key)) continue;
        out.push(text);
        seen.add(key);
        if (out.length >= limit) break;
    }
    return out;
}

export const LOCAL_SYMBOL_ICON_MAP: Record<string, string> = {
    XUTUM: '/market-icons/index-xu100.svg',
    XU100: '/market-icons/index-xu100.svg',
    XU030: '/market-icons/index-xu030.svg',
    SP500: '/market-icons/index-sp500.svg',
    NASDAQ: '/market-icons/index-nasdaq.svg',
    DOW: '/market-icons/index-dow.svg',
    DAX: '/market-icons/index-dax.svg',
    FTSE: '/market-icons/index-ftse.svg',
    NIKKEI: '/market-icons/index-nikkei.svg',
    HANGSENG: '/market-icons/index-hangseng.svg',
    CAC40: '/market-icons/index-cac40.svg',
    BRENT: '/market-icons/commodity-brent.svg',
    WTI: '/market-icons/commodity-wti.svg',
    USOIL: '/market-icons/commodity-wti.svg',
    ALTIN: '/market-icons/commodity-altin.svg',
    GUMUS: '/market-icons/commodity-gumus.svg',
    DOGALGAZ: '/market-icons/commodity-dogalgaz.svg',
    BAKIR: '/market-icons/commodity-bakir.svg',
    PLATIN: '/market-icons/commodity-platin.svg',
    PALADYUM: '/market-icons/commodity-paladyum.svg',
    KAHVE: '/market-icons/commodity-kahve.svg',
    SEKER: '/market-icons/commodity-seker.svg',
    BUGDAY: '/market-icons/commodity-bugday.svg',
    MISIR: '/market-icons/commodity-misir.svg',
    PAMUK: '/market-icons/commodity-pamuk.svg',
    KAKAO: '/market-icons/commodity-kakao.svg',
    SOYA: '/market-icons/commodity-soya.svg',
    'USD/TRY': '/market-icons/fx-usdtry.svg',
    'EUR/TRY': '/market-icons/fx-eurtry.svg',
    'GBP/TRY': '/market-icons/fx-gbptry.svg',
    'EUR/USD': '/market-icons/fx-eurusd.svg',
    DXY: '/market-icons/fx-dxy.svg',
};

export const INDEX_LOGO_SLUGS: Record<string, string> = {
    DAX: 'country/DE--big',
    FTSE: 'country/GB--big',
    NIKKEI: 'country/JP--big',
    HANGSENG: 'country/CN--big',
    CAC40: 'country/FR--big',
    XUTUM: 'country/TR--big',
    XU100: 'country/TR--big',
    XU030: 'country/TR--big',
    XUSIN: 'country/TR--big',
    XUHIZ: 'country/TR--big',
    XUMAL: 'country/TR--big',
    XUTEK: 'country/TR--big',
    XBANK: 'country/TR--big',
    XAKUR: 'country/TR--big',
    XBLSM: 'country/TR--big',
    XELKT: 'country/TR--big',
    XFINK: 'country/TR--big',
    XGMYO: 'country/TR--big',
    XGIDA: 'country/TR--big',
    XHOLD: 'country/TR--big',
    XILTM: 'country/TR--big',
    XINSA: 'country/TR--big',
    XKAGT: 'country/TR--big',
    XKMYA: 'country/TR--big',
    XMADN: 'country/TR--big',
    XMANA: 'country/TR--big',
    XMESY: 'country/TR--big',
    XSGRT: 'country/TR--big',
    XSPOR: 'country/TR--big',
    XTAST: 'country/TR--big',
    XTCRT: 'country/TR--big',
    XTEKS: 'country/TR--big',
    XTRZM: 'country/TR--big',
    XULAS: 'country/TR--big',
    XYORT: 'country/TR--big',
};

export const FX_COUNTRY_FLAGS: Record<string, { base: string; quote?: string }> = {
    'USD/TRY': { base: 'US', quote: 'TR' },
    'EUR/TRY': { base: 'EU', quote: 'TR' },
    'GBP/TRY': { base: 'GB', quote: 'TR' },
    'CHF/TRY': { base: 'CH', quote: 'TR' },
    'AUD/TRY': { base: 'AU', quote: 'TR' },
    'CAD/TRY': { base: 'CA', quote: 'TR' },
    'JPY/TRY': { base: 'JP', quote: 'TR' },
    'CNY/TRY': { base: 'CN', quote: 'TR' },
    'EUR/USD': { base: 'EU', quote: 'US' },
    'GBP/USD': { base: 'GB', quote: 'US' },
    'USD/JPY': { base: 'US', quote: 'JP' },
    'EUR/JPY': { base: 'EU', quote: 'JP' },
    'GBP/JPY': { base: 'GB', quote: 'JP' },
    'USD/CNY': { base: 'US', quote: 'CN' },
    'EUR/CNY': { base: 'EU', quote: 'CN' },
    'GBP/CNY': { base: 'GB', quote: 'CN' },
    'CNY/JPY': { base: 'CN', quote: 'JP' },
    'CHF/JPY': { base: 'CH', quote: 'JP' },
    DXY: { base: 'US' },
};

export const TRADINGVIEW_SYMBOL_SLUG_MAP: Record<string, string> = {
    SP500: 'indices/s-and-p-500--big',
    NASDAQ: 'indices/nasdaq-composite--big',
    DOW: 'dow',
    ...INDEX_LOGO_SLUGS,
    BRENT: 'crude-oil--big',
    WTI: 'crude-oil',
    ALTIN: 'metal/gold--big',
    GUMUS: 'metal/silver--big',
    DOGALGAZ: 'natural-gas',
    BAKIR: 'metal/copper--big',
    PLATIN: 'metal/platinum--big',
    PALADYUM: 'metal/palladium--big',
    KAHVE: 'commodity/coffee--big',
    SEKER: 'commodity/sugar--big',
    BUGDAY: 'commodity/wheat--big',
    MISIR: 'commodity/corn--big',
    PAMUK: 'commodity/cotton--big',
    KAKAO: 'cocoa',
    SOYA: 'commodity/soybean--big',
    USOIL: 'crude-oil',
    'USD/TRY': 'usd',
    'EUR/TRY': 'eur',
    AKBNK: 'akbank',
    AEFES: 'anadolu-efes',
    ASELS: 'aselsan',
    BIMAS: 'bim',
    ENKAI: 'enka-insaat',
    EREGL: 'eregli-demir',
    EBEBK: 'ebebek-magazacilik--big',
    ECOGR: 'ecogreen-enerji-as--600.png',
    EGEGY: 'egeyapi-avrupa-gmyo--600.png',
    GUNDG: 'gundogdu-gida--big',
    KCHOL: 'koc',
    KTLEV: 'katilimevim-tas-fin--big',
    ODINE: 'odine-teknoloji--big',
    PASEU: 'pasifik-eurasia-lojistik--big',
    PGSUS: 'pegasus--big',
    TAVHL: 'tav-havalimanlari',
    THYAO: 'turkish-airlines',
    TUPRS: 'tupras',
    YKBNK: 'yapi-ve-kredi--big',
};

const rawLogoDevToken = (import.meta.env.VITE_LOGO_DEV_TOKEN as string | undefined)?.trim();
const rawTradingViewFlag = (import.meta.env.VITE_ENABLE_TRADINGVIEW_LOGO_FALLBACK as string | undefined)?.trim().toLowerCase();
const rawFintablesLogoFlag = (import.meta.env.VITE_FINTABLES_LOGOS_ENABLED as string | undefined)?.trim().toLowerCase();

export const LOGO_DEV_TOKEN = rawLogoDevToken || '';
export const LOGO_DEV_ENABLED = Boolean(LOGO_DEV_TOKEN);
export const TRADINGVIEW_FALLBACK_ENABLED = rawTradingViewFlag === '1' || rawTradingViewFlag === 'true' || rawTradingViewFlag === 'yes';
export const FINTABLES_LOGOS_ENABLED = !['0', 'false', 'no', 'off'].includes(rawFintablesLogoFlag || '');
const FINTABLES_LOGO_BASE_URL = 'https://storage.fintables.com/media/uploads/company-logos';
const FINTABLES_FUND_MANAGER_LOGO_BASE_URL = 'https://storage.fintables.com/media/uploads/fund-management-logos';
const SYMBOL_LOGO_ALIAS_MAP: Record<string, string> = {
    TEHOL: 'TERA',
};
const EXPLICIT_SYMBOL_LOGO_URL_MAP: Record<string, string[]> = {
    EBEBK: ['https://s3-symbol-logo.tradingview.com/ebebek-magazacilik--big.svg'],
    GUNDG: ['https://s3-symbol-logo.tradingview.com/gundogdu-gida--big.svg'],
    TRHOL: ['https://s3-symbol-logo.tradingview.com/dagi-yatirim-holding--big.svg'],
    TEHOL: [
        `${FINTABLES_LOGO_BASE_URL}/tera_icon.png`,
        `${FINTABLES_LOGO_BASE_URL}/TERA.png`,
    ],
};
const FUND_MANAGER_LOGO_SLUG_MAP: Record<string, string[]> = {
    a1_capital_portfoy: ['a1_portfoy_icon_So2sGTy', 'a1_portfoy_icon'],
    a1_portfoy: ['a1_portfoy_icon_So2sGTy', 'a1_portfoy_icon'],
    aktif_portfoy: ['aktif_portfoy', 'aktif_portfoy_icon'],
    ak_portfoy: ['akportfoy_icon'],
    allbatross_portfoy: ['allbatross_portfoy_icon'],
    atlas_portfoy: ['atlas_portfoy_icon'],
    ata_portfoy: ['ata_portfoy_icon'],
    aura_portfoy: ['aura_portfoy'],
    bulls_portfoy: ['bulls_portfoy_icon'],
    inveo_portfoy: ['inveo_portfoy'],
    is_portfoy: ['is_portfoy_icon'],
    istanbul_portfoy: ['istanbul_portfoy'],
    pardus_portfoy: ['pardus_portfoy_icon'],
    perform_portfoy: ['perform_portfoy'],
    pusula_portfoy: ['pusula_portfoy_icon'],
    qinvest_portfoy: ['qinvest_portfoy'],
    qnb_finans_portfoy: ['qnb_finans_portfoy_icon'],
    qnb_portfoy: ['qnb_finans_portfoy_icon'],
    tacirler_portfoy: ['tacirler_portfoy'],
    tera_portfoy: ['tera_portfoy_icon'],
    vega_portfoy: ['vega_portfoy_icon'],
    yapi_kredi_portfoy: ['yapikredi_portfoy_icon'],
};

export function normalizeLogoSymbol(symbol: string): string {
    const normalized = String(symbol || '')
        .replace(/\.[A-Z]{1,4}$/i, '')
        .trim()
        .toUpperCase();
    const classSuffixMatch = normalized.match(/^([A-Z0-9]{2,12})[\s._-]+[A-Z]$/);
    return classSuffixMatch ? classSuffixMatch[1] : normalized;
}

function logoLookupSymbol(symbol: string): string {
    const normalized = normalizeLogoSymbol(symbol);
    return SYMBOL_LOGO_ALIAS_MAP[normalized] || normalized;
}

export function explicitLogoUrlsForSymbol(symbol: string): string[] {
    const normalized = normalizeLogoSymbol(symbol);
    return EXPLICIT_SYMBOL_LOGO_URL_MAP[normalized] || [];
}

export function fintablesLogoUrlsForSymbol(symbol: string): string[] {
    if (!FINTABLES_LOGOS_ENABLED) return [];
    const normalized = logoLookupSymbol(symbol).replace(/[^A-Z0-9]/g, '');
    if (!/^[A-Z0-9]{2,12}$/.test(normalized)) return [];
    return [
        `${FINTABLES_LOGO_BASE_URL}/${normalized.toLowerCase()}_icon.png`,
        `${FINTABLES_LOGO_BASE_URL}/${normalized}.png`,
    ];
}

function normalizeFundManagerSlugSeed(raw: string): string {
    return foldTurkishToAscii(raw)
        .replace(/\ba\s*[\.\s]*s\b\.?/g, '')
        .replace(/\bportfoy yonetimi\b/g, 'portfoy')
        .replace(/\byonetimi\b/g, '')
        .replace(/\banonim sirketi\b/g, '')
        .replace(/\bsirketi\b/g, '')
        .replace(/\bpys\b/g, 'portfoy')
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '')
        .replace(/_+/g, '_');
}

export function fintablesFundManagerLogoUrls(managerName?: string | null): string[] {
    if (!FINTABLES_LOGOS_ENABLED) return [];
    const slug = normalizeFundManagerSlugSeed(String(managerName || ''));
    if (!slug) return [];
    const mappedKeys = Object.keys(FUND_MANAGER_LOGO_SLUG_MAP).filter(
        (key) => slug === key || slug.startsWith(`${key}_`) || slug.includes(`_${key}_`),
    );
    const basenames = uniquePreserveCase([
        ...mappedKeys.flatMap((key) => FUND_MANAGER_LOGO_SLUG_MAP[key] || []),
        `${slug}_icon`,
        slug,
    ]);
    return basenames.flatMap((item) => [
        `${FINTABLES_FUND_MANAGER_LOGO_BASE_URL}/${item}.png`,
        `${FINTABLES_FUND_MANAGER_LOGO_BASE_URL}/${item}.jpeg`,
    ]);
}

function sanitizeDomain(raw: string | null | undefined): string | null {
    const value = String(raw || '').trim().toLowerCase();
    if (!value) return null;
    const withoutProtocol = value.replace(/^https?:\/\//, '');
    const withoutPath = withoutProtocol.split('/')[0].trim();
    const withoutWww = withoutPath.replace(/^www\./, '');
    if (!withoutWww || !withoutWww.includes('.')) return null;
    return withoutWww;
}

export function stockDomainsForSymbol(symbol: string): string[] {
    const normalized = normalizeLogoSymbol(symbol);
    const entry = STOCK_LOGO_DOMAIN_MAP[normalized];
    const rawItems = Array.isArray(entry) ? entry : entry ? [entry] : [];
    const sanitized = rawItems
        .map((item) => sanitizeDomain(item))
        .filter((item): item is string => Boolean(item));
    return uniqueLimited(sanitized, 12);
}

export function stockDomainForSymbol(symbol: string): string | null {
    const domains = stockDomainsForSymbol(symbol);
    return domains[0] || null;
}

function companySlugCandidates(symbol: string, companyName?: string): string[] {
    const normalizedSymbol = normalizeLogoSymbol(symbol);
    const symbolSlug = normalizedSymbol.replace(/[^A-Z0-9]/g, '').toLowerCase();

    const foldedName = foldTurkishToAscii(String(companyName || ''));
    const rawTokens = foldedName
        .replace(/&/g, ' and ')
        .replace(/[^a-z0-9]+/g, ' ')
        .split(' ')
        .filter(Boolean);

    const filteredTokens = rawTokens.filter((token) => token.length > 1 && !COMPANY_TOKEN_STOPWORDS.has(token));
    const first = filteredTokens[0] || '';
    const second = filteredTokens[1] || '';
    const joinedTwo = first && second ? `${first}${second}` : '';
    const joinedThree = filteredTokens.slice(0, 3).join('');
    const hasGrupLikeWord = rawTokens.includes('grup') || rawTokens.includes('grubu') || rawTokens.includes('group');
    const hasHoldingWord = rawTokens.includes('holding') || rawTokens.includes('holdings');

    const dynamicSeeds = uniqueLimited([
        symbolSlug,
        first,
        joinedTwo,
        joinedThree,
        hasGrupLikeWord && first ? `${first}group` : '',
        hasHoldingWord && first ? `${first}holding` : '',
    ]);

    return dynamicSeeds;
}

export function stockLogoDevDomains(symbol: string, companyName?: string): string[] {
    const normalized = normalizeLogoSymbol(symbol);
    const explicitDomains = stockDomainsForSymbol(normalized);
    const seeds = companySlugCandidates(normalized, companyName);
    const inferredDomains = seeds.flatMap((seed) => [`${seed}.com.tr`, `${seed}.com`]);
    return uniqueLimited([...explicitDomains, ...inferredDomains], 12);
}

export function localIconForSymbol(symbol: string): string | null {
    const normalized = normalizeLogoSymbol(symbol);
    return LOCAL_SYMBOL_ICON_MAP[normalized] || null;
}

export function tradingViewCountryFlagUrl(code: string): string {
    return `https://s3-symbol-logo.tradingview.com/country/${String(code || '').trim().toUpperCase()}--big.svg`;
}

export function tradingViewLogoUrl(
    symbol: string,
    options?: { force?: boolean; requireMappedSlug?: boolean },
): string | null {
    if (!options?.force && !TRADINGVIEW_FALLBACK_ENABLED) return null;
    const normalized = normalizeLogoSymbol(symbol);
    const mappedSlug = TRADINGVIEW_SYMBOL_SLUG_MAP[normalized] || '';
    const slug =
        mappedSlug ||
        (options?.requireMappedSlug ? '' : normalized.toLowerCase().replace(/\//g, '-'));
    if (!slug) return null;
    const hasKnownExtension = /\.(svg|png|jpe?g|webp)$/i.test(slug);
    return `https://s3-symbol-logo.tradingview.com/${slug}${hasKnownExtension ? '' : '.svg'}`;
}

export function logoDevDomainUrl(domain: string | null): string | null {
    if (!LOGO_DEV_ENABLED || !domain) return null;
    const cleanDomain = String(domain).trim().toLowerCase();
    if (!cleanDomain) return null;
    return `https://img.logo.dev/${cleanDomain}?token=${encodeURIComponent(LOGO_DEV_TOKEN)}`;
}
