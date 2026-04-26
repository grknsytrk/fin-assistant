export type SymbolLogoKind = 'stock' | 'index' | 'commodity' | 'fx';

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
    TAVHL: 'tavhavalimanlari.com.tr',
    THYAO: 'turkishairlines.com',
    TKFEN: 'tekfen.com.tr',
    TOASO: 'tofas.com.tr',
    TSPOR: 'trabzonspor.org.tr',
    TTKOM: 'turktelekom.com.tr',
    TTRAK: 'turktraktor.com.tr',
    TRALT: 'turkaltinisletmeleri.com',
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

export const LOCAL_SYMBOL_ICON_MAP: Record<string, string> = {
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
    XU100: 'country/TR--big',
    XU030: 'country/TR--big',
};

export const FX_COUNTRY_FLAGS: Record<string, { base: string; quote?: string }> = {
    'USD/TRY': { base: 'US', quote: 'TR' },
    'EUR/TRY': { base: 'EU', quote: 'TR' },
    'GBP/TRY': { base: 'GB', quote: 'TR' },
    'EUR/USD': { base: 'EU', quote: 'US' },
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
    KCHOL: 'koc',
    PGSUS: 'pegasus',
    TAVHL: 'tav-havalimanlari',
    THYAO: 'turkish-airlines',
    TUPRS: 'tupras',
};

const rawLogoDevToken = (import.meta.env.VITE_LOGO_DEV_TOKEN as string | undefined)?.trim();
const rawTradingViewFlag = (import.meta.env.VITE_ENABLE_TRADINGVIEW_LOGO_FALLBACK as string | undefined)?.trim().toLowerCase();

export const LOGO_DEV_TOKEN = rawLogoDevToken || '';
export const LOGO_DEV_ENABLED = Boolean(LOGO_DEV_TOKEN);
export const TRADINGVIEW_FALLBACK_ENABLED = rawTradingViewFlag === '1' || rawTradingViewFlag === 'true' || rawTradingViewFlag === 'yes';

export function normalizeLogoSymbol(symbol: string): string {
    const normalized = String(symbol || '')
        .replace(/\.[A-Z]{1,4}$/i, '')
        .trim()
        .toUpperCase();
    const classSuffixMatch = normalized.match(/^([A-Z0-9]{2,12})[\s._-]+[A-Z]$/);
    return classSuffixMatch ? classSuffixMatch[1] : normalized;
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
    return `https://s3-symbol-logo.tradingview.com/${slug}.svg`;
}

export function logoDevDomainUrl(domain: string | null): string | null {
    if (!LOGO_DEV_ENABLED || !domain) return null;
    const cleanDomain = String(domain).trim().toLowerCase();
    if (!cleanDomain) return null;
    return `https://img.logo.dev/${cleanDomain}?token=${encodeURIComponent(LOGO_DEV_TOKEN)}`;
}
