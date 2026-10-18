import { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { Search, X } from 'lucide-react';
import { apiClient } from '../api/client';
import type { FundSummary, MarketUniverseRow } from '../api/types';
import { DEFAULT_FUND_TAB, DEFAULT_STOCK_TAB, toFundDetail, toStockDetail } from '../routing/routes';
import AssetSearchResult from './AssetSearchResult';
import './GlobalTickerSearch.css';

type SearchAssetType = 'stock' | 'fund';

type SearchResult = {
  id: string;
  type: SearchAssetType;
  symbol: string;
  title: string;
  subtitle: string;
  aliases?: string[];
  price: number | null;
  changePct: number | null;
  logoUrl?: string | null;
  logoName?: string | null;
  searchFields?: SearchField[];
};

type SearchField = {
  text: string;
  compact: string;
  noVowels: string;
};

type SearchTerms = SearchField;
type SearchCatalogStatus = 'loading' | 'ready' | 'error';

interface GlobalTickerSearchProps {
  currentTicker?: string | null;
  onSelectTicker: (ticker: string) => void;
  onSelectFund?: (fundCode: string) => void;
}

const RECENT_SEARCHES_KEY = 'ragfin.globalSearch.recent.v1';
const MAX_RECENT_SEARCHES = 8;
const MIN_FUND_SEARCH_QUERY_LENGTH = 1;
const SEARCH_CATALOG_CACHE_TTL_MS = 5 * 60 * 1000;
const STOCK_SEARCH_CATALOG_CACHE_TTL_MS = 60 * 1000;

type SearchCatalogCache<T> = { rows: T[]; fetchedAt: number };
let stockSearchCatalogCache: SearchCatalogCache<MarketUniverseRow> | null = null;
let fundSearchCatalogCache: SearchCatalogCache<FundSummary> | null = null;
let stockSearchCatalogRequest: Promise<MarketUniverseRow[]> | null = null;
let fundSearchCatalogRequest: Promise<FundSummary[]> | null = null;

function cachedCatalogRows<T>(
  cache: SearchCatalogCache<T> | null,
  ttlMs = SEARCH_CATALOG_CACHE_TTL_MS,
): T[] | null {
  if (!cache || Date.now() - cache.fetchedAt >= ttlMs) return null;
  return cache.rows;
}

function availableStockSearchRows(): MarketUniverseRow[] {
  return stockSearchCatalogCache?.rows
    || apiClient.getCachedMarketUniverse?.()?.rows
    || [];
}

function hasStockSearchRows(): boolean {
  return Boolean(stockSearchCatalogCache || apiClient.getCachedMarketUniverse?.());
}

function availableFundSearchRows(): FundSummary[] {
  const shared = apiClient.getCachedFundsCatalog?.();
  if (shared && (!fundSearchCatalogCache || shared.fetchedAt >= fundSearchCatalogCache.fetchedAt)) {
    return shared.payload.rows || [];
  }
  return fundSearchCatalogCache?.rows || [];
}

function hasFundSearchRows(): boolean {
  return Boolean(fundSearchCatalogCache || apiClient.getCachedFundsCatalog?.());
}

function cachedMarketStockRows(): MarketUniverseRow[] {
  const cached = apiClient.getCachedMarketStocksEntry?.('XUTUM');
  if (!cached || Date.now() - cached.fetchedAt >= 15_000) return [];
  return cached.payload.rows || [];
}

function loadStockSearchCatalog(): Promise<MarketUniverseRow[]> {
  const cached = cachedCatalogRows(stockSearchCatalogCache, STOCK_SEARCH_CATALOG_CACHE_TTL_MS);
  if (cached) return Promise.resolve(cached);
  if (stockSearchCatalogRequest) return stockSearchCatalogRequest;

  const request = Promise.resolve()
    .then(() => apiClient.marketUniverse())
    .then((payload) => {
      const rows = payload.rows || [];
      stockSearchCatalogCache = { rows, fetchedAt: Date.now() };
      return rows;
    })
    .finally(() => {
      if (stockSearchCatalogRequest === request) stockSearchCatalogRequest = null;
    });
  stockSearchCatalogRequest = request;
  return request;
}

function loadFundSearchCatalog(): Promise<FundSummary[]> {
  const shared = apiClient.getCachedFundsCatalog?.();
  const sharedIsFresh = shared && Date.now() - shared.fetchedAt < SEARCH_CATALOG_CACHE_TTL_MS;
  if (
    sharedIsFresh
    && (!fundSearchCatalogCache || shared.fetchedAt >= fundSearchCatalogCache.fetchedAt)
  ) {
    const rows = shared.payload.rows || [];
    fundSearchCatalogCache = { rows, fetchedAt: shared.fetchedAt };
    return Promise.resolve(rows);
  }

  const cached = cachedCatalogRows(fundSearchCatalogCache);
  if (cached) return Promise.resolve(cached);
  if (fundSearchCatalogRequest) return fundSearchCatalogRequest;

  const request = Promise.resolve()
    .then(() => apiClient.funds())
    .then((payload) => {
      const rows = payload.rows || [];
      fundSearchCatalogCache = { rows, fetchedAt: Date.now() };
      return rows;
    })
    .finally(() => {
      if (fundSearchCatalogRequest === request) fundSearchCatalogRequest = null;
    });
  fundSearchCatalogRequest = request;
  return request;
}

const STOCK_SYMBOL_ALIASES: Record<string, string> = {
  BIM: 'BIMAS',
  BIMAS: 'BIMAS',
  MIGROS: 'MGROS',
  MGROS: 'MGROS',
  SOK: 'SOKM',
  SOKM: 'SOKM',
  TAV: 'TAVHL',
  TAVHL: 'TAVHL',
};

const STOCK_SEARCH_LABELS: Record<string, string[]> = {
  ASTOR: ['Astor Enerji A.Ş.'],
  BIMAS: ['BİM Birleşik Mağazalar A.Ş.', 'BIM'],
  MGROS: ['Migros Ticaret A.Ş.', 'Migros', 'MIGROS'],
  ORGE: ['Orge Enerji Elektrik Taahhüt A.Ş.'],
  SOKM: ['Şok Marketler Ticaret A.Ş.', 'SOK'],
  TAVHL: ['TAV Havalimanları Holding A.Ş.', 'TAV'],
  YEOTK: ['YEO Teknoloji Enerji ve Endüstri A.Ş.'],
};

function normalizeSymbol(value: string | null | undefined): string {
  return String(value || '').trim().toUpperCase().replace(/\s+/g, '');
}

function normalizeStockSymbol(value: string | null | undefined): string {
  const symbol = normalizeSymbol(value);
  return STOCK_SYMBOL_ALIASES[symbol] || symbol;
}

function normalizeSearchText(value: string | null | undefined): string {
  return String(value || '')
    .toLocaleUpperCase('tr-TR')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^A-Z0-9]+/g, ' ')
    .trim();
}

function getStockAliases(symbol: string, sourceSymbol?: string | null): string[] {
  const aliases = new Set<string>([symbol]);
  const rawSymbol = normalizeSymbol(sourceSymbol);
  if (rawSymbol && rawSymbol !== symbol) aliases.add(rawSymbol);
  for (const label of STOCK_SEARCH_LABELS[symbol] || []) {
    aliases.add(label);
  }
  return [...aliases];
}

function resultId(type: SearchAssetType, symbol: string): string {
  return `${type}:${normalizeSymbol(symbol)}`;
}

function readRecentSearches(): SearchResult[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(RECENT_SEARCHES_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item): item is SearchResult => {
        return (
          item
          && (item.type === 'stock' || item.type === 'fund')
          && typeof item.symbol === 'string'
          && typeof item.title === 'string'
        );
      })
      .slice(0, MAX_RECENT_SEARCHES)
      .map((item) => ({ ...item, price: null, changePct: null, searchFields: undefined }));
  } catch {
    return [];
  }
}

function writeRecentSearches(items: SearchResult[]): void {
  if (typeof window === 'undefined') return;
  try {
    const persisted = items.slice(0, MAX_RECENT_SEARCHES).map((item) => {
      const copy = { ...item, price: null, changePct: null };
      delete copy.searchFields;
      return copy;
    });
    window.localStorage.setItem(RECENT_SEARCHES_KEY, JSON.stringify(persisted));
  } catch {
    // Ignore storage failures; the modal still works without persisted recents.
  }
}

function prepareSearchResult(result: SearchResult): SearchResult {
  const values = [result.symbol, result.title, result.subtitle, ...(result.aliases || [])]
    .filter(Boolean);
  return {
    ...result,
    searchFields: values.map((value) => {
      const text = normalizeSearchText(value);
      const compact = text.replace(/\s+/g, '');
      return { text, compact, noVowels: compact.replace(/[AEIOU]/g, '') };
    }),
  };
}

function stockRowToResult(row: MarketUniverseRow): SearchResult {
  const sourceSymbol = row.symbol || row.company;
  const symbol = normalizeStockSymbol(sourceSymbol);
  const title = String(row.name || '').trim();
  const displayName = title || STOCK_SEARCH_LABELS[symbol]?.[0] || 'Hisse';
  const aliases = getStockAliases(symbol, sourceSymbol);
  return {
    id: resultId('stock', symbol),
    type: 'stock',
    symbol,
    title: displayName,
    subtitle: displayName,
    aliases,
    price: row.price ?? null,
    changePct: row.change_pct ?? null,
    logoUrl: row.logo_url,
    logoName: displayName,
  };
}

function mergeStockQuoteRows(
  catalogRows: MarketUniverseRow[],
  quoteRows: MarketUniverseRow[],
): MarketUniverseRow[] {
  const quotesBySymbol = new Map(
    quoteRows.map((row) => [normalizeStockSymbol(row.symbol || row.company), row]),
  );
  return catalogRows.map((row) => {
    const quote = quotesBySymbol.get(normalizeStockSymbol(row.symbol || row.company));
    if (!quote) return row;
    return {
      ...row,
      price: quote.price,
      price_currency: quote.price_currency,
      change: quote.change,
      change_pct: quote.change_pct,
      price_as_of: quote.price_as_of,
      market_cap: quote.market_cap ?? row.market_cap,
    };
  });
}

function fundRowToResult(row: FundSummary): SearchResult {
  const symbol = normalizeSymbol(row.fund_code);
  return {
    id: resultId('fund', symbol),
    type: 'fund',
    symbol,
    title: symbol,
    subtitle: row.name,
    aliases: [row.fund_code, row.name, row.founder_company || '', row.manager_company || ''].filter(Boolean),
    price: row.price,
    changePct: row.daily_return,
    logoName: row.founder_company || row.manager_company || row.name,
  };
}

function searchTerms(value: string): SearchTerms {
  const text = normalizeSearchText(value);
  const compact = text.replace(/\s+/g, '');
  return { text, compact, noVowels: compact.replace(/[AEIOU]/g, '') };
}

const NO_SEARCH_MATCH = Number.POSITIVE_INFINITY;

function resultMatchRank(result: SearchResult, terms: SearchTerms, normalizedSymbolQuery = ''): number {
  if (!terms.text) return 0;

  if (normalizedSymbolQuery && normalizeSymbol(result.symbol) === normalizedSymbolQuery) {
    // Fund codes are the authoritative interpretation for a three-letter
    // query, while still keeping exact fund codes ahead of stock codes for
    // any future symbol collision.
    return result.type === 'fund' ? 0 : 1;
  }

  const fields = result.searchFields || prepareSearchResult(result).searchFields || [];
  let bestRank = NO_SEARCH_MATCH;
  for (const field of fields) {
    if (field.text === terms.text || field.compact === terms.compact) {
      bestRank = Math.min(bestRank, 2);
      continue;
    }
    if (field.text.startsWith(terms.text) || field.compact.startsWith(terms.compact)) {
      bestRank = Math.min(bestRank, 3);
      continue;
    }
    if (field.text.includes(terms.text) || field.compact.includes(terms.compact)) {
      bestRank = Math.min(bestRank, 4);
      continue;
    }
    if (terms.noVowels.length >= 3 && field.noVowels.includes(terms.noVowels)) {
      bestRank = Math.min(bestRank, 5);
    }
  }
  return bestRank;
}

function resultMatches(result: SearchResult, terms: SearchTerms): boolean {
  return resultMatchRank(result, terms) < NO_SEARCH_MATCH;
}

function compareSearchResults(a: SearchResult, b: SearchResult, terms: SearchTerms, normalizedSymbolQuery: string): number {
  const rankDifference = resultMatchRank(a, terms, normalizedSymbolQuery)
    - resultMatchRank(b, terms, normalizedSymbolQuery);
  if (rankDifference !== 0) return rankDifference;
  if (a.type !== b.type) return a.type === 'fund' ? -1 : 1;
  return a.symbol.localeCompare(b.symbol, 'tr');
}

function isThreeLetterFundQuery(value: string): boolean {
  return /^[A-Z]{3}$/.test(normalizeSymbol(value));
}

function isDirectFundCodeQuery(value: string): boolean {
  const normalized = normalizeSymbol(value);
  return isThreeLetterFundQuery(normalized) && !STOCK_SYMBOL_ALIASES[normalized];
}

function rankSearchResults(results: SearchResult[], query: string, currentTicker?: string | null): SearchResult[] {
  const normalizedQuery = normalizeSearchText(query);
  const terms = searchTerms(normalizedQuery);
  const normalizedSymbolQuery = normalizeSymbol(query);
  const currentStock = normalizeSymbol(currentTicker);
  const seen = new Set<string>();

  return results
    .filter((result) => result.type !== 'stock' || result.symbol !== currentStock)
    .filter((result) => resultMatchRank(result, terms, normalizedSymbolQuery) < NO_SEARCH_MATCH)
    .filter((result) => {
      if (seen.has(result.id)) return false;
      seen.add(result.id);
      return true;
    })
    .sort((a, b) => compareSearchResults(a, b, terms, normalizedSymbolQuery));
}

export default function GlobalTickerSearch({
  currentTicker = null,
  onSelectTicker,
  onSelectFund,
}: GlobalTickerSearchProps) {
  const [stockCatalogRows, setStockCatalogRows] = useState<MarketUniverseRow[]>(availableStockSearchRows);
  const [stockCatalogStatus, setStockCatalogStatus] = useState<SearchCatalogStatus>(() => (
    hasStockSearchRows() ? 'ready' : 'loading'
  ));
  const [stockQuoteRows, setStockQuoteRows] = useState<MarketUniverseRow[]>(() => (
    cachedMarketStockRows()
  ));
  const [fundCatalogRows, setFundCatalogRows] = useState<FundSummary[]>(availableFundSearchRows);
  const [fundCatalogStatus, setFundCatalogStatus] = useState<SearchCatalogStatus>(() => (
    hasFundSearchRows() ? 'ready' : 'loading'
  ));
  const [stockRows, setStockRows] = useState<MarketUniverseRow[]>([]);
  const [stocksLoading, setStocksLoading] = useState(false);
  const [fundRows, setFundRows] = useState<FundSummary[]>([]);
  const [fundsLoading, setFundsLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const [recentSearches, setRecentSearches] = useState<SearchResult[]>(() => readRecentSearches());
  const [isResolving, setIsResolving] = useState(false);
  const [searchError, setSearchError] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const stockSearchQueryRef = useRef('');
  const fundSearchQueryRef = useRef('');
  const submitGenerationRef = useRef(0);
  const submitControllerRef = useRef<AbortController | null>(null);

  const deferredQuery = useDeferredValue(query);
  const trimmedQuery = deferredQuery.trim();
  const normalizedQuery = normalizeSearchText(trimmedQuery);
  const queryTerms = useMemo(() => searchTerms(normalizedQuery), [normalizedQuery]);
  const normalizedSymbolQuery = normalizeSymbol(query.trim());
  const immediateQueryTerms = useMemo(() => searchTerms(query.trim()), [query]);
  const useStockSearchFallback = stockCatalogStatus !== 'ready';
  const useFundSearchFallback = fundCatalogStatus !== 'ready';

  useEffect(() => {
    let active = true;

    setStockQuoteRows(cachedMarketStockRows());
    const unsubscribeStocks = apiClient.subscribeMarketStocks?.((index, payload) => {
      if (active && index === 'XUTUM') {
        setStockQuoteRows(payload.rows || []);
      }
    });
    const unsubscribeFunds = apiClient.subscribeFundsCatalog?.((payload) => {
      if (!active || payload.status === 'pending') return;
      const rows = payload.rows || [];
      fundSearchCatalogCache = { rows, fetchedAt: Date.now() };
      setFundCatalogRows(rows);
      setFundCatalogStatus('ready');
    });

    // Warm both searchable catalogues when the navigation mounts. Once they
    // are ready, keystrokes only filter these in-memory rows, just like the
    // fund list search. The market and fund pages share the API client's
    // existing request/cache paths, so an already-running catalogue load is
    // reused where possible.
    loadStockSearchCatalog()
      .then((rows) => {
        if (!active) return;
        setStockCatalogRows(rows);
        setStockCatalogStatus('ready');
      })
      .catch(() => {
        if (active && !hasStockSearchRows()) setStockCatalogStatus('error');
      });

    loadFundSearchCatalog()
      .then((rows) => {
        if (!active) return;
        setFundCatalogRows(rows);
        setFundCatalogStatus('ready');
      })
      .catch(() => {
        if (active && !hasFundSearchRows()) setFundCatalogStatus('error');
      });

    return () => {
      active = false;
      unsubscribeStocks?.();
      unsubscribeFunds?.();
    };
  }, []);

  const cancelSearch = useCallback(() => {
    submitGenerationRef.current += 1;
    submitControllerRef.current?.abort();
    submitControllerRef.current = null;
    setIsResolving(false);
    setSearchError(false);
    setIsOpen(false);
    setQuery('');
  }, []);

  useEffect(() => {
    submitGenerationRef.current += 1;
    submitControllerRef.current?.abort();
    submitControllerRef.current = null;
    setIsResolving(false);
    setSearchError(false);
  }, [query]);

  useEffect(() => () => {
    submitGenerationRef.current += 1;
    submitControllerRef.current?.abort();
  }, []);

  useEffect(() => {
    if (!useStockSearchFallback) {
      stockSearchQueryRef.current = trimmedQuery;
      setStockRows([]);
      setStocksLoading(false);
      return;
    }
    if (!isOpen || trimmedQuery.length < 1) {
      stockSearchQueryRef.current = trimmedQuery;
      setStockRows([]);
      setStocksLoading(false);
      return;
    }
    let active = true;
    stockSearchQueryRef.current = '';
    setStockRows([]);
    setStocksLoading(true);
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      apiClient.marketStockSearch({
        q: trimmedQuery,
        index: 'XUTUM',
        limit: 12,
        signal: controller.signal,
      })
        .then((payload) => {
          if (active) {
            stockSearchQueryRef.current = trimmedQuery;
            setStockRows(payload.rows || []);
          }
        })
        .catch((error) => {
          if ((error as Error)?.name === 'AbortError') return;
          if (active) {
            stockSearchQueryRef.current = trimmedQuery;
            setStockRows([]);
          }
        })
        .finally(() => {
          if (active) setStocksLoading(false);
        });
    }, 180);
    return () => {
      active = false;
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [isOpen, trimmedQuery, useStockSearchFallback]);

  useEffect(() => {
    if (!useFundSearchFallback) {
      fundSearchQueryRef.current = trimmedQuery;
      setFundRows([]);
      setFundsLoading(false);
      return;
    }
    if (!isOpen || trimmedQuery.length < MIN_FUND_SEARCH_QUERY_LENGTH) {
      fundSearchQueryRef.current = trimmedQuery;
      setFundRows([]);
      setFundsLoading(false);
      return;
    }
    let active = true;
    fundSearchQueryRef.current = '';
    setFundRows([]);
    setFundsLoading(true);
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      if (!active) return;
      apiClient
        .fundSearch(trimmedQuery, 12, { signal: controller.signal })
        .then((payload) => {
          if (active) {
            fundSearchQueryRef.current = trimmedQuery;
            setFundRows(payload.rows || []);
          }
        })
        .catch((error) => {
          if ((error as Error)?.name === 'AbortError') return;
          if (active) {
            fundSearchQueryRef.current = trimmedQuery;
            setFundRows([]);
          }
        })
        .finally(() => {
          if (active) setFundsLoading(false);
        });
    }, 180);
    return () => {
      active = false;
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [isOpen, trimmedQuery, useFundSearchFallback]);

  useEffect(() => {
    if (!isOpen) return;
    const timer = window.setTimeout(() => inputRef.current?.focus(), 0);
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') {
        cancelSearch();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [cancelSearch, isOpen]);

  const stockBaseResults = useMemo(() => {
    const rows = stockCatalogStatus === 'ready' ? stockCatalogRows : stockRows;
    return mergeStockQuoteRows(rows, stockQuoteRows).map(stockRowToResult).map(prepareSearchResult);
  }, [stockCatalogRows, stockCatalogStatus, stockQuoteRows, stockRows]);

  const fundBaseResults = useMemo(() => {
    const rows = fundCatalogStatus === 'ready' ? fundCatalogRows : fundRows;
    return rows
      .map(fundRowToResult)
      .map(prepareSearchResult);
  }, [fundCatalogRows, fundCatalogStatus, fundRows]);

  const fundResults = useMemo(() => {
    return fundBaseResults
      .filter((item) => resultMatches(item, queryTerms))
      .sort((a, b) => compareSearchResults(a, b, queryTerms, normalizeSymbol(trimmedQuery)))
      .slice(0, 10);
  }, [fundBaseResults, queryTerms, trimmedQuery]);

  const allSearchCandidates = useMemo(
    () => [...stockBaseResults, ...fundBaseResults],
    [fundBaseResults, stockBaseResults],
  );

  useEffect(() => {
    if (recentSearches.length === 0) return;
    const currentResults = new Map(allSearchCandidates.map((result) => [result.id, result]));
    setRecentSearches((current) => {
      let changed = false;
      const next = current.map((recent) => {
        const refreshed = currentResults.get(recent.id) || { ...recent, price: null, changePct: null };
        if (refreshed.price !== recent.price || refreshed.changePct !== recent.changePct) changed = true;
        return refreshed;
      });
      if (changed) writeRecentSearches(next);
      return changed ? next : current;
    });
  }, [allSearchCandidates, recentSearches.length]);

  const visibleResults = useMemo(() => {
    if (!normalizedQuery) return recentSearches;
    return rankSearchResults(allSearchCandidates, trimmedQuery, currentTicker).slice(0, 14);
  }, [allSearchCandidates, currentTicker, normalizedQuery, recentSearches, trimmedQuery]);

  useEffect(() => {
    setHighlightedIndex(0);
  }, [normalizedQuery, isOpen]);

  const rememberResult = (result: SearchResult) => {
    setRecentSearches((current) => {
      const next = [result, ...current.filter((item) => item.id !== result.id)].slice(0, MAX_RECENT_SEARCHES);
      writeRecentSearches(next);
      return next;
    });
  };

  const commitKnownResult = (selected: SearchResult) => {
    rememberResult(selected);
    setIsOpen(false);
    setQuery('');
    if (selected.type === 'fund') {
      if (onSelectFund) {
        onSelectFund(selected.symbol);
      } else {
        window.location.href = `/funds/${encodeURIComponent(selected.symbol)}/overview`;
      }
      return;
    }
    onSelectTicker(selected.symbol);
  };

  const commitFundCode = (fundCode: string) => {
    const normalizedFundCode = normalizeSymbol(fundCode);
    if (!normalizedFundCode) return;
    cancelSearch();
    if (onSelectFund) {
      onSelectFund(normalizedFundCode);
    } else {
      window.location.href = `/funds/${encodeURIComponent(normalizedFundCode)}/overview`;
    }
  };

  const openSearch = () => {
    setIsOpen(true);

    setStockQuoteRows(cachedMarketStockRows());

    if (!cachedCatalogRows(stockSearchCatalogCache, STOCK_SEARCH_CATALOG_CACHE_TTL_MS)) {
      if (!hasStockSearchRows()) setStockCatalogStatus('loading');
      loadStockSearchCatalog()
        .then((rows) => {
          setStockCatalogRows(rows);
          setStockCatalogStatus('ready');
        })
        .catch(() => {
          if (!hasStockSearchRows()) setStockCatalogStatus('error');
        });
    }

    if (!cachedCatalogRows(fundSearchCatalogCache)) {
      if (!hasFundSearchRows()) setFundCatalogStatus('loading');
      loadFundSearchCatalog()
        .then((rows) => {
          setFundCatalogRows(rows);
          setFundCatalogStatus('ready');
        })
        .catch(() => {
          if (!hasFundSearchRows()) setFundCatalogStatus('error');
        });
    }
  };

  const resolveQueryOnEnter = async () => {
    const searchQuery = query.trim();
    const normalizedSubmitQuery = normalizeSymbol(searchQuery);
    if (!normalizedSubmitQuery || isResolving) return;

    const generation = submitGenerationRef.current;
    const controller = new AbortController();
    submitControllerRef.current?.abort();
    submitControllerRef.current = controller;
    setIsResolving(true);
    setSearchError(false);

    try {
      const candidates: SearchResult[] = [];
      let hadError = false;

      if (isThreeLetterFundQuery(searchQuery)) {
        try {
          const payload = await apiClient.fundSearch(searchQuery, 12, { signal: controller.signal });
          const exactFundRows = (payload.rows || []).filter((row) => (
            normalizeSymbol(row.fund_code) === normalizedSubmitQuery
          ));
          candidates.push(...exactFundRows.map(fundRowToResult).map(prepareSearchResult));
        } catch {
          if (controller.signal.aborted) return;
          hadError = true;
        }
      } else {
        const [stockResult, fundResult] = await Promise.allSettled([
          apiClient.marketStockSearch({
            q: searchQuery,
            index: 'XUTUM',
            limit: 12,
            signal: controller.signal,
          }),
          apiClient.fundSearch(searchQuery, 12, { signal: controller.signal }),
        ]);

        if (stockResult.status === 'fulfilled') {
          candidates.push(...(stockResult.value.rows || []).map(stockRowToResult).map(prepareSearchResult));
        } else if (!controller.signal.aborted) {
          hadError = true;
        }
        if (fundResult.status === 'fulfilled') {
          candidates.push(...(fundResult.value.rows || []).map(fundRowToResult).map(prepareSearchResult));
        } else if (!controller.signal.aborted) {
          hadError = true;
        }
      }

      if (controller.signal.aborted || generation !== submitGenerationRef.current) return;
      if (hadError) {
        setSearchError(true);
        return;
      }

      const rankedResults = rankSearchResults(candidates, searchQuery, currentTicker);
      const selected = rankedResults[0];
      if (!selected) {
        setSearchError(hadError);
        return;
      }
      commitKnownResult(selected);
    } finally {
      if (generation === submitGenerationRef.current) {
        setIsResolving(false);
        if (submitControllerRef.current === controller) submitControllerRef.current = null;
      }
    }
  };

  const commitResult = (result?: SearchResult) => {
    if (isResolving) return;

    if (result) {
      const matchesCurrentQuery = !immediateQueryTerms.text
        || resultMatchRank(result, immediateQueryTerms, normalizedSymbolQuery) < NO_SEARCH_MATCH;
      const requiresFundResolution = isThreeLetterFundQuery(query) && result.type === 'stock';
      if (matchesCurrentQuery && !requiresFundResolution) {
        commitKnownResult(result);
        return;
      }
      if (stockCatalogStatus === 'ready' && fundCatalogStatus === 'ready') {
        const matchingResult = rankSearchResults(allSearchCandidates, query, currentTicker)
          .find((candidate) => candidate.id === result.id);
        if (matchingResult) commitKnownResult(matchingResult);
        return;
      }
    } else if (!immediateQueryTerms.text) {
      const highlighted = visibleResults[highlightedIndex];
      if (highlighted) {
        commitKnownResult(highlighted);
      }
      return;
    } else if (stockCatalogStatus === 'ready' && fundCatalogStatus === 'ready') {
      const selected = rankSearchResults(allSearchCandidates, query, currentTicker)[highlightedIndex];
      if (selected) commitKnownResult(selected);
      return;
    } else if (
      trimmedQuery === query.trim()
      && stockSearchQueryRef.current === trimmedQuery
      && fundSearchQueryRef.current === trimmedQuery
    ) {
      const highlighted = visibleResults[highlightedIndex];
      const isAllowedFundCodeResult = !isThreeLetterFundQuery(query)
        || (
          highlighted?.type === 'fund'
          && normalizeSymbol(highlighted.symbol) === normalizedSymbolQuery
        );
      if (highlighted && isAllowedFundCodeResult) {
        commitKnownResult(highlighted);
        return;
      }
    }

    void resolveQueryOnEnter();
  };

  const handleInputKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      if (visibleResults.length > 0) {
        setHighlightedIndex((prev) => (prev + 1) % visibleResults.length);
      }
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      if (visibleResults.length > 0) {
        setHighlightedIndex((prev) => (prev === 0 ? visibleResults.length - 1 : prev - 1));
      }
      return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      if (isDirectFundCodeQuery(event.currentTarget.value)) {
        commitFundCode(event.currentTarget.value);
        return;
      }
      commitResult();
    }
  };

  const isSearching = Boolean(normalizedQuery) && (
    (stocksLoading && stockBaseResults.length === 0) ||
    (fundsLoading && fundResults.length === 0)
  );

  return (
    <div className="global-search">
      <button
        type="button"
        className="global-search-trigger"
        onClick={openSearch}
        aria-haspopup="dialog"
        aria-expanded={isOpen}
      >
        <Search size={17} aria-hidden="true" />
        <span>Ara</span>
      </button>

      {isOpen && (
        <div className="global-search-modal-backdrop" role="presentation" onMouseDown={cancelSearch}>
          <section
            className="global-search-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Hisse ve fon ara"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="global-search-modal-head">
              <Search size={24} aria-hidden="true" />
              <input
                ref={inputRef}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={handleInputKeyDown}
                placeholder="Ara..."
                aria-label="Hisse veya fon ara"
              />
              {query ? (
                <button type="button" className="global-search-clear" onClick={() => setQuery('')} aria-label="Aramayı temizle">
                  <X size={18} aria-hidden="true" />
                </button>
              ) : null}
            </div>
            <div className="global-search-modal-body">
              <div className="global-search-section-title">
                {normalizedQuery ? 'Sonuçlar' : 'En Son Arananlar'}
              </div>
              {visibleResults.length > 0 ? (
                <div className="global-search-results" role="listbox">
                  {visibleResults.map((result, index) => (
                    <AssetSearchResult
                      key={result.id}
                      href={result.type === 'fund'
                        ? toFundDetail(result.symbol, DEFAULT_FUND_TAB)
                        : toStockDetail(result.symbol, DEFAULT_STOCK_TAB)}
                      kind={result.type}
                      symbol={result.symbol}
                      name={result.subtitle}
                      price={result.price}
                      changePct={result.changePct}
                      logoUrl={result.logoUrl}
                      logoName={result.logoName || result.subtitle || result.title}
                      active={index === highlightedIndex}
                      onMouseEnter={() => setHighlightedIndex(index)}
                      onClick={(event) => {
                        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                        event.preventDefault();
                        commitResult(result);
                      }}
                    />
                  ))}
                </div>
              ) : (
                <div className="global-search-empty">
                  {isResolving || isSearching
                    ? 'Aranıyor...'
                    : searchError
                      ? 'Arama yapılamadı. Lütfen tekrar deneyin.'
                    : normalizedQuery
                      ? 'Sonuç bulunamadı.'
                      : 'Arama yaptıkça fonlar ve hisseler burada görünecek.'}
                </div>
              )}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
