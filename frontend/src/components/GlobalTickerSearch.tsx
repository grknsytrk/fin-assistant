import { useDeferredValue, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { Search, X } from 'lucide-react';
import { apiClient } from '../api/client';
import type { FundSummary, MarketUniverseRow } from '../api/types';
import SymbolLogo from './SymbolLogo';
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

interface GlobalTickerSearchProps {
  currentTicker?: string | null;
  onSelectTicker: (ticker: string) => void;
  onSelectFund?: (fundCode: string) => void;
}

const RECENT_SEARCHES_KEY = 'ragfin.globalSearch.recent.v1';
const MAX_RECENT_SEARCHES = 8;
const MIN_FUND_SEARCH_QUERY_LENGTH = 2;

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
      .slice(0, MAX_RECENT_SEARCHES);
  } catch {
    return [];
  }
}

function writeRecentSearches(items: SearchResult[]): void {
  if (typeof window === 'undefined') return;
  try {
    const persisted = items.slice(0, MAX_RECENT_SEARCHES).map((item) => {
      const copy = { ...item };
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

function formatSearchPrice(value: number | null | undefined, type: SearchAssetType): string {
  if (value == null || !Number.isFinite(value)) return '-';
  return value.toLocaleString('tr-TR', {
    minimumFractionDigits: type === 'fund' ? 4 : 2,
    maximumFractionDigits: type === 'fund' ? 6 : 2,
  });
}

function formatSearchPct(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '-';
  const sign = value > 0 ? '+' : '';
  return `% ${sign}${value.toLocaleString('tr-TR', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function changeClass(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return 'flat';
  if (value > 0) return 'up';
  if (value < 0) return 'down';
  return 'flat';
}

function searchTerms(value: string): SearchTerms {
  const text = normalizeSearchText(value);
  const compact = text.replace(/\s+/g, '');
  return { text, compact, noVowels: compact.replace(/[AEIOU]/g, '') };
}

function resultMatches(result: SearchResult, terms: SearchTerms): boolean {
  if (!terms.text) return true;
  const fields = result.searchFields || prepareSearchResult(result).searchFields || [];
  return fields.some((field) => (
    field.text.includes(terms.text)
    || field.compact.includes(terms.compact)
    || (terms.noVowels.length >= 3 && field.noVowels.includes(terms.noVowels))
  ));
}

function resultStartsWith(result: SearchResult, terms: SearchTerms): boolean {
  if (!terms.compact) return false;
  const fields = result.searchFields || prepareSearchResult(result).searchFields || [];
  return fields.some((field) => field.compact.startsWith(terms.compact));
}

export default function GlobalTickerSearch({
  currentTicker = null,
  onSelectTicker,
  onSelectFund,
}: GlobalTickerSearchProps) {
  const [stockRows, setStockRows] = useState<MarketUniverseRow[]>([]);
  const [stocksLoading, setStocksLoading] = useState(false);
  const [fundRows, setFundRows] = useState<FundSummary[]>([]);
  const [fundsLoading, setFundsLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const [recentSearches, setRecentSearches] = useState<SearchResult[]>(() => readRecentSearches());
  const inputRef = useRef<HTMLInputElement | null>(null);

  const deferredQuery = useDeferredValue(query);
  const trimmedQuery = deferredQuery.trim();
  const normalizedQuery = normalizeSearchText(trimmedQuery);
  const queryTerms = useMemo(() => searchTerms(normalizedQuery), [normalizedQuery]);
  const normalizedSymbolQuery = normalizeSymbol(query.trim());

  useEffect(() => {
    if (!isOpen || trimmedQuery.length < 1) {
      setStockRows([]);
      setStocksLoading(false);
      return;
    }
    let active = true;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setStocksLoading(true);
      apiClient.marketStockSearch({
        q: trimmedQuery,
        index: 'XUTUM',
        limit: 12,
        signal: controller.signal,
      })
        .then((payload) => {
          if (active) setStockRows(payload.rows || []);
        })
        .catch((error) => {
          if ((error as Error)?.name === 'AbortError') return;
          if (active) setStockRows([]);
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
  }, [isOpen, trimmedQuery]);

  useEffect(() => {
    if (!isOpen || trimmedQuery.length < MIN_FUND_SEARCH_QUERY_LENGTH) {
      setFundRows([]);
      setFundsLoading(false);
      return;
    }
    let active = true;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      if (!active) return;
      setFundsLoading(true);
      apiClient
        .fundSearch(trimmedQuery, 12, { signal: controller.signal })
        .then((payload) => {
          if (active) setFundRows(payload.rows || []);
        })
        .catch((error) => {
          if ((error as Error)?.name === 'AbortError') return;
          if (active) setFundRows([]);
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
  }, [isOpen, trimmedQuery]);

  useEffect(() => {
    if (!isOpen) return;
    const timer = window.setTimeout(() => inputRef.current?.focus(), 0);
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
        setQuery('');
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  const stockBaseResults = useMemo(() => {
    return stockRows.map(stockRowToResult).map(prepareSearchResult);
  }, [stockRows]);

  const stockResults = useMemo(() => {
    return stockBaseResults
      .filter((item) => item.symbol !== normalizeSymbol(currentTicker))
      .filter((item) => resultMatches(item, queryTerms))
      .map((item) => ({ item, starts: resultStartsWith(item, queryTerms) ? 0 : 1 }))
      .sort((a, b) => a.starts - b.starts || a.item.symbol.localeCompare(b.item.symbol, 'tr'))
      .slice(0, 10)
      .map(({ item }) => item);
  }, [currentTicker, queryTerms, stockBaseResults]);

  const fundResults = useMemo(() => {
    return fundRows
      .map(fundRowToResult)
      .map(prepareSearchResult)
      .filter((item) => resultMatches(item, queryTerms))
      .slice(0, 10);
  }, [fundRows, queryTerms]);

  const visibleResults = useMemo(() => {
    if (!normalizedQuery) return recentSearches;
    const seen = new Set<string>();
    return [...stockResults, ...fundResults].filter((item) => {
      if (seen.has(item.id)) return false;
      seen.add(item.id);
      return true;
    }).slice(0, 14);
  }, [fundResults, normalizedQuery, recentSearches, stockResults]);

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

  const commitResult = (result?: SearchResult) => {
    const selected = result || visibleResults[highlightedIndex];
    if (!selected) {
      if (normalizedSymbolQuery.length >= 2) {
        onSelectTicker(normalizedSymbolQuery);
        setIsOpen(false);
        setQuery('');
      }
      return;
    }
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
      commitResult();
    }
  };

  const isSearching = Boolean(normalizedQuery) && (
    (stocksLoading && stockBaseResults.length === 0) ||
    fundsLoading
  );

  return (
    <div className="global-search">
      <button
        type="button"
        className="global-search-trigger"
        onClick={() => setIsOpen(true)}
        aria-haspopup="dialog"
        aria-expanded={isOpen}
      >
        <Search size={17} aria-hidden="true" />
        <span>Ara</span>
      </button>

      {isOpen && (
        <div className="global-search-modal-backdrop" role="presentation" onMouseDown={() => setIsOpen(false)}>
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
                    <button
                      key={result.id}
                      type="button"
                      className={`global-search-result${index === highlightedIndex ? ' active' : ''}`}
                      onMouseEnter={() => setHighlightedIndex(index)}
                      onClick={() => commitResult(result)}
                      role="option"
                      aria-selected={index === highlightedIndex}
                    >
                      <SymbolLogo
                        symbol={result.symbol}
                        name={result.logoName || result.subtitle || result.title}
                        kind={result.type === 'fund' ? 'fund' : 'stock'}
                        logoUrl={result.logoUrl}
                        size="md"
                      />
                      <span className="global-search-result-main">
                        <strong>{result.symbol}</strong>
                        <em>{result.subtitle}</em>
                      </span>
                      <span className="global-search-result-meta">
                        {result.price == null && result.changePct == null ? (
                          <span className="global-search-result-kind">
                            {result.type === 'fund' ? 'Fon' : 'Hisse'}
                          </span>
                        ) : (
                          <>
                            <span className="global-search-result-price">
                              {result.type === 'stock' && result.price != null ? <small>G</small> : null}
                              {formatSearchPrice(result.price, result.type)}
                            </span>
                            <span className={`global-search-result-change ${changeClass(result.changePct)}`}>
                              {formatSearchPct(result.changePct)}
                            </span>
                          </>
                        )}
                      </span>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="global-search-empty">
                  {isSearching
                    ? 'Aranıyor...'
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
