import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { Search } from 'lucide-react';
import { apiClient } from '../api/client';
import './GlobalTickerSearch.css';

interface GlobalTickerSearchProps {
  currentTicker?: string | null;
  onSelectTicker: (ticker: string) => void;
}

export default function GlobalTickerSearch({
  currentTicker = null,
  onSelectTicker,
}: GlobalTickerSearchProps) {
  const [companies, setCompanies] = useState<string[]>([]);
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let active = true;
    apiClient
      .kapCompanies()
      .then((payload) => {
        if (!active) {
          return;
        }
        const nextCompanies = [...(payload.companies || [])]
          .map((company) => company.toUpperCase())
          .sort((a, b) => a.localeCompare(b, 'tr'));
        setCompanies(nextCompanies);
      })
      .catch(() => {
        if (active) {
          setCompanies([]);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handlePointerDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
    };
  }, []);

  const normalizedQuery = query.trim().toUpperCase();

  const suggestions = useMemo(() => {
    if (!companies.length) {
      return [];
    }

    if (!normalizedQuery) {
      const current = currentTicker?.toUpperCase();
      return companies.filter((company) => company !== current).slice(0, 8);
    }

    const startsWithMatches = companies.filter((company) => company.startsWith(normalizedQuery));
    const includesMatches = companies.filter(
      (company) => !company.startsWith(normalizedQuery) && company.includes(normalizedQuery),
    );
    return [...startsWithMatches, ...includesMatches].slice(0, 8);
  }, [companies, currentTicker, normalizedQuery]);

  useEffect(() => {
    setHighlightedIndex(0);
  }, [normalizedQuery, isOpen]);

  useEffect(() => {
    setQuery('');
    setIsOpen(false);
    setHighlightedIndex(0);
  }, [currentTicker]);

  const commitSelection = (candidate?: string) => {
    const typedTicker = (candidate || normalizedQuery).trim().toUpperCase();
    if (!typedTicker) {
      return;
    }

    const exactMatch = companies.find((company) => company === typedTicker);
    const suggestedMatch = suggestions[0];
    const nextTicker = exactMatch || suggestedMatch || (typedTicker.length >= 3 ? typedTicker : null);
    if (!nextTicker) {
      return;
    }

    onSelectTicker(nextTicker);
    setQuery('');
    setIsOpen(false);
    setHighlightedIndex(0);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Escape') {
      setIsOpen(false);
      return;
    }

    if (event.key === 'ArrowDown') {
      event.preventDefault();
      if (!isOpen) {
        setIsOpen(true);
        return;
      }
      if (suggestions.length > 0) {
        setHighlightedIndex((prev) => (prev + 1) % suggestions.length);
      }
      return;
    }

    if (event.key === 'ArrowUp') {
      event.preventDefault();
      if (!isOpen) {
        setIsOpen(true);
        return;
      }
      if (suggestions.length > 0) {
        setHighlightedIndex((prev) => (prev === 0 ? suggestions.length - 1 : prev - 1));
      }
      return;
    }

    if (event.key === 'Enter') {
      event.preventDefault();
      if (isOpen && suggestions.length > 0) {
        commitSelection(suggestions[highlightedIndex]);
        return;
      }
      commitSelection();
    }
  };

  return (
    <div className="global-search" ref={rootRef}>
      <div className="global-search-shell">
        <Search size={16} className="global-search-icon" />
        <input
          type="text"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          onKeyDown={handleKeyDown}
          className="global-search-input"
          placeholder={currentTicker ? `Hisse değiştir... (${currentTicker})` : 'Hisse ara...'}
          aria-label="Hisse ara"
        />
        <button
          type="button"
          className="global-search-action"
          onClick={() => commitSelection()}
          title="Hisse aç"
        >
          {currentTicker || '/'}
        </button>
      </div>

      {isOpen && (
        <div className="global-search-dropdown">
          {suggestions.length > 0 ? (
            suggestions.map((company, index) => (
              <button
                key={company}
                type="button"
                className={`global-search-item ${index === highlightedIndex ? 'active' : ''}`}
                onMouseEnter={() => setHighlightedIndex(index)}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => commitSelection(company)}
              >
                <span className="global-search-symbol">{company}</span>
                {company === currentTicker && <span className="global-search-badge">Aktif</span>}
              </button>
            ))
          ) : normalizedQuery ? (
            <button
              type="button"
              className="global-search-item global-search-item-manual"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => commitSelection(normalizedQuery)}
            >
              <span className="global-search-symbol">{normalizedQuery}</span>
              <span className="global-search-hint">Enter ile aç</span>
            </button>
          ) : (
            <div className="global-search-empty">KAP şirketleri yükleniyor...</div>
          )}
        </div>
      )}
    </div>
  );
}
