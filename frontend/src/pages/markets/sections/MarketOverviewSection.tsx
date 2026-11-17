import type { DragEvent as ReactDragEvent } from 'react';
import { Plus, Search, X } from 'lucide-react';
import type {
    FundSummary,
    MarketIndexCode,
    MarketIndexDetailResponse,
    MarketStockCardItem,
    MarketStockCardsResponse,
    MarketStockRow,
    MarketUniverseResponse,
    MarketUniverseRow,
} from '../../../api/types';
import MarketWatchStrip from '../../../components/MarketWatchStrip';
import SymbolLogo from '../../../components/SymbolLogo';
import AnimatedChartSkeleton from '../../../components/charts/AnimatedChartSkeleton';
import { TerminalClock } from '../../../components/TerminalClock';
import type { WatchlistItem } from '../../../hooks/useWatchlist';
import { MarketStockCard } from '../components/MarketStockCard';
import { MobileMarketOverview } from '../components/MobileMarketOverview';
import { MAX_STOCK_CARDS, emptyStockCardItem } from '../stockScreenerConfig';
import type { StockCardDropPlacement } from '../types';

export interface MarketOverviewSectionProps {
    loading: boolean;
    error: string | null;
    market: MarketUniverseResponse | null;
    isMobileViewport: boolean;
    mobileIndexDetail: MarketIndexDetailResponse | null;
    mobileWatchlistSearchOpen: boolean;
    mobileWatchlistFundRows: Record<string, FundSummary | null>;
    stockQuoteRows: MarketStockRow[];
    watchlistItems: WatchlistItem[];
    stockCardSymbols: string[];
    stockCards: MarketStockCardsResponse | null;
    stockCardsLoading: boolean;
    stockCardsError: string | null;
    stockCardsBySymbol: Map<string, MarketStockCardItem>;
    stockCardPendingSet: Set<string>;
    canAddStockCards: boolean;
    stockCardPickerOpen: boolean;
    stockCardSearchTerm: string;
    stockCardCandidates: MarketUniverseRow[];
    draggingStockCardSymbol: string | null;
    onRetryLoadStats: () => void;
    onSelectTicker: (ticker: string) => void;
    onSelectIndex: (index: MarketIndexCode) => void;
    onOpenFund?: (fundCode: string) => void;
    onAddWatchlistStock: (symbol: string) => void;
    onAddWatchlistFund: (fund: FundSummary) => void;
    onRemoveWatchlistItem: (item: WatchlistItem) => void;
    onMobileWatchlistSearchOpenChange: (open: boolean) => void;
    onToggleStockCardPicker: () => void;
    onCloseStockCardPicker: () => void;
    onStockCardSearchTermChange: (value: string) => void;
    onAddStockCard: (symbol: string) => void;
    onRetryStockCards: () => void;
    onOpenStockCardPicker: () => void;
    onStockCardGridDragOver: (event: ReactDragEvent<HTMLDivElement>) => void;
    onStockCardMoveStart: (symbol: string) => void;
    onStockCardMoveOver: (symbol: string, placement: StockCardDropPlacement) => void;
    onStockCardMoveEnd: () => void;
    onRemoveStockCard: (symbol: string) => void;
}

export function MarketOverviewSection({
    loading,
    error,
    market,
    isMobileViewport,
    mobileIndexDetail,
    mobileWatchlistSearchOpen,
    mobileWatchlistFundRows,
    stockQuoteRows,
    watchlistItems,
    stockCardSymbols,
    stockCards,
    stockCardsLoading,
    stockCardsError,
    stockCardsBySymbol,
    stockCardPendingSet,
    canAddStockCards,
    stockCardPickerOpen,
    stockCardSearchTerm,
    stockCardCandidates,
    draggingStockCardSymbol,
    onRetryLoadStats,
    onSelectTicker,
    onSelectIndex,
    onOpenFund,
    onAddWatchlistStock,
    onAddWatchlistFund,
    onRemoveWatchlistItem,
    onMobileWatchlistSearchOpenChange,
    onToggleStockCardPicker,
    onCloseStockCardPicker,
    onStockCardSearchTermChange,
    onAddStockCard,
    onRetryStockCards,
    onOpenStockCardPicker,
    onStockCardGridDragOver,
    onStockCardMoveStart,
    onStockCardMoveOver,
    onStockCardMoveEnd,
    onRemoveStockCard,
}: MarketOverviewSectionProps) {
    return (
        <>
            <div className="market-terminal-shell">
                <div className="market-overview-head">
                    <div className="market-title">
                        <h1>Piyasa Görünümü</h1>
                        <p>Güncel fiyatlar, finansal görünüm ve seçtiğiniz hisse kartları tek ekranda.</p>
                    </div>
                </div>
                <div className="market-session-bar">
                    <div className="market-session-date">
                        <span className="market-session-dot" aria-label="Piyasa izleme aktif" />
                        <TerminalClock intervalMs={isMobileViewport ? 10000 : 1000} />
                    </div>
                </div>
                <MarketWatchStrip variant="compact" onSelectIndex={onSelectIndex} />
            </div>

            {loading && (
                <div className="market-content market-content-terminal" role="status" aria-live="polite" aria-label="Piyasa verileri yükleniyor">
                    <div className="market-main-column">
                        <div className="stock-card-skeleton-grid" aria-hidden="true">
                            {[0, 1, 2].map((cardIdx) => (
                                <div key={cardIdx} className="stock-card-skeleton">
                                    <div className="stock-card-skeleton-head">
                                        <div className="stock-card-skeleton-logo stock-card-skeleton-pulse" />
                                        <div className="stock-card-skeleton-id">
                                            <div className="stock-card-skeleton-symbol stock-card-skeleton-pulse" />
                                            <div className="stock-card-skeleton-company stock-card-skeleton-pulse" />
                                        </div>
                                    </div>
                                    <div className="stock-card-skeleton-price-row">
                                        <div className="stock-card-skeleton-price stock-card-skeleton-pulse" />
                                        <div className="stock-card-skeleton-change stock-card-skeleton-pulse" />
                                    </div>
                                    <div className="stock-card-skeleton-metrics">
                                        <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                        <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                        <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                    </div>
                                    <div className="stock-card-skeleton-chart">
                                        <AnimatedChartSkeleton
                                            seed={`overview-loading-${cardIdx}`}
                                            width={340}
                                            height={100}
                                            padding={{ top: 10, right: 6, bottom: 12, left: 6 }}
                                            preserveAspectRatio="none"
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {error && (
                <div className="error-message">
                    <strong>Hata:</strong> {error}
                    <button onClick={onRetryLoadStats} className="btn-secondary" style={{ marginLeft: '1rem' }}>Tekrar Dene</button>
                </div>
            )}

            {!loading && !error && market && (
                <div className={`market-content market-content-terminal${mobileWatchlistSearchOpen ? ' mobile-watchlist-search-open' : ''}`}>
                    <div className="market-main-column">
                        <MobileMarketOverview
                            index={mobileIndexDetail}
                            rows={market.rows}
                            stockQuoteRows={stockQuoteRows}
                            stockCardItems={stockCards?.items ?? []}
                            watchlistItems={watchlistItems}
                            fundRows={mobileWatchlistFundRows}
                            isMobileViewport={isMobileViewport}
                            onSelectTicker={onSelectTicker}
                            onSelectIndex={onSelectIndex}
                            onOpenFund={onOpenFund}
                            onAddStock={onAddWatchlistStock}
                            onAddFund={onAddWatchlistFund}
                            onRemoveWatchlistItem={onRemoveWatchlistItem}
                            onSearchOpenChange={onMobileWatchlistSearchOpenChange}
                        />
                        {!isMobileViewport && (
                            <section className="panel stock-cards-panel">
                                <div className="stock-cards-toolbar">
                                    <div className="stock-cards-title-row">
                                        <h2>Hisse Kartları</h2>
                                        <span className="panel-kicker">
                                            {stockCardSymbols.length}/{MAX_STOCK_CARDS} kart
                                        </span>
                                        <button
                                            type="button"
                                            className="stock-card-add-link"
                                            onClick={onToggleStockCardPicker}
                                            disabled={!canAddStockCards}
                                        >
                                            <span>Ekle</span>
                                            <Plus size={16} aria-hidden="true" />
                                        </button>
                                    </div>
                                </div>

                                {stockCardPickerOpen && (
                                    <div className="stock-card-picker">
                                        <div className="stock-card-picker-search">
                                            <Search size={16} aria-hidden="true" />
                                            <input
                                                type="text"
                                                placeholder="Hisse ara..."
                                                value={stockCardSearchTerm}
                                                onChange={(event) => onStockCardSearchTermChange(event.target.value)}
                                                autoFocus
                                            />
                                            <button
                                                type="button"
                                                className="stock-card-picker-close"
                                                onClick={onCloseStockCardPicker}
                                                aria-label="Hisse seçiciyi kapat"
                                                title="Kapat"
                                            >
                                                <X size={16} aria-hidden="true" />
                                            </button>
                                        </div>
                                        <div className="stock-card-picker-heading">Hisse Senetleri</div>
                                        <div className="stock-card-picker-list">
                                            {stockCardCandidates.length > 0 ? (
                                                stockCardCandidates.map((row) => (
                                                    <button
                                                        key={row.company}
                                                        type="button"
                                                        onClick={() => onAddStockCard(row.company)}
                                                    >
                                                        <SymbolLogo
                                                            symbol={row.company}
                                                            name={row.company}
                                                            kind="stock"
                                                            logoUrl={row.logo_url}
                                                            size="sm"
                                                            className="stock-card-picker-logo"
                                                        />
                                                        <span className="stock-card-picker-copy">
                                                            <strong>{row.company}</strong>
                                                        </span>
                                                    </button>
                                                ))
                                            ) : (
                                                <div className="stock-card-picker-empty">
                                                    Eklenebilecek hisse bulunamadı.
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}

                                {stockCardsError && (
                                    <div className="stock-cards-inline-alert" role="status" aria-live="polite">
                                        <span>Hisse kartları güncellenemedi.</span>
                                        <button
                                            type="button"
                                            onClick={onRetryStockCards}
                                        >
                                            Tekrar dene
                                        </button>
                                    </div>
                                )}

                                {stockCardSymbols.length === 0 ? (
                                    <div className="stock-card-empty-state">
                                        <h3>Henüz hisse kartı yok</h3>
                                        <p>Takip etmek istediğiniz hisseleri ekleyin; fiyat, gün içi grafik ve temel piyasa verileri burada görünsün.</p>
                                        <button
                                            type="button"
                                            className="stock-card-empty-action"
                                            onClick={onOpenStockCardPicker}
                                        >
                                            <Plus size={16} aria-hidden="true" />
                                            Hisse kartı ekle
                                        </button>
                                    </div>
                                ) : !stockCards && !stockCardsError ? (
                                    <div className="stock-card-skeleton-grid">
                                        {stockCardSymbols.map((symbol) => (
                                            <div key={symbol} className="stock-card-skeleton">
                                                <div className="stock-card-skeleton-head">
                                                    <div className="stock-card-skeleton-logo stock-card-skeleton-pulse" />
                                                    <div className="stock-card-skeleton-id">
                                                        <div className="stock-card-skeleton-symbol stock-card-skeleton-pulse" />
                                                        <div className="stock-card-skeleton-company stock-card-skeleton-pulse" />
                                                    </div>
                                                </div>
                                                <div className="stock-card-skeleton-price-row">
                                                    <div className="stock-card-skeleton-price stock-card-skeleton-pulse" />
                                                    <div className="stock-card-skeleton-change stock-card-skeleton-pulse" />
                                                </div>
                                                <div className="stock-card-skeleton-metrics">
                                                    <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                                    <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                                    <div className="stock-card-skeleton-metric stock-card-skeleton-pulse" />
                                                </div>
                                                <div className="stock-card-skeleton-chart">
                                                    <AnimatedChartSkeleton
                                                        seed={`overview-card-${symbol}`}
                                                        width={340}
                                                        height={100}
                                                        padding={{ top: 10, right: 6, bottom: 12, left: 6 }}
                                                        preserveAspectRatio="none"
                                                        ariaLabel={`${symbol} grafiği yükleniyor`}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div
                                        className="stock-card-grid"
                                        onDragOver={onStockCardGridDragOver}
                                        onDrop={(event) => {
                                            event.preventDefault();
                                            onStockCardMoveEnd();
                                        }}
                                    >
                                        {stockCardSymbols.map((symbol) => {
                                            const item = stockCardsBySymbol.get(symbol) || emptyStockCardItem(symbol);
                                            const hasLoadedItem = stockCardsBySymbol.has(symbol);
                                            const isCardLoading =
                                                !hasLoadedItem
                                                && (stockCardPendingSet.has(symbol) || stockCardsLoading);
                                            return (
                                                <MarketStockCard
                                                    key={symbol}
                                                    item={item}
                                                    onOpen={() => onSelectTicker(symbol)}
                                                    onRemove={() => onRemoveStockCard(symbol)}
                                                    onMoveStart={() => onStockCardMoveStart(symbol)}
                                                    onMoveOver={(placement) => onStockCardMoveOver(symbol, placement)}
                                                    onMoveEnd={onStockCardMoveEnd}
                                                    isLoading={isCardLoading}
                                                    isDragging={draggingStockCardSymbol === symbol}
                                                />
                                            );
                                        })}
                                    </div>
                                )}
                            </section>
                        )}
                    </div>
                </div>
            )}
        </>
    );
}
