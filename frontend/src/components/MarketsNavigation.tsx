import { useEffect, useState } from 'react';
import type { MouseEvent as ReactMouseEvent } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { 
    Calendar, 
    Layers, 
    Coins, 
    BriefcaseBusiness,
    ListOrdered, 
    Users, 
    Crosshair, 
    ChevronRight,
    PanelLeftClose,
} from 'lucide-react';
import GlobalTickerSearch from './GlobalTickerSearch';
import {
    DEFAULT_MARKET_INDEX,
    ROUTE_PATHS,
    toFundComparison,
    toFunds,
    toMarketsIndices,
    toMarketsOverview,
    toMarketsStocks,
    toModelPortfolio,
    toStockComparison,
} from '../routing/routes';
import './MarketsNavigation.css';

export type MarketsNavigationSection = 'markets' | 'stocks' | 'funds' | 'indices';
export type MarketsNavigationStockSection = 'list' | 'compare';
export type MarketsNavigationFundSection = 'investment' | 'compare';

type MarketsNavigationProps = {
    collapsed: boolean;
    activeSection: MarketsNavigationSection;
    activeStockSection?: MarketsNavigationStockSection;
    activeFundSection?: MarketsNavigationFundSection;
    onCollapsedChange: (collapsed: boolean) => void;
    onSectionChange: (section: MarketsNavigationSection) => void;
    onStockSectionChange?: (section: MarketsNavigationStockSection) => void;
    onFundSectionChange?: (section: MarketsNavigationFundSection) => void;
    onSelectTicker?: (ticker: string) => void;
    onSelectFund?: (fundCode: string) => void;
};

export default function MarketsNavigation({
    collapsed,
    activeSection,
    activeStockSection = 'list',
    activeFundSection = 'investment',
    onCollapsedChange,
    onSectionChange,
    onStockSectionChange,
    onFundSectionChange,
    onSelectTicker,
    onSelectFund,
}: MarketsNavigationProps) {
    const [stocksExpanded, setStocksExpanded] = useState<boolean>(activeSection === 'stocks');
    const [fundsExpanded, setFundsExpanded] = useState<boolean>(activeSection === 'funds');
    const location = useLocation();
    const isModelPortfolioPage = location.pathname === ROUTE_PATHS.modelPortfolio;
    const stockHref = activeSection === 'stocks' && location.pathname.startsWith(ROUTE_PATHS.marketsStocks)
        ? `${location.pathname}${location.search}`
        : toMarketsStocks(DEFAULT_MARKET_INDEX);

    const handleLinkClick = (event: ReactMouseEvent<HTMLAnchorElement>, action?: () => void) => {
        if (
            !action
            || event.button !== 0
            || event.metaKey
            || event.ctrlKey
            || event.shiftKey
            || event.altKey
        ) {
            return;
        }
        event.preventDefault();
        action();
    };

    // Auto-open the submenu whenever the user is already on the stocks or funds section
    // (e.g. arriving via a stock/fund detail URL), but allow manual collapse afterwards.
    useEffect(() => {
        if (activeSection === 'stocks') {
            setStocksExpanded(true);
        }
        if (activeSection === 'funds') {
            setFundsExpanded(true);
        }
    }, [activeSection]);

    const navigate = (ticker: string) => {
        const normalizedTicker = ticker.trim().toUpperCase();
        if (!normalizedTicker) return;
        if (onSelectTicker) {
            onSelectTicker(normalizedTicker);
        } else {
            window.location.href = '/?ticker=' + encodeURIComponent(normalizedTicker);
        }
    };

    const navItems = (
        <>
            {/* Group 1 */}
            <div className="mn-menu-group">
                <Link
                    to={toMarketsOverview()}
                    className={`mn-menu-item ${activeSection === 'markets' && !isModelPortfolioPage ? 'active' : ''}`}
                    title="Piyasalar"
                    onClick={(event) => handleLinkClick(event, () => onSectionChange('markets'))}
                >
                    <div className="mn-item-main">
                        <Calendar size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Piyasalar</span>}
                    </div>
                </Link>
                <div className={`mn-menu-subtree ${stocksExpanded ? 'is-open' : ''}`}>
                    <button
                        type="button"
                        className={`mn-menu-item has-submenu ${activeSection === 'stocks' && !isModelPortfolioPage ? 'active' : ''}`}
                        title="Hisseler"
                        aria-expanded={stocksExpanded}
                        onClick={() => {
                            if (typeof window !== 'undefined' && window.matchMedia?.('(max-width: 760px)').matches) {
                                onSectionChange('stocks');
                                return;
                            }
                            if (collapsed) {
                                onCollapsedChange(false);
                                setStocksExpanded(true);
                                return;
                            }
                            setStocksExpanded((prev) => !prev);
                        }}
                    >
                        <div className="mn-item-main">
                            <Layers size={18} className="mn-icon" />
                            {!collapsed && <span className="mn-text">Hisseler</span>}
                        </div>
                        {!collapsed && <ChevronRight size={14} className="mn-chevron" />}
                    </button>
                    {!collapsed && (
                        <div className={`mn-submenu-collapse ${stocksExpanded ? 'is-open' : ''}`} aria-hidden={!stocksExpanded}>
                            <div className="mn-submenu-collapse-inner">
                                <div className="mn-submenu" aria-label="Hisseler alt menüsü">
                                    <Link
                                        to={stockHref}
                                        tabIndex={stocksExpanded ? undefined : -1}
                                        className={`mn-submenu-item ${activeSection === 'stocks' && activeStockSection === 'list' ? 'active' : ''}`}
                                        onClick={(event) => handleLinkClick(event, onStockSectionChange
                                            ? () => onStockSectionChange('list')
                                            : () => onSectionChange('stocks'))}
                                    >
                                        Hisse Listesi
                                    </Link>
                                    <Link
                                        to={toStockComparison()}
                                        tabIndex={stocksExpanded ? undefined : -1}
                                        className={`mn-submenu-item ${activeSection === 'stocks' && activeStockSection === 'compare' ? 'active' : ''}`}
                                        onClick={(event) => handleLinkClick(event, onStockSectionChange
                                            ? () => onStockSectionChange('compare')
                                            : undefined)}
                                    >
                                        Hisse Karşılaştırma
                                    </Link>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
                <div className={`mn-menu-subtree ${fundsExpanded ? 'is-open' : ''}`}>
                    <button
                        type="button"
                        className={`mn-menu-item has-submenu ${activeSection === 'funds' && !isModelPortfolioPage ? 'active' : ''}`}
                        title="Fonlar"
                        aria-expanded={fundsExpanded}
                        onClick={() => {
                            if (collapsed) {
                                onCollapsedChange(false);
                                setFundsExpanded(true);
                                return;
                            }
                            setFundsExpanded((prev) => !prev);
                        }}
                    >
                        <div className="mn-item-main">
                            <Coins size={18} className="mn-icon" />
                            {!collapsed && <span className="mn-text">Fonlar</span>}
                        </div>
                        {!collapsed && <ChevronRight size={14} className="mn-chevron" />}
                    </button>
                    {!collapsed && (
                        <div className={`mn-submenu-collapse ${fundsExpanded ? 'is-open' : ''}`} aria-hidden={!fundsExpanded}>
                            <div className="mn-submenu-collapse-inner">
                                <div className="mn-submenu" aria-label="Fonlar alt menüsü">
                                    <Link
                                        to={toFunds()}
                                        tabIndex={fundsExpanded ? undefined : -1}
                                        className={`mn-submenu-item ${activeSection === 'funds' && activeFundSection === 'investment' ? 'active' : ''}`}
                                        onClick={(event) => handleLinkClick(event, onFundSectionChange
                                            ? () => onFundSectionChange('investment')
                                            : () => onSectionChange('funds'))}
                                    >
                                        Yatırım Fonları
                                    </Link>
                                    <Link
                                        to={toFundComparison()}
                                        tabIndex={fundsExpanded ? undefined : -1}
                                        className={`mn-submenu-item ${activeSection === 'funds' && activeFundSection === 'compare' ? 'active' : ''}`}
                                        onClick={(event) => handleLinkClick(event, onFundSectionChange
                                            ? () => onFundSectionChange('compare')
                                            : undefined)}
                                    >
                                        Fon Karşılaştırma
                                    </Link>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
                <Link
                    to={toMarketsIndices()}
                    className={`mn-menu-item ${activeSection === 'indices' && !isModelPortfolioPage ? 'active' : ''}`}
                    title="Endeksler"
                    onClick={(event) => handleLinkClick(event, () => onSectionChange('indices'))}
                >
                    <div className="mn-item-main">
                        <ListOrdered size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Endeksler</span>}
                    </div>
                </Link>
                <NavLink
                    to={toModelPortfolio()}
                    className={({ isActive }) => `mn-menu-item ${isActive ? 'active' : ''}`}
                    title="Model Portföy"
                >
                    <div className="mn-item-main">
                        <BriefcaseBusiness size={18} className="mn-icon" />
                        {!collapsed && (
                            <span className="mn-text">
                                <span className="mn-text-long">Model Portföy</span>
                                <span className="mn-text-short">Portföy</span>
                            </span>
                        )}
                    </div>
                </NavLink>
                <button
                    type="button"
                    className="mn-menu-item has-submenu"
                    title="Sektörler (yakında)"
                    disabled
                    aria-disabled="true"
                >
                    <div className="mn-item-main">
                        <Users size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Sektörler</span>}
                    </div>
                    {!collapsed && <ChevronRight size={14} className="mn-chevron" />}
                </button>
                <button
                    type="button"
                    className="mn-menu-item has-submenu"
                    title="Analizler (yakında)"
                    disabled
                    aria-disabled="true"
                >
                    <div className="mn-item-main">
                        <Crosshair size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Analizler</span>}
                    </div>
                    {!collapsed && <ChevronRight size={14} className="mn-chevron" />}
                </button>
            </div>
        </>
    );

    return (
        <nav className={`mn-sidebar ${collapsed ? 'mn-collapsed' : ''}`}>
            {/* Header */}
            <div className="mn-header">
                <button
                    type="button"
                    className="mn-logo"
                    onClick={() => {
                        if (collapsed) onCollapsedChange(false);
                    }}
                    aria-label={collapsed ? 'Menüyü genişlet' : 'Fin-Rag'}
                    title={collapsed ? 'Menüyü genişlet' : 'Fin-Rag'}
                >
                    <div className="mn-logo-icon">F</div>
                    {!collapsed && <span className="mn-logo-text">Fin-Rag</span>}
                </button>
                {!collapsed && (
                    <button
                        type="button"
                        className="mn-sidebar-toggle"
                        onClick={() => onCollapsedChange(true)}
                        aria-label="Menüyü daralt"
                        title="Daralt"
                    >
                        <PanelLeftClose size={16} />
                    </button>
                )}
            </div>

            {!collapsed && (
                <div className="mn-sidebar-search">
                    <GlobalTickerSearch onSelectTicker={navigate} onSelectFund={onSelectFund} />
                </div>
            )}

            {navItems}

            <div className="mn-spacer"></div>

            {!collapsed && (
                <div className="mn-auth-actions">
                    <button className="mn-btn-login">Giriş yap</button>
                    <button className="mn-btn-signup">Ücretsiz kaydol</button>
                </div>
            )}
        </nav>
    );
}
