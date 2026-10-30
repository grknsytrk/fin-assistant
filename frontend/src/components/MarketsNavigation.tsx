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
} from '../routing/routes';
import './MarketsNavigation.css';

export type MarketsNavigationSection = 'markets' | 'stocks' | 'funds' | 'indices';
export type MarketsNavigationFundSection = 'investment' | 'compare';

type MarketsNavigationProps = {
    collapsed: boolean;
    activeSection: MarketsNavigationSection;
    activeFundSection?: MarketsNavigationFundSection;
    onCollapsedChange: (collapsed: boolean) => void;
    onSectionChange: (section: MarketsNavigationSection) => void;
    onFundSectionChange?: (section: MarketsNavigationFundSection) => void;
    onSelectTicker?: (ticker: string) => void;
    onSelectFund?: (fundCode: string) => void;
};

export default function MarketsNavigation({
    collapsed,
    activeSection,
    activeFundSection = 'investment',
    onCollapsedChange,
    onSectionChange,
    onFundSectionChange,
    onSelectTicker,
    onSelectFund,
}: MarketsNavigationProps) {
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

    // Auto-open the submenu whenever the user is already on the funds section
    // (e.g. arriving via a fund detail URL), but allow manual collapse afterwards.
    useEffect(() => {
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
                <Link
                    to={stockHref}
                    className={`mn-menu-item ${activeSection === 'stocks' && !isModelPortfolioPage ? 'active' : ''}`}
                    title="Hisseler"
                    onClick={(event) => handleLinkClick(event, () => onSectionChange('stocks'))}
                >
                    <div className="mn-item-main">
                        <Layers size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Hisseler</span>}
                    </div>
                </Link>
                <div className={`mn-menu-subtree ${fundsExpanded ? 'is-open' : ''}`}>
                    <div className="mn-submenu-head">
                        <Link
                            to={toFunds()}
                            className={`mn-menu-item mn-submenu-link ${activeSection === 'funds' && !isModelPortfolioPage ? 'active' : ''}`}
                            title="Fonlar"
                            onClick={(event) => {
                                if (collapsed && event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey) {
                                    event.preventDefault();
                                    onCollapsedChange(false);
                                    setFundsExpanded(true);
                                    return;
                                }
                                handleLinkClick(event, onFundSectionChange
                                    ? () => onFundSectionChange('investment')
                                    : () => onSectionChange('funds'));
                            }}
                        >
                            <div className="mn-item-main">
                                <Coins size={18} className="mn-icon" />
                                {!collapsed && <span className="mn-text">Fonlar</span>}
                            </div>
                        </Link>
                        {!collapsed && (
                            <button
                                type="button"
                                className="mn-submenu-toggle"
                                aria-label={fundsExpanded ? 'Fonlar alt menüsünü daralt' : 'Fonlar alt menüsünü genişlet'}
                                aria-expanded={fundsExpanded}
                                onClick={() => setFundsExpanded((prev) => !prev)}
                            >
                                <ChevronRight size={14} className="mn-chevron" />
                            </button>
                        )}
                    </div>
                    {!collapsed && fundsExpanded && (
                        <div className="mn-submenu" aria-label="Fonlar alt menüsü">
                            <Link
                                to={toFunds()}
                                className={`mn-submenu-item ${activeSection === 'funds' && activeFundSection === 'investment' ? 'active' : ''}`}
                                onClick={(event) => handleLinkClick(event, onFundSectionChange
                                    ? () => onFundSectionChange('investment')
                                    : () => onSectionChange('funds'))}
                            >
                                Yatırım Fonları
                            </Link>
                            <Link
                                to={toFundComparison()}
                                className={`mn-submenu-item ${activeSection === 'funds' && activeFundSection === 'compare' ? 'active' : ''}`}
                                onClick={(event) => handleLinkClick(event, onFundSectionChange
                                    ? () => onFundSectionChange('compare')
                                    : undefined)}
                            >
                                Fon Karşılaştırma
                            </Link>
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
                        {!collapsed && <span className="mn-text">Model Portföy</span>}
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
