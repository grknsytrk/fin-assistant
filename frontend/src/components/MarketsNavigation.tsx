import { 
    Calendar, 
    Layers, 
    Coins, 
    ListOrdered, 
    Box, 
    Users, 
    Crosshair, 
    TerminalSquare, 
    MonitorPlay, 
    Atom, 
    GraduationCap, 
    ChevronRight,
    ArrowUpRight,
    PanelLeftClose,
} from 'lucide-react';
import GlobalTickerSearch from './GlobalTickerSearch';
import './MarketsNavigation.css';

export type MarketsNavigationSection = 'markets' | 'stocks' | 'funds' | 'indices';

type MarketsNavigationProps = {
    collapsed: boolean;
    activeSection: MarketsNavigationSection;
    onCollapsedChange: (collapsed: boolean) => void;
    onSectionChange: (section: MarketsNavigationSection) => void;
    onSelectTicker?: (ticker: string) => void;
    onSelectFund?: (fundCode: string) => void;
};

export default function MarketsNavigation({
    collapsed,
    activeSection,
    onCollapsedChange,
    onSectionChange,
    onSelectTicker,
    onSelectFund,
}: MarketsNavigationProps) {
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
                <button
                    type="button"
                    className={`mn-menu-item ${activeSection === 'markets' ? 'active' : ''}`}
                    title="Piyasalar"
                    onClick={() => onSectionChange('markets')}
                >
                    <div className="mn-item-main">
                        <Calendar size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Piyasalar</span>}
                    </div>
                </button>
                <button
                    type="button"
                    className={`mn-menu-item ${activeSection === 'stocks' ? 'active' : ''}`}
                    title="Hisseler"
                    onClick={() => onSectionChange('stocks')}
                >
                    <div className="mn-item-main">
                        <Layers size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Hisseler</span>}
                    </div>
                </button>
                <button
                    type="button"
                    className={`mn-menu-item ${activeSection === 'funds' ? 'active' : ''}`}
                    title="Fonlar"
                    onClick={() => onSectionChange('funds')}
                >
                    <div className="mn-item-main">
                        <Coins size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Fonlar</span>}
                    </div>
                </button>
                <button
                    type="button"
                    className={`mn-menu-item ${activeSection === 'indices' ? 'active' : ''}`}
                    title="Endeksler"
                    onClick={() => onSectionChange('indices')}
                >
                    <div className="mn-item-main">
                        <ListOrdered size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Endeksler</span>}
                    </div>
                </button>
                <button
                    type="button"
                    className="mn-menu-item"
                    title="Aracı Kurumlar (yakında)"
                    disabled
                    aria-disabled="true"
                >
                    <div className="mn-item-main">
                        <Box size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Aracı Kurumlar</span>}
                    </div>
                </button>
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

            <div className="mn-divider"></div>

            {/* Group 2 */}
            <div className="mn-menu-group">
                <button
                    type="button"
                    className="mn-menu-item"
                    title="Trade Ekranı (yakında)"
                    disabled
                    aria-disabled="true"
                >
                    <div className="mn-item-main">
                        <TerminalSquare size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Trade Ekranı</span>}
                    </div>
                </button>
                <button
                    type="button"
                    className="mn-menu-item"
                    title="Terminal (yakında)"
                    disabled
                    aria-disabled="true"
                >
                    <div className="mn-item-main">
                        <MonitorPlay size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Terminal</span>}
                    </div>
                    {!collapsed && <ArrowUpRight size={14} className="mn-external-icon" />}
                </button>
            </div>

            <div className="mn-divider"></div>

            {/* Group 3 */}
            <div className="mn-menu-group">
                <button
                    type="button"
                    className="mn-menu-item has-submenu"
                    title="Araştırma (yakında)"
                    disabled
                    aria-disabled="true"
                >
                    <div className="mn-item-main">
                        <Atom size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Araştırma</span>}
                    </div>
                    {!collapsed && <ChevronRight size={14} className="mn-chevron" />}
                </button>
                <button
                    type="button"
                    className="mn-menu-item"
                    title="SPL Eğitimleri (yakında)"
                    disabled
                    aria-disabled="true"
                >
                    <div className="mn-item-main">
                        <GraduationCap size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">SPL Eğitimleri</span>}
                    </div>
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
