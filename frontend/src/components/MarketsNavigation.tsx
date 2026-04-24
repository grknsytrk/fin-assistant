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

type MarketsNavigationProps = {
    collapsed: boolean;
    activeSection: 'markets' | 'stocks' | 'indices';
    onCollapsedChange: (collapsed: boolean) => void;
    onSectionChange: (section: 'markets' | 'stocks' | 'indices') => void;
};

export default function MarketsNavigation({
    collapsed,
    activeSection,
    onCollapsedChange,
    onSectionChange,
}: MarketsNavigationProps) {
    const navigate = (ticker: string) => {
        const normalizedTicker = ticker.trim().toUpperCase();
        if (!normalizedTicker) return;
        window.location.href = '/?ticker=' + encodeURIComponent(normalizedTicker);
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
                <a href="#" className="mn-menu-item has-submenu" title="Fonlar">
                    <div className="mn-item-main">
                        <Coins size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Fonlar</span>}
                    </div>
                    {!collapsed && <ChevronRight size={14} className="mn-chevron" />}
                </a>
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
                <a href="#" className="mn-menu-item" title="Aracı Kurumlar">
                    <div className="mn-item-main">
                        <Box size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Aracı Kurumlar</span>}
                    </div>
                </a>
                <a href="#" className="mn-menu-item has-submenu" title="Sektörler">
                    <div className="mn-item-main">
                        <Users size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Sektörler</span>}
                    </div>
                    {!collapsed && <ChevronRight size={14} className="mn-chevron" />}
                </a>
                <a href="#" className="mn-menu-item has-submenu" title="Analizler">
                    <div className="mn-item-main">
                        <Crosshair size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Analizler</span>}
                    </div>
                    {!collapsed && <ChevronRight size={14} className="mn-chevron" />}
                </a>
            </div>

            <div className="mn-divider"></div>

            {/* Group 2 */}
            <div className="mn-menu-group">
                <a href="#" className="mn-menu-item" title="Trade Ekranı">
                    <div className="mn-item-main">
                        <TerminalSquare size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Trade Ekranı</span>}
                    </div>
                </a>
                <a href="#" className="mn-menu-item" title="Terminal">
                    <div className="mn-item-main">
                        <MonitorPlay size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Terminal</span>}
                    </div>
                    {!collapsed && <ArrowUpRight size={14} className="mn-external-icon" />}
                </a>
            </div>

            <div className="mn-divider"></div>

            {/* Group 3 */}
            <div className="mn-menu-group">
                <a href="#" className="mn-menu-item has-submenu" title="Araştırma">
                    <div className="mn-item-main">
                        <Atom size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">Araştırma</span>}
                    </div>
                    {!collapsed && <ChevronRight size={14} className="mn-chevron" />}
                </a>
                <a href="#" className="mn-menu-item" title="SPL Eğitimleri">
                    <div className="mn-item-main">
                        <GraduationCap size={18} className="mn-icon" />
                        {!collapsed && <span className="mn-text">SPL Eğitimleri</span>}
                    </div>
                </a>
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
                    <GlobalTickerSearch onSelectTicker={navigate} />
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
