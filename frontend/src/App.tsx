import { useState } from 'react';
import { AskScreen } from './components/AskScreen';
import { SettingsPanel } from './components/SettingsPanel';
import { StatsCards } from './components/StatsCards';
import { ExportPanel } from './components/ExportPanel';
import type { AskRequest } from './api/types';
import './index.css';

function App() {
  const [settings, setSettings] = useState<{
    retriever: AskRequest['retriever'];
    mode: AskRequest['mode'];
  }>({
    retriever: 'v3',
    mode: 'single',
  });

  return (
    <div className="app-layout">
      <header className="app-header">
        <div className="header-content">
          <div className="logo-area">
            <h1>RAG-Fin</h1>
            <span className="badge-beta">React Beta</span>
          </div>
          <p className="subtitle">Banka, Sigorta, Sanayi - Q1/Q2/Q3 2024 Analiz</p>
        </div>
      </header>

      <main className="app-main">
        <div className="main-content">
          <AskScreen settings={settings} />
        </div>
        <aside className="app-sidebar">
          <SettingsPanel settings={settings} onSettingsChange={setSettings} />
          <StatsCards />
          <ExportPanel />
        </aside>
      </main>
    </div>
  );
}

export default App;
