import type { AskRequest } from '../api/types';

interface SettingsPanelProps {
    settings: {
        retriever: AskRequest['retriever'];
        mode: AskRequest['mode'];
    };
    onSettingsChange: (newSettings: {
        retriever: AskRequest['retriever'];
        mode: AskRequest['mode'];
    }) => void;
}

export function SettingsPanel({ settings, onSettingsChange }: SettingsPanelProps) {
    const handleRetrieverChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        onSettingsChange({
            ...settings,
            retriever: e.target.value as AskRequest['retriever'],
        });
    };

    const handleModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        onSettingsChange({
            ...settings,
            mode: e.target.value as AskRequest['mode'],
        });
    };

    return (
        <div className="settings-panel card-glass">
            <h3>Ayarlar</h3>

            <div className="form-group">
                <label>Arama Modeli (Retriever)</label>
                <select
                    value={settings.retriever}
                    onChange={handleRetrieverChange}
                    className="select-field"
                >
                    <option value="v1">v1 (Basic Vector)</option>
                    <option value="v2">v2 (Vector with Boost)</option>
                    <option value="v3">v3 (Query Aware)</option>
                    <option value="v5">v5 (Hybrid Vector + BM25)</option>
                    <option value="v6">v6 (Cross Encoder Re-rank)</option>
                </select>
            </div>

            <div className="form-group">
                <label>Soru Tipi (Mode)</label>
                <select
                    value={settings.mode}
                    onChange={handleModeChange}
                    className="select-field"
                >
                    <option value="single">Tekil Soru (Single)</option>
                    <option value="trend">Trend Analizi (Trend)</option>
                </select>
            </div>

        </div>
    );
}
