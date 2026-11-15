import { useEffect, useState } from 'react';

const THINKING_STEPS = [
    'Mali tablolar ve son dönem finansalları inceleniyor...',
    'Kârlılık, marj trendleri ve nakit akışı analiz ediliyor...',
    'Değerleme çarpanları ve bilanço dengesi kontrol ediliyor...',
    'Şirket performansı ve finansal göstergeler değerlendiriliyor...',
    'Analiz özeti ve finansal skor kartı hazırlanıyor...',
];

interface OverviewAiLoadingIllustrationProps {
    providerLabel?: string;
    modelLabel?: string;
    reasoningEffort?: 'low' | 'medium' | 'high';
    supportsReasoningEffort?: boolean;
}

export function OverviewAiLoadingIllustration({
    providerLabel,
    modelLabel,
    reasoningEffort = 'low',
    supportsReasoningEffort = true,
}: OverviewAiLoadingIllustrationProps) {
    const [stepIndex, setStepIndex] = useState(0);

    useEffect(() => {
        setStepIndex(0);
        const timer = window.setInterval(() => {
            setStepIndex((prev) => Math.min(prev + 1, THINKING_STEPS.length - 1));
        }, 3000);
        return () => window.clearInterval(timer);
    }, []);

    const currentStep = THINKING_STEPS[stepIndex];

    const effortLabel =
        reasoningEffort === 'high'
            ? 'High'
            : reasoningEffort === 'medium'
                ? 'Medium'
                : 'Low';

    const displayModel = modelLabel
        ? (providerLabel && !modelLabel.toLowerCase().startsWith(providerLabel.toLowerCase())
            ? `${providerLabel} · ${modelLabel}`
            : modelLabel)
        : (providerLabel || 'AI');

    const badgeText = supportsReasoningEffort
        ? `${displayModel} · ${effortLabel}`
        : displayModel;

    return (
        <div className="overview-ai-thinking-card" role="status" aria-live="polite">
            <div className="overview-ai-thinking-meta">
                <span className="overview-ai-thinking-badge">
                    {badgeText}
                </span>
            </div>
            <div className="overview-ai-thinking-headline">
                <span key={stepIndex} className="overview-ai-thinking-shimmer">
                    {currentStep}
                </span>
            </div>
            <div className="overview-ai-thinking-lines" aria-hidden="true">
                <span className="overview-ai-thinking-line is-wide" />
                <span className="overview-ai-thinking-line is-medium" />
                <span className="overview-ai-thinking-line is-short" />
            </div>
        </div>
    );
}
