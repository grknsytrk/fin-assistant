import { useEffect, useState } from 'react';

const THINKING_STEPS = [
    'Mali tablolar ve son dönem finansalları inceleniyor...',
    'Kârlılık, marj trendleri ve nakit akışı analiz ediliyor...',
    'Değerleme çarpanları ve bilanço dengesi kontrol ediliyor...',
    'Şirket performansı ve finansal göstergeler değerlendiriliyor...',
    'Analiz özeti ve finansal skor kartı hazırlanıyor...',
];

interface OverviewAiLoadingIllustrationProps {
    reasoningEffort?: 'low' | 'medium' | 'high';
}

export function OverviewAiLoadingIllustration({
    reasoningEffort = 'low',
}: OverviewAiLoadingIllustrationProps) {
    const [elapsedSeconds, setElapsedSeconds] = useState(0);

    useEffect(() => {
        setElapsedSeconds(0);
        const timer = window.setInterval(() => {
            setElapsedSeconds((prev) => prev + 1);
        }, 1000);
        return () => window.clearInterval(timer);
    }, []);

    const stepIndex = Math.min(
        Math.floor(elapsedSeconds / 3),
        THINKING_STEPS.length - 1,
    );
    const currentStep = THINKING_STEPS[stepIndex];

    const effortLabel =
        reasoningEffort === 'high'
            ? 'High'
            : reasoningEffort === 'medium'
                ? 'Medium'
                : 'Low';

    return (
        <div className="overview-ai-thinking-card" role="status" aria-live="polite">
            <div className="overview-ai-thinking-meta">
                <span className="overview-ai-thinking-indicator" aria-hidden="true">
                    <span className="overview-ai-thinking-dot" />
                    <span className="overview-ai-thinking-ping" />
                </span>
                <span className="overview-ai-thinking-timer">
                    {elapsedSeconds}sn süredir {elapsedSeconds >= 4 ? 'düşünüyor...' : 'çalışıyor...'}
                </span>
                <span className="overview-ai-thinking-badge">
                    IFM · {effortLabel}
                </span>
            </div>
            <div className="overview-ai-thinking-headline">
                <span key={stepIndex} className="overview-ai-thinking-shimmer">
                    {currentStep}
                </span>
            </div>
        </div>
    );
}
