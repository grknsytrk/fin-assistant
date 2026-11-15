import './ModelPortfolioDonut.css';

export function ModelPortfolioDonut({
    items,
    activeIndex,
    onActiveIndexChange,
}: {
    items: Array<{ symbol: string; label: string; weight: number; color: string; isCash: boolean }>;
    activeIndex: number | null;
    onActiveIndexChange: (idx: number | null) => void;
}) {
    const radius = 64;
    const circumference = 2 * Math.PI * radius;
    const total = Math.max(1e-6, items.reduce((acc, item) => acc + Math.max(0, item.weight), 0));

    let accumulated = 0;
    const segments = items.map((item, idx) => {
        const fraction = Math.max(0, item.weight) / total;
        const length = fraction * circumference;
        const gap = Math.max(0, circumference - length);
        const offset = -accumulated;
        accumulated += length;
        return { item, idx, length, gap, offset };
    });

    const activeItem = activeIndex != null ? items[activeIndex] : null;

    return (
        <div className="mp-donut-wrap">
            <svg viewBox="0 0 160 160" className="mp-donut-svg">
                <circle cx="80" cy="80" r={radius} className="mp-donut-track" />
                {segments.map((seg) => (
                    <circle
                        key={seg.item.symbol}
                        cx="80"
                        cy="80"
                        r={radius}
                        fill="none"
                        stroke={seg.item.color}
                        strokeWidth={activeIndex === seg.idx ? 18 : 14}
                        strokeDasharray={`${seg.length} ${seg.gap}`}
                        strokeDashoffset={seg.offset}
                        transform="rotate(-90 80 80)"
                        className="mp-donut-segment"
                        style={{
                            opacity: activeIndex != null && activeIndex !== seg.idx ? 0.38 : 1,
                        }}
                        onMouseEnter={() => onActiveIndexChange(seg.idx)}
                        onMouseLeave={() => onActiveIndexChange(null)}
                    />
                ))}
            </svg>
            <div className="mp-donut-center">
                <span>{activeItem ? activeItem.symbol : 'HİSSE PAYI'}</span>
                <strong className="mp-mono">
                    %{activeItem ? activeItem.weight.toFixed(1) : items.filter((i) => !i.isCash).reduce((a, b) => a + b.weight, 0).toFixed(0)}
                </strong>
            </div>
        </div>
    );
}
