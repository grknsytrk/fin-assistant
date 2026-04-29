import re
with open('frontend/src/pages/MarketsView.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

helper = """
const isBistSymbol = (sym?: string) => {
    if (!sym) return true;
    const globalSymbols = ['SP500', 'NASDAQ', 'DOW', 'DAX', 'FTSE', 'CAC40', 'NIKKEI', 'HANGSENG', 'VIX', 'DXY'];
    if (globalSymbols.includes(sym.toUpperCase())) return false;
    if (sym.includes('/')) return false;
    return true;
};

function IndexLineChart({
"""
content = content.replace('function IndexLineChart({', helper)

index_line_chart_sig_old = """function IndexLineChart({
    points,
    prevClose,
    changePct,
}: {
    points: MarketIndexDetailResponse['line_points'];
    prevClose: number | null;
    changePct: number | null;
}) {"""
index_line_chart_sig_new = """function IndexLineChart({
    symbol,
    points,
    prevClose,
    changePct,
}: {
    symbol?: string;
    points: MarketIndexDetailResponse['line_points'];
    prevClose: number | null;
    changePct: number | null;
}) {"""
content = content.replace(index_line_chart_sig_old, index_line_chart_sig_new)

index_line_chart_xfor_old = """    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const xFor = (index: number) =>
        padding.left + (validPoints.length === 1 ? 0 : (index / (validPoints.length - 1)) * plotWidth);
    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;

    const last = validPoints[validPoints.length - 1].close;

    // Y ekseni çizgileri (5 adet)
    const tickCount = 6;
    const tickValues = Array.from({ length: tickCount }).map((_, i) => minValue + (span * i) / (tickCount - 1));

    // X ekseni (saat başları için tahmini çizim veya eşit dağılımlı)
    const timeTickCount = 8;
    const timeTicks = Array.from({ length: timeTickCount }).map((_, i) => Math.floor((validPoints.length - 1) * (i / (timeTickCount - 1))));"""

index_line_chart_xfor_new = """    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const isBist = isBistSymbol(symbol);
    const useTimeScale = isBist && validPoints.length > 0;
    let startTimeMs = 0;
    let endTimeMs = 0;
    if (useTimeScale) {
        const d = new Date(validPoints[0].time);
        const start = new Date(d); start.setHours(10, 0, 0, 0);
        const end = new Date(d); end.setHours(18, 0, 0, 0);
        startTimeMs = start.getTime();
        endTimeMs = end.getTime();
    }

    const xFor = (index: number) => {
        if (useTimeScale) {
            const pointTime = new Date(validPoints[index].time).getTime();
            let ratio = (pointTime - startTimeMs) / (endTimeMs - startTimeMs);
            ratio = Math.max(0, Math.min(1, ratio));
            return padding.left + ratio * plotWidth;
        }
        return padding.left + (validPoints.length === 1 ? 0 : (index / (validPoints.length - 1)) * plotWidth);
    };
    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;

    const last = validPoints[validPoints.length - 1].close;

    // Y ekseni çizgileri (5 adet)
    const tickCount = 6;
    const tickValues = Array.from({ length: tickCount }).map((_, i) => minValue + (span * i) / (tickCount - 1));

    // X ekseni (saat başları için tahmini çizim veya eşit dağılımlı)
    const timeTickLabels: Array<{x: number, label: string, key: string}> = [];
    if (useTimeScale) {
        for (let h = 10; h <= 18; h += 1) {
            const d = new Date(validPoints[0].time);
            d.setHours(h, 0, 0, 0);
            timeTickLabels.push({
                x: padding.left + ((d.getTime() - startTimeMs) / (endTimeMs - startTimeMs)) * plotWidth,
                label: `${h.toString().padStart(2, '0')}:00`,
                key: `fixed-${h}`
            });
        }
    } else {
        const timeTickCount = 8;
        const timeTicks = Array.from({ length: timeTickCount }).map((_, i) => Math.floor((validPoints.length - 1) * (i / (timeTickCount - 1))));
        timeTicks.forEach(index => {
            const dt = new Date(validPoints[index].time);
            const label = Number.isNaN(dt.getTime()) ? '' : dt.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
            timeTickLabels.push({ x: xFor(index), label, key: `dyn-${index}` });
        });
    }"""
content = content.replace(index_line_chart_xfor_old, index_line_chart_xfor_new)

index_line_chart_ticks_old = """            {timeTicks.map((index) => {
                const dt = new Date(validPoints[index].time);
                const label = Number.isNaN(dt.getTime())
                    ? ''
                    : dt.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
                const x = xFor(index);
                return (
                    <g key={index}>
                        <line
                            x1={x}
                            x2={x}
                            y1={padding.top}
                            y2={height - padding.bottom}
                            stroke="rgba(255,255,255,0.03)"
                            strokeWidth="1"
                        />
                        <text x={x} y={height - 15} fill="rgba(255,255,255,0.5)" fontSize="11" fontFamily="monospace" textAnchor="middle">
                            {label}
                        </text>
                    </g>
                );
            })}"""

index_line_chart_ticks_new = """            {timeTickLabels.map(({ x, label, key }) => (
                <g key={key}>
                    <line
                        x1={x}
                        x2={x}
                        y1={padding.top}
                        y2={height - padding.bottom}
                        stroke="rgba(255,255,255,0.03)"
                        strokeWidth="1"
                    />
                    <text x={x} y={height - 15} fill="rgba(255,255,255,0.5)" fontSize="11" fontFamily="monospace" textAnchor="middle">
                        {label}
                    </text>
                </g>
            ))}"""
content = content.replace(index_line_chart_ticks_old, index_line_chart_ticks_new)

index_usage_old = """                            <section className="indices-chart-panel">
                                <IndexLineChart
                                    points={indexDetail.line_points}
                                    prevClose={indexDetail.prev_close}
                                    changePct={indexDetail.change_pct}
                                />
                            </section>"""
index_usage_new = """                            <section className="indices-chart-panel">
                                <IndexLineChart
                                    symbol={indexDetail.symbol}
                                    points={indexDetail.line_points}
                                    prevClose={indexDetail.prev_close}
                                    changePct={indexDetail.change_pct}
                                />
                            </section>"""
content = content.replace(index_usage_old, index_usage_new)

stock_card_mini_xfor_old = """    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const xFor = (index: number) =>
        padding.left + (validPoints.length === 1 ? 0 : (index / (validPoints.length - 1)) * plotWidth);
    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;
    const pathData = validPoints"""

stock_card_mini_xfor_new = """    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const isBist = isBistSymbol(symbol);
    const useTimeScale = isBist && selectedRange === '1d' && validPoints.length > 0;
    let startTimeMs = 0;
    let endTimeMs = 0;
    if (useTimeScale) {
        const d = new Date(validPoints[0].time);
        const start = new Date(d); start.setHours(10, 0, 0, 0);
        const end = new Date(d); end.setHours(18, 0, 0, 0);
        startTimeMs = start.getTime();
        endTimeMs = end.getTime();
    }

    const xFor = (index: number) => {
        if (useTimeScale) {
            const pointTime = new Date(validPoints[index].time).getTime();
            let ratio = (pointTime - startTimeMs) / (endTimeMs - startTimeMs);
            ratio = Math.max(0, Math.min(1, ratio));
            return padding.left + ratio * plotWidth;
        }
        return padding.left + (validPoints.length === 1 ? 0 : (index / (validPoints.length - 1)) * plotWidth);
    };

    const yFor = (value: number) => padding.top + ((maxValue - value) / span) * plotHeight;
    const pathData = validPoints"""
content = content.replace(stock_card_mini_xfor_old, stock_card_mini_xfor_new)

stock_card_mini_hover_old = """    const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        if (rect.width <= 0) return;
        const x = ((event.clientX - rect.left) / rect.width) * width;
        const rawIndex = Math.round(((x - padding.left) / plotWidth) * (validPoints.length - 1));
        setHoverIndex(clamp(rawIndex, 0, validPoints.length - 1));
    };"""

stock_card_mini_hover_new = """    const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        if (rect.width <= 0) return;
        const x = ((event.clientX - rect.left) / rect.width) * width;
        let closestIndex = 0;
        let minDiff = Infinity;
        for (let i = 0; i < validPoints.length; i++) {
            const diff = Math.abs(xFor(i) - x);
            if (diff < minDiff) {
                minDiff = diff;
                closestIndex = i;
            }
        }
        setHoverIndex(closestIndex);
    };"""
content = content.replace(stock_card_mini_hover_old, stock_card_mini_hover_new)

stock_card_mini_ticks_old = """                {timeTickIndexes.map((index) => {
                    const x = xFor(index);
                    return (
                        <text
                            key={index}
                            x={x}
                            y={height - 6}
                            fill="rgba(255,255,255,0.24)"
                            fontSize="10"
                            fontFamily="monospace"
                            textAnchor={index === 0 ? 'start' : index === validPoints.length - 1 ? 'end' : 'middle'}
                        >
                            {formatStockCardAxisDate(validPoints[index].time, selectedRange)}
                        </text>
                    );
                })}"""

stock_card_mini_ticks_new = """                {useTimeScale ? (
                    <>
                        <text x={padding.left} y={height - 6} fill="rgba(255,255,255,0.24)" fontSize="10" fontFamily="monospace" textAnchor="start">10:00</text>
                        <text x={padding.left + plotWidth / 2} y={height - 6} fill="rgba(255,255,255,0.24)" fontSize="10" fontFamily="monospace" textAnchor="middle">14:00</text>
                        <text x={width - padding.right} y={height - 6} fill="rgba(255,255,255,0.24)" fontSize="10" fontFamily="monospace" textAnchor="end">18:00</text>
                    </>
                ) : (
                    timeTickIndexes.map((index) => {
                        const x = xFor(index);
                        return (
                            <text
                                key={index}
                                x={x}
                                y={height - 6}
                                fill="rgba(255,255,255,0.24)"
                                fontSize="10"
                                fontFamily="monospace"
                                textAnchor={index === 0 ? 'start' : index === validPoints.length - 1 ? 'end' : 'middle'}
                            >
                                {formatStockCardAxisDate(validPoints[index].time, selectedRange)}
                            </text>
                        );
                    })
                )}"""
content = content.replace(stock_card_mini_ticks_old, stock_card_mini_ticks_new)

with open('frontend/src/pages/MarketsView.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Replacement Complete")
