import re
with open('frontend/src/pages/MarketsView.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

index_dot_old = """            {/* Uç Noktası (Kapanış) Dot */}
            <circle cx={xFor(validPoints.length - 1)} cy={yFor(last)} r="4" fill={chartColor} />"""
index_dot_new = """            {/* Uç Noktası (Kapanış) Dot ve Pulse */}
            <circle cx={xFor(validPoints.length - 1)} cy={yFor(last)} r="4" fill={chartColor} opacity="0.6">
                <animate attributeName="r" values="4; 14; 14" keyTimes="0; 0.5; 1" dur="2s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.6; 0; 0" keyTimes="0; 0.5; 1" dur="2s" repeatCount="indefinite" />
            </circle>
            <circle cx={xFor(validPoints.length - 1)} cy={yFor(last)} r="4" fill={chartColor} />"""
content = content.replace(index_dot_old, index_dot_new)

stock_card_old = """                <path d={areaData} fill={`url(#${gradientId})`} />
                <path d={pathData} fill="none" stroke={color} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
                {hoverPoint && hoverX != null && hoverY != null && ("""
stock_card_new = """                <path d={areaData} fill={`url(#${gradientId})`} />
                <path d={pathData} fill="none" stroke={color} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
                
                {selectedRange === '1d' && (
                    <>
                        <circle cx={xFor(validPoints.length - 1)} cy={yFor(validPoints[validPoints.length - 1].close)} r="4" fill={color} opacity="0.6">
                            <animate attributeName="r" values="4; 14; 14" keyTimes="0; 0.5; 1" dur="2s" repeatCount="indefinite" />
                            <animate attributeName="opacity" values="0.6; 0; 0" keyTimes="0; 0.5; 1" dur="2s" repeatCount="indefinite" />
                        </circle>
                        <circle cx={xFor(validPoints.length - 1)} cy={yFor(validPoints[validPoints.length - 1].close)} r="4" fill={color} />
                    </>
                )}

                {hoverPoint && hoverX != null && hoverY != null && ("""
content = content.replace(stock_card_old, stock_card_new)

with open('frontend/src/pages/MarketsView.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Pulse patch applied")
