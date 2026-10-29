export function OverviewAiLoadingIllustration() {
    return (
        <div className="overview-ai-loading-visual" aria-live="polite">
            <svg
                className="overview-ai-loading-svg overview-ai-loading-svg-desktop"
                viewBox="0 0 1200 176"
                role="img"
                aria-labelledby="overview-ai-loading-title overview-ai-loading-description"
                xmlns="http://www.w3.org/2000/svg"
            >
                <title id="overview-ai-loading-title">Yapay zeka finansal verileri inceliyor</title>
                <desc id="overview-ai-loading-description">
                    Son açıklanan çeyrek, yıllık kıyas, nakit akışı ve bilanço verileri animasyonlu grafiklerle analiz ediliyor.
                </desc>
                <defs>
                    <linearGradient id="overview-ai-desktop-surface" x1="0" y1="0" x2="1" y2="1">
                        <stop stopColor="#1d2a3d" />
                        <stop offset=".55" stopColor="#202a38" />
                        <stop offset="1" stopColor="#1c2532" />
                    </linearGradient>
                    <radialGradient id="overview-ai-desktop-glow" cx="0" cy="0" r="1" gradientTransform="translate(45 30) rotate(47) scale(310 180)" gradientUnits="userSpaceOnUse">
                        <stop stopColor="#4389fa" stopOpacity=".22" />
                        <stop offset="1" stopColor="#4389fa" stopOpacity="0" />
                    </radialGradient>
                    <linearGradient id="overview-ai-desktop-accent" x1="28" y1="127" x2="106" y2="50" gradientUnits="userSpaceOnUse">
                        <stop stopColor="#548df6" />
                        <stop offset="1" stopColor="#73e2cb" />
                    </linearGradient>
                    <linearGradient id="overview-ai-desktop-chart" x1="714" y1="134" x2="1148" y2="48" gradientUnits="userSpaceOnUse">
                        <stop stopColor="#6396ff" />
                        <stop offset="1" stopColor="#78e1ca" />
                    </linearGradient>
                    <linearGradient id="overview-ai-desktop-scan" x1="0" y1="0" x2="1" y2="0">
                        <stop stopColor="#78e1ca" stopOpacity="0" />
                        <stop offset=".5" stopColor="#78e1ca" stopOpacity=".72" />
                        <stop offset="1" stopColor="#78e1ca" stopOpacity="0" />
                    </linearGradient>
                    <clipPath id="overview-ai-desktop-chart-clip">
                        <rect x="682" y="35" width="484" height="112" rx="13" />
                    </clipPath>
                </defs>

                <rect x="1" y="1" width="1198" height="174" rx="18" fill="url(#overview-ai-desktop-surface)" stroke="#43618a" strokeOpacity=".58" />
                <rect x="1" y="1" width="1198" height="174" rx="18" fill="url(#overview-ai-desktop-glow)" />
                <path d="M615 20v136" stroke="#8294ad" strokeOpacity=".16" />

                <g transform="translate(35 44)">
                    <circle cx="40" cy="40" r="36" fill="#203752" stroke="#5284d8" strokeOpacity=".48" />
                    <circle cx="40" cy="40" r="29" fill="none" stroke="#79a9f7" strokeOpacity=".28" strokeDasharray="2 5" />
                    <circle className="overview-ai-svg-motion" cx="40" cy="40" r="24" fill="none" stroke="#8ce8d2" strokeOpacity=".56" strokeDasharray="44 108" strokeLinecap="round">
                        <animateTransform attributeName="transform" type="rotate" from="0 40 40" to="360 40 40" dur="9s" repeatCount="indefinite" />
                    </circle>
                    <path d="M28 43c0-7 5-12 12-12s12 5 12 12c0 4-2 7-5 9v5H33v-5c-3-2-5-5-5-9Z" fill="#293e58" stroke="url(#overview-ai-desktop-accent)" strokeWidth="1.8" />
                    <path d="M35 45c1.8-2 3.5-3 5-3s3.2 1 5 3M35 49h10" fill="none" stroke="#c3f9ec" strokeLinecap="round" strokeWidth="1.8" />
                    <circle className="overview-ai-svg-motion" cx="20" cy="24" r="2.5" fill="#8ee8d3">
                        <animate attributeName="opacity" values=".3;1;.3" dur="1.9s" repeatCount="indefinite" />
                    </circle>
                    <circle className="overview-ai-svg-motion" cx="61" cy="57" r="2" fill="#83adff">
                        <animate attributeName="opacity" values="1;.25;1" dur="2.4s" repeatCount="indefinite" />
                    </circle>
                </g>

                <text x="128" y="66" fill="#e6edf8" fontFamily="Inter, Segoe UI, sans-serif" fontSize="20" fontWeight="700">
                    AI finansal verileri birlikte inceliyor
                </text>
                <text x="128" y="92" fill="#aab7c9" fontFamily="Inter, Segoe UI, sans-serif" fontSize="12.5">
                    Son çeyrek; geçen yıl, nakit akışı ve bilanço sinyalleriyle karşılaştırılıyor.
                </text>
                <g fontFamily="Inter, Segoe UI, sans-serif" fontSize="9" fontWeight="650" letterSpacing=".5">
                    <circle cx="132" cy="121" r="3" fill="#79a9ff" />
                    <text x="141" y="124" fill="#c0cce0">SON ÇEYREK</text>
                    <circle cx="235" cy="121" r="3" fill="#7be1cb" />
                    <text x="244" y="124" fill="#c0cce0">YILLIK KIYAS</text>
                    <circle cx="347" cy="121" r="3" fill="#a1b5ff" />
                    <text x="356" y="124" fill="#c0cce0">NAKİT + BİLANÇO</text>
                </g>
                <text x="128" y="149" fill="#8498b3" fontFamily="Inter, Segoe UI, sans-serif" fontSize="10.5">
                    IFM yalnızca ekranda görünen finansal verilerden yorum üretiyor.
                </text>

                <rect x="670" y="24" width="510" height="128" rx="15" fill="#111a27" fillOpacity=".52" stroke="#6680a3" strokeOpacity=".3" />
                <g fontFamily="Inter, Segoe UI, sans-serif">
                    <text x="690" y="44" fill="#91a4be" fontSize="8" fontWeight="700" letterSpacing=".8">SON AÇIKLANAN</text>
                    <text x="690" y="61" fill="#e4edf9" fontSize="12" fontWeight="700">Çeyrek akışı</text>
                    <path d="M690 123h130" stroke="#667a96" strokeOpacity=".32" />
                    <rect x="699" y="101" width="13" height="22" rx="3" fill="#4c78c5" fillOpacity=".74" />
                    <rect x="720" y="91" width="13" height="32" rx="3" fill="#5c8de5" fillOpacity=".82" />
                    <rect x="741" y="82" width="13" height="41" rx="3" fill="#6ca4f8" fillOpacity=".9" />
                    <rect x="762" y="73" width="13" height="50" rx="3" fill="#7bdec9" />
                    <path d="M693 90 724 79l21 4 35-22" fill="none" stroke="url(#overview-ai-desktop-chart)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
                    <circle cx="780" cy="61" r="3" fill="#b6f7e7" />

                    <path d="M839 38v98" stroke="#71839b" strokeOpacity=".22" />
                    <text x="855" y="44" fill="#91a4be" fontSize="8" fontWeight="700" letterSpacing=".8">GEÇMİŞ DÖNEMLER</text>
                    <text x="855" y="61" fill="#e4edf9" fontSize="12" fontWeight="700">Yıllık kıyas</text>
                    <path d="M856 123h139M856 104h139M856 85h139" stroke="#71839b" strokeOpacity=".13" />
                    <path d="M861 114c17-2 18-21 34-17s17 13 31 3 18-4 29-16 22 2 36-11" fill="none" stroke="#668bc9" strokeOpacity=".5" strokeWidth="2" />
                    <path className="overview-ai-svg-motion" d="M861 110c17-4 18-12 34-11s17-18 31-10 18-9 29-7 22-17 36-24" fill="none" stroke="url(#overview-ai-desktop-chart)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" strokeDasharray="180" strokeDashoffset="180">
                        <animate attributeName="stroke-dashoffset" from="180" to="0" dur="2.8s" repeatCount="indefinite" />
                    </path>
                    <circle className="overview-ai-svg-motion" cx="991" cy="58" r="3.5" fill="#8ce8d2">
                        <animate attributeName="r" values="2.5;4;2.5" dur="1.6s" repeatCount="indefinite" />
                    </circle>

                    <path d="M1011 38v98" stroke="#71839b" strokeOpacity=".22" />
                    <text x="1027" y="44" fill="#91a4be" fontSize="8" fontWeight="700" letterSpacing=".8">KALİTE KONTROLÜ</text>
                    <text x="1027" y="61" fill="#e4edf9" fontSize="12" fontWeight="700">Nakit &amp; bilanço</text>
                    <text x="1027" y="83" fill="#9eacc0" fontSize="8.5">Kârın nakde dönüşümü</text>
                    <rect x="1027" y="91" width="129" height="5" rx="2.5" fill="#364459" />
                    <rect x="1027" y="91" width="88" height="5" rx="2.5" fill="#70cdbd" fillOpacity=".9" />
                    <text x="1027" y="113" fill="#9eacc0" fontSize="8.5">Borç ve özkaynak dengesi</text>
                    <rect x="1027" y="121" width="129" height="5" rx="2.5" fill="#364459" />
                    <rect x="1027" y="121" width="103" height="5" rx="2.5" fill="#719cf0" fillOpacity=".9" />
                </g>

                <g className="overview-ai-svg-motion" clipPath="url(#overview-ai-desktop-chart-clip)">
                    <rect x="678" y="36" width="22" height="110" fill="url(#overview-ai-desktop-scan)" opacity=".46">
                        <animate attributeName="x" values="678;1150;678" dur="6.5s" repeatCount="indefinite" />
                    </rect>
                </g>
            </svg>

            <svg
                className="overview-ai-loading-svg overview-ai-loading-svg-mobile"
                viewBox="0 0 360 284"
                role="img"
                aria-labelledby="overview-ai-loading-mobile-title overview-ai-loading-mobile-description"
                xmlns="http://www.w3.org/2000/svg"
            >
                <title id="overview-ai-loading-mobile-title">Yapay zeka finansal verileri inceliyor</title>
                <desc id="overview-ai-loading-mobile-description">
                    Son çeyrek, yıllık kıyas, nakit akışı ve bilanço verileri animasyonlu grafiklerle analiz ediliyor.
                </desc>
                <defs>
                    <linearGradient id="overview-ai-mobile-surface" x1="0" y1="0" x2="1" y2="1">
                        <stop stopColor="#1d2a3d" />
                        <stop offset="1" stopColor="#1c2532" />
                    </linearGradient>
                    <radialGradient id="overview-ai-mobile-glow" cx="0" cy="0" r="1" gradientTransform="translate(15 12) rotate(48) scale(220 180)" gradientUnits="userSpaceOnUse">
                        <stop stopColor="#4389fa" stopOpacity=".24" />
                        <stop offset="1" stopColor="#4389fa" stopOpacity="0" />
                    </radialGradient>
                    <linearGradient id="overview-ai-mobile-chart" x1="37" y1="223" x2="208" y2="163" gradientUnits="userSpaceOnUse">
                        <stop stopColor="#6396ff" />
                        <stop offset="1" stopColor="#78e1ca" />
                    </linearGradient>
                    <linearGradient id="overview-ai-mobile-scan" x1="0" y1="0" x2="1" y2="0">
                        <stop stopColor="#78e1ca" stopOpacity="0" />
                        <stop offset=".5" stopColor="#78e1ca" stopOpacity=".68" />
                        <stop offset="1" stopColor="#78e1ca" stopOpacity="0" />
                    </linearGradient>
                    <clipPath id="overview-ai-mobile-clip">
                        <rect x="16" y="113" width="328" height="153" rx="13" />
                    </clipPath>
                </defs>

                <rect x="1" y="1" width="358" height="282" rx="18" fill="url(#overview-ai-mobile-surface)" stroke="#43618a" strokeOpacity=".58" />
                <rect x="1" y="1" width="358" height="282" rx="18" fill="url(#overview-ai-mobile-glow)" />
                <g transform="translate(18 16)">
                    <circle cx="25" cy="25" r="23" fill="#203752" stroke="#5284d8" strokeOpacity=".48" />
                    <circle className="overview-ai-svg-motion" cx="25" cy="25" r="17" fill="none" stroke="#8ce8d2" strokeOpacity=".62" strokeDasharray="30 78" strokeLinecap="round">
                        <animateTransform attributeName="transform" type="rotate" from="0 25 25" to="360 25 25" dur="8s" repeatCount="indefinite" />
                    </circle>
                    <path d="M17 25c0-5 3-8 8-8s8 3 8 8c0 3-1 5-3 6v4H20v-4c-2-1-3-3-3-6Z" fill="#293e58" stroke="#70cdbd" strokeWidth="1.4" />
                    <path d="M22 26c1-1 2-2 3-2s2 1 3 2M22 29h6" fill="none" stroke="#c3f9ec" strokeLinecap="round" strokeWidth="1.3" />
                    <circle className="overview-ai-svg-motion" cx="9" cy="10" r="2" fill="#8ee8d3">
                        <animate attributeName="opacity" values=".25;1;.25" dur="2s" repeatCount="indefinite" />
                    </circle>
                </g>
                <text x="78" y="36" fill="#e6edf8" fontFamily="Inter, Segoe UI, sans-serif" fontSize="15" fontWeight="700">
                    AI finansal analizi
                </text>
                <text x="78" y="56" fill="#aab7c9" fontFamily="Inter, Segoe UI, sans-serif" fontSize="10.5">
                    Son çeyrek, yıllık değişim,
                </text>
                <text x="78" y="71" fill="#aab7c9" fontFamily="Inter, Segoe UI, sans-serif" fontSize="10.5">
                    nakit akışı ve bilanço birlikte.
                </text>

                <rect x="16" y="113" width="328" height="153" rx="13" fill="#111a27" fillOpacity=".54" stroke="#6680a3" strokeOpacity=".3" />
                <g fontFamily="Inter, Segoe UI, sans-serif">
                    <text x="31" y="136" fill="#91a4be" fontSize="8" fontWeight="700" letterSpacing=".5">SON ÇEYREK</text>
                    <text x="31" y="152" fill="#e4edf9" fontSize="10" fontWeight="650">Gelir ve kâr</text>
                    <text x="134" y="136" fill="#91a4be" fontSize="8" fontWeight="700" letterSpacing=".5">YILLIK KIYAS</text>
                    <text x="134" y="152" fill="#e4edf9" fontSize="10" fontWeight="650">Aynı dönem</text>
                    <text x="250" y="136" fill="#91a4be" fontSize="8" fontWeight="700" letterSpacing=".5">BİLANÇO</text>
                    <text x="250" y="152" fill="#e4edf9" fontSize="10" fontWeight="650">Nakit ve borç</text>
                    <path d="M122 125v115M237 125v115" stroke="#71839b" strokeOpacity=".2" />
                    <path d="M31 228h79" stroke="#667a96" strokeOpacity=".32" />
                    <rect x="38" y="207" width="10" height="21" rx="2.5" fill="#4c78c5" fillOpacity=".78" />
                    <rect x="54" y="195" width="10" height="33" rx="2.5" fill="#5c8de5" fillOpacity=".84" />
                    <rect x="70" y="187" width="10" height="41" rx="2.5" fill="#6ca4f8" fillOpacity=".9" />
                    <rect x="86" y="176" width="10" height="52" rx="2.5" fill="#7bdec9" />
                    <path d="M136 222h89M136 203h89M136 184h89" stroke="#71839b" strokeOpacity=".14" />
                    <path d="M140 216c12-3 13-18 24-13s14 11 22 1 13-4 20-13 12 1 18-7" fill="none" stroke="#668bc9" strokeOpacity=".46" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
                    <path className="overview-ai-svg-motion" d="M140 216c12-3 13-18 24-13s14 11 22 1 13-4 20-13 12 1 18-7" fill="none" stroke="url(#overview-ai-mobile-chart)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.4" strokeDasharray="150" strokeDashoffset="150">
                        <animate attributeName="stroke-dashoffset" from="150" to="0" dur="2.6s" repeatCount="indefinite" />
                    </path>
                    <path d="M252 181h75M252 202h62M252 223h82" stroke="#70cdbd" strokeOpacity=".84" strokeLinecap="round" strokeWidth="4" />
                    <path d="M252 181h75M252 202h62M252 223h82" stroke="#71839b" strokeOpacity=".2" strokeLinecap="round" strokeWidth="4" />
                </g>
                <g className="overview-ai-svg-motion" clipPath="url(#overview-ai-mobile-clip)">
                    <rect x="16" y="115" width="17" height="149" fill="url(#overview-ai-mobile-scan)" opacity=".42">
                        <animate attributeName="x" values="16;340;16" dur="5.8s" repeatCount="indefinite" />
                    </rect>
                </g>
                <text x="18" y="276" fill="#8498b3" fontFamily="Inter, Segoe UI, sans-serif" fontSize="8.5" letterSpacing=".3">
                    IFM · Son açıklanan dönemden otomatik yorum
                </text>
            </svg>
        </div>
    );
}
