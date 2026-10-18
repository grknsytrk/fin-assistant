import { useEffect, useState } from 'react';
import './FundsListLoader.css';

const LOADING_MESSAGES = [
    'Fon listesi hazırlanıyor…',
    'Yatırım fonları getiriliyor…',
    'Fon kayıtları yükleniyor…',
    'Tablo görünümü hazırlanıyor…',
    'Liste güncelleniyor…',
];

export default function FundsListLoader() {
    const [messageIndex, setMessageIndex] = useState(0);

    useEffect(() => {
        if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return undefined;

        const intervalId = window.setInterval(() => {
            setMessageIndex((current) => (current + 1) % LOADING_MESSAGES.length);
        }, 2600);

        return () => window.clearInterval(intervalId);
    }, []);

    return (
        <div className="funds-list-loader" role="status" aria-live="polite" aria-label="Fon listesi yükleniyor">
            <svg
                className="funds-list-loader-art"
                viewBox="0 0 300 124"
                aria-hidden="true"
                focusable="false"
            >
                <defs>
                    <linearGradient id="funds-list-loader-sheen" x1="0" x2="1" y1="0" y2="0">
                        <stop offset="0" stopColor="#3979d8" stopOpacity="0" />
                        <stop offset="0.5" stopColor="#75a4e8" stopOpacity="0.16" />
                        <stop offset="1" stopColor="#3979d8" stopOpacity="0" />
                    </linearGradient>
                    <linearGradient id="funds-list-loader-progress-fill" x1="0" x2="1" y1="0" y2="0">
                        <stop offset="0" stopColor="#3676cf" />
                        <stop offset="0.55" stopColor="#84a8dd" />
                        <stop offset="1" stopColor="#4b83d2" />
                    </linearGradient>
                    <clipPath id="funds-list-loader-clip">
                        <rect x="17" y="24" width="266" height="70" rx="4" />
                    </clipPath>
                    <clipPath id="funds-list-loader-progress-clip">
                        <rect x="65" y="115" width="170" height="6" rx="3" />
                    </clipPath>
                </defs>

                <rect className="funds-list-loader-frame" x="8" y="4" width="284" height="104" rx="8" />
                <path className="funds-list-loader-divider" d="M18 24.5h264" />
                <path className="funds-list-loader-heading" d="M22 16h22m23 0h50m87 0h28m11 0h22" />

                <g className="funds-list-loader-row row-one">
                    <rect className="funds-list-loader-row-bg" x="17" y="27" width="266" height="20" rx="4" />
                    <rect className="funds-list-loader-logo" x="22" y="30" width="15" height="14" rx="3" />
                    <path className="funds-list-loader-mark" d="M26 40v-5l3 2 3-4v7" />
                    <path className="funds-list-loader-code" d="M44 34h20" />
                    <path className="funds-list-loader-copy" d="M73 33h70m-70 6h39" />
                    <rect className="funds-list-loader-tag" x="184" y="31" width="29" height="12" rx="4" />
                    <path className="funds-list-loader-value" d="M222 34h35m10 0h12" />
                </g>

                <g className="funds-list-loader-row row-two">
                    <rect className="funds-list-loader-row-bg" x="17" y="51" width="266" height="20" rx="4" />
                    <rect className="funds-list-loader-logo" x="22" y="54" width="15" height="14" rx="3" />
                    <path className="funds-list-loader-mark" d="M26 64v-5l3 2 3-4v7" />
                    <path className="funds-list-loader-code" d="M44 58h24" />
                    <path className="funds-list-loader-copy" d="M77 57h86m-86 6h49" />
                    <rect className="funds-list-loader-tag" x="184" y="55" width="29" height="12" rx="4" />
                    <path className="funds-list-loader-value" d="M222 58h28m17 0h14" />
                </g>

                <g className="funds-list-loader-row row-three">
                    <rect className="funds-list-loader-row-bg" x="17" y="75" width="266" height="20" rx="4" />
                    <rect className="funds-list-loader-logo" x="22" y="78" width="15" height="14" rx="3" />
                    <path className="funds-list-loader-mark" d="M26 88v-5l3 2 3-4v7" />
                    <path className="funds-list-loader-code" d="M44 82h18" />
                    <path className="funds-list-loader-copy" d="M71 81h76m-76 6h54" />
                    <rect className="funds-list-loader-tag" x="184" y="79" width="29" height="12" rx="4" />
                    <path className="funds-list-loader-value" d="M222 82h34m10 0h12" />
                </g>

                <g clipPath="url(#funds-list-loader-clip)" aria-hidden="true">
                    <rect className="funds-list-loader-sheen row-one-sheen" x="17" y="27" width="92" height="20" fill="url(#funds-list-loader-sheen)" />
                    <rect className="funds-list-loader-sheen row-two-sheen" x="17" y="51" width="92" height="20" fill="url(#funds-list-loader-sheen)" />
                    <rect className="funds-list-loader-sheen row-three-sheen" x="17" y="75" width="92" height="20" fill="url(#funds-list-loader-sheen)" />
                </g>

                <rect className="funds-list-loader-progress-track" x="65" y="115" width="170" height="6" rx="3" />
                <g clipPath="url(#funds-list-loader-progress-clip)">
                    <rect className="funds-list-loader-progress" x="65" y="115" width="170" height="6" rx="3" fill="url(#funds-list-loader-progress-fill)" />
                </g>
            </svg>
            <span key={messageIndex} className="funds-list-loader-text" aria-hidden="true">
                {LOADING_MESSAGES[messageIndex]}
            </span>
        </div>
    );
}
