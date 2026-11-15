import './FundsListLoader.css';

const SKELETON_ROWS = [
    { nameWidth: '72%', founderWidth: '64%' },
    { nameWidth: '84%', founderWidth: '58%' },
    { nameWidth: '66%', founderWidth: '76%' },
    { nameWidth: '78%', founderWidth: '60%' },
    { nameWidth: '88%', founderWidth: '68%' },
    { nameWidth: '62%', founderWidth: '54%' },
    { nameWidth: '80%', founderWidth: '72%' },
    { nameWidth: '70%', founderWidth: '62%' },
    { nameWidth: '76%', founderWidth: '66%' },
];

export default function FundsListLoader() {
    return (
        <div
            className="funds-list-loader"
            role="status"
            aria-live="polite"
            aria-label="Fon listesi yükleniyor"
        >
            <div className="funds-list-skeleton-table" aria-hidden="true">
                <div className="funds-list-skeleton-header">
                    <span className="funds-list-sk-head funds-list-sk-col-rank" />
                    <span className="funds-list-sk-head funds-list-sk-col-fund" />
                    <span className="funds-list-sk-head funds-list-sk-col-type" />
                    <span className="funds-list-sk-head funds-list-sk-col-founder" />
                    <span className="funds-list-sk-head funds-list-sk-col-num" />
                    <span className="funds-list-sk-head funds-list-sk-col-num" />
                    <span className="funds-list-sk-head funds-list-sk-col-risk" />
                    <span className="funds-list-sk-head funds-list-sk-col-wide" />
                    <span className="funds-list-sk-head funds-list-sk-col-num" />
                </div>

                <div className="funds-list-skeleton-body">
                    {SKELETON_ROWS.map((row, index) => (
                        <div className="funds-list-skeleton-row" key={index}>
                            <div className="funds-list-sk-cell funds-list-sk-col-rank">
                                <span className="funds-list-sk-block funds-list-sk-rank" />
                            </div>
                            <div className="funds-list-sk-cell funds-list-sk-col-fund">
                                <span className="funds-list-sk-block funds-list-sk-code" />
                                <span
                                    className="funds-list-sk-block funds-list-sk-name"
                                    style={{ width: row.nameWidth }}
                                />
                            </div>
                            <div className="funds-list-sk-cell funds-list-sk-col-type">
                                <span className="funds-list-sk-block funds-list-sk-tag" />
                            </div>
                            <div className="funds-list-sk-cell funds-list-sk-col-founder">
                                <span
                                    className="funds-list-sk-block funds-list-sk-founder"
                                    style={{ width: row.founderWidth }}
                                />
                            </div>
                            <div className="funds-list-sk-cell funds-list-sk-col-num">
                                <span className="funds-list-sk-block funds-list-sk-metric" />
                            </div>
                            <div className="funds-list-sk-cell funds-list-sk-col-num">
                                <span className="funds-list-sk-block funds-list-sk-metric-sm" />
                            </div>
                            <div className="funds-list-sk-cell funds-list-sk-col-risk">
                                <span className="funds-list-sk-block funds-list-sk-risk" />
                            </div>
                            <div className="funds-list-sk-cell funds-list-sk-col-wide">
                                <span className="funds-list-sk-block funds-list-sk-aum" />
                            </div>
                            <div className="funds-list-sk-cell funds-list-sk-col-num">
                                <span className="funds-list-sk-block funds-list-sk-metric" />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
