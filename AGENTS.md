# RAG-FIN Agent Notes

## User Preferences

- Keep frontend changes scoped and practical. Do not run broad or repetitive test suites unless the change clearly needs them.
- Prefer targeted checks such as `npm run build` for frontend-only work, or a single focused pytest file/test for backend logic.
- Fund distribution history should live inside the fund detail `Geçmiş Veriler` area, not in a modal. That area should have two tabs:
  - `Fiyat / Yatırımcı`: historical price, return, portfolio size, investor count.
  - `Fon Dağılımı`: historical asset allocation / portfolio breakdown.

## TEFAS / tefasfon v1.1.0

Use `tefasfon` as the primary TEFAS adapter for fund data. It does not require Chrome/browser automation.

Install:

```bash
pip install tefasfon
```

Supported fund types:

- `SEC`: Securities
- `PEN`: Pension
- `ETF`: Exchange-Traded Fund
- `RE`: Real Estate
- `VC`: Venture Capital

### Python API

`get_funds(...)` fetches general fund information by date range:

```python
from tefasfon import get_funds

df = get_funds(
    fund_type="SEC",
    start_date="01.04.2026",
    end_date="30.04.2026",
    fund_codes=["AEK", "AEY"],              # optional
    fund_title_contains=["ALTIN"],          # optional
    save_to_excel=False,
)
```

Important columns include `fonKodu`, `fonUnvan`, `tarih`, `fiyat`, `tedPaySayisi`, `kisiSayisi`, `portfoyBuyukluk`, `borsaBultenFiyat`.

`get_portfolio(...)` fetches portfolio breakdown / asset allocation by date range:

```python
from tefasfon import get_portfolio

df = get_portfolio(
    fund_type="PEN",
    start_date="01.04.2026",
    end_date="30.04.2026",
    fund_codes=["AEK"],
)
```

Important columns include `fonKodu`, `fonUnvan`, `tarih`, plus allocation keys such as `hs`, `yhs`, `dt`, `hb`, `vdm`, `vmtl`, `tr`, `byf`, `yyf`, `km`, `kkstl`, `osks`. Other non-null allocation columns may appear and should be preserved.

`get_returns(...)` fetches return data:

```python
from tefasfon import get_returns

df = get_returns(fund_type="SEC", basis="RB")
df = get_returns(
    fund_type="PEN",
    basis="RB",
    start_date="01.04.2026",
    end_date="30.04.2026",
)
df = get_returns(
    fund_type="RE",
    basis="SB",
    start_date="01.04.2026",
    end_date="30.04.2026",
)
```

For returns, `basis` is required:

- `RB`: return-based. `start_date` and `end_date` are optional for standard periods.
- `SB`: size-based. `start_date` and `end_date` are required.
- `MB`: management fee-based. `start_date` and `end_date` are required.

`analyze_funds(df, ...)` calculates performance metrics from `get_funds()` output:

```python
from tefasfon import analyze_funds

metrics = analyze_funds(
    df,
    freq="B",
    method="log",
    risk_free_annual=0.40,
)
```

### CLI

The package installs a `tefasfon` command.

```bash
tefasfon funds -t SEC -s 01.04.2026 -e 30.04.2026 -x
tefasfon funds -t PEN -s 01.04.2026 -e 30.04.2026 -c AEK AEY -o csv
tefasfon funds -t ETF -s 01.04.2026 -e 30.04.2026 -f ALTIN -o json
tefasfon returns -t SEC -b RB -x
tefasfon returns -t PEN -b RB -s 01.04.2026 -e 30.04.2026 -o csv
tefasfon returns -t RE -b SB -s 01.04.2026 -e 30.04.2026 -x
tefasfon funds -t SEC -s 01.04.2026 -e 30.04.2026 -o csv > funds.csv
```

Common CLI options:

- `-t, --fund-type`: `SEC`, `PEN`, `ETF`, `RE`, `VC`
- `-s, --start-date`: `DD.MM.YYYY`
- `-e, --end-date`: `DD.MM.YYYY`
- `-c, --fund-codes`: exact fund code filters
- `-f, --fund-title-contains`: title keyword filters
- `-x, --save-to-excel`: save `.xlsx`
- `-o, --output`: `csv` or `json`

### Implementation Notes

- Prefer `get_portfolio()` range calls for allocation history. Per-day portfolio calls are a slow fallback and should not be the first strategy.
- A slow first `Geçmişi gör` load usually means the cache is cold and the backend is fetching TEFAS allocation history. If the range query returns empty, the current fallback may fan out into one request per business day.
- Preserve TEFAS fields that are not yet mapped; allocation columns can expand over time.
