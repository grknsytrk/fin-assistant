"""Modular domain submodules extracted from ``app.fund_service``."""

from app.funds.holdings_kap_parser import (
    get_fund_holdings_payload,
    refresh_fund_holdings,
)
from app.funds.providers import (
    FintablesClient,
    TefasClient,
    TefasFonClient,
    fetch_fintables_udf_history,
    fetch_fintables_yield_summary,
)

__all__ = [
    "FintablesClient",
    "TefasClient",
    "TefasFonClient",
    "fetch_fintables_udf_history",
    "fetch_fintables_yield_summary",
    "get_fund_holdings_payload",
    "refresh_fund_holdings",
]
