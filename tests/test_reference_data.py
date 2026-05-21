from __future__ import annotations

import json

from app.reference_data import get_instrument, sync_reference_data_from_caches, upsert_instrument


def test_reference_data_keeps_higher_priority_static_fields(tmp_path) -> None:
    upsert_instrument(
        tmp_path,
        kind="stock",
        symbol="TRHOL",
        name="Manual TRHOL",
        logo_url="https://example.test/trhol.svg",
        logo_source="manual",
        source="manual",
    )
    upsert_instrument(
        tmp_path,
        kind="stock",
        symbol="TRHOL",
        name="KAP TRHOL",
        logo_url="https://example.test/kap.svg",
        logo_source="kap",
        source="kap",
    )

    row = get_instrument(tmp_path, "stock", "TRHOL")

    assert row is not None
    assert row["name"] == "Manual TRHOL"
    assert row["logo_url"] == "https://example.test/trhol.svg"


def test_reference_data_resolves_alias(tmp_path) -> None:
    upsert_instrument(
        tmp_path,
        kind="stock",
        symbol="BIMAS",
        name="BİM BİRLEŞİK MAĞAZALAR A.Ş.",
        source="kap",
        aliases=["BIM"],
    )

    row = get_instrument(tmp_path, "stock", "BIM")

    assert row is not None
    assert row["symbol"] == "BIMAS"


def test_reference_data_bootstraps_existing_json_caches(tmp_path) -> None:
    funds_dir = tmp_path / "funds_cache"
    funds_dir.mkdir()
    (funds_dir / "funds_latest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "fund_code": "TLY",
                        "name": "TERA PORTFÖY BİRİNCİ SERBEST FON",
                        "founder_company": "TERA PORTFÖY",
                        "as_of": "2026-05-20",
                        "source": "tefasfon_funds",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    kap_dir = tmp_path / "kap_cache"
    kap_dir.mkdir()
    (kap_dir / "DSTKF.json").write_text(
        json.dumps(
            {
                "company": "DSTKF",
                "stock_code": "DSTKF",
                "company_title": "DESTEK FİNANS FAKTORİNG A.Ş.",
                "member_oid": "member-dstkf",
                "fetched_at": "2026-05-20T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    result = sync_reference_data_from_caches(tmp_path)

    assert result["record_count"] == 2
    assert get_instrument(tmp_path, "fund", "TLY")["name"] == "TERA PORTFÖY BİRİNCİ SERBEST FON"
    assert get_instrument(tmp_path, "stock", "DSTKF")["name"] == "DESTEK FİNANS FAKTORİNG A.Ş."
