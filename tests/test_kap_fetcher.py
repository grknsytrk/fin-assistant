import io
import json
import urllib.error
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src import kap_fetcher
from src.kap_fetcher import (
    KAP_BROWSER_USER_AGENT,
    KAP_CACHE_SCHEMA_VERSION,
    _http_get_json,
    _list_company_disclosures,
    _pick_metric_value,
    fetch_kap_company_snapshot,
)


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _cfg(user_agent: str = "ragfin-test/1.0") -> SimpleNamespace:
    return SimpleNamespace(user_agent=user_agent, timeout_seconds=1.0)


def test_list_member_disclosures_by_criteria_chunks_long_lookback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        kap_fetcher,
        "_utc_now",
        lambda: datetime(2026, 5, 17, tzinfo=timezone.utc),
    )

    def fake_http_post_json(_url: str, body: dict[str, object], _cfg_obj: object) -> list[dict[str, object]]:
        calls.append(body)
        return [{"disclosureIndex": str(len(calls)), "summary": "Brüt Yazılan Prim"}]

    monkeypatch.setattr(kap_fetcher, "_http_post_json", fake_http_post_json)

    rows = kap_fetcher._list_member_disclosures_by_criteria(
        member_oid="oid",
        cfg=_cfg(),
        lookback_days=760,
    )

    assert [call["fromDate"] for call in calls] == ["2025-05-17", "2024-05-16", "2024-04-15"]
    assert [call["toDate"] for call in calls] == ["2026-05-17", "2025-05-16", "2024-05-15"]
    assert [row["disclosureIndex"] for row in rows] == ["1", "2", "3"]


def test_list_company_disclosures_falls_back_to_bycriteria_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rate_limited(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(kap_fetcher, "_http_get_json", rate_limited)
    monkeypatch.setattr(
        kap_fetcher,
        "_list_member_disclosures_by_criteria",
        lambda **_kwargs: [
            {
                "disclosureIndex": 10,
                "subject": "Finansal Rapor",
                "year": 2026,
                "period": 1,
                "summary": "Konsolide olmayan",
            },
            {
                "disclosureIndex": 11,
                "subject": "Finansal Rapor",
                "year": 2026,
                "period": 1,
                "summary": "Konsolide",
            },
            {
                "disclosureIndex": 8,
                "subject": "Finansal Rapor",
                "year": 2025,
                "period": 4,
                "summary": "2025/4",
            },
            {
                "disclosureIndex": 99,
                "subject": "Faaliyet Raporu",
                "year": 2025,
                "period": 4,
                "summary": "ignored",
            },
        ],
    )

    result = kap_fetcher._list_company_disclosures(
        member_oid="oid",
        cfg=_cfg(),
        max_periods=2,
    )

    assert [(row["year"], row["period"], row["disclosure_index"]) for row in result] == [
        (2026, 1, 11),
        (2025, 4, 8),
    ]


def test_list_company_disclosures_supplements_incomplete_excel_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kap_fetcher.time, "sleep", lambda _seconds: None)

    def fake_http_get_json(url: str, _cfg_obj: object) -> list[dict[str, object]]:
        if "/2026/" in url:
            return [
                {
                    "mkkMemberOid": "oid",
                    "stockCode": "ANHYT",
                    "disclosureIndex": 30,
                    "period": 1,
                    "year": 2026,
                    "title": "2026/1",
                }
            ]
        return []

    monkeypatch.setattr(kap_fetcher, "_http_get_json", fake_http_get_json)
    monkeypatch.setattr(
        kap_fetcher,
        "_list_member_disclosures_by_criteria",
        lambda **_kwargs: [
            {"disclosureIndex": 30, "subject": "Finansal Rapor", "year": 2026, "period": 1},
            {"disclosureIndex": 20, "subject": "Finansal Rapor", "year": 2025, "period": 4},
            {"disclosureIndex": 19, "subject": "Finansal Rapor", "year": 2025, "period": 3},
        ],
    )

    result = kap_fetcher._list_company_disclosures(
        member_oid="oid",
        cfg=_cfg(),
        max_periods=3,
    )

    assert [(row["year"], row["period"], row["disclosure_index"]) for row in result] == [
        (2026, 1, 30),
        (2025, 4, 20),
        (2025, 3, 19),
    ]


def test_http_get_json_uses_browser_user_agent_before_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_user_agents: list[str] = []

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        del timeout
        headers = {key.lower(): value for key, value in request.header_items()}  # type: ignore[attr-defined]
        seen_user_agents.append(str(headers.get("user-agent", "")))
        return _FakeResponse([{"ok": True}])

    monkeypatch.setattr(kap_fetcher.urllib.request, "urlopen", fake_urlopen)

    payload = _http_get_json("https://example.test/kap", _cfg())

    assert payload == [{"ok": True}]
    assert seen_user_agents == [KAP_BROWSER_USER_AGENT]


def test_http_get_json_stops_after_429_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_user_agents: list[str] = []

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        del timeout
        headers = {key.lower(): value for key, value in request.header_items()}  # type: ignore[attr-defined]
        seen_user_agents.append(str(headers.get("user-agent", "")))
        raise urllib.error.HTTPError(
            "https://example.test/kap",
            429,
            "Request Limit Exceeded",
            hdrs=None,
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr(kap_fetcher.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(kap_fetcher.time, "sleep", lambda _seconds: None)

    with pytest.raises(urllib.error.HTTPError):
        _http_get_json("https://example.test/kap", _cfg())

    assert seen_user_agents == [KAP_BROWSER_USER_AGENT, KAP_BROWSER_USER_AGENT]


def test_list_company_disclosures_stops_after_requested_period_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_http_get_json(url: str, _cfg_obj: object) -> list[dict[str, object]]:
        calls.append(url)
        if "/2026/" in url:
            return [
                {
                    "mkkMemberOid": "oid",
                    "stockCode": "MGROS",
                    "pdOid": "pd-2026-1",
                    "disclosureIndex": 1601478,
                    "year": 2026,
                    "period": 1,
                    "title": "MIGROS TICARET A.S.",
                }
            ]
        if "/2025/" in url:
            return [
                {
                    "mkkMemberOid": "oid",
                    "stockCode": "MGROS",
                    "pdOid": "pd-2025-4",
                    "disclosureIndex": 1566129,
                    "year": 2025,
                    "period": 4,
                    "title": "MIGROS TICARET A.S.",
                }
            ]
        raise AssertionError(f"unexpected year fetch: {url}")

    monkeypatch.setattr(kap_fetcher, "_utc_now", lambda: kap_fetcher.datetime(2026, 5, 11, tzinfo=kap_fetcher.timezone.utc))
    monkeypatch.setattr(kap_fetcher, "_http_get_json", fake_http_get_json)
    monkeypatch.setattr(kap_fetcher.time, "sleep", lambda _seconds: None)

    rows = _list_company_disclosures("oid", _cfg(), max_periods=2)

    assert [row["period"] for row in rows] == [1, 4]
    assert len(calls) == 2


def test_fetch_snapshot_uses_ticker_alias_cache_when_live_list_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cache_dir = tmp_path / "kap_cache"
    cache_dir.mkdir()
    (cache_dir / "ENKAI.json").write_text(
        json.dumps(
            {
                "ok": True,
                "schema_version": KAP_CACHE_SCHEMA_VERSION,
                "company": "ENKAI",
                "stock_code": "ENKAI",
                "member_oid": "oid-enka",
                "fetched_at": "2026-05-08T18:12:50+00:00",
                "quarters": [{"quarter": "2026Q1", "year": 2026, "period": 1}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        kap_fetcher,
        "_resolve_member",
        lambda _company, _cfg_obj: {"mkk_member_oid": "oid-enka", "title": "ENKA INSAAT VE SANAYI A.S."},
    )
    monkeypatch.setattr(kap_fetcher, "_list_company_disclosures", lambda **_kwargs: [])

    payload = fetch_kap_company_snapshot(
        company="ENKA",
        cfg=SimpleNamespace(cache_ttl_hours=0.0, timeout_seconds=1.0, user_agent="ragfin-test/1.0"),
        processed_dir=tmp_path,
        force_refresh=False,
        max_quarters=20,
    )

    assert payload["ok"] is True
    assert payload["cache_hit"] is True
    assert payload["cache_stale"] is True
    assert payload["stock_code"] == "ENKAI"
    assert payload["quarters"][0]["quarter"] == "2026Q1"


def test_fetch_snapshot_can_serve_complete_cache_without_live_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cache_dir = tmp_path / "kap_cache"
    cache_dir.mkdir()
    quarters = [
        {"quarter": f"202{i // 4 + 1}Q{i % 4 + 1}", "year": 2021 + i // 4, "period": i % 4 + 1}
        for i in range(20)
    ]
    (cache_dir / "TTKOM.json").write_text(
        json.dumps(
            {
                "ok": True,
                "schema_version": KAP_CACHE_SCHEMA_VERSION,
                "company": "TTKOM",
                "stock_code": "TTKOM",
                "fetched_at": "2026-01-01T00:00:00+00:00",
                "live_disclosure_checked_at": kap_fetcher._utc_now().isoformat(),
                "error": "previous live error",
                "quarters": quarters,
            }
        ),
        encoding="utf-8",
    )

    def fail_live_request(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("complete cache should be served before live KAP lookup")

    monkeypatch.setattr(kap_fetcher, "_resolve_member", fail_live_request)

    payload = fetch_kap_company_snapshot(
        company="TTKOM",
        cfg=SimpleNamespace(cache_ttl_hours=0.0, timeout_seconds=1.0, user_agent="ragfin-test/1.0"),
        processed_dir=tmp_path,
        force_refresh=False,
        max_quarters=20,
        use_cache_when_complete=True,
    )

    assert payload["ok"] is True
    assert payload["cache_hit"] is True
    assert payload["cache_stale"] is False
    assert "error" not in payload
    assert len(payload["quarters"]) == 20


def test_parse_insurance_premium_pdf_text_extracts_general_total() -> None:
    text = (
        "2025 2026 Değişim 2025 2026 Değişim "
        "2.193.317 3.331.740 52% GENEL TOPLAM 10.990.588 15.024.788 37% "
        "Aylık Nisan Ayı Prim Üretimi Sonuçları (Bin TL) Yıl Başından Bugüne"
    )

    parsed = kap_fetcher._parse_insurance_premium_pdf_text(text)

    assert parsed == {
        "monthly_gross_premium": 3_331_740_000.0,
        "ytd_gross_premium": 15_024_788_000.0,
        "previous_year_monthly_gross_premium": 2_193_317_000.0,
        "previous_year_ytd_gross_premium": 10_990_588_000.0,
        "monthly_yoy_pct": 52.0,
        "ytd_yoy_pct": 37.0,
    }


def test_fetch_insurance_premium_disclosures_filters_and_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "publishDate": "08.05.2026 18:27:14",
            "disclosureIndex": 1603625,
            "summary": "Aksigorta A.Ş. 01.01.2026 - 30.04.2026 Tarihleri Arası Brüt Yazılan Prim",
            "subject": "Özel Durum Açıklaması (Genel)",
        },
        {
            "publishDate": "14.05.2026 16:35:15",
            "disclosureIndex": 1607328,
            "summary": "Üst Düzey Yönetici Değişikliği",
            "subject": "Özel Durum Açıklaması (Genel)",
        },
    ]

    criteria_calls = []

    def fake_list_member_disclosures_by_criteria(**kwargs):
        criteria_calls.append(kwargs)
        return rows

    monkeypatch.setattr(kap_fetcher, "_list_member_disclosures_by_criteria", fake_list_member_disclosures_by_criteria)
    monkeypatch.setattr(kap_fetcher, "_fetch_attachment_detail", lambda _idx, _cfg_obj: {"attachments": []})
    monkeypatch.setattr(
        kap_fetcher,
        "_premium_attachment_text",
        lambda _detail, _cfg_obj: (
            "2.193.317 3.331.740 52% GENEL TOPLAM 10.990.588 15.024.788 37% "
            "Prim Üretimi Sonuçları (Bin TL)"
        ),
    )

    result = kap_fetcher._fetch_insurance_premium_disclosures(
        member_oid="oid",
        cfg=_cfg(),
        max_items=12,
    )

    assert len(result) == 1
    assert criteria_calls[0]["lookback_days"] == kap_fetcher.KAP_INSURANCE_PREMIUM_DISCLOSURE_LOOKBACK_DAYS
    assert result[0]["period_label"] == "2026/4"
    assert result[0]["monthly_gross_premium"] == 3_331_740_000.0
    assert result[0]["ytd_gross_premium"] == 15_024_788_000.0
    assert result[0]["source_url"].endswith("/1603625")


def test_fetch_insurance_premium_ytd_only_disclosures_derives_monthly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "publishDate": "06.04.2026 11:26:13",
            "disclosureIndex": 1584367,
            "summary": "03.2026 Dönemi Brüt Prim Üretimi",
            "subject": "Özel Durum Açıklaması (Genel)",
        },
        {
            "publishDate": "07.05.2026 21:21:28",
            "disclosureIndex": 1602863,
            "summary": "04.2026 Dönemi Brüt Prim Üretimi",
            "subject": "Özel Durum Açıklaması (Genel)",
        },
    ]
    text_by_index = {
        1584367: "GENEL TOPLAM 300 200 50,0 01.01.2026 - 31.03.2026 DÖNEMİ BRÜT PRİM ÜRETİMİ 03.2026 03.2025",
        1602863: "GENEL TOPLAM 450 260 73,1 01.01.2026 - 30.04.2026 DÖNEMİ BRÜT PRİM ÜRETİMİ 04.2026 04.2025",
    }

    monkeypatch.setattr(kap_fetcher, "_list_member_disclosures_by_criteria", lambda **_kwargs: rows)
    monkeypatch.setattr(kap_fetcher, "_fetch_attachment_detail", lambda idx, _cfg_obj: {"idx": idx})
    monkeypatch.setattr(
        kap_fetcher,
        "_premium_attachment_text",
        lambda detail, _cfg_obj: text_by_index[int(detail["idx"])],
    )

    result = kap_fetcher._fetch_insurance_premium_disclosures(
        member_oid="oid",
        cfg=_cfg(),
        max_items=12,
    )

    assert [(row["year"], row["month"]) for row in result] == [(2026, 3), (2026, 4)]
    assert result[1]["monthly_gross_premium"] == 150.0
    assert result[1]["previous_year_monthly_gross_premium"] == 60.0
    assert result[1]["monthly_yoy_pct"] == 150.0
    assert result[1]["period_start"] == "01.01.2026"
    assert result[1]["period_end"] == "30.04.2026"


def test_fetch_insurance_premium_month_name_summary_and_total_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "publishDate": "07.04.2026 07:28:51",
            "disclosureIndex": 1585302,
            "summary": "2026 Mart Sonu Brüt Prim Üretimi",
            "subject": "Özel Durum Açıklaması (Genel)",
        },
        {
            "publishDate": "12.05.2026 07:27:18",
            "disclosureIndex": 1605913,
            "summary": "2026 Nisan Sonu Brüt Prim Üretimi",
            "subject": "Özel Durum Açıklaması (Genel)",
        },
    ]
    text_by_index = {
        1585302: "BRANŞ 01.01-31.03.2026 01.01-31.03.2025 Değişim TOPLAM 53.805.814.884 41.401.841.724 %30",
        1605913: "BRANŞ 01.01-30.04.2026 01.01-30.04.2025 Değişim TOPLAM 72.000.000.000 55.000.000.000 %31",
    }

    monkeypatch.setattr(kap_fetcher, "_list_member_disclosures_by_criteria", lambda **_kwargs: rows)
    monkeypatch.setattr(kap_fetcher, "_fetch_attachment_detail", lambda idx, _cfg_obj: {"idx": idx})
    monkeypatch.setattr(kap_fetcher, "_premium_disclosure_text", lambda _detail: "")
    monkeypatch.setattr(
        kap_fetcher,
        "_premium_attachment_text",
        lambda detail, _cfg_obj: text_by_index[int(detail["idx"])],
    )

    result = kap_fetcher._fetch_insurance_premium_disclosures(
        member_oid="oid",
        cfg=_cfg(),
        max_items=12,
    )

    assert [(row["year"], row["month"]) for row in result] == [(2026, 3), (2026, 4)]
    assert result[0]["ytd_gross_premium"] == 53_805_814_884.0
    assert result[0]["previous_year_ytd_gross_premium"] == 41_401_841_724.0
    assert result[1]["monthly_gross_premium"] == 18_194_185_116.0
    assert result[1]["previous_year_monthly_gross_premium"] == 13_598_158_276.0


def test_fetch_insurance_premium_month_range_sentence_derives_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "publishDate": "13.06.2025 08:19:59",
            "disclosureIndex": 1448370,
            "summary": "Gross Premium Production as of May 2025",
            "subject": "Özel Durum Açıklaması (Genel)",
        },
    ]

    monkeypatch.setattr(kap_fetcher, "_list_member_disclosures_by_criteria", lambda **_kwargs: rows)
    monkeypatch.setattr(kap_fetcher, "_fetch_attachment_detail", lambda idx, _cfg_obj: {"idx": idx})
    monkeypatch.setattr(
        kap_fetcher,
        "_premium_disclosure_text",
        lambda _detail: (
            "Şirketimizin 2025 yılı Ocak-Mayıs aylarını kapsayan döneme ait "
            "denetimden geçmemiş, tahmini prim üretimi 17.064.261.088,15 TL'dir. "
            "Prim üretiminde bir önceki yılın aynı dönemine göre % 51,21 oranında artış meydana gelmiştir."
        ),
    )
    monkeypatch.setattr(kap_fetcher, "_premium_attachment_text", lambda _detail, _cfg_obj: "")

    result = kap_fetcher._fetch_insurance_premium_disclosures(
        member_oid="oid",
        cfg=_cfg(),
        max_items=12,
    )

    assert [(row["year"], row["month"]) for row in result] == [(2025, 5)]
    assert result[0]["ytd_gross_premium"] == 17_064_261_088.15
    assert result[0]["ytd_yoy_pct"] == 51.21
    assert result[0]["previous_year_ytd_gross_premium"] == pytest.approx(17_064_261_088.15 / 1.5121)


def test_parse_marketvisuals_insurance_premium_page_builds_monthly_rows() -> None:
    sections = [
        {"title": "AgeSA Hayat ve Emeklilik AŞ — Aylık Prim", "chartId": "chart-0"},
        {"title": "AgeSA Hayat ve Emeklilik AŞ — Çeyreksel Prim", "chartId": "chart-1"},
    ]
    charts = [
        {
            "id": "chart-0",
            "data": {
                "labels": [
                    "Ocak",
                    "Şubat",
                    "Mart",
                    "Nisan",
                    "Mayıs",
                    "Haziran",
                    "Temmuz",
                    "Ağustos",
                    "Eylül",
                    "Ekim",
                    "Kasım",
                    "Aralık",
                ],
                "datasets": [
                    {"label": "2025", "data": [1484.37, 1550.27, 1689.24, 1810.66, 0, 0, 0, 0, 0, 0, 0, 0]},
                    {"label": "2026", "data": [1655.18, 2588.23, 2807.74, 2631.16, 0, 0, 0, 0, 0, 0, 0, 0]},
                ],
            },
        }
    ]
    html = (
        "<script>"
        f"var SECTIONS = {json.dumps(sections, ensure_ascii=False)};"
        "var DASHBOARD = {};"
        f"var CHARTS = {json.dumps(charts, ensure_ascii=False)};"
        "</script>"
    )

    result = kap_fetcher._parse_marketvisuals_insurance_premium_page(
        html,
        company_key="AGESA",
        company_title="AgeSA Hayat ve Emeklilik AŞ",
        source_url="https://marketvisuals.net/insurance_hayat.html",
    )

    assert [(row["year"], row["month"]) for row in result] == [
        (2025, 1),
        (2025, 2),
        (2025, 3),
        (2025, 4),
        (2026, 1),
        (2026, 2),
        (2026, 3),
        (2026, 4),
    ]
    assert result[-1]["monthly_gross_premium"] == 2_631_160_000.0
    assert result[-1]["previous_year_monthly_gross_premium"] == 1_810_660_000.0
    assert result[-1]["monthly_yoy_pct"] == 45.3
    assert result[-1]["summary"] == kap_fetcher.MARKETVISUALS_TSB_SOURCE_LABEL
    assert result[-1]["source_url"] == "https://marketvisuals.net/insurance_hayat.html"


def test_fetch_tsb_marketvisuals_premium_disclosures_uses_company_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sections = [{"title": "Anadolu Hayat Emeklilik AŞ — Aylık Prim", "chartId": "chart-2"}]
    charts = [
        {
            "id": "chart-2",
            "data": {
                "labels": ["Ocak", "Şubat"],
                "datasets": [{"label": "2026", "data": [1567.0, 2148.24]}],
            },
        }
    ]
    html = (
        "<script>"
        f"var SECTIONS = {json.dumps(sections, ensure_ascii=False)};"
        f"var CHARTS = {json.dumps(charts, ensure_ascii=False)};"
        "</script>"
    )
    calls: list[str] = []

    def fake_get_text(url: str, _cfg_obj: object) -> str:
        calls.append(url)
        return html

    monkeypatch.setattr(kap_fetcher, "_http_get_text", fake_get_text)

    result = kap_fetcher._fetch_tsb_marketvisuals_premium_disclosures(
        company_key="ANHYT",
        company_title="Anadolu Hayat Emeklilik A.Ş.",
        cfg=_cfg(),
    )

    assert calls == [kap_fetcher.MARKETVISUALS_INSURANCE_HAYAT_URL]
    assert [(row["year"], row["month"], row["monthly_gross_premium"]) for row in result] == [
        (2026, 1, 1_567_000_000.0),
        (2026, 2, 2_148_240_000.0),
    ]


def test_merge_insurance_premium_disclosures_preserves_existing_periods() -> None:
    merged = kap_fetcher._merge_insurance_premium_disclosures(
        [
            {"year": 2024, "month": 1, "monthly_gross_premium": 1.0},
            {"year": 2024, "month": 2, "monthly_gross_premium": 2.0},
        ],
        [
            {"year": 2024, "month": 2, "monthly_gross_premium": 22.0},
            {"year": 2024, "month": 3, "monthly_gross_premium": 3.0},
        ],
    )

    assert [(row["year"], row["month"], row["monthly_gross_premium"]) for row in merged] == [
        (2024, 1, 1.0),
        (2024, 2, 22.0),
        (2024, 3, 3.0),
    ]


def test_fetch_snapshot_cached_fallback_refreshes_insurance_premiums(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cache_dir = tmp_path / "kap_cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "ANSGR.json"
    cache_file.write_text(
        json.dumps(
            {
                "ok": True,
                "schema_version": KAP_CACHE_SCHEMA_VERSION,
                "company": "ANSGR",
                "company_title": "Anadolu Sigorta",
                "stock_code": "ANSGR",
                "member_oid": "oid",
                "fetched_at": "2020-01-01T00:00:00+00:00",
                "quarters": [
                    {
                        "year": 2025,
                        "period": 1,
                        "quarter": "2025Q1",
                        "metrics": {"prim_uretimi": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fail_resolve_member(*_args, **_kwargs):
        raise RuntimeError("live kap unavailable")

    monkeypatch.setattr(kap_fetcher, "_resolve_member", fail_resolve_member)
    monkeypatch.setattr(
        kap_fetcher,
        "_fetch_insurance_premium_disclosures",
        lambda **_kwargs: [{"year": 2026, "month": 4, "monthly_gross_premium": 3.0}],
    )
    monkeypatch.setattr(kap_fetcher, "_fetch_tsb_marketvisuals_premium_disclosures", lambda **_kwargs: [])

    payload = fetch_kap_company_snapshot(
        company="ANSGR",
        cfg=SimpleNamespace(user_agent="ragfin-test/1.0", timeout_seconds=1.0, cache_ttl_hours=0.0),
        processed_dir=tmp_path,
        max_quarters=1,
    )

    assert payload["cache_hit"] is True
    assert payload["cache_stale"] is False
    assert payload["insurance_premium_disclosures"] == [
        {"year": 2026, "month": 4, "monthly_gross_premium": 3.0}
    ]
    saved = json.loads(cache_file.read_text(encoding="utf-8"))
    assert saved["insurance_premium_disclosures_version"] == kap_fetcher.KAP_INSURANCE_PREMIUM_CACHE_VERSION


def test_ensure_insurance_premium_disclosures_uses_marketvisuals_when_kap_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cache_path = tmp_path / "AGESA.json"
    payload = {
        "ok": True,
        "schema_version": KAP_CACHE_SCHEMA_VERSION,
        "company": "AGESA",
        "company_title": "AGESA HAYAT VE EMEKLİLİK A.Ş.",
        "stock_code": "AGESA",
        "member_oid": "oid",
        "quarters": [],
    }

    monkeypatch.setattr(kap_fetcher, "_fetch_insurance_premium_disclosures", lambda **_kwargs: [])
    monkeypatch.setattr(
        kap_fetcher,
        "_fetch_tsb_marketvisuals_premium_disclosures",
        lambda **_kwargs: [{"year": 2026, "month": 4, "monthly_gross_premium": 2_631_160_000.0}],
    )

    result = kap_fetcher._ensure_insurance_premium_disclosures(
        payload=payload,
        cache_path=cache_path,
        cfg=_cfg(),
    )

    assert result["insurance_premium_disclosures"] == [
        {"year": 2026, "month": 4, "monthly_gross_premium": 2_631_160_000.0}
    ]
    assert result["insurance_premium_disclosures_version"] == kap_fetcher.KAP_INSURANCE_PREMIUM_CACHE_VERSION
    assert json.loads(cache_path.read_text(encoding="utf-8"))["insurance_premium_disclosures"][0]["year"] == 2026


def test_ensure_insurance_premium_disclosures_prefers_marketvisuals_for_known_company(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    payload = {
        "ok": True,
        "schema_version": KAP_CACHE_SCHEMA_VERSION,
        "company": "RAYSG",
        "company_title": "RAY SİGORTA A.Ş.",
        "stock_code": "RAYSG",
        "member_oid": "oid",
        "quarters": [],
        "insurance_premium_disclosures": [
            {"year": 2026, "month": 4, "monthly_gross_premium": -1.0},
        ],
    }

    monkeypatch.setattr(
        kap_fetcher,
        "_fetch_insurance_premium_disclosures",
        lambda **_kwargs: [{"year": 2026, "month": 4, "monthly_gross_premium": 10.0}],
    )
    monkeypatch.setattr(
        kap_fetcher,
        "_fetch_tsb_marketvisuals_premium_disclosures",
        lambda **_kwargs: [{"year": 2026, "month": 4, "monthly_gross_premium": 3_153_910_000.0}],
    )

    result = kap_fetcher._ensure_insurance_premium_disclosures(
        payload=payload,
        cache_path=tmp_path / "RAYSG.json",
        cfg=_cfg(),
    )

    assert result["insurance_premium_disclosures"] == [
        {"year": 2026, "month": 4, "monthly_gross_premium": 3_153_910_000.0}
    ]


def test_fetch_snapshot_uses_premium_only_when_financials_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        kap_fetcher,
        "_resolve_member",
        lambda _company, _cfg_obj: {
            "company_code": "RAYSG",
            "mkk_member_oid": "oid",
            "title": "RAY SİGORTA A.Ş.",
            "permalink": "1063-ray-sigorta-a-s",
        },
    )
    monkeypatch.setattr(
        kap_fetcher,
        "_list_company_disclosures",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("KAP istek limiti")),
    )
    monkeypatch.setattr(
        kap_fetcher,
        "_fetch_insurance_premium_disclosures",
        lambda **_kwargs: [{"year": 2026, "month": 4, "monthly_gross_premium": 10.0}],
    )
    monkeypatch.setattr(kap_fetcher, "_fetch_tsb_marketvisuals_premium_disclosures", lambda **_kwargs: [])

    payload = fetch_kap_company_snapshot(
        company="RAYSG",
        cfg=SimpleNamespace(user_agent="ragfin-test/1.0", timeout_seconds=1.0, cache_ttl_hours=0.0),
        processed_dir=tmp_path,
        max_quarters=1,
    )

    assert payload["ok"] is True
    assert payload["cache_stale"] is True
    assert payload["quarters"] == []
    assert payload["insurance_premium_disclosures"] == [
        {"year": 2026, "month": 4, "monthly_gross_premium": 10.0}
    ]


def test_fetch_snapshot_uses_marketvisuals_premium_only_when_member_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(kap_fetcher, "_resolve_member", lambda _company, _cfg_obj: None)
    monkeypatch.setattr(
        kap_fetcher,
        "_fetch_tsb_marketvisuals_premium_disclosures",
        lambda **_kwargs: [{"year": 2026, "month": 4, "monthly_gross_premium": 2_056_630_000.0}],
    )

    payload = fetch_kap_company_snapshot(
        company="ANHYT",
        cfg=SimpleNamespace(user_agent="ragfin-test/1.0", timeout_seconds=1.0, cache_ttl_hours=0.0),
        processed_dir=tmp_path,
        max_quarters=1,
    )

    assert payload["ok"] is True
    assert payload["company_title"] == "Anadolu Hayat Emeklilik AŞ"
    assert payload["quarters"] == []
    assert payload["insurance_premium_disclosures"] == [
        {"year": 2026, "month": 4, "monthly_gross_premium": 2_056_630_000.0}
    ]


def test_net_kar_prefers_parent_or_net_profit_rows() -> None:
    rows = [
        {
            "label_norm": "surdurulen faaliyetler donem kari (zarari)",
            "body_index": 1,
            "col_order": 4,
            "value": 34_628_000_000.0,
        },
        {
            "label_norm": "net donem kari veya zarari",
            "body_index": 0,
            "col_order": 4,
            "value": 22_001_000_000.0,
        },
        {
            "label_norm": "ana ortaklik paylari",
            "body_index": 1,
            "col_order": 4,
            "value": 22_001_000_000.0,
        },
        {
            "label_norm": "kontrol gucu olmayan paylar",
            "body_index": 1,
            "col_order": 4,
            "value": 50_000_000_000.0,
        },
    ]

    picked = _pick_metric_value("net_kar", rows, period=4)
    assert picked == 22_001_000_000.0


def test_net_kar_falls_back_when_only_continued_operations_exists() -> None:
    rows = [
        {
            "label_norm": "surdurulen faaliyetler donem kari (zarari)",
            "body_index": 1,
            "col_order": 4,
            "value": 3_500_000_000.0,
        }
    ]

    picked = _pick_metric_value("net_kar", rows, period=3)
    assert picked == 3_500_000_000.0
