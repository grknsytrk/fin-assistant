from __future__ import annotations

from types import SimpleNamespace

from src.commentary import SAFE_EMPTY_COMMENTARY, build_commentary_input, generate_commentary


def _cfg(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        llm_assistant=SimpleNamespace(
            enabled=enabled,
            provider="openrouter",
            model="arcee-ai/trinity-large-preview:free",
            max_tokens=None,
            timeout_s=8.0,
            temperature=0.2,
            reasoning_enabled=True,
        )
    )


def test_build_commentary_input_shape() -> None:
    payload = build_commentary_input(
        company="BIM",
        period="Q3",
        metrics={"net_kar": 11_370_000_000, "satis_gelirleri": 512_770_000_000},
        ratios={"net_margin": 2.2, "favok_margin": 5.7},
        deltas={"net_kar_qoq": 1_200_000_000},
        confidence_map={"net_kar": 0.92, "net_margin": 0.88},
        verify_map={"net_kar": "PASS", "net_margin": "WARN"},
        evidence_snippets=["Q3 net kar artisi", "FAVOK marji dengeli"],
    )
    assert payload["company"] == "BIM"
    assert payload["period"] == "Q3"
    assert isinstance(payload["metrics"], dict)
    assert isinstance(payload["ratios"], dict)
    assert isinstance(payload["deltas"], dict)
    assert isinstance(payload["confidence_map"], dict)
    assert isinstance(payload["verify_map"], dict)
    assert isinstance(payload["evidence_snippets"], list)
    assert "quality_flags" in payload
    assert "has_low_quality_signal" in payload["quality_flags"]


def test_generate_commentary_returns_empty_on_found_false(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    result = generate_commentary(
        answer_payload={
            "found": False,
            "answer": {"found": False},
            "evidence": [{"excerpt": "ornek"}],
        },
        question="Q3 net kar yorumu",
        cfg=_cfg(enabled=True),
    )
    assert result == SAFE_EMPTY_COMMENTARY


def test_generate_commentary_rejects_digits_with_rule_based_fallback(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self, api_key: str, base_url: str = "", timeout_sec: float = 0.0) -> None:
            self.api_key = api_key

        def chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
            return (
                '{"headline":"Q3 guclu","bullets":["Marj 2 puan iyilesti"],'
                '"risk_note":"Risk 1 alanda","next_question":"Q4 ne olur?"}'
            )

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    monkeypatch.setattr("src.commentary.OpenRouterClient", _FakeClient)

    result = generate_commentary(
        answer_payload={
            "found": True,
            "answer": {"found": True, "verify_status": "PASS"},
            "metrics": {"net_kar": 1.0},
            "ratios": {"net_margin": 2.0},
            "deltas": {"net_kar_qoq": 0.1},
            "confidence_map": {"net_kar": 0.9},
            "verify_map": {"net_kar": "PASS"},
            "evidence": [{"excerpt": "Q3 net kar artisi suruyor."}],
        },
        question="Q3 net kar yorumu",
        cfg=_cfg(enabled=True),
    )
    assert set(result.keys()) == {"headline", "bullets", "risk_note", "next_question"}
    assert result["headline"]
    assert isinstance(result["bullets"], list)
    assert len(result["bullets"]) >= 1
    # Strict mode should still avoid numeric artifacts in fallback text.
    assert all(not any(ch.isdigit() for ch in str(item)) for item in result["bullets"])


def test_generate_commentary_rejects_percent_placeholders_with_rule_based_fallback(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self, api_key: str, base_url: str = "", timeout_sec: float = 0.0) -> None:
            self.api_key = api_key

        def chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
            return (
                '{"headline":"Q3 performans","bullets":["Net kar % artis gosterdi","Satis milyar TL oldu"],'
                '"risk_note":"", "next_question":""}'
            )

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    monkeypatch.setattr("src.commentary.OpenRouterClient", _FakeClient)

    result = generate_commentary(
        answer_payload={
            "found": True,
            "answer": {"found": True, "verify_status": "WARN"},
            "metrics": {"net_kar": 1.0},
            "ratios": {},
            "deltas": {"net_kar_qoq": 0.1},
            "confidence_map": {"net_kar": 0.8},
            "verify_map": {"net_kar": "WARN"},
            "evidence": [{"excerpt": "Net kar artisi suruyor."}],
        },
        question="Q3 yorumu",
        cfg=_cfg(enabled=True),
    )
    assert set(result.keys()) == {"headline", "bullets", "risk_note", "next_question"}
    assert result["headline"]
    assert isinstance(result["bullets"], list)
    assert len(result["bullets"]) >= 1


def test_generate_commentary_valid_schema(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self, api_key: str, base_url: str = "", timeout_sec: float = 0.0) -> None:
            self.api_key = api_key

        def chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
            return (
                '{"headline":"Kisa trend ozeti",'
                '"bullets":["Kar marji yukari egilimde","Ciro ivmesi korunuyor","Operasyonel denge suruyor"],'
                '"risk_note":"Bazi satirlarda verify uyarisi var.",'
                '"next_question":"Bir sonraki ceyrekte marj devami var mi?"}'
            )

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    monkeypatch.setattr("src.commentary.OpenRouterClient", _FakeClient)

    result = generate_commentary(
        answer_payload={
            "found": True,
            "parsed": {"quarter": "Q3"},
            "answer": {"found": True, "verify_status": "WARN"},
            "company": "BIM",
            "metrics": {"net_kar": 1.0, "satis_gelirleri": 2.0},
            "ratios": {"net_margin": 2.0},
            "deltas": {"net_kar_qoq": 0.1},
            "confidence_map": {"net_kar": 0.7},
            "verify_map": {"net_kar": "WARN"},
            "evidence": [{"excerpt": "Q3 doneminde operasyonel performans korunmustur."}],
        },
        question="Q3 net kar yorumu",
        cfg=_cfg(enabled=True),
    )

    assert set(result.keys()) == {"headline", "bullets", "risk_note", "next_question"}
    assert isinstance(result["headline"], str)
    assert isinstance(result["bullets"], list)
    assert isinstance(result["risk_note"], str)
    assert isinstance(result["next_question"], str)
    assert result["headline"]
    assert len(result["bullets"]) >= 1


def test_generate_commentary_model_override(monkeypatch) -> None:
    captured: dict = {}

    class _FakeClient:
        def __init__(self, api_key: str, base_url: str = "", timeout_sec: float = 0.0) -> None:
            self.api_key = api_key

        def chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
            captured["model"] = kwargs.get("model")
            return (
                '{"headline":"Trend ozeti","bullets":["Marj ivmesi korunuyor"],'
                '"risk_note":"Dogrulama sinyali izlenmeli","next_question":"Marj devami beklenir mi?"}'
            )

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    monkeypatch.setattr("src.commentary.OpenRouterClient", _FakeClient)

    _ = generate_commentary(
        answer_payload={
            "found": True,
            "answer": {"found": True, "verify_status": "WARN"},
            "metrics": {"net_kar": 1.0},
            "ratios": {},
            "deltas": {"net_kar_qoq": 0.2},
            "confidence_map": {"net_kar": 0.8},
            "verify_map": {"net_kar": "WARN"},
            "evidence": [{"excerpt": "Net kar artisi suruyor."}],
        },
        question="Q3 yorumu",
        cfg=_cfg(enabled=True),
        model_override="stepfun/step-3.5-flash:free",
    )

    assert captured.get("model") == "stepfun/step-3.5-flash:free"


def test_generate_commentary_kap_mode_allows_numbers(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self, api_key: str, base_url: str = "", timeout_sec: float = 0.0) -> None:
            self.api_key = api_key

        def chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
            return (
                '{"headline":"KAP Q3 ozeti","bullets":["Net kar 2.02 mlr TL"],'
                '"risk_note":"", "next_question":"Q4 trendi nasil?"}'
            )

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    monkeypatch.setattr("src.commentary.OpenRouterClient", _FakeClient)

    result = generate_commentary(
        answer_payload={
            "found": True,
            "commentary_mode": "kap",
            "answer": {"found": True, "verify_status": "PASS"},
            "metrics": {"net_kar": 2_020_000_000.0},
            "ratios": {},
            "deltas": {"net_kar_qoq": 100_000_000.0},
            "confidence_map": {},
            "verify_map": {},
            "evidence": [{"excerpt": "2025/9 | Net Donem Kari: 2.020.000"}],
        },
        question="KAP ceyrek ozeti",
        cfg=_cfg(enabled=True),
    )

    assert "2.02" in " ".join(result.get("bullets", []))
