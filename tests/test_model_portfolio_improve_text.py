from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from api_module_facade import api_module as api_mod
from app.model_portfolio_service import (
    reset_admin_rate_limit_for_tests,
    reset_model_portfolio_cache_for_tests,
)
from src.nvidia_commentary import (
    NvidiaCommentaryError,
    PayloadValidationError,
    improve_model_portfolio_text,
)


@pytest.fixture(autouse=True)
def clean_portfolio_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reset_model_portfolio_cache_for_tests()
    reset_admin_rate_limit_for_tests()
    monkeypatch.delenv("RAGFIN_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("RAGFIN_ADMIN_TOKEN", "test-founder-secret")


def _get_auth_headers(client: TestClient) -> Dict[str, str]:
    verify_res = client.post(
        "/model-portfolio/admin/verify",
        headers={"X-Admin-Token": "test-founder-secret"},
    )
    assert verify_res.status_code == 200
    token = verify_res.json()["session_token"]
    return {"Authorization": f"Bearer {token}"}


def test_improve_text_auth_required() -> None:
    client = TestClient(api_mod.app)

    # Missing auth
    res_no_auth = client.post(
        "/model-portfolio/improve-text",
        json={"field": "thesis", "text": "Hisse yükselebilir."},
    )
    assert res_no_auth.status_code == 401

    # Invalid token
    res_bad_auth = client.post(
        "/model-portfolio/improve-text",
        headers={"Authorization": "Bearer bad-token"},
        json={"field": "thesis", "text": "Hisse yükselebilir."},
    )
    assert res_bad_auth.status_code == 401


def test_improve_text_payload_validation() -> None:
    client = TestClient(api_mod.app)
    auth_headers = _get_auth_headers(client)

    # Invalid field
    res_bad_field = client.post(
        "/model-portfolio/improve-text",
        headers=auth_headers,
        json={"field": "unknown_field", "text": "Deneme metni."},
    )
    assert res_bad_field.status_code == 422
    assert "Düzenlenecek alan geçersiz" in res_bad_field.json()["detail"]

    # Empty text
    res_empty_text = client.post(
        "/model-portfolio/improve-text",
        headers=auth_headers,
        json={"field": "thesis", "text": "   "},
    )
    assert res_empty_text.status_code == 422
    assert "boş olamaz" in res_empty_text.json()["detail"]

    # Text length limits: risk_note max 1500
    res_too_long_risk = client.post(
        "/model-portfolio/improve-text",
        headers=auth_headers,
        json={"field": "risk_note", "text": "A" * 1501},
    )
    assert res_too_long_risk.status_code == 422
    assert "en fazla 1500 karakter" in res_too_long_risk.json()["detail"]

    # Text length limits: thesis max 4000
    res_too_long_thesis = client.post(
        "/model-portfolio/improve-text",
        headers=auth_headers,
        json={"field": "thesis", "text": "B" * 4001},
    )
    assert res_too_long_thesis.status_code == 422
    assert "en fazla 4000 karakter" in res_too_long_thesis.json()["detail"]

    # Oversized raw body (> 24 KB)
    res_huge = client.post(
        "/model-portfolio/improve-text",
        headers=auth_headers,
        content=b"x" * (25 * 1024),
    )
    assert res_huge.status_code == 413


def test_improve_text_service_unconfigured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IFM_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    client = TestClient(api_mod.app)
    auth_headers = _get_auth_headers(client)

    res = client.post(
        "/model-portfolio/improve-text",
        headers=auth_headers,
        json={"field": "thesis", "text": "Operasyonel marjlar artacak."},
    )
    assert res.status_code == 503
    assert "AI düzenleme servisi yapılandırılmamış" in res.json()["detail"]


@pytest.mark.anyio
async def test_improve_text_endpoint_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(api_mod.app)
    auth_headers = _get_auth_headers(client)

    with patch(
        "src.nvidia_commentary.improve_model_portfolio_text",
        new_callable=AsyncMock,
        return_value="Operasyonel kârlılık ve nakit akışı artışı beklenmektedir.",
    ) as mock_improve:
        res = client.post(
            "/model-portfolio/improve-text",
            headers=auth_headers,
            json={
                "field": "thesis",
                "text": "kar nakit akisi iyi olacak",
                "symbol": "THYAO",
                "company_name": "TÜRK HAVA YOLLARI",
                "sector": "Ulaştırma",
            },
        )
        assert res.status_code == 200
        assert res.json() == {
            "text": "Operasyonel kârlılık ve nakit akışı artışı beklenmektedir."
        }
        mock_improve.assert_called_once_with(
            "kar nakit akisi iyi olacak",
            "thesis",
            symbol="THYAO",
            company_name="TÜRK HAVA YOLLARI",
            sector="Ulaştırma",
        )


@pytest.mark.anyio
async def test_improve_model_portfolio_text_unit_ifm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IFM_API_KEY", "test-ifm-key")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    mock_response = httpx.Response(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "content": "“Maliyet baskısı ve döviz kuru dalgalanmaları ana risk unsurlarıdır.”"
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://api.ifm.example/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        result = await improve_model_portfolio_text(
            "maliyet ve dolar riski var",
            "risk_note",
            symbol="BIMAS",
            company_name="BİM Birleşik Mağazalar",
        )
        assert result == "Maliyet baskısı ve döviz kuru dalgalanmaları ana risk unsurlarıdır."
        mock_post.assert_called_once()
        sent_body = mock_post.call_args[1]["json"]
        assert sent_body["temperature"] == 0.25
        assert sent_body["chat_template_kwargs"] == {"reasoning_effort": "low"}
        assert "BIST model portföy yöneticisinin Türkçe metin editörüsün" in sent_body["messages"][0]["content"]
        assert "Hisse kodu: BIMAS" in sent_body["messages"][1]["content"]


@pytest.mark.anyio
async def test_improve_model_portfolio_text_unit_nvidia_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IFM_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nv-key")

    mock_response = httpx.Response(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "content": "Güçlü bilanço büyümesi öngörülmektedir."
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://api.nvidia.example/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        result = await improve_model_portfolio_text("bilanco buyumesi iyi", "thesis")
        assert result == "Güçlü bilanço büyümesi öngörülmektedir."
        sent_body = mock_post.call_args[1]["json"]
        assert sent_body["chat_template_kwargs"] == {"thinking": False}


@pytest.mark.anyio
async def test_improve_model_portfolio_text_unit_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IFM_API_KEY", "test-ifm-key")

    # Invalid field
    with pytest.raises(PayloadValidationError):
        await improve_model_portfolio_text("metin", "invalid_field")

    # Empty AI response
    empty_resp = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "   "}}]},
        request=httpx.Request("POST", "https://api.ifm.example"),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=empty_resp):
        with pytest.raises(NvidiaCommentaryError):
            await improve_model_portfolio_text("metin", "thesis")

    # HTTP Error from provider
    error_resp = httpx.Response(
        status_code=500,
        text="Internal Server Error",
        request=httpx.Request("POST", "https://api.ifm.example"),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=error_resp):
        with pytest.raises(NvidiaCommentaryError):
            await improve_model_portfolio_text("metin", "thesis")
