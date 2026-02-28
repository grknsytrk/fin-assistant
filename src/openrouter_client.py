from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_sec: float = 8.0,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url).rstrip("/")
        self.timeout_sec = float(timeout_sec)
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY bos olamaz")

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": str(model),
            "messages": messages,
            "temperature": float(temperature),
        }
        if max_tokens is not None:
            try:
                parsed_max_tokens = int(max_tokens)
                if parsed_max_tokens > 0:
                    payload["max_tokens"] = parsed_max_tokens
            except Exception:
                pass
        if response_format:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(dict(extra_body))

        raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=raw_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "rag-fin",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            raise OpenRouterError(
                f"OpenRouter HTTP hata: {exc.code} {exc.reason} {detail[:240]}".strip()
            ) from exc
        except Exception as exc:
            raise OpenRouterError(f"OpenRouter baglanti hatasi: {exc}") from exc

        try:
            payload = json.loads(body)
        except Exception as exc:
            raise OpenRouterError(f"OpenRouter JSON parse hatasi: {exc}") from exc

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenRouterError("OpenRouter yanitinda choices yok")

        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text is not None:
                        text_parts.append(str(text))
            content_text = "\n".join(text_parts).strip()
        else:
            content_text = str(content or "").strip()

        if not content_text:
            raise OpenRouterError("OpenRouter bos icerik dondu")
        return content_text
