"""LLM provider abstraction.

Provider chain: Gemini → Groq → OpenRouter → safe refusal.
Each provider exposes generate(system_prompt, user_message) → (text, metadata).
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    APIStatusError,
)

logger = logging.getLogger(__name__)

PROVIDER_TIMEOUT_SECONDS = 30.0


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging (e.g. 'gemini', 'groq')."""

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True if the provider has the required API key set."""

    @abstractmethod
    def _do_generate(
        self, system_prompt: str, user_message: str,
        max_tokens: int, temperature: float,
    ) -> tuple[Optional[str], dict]:
        """Subclass implementation. Returns (text, meta) or (None, meta) on failure."""

    def generate(
        self, system_prompt: str, user_message: str,
        max_tokens: int = 2048, temperature: float = 0.1,
    ) -> tuple[Optional[str], dict]:
        """Generate with structured logging and error wrapping."""
        if not self.is_configured:
            return None, {
                "provider": self.name,
                "error": "not_configured",
                "failure_type": "not_configured",
            }

        t0 = time.time()
        try:
            text, meta = self._do_generate(system_prompt, user_message, max_tokens, temperature)
            meta["latency_ms"] = round((time.time() - t0) * 1000, 1)
            if text:
                logger.info(
                    f"[PROVIDER] {self.name} OK: {meta.get('model', '?')} "
                    f"({meta['latency_ms']:.0f}ms, {meta.get('total_tokens', '?')} tokens)"
                )
            else:
                logger.warning(
                    f"[PROVIDER] {self.name} failed: {meta.get('error', 'unknown')} "
                    f"({meta.get('failure_type', '?')}, {meta['latency_ms']:.0f}ms)"
                )
            return text, meta
        except Exception as e:
            latency = round((time.time() - t0) * 1000, 1)
            logger.error(f"[PROVIDER] {self.name} exception: {e}")
            return None, {
                "provider": self.name,
                "model": "",
                "error": str(e)[:200],
                "failure_type": "connection",
                "latency_ms": latency,
            }


class GeminiProvider(LLMProvider):
    """Google Gemini via REST API (no SDK required)."""

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _do_generate(
        self, system_prompt: str, user_message: str,
        max_tokens: int, temperature: float,
    ) -> tuple[Optional[str], dict]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        meta = {"provider": "gemini", "model": self._model}

        with httpx.Client(timeout=PROVIDER_TIMEOUT_SECONDS) as client:
            resp = client.post(url, json=payload)

        meta["status_code"] = resp.status_code

        if resp.status_code == 429:
            meta["error"] = "rate_limited"
            meta["failure_type"] = "429"
            return None, meta
        if resp.status_code == 403:
            meta["error"] = "forbidden_invalid_key"
            meta["failure_type"] = "403"
            return None, meta
        if resp.status_code != 200:
            meta["error"] = resp.text[:200]
            meta["failure_type"] = str(resp.status_code)
            return None, meta

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            meta["error"] = "no_candidates"
            meta["failure_type"] = "empty_content"
            return None, meta

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()

        if not text:
            meta["error"] = "empty_content"
            meta["failure_type"] = "empty_content"
            return None, meta

        usage = data.get("usageMetadata", {})
        meta["success"] = True
        meta["prompt_tokens"] = usage.get("promptTokenCount", 0)
        meta["completion_tokens"] = usage.get("candidatesTokenCount", 0)
        meta["total_tokens"] = usage.get("totalTokenCount", 0)
        return text, meta


class GroqProvider(LLMProvider):
    """Groq via OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.groq.com/openai/v1"):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "groq"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _do_generate(
        self, system_prompt: str, user_message: str,
        max_tokens: int, temperature: float,
    ) -> tuple[Optional[str], dict]:
        return self._call_openai_compatible(
            system_prompt, user_message, max_tokens, temperature, "groq",
        )

    def _call_openai_compatible(
        self, system_prompt, user_message, max_tokens, temperature, provider_name,
    ) -> tuple[Optional[str], dict]:
        meta = {"provider": provider_name, "model": self._model}
        client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=PROVIDER_TIMEOUT_SECONDS,
        )

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        usage = getattr(response, "usage", None) or {}
        meta["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
        meta["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
        meta["total_tokens"] = getattr(usage, "total_tokens", 0) or 0

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content if choice else "") or ""
        meta["finish_reason"] = getattr(choice, "finish_reason", "unknown") if choice else "unknown"

        if not text.strip():
            meta["error"] = "empty_content"
            meta["failure_type"] = "empty_content"
            return None, meta

        meta["success"] = True
        return text.strip(), meta


class OpenRouterProvider(LLMProvider):
    """OpenRouter via OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1"):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _do_generate(
        self, system_prompt: str, user_message: str,
        max_tokens: int, temperature: float,
    ) -> tuple[Optional[str], dict]:
        meta = {"provider": "openrouter", "model": self._model}
        client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=PROVIDER_TIMEOUT_SECONDS,
        )

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        usage = getattr(response, "usage", None) or {}
        meta["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
        meta["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
        meta["total_tokens"] = getattr(usage, "total_tokens", 0) or 0

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content if choice else "") or ""
        meta["finish_reason"] = getattr(choice, "finish_reason", "unknown") if choice else "unknown"

        if not text.strip():
            meta["error"] = "empty_content"
            meta["failure_type"] = "empty_content"
            return None, meta

        meta["success"] = True
        return text.strip(), meta


def build_provider_chain(
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.0-flash",
    groq_api_key: str = "",
    groq_model: str = "llama-3.3-70b-versatile",
    groq_base_url: str = "https://api.groq.com/openai/v1",
    openrouter_api_key: str = "",
    openrouter_model: str = "openai/gpt-oss-20b:free",
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    forced_provider: str = "",
) -> list[LLMProvider]:
    """Build ordered provider list based on available keys.

    If forced_provider is set (e.g. 'gemini'), only that provider is returned.
    Otherwise: Gemini → Groq → OpenRouter (skipping unconfigured).
    """
    if forced_provider and forced_provider != "auto":
        pmap = {
            "gemini": lambda: GeminiProvider(gemini_api_key, gemini_model),
            "groq": lambda: GroqProvider(groq_api_key, groq_model, groq_base_url),
            "openrouter": lambda: OpenRouterProvider(openrouter_api_key, openrouter_model, openrouter_base_url),
        }
        factory = pmap.get(forced_provider)
        if factory:
            p = factory()
            return [p] if p.is_configured else []
        return []

    chain = []
    gemini = GeminiProvider(gemini_api_key, gemini_model)
    if gemini.is_configured:
        chain.append(gemini)

    groq = GroqProvider(groq_api_key, groq_model, groq_base_url)
    if groq.is_configured:
        chain.append(groq)

    openrouter = OpenRouterProvider(openrouter_api_key, openrouter_model, openrouter_base_url)
    if openrouter.is_configured:
        chain.append(openrouter)

    return chain
