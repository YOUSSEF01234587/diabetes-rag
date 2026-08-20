"""LLM resilience: timeout, retry, fallback, malformed detection, structured logging."""
import logging
import time
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError, InternalServerError, APIStatusError

logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    NONE = "none"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "429"
    SERVER_ERROR = "5xx"
    EMPTY_CONTENT = "empty_content"
    REASONING_ONLY = "reasoning_only"
    MALFORMED = "malformed"
    UNKNOWN = "unknown"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


@dataclass
class LLMCallLog:
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    retry_count: int = 0
    failure_type: str = "none"
    success: bool = False
    attempt_count: int = 0
    fallback_used: bool = False
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "retry_count": self.retry_count,
            "failure_type": self.failure_type,
            "success": self.success,
            "attempt_count": self.attempt_count,
            "fallback_used": self.fallback_used,
        }

    def log_summary(self) -> str:
        status = "OK" if self.success else f"FAIL({self.failure_type})"
        return (
            f"[LLM {status}] model={self.model} latency={self.latency_ms:.0f}ms "
            f"tokens={self.prompt_tokens}+{self.completion_tokens} "
            f"finish={self.finish_reason} retries={self.retry_count}"
        )


TRANSIENT_ERROR_TYPES = (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError)


def _classify_failure(exc: Exception) -> FailureType:
    if isinstance(exc, APITimeoutError):
        return FailureType.TIMEOUT
    if isinstance(exc, APIConnectionError):
        return FailureType.CONNECTION
    if isinstance(exc, RateLimitError):
        return FailureType.RATE_LIMIT
    if isinstance(exc, InternalServerError):
        return FailureType.SERVER_ERROR
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", 0)
        if status == 429:
            return FailureType.RATE_LIMIT
        if 500 <= status < 600:
            return FailureType.SERVER_ERROR
    return FailureType.UNKNOWN


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, TRANSIENT_ERROR_TYPES):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", 0)
        return status == 429 or 500 <= status < 600
    return False


def _extract_usage(response) -> dict:
    usage = getattr(response, "usage", None) or {}
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0
    reasoning_tokens = 0
    completion_details = getattr(usage, "completion_tokens_details", None)
    if completion_details:
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def _extract_finish_reason(response) -> str:
    choice = response.choices[0] if response.choices else None
    if choice:
        fr = getattr(choice, "finish_reason", None) or "unknown"
        return fr
    return "unknown"


def _validate_content(content: str, reasoning_content: str = None) -> FailureType:
    if not content or not content.strip():
        if reasoning_content and reasoning_content.strip():
            return FailureType.REASONING_ONLY
        return FailureType.EMPTY_CONTENT
    return FailureType.NONE


CONCISE_ANSWER_PROMPT = (
    "You previously produced a reasoning response but no final answer. "
    "Please provide a direct, concise answer based on the evidence. "
    "Do NOT include internal reasoning. Output only the final answer with citations."
)


def call_llm_with_resilience(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.1,
    timeout_seconds: float = 90.0,
    max_retries: int = 3,
    base_retry_delay: float = 5.0,
    fallback_model: str = None,
    fallback_messages: list[dict] = None,
    provider: str = "unknown",
) -> tuple[Optional[str], LLMCallLog]:
    """Call LLM with timeout, retry, fallback, and structured logging.

    Returns (answer_text_or_None, call_log).
    On success, answer_text is a non-empty string.
    On total failure, answer_text is None and call_log.failure_type indicates why.
    """
    log = LLMCallLog(model=model, provider=provider)

    all_messages = messages
    models_to_try = [model]
    if fallback_model:
        models_to_try.append(fallback_model)

    attempt = 0
    for model_idx, current_model in enumerate(models_to_try):
        if model_idx > 0:
            log.fallback_used = True
            log.model = current_model
            all_messages = fallback_messages or messages
            logger.info(f"Falling back to model: {current_model}")

        for retry in range(max_retries):
            attempt += 1
            log.attempt_count = attempt
            log.retry_count = retry

            try:
                t0 = time.time()
                response = client.chat.completions.create(
                    model=current_model,
                    messages=all_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout_seconds,
                )
                latency_ms = (time.time() - t0) * 1000
                log.latency_ms = round(latency_ms, 1)

                usage = _extract_usage(response)
                log.prompt_tokens = usage["prompt_tokens"]
                log.completion_tokens = usage["completion_tokens"]
                log.total_tokens = usage["total_tokens"]
                log.reasoning_tokens = usage["reasoning_tokens"]

                log.finish_reason = _extract_finish_reason(response)

                content = response.choices[0].message.content if response.choices else ""
                reasoning_content = None
                if response.choices and hasattr(response.choices[0].message, "reasoning_content"):
                    reasoning_content = response.choices[0].message.reasoning_content

                failure = _validate_content(content, reasoning_content)
                if failure != FailureType.NONE:
                    log.failure_type = failure.value
                    logger.warning(f"Content validation failed: {failure.value} (attempt {attempt})")
                    if failure == FailureType.REASONING_ONLY and retry == 0 and model_idx == 0:
                        retry_messages = all_messages + [{"role": "user", "content": CONCISE_ANSWER_PROMPT}]
                        all_messages = retry_messages
                        continue
                    continue

                log.success = True
                log.failure_type = FailureType.NONE.value
                logger.info(log.log_summary())
                return content.strip(), log

            except Exception as exc:
                log.latency_ms = round((time.time() - t0) * 1000, 1) if 't0' in dir() else 0
                log.error_message = str(exc)[:200]

                if _is_transient(exc) and retry < max_retries - 1:
                    delay = base_retry_delay * (2 ** retry)
                    if isinstance(exc, RateLimitError):
                        delay = max(delay, 30.0)
                        retry_after = getattr(exc, "response", None)
                        if retry_after and hasattr(retry_after, "headers"):
                            ra = retry_after.headers.get("retry-after")
                            if ra:
                                try:
                                    delay = max(delay, float(ra))
                                except ValueError:
                                    pass
                    log.failure_type = _classify_failure(exc).value
                    logger.warning(
                        f"Transient error ({log.failure_type}), retrying in {delay:.0f}s "
                        f"(attempt {attempt}/{max_retries * len(models_to_try)}): {str(exc)[:100]}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    log.failure_type = _classify_failure(exc).value
                    logger.error(f"Non-transient or exhausted retries: {log.failure_type}: {str(exc)[:200]}")
                    break

    logger.error(log.log_summary())
    return None, log
