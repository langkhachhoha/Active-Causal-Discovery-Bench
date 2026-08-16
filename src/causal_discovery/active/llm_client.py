"""Minimal, fully-instrumented OpenRouter chat client.

Deliberately *not* LiteLLM: we want the raw OpenRouter JSON so that
`usage.cost`, `usage.prompt_tokens_details.cached_tokens` and the serving
`provider` are recorded verbatim for the paper's cost/efficiency tables.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_ALIASES = {
    "qwen3-coder-30b": "qwen/qwen3-coder-30b-a3b-instruct",
    "qwen": "qwen/qwen3-coder-30b-a3b-instruct",
    "gpt-4o-mini": "openai/gpt-4o-mini-2024-07-18",
    "4o-mini": "openai/gpt-4o-mini-2024-07-18",
}

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


def resolve_model(name: str) -> str:
    key = name.strip()
    return MODEL_ALIASES.get(key, key)


def short_model_name(name: str) -> str:
    """Filesystem/CSV-friendly tag, e.g. `qwen3-coder-30b-a3b-instruct`."""
    return resolve_model(name).split("/")[-1]


def resolve_api_key(env_file: str | Path | None = None) -> str:
    """OpenRouter key. Accepts `OPENROUTER_API_KEY` or a `sk-or-` value in `OPENAI_API_KEY`."""
    if env_file:
        path = Path(env_file)
        if path.exists():
            from dotenv import load_dotenv

            load_dotenv(path, override=True)
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise RuntimeError(
        "No OpenRouter API key found. Set OPENROUTER_API_KEY (or OPENAI_API_KEY) "
        "in your .env file."
    )


class LLMError(RuntimeError):
    """Raised when a call cannot be completed or repaired."""


def coerce_int(value: Any, field: str) -> int:
    """Accept `3`, `"3"`, `"X3"`, `3.0` — small models are inconsistent about JSON types."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in {"X", "x"} and text[1:].isdigit():
            return int(text[1:])
        try:
            parsed = float(text)
        except ValueError:
            raise ValueError(f"{field} must be an integer, got {value!r}") from None
        if parsed.is_integer():
            return int(parsed)
    raise ValueError(f"{field} must be an integer, got {value!r}")


def coerce_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number, got a boolean")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            raise ValueError(f"{field} must be a number, got {value!r}") from None
    raise ValueError(f"{field} must be a number, got {value!r}")


@dataclass(slots=True)
class UsageTotals:
    calls: int = 0
    failed_calls: int = 0
    repair_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_sec: float = 0.0
    per_tag: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, tag: str, usage: dict[str, Any], latency: float) -> None:
        details = usage.get("prompt_tokens_details") or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        cached = int(details.get("cached_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or 0) or (prompt + completion)
        cost = float(usage.get("cost", 0.0) or 0.0)

        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cached_tokens += cached
        self.total_tokens += total
        self.cost_usd += cost
        self.latency_sec += latency

        bucket = self.per_tag.setdefault(
            tag, {"calls": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "cost_usd": 0.0}
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += prompt
        bucket["completion_tokens"] += completion
        bucket["cost_usd"] += cost

    def as_row(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "llm_failed_calls": self.failed_calls,
            "llm_repair_calls": self.repair_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "llm_latency_sec": round(self.latency_sec, 4),
        }


@dataclass(slots=True)
class ToolResponse:
    payload: dict[str, Any]
    usage: dict[str, Any]
    latency_sec: float
    provider: str
    repairs: int


class OpenRouterClient:
    """One client per episode, so usage totals are per-episode by construction."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 180,
        max_retries: int = 5,
        max_repairs: int = 2,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = resolve_model(model)
        self._api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.max_repairs = int(max_repairs)
        self.usage = UsageTotals()
        self._on_event = on_event
        self._session = requests.Session()

    # -- transport ---------------------------------------------------------- #
    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/acdb/active-causal-discovery",
                        "X-Title": "ACDB active experiment studies",
                    },
                    data=json.dumps(body),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(2.0 * attempt, 20.0))
                continue

            if response.status_code == 200:
                payload = response.json()
                if "error" in payload and not payload.get("choices"):
                    last_error = LLMError(f"provider error: {payload['error']}")
                    time.sleep(min(2.0 * attempt, 20.0))
                    continue
                return payload

            last_error = LLMError(f"HTTP {response.status_code}: {response.text[:400]}")
            if response.status_code not in _RETRYABLE_STATUS:
                break
            time.sleep(min(2.0 * attempt, 20.0))

        raise LLMError(f"OpenRouter call failed after {self.max_retries} attempts: {last_error}")

    # -- public API --------------------------------------------------------- #
    def call_tool(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tool: dict[str, Any],
        validate: Callable[[dict[str, Any]], None] | None = None,
        tag: str = "call",
        context: dict[str, Any] | None = None,
    ) -> ToolResponse:
        """Force exactly one call of `tool`, parse and validate it, repair on failure."""
        tool_name = tool["function"]["name"]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        repairs = 0
        last_error = ""

        while True:
            body = {
                "model": self.model,
                "messages": messages,
                "tools": [tool],
                "tool_choice": {"type": "function", "function": {"name": tool_name}},
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "usage": {"include": True},
            }
            started = time.perf_counter()
            try:
                raw = self._post(body)
            except LLMError:
                self.usage.failed_calls += 1
                self._log(
                    tag,
                    {"status": "transport_failed", "repairs": repairs, **(context or {})},
                )
                raise
            latency = time.perf_counter() - started

            usage = raw.get("usage") or {}
            self.usage.add(tag, usage, latency)
            provider = str(raw.get("provider", ""))

            error_message = ""
            payload: dict[str, Any] = {}
            try:
                payload = _extract_tool_arguments(raw, tool_name)
                if validate is not None:
                    validate(payload)
            except Exception as exc:  # noqa: BLE001
                error_message = f"{type(exc).__name__}: {exc}"

            self._log(
                tag,
                {
                    "status": "ok" if not error_message else "invalid",
                    "model": self.model,
                    "provider": provider,
                    "repairs": repairs,
                    "latency_sec": round(latency, 4),
                    "usage": usage,
                    "payload": payload if not error_message else None,
                    "error": error_message or None,
                    "raw_message": None if not error_message else _raw_message(raw),
                    **(context or {}),
                },
            )

            if not error_message:
                return ToolResponse(
                    payload=payload,
                    usage=usage,
                    latency_sec=latency,
                    provider=provider,
                    repairs=repairs,
                )

            last_error = error_message
            if repairs >= self.max_repairs:
                self.usage.failed_calls += 1
                raise LLMError(f"tool call could not be repaired ({tag}): {last_error}")

            repairs += 1
            self.usage.repair_calls += 1
            messages = messages[:2] + [
                {
                    "role": "user",
                    "content": (
                        "Your previous tool call was rejected with this error:\n"
                        f"{last_error}\n"
                        "Emit a corrected call to the same tool. Obey every field constraint. "
                        "Use 0-based integer variable indices only."
                    ),
                }
            ]

    def _log(self, tag: str, payload: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(f"llm_call:{tag}", payload)


def _raw_message(raw: dict[str, Any]) -> Any:
    try:
        return raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None


def _extract_tool_arguments(raw: dict[str, Any], tool_name: str) -> dict[str, Any]:
    choices = raw.get("choices") or []
    if not choices:
        raise LLMError("response has no choices")
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []

    arguments: str | dict[str, Any] | None = None
    if tool_calls:
        call = tool_calls[0]
        name = (call.get("function") or {}).get("name")
        if name != tool_name:
            raise LLMError(f"unexpected tool {name!r}, expected {tool_name!r}")
        arguments = (call.get("function") or {}).get("arguments")
    else:
        # Some providers ignore tool_choice and answer in content; salvage the JSON.
        arguments = message.get("content")
        if not arguments:
            raise LLMError("no tool call and no content in response")

    if isinstance(arguments, dict):
        return arguments
    parsed = _loads_lenient(str(arguments))
    if not isinstance(parsed, dict):
        raise LLMError(f"tool arguments must be a JSON object, got {type(parsed).__name__}")
    return parsed


def _loads_lenient(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start >= 0:
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError as exc:
            raise LLMError(f"could not parse JSON: {exc}") from exc
    raise LLMError("no JSON object found in model output")
