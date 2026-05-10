from __future__ import annotations

import json
import os
import time
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except Exception:  # noqa: BLE001
    retry = None
    stop_after_attempt = None
    wait_exponential = None

from .cache import JsonCache
from .rate_limit import RateLimiter
from src.utils.json_utils import parse_json_object
from src.utils.text_utils import deterministic_embedding, generate_local_answer


DEFAULT_PRICING_PER_1M = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


def _identity_retry(func):
    return func


def _retry_decorator():
    if retry is None:
        return _identity_retry
    return retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))


class OpenAIClient:
    def __init__(
        self,
        cache_dir: str | None = None,
        use_cache: bool = True,
        use_api: bool | None = None,
        pricing: dict[str, dict[str, float]] | None = None,
        rate_limit_seconds: float = 0.0,
    ):
        if load_dotenv is not None:
            load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_URL") or os.getenv("OPENAI_BASE_URL")
        self.use_api = bool(self.api_key) if use_api is None else bool(use_api and self.api_key)
        self.cache = JsonCache(cache_dir or os.getenv("CACHE_DIR", ".cache/openai"), enabled=use_cache)
        self.pricing = pricing or DEFAULT_PRICING_PER_1M
        self.rate_limiter = RateLimiter(rate_limit_seconds)
        self._client = None

        if self.use_api:
            try:
                from openai import OpenAI

                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
            except Exception:
                self.use_api = False
                self._client = None

    def complete(
        self,
        prompt: str,
        model: str = "gpt-4.1-mini",
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 600,
        json_mode: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "kind": "chat",
            "model": model,
            "system": system,
            "prompt": prompt,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "json_mode": json_mode,
            "metadata": metadata or {},
        }
        cached = self.cache.get(payload)
        if cached is not None:
            cached["cached"] = True
            return cached

        started = time.time()
        if self.use_api and self._client is not None:
            result = self._complete_api(payload)
        else:
            result = self._complete_local(prompt, model, json_mode=json_mode)
        result["latency_seconds"] = time.time() - started
        result["cached"] = False
        self.cache.set(payload, result)
        return result

    @_retry_decorator()
    def _complete_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.rate_limiter.wait()
        messages = []
        if payload.get("system"):
            messages.append({"role": "system", "content": payload["system"]})
        messages.append({"role": "user", "content": payload["prompt"]})
        kwargs = {
            "model": payload["model"],
            "messages": messages,
            "temperature": payload["temperature"],
            "max_tokens": payload["max_output_tokens"],
        }
        if payload.get("json_mode"):
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
            "output_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
            "total_tokens": getattr(response.usage, "total_tokens", 0) if response.usage else 0,
        }
        return {
            "text": text,
            "json": parse_json_object(text) if payload.get("json_mode") else None,
            "usage": usage,
            "estimated_cost_usd": self.estimate_cost(payload["model"], usage),
            "model": payload["model"],
        }

    def _complete_local(self, prompt: str, model: str, json_mode: bool = False) -> dict[str, Any]:
        if json_mode:
            if "task_success_score" in prompt:
                data = {
                    "task_success_score": 0.75,
                    "memory_use_score": 0.75,
                    "harmfulness_score": 0.0,
                    "passes": True,
                    "explanation": "Deterministic fallback judge.",
                }
            elif "likely_useful" in prompt:
                prompt_l = prompt.lower()
                useful = "harmful" not in prompt_l and "poison" not in prompt_l and "irrelevant" not in prompt_l
                data = {"likely_useful": useful, "reason": "Deterministic lexical fallback.", "risk": "low" if useful else "high"}
            elif "perturbed_memory" in prompt:
                data = {"perturbed_memory": "Possibly unreliable memory.", "perturbation_explanation": "Deterministic fallback perturbation."}
            else:
                data = {}
            text = json.dumps(data)
            parsed = data
        else:
            text = generate_local_answer(prompt, [])
            parsed = None
        usage = self._estimate_usage(prompt, text)
        return {
            "text": text,
            "json": parsed,
            "usage": usage,
            "estimated_cost_usd": 0.0,
            "model": model,
        }

    def json_complete(self, prompt: str, **kwargs) -> dict[str, Any]:
        result = self.complete(prompt, json_mode=True, **kwargs)
        return result.get("json") or parse_json_object(result.get("text", ""))

    def embed(self, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
        payload = {"kind": "embedding", "model": model, "texts": texts}
        cached = self.cache.get(payload)
        if cached is not None:
            return cached["embeddings"]
        if self.use_api and self._client is not None:
            embeddings = self._embed_api(texts, model)
        else:
            embeddings = [deterministic_embedding(text) for text in texts]
        self.cache.set(payload, {"embeddings": embeddings})
        return embeddings

    @_retry_decorator()
    def _embed_api(self, texts: list[str], model: str) -> list[list[float]]:
        self.rate_limiter.wait()
        response = self._client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]

    def estimate_cost(self, model: str, usage: dict[str, int]) -> float:
        prices = self.pricing.get(model, {"input": 0.0, "output": 0.0})
        return (
            usage.get("input_tokens", 0) * prices.get("input", 0.0)
            + usage.get("output_tokens", 0) * prices.get("output", 0.0)
        ) / 1_000_000

    @staticmethod
    def _estimate_usage(prompt: str, text: str) -> dict[str, int]:
        input_tokens = max(1, len(prompt.split()))
        output_tokens = max(1, len(text.split()))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
