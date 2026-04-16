from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

from openai import OpenAI


class VLLMChatClient:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0, timeout: float = 300.0):
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("all_proxy", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature

    def list_models(self) -> List[str]:
        response = self.client.models.list()
        return [m.id for m in response.data]

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 160) -> Tuple[str, float]:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        content = response.choices[0].message.content or ""
        return content.strip(), latency_ms
