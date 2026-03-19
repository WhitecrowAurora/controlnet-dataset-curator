"""Minimal VLM client wrappers (local HTTP backends).

Supports:
- OpenAI-compatible /v1/chat/completions
- Ollama /api/chat
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional


def _read_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _http_post_json(url: str, payload: Dict, timeout: int = 120, headers: Optional[Dict] = None) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        if v is None:
            continue
        req.add_header(k, str(v))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


@dataclass
class VlmConfig:
    backend: str = "openai"  # openai | ollama
    base_url: str = "http://127.0.0.1:1234"
    api_key: str = ""
    model: str = ""
    timeout_seconds: int = 120


def build_rewrite_prompt(tags: str) -> str:
    return (
        "You are checking image tags for a dataset.\n"
        "Given the image and the tag list, do two things:\n"
        "1) Decide if the tags are plausible for the image.\n"
        "2) Rewrite into a short natural-language prompt.\n\n"
        "Return STRICT JSON with keys:\n"
        "- needs_review: boolean (true if tags look wrong/uncertain)\n"
        "- nl_prompt: string (natural language prompt)\n"
        "- reason: string (short reason)\n\n"
        f"tags: {tags}"
    )


def _extract_json_from_text(text: str) -> Dict:
    text = (text or "").strip()
    if not text:
        return {}
    # Try direct parse first.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find a JSON object block.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


class VlmClient:
    def __init__(self, cfg: VlmConfig):
        self.cfg = cfg

    def rewrite(self, image_path: str, tags: str) -> Dict:
        backend = (self.cfg.backend or "openai").lower()
        if backend == "ollama":
            return self._rewrite_ollama(image_path, tags)
        return self._rewrite_openai(image_path, tags)

    def _rewrite_openai(self, image_path: str, tags: str) -> Dict:
        url = (self.cfg.base_url or "").rstrip("/") + "/v1/chat/completions"
        b64 = _read_image_b64(image_path)

        messages = [
            {"role": "system", "content": "You output only JSON, no markdown."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_rewrite_prompt(tags)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            },
        ]

        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        payload = {
            "model": self.cfg.model or "local",
            "messages": messages,
            "temperature": 0.2,
        }

        data = _http_post_json(url, payload, timeout=int(self.cfg.timeout_seconds), headers=headers)
        text = (
            (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
            if isinstance(data, dict)
            else ""
        )
        parsed = _extract_json_from_text(text if isinstance(text, str) else json.dumps(text))
        return parsed or {"needs_review": True, "nl_prompt": "", "reason": "model_output_parse_failed"}

    def _rewrite_ollama(self, image_path: str, tags: str) -> Dict:
        url = (self.cfg.base_url or "").rstrip("/") + "/api/chat"
        b64 = _read_image_b64(image_path)
        payload = {
            "model": self.cfg.model or "llava",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": build_rewrite_prompt(tags),
                    "images": [b64],
                }
            ],
            "options": {"temperature": 0.2},
        }
        data = _http_post_json(url, payload, timeout=int(self.cfg.timeout_seconds))
        text = ((data.get("message") or {}).get("content") if isinstance(data, dict) else "")
        parsed = _extract_json_from_text(text if isinstance(text, str) else json.dumps(text))
        return parsed or {"needs_review": True, "nl_prompt": "", "reason": "model_output_parse_failed"}

