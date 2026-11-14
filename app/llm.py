"""Minimal client for Vultr Serverless Inference (OpenAI-compatible API).

Settings come from the .env file:
    INFERENCE_API_KEY=...            (from my.vultr.com -> Serverless Inference)
    INFERENCE_MODEL=...              (see: python -m app.llm models)
    INFERENCE_BASE_URL=https://api.vultrinference.com/v1   (optional)
"""
import os
import re
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("INFERENCE_BASE_URL", "https://api.vultrinference.com/v1").rstrip("/")
API_KEY = os.getenv("INFERENCE_API_KEY", "")
MODEL = os.getenv("INFERENCE_MODEL", "")
TIMEOUT = 60


class LLMError(Exception):
    pass


def _headers() -> dict:
    if not API_KEY:
        raise LLMError("Lipsește INFERENCE_API_KEY în fișierul .env")
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def list_models() -> list:
    r = requests.get(f"{BASE_URL}/models", headers=_headers(), timeout=TIMEOUT)
    if r.status_code != 200:
        raise LLMError(f"Eroare {r.status_code}: {r.text[:300]}")
    data = r.json().get("data", [])
    return [m.get("id") for m in data if m.get("id")]


def chat(messages: list, temperature: float = 0.2, max_tokens: int = 900) -> str:
    if not MODEL:
        raise LLMError("Lipsește INFERENCE_MODEL în fișierul .env (vezi: python -m app.llm models)")
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", headers=_headers(),
                          json=payload, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise LLMError(f"Nu pot contacta serviciul AI: {exc}") from exc
    if r.status_code != 200:
        raise LLMError(f"Eroare AI {r.status_code}: {r.text[:300]}")
    try:
        content = r.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"Răspuns neașteptat de la AI: {r.text[:300]}") from exc
    # Some reasoning models include their hidden thinking; remove it.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "models":
        for model_id in list_models():
            print(model_id)
    else:
        print(chat([{"role": "user", "content": "Spune 'salut' în română, într-un cuvânt."}]))
