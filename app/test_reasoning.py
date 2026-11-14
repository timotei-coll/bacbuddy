"""Tries different ways to limit the model's hidden 'thinking'. Run: python -m app.test_reasoning"""
import time

import requests

from app import llm

PROMPT = ("Rescrie clar acest exercițiu de Bac și dă un indiciu scurt.\n"
          "Format: PROBLEM: ...\nHINT: ...\n\nExercițiu: 5p 1. Arătați că ( )1 0 f − = , unde f = X^3 + aX + b, a=4, b=0.")
VARIANTS = {
    "fara parametri": {},
    "reasoning_effort=low": {"reasoning_effort": "low"},
    "reasoning.effort=low": {"reasoning": {"effort": "low"}},
    "thinking dezactivat": {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}},
}

for name, extra in VARIANTS.items():
    body = {"model": llm.MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 1500, "temperature": 0.3, **extra}
    start = time.time()
    r = requests.post(f"{llm.BASE_URL}/chat/completions", headers=llm._headers(), json=body, timeout=90)
    secs = time.time() - start
    if r.status_code != 200:
        print(f"[{name}] EROARE {r.status_code}: {r.text[:150]}")
        continue
    data = r.json()
    usage = data.get("usage", {})
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    content = (data["choices"][0]["message"].get("content") or "").strip()
    print(f"[{name}] {secs:.1f}s | reasoning_tokens={reasoning} | "
          f"finish={data['choices'][0].get('finish_reason')} | content: {content[:120]!r}")