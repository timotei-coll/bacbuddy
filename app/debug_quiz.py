"""Shows the raw AI response for one quiz problem (for debugging). Run: python -m app.debug_quiz"""
import json
import random

import requests

from app import llm, rag

chunks = [c for c in rag.INDEX.chunks if c["doc_type"] == "subiect" and c["subject"] == "matematica"]
exercise = random.choice(rag.split_exercises(random.choice(chunks)["text"]))
print("EXERCISE:", exercise[:300], "\n")
prompt = rag.QUIZ_PROMPT.format(label="test", language="Romanian", text=exercise)
r = requests.post(f"{llm.BASE_URL}/chat/completions", headers=llm._headers(), timeout=90,
                  json={"model": llm.MODEL, "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1500, "temperature": 0.3})
data = r.json()
choice = data["choices"][0]
msg = choice["message"]
print("STATUS:", r.status_code, "| FINISH:", choice.get("finish_reason"), "| USAGE:", data.get("usage"))
print("MESSAGE KEYS:", list(msg.keys()))
print("CONTENT:", repr((msg.get("content") or "")[:700]))
for key in ("reasoning_content", "reasoning"):
    if msg.get(key):
        print(f"{key.upper()} (first 300):", repr(str(msg[key])[:300]))