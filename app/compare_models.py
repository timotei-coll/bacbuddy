"""Ask the same real question to several AI models and compare quality and speed.

Run:  python -m app.compare_models
"""
import sys
import time

from app import llm, rag

CANDIDATES = [
    "glm-5.3",
    "deepseek-v4.1-flash-1017",
    "qwen3.8-27b",
    "mimo-v2.6-pro",
]
QUESTION = "Cum se punctează studiul monotoniei unei funcții la Bacalaureat?"

if __name__ == "__main__":
    models = sys.argv[1:] or CANDIDATES
    for model in models:
        llm.MODEL = model
        print("=" * 70)
        print(f"MODEL: {model}")
        start = time.time()
        try:
            result = rag.ask(QUESTION, lang="ro")
            seconds = time.time() - start
            print(f"Timp: {seconds:.1f}s | Surse citate: {len(result['sources'])} | Găsit: {result['found']}")
            print("Surse:", "; ".join(f"{x['label']} p.{x['page']}" for x in result["sources"]) or "-")
            print(result["answer"][:900])
        except llm.LLMError as exc:
            print(f"EROARE: {exc}")
    print("=" * 70)
