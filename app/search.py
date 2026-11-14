"""BM25 keyword search over the indexed Bac documents (no external services)."""
import json
import math
from collections import Counter
from pathlib import Path

from app.text_utils import tokenize

CHUNKS_FILE = Path(__file__).resolve().parent.parent / "index" / "chunks.json"


class BM25Index:
    def __init__(self, chunks: list, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 6, subject=None, year=None, doc_type=None) -> list:
        terms = tokenize(query)
        if not terms or not self.docs:
            return []
        scored = []
        for i, doc in enumerate(self.docs):
            c = self.chunks[i]
            if subject and c["subject"] != subject:
                continue
            if year and c["year"] != year:
                continue
            if doc_type and c["doc_type"] != doc_type:
                continue
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    score += self.idf.get(t, 0) * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [{**self.chunks[i], "score": round(s, 3)} for s, i in scored[:k]]


def load_index() -> BM25Index:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("Nu există index/chunks.json. Rulează întâi: python -m app.ingest")
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return BM25Index(chunks)


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "derivata functiei"
    index = load_index()
    print(f"Întrebare: {question}\n")
    for r in index.search(question, k=5):
        print(f"[{r['score']}] {r['label']}, pagina {r['page']}")
        print("   ", r["text"][:200].replace("\n", " "), "...\n")