"""BM25 keyword search over the indexed Bac documents (no external services)."""
import json
import math
from collections import Counter
from pathlib import Path

from app.text_utils import tokenize

CHUNKS_FILE = Path(__file__).resolve().parent.parent / "index" / "chunks.json"
VECTORS_FILE = Path(__file__).resolve().parent.parent / "index" / "vectors.npy"


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


def _matches(c: dict, subject, year, doc_type) -> bool:
    return ((not subject or c["subject"] == subject) and (not year or c["year"] == year)
            and (not doc_type or c["doc_type"] == doc_type))


def load_vectors(n_chunks: int):
    """Load normalized embedding matrix, or None if hybrid search isn't available."""
    if not VECTORS_FILE.exists():
        return None
    import numpy as np
    matrix = np.load(VECTORS_FILE)
    return matrix if matrix.shape[0] == n_chunks else None


def vector_search(matrix, chunks: list, query_vector, k: int = 20, subject=None, year=None,
                  doc_type=None) -> list:
    import numpy as np
    q = np.asarray(query_vector, dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-12
    scores = matrix @ q
    order = np.argsort(-scores)
    out = []
    for i in order:
        c = chunks[int(i)]
        if _matches(c, subject, year, doc_type):
            out.append({**c, "score": round(float(scores[i]), 3)})
            if len(out) >= k:
                break
    return out


def fuse(*result_lists, k: int = 6, rrf_k: int = 60) -> list:
    """Reciprocal Rank Fusion: combine rankings from keyword and semantic search."""
    scores, items = {}, {}
    for results in result_lists:
        for rank, r in enumerate(results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
            items.setdefault(r["id"], r)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**items[i], "score": round(scores[i], 4)} for i in best]


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