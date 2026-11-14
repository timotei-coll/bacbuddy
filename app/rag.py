"""Retrieval-augmented answers and quiz generation, grounded in official Bac documents."""
import json
import random
import re
from pathlib import Path
from urllib.parse import quote

from app import llm
from app.search import fuse, load_index, load_vectors, vector_search
from app.text_utils import strip_diacritics

INDEX = load_index()
_PAGES_FILE = Path(__file__).resolve().parent.parent / "index" / "exam_pages.json"
EXAM_PAGES = json.loads(_PAGES_FILE.read_text(encoding="utf-8")) if _PAGES_FILE.exists() else []
VECTORS = load_vectors(len(INDEX.chunks))   # None -> keyword search only
QUERY_INSTRUCT = "Instruct: Given a student's question about the Romanian Baccalaureate exam, " \
                 "retrieve official curriculum, exam or grading-scheme passages that answer it\nQuery: "
TOP_K = 6
MIN_SCORE = 1.0  # below this, sources are too weak to answer from

NOT_FOUND = {
    "ro": "Nu am găsit informația asta în materialele oficiale de Bacalaureat pe care le am "
          "(programe, subiecte și bareme la Matematică M_mate-info și Informatică, 2024–2026). "
          "Încearcă să reformulezi întrebarea sau să alegi altă materie/an.",
    "en": "I couldn't find this in the official Baccalaureate materials I have "
          "(curricula, exam papers and grading schemes for Math M_mate-info and Computer Science, "
          "2024–2026). Try rephrasing or choosing another subject/year.",
}

SYSTEM_PROMPT = """You are BacBuddy, a study assistant for Romania's national Baccalaureate exam \
(Mathematics M_mate-info and Computer Science, C/C++).

STRICT RULES:
1. Answer ONLY using the numbered SOURCES provided. Never use outside knowledge to state facts \
about the exam, its content, points or grading.
2. After every claim, cite the source number in square brackets, e.g. [1] or [2][3].
3. If the sources answer the question only partly, answer the part they cover and say clearly \
what the official materials do not specify. Reply exactly with NOT_FOUND only if the sources are \
completely unrelated to the question.
4. When asked about grading, quote the exact points (e.g. "3p") from the grading scheme (barem).
5. Text was extracted from PDFs, so math formulas may look scrambled. You may tidy obvious \
notation, but never invent numbers, formulas or points that are not in the sources. If a formula \
is unreadable, say so and point the student to the cited source.
6. Be clear and student-friendly. Use short paragraphs or bullet points. Max ~200 words.
7. Write math in LaTeX between single dollar signs, e.g. $f'(x)=2x$.
8. Write your answer in {language}."""


def _source_url(chunk: dict) -> str:
    return f"/data/{quote(chunk['file'])}#page={chunk['page']}"


def _format_sources(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] {c['label']}, pagina {c['page']}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _public_source(i: int, c: dict) -> dict:
    snippet = re.sub(r"\s+", " ", c["text"]).strip()
    return {
        "n": i,
        "label": c["label"],
        "page": c["page"],
        "url": _source_url(c),
        "snippet": snippet[:280] + ("…" if len(snippet) > 280 else ""),
        "doc_type": c["doc_type"],
        "year": c["year"],
    }


EXPAND_PROMPT = """A student asks about the Romanian Baccalaureate exam (Math M_mate-info or \
Computer Science C/C++). Write 8-15 Romanian search keywords that would appear in the official \
curriculum (programa), exam papers (subiecte) or grading schemes (bareme) relevant to the question. \
Include synonyms and the concrete math/CS terms used in exam papers (e.g. for "monotonie": \
crescătoare descrescătoare derivata semnul). Output ONLY the keywords separated by spaces.

QUESTION: {question}"""

GRADING_WORDS = ("punct", "barem", "noteaz", "notare", "grad", "point", "score", "mark")


def expand_query(question: str) -> str:
    """Ask the model for extra search terms (query rewriting). Falls back silently."""
    try:
        extra = llm.chat([{"role": "user", "content": EXPAND_PROMPT.format(question=question)}],
                         temperature=0.0, max_tokens=300)
        extra = re.sub(r"[^\w\s\-]", " ", extra)[:300]
        return f"{question} {extra}"
    except llm.LLMError:
        return question


MATH_HINTS = ("functi", "derivat", "integral", "primitiv", "limit", "matrice", "determinant",
              "ecuati", "logaritm", "monoton", "asimptot", "continu", "polinom", "progresi",
              "trigonometr", "geometri", "probabilit", "complex", "lege de compozitie",
              "extrem", "convex", "concav", "grafic", "sistem de ecuatii", "m1", "matemat",
              "math", "calculus", "derivative", "function")
INFO_HINTS = ("algoritm", "programul", "subprogram", "c++", "pseudocod", "tablou", "graf ",
              "graful", "grafuri", "computer science", "coding", "algorithm", "sorting",
              "arbore", "recursiv", "sortare", "cautare binara", "fisier", "struct",
              "variabil", "backtracking", "sir de caractere", "instructiun", "cod ",
              "informatic", "bubble", "selectie", "interclasare", "stiva", "coada")


def detect_subject(question: str):
    """Guess the subject from the question when the student didn't pick one."""
    q = strip_diacritics(question.lower()) + " "
    math = sum(h in q for h in MATH_HINTS)
    info = sum(h in q for h in INFO_HINTS)
    if math > info:
        return "matematica"
    if info > math:
        return "informatica"
    return None


def _hybrid(query: str, qvec, k: int, subject, year, doc_type=None) -> list:
    keyword = [h for h in INDEX.search(query, k=20, subject=subject, year=year, doc_type=doc_type)
               if h["score"] >= MIN_SCORE]
    if qvec is None:
        return keyword[:k]
    semantic = vector_search(VECTORS, INDEX.chunks, qvec, k=20, subject=subject, year=year,
                             doc_type=doc_type)
    return fuse(keyword, semantic, k=k)


def retrieve(question: str, subject=None, year=None) -> list:
    """Hybrid retrieval: LLM query expansion + BM25 keywords + semantic embeddings (RRF)."""
    subject = subject or detect_subject(question)
    query = expand_query(question)
    qvec = None
    if VECTORS is not None:
        try:
            qvec = llm.embed([QUERY_INSTRUCT + question])[0]
        except llm.LLMError as exc:
            print("[embeddings unavailable, keyword search only]", exc)
    hits = _hybrid(query, qvec, TOP_K, subject, year)
    # Grading questions must see the actual grading schemes (bareme), not only the curriculum.
    if any(w in question.lower() for w in GRADING_WORDS):
        barem_hits = _hybrid(query, qvec, 3, subject, year, doc_type="barem")
        ids = {h["id"] for h in hits}
        new = [b for b in barem_hits if b["id"] not in ids]
        hits = hits[:TOP_K - len(new)] + new
    return hits


def ask(question: str, subject=None, year=None, lang: str = "ro") -> dict:
    lang = "en" if lang == "en" else "ro"
    hits = retrieve(question, subject=subject, year=year)
    if not hits:
        return {"answer": NOT_FOUND[lang], "sources": [], "found": False}

    language = "English" if lang == "en" else "Romanian"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=language)},
        {"role": "user", "content": f"SOURCES:\n\n{_format_sources(hits)}\n\nQUESTION: {question}"},
    ]
    answer = llm.chat(messages)
    if not answer or "NOT_FOUND" in answer[:40]:
        return {"answer": NOT_FOUND[lang], "sources": [], "found": False}

    # Return only the sources the model actually cited (fallback: all).
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(hits)})
    numbers = cited or list(range(1, len(hits) + 1))
    sources = [_public_source(n, hits[n - 1]) for n in numbers]
    return {"answer": answer, "sources": sources, "found": True}


QUIZ_PROMPT = """You are BacBuddy. Below is ONE exercise taken from an official Romanian \
Baccalaureate exam paper ({label}). Present it to the student as a single practice problem.

Rules:
- If the exercise has several sub-questions (a), b), c)), keep the shared statement and ONLY the \
first sub-question. Keep multiple-choice options (a. b. c. d.) intact.
- Copy it faithfully; only tidy formatting. Do not invent new data or numbers.
- If formulas are scrambled from PDF extraction, rewrite them in clear math notation only when \
the meaning is unambiguous.
- Write math in LaTeX between single dollar signs, e.g. $\\lim_{{x\\to 0}} f(x)$.
- Then add one short hint (without solving it).
- Output format exactly:
PROBLEM: <the exercise>
HINT: <one short hint>
- Write in {language}.

EXERCISE:
{text}"""

# Exercises in Bac papers start on a new line with an optional score and a number, e.g. "5p 2. ..."
_EXERCISE_START = re.compile(r"\n(?=\s*(?:\d{1,2}\s*p\s*)?\d{1,2}\.\s)")


def split_exercises(text: str) -> list:
    """Cut a fragment of an exam paper into separate numbered exercises."""
    text = re.sub(r"(?im)^\s*SUBIECTUL.*$", "", text)   # drop section headers
    parts = [p.strip() for p in _EXERCISE_START.split("\n" + text) if p.strip()]
    # keep real exercises: they start with a number (not mid-sentence) and have a sane length
    good = [p for p in parts if 60 <= len(p) <= 1500
            and re.match(r"(?:\d{1,2}\s*p\s*)?\d{1,2}\.\s", p)]
    return good


def quiz(subject=None, lang: str = "ro") -> dict:
    lang = "en" if lang == "en" else "ro"
    # Use whole exam pages (not search chunks) so exercises are never cut in half.
    pool = [p for p in (EXAM_PAGES or INDEX.chunks)
            if p["doc_type"] == "subiect" and (not subject or p["subject"] == subject)]
    candidates = [(p, ex) for p in pool for ex in split_exercises(p["text"])]
    if not candidates:
        return {"error": "no_subjects"}
    chunk, exercise = random.choice(candidates)
    language = "English" if lang == "en" else "Romanian"
    raw = llm.chat([{"role": "user", "content": QUIZ_PROMPT.format(
        label=chunk["label"], language=language, text=exercise)}], temperature=0.3, max_tokens=1500)

    raw = raw.replace("**", "")
    problem, hint = raw.strip(), ""
    m = re.search(r"PROBLEM:\s*(.*?)\s*HINT:\s*(.*)", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        problem, hint = m.group(1).strip(), m.group(2).strip()
    else:
        problem = re.sub(r"^\s*PROBLEM:\s*", "", problem, flags=re.IGNORECASE)
    if len(problem) < 15:          # empty/odd model output -> show the official text as is
        problem, hint = exercise, ""
    return {
        "problem": problem,
        "hint": hint,
        "source": _public_source(1, chunk),
        "quiz_ref": {"subject": chunk["subject"], "year": chunk["year"],
                     "session": chunk["session"], "problem": exercise[:600]},
    }


def quiz_solution(ref: dict, lang: str = "ro") -> dict:
    """Find the official grading scheme (barem) from the same exam session and explain it."""
    lang = "en" if lang == "en" else "ro"
    candidates = [c for c in INDEX.chunks
                  if c["doc_type"] == "barem" and c["subject"] == ref.get("subject")
                  and c["year"] == ref.get("year") and c["session"] == ref.get("session")]
    if not candidates:
        return {"answer": NOT_FOUND[lang], "sources": [], "found": False}
    # Rank the barem pages of that session by similarity to the problem text.
    from app.search import BM25Index
    ranked = BM25Index(candidates).search(ref.get("problem", ""), k=3) or candidates[:3]
    language = "English" if lang == "en" else "Romanian"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=language)},
        {"role": "user", "content":
            f"SOURCES:\n\n{_format_sources(ranked)}\n\nQUESTION: Using the official grading scheme, "
            f"show how this exercise is solved and how many points each step gets:\n{ref.get('problem', '')}"},
    ]
    answer = llm.chat(messages, max_tokens=1500)
    if not answer or "NOT_FOUND" in answer[:40]:
        return {"answer": NOT_FOUND[lang], "sources": [_public_source(i, c) for i, c in enumerate(ranked, 1)],
                "found": False}
    return {"answer": answer, "sources": [_public_source(i, c) for i, c in enumerate(ranked, 1)],
            "found": True}