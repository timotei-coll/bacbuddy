"""Quiz me: practice problems from official Bac papers, answer checking against the official
grading scheme (barem), and the official solution.

Flow:
1. pick an exercise that asks for a final answer (Calculați / Determinați / Indicați ...)
2. ONE AI call turns it into a clean problem + hint and reads the expected answer from the barem
3. the expected answer stays on the server; the student's answer is checked in two layers:
   exact match after normalization, then an AI equivalence check (1/2 == 0.5, {3,1} == {1,3})
"""
import hashlib
import json
import random
import re
import threading
import time
import uuid

from app import llm
from app.rag import EXAM_PAGES, INDEX, NOT_FOUND, SYSTEM_PROMPT, _format_sources, _public_source
from app.search import BM25Index
from app.text_utils import strip_diacritics

# ---------------------------------------------------------------- exercise selection

_EXERCISE_START = re.compile(r"\n(?=\s*(?:\d{1,2}\s*p\s*)?\d{1,2}\.\s)")
# verbs that ask for a result we can check (diacritics removed, lowercase)
_ASKS_FOR_ANSWER = re.compile(
    r"\b(calcula|determina|rezolva|afla|gasi|indica|care (este|sunt)|cate |cati |"
    r"scrieti (valoarea|valorile|numarul|cel |cea |ce se afiseaza|sirul|toate))")


def split_exercises(text: str) -> list:
    """Cut an exam page into separate numbered exercises."""
    text = re.sub(r"(?im)^\s*SUBIECTUL.*$", "", text)   # drop section headers
    parts = [p.strip() for p in _EXERCISE_START.split("\n" + text) if p.strip()]
    return [p for p in parts if 60 <= len(p) <= 1500
            and re.match(r"(?:\d{1,2}\s*p\s*)?\d{1,2}\.\s", p)]


# tasks that ask the student to WRITE code / an algorithm have no single final answer
_WRITE_CODE = re.compile(
    r"(scrieti (definitia|un program|programul|un subprogram|subprogramul|algoritmul|pseudocodul|"
    r"un algoritm)|subprogramul \w+ are|proiectati|descrieti in limbaj|fisierul \w+\.(txt|in|out))")


# explicit "give a result" requests; if present, the AI can pick that sub-question even when
# another sub-question of the same exercise asks for a program
_CLEAR_RESULT = re.compile(
    r"\b(calculati|determinati|rezolvati|indicati|scrieti (valoarea|valorile|numarul|ce se afiseaza))")


def answerable(exercise: str) -> bool:
    t = strip_diacritics(exercise.lower())
    if _CLEAR_RESULT.search(t):
        return True
    return bool(_ASKS_FOR_ANSWER.search(t)) and not _WRITE_CODE.search(t)


# ---------------------------------------------------------------- quiz generation

QUIZ_PROMPT = """You are BacBuddy. Below is ONE exercise from an official Romanian Baccalaureate \
exam paper ({label}) and excerpts from the OFFICIAL GRADING SCHEME (barem) of the same exam.

Your job:
1. Choose ONE sub-question that asks the student to compute, determine, solve or choose something \
with a short FINAL ANSWER (a number, an expression, a set of values, or a multiple-choice letter). \
Do NOT choose "Arătați că" / "Demonstrați" sub-questions or "write a program" tasks.
2. Present it as a single practice problem: keep the shared statement + only that sub-question. \
Keep multiple-choice options (a. b. c. d.) intact. Copy faithfully; only tidy formatting. Rewrite \
scrambled formulas in clear LaTeX between $...$ only when the meaning is unambiguous.
3. Give one short hint (without solving it).
4. Read the expected FINAL ANSWER for that sub-question from the grading scheme. If the grading \
scheme excerpts don't clearly contain it, write NONE. Never compute it yourself.

Output format exactly:
PROBLEM: <the problem>
HINT: <one short hint>
ANSWER: <final answer only, e.g. x=3 or {{1,2}} or b or NONE>

Write PROBLEM and HINT in {language}.

EXERCISE:
{exercise}

GRADING SCHEME EXCERPTS:
{barem}"""

_store = {}                 # quiz_id -> data (expected answer never leaves the server)
_store_lock = threading.Lock()
STORE_TTL = 3 * 3600
STORE_MAX = 5000


def _remember(data: dict) -> str:
    quiz_id = uuid.uuid4().hex
    now = time.time()
    with _store_lock:
        if len(_store) > STORE_MAX:
            for key in [k for k, v in _store.items() if now - v["created"] > STORE_TTL]:
                _store.pop(key, None)
        _store[quiz_id] = {**data, "created": now, "answered": False}
    return quiz_id


def _barem_for(page: dict, exercise: str, k: int = 3) -> list:
    """Grading-scheme chunks from the same exam session, ranked by similarity to the exercise."""
    candidates = [c for c in INDEX.chunks
                  if c["doc_type"] == "barem" and c["subject"] == page["subject"]
                  and c["year"] == page["year"] and c["session"] == page["session"]]
    if not candidates:
        return []
    return BM25Index(candidates).search(exercise, k=k) or candidates[:k]


def _parse(raw: str) -> dict:
    raw = raw.replace("**", "")
    out = {}
    for key, nxt in (("PROBLEM", "HINT"), ("HINT", "ANSWER"), ("ANSWER", None)):
        pattern = rf"{key}:\s*(.*?)\s*(?={nxt}:|$)" if nxt else rf"{key}:\s*(.*)"
        m = re.search(pattern, raw, flags=re.DOTALL | re.IGNORECASE)
        out[key.lower()] = m.group(1).strip() if m else ""
    answer = out["answer"].splitlines()[0].strip() if out["answer"] else ""
    out["answer"] = "" if answer.upper().startswith("NONE") else answer.strip(" .$")
    return out


def exercise_key(exercise: str) -> str:
    """Stable id of an exercise text (the same exercise in two papers gets the same id)."""
    norm = re.sub(r"\s+", " ", strip_diacritics(exercise.lower())).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def new_quiz(subject=None, lang: str = "ro", seen=None, attempts: int = 4) -> dict:
    lang = "en" if lang == "en" else "ro"
    pages = [p for p in (EXAM_PAGES or INDEX.chunks)
             if p["doc_type"] == "subiect" and (not subject or p["subject"] == subject)]
    unique = {}
    for p in pages:
        for ex in split_exercises(p["text"]):
            if answerable(ex):
                unique.setdefault(exercise_key(ex), (p, ex))   # drop duplicates across papers
    if not unique:
        return {"error": "no_subjects"}
    seen = set(seen or [])
    fresh = [k for k in unique if k not in seen]
    keys = fresh or list(unique)          # everything seen -> start a new round
    random.shuffle(keys)
    language = "English" if lang == "en" else "Romanian"

    best = None
    for key in keys[:attempts]:
        page, exercise = unique[key]
        barem = _barem_for(page, exercise)
        raw = llm.chat([{"role": "user", "content": QUIZ_PROMPT.format(
            label=page["label"], language=language, exercise=exercise,
            barem=_format_sources(barem) if barem else "(not available)")}],
            temperature=0.2, max_tokens=1500)
        parsed = _parse(raw)
        if len(parsed["problem"]) < 15:
            parsed["problem"] = exercise
        best = (key, page, exercise, barem, parsed)
        if parsed["answer"]:
            break

    key, page, exercise, barem, parsed = best
    ref = {"subject": page["subject"], "year": page["year"], "session": page["session"],
           "problem": exercise[:600]}
    quiz_id = _remember({"expected": parsed["answer"], "problem": parsed["problem"], "ref": ref,
                         "barem_ids": [c["id"] for c in barem], "lang": lang})
    return {
        "quiz_id": quiz_id,
        "exercise_key": key,
        "remaining": max(0, len(fresh) - 1),
        "problem": parsed["problem"],
        "hint": parsed["hint"],
        "checkable": bool(parsed["answer"]),
        "source": _public_source(1, page),
        "quiz_ref": ref,
    }


# ---------------------------------------------------------------- answer checking

def normalize_answer(text: str, ordered: bool = False) -> str:
    """Make harmless differences disappear: spaces, case, diacritics, 'x =', decimal comma,
    unicode minus, braces/brackets and the order of values in a set."""
    t = strip_diacritics(str(text).lower()).strip()
    t = t.replace("−", "-").replace("–", "-").replace("$", "").replace("\\", "")
    t = re.sub(r"^(raspuns(ul)?|answer|r)\s*[:=-]\s*", "", t)
    t = re.sub(r"^[a-z]\d*\s*(=|\u2208|in\b)\s*", "", t)        # "x = 3", "m ∈ {2,3}" -> value
    t = re.sub(r"\s+(sau|si|or|and)\s+", ";", t)               # "x=1 sau x=9" -> list
    t = re.sub(r"(?<=\d),(?=\d)", ".", t) if not re.search(r"[{};]", t) else t
    t = re.sub(r"\s+", "", t).strip(".;")
    if re.fullmatch(r"[a-d][.)]?(.*)?", t) and len(t) <= 2:   # "b)" / "b." -> "b"
        t = t[0]
    if "frac" not in t and (re.search(r"[{};]", t) or t.count(",") >= 1):   # sets / lists
        items = [i for i in re.split(r"[;,]", t.strip("{}[]()")) if i]
        items = [re.sub(r"^[a-z]\d*=", "", i) for i in items]
        t = "{" + ",".join(items if ordered else sorted(items)) + "}"
    return t


def _choice_letter(text: str):
    m = re.fullmatch(r"\s*([a-dA-D])(?:\s*[.)]|\s|$).*", text.strip(), flags=re.DOTALL) \
        if len(text.strip()) <= 60 else None
    return m.group(1).lower() if m else None


CHECK_PROMPT = """Compare a student's final answer with the official answer for a Bac exercise.
Decide if they are mathematically/logically EQUIVALENT (e.g. 1/2 and 0.5, x=2 and 2, {{1,3}} and \
{{3,1}}, sqrt(4) and 2). Ignore formatting. If the official answer has several values, the student \
must give all of them. If the exercise says "respectiv", the values must be in the same order.

EXERCISE: {problem}
OFFICIAL ANSWER: {expected}
STUDENT ANSWER: {student}

Reply with JSON only: {{"correct": true or false, "explanation": "<one short sentence in {language}>"}}"""


def check_answer(quiz_id: str, student: str) -> dict:
    with _store_lock:
        quiz = _store.get(quiz_id)
        if quiz is None:
            return {"error": "expired"}
        if quiz["answered"]:
            return {"error": "already_answered"}
        if not quiz["expected"]:
            return {"error": "not_checkable"}
        quiz["answered"] = True
    expected, lang = quiz["expected"], quiz["lang"]
    language = "English" if lang == "en" else "Romanian"
    student = str(student or "").strip()[:200]

    # Layer 1: exact match after normalization (also multiple-choice letters).
    exp_letter = _choice_letter(expected) if len(expected.strip()) <= 3 else None
    if exp_letter:
        correct = _choice_letter(student) == exp_letter
        return {"correct": correct, "expected": expected, "method": "letter",
                "explanation": ""}
    # "Scrieți f(2), respectiv f(17)" -> the order of the values matters
    ordered = "respectiv" in strip_diacritics(quiz["problem"].lower())
    if normalize_answer(student, ordered) == normalize_answer(expected, ordered):
        return {"correct": True, "expected": expected, "method": "exact", "explanation": ""}

    # Layer 2: AI equivalence check.
    try:
        raw = llm.chat([{"role": "user", "content": CHECK_PROMPT.format(
            problem=quiz["problem"][:800], expected=expected, student=student, language=language)}],
            temperature=0.0, max_tokens=200)
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        correct = bool(data.get("correct"))
        explanation = str(data.get("explanation", ""))[:300]
    except (llm.LLMError, ValueError):
        with _store_lock:
            quiz["answered"] = False      # let the student try again
        raise llm.LLMError("check failed")
    return {"correct": correct, "expected": expected, "method": "ai", "explanation": explanation}


# ---------------------------------------------------------------- official solution

def solution(ref: dict, lang: str = "ro") -> dict:
    """Explain the official grading scheme (barem) of the same exam session."""
    lang = "en" if lang == "en" else "ro"
    page = {"subject": ref.get("subject"), "year": ref.get("year"), "session": ref.get("session")}
    ranked = _barem_for(page, ref.get("problem", ""))
    if not ranked:
        return {"answer": NOT_FOUND[lang], "sources": [], "found": False}
    language = "English" if lang == "en" else "Romanian"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=language)},
        {"role": "user", "content":
            f"SOURCES:\n\n{_format_sources(ranked)}\n\nQUESTION: Using the official grading scheme, "
            f"show how this exercise is solved and how many points each step gets:\n{ref.get('problem', '')}"},
    ]
    answer = llm.chat(messages, max_tokens=1500)
    sources = [_public_source(i, c) for i, c in enumerate(ranked, 1)]
    if not answer or "NOT_FOUND" in answer[:40]:
        return {"answer": NOT_FOUND[lang], "sources": sources, "found": False}
    return {"answer": answer, "sources": sources, "found": True}