"""Quiz me: practice problems from official Bac papers, answer checking against the official
grading scheme (barem), and the official solution.

Flow:
1. problems come ONLY from a verified bank (app/quiz_bank.json): every statement was transcribed
   by hand from the official PDF (formulas in LaTeX, code in code blocks) and every expected
   answer was taken from the official barem and re-checked (computed or simulated). Raw PDF text
   scrambles matrices, systems and fractions, so problems are never generated from it on the fly.
2. the expected answer stays on the server; the student's answer is checked in layers:
   multiple-choice letter (or the text of an option), exact match after normalization, then an
   AI equivalence check (1/2 == 0.5, {3,1} == {1,3})
"""
import hashlib
import json
import random
import re
import threading
import time
import uuid
from pathlib import Path

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


# ---------------------------------------------------------------- verified problem bank

BANK_FILE = Path(__file__).with_name("quiz_bank.json")


def _load_bank() -> list:
    try:
        bank = json.loads(BANK_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    ok = []
    for item in bank:
        # a problem is shown only if it is complete and its expected answer is known
        if not (item.get("id") and item.get("answer") and item.get("problem", {}).get("ro")
                and item.get("problem", {}).get("en")):
            continue
        if item.get("options") and item["answer"] not in item["options"]:
            continue
        ok.append(item)
    return ok


BANK = _load_bank()
_PAGES = {(p["file"], p["page"]): p for p in EXAM_PAGES}


def _page_for(item: dict) -> dict:
    page = _PAGES.get((item["file"], item["page"]))
    if page:
        return page
    m = re.match(r"(?:\d+_)?(\d{4})_(.+)_subiect\.pdf$", item["file"].split("/")[-1])
    year, session = (int(m.group(1)), m.group(2)) if m else (None, "")
    name = "Matematică" if item["subject"] == "matematica" else "Informatică"
    return {"file": item["file"], "page": item["page"], "subject": item["subject"], "year": year,
            "session": session, "doc_type": "subiect", "text": "",
            "label": f"Subiect {name}, {year}, {session.replace('_', ' ')}"}


def new_quiz(subject=None, lang: str = "ro", seen=None) -> dict:
    lang = "en" if lang == "en" else "ro"
    pool = [b for b in BANK if not subject or b["subject"] == subject]
    if not pool:
        return {"error": "no_subjects"}
    seen = set(seen or [])
    fresh = [b for b in pool if b["id"] not in seen]
    item = random.choice(fresh or pool)   # everything seen -> start a new round
    page = _page_for(item)
    ref = {"subject": page["subject"], "year": page["year"], "session": page["session"],
           "problem": item["problem"]["ro"][:600]}
    quiz_id = _remember({"expected": item["answer"], "options": item.get("options") or {},
                         "ordered": bool(item.get("ordered")), "problem": item["problem"][lang],
                         "ref": ref, "lang": lang})
    return {
        "quiz_id": quiz_id,
        "exercise_key": item["id"],
        "remaining": max(0, len(fresh) - 1),
        "problem": item["problem"][lang],
        "hint": item["hint"][lang],
        "checkable": True,
        "source": _public_source(1, page),
        "quiz_ref": ref,
    }


# ---------------------------------------------------------------- answer checking

def normalize_answer(text: str, ordered: bool = False) -> str:
    """Make harmless differences disappear: spaces, case, diacritics, 'x =', decimal comma,
    unicode minus, braces and the order of values in a set. Intervals and points keep their
    brackets and order, so [a,b] != (a,b) and (4,5) != (5,4)."""
    t = strip_diacritics(str(text).lower()).strip()
    t = t.replace("−", "-").replace("–", "-").replace("$", "").replace("\\", "").replace("`", "")
    t = t.replace("infty", "∞").replace("inf", "∞")
    t = re.sub(r"^(raspuns(ul)?|answer|r)\s*[:=-]\s*", "", t)
    t = re.sub(r"^[a-z]\d*\s*(=|∈|in\b)\s*", "", t)        # "x = 3", "m ∈ {2,3}" -> value
    t = re.sub(r"\s+(sau|si|și|or|and)\s+", ";", t)            # "x=1 sau x=9" -> list
    t = re.sub(r"(?<=\d),(?=\d)", ".", t) if not re.search(r"[{};()\[\]]", t) else t
    t = re.sub(r"\s+", "", t).strip(".;")
    t = re.sub(r"^[a-z](?=\()", "", t)                         # "C(4,5)" -> "(4,5)"
    if re.fullmatch(r"[a-d][.)]?", t):                        # "b)" / "b." -> "b"
        return t[0]
    pairs = re.findall(r"\(([^()]*)\)", t)
    if len(pairs) >= 2 and re.fullmatch(r"(?:[;,]?\([^()]*\))+", t):   # several points / pairs
        items = ["(" + p + ")" for p in pairs]
        return "{" + ",".join(items if ordered else sorted(items)) + "}"
    if re.fullmatch(r"[\[(][^\[\]()]*,[^\[\]()]*[\])]", t):   # one interval or one point
        return t
    if "frac" not in t and (re.search(r"[{};]", t) or t.count(",") >= 1):   # sets / lists
        items = [i for i in re.split(r"[;,]", t.strip("{}")) if i]
        items = [re.sub(r"^[a-z]\d*=", "", i) for i in items]
        return "{" + ",".join(items if ordered else sorted(items)) + "}"
    return t


def _choice_letter(text: str):
    m = re.fullmatch(r"\s*([a-dA-D])(?:\s*[.)]|\s|$).*", text.strip(), flags=re.DOTALL) \
        if len(text.strip()) <= 60 else None
    return m.group(1).lower() if m else None


def _plain(option: str) -> str:
    return option.replace("`", "").strip()


CHECK_PROMPT = """Compare a student's final answer with the official answer for a Bac exercise.
Decide if they are mathematically/logically EQUIVALENT (e.g. 1/2 and 0.5, x=2 and 2, {{1,3}} and \
{{3,1}}, sqrt(4) and 2). Ignore formatting. Be strict: if the official answer has several values, \
the student must give all of them and no extra ones; intervals must have the same endpoints and the \
same open/closed brackets. If the exercise says "respectiv" or asks for printed values, the values \
must be in the same order. If the answer is unclear or only partially correct, it is NOT correct.

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
    options = quiz.get("options") or {}
    language = "English" if lang == "en" else "Romanian"
    student = str(student or "").strip()[:200]
    shown = f"{expected}. {_plain(options[expected])}" if expected in options else expected

    # Layer 1a: multiple choice. A letter is compared directly; the text of an option counts too
    # ("2" for the option "a. 2"), so a right value is never marked wrong for a missing letter.
    if options:
        letter = _choice_letter(student)
        if letter in options:
            return {"correct": letter == expected, "expected": shown, "method": "letter",
                    "explanation": ""}
        given = normalize_answer(student)
        matches = [k for k, v in options.items() if normalize_answer(_plain(v)) == given]
        if len(matches) == 1:
            return {"correct": matches[0] == expected, "expected": shown, "method": "option",
                    "explanation": ""}
        expected_for_ai = shown
    else:
        expected_for_ai = expected

    # Layer 1b: exact match after normalization.
    ordered = quiz.get("ordered") or "respectiv" in strip_diacritics(quiz["problem"].lower())
    if not options and normalize_answer(student, ordered) == normalize_answer(expected, ordered):
        return {"correct": True, "expected": shown, "method": "exact", "explanation": ""}

    # Layer 2: AI equivalence check.
    try:
        raw = llm.chat([{"role": "user", "content": CHECK_PROMPT.format(
            problem=quiz["problem"][:1500], expected=expected_for_ai, student=student,
            language=language)}], temperature=0.0, max_tokens=200)
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        correct = data.get("correct") is True
        explanation = str(data.get("explanation", ""))[:300]
    except (llm.LLMError, ValueError):
        with _store_lock:
            quiz["answered"] = False      # let the student try again
        raise llm.LLMError("check failed")
    return {"correct": correct, "expected": shown, "method": "ai", "explanation": explanation}


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