"""BacBuddy web server.

Run locally:   uvicorn app.server:app --reload
Then open:     http://127.0.0.1:8000
"""
import time
from collections import defaultdict, deque
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from app import db, quiz as quizzes, rag
from app.llm import LLMError

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"

SUBJECTS = {"matematica", "informatica"}
MAX_QUESTION_CHARS = 500
RATE_WINDOW = 60         # seconds
LIMIT_PER_CLIENT = 12    # AI requests per minute for one browser
LIMIT_PER_IP = 60        # higher, because a whole school can share one IP
_hits = defaultdict(deque)


def _client_ip(request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _over_limit(key: str, limit: int, now: float) -> bool:
    q = _hits[key]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= limit:
        return True
    q.append(now)
    return False


def _rate_limited(request, body: dict) -> bool:
    now = time.time()
    client = "c:" + str(body.get("client") or "anon")[:64]
    ip = "ip:" + _client_ip(request)
    return _over_limit(ip, LIMIT_PER_IP, now) or _over_limit(client, LIMIT_PER_CLIENT, now)


def _clean(body: dict):
    subject = body.get("subject") if body.get("subject") in SUBJECTS else None
    try:
        year = int(body["year"]) if body.get("year") else None
    except (TypeError, ValueError):
        year = None
    lang = "en" if body.get("lang") == "en" else "ro"
    client = str(body.get("client") or "anon")
    country = str(body.get("country") or "")
    return subject, year, lang, client, country


async def _json_body(request) -> dict:
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


def _too_many():
    return JSONResponse({"error": "Prea multe cereri. Încearcă din nou peste un minut. / "
                                  "Too many requests, try again in a minute."}, status_code=429)


def _ai_error(exc):
    print("[AI error]", exc)
    return JSONResponse({"error": "Serviciul AI nu răspunde momentan. Încearcă din nou. / "
                                  "The AI service is not responding, please try again."},
                        status_code=502)


async def ask(request):
    body = await _json_body(request)
    if _rate_limited(request, body):
        return _too_many()
    question = str(body.get("question", "")).strip()[:MAX_QUESTION_CHARS]
    if len(question) < 3:
        return JSONResponse({"error": "Scrie o întrebare. / Please type a question."}, status_code=400)
    subject, year, lang, client, country = _clean(body)
    try:
        result = await run_in_threadpool(rag.ask, question, subject, year, lang)
    except LLMError as exc:
        return _ai_error(exc)
    db.log_event("ask", client, country, lang, subject, result["found"])
    return JSONResponse(result)


async def quiz(request):
    body = await _json_body(request)
    if _rate_limited(request, body):
        return _too_many()
    subject, _, lang, client, country = _clean(body)
    try:
        seen = [str(x)[:16] for x in body.get("seen", [])][:300] if isinstance(body.get("seen"), list) else []
        result = await run_in_threadpool(quizzes.new_quiz, subject, lang, seen)
    except LLMError as exc:
        return _ai_error(exc)
    db.log_event("quiz", client, country, lang, subject, "error" not in result)
    return JSONResponse(result)


async def quiz_solution(request):
    body = await _json_body(request)
    if _rate_limited(request, body):
        return _too_many()
    ref = body.get("quiz_ref") if isinstance(body.get("quiz_ref"), dict) else {}
    _, _, lang, _, _ = _clean(body)
    try:
        return JSONResponse(await run_in_threadpool(quizzes.solution, ref, lang))
    except LLMError as exc:
        return _ai_error(exc)


async def quiz_check(request):
    body = await _json_body(request)
    if _rate_limited(request, body):
        return _too_many()
    _, _, lang, client, country = _clean(body)
    quiz_id = str(body.get("quiz_id", ""))[:64]
    try:
        result = await run_in_threadpool(quizzes.check_answer, quiz_id, body.get("answer", ""))
    except LLMError as exc:
        return _ai_error(exc)
    if "error" not in result:
        db.log_event("quiz_answer", client, country, lang, None, result["correct"])
    return JSONResponse(result)


EVENT_KINDS = {"share_card", "level_up"}


async def event(request):
    """Small anonymous events from the UI (e.g. an achievement card was shared)."""
    body = await _json_body(request)
    kind = str(body.get("kind", ""))
    if kind not in EVENT_KINDS:
        return JSONResponse({"error": "unknown event"}, status_code=400)
    _, _, lang, client, country = _clean(body)
    db.log_event(kind, client, country, lang, None, True)
    return JSONResponse({"ok": True})


async def feedback(request):
    body = await _json_body(request)
    _, _, _, client, _ = _clean(body)
    db.log_feedback(client, bool(body.get("helpful")), str(body.get("kind", "ask")))
    return JSONResponse({"ok": True})


async def stats(request):
    return JSONResponse(db.stats())


async def health(request):
    return JSONResponse({"status": "ok", "chunks": len(rag.INDEX.chunks)})


async def homepage(request):
    return FileResponse(STATIC_DIR / "index.html")


async def stats_page(request):
    return FileResponse(STATIC_DIR / "stats.html")


app = Starlette(routes=[
    Route("/", homepage),
    Route("/stats", stats_page),
    Route("/api/ask", ask, methods=["POST"]),
    Route("/api/quiz", quiz, methods=["POST"]),
    Route("/api/quiz/solution", quiz_solution, methods=["POST"]),
    Route("/api/quiz/check", quiz_check, methods=["POST"]),
    Route("/api/event", event, methods=["POST"]),
    Route("/api/feedback", feedback, methods=["POST"]),
    Route("/api/stats", stats),
    Route("/health", health),
    Mount("/data", app=StaticFiles(directory=DATA_DIR), name="data"),
    Mount("/static", app=StaticFiles(directory=STATIC_DIR), name="static"),
])