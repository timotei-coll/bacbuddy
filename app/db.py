"""Tiny SQLite store for anonymous usage stats and feedback (no personal data)."""
import sqlite3
import threading
import time
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent.parent / "index" / "bacbuddy.db"
_lock = threading.Lock()


def _connect():
    DB_FILE.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, kind TEXT, client TEXT, country TEXT, lang TEXT, subject TEXT, found INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, client TEXT, helpful INTEGER, kind TEXT)""")
    return conn


_conn = _connect()


def log_event(kind, client, country, lang, subject, found):
    with _lock:
        _conn.execute(
            "INSERT INTO events (ts, kind, client, country, lang, subject, found) VALUES (?,?,?,?,?,?,?)",
            (time.time(), kind, client[:64], (country or "")[:40], lang, subject or "", int(bool(found))))
        _conn.commit()


def log_feedback(client, helpful, kind):
    with _lock:
        _conn.execute("INSERT INTO feedback (ts, client, helpful, kind) VALUES (?,?,?,?)",
                      (time.time(), client[:64], int(bool(helpful)), kind[:20]))
        _conn.commit()


def stats() -> dict:
    with _lock:
        cur = _conn.cursor()
        questions = cur.execute("SELECT COUNT(*) FROM events WHERE kind='ask'").fetchone()[0]
        quizzes = cur.execute("SELECT COUNT(*) FROM events WHERE kind='quiz'").fetchone()[0]
        users = cur.execute("SELECT COUNT(DISTINCT client) FROM events").fetchone()[0]
        countries = cur.execute(
            "SELECT country, COUNT(DISTINCT client) FROM events WHERE country != '' "
            "GROUP BY country ORDER BY 2 DESC").fetchall()
        fb = cur.execute("SELECT COUNT(*), COALESCE(SUM(helpful),0) FROM feedback").fetchone()
    return {
        "questions": questions,
        "quizzes": quizzes,
        "users": users,
        "countries": [{"country": c, "users": n} for c, n in countries],
        "feedback_total": fb[0],
        "helpful_percent": round(100 * fb[1] / fb[0]) if fb[0] else None,
    }
