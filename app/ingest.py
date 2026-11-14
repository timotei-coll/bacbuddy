"""Reads the official Bac PDFs from data/ and builds index/chunks.json.

Run from the project folder:
    python -m app.ingest
"""
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

from app.text_utils import normalize_text, strip_diacritics

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

SUBJECTS = {"matematica": "Matematică", "informatica": "Informatică"}
SESSIONS = {
    "iunie": "iunie",
    "iunie_rezerva": "iunie (rezervă)",
    "iulie": "iulie",
    "august": "august",
    "speciala": "sesiunea specială",
}
DOC_TYPES = {"subiect": "Subiect", "barem": "Barem"}

CHUNK_CHARS = 900      # target size of one searchable piece of text
OVERLAP_CHARS = 150    # small overlap so ideas aren't cut in half
MIN_PAGE_CHARS = 40    # less text than this on a page = probably a scanned image

# e.g. 05_2025_iunie_rezerva_barem.pdf
EXAM_FILE = re.compile(r"^\d+_(\d{4})_(.+)_(subiect|barem)\.pdf$", re.IGNORECASE)
# e.g. Programa_BAC_2026_Matematica.pdf
PROGRAM_FILE = re.compile(r"^programa_bac_(\d{4})_(\w+)\.pdf$", re.IGNORECASE)


def describe_file(folder: str, filename: str):
    """Turn a file name into metadata used for filtering and citations."""
    if folder in SUBJECTS:
        m = EXAM_FILE.match(filename)
        if not m:
            return None
        year, session, doc_type = m.group(1), m.group(2).lower(), m.group(3).lower()
        subject = SUBJECTS[folder]
        session_label = SESSIONS.get(session, session.replace("_", " "))
        return {
            "subject": folder,
            "year": int(year),
            "session": session,
            "doc_type": doc_type,
            "label": f"{DOC_TYPES[doc_type]} {subject}, {year}, {session_label}",
        }
    if folder == "programe":
        m = PROGRAM_FILE.match(filename)
        if not m:
            return None
        year, subj = m.group(1), m.group(2).lower()
        subject_key = "matematica" if subj.startswith("mat") else "informatica"
        return {
            "subject": subject_key,
            "year": int(year),
            "session": None,
            "doc_type": "programa",
            "label": f"Programa de examen {SUBJECTS[subject_key]}, Bac {year}",
        }
    return None


# The official "programe" PDFs contain several profiles in one file
# (e.g. M_mate-info, M_st-nat, M_tehnologic, M_pedagogic). We keep only mate-info.
PROFILE_MARKERS = {
    "matematica": {"keep": ["m_mate-info"],
                   "drop": ["m_st-nat", "m_tehnologic", "m_pedagogic"]},
    "informatica": {"keep": ["matematica-informatica"],
                    "drop": ["stiinte ale naturii", "stiintele naturii"]},
}


def detect_profile(subject: str, page_text: str):
    """Return 'keep', 'drop' or None (no clear profile marker on this page)."""
    markers = PROFILE_MARKERS.get(subject)
    if not markers:
        return None
    t = strip_diacritics(page_text.lower())
    has_keep = any(m in t for m in markers["keep"])
    has_drop = any(m in t for m in markers["drop"])
    if has_keep and not has_drop:
        return "keep"
    if has_drop and not has_keep:
        return "drop"
    return None  # both (intro/summary page) or neither: don't change state


def split_into_chunks(text: str) -> list:
    """Split page text into overlapping pieces, cutting at paragraph/line ends."""
    if len(text) <= CHUNK_CHARS:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start + CHUNK_CHARS // 2, end),
                      text.rfind(". ", start + CHUNK_CHARS // 2, end))
            if cut > start:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return chunks


def build_index() -> dict:
    if not DATA_DIR.exists():
        sys.exit(f"Nu găsesc folderul {DATA_DIR}. Copiază PDF-urile în data/.")

    chunks, report = [], {"files": 0, "skipped": [], "low_text_pages": [],
                           "other_profile_pages": 0}
    for folder in ["programe", "matematica", "informatica"]:
        folder_path = DATA_DIR / folder
        if not folder_path.exists():
            print(f"[!] Lipsește folderul data/{folder}")
            continue
        for pdf_path in sorted(folder_path.glob("*.pdf")):
            meta = describe_file(folder, pdf_path.name)
            if meta is None:
                report["skipped"].append(f"{folder}/{pdf_path.name} (nume nerecunoscut)")
                continue
            try:
                reader = PdfReader(str(pdf_path))
            except Exception as exc:  # broken or protected PDF
                report["skipped"].append(f"{folder}/{pdf_path.name} ({exc})")
                continue
            report["files"] += 1
            profile_state = "keep"
            for page_no, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                text = normalize_text(raw)
                if len(text) < MIN_PAGE_CHARS:
                    report["low_text_pages"].append(f"{folder}/{pdf_path.name} p.{page_no}")
                    continue
                if meta["doc_type"] == "programa":
                    profile_state = detect_profile(meta["subject"], text) or profile_state
                    if profile_state == "drop":
                        report["other_profile_pages"] += 1
                        continue
                for piece in split_into_chunks(text):
                    chunks.append({
                        "id": len(chunks),
                        "file": f"{folder}/{pdf_path.name}",
                        "page": page_no,
                        "text": piece,
                        **meta,
                    })

    INDEX_DIR.mkdir(exist_ok=True)
    CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    report["chunks"] = len(chunks)
    return report


if __name__ == "__main__":
    result = build_index()
    print(f"✅ Fișiere citite: {result['files']}")
    print(f"✅ Bucăți de text salvate: {result['chunks']} -> index/chunks.json")
    print(f"ℹ️  Pagini din programe pentru alte profiluri (ignorate): {result['other_profile_pages']}")
    if result["skipped"]:
        print("⚠️  Fișiere sărite:")
        for item in result["skipped"]:
            print("   -", item)
    if result["low_text_pages"]:
        print(f"⚠️  Pagini fără text (posibil scanate): {len(result['low_text_pages'])}")
        for item in result["low_text_pages"][:10]:
            print("   -", item)
