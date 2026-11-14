"""Text helpers for Romanian: diacritic normalization and tokenization."""
import re
import unicodedata

# Old Romanian PDFs often use cedilla letters (ş, ţ) instead of the correct
# comma-below letters (ș, ț). We convert everything to the correct form.
_CEDILLA_FIX = str.maketrans({
    "\u015f": "\u0219",  # cedilla s -> comma s
    "\u015e": "\u0218",  # cedilla S -> comma S
    "\u0163": "\u021b",  # cedilla t -> comma t
    "\u0162": "\u021a",  # cedilla T -> comma T
})

# Very common Romanian words that don't help search.
STOPWORDS = {
    "si", "sa", "se", "de", "la", "in", "cu", "pe", "din", "un", "o", "a", "al",
    "ale", "ai", "ca", "ce", "care", "este", "sunt", "fi", "nu", "mai", "sau",
    "prin", "pentru", "lui", "lor", "le", "il", "ii", "i", "e", "am", "ati",
    "au", "fie", "dupa", "cum", "iar", "dar", "daca", "cat", "ori", "the",
    "and", "of", "to", "is", "what", "how", "for", "in", "on", "an",
}


def normalize_text(text: str) -> str:
    """Fix diacritics, join hyphenated line breaks and tidy whitespace."""
    text = unicodedata.normalize("NFC", text).translate(_CEDILLA_FIX)
    text = re.sub(r"-\n(\w)", r"\1", text)      # word-\nbreak -> wordbreak
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = re.sub(r"[ \t]+", " ", text)
    # PDF extraction sometimes splits words around diacritics:
    # "î ntr-un" -> "într-un", "memoreaz ă" -> "memorează"
    text = re.sub(r"(?<![^\W\d_])([îÎ]) (?=[^\W\d_])", r"\1", text)
    text = re.sub(r"(?<=[^\W\d_]) ([ăâșțĂÂȘȚ])(?![^\W\d_])", r"\1", text)
    text = re.sub(r"(?<=[^\W\d_]) ([ăâĂÂ])(?=[^\W\d_])", r"\1", text)  # "num ărul" -> "numărul"
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def strip_diacritics(text: str) -> str:
    """'funcție' -> 'functie', so students can type without diacritics."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


STEM_LEN = 6  # simple prefix stemming: 'funcției', 'functia', 'funcții' -> 'functi'


def tokenize(text: str) -> list:
    """Lowercase, remove diacritics and stopwords, then cut words to a common prefix
    so Romanian word endings (-ul, -ei, -ilor...) still match."""
    text = strip_diacritics(text.lower())
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t[:STEM_LEN] for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]