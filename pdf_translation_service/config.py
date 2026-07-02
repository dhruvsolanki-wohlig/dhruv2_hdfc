import asyncio
import logging
import os

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

# Resolve paths relative to the project root (parent of this package directory)
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(_PACKAGE_ROOT, ".env"))

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pdf_translation")

# ── Environment variables ─────────────────────────────────────────────────────

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
MONGO_URL = os.getenv("MONGO_URL")
UPLOAD_URL = os.getenv("UPLOAD_URL", "http://localhost:1330")
CDN_URL = os.getenv("CDN_URL", "https://cdn.pocketstudio.ai")
PDF_SERVICE_PORT = int(os.getenv("PDF_SERVICE_PORT", "8100"))
FONTS_DIR = os.getenv("FONTS_DIR", os.path.join(_PACKAGE_ROOT, "fonts"))
JWT_SECRET_KEY = os.getenv("AUTHENTICATION_JWT_SECRET_KEY")

LOCAL_OUTPUT_DIR = os.path.join(_PACKAGE_ROOT, "translated images")
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# ── Model names ───────────────────────────────────────────────────────────────

# Text-translation model. Default gemini-3.5-flash: newest/best Flash AND available
# in asia-south1 (Mumbai) — required for India data residency (the global endpoint
# does not guarantee in-region processing; gemini-3-flash-preview is NOT served in
# asia-south1). Override via env if needed.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Erase is deterministic: PyMuPDF redaction deletes the source glyphs and reveals
# the real background (no AI). See text_mode.translate_page_text_mode.

# Tables: every cell is translated and placed at its OWN detected bbox via the
# normal per-block flow (the table's vector grid-lines are left untouched by
# redaction, so it still reads as a table). No structured-grid path.

# ── Text renderer ─────────────────────────────────────────────────────────────
# How translated complex-script text is drawn onto the page. Both renderers are
# kept; switch freely.
#   "vector" (default) — HarfBuzz + fontTools filled glyph outlines drawn directly
#                        on the page. Crisp at any zoom, and NO transparent/soft-mask
#                        image, so it renders correctly in every PDF viewer (the
#                        raster path's transparent text layer washes out in some
#                        viewers). (pdf_translation_service.vector_text.vector_insert_text)
#   "raster"           — PIL/FreeType transparent glyph PNG (pdf_translation_service.text_mode._pil_insert_text)
# Vector falls back to raster automatically if it can't render a block.
TEXT_RENDER_MODE = os.getenv("TEXT_RENDER_MODE", "vector").strip().lower()

# ── Concurrency semaphores ────────────────────────────────────────────────────

DOC_SEMAPHORE = asyncio.Semaphore(5)
LANG_SEMAPHORE = asyncio.Semaphore(10)
GEMINI_SEMAPHORE = asyncio.Semaphore(15)

# ── Batch constants ───────────────────────────────────────────────────────────

GEMINI_BATCH_SIZE = 20

# ── Noto Fonts base URL ───────────────────────────────────────────────────────

NOTO_FONTS_BASE = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts"

# ── Scripts requiring complex OpenType shaping ────────────────────────────────
# PyMuPDF cannot reliably re-read conjuncts/ligatures written via insert_htmlbox
# (e.g. वर्ष re-extracts as 'व�ष'). Completeness checks for these scripts MUST
# operate on pre-insertion translation strings, never on re-extracted PDF text.
COMPLEX_SCRIPTS: frozenset[str] = frozenset({
    "Devanagari", "Tamil", "Telugu", "Malayalam",
    "Bengali", "Gujarati", "Kannada", "Gurmukhi", "Odia",
})

# ── Language configuration ────────────────────────────────────────────────────

LANGUAGE_CONFIG = {
    "Hindi": {
        "code": "hi",
        "script": "Devanagari",
        "font_family": "Noto Sans Devanagari",
        "regular": "NotoSansDevanagari-Regular.ttf",
        "bold": "NotoSansDevanagari-Bold.ttf",
    },
    "Marathi": {
        "code": "mr",
        "script": "Devanagari",
        "font_family": "Noto Sans Devanagari",
        "regular": "NotoSansDevanagari-Regular.ttf",
        "bold": "NotoSansDevanagari-Bold.ttf",
    },
    "Sanskrit": {
        "code": "sa",
        "script": "Devanagari",
        "font_family": "Noto Sans Devanagari",
        "regular": "NotoSansDevanagari-Regular.ttf",
        "bold": "NotoSansDevanagari-Bold.ttf",
    },
    "Gujarati": {
        "code": "gu",
        "script": "Gujarati",
        "font_family": "Noto Sans Gujarati",
        "regular": "NotoSansGujarati-Regular.ttf",
        "bold": "NotoSansGujarati-Bold.ttf",
    },
    "Kannada": {
        "code": "kn",
        "script": "Kannada",
        "font_family": "Noto Sans Kannada",
        "regular": "NotoSansKannada-Regular.ttf",
        "bold": "NotoSansKannada-Bold.ttf",
    },
    "Tamil": {
        "code": "ta",
        "script": "Tamil",
        "font_family": "Noto Sans Tamil",
        "regular": "NotoSansTamil-Regular.ttf",
        "bold": "NotoSansTamil-Bold.ttf",
    },
    "Telugu": {
        "code": "te",
        "script": "Telugu",
        "font_family": "Noto Sans Telugu",
        "regular": "NotoSansTelugu-Regular.ttf",
        "bold": "NotoSansTelugu-Bold.ttf",
    },
    "Bengali": {
        "code": "bn",
        "script": "Bengali",
        "font_family": "Noto Sans Bengali",
        "regular": "NotoSansBengali-Regular.ttf",
        "bold": "NotoSansBengali-Bold.ttf",
    },
    "Malayalam": {
        "code": "ml",
        "script": "Malayalam",
        "font_family": "Noto Sans Malayalam",
        "regular": "NotoSansMalayalam-Regular.ttf",
        "bold": "NotoSansMalayalam-Bold.ttf",
    },
    "Punjabi": {
        "code": "pa",
        "script": "Gurmukhi",
        "font_family": "Noto Sans Gurmukhi",
        "regular": "NotoSansGurmukhi-Regular.ttf",
        "bold": "NotoSansGurmukhi-Bold.ttf",
    },
    "Odia": {
        "code": "or",
        "script": "Odia",
        "font_family": "Noto Sans Oriya",
        "regular": "NotoSansOriya-Regular.ttf",
        "bold": "NotoSansOriya-Bold.ttf",
    },
    "Assamese": {
        "code": "as",
        "script": "Bengali",
        "font_family": "Noto Sans Bengali",
        "regular": "NotoSansBengali-Regular.ttf",
        "bold": "NotoSansBengali-Bold.ttf",
    },
}

# ── Terminology glossary ──────────────────────────────────────────────────────
# Preferred translations for HDFC/insurance domain terms, locked for consistency
# across runs and languages. Injected into the translation prompt. Add a language
# key to lock its terminology; languages without an entry are translated freely
# by the model. These are starter terms — the client's approved glossary should
# replace/extend them before production.
INSURANCE_GLOSSARY: dict[str, dict[str, str]] = {
    "Hindi": {
        "Sum Assured": "बीमा राशि",
        "Death Benefit": "मृत्यु लाभ",
        "Maturity Benefit": "परिपक्वता लाभ",
        "Survival Benefit": "उत्तरजीविता लाभ",
        "Income Benefit": "आय लाभ",
        "Guaranteed": "गारंटीड",
        "Premium": "प्रीमियम",
        "Premium Payment Term": "प्रीमियम भुगतान अवधि",
        "Policy Term": "पॉलिसी अवधि",
        "Policyholder": "पॉलिसीधारक",
        "Life Insurance": "जीवन बीमा",
        "Maturity": "परिपक्वता",
        "Nominee": "नामिती",
        "Rider": "राइडर",
        "Lumpsum": "एकमुश्त",
        "Annual": "वार्षिक",
        "Half-Yearly": "अर्ध-वार्षिक",
        "Monthly": "मासिक",
        "years": "वर्ष",
        "Eligibility": "पात्रता",
        "Minimum": "न्यूनतम",
        "Maximum": "अधिकतम",
        "Waiver of Premium": "प्रीमियम की छूट",
    },
}


def get_glossary(target_language: str) -> dict[str, str]:
    """Return the locked terminology map for a language (empty if none)."""
    return INSURANCE_GLOSSARY.get(target_language, {})


# ── Retry decorator ───────────────────────────────────────────────────────────


def is_retryable_error(exception: BaseException) -> bool:
    """Only retry on 429 Resource Exhausted, 500/503 server errors, or transient network errors."""
    err_str = str(exception).lower()
    retryable = False
    reason = ""
    if "429" in err_str or "resource exhausted" in err_str or "resource_exhausted" in err_str:
        retryable, reason = True, "429 Resource Exhausted"
    elif "500" in err_str or "503" in err_str or "internal" in err_str or "unavailable" in err_str:
        retryable, reason = True, "Server error"
    elif any(kw in err_str for kw in ("timeout", "connection", "reset", "broken pipe")):
        retryable, reason = True, "Network error"
    elif "empty response" in err_str:
        retryable, reason = True, "Empty response"
    if retryable:
        logger.warning(f"Retryable error detected ({reason}): {str(exception)[:120]}")
    return retryable


GEMINI_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 2),
    retry=retry_if_exception(is_retryable_error),
    reraise=True,
)
