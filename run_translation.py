#!/usr/bin/env python3
"""Standalone runner — translates a PDF file to a target language without
requiring the FastAPI server, MongoDB, JWT auth, or GCP upload.

Usage:
    python3 run_translation.py <pdf_path> <language> [--output <path>]

Examples:
    python3 run_translation.py Cutouts_Page_1.pdf Hindi
    python3 run_translation.py Cutouts_Page_1.pdf Hindi --output translated_hindi.pdf
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# ── Load .env and map GEMINI_API_KEY → GOOGLE_API_KEY ──────────────────────
from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

# The .env has GEMINI_API_KEY; gemini.py reads GOOGLE_API_KEY
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# ── Now import the service modules ──────────────────────────────────────────
from pdf_translation_service.config import (
    LANGUAGE_CONFIG,
    LOCAL_OUTPUT_DIR,
    logger,
)
from pdf_translation_service.extraction import extract_page_inventory
from pdf_translation_service.fonts import ensure_fonts_available
from pdf_translation_service.gemini import init_gemini_client
from pdf_translation_service.text_mode import translate_page_text_mode


# ── Stub WebSocket manager (no server, just logs) ───────────────────────────
class StubWSManager:
    """No-op WebSocket manager — logs broadcasts instead of sending to clients."""
    async def broadcast(self, job_id: str, message: dict):
        msg_type = message.get("type", "?")
        if msg_type == "page:progress":
            lang = message.get("language", "")
            page = message.get("page", 0)
            total = message.get("totalPages", 0)
            phase = message.get("phase", "")
            pct = message.get("pctTranslated", "")
            logger.info(f"  [WS] {lang} page {page}/{total} — {phase} ({pct}%)")
        elif msg_type in ("doc:started", "doc:language:done", "job:completed"):
            logger.info(f"  [WS] {msg_type}: {message}")


async def translate_pdf(pdf_path: str, target_language: str, output_path: str = None):
    """Translate a single PDF to a single language and save locally."""
    import pymupdf

    if target_language not in LANGUAGE_CONFIG:
        supported = ", ".join(sorted(LANGUAGE_CONFIG.keys()))
        print(f"ERROR: Unsupported language '{target_language}'. Supported: {supported}")
        sys.exit(1)

    lang_config = LANGUAGE_CONFIG[target_language]
    target_script = lang_config["script"]

    # ── Init Gemini client ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Initializing Gemini client...")
    print("=" * 60)
    init_gemini_client()

    # ── Ensure fonts ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Ensuring Noto fonts are available...")
    print("=" * 60)
    await ensure_fonts_available()

    # ── Load PDF ────────────────────────────────────────────────────────────
    pdf_bytes = Path(pdf_path).read_bytes()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    print(f"\n{'=' * 60}")
    print(f"  PDF: {pdf_path} ({total_pages} page(s))")
    print(f"  Target language: {target_language} ({target_script})")
    print(f"{'=' * 60}\n")

    ws = StubWSManager()
    job_id = "standalone-run"
    doc_index = 0

    # ── Pre-extract all page inventories (shared cache) ─────────────────────
    print("Pre-extracting page inventories...")
    loop = asyncio.get_event_loop()

    def _extract_all(d):
        result = []
        for i in range(len(d)):
            try:
                result.append(extract_page_inventory(d[i]))
            except Exception as ex:
                logger.warning(f"Pre-extraction failed for page {i + 1}: {ex}")
                result.append(None)
        return result

    page_inventories = await loop.run_in_executor(None, _extract_all, doc)
    print(f"  → {len(page_inventories)} page(s) extracted\n")

    # ── Translate each page ─────────────────────────────────────────────────
    all_failed_pages = []
    pct_values = []
    review_pages = []

    for page_num in range(total_pages):
        page = doc[page_num]
        inventory = page_inventories[page_num] if page_num < len(page_inventories) else None

        # Rate limit: wait between pages to avoid 429 quota exhaustion
        if page_num > 0:
            logger.info(f"  Waiting 15s before page {page_num + 1} (rate limit)...")
            await asyncio.sleep(15)

        try:
            page_res = await translate_page_text_mode(
                page, page_num, target_language, target_script,
                ws, job_id, doc_index, total_pages,
                inventory=inventory,
            )
            all_failed_pages.extend(page_res["failed_pages"])
            pct_values.append(page_res["pct_translated"])
            if page_res["needs_review"]:
                review_pages.append(page_num + 1)
        except Exception as e:
            logger.error(f"Error on page {page_num + 1}: {e}")
            all_failed_pages.append(page_num + 1)
            review_pages.append(page_num + 1)

    # ── Save output ─────────────────────────────────────────────────────────
    doc_pct = round((sum(pct_values) / len(pct_values) * 100), 1) if pct_values else 100.0

    doc.subset_fonts()
    output_bytes = doc.tobytes(deflate=True, garbage=3)
    doc.close()

    if output_path is None:
        lang_suffix = target_language.lower().replace(" ", "_")
        stem = Path(pdf_path).stem
        output_path = os.path.join(LOCAL_OUTPUT_DIR, f"{stem}_{lang_suffix}.pdf")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(output_bytes)

    print(f"\n{'=' * 60}")
    print(f"  ✅ DONE — Translated PDF saved:")
    print(f"     {output_path}")
    print(f"  Completeness: {doc_pct}% translated")
    print(f"  Failed pages: {all_failed_pages or '—'}")
    print(f"  Pages needing review: {review_pages or '—'}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Translate a PDF to an Indian language")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("language", help="Target language (e.g. Hindi, Marathi, Tamil)")
    parser.add_argument("--output", "-o", default=None, help="Output file path (default: translated images/ dir)")
    args = parser.parse_args()

    if not Path(args.pdf_path).exists():
        print(f"ERROR: PDF not found: {args.pdf_path}")
        sys.exit(1)

    asyncio.run(translate_pdf(args.pdf_path, args.language, args.output))


if __name__ == "__main__":
    main()