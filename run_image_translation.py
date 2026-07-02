#!/usr/bin/env python3
"""Translate text in PNG images from the cuutouts/ folder to Hindi.

Pipeline per image:
  1. Gemini Vision extracts text blocks with bounding boxes (JSON)
  2. Gemini text-to-text translates each block (with glossary + fit constraints)
  3. Residual English check + auto-repair pass
  4. PIL erases original text (sampling actual bg pixels from bbox border)
     and renders Hindi text with Noto fonts at the same positions

FIXES APPLIED:
  - Rate limiting: 15s between Gemini calls, 45s/90s backoff on 429
  - Erase: sample actual background pixels from bbox border (not flat color)
  - Hindi text sizing: start at 0.85x for Indic, min 0.50x (not 0.30x)
  - Residual English check + auto-repair (same as PDF pipeline)
  - Overlap prevention: sort blocks by position, skip >60% overlapping bboxes

Output: 8_section_output/<original_name>_hindi.png
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import io

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# ── Import service modules ───────────────────────────────────────────────────
from pdf_translation_service.config import (
    LANGUAGE_CONFIG,
    get_glossary,
    logger,
)
from pdf_translation_service.fonts import ensure_fonts_available, get_pil_font, _get_latin_pil_font
from pdf_translation_service.gemini import init_gemini_client
from pdf_translation_service.utils import clean_symbol_text, should_translate
from pdf_translation_service.extraction import is_residual_english, apply_proportional_styles
import google.genai as genai

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR = Path(__file__).parent / "cuutouts"
OUTPUT_DIR = Path(__file__).parent / "8_section_output"
TARGET_LANGUAGE = "Hindi"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FONT_DIR = Path(__file__).parent / "fonts"

_DEVA_LO, _DEVA_HI = 0x0900, 0x0DFF


def _is_indic(text):
    return any(_DEVA_LO <= ord(c) <= _DEVA_HI for c in text)


# ── Rate limiter ──────────────────────────────────────────────────────────────

_last_gemini_call = 0.0
_MIN_INTERVAL = 15.0  # 15s between Gemini calls = 4 req/min max


async def _rate_limit():
    """Ensure at least _MIN_INTERVAL seconds between Gemini API calls."""
    global _last_gemini_call
    now = time.monotonic()
    elapsed = now - _last_gemini_call
    if elapsed < _MIN_INTERVAL:
        wait = _MIN_INTERVAL - elapsed
        logger.info(f"  [RateLimit] Waiting {wait:.1f}s before next Gemini call...")
        await asyncio.sleep(wait)
    _last_gemini_call = time.monotonic()


# ── Step 1: Vision-based text extraction ─────────────────────────────────────

VISION_PROMPT = """You are a precise OCR system. Analyze this image and extract ALL text blocks.

For each text block, provide:
- "text": the exact text content (preserve line breaks as \\n)
- "bbox": [x1, y1, x2, y2] — pixel coordinates of the bounding box (top-left origin)
- "font_size": approximate font size in pixels
- "color": approximate text color as hex (e.g. "#FFFFFF" for white, "#000000" for black)
- "bold": true if the text appears bold, false otherwise
- "bg_color": approximate background color behind this text as hex

Rules:
- Include EVERY text element: headings, body text, table cells, labels, values, footnotes, disclaimers
- Skip logos, icons, and pure decorative graphics
- For text on colored backgrounds, estimate the background color
- Coordinates must be in image pixel space (0,0 = top-left)
- Return ONLY valid JSON, no markdown

Return format:
{"blocks": [{"text": "...", "bbox": [x1,y1,x2,y2], "font_size": N, "color": "#RRGGBB", "bold": false, "bg_color": "#RRGGBB"}, ...]}
"""


async def extract_text_blocks_from_image(image_bytes: bytes, mime_type: str = "image/png") -> list[dict]:
    """Use Gemini Vision to extract text blocks with bounding boxes from an image.
    
    Rate-limited with 45s/90s backoff on 429 quota errors.
    """
    from pdf_translation_service.gemini import gemini_client

    for attempt in range(3):
        try:
            await _rate_limit()
            response = await asyncio.wait_for(
                gemini_client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        VISION_PROMPT,
                    ],
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1,
                    },
                ),
                timeout=90,
            )

            if not response or not response.text:
                raise RuntimeError("Empty response from Gemini Vision")

            data = json.loads(response.text)
            blocks = data.get("blocks", [])
            logger.info(f"  [Vision] Extracted {len(blocks)} text block(s)")
            return blocks

        except Exception as e:
            err_str = str(e).lower()
            # On 429 quota errors, wait much longer (45s, 90s)
            if "429" in err_str or "resource exhausted" in err_str or "quota" in err_str:
                wait_time = 45 * (attempt + 1)
                logger.warning(f"  [Vision] 429 quota hit — waiting {wait_time}s before retry {attempt+1}/3")
                await asyncio.sleep(wait_time)
            else:
                logger.warning(f"  [Vision] Attempt {attempt+1}/3 failed: {str(e)[:120]}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

    logger.error("  [Vision] All attempts failed — returning empty block list")
    return []


# ── Step 2: Translation (reuse existing Gemini pipeline) ─────────────────────

async def translate_blocks(blocks: list[dict], target_language: str, target_script: str) -> list[str]:
    """Translate extracted text blocks using the existing Gemini translation pipeline."""
    from pdf_translation_service.gemini import translate_batch_with_gemini

    text_blocks = [{"text": b["text"]} for b in blocks if b.get("text", "").strip()]

    if not text_blocks:
        return []

    page_context = "\n".join(b["text"] for b in text_blocks)

    await _rate_limit()
    translations = await translate_batch_with_gemini(
        text_blocks, target_language, target_script, page_context=page_context
    )
    return translations


# ── Step 2b: Residual English check + repair ─────────────────────────────────

async def repair_residuals(blocks, translations, target_language, target_script):
    """Re-translate any blocks that are still in English."""
    residuals = [
        i for i, t in enumerate(translations)
        if should_translate(blocks[i].get("text", "")) and is_residual_english(t or "")
    ]
    if not residuals:
        return translations

    logger.info(f"  [Repair] {len(residuals)} block(s) still in English — re-translating")
    repair_blocks = [{"text": blocks[i]["text"]} for i in residuals]
    await _rate_limit()
    from pdf_translation_service.gemini import translate_batch_with_gemini
    retry = await translate_batch_with_gemini(
        repair_blocks, target_language, target_script,
        page_context="\n".join(b["text"] for b in repair_blocks),
    )
    for j, i in enumerate(residuals):
        if j < len(retry) and retry[j] and not is_residual_english(retry[j]):
            translations[i] = retry[j]
    return translations


# ── Step 3: PIL rendering (erase + insert Hindi text) ─────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple:
    h = (hex_color or "#000000").lstrip("#")
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except Exception:
        return (0, 0, 0)


def _font_for(text: str, language: str, size_px: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Pick the right font: Indic script for Hindi, Latin for English/digits."""
    if _is_indic(text):
        return get_pil_font(language, size_px, bold=bold)
    return _get_latin_pil_font(size_px, bold=bold)


def _text_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0]
    except Exception:
        return len(text) * 10


def _sample_bg_color(image: Image.Image, bbox: list, pad: int = 4) -> tuple:
    """Sample the actual background color from the border pixels around a text bbox.
    
    Instead of trusting Gemini's bg_color estimate, we sample pixels from a
    2px-wide ring just outside the text bbox. This gives the true background
    color even on gradients or photos.
    """
    import numpy as np
    x1, y1, x2, y2 = bbox
    img_w, img_h = image.size
    arr = np.array(image.convert("RGB"))

    # Clamp sampling area to image bounds
    sx1 = max(0, x1 - pad)
    sy1 = max(0, y1 - pad)
    sx2 = min(img_w, x2 + pad)
    sy2 = min(img_h, y2 + pad)

    # Collect border pixels (ring around the text bbox)
    border_pixels = []
    # Top and bottom strips
    if sy1 < y1:
        border_pixels.extend(arr[sy1:y1, sx1:sx2].reshape(-1, 3).tolist())
    if sy2 > y2:
        border_pixels.extend(arr[y2:sy2, sx1:sx2].reshape(-1, 3).tolist())
    # Left and right strips (excluding corners already captured)
    if sx1 < x1:
        border_pixels.extend(arr[y1:y2, sx1:x1].reshape(-1, 3).tolist())
    if sx2 > x2:
        border_pixels.extend(arr[y1:y2, x2:sx2].reshape(-1, 3).tolist())

    if not border_pixels:
        # Fallback: use Gemini's estimate or white
        return (255, 255, 255)

    # Use median to be robust against outliers
    arr_bp = np.array(border_pixels)
    med = np.median(arr_bp, axis=0).astype(int)
    return tuple(med.tolist())


def _overlap_ratio(a: list, b: list) -> float:
    """How much bbox A overlaps bbox B (0.0 - 1.0)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    if a_area <= 0:
        return 0.0
    return inter / a_area


def render_translated_image(
    image: Image.Image,
    blocks: list[dict],
    translations: list[str],
    language: str,
) -> Image.Image:
    """Erase original text and render translated text at the same positions.
    
    FIXES:
    - Erase: sample actual bg pixels from bbox border (not flat Gemini estimate)
    - Sort blocks by y-position before rendering (prevents overwrite)
    - Skip blocks that overlap a previously rendered block by >60%
    - Hindi text sizing: start at 0.85x for Indic, min 0.50x
    """
    draw = ImageDraw.Draw(image, "RGBA")
    img_w, img_h = image.size

    # Sort blocks by y-position, then x — render top-to-bottom
    indexed_blocks = list(enumerate(blocks))
    indexed_blocks.sort(key=lambda pair: (pair[1].get("bbox", [0, 0, 0, 0])[1], pair[1].get("bbox", [0, 0, 0, 0])[0]))

    rendered_bboxes = []
    trans_idx = 0

    for _, block in indexed_blocks:
        text = block.get("text", "").strip()
        if not text:
            continue

        bbox = block.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 2 or box_h < 2:
            continue

        # Get translation for this block
        if trans_idx >= len(translations):
            break
        translated = translations[trans_idx]
        trans_idx += 1

        if not translated or not translated.strip():
            continue

        # Skip if this block overlaps a previously rendered block by >60%
        skip = False
        for prev_bbox in rendered_bboxes:
            if _overlap_ratio(bbox, prev_bbox) > 0.60:
                logger.warning(f"  [Render] Skipping block at {bbox} — overlaps previous render")
                skip = True
                break
        if skip:
            continue

        # ── Erase original text ──────────────────────────────────────────────
        # FIX: Sample actual background pixels from the border ring around bbox
        bg_color = _sample_bg_color(image, bbox, pad=4)
        pad = 3
        draw.rectangle(
            [x1 - pad, y1 - pad, x2 + pad, y2 + pad],
            fill=bg_color + (255,),
        )

        # ── Render translated text ───────────────────────────────────────────
        original_size = block.get("font_size", 14)
        bold = block.get("bold", False)
        text_color = _hex_to_rgb(block.get("color", "#000000"))

        # Split translated text into words
        words = translated.replace("\n", " ").split()
        if not words:
            continue

        # FIX: Indic text renders ~20% wider — start at 0.85x, min 0.50x
        is_indic_text = _is_indic(translated)
        if is_indic_text:
            initial_shrink = 0.85
            min_shrink = 0.50
        else:
            initial_shrink = 1.0
            min_shrink = 0.30

        # Determine if single-line or multi-line box
        single_line = box_h < original_size * 2.5

        if single_line:
            # Shrink font until it fits the box width
            shrink = initial_shrink
            runs = []
            while shrink >= min_shrink:
                font_size = max(6, int(original_size * shrink))
                font = _font_for(translated, language, font_size, bold=bold)
                # Build runs of same-font words
                runs = []
                for w in words:
                    f = _font_for(w, language, font_size, bold=bold)
                    if runs and runs[-1][1] is f:
                        runs[-1][0] += " " + w
                    else:
                        runs.append([w, f])
                sp_w = _text_width(font, " ")
                total_w = sum(_text_width(f, t) for t, f in runs) + sp_w * (len(runs) - 1)
                if total_w <= box_w - 4:
                    break
                shrink -= 0.05

            # Render centered vertically
            try:
                ascent, descent = font.getmetrics()
                lh = ascent + descent
            except Exception:
                lh = font_size
            y_start = y1 + max(0, (box_h - lh) // 2)
            x = x1 + 2
            for ri, (run_text, run_font) in enumerate(runs):
                draw.text((x, y_start), run_text, font=run_font, fill=text_color + (255,))
                x += _text_width(run_font, run_text)
                if ri < len(runs) - 1:
                    x += _text_width(run_font, " ")
        else:
            # Multi-line: wrap text to fit box
            shrink = initial_shrink
            final_lines = None
            final_font_size = original_size
            while shrink >= min_shrink:
                font_size = max(6, int(original_size * shrink))
                font = _font_for(translated, language, font_size, bold=bold)
                sp_w = _text_width(font, " ")

                # Wrap words into lines
                lines = []
                cur, cur_w = [], 0
                for w in words:
                    wf = _font_for(w, language, font_size, bold=bold)
                    ww = _text_width(wf, w)
                    need = (cur_w + sp_w + ww) if cur else ww
                    if need <= box_w - 4 or not cur:
                        cur.append((w, wf))
                        cur_w = need
                    else:
                        lines.append(cur)
                        cur = [(w, wf)]
                        cur_w = ww
                if cur:
                    lines.append(cur)

                try:
                    ascent, descent = font.getmetrics()
                    lh = ascent + descent + 1
                except Exception:
                    lh = font_size + 2

                if lh * len(lines) <= box_h:
                    final_lines = lines
                    final_font_size = font_size
                    break
                shrink -= 0.07

            if final_lines is None:
                # Accept overflow at smallest size
                font_size = max(6, int(original_size * min_shrink))
                font = _font_for(translated, language, font_size, bold=bold)
                sp_w = _text_width(font, " ")
                lines = []
                cur, cur_w = [], 0
                for w in words:
                    wf = _font_for(w, language, font_size, bold=bold)
                    ww = _text_width(wf, w)
                    need = (cur_w + sp_w + ww) if cur else ww
                    if need <= box_w - 4 or not cur:
                        cur.append((w, wf))
                        cur_w = need
                    else:
                        lines.append(cur)
                        cur = [(w, wf)]
                        cur_w = ww
                if cur:
                    lines.append(cur)
                final_lines = lines
                try:
                    ascent, descent = font.getmetrics()
                    lh = ascent + descent + 1
                except Exception:
                    lh = font_size + 2

            # Render lines
            try:
                ascent, _d = font.getmetrics()
            except Exception:
                ascent = font_size
            total_h = lh * len(final_lines)
            y_start = y1 + max(0, (box_h - total_h) // 2)

            for li, line in enumerate(final_lines):
                sp_w = _text_width(font, " ")
                line_widths = [_text_width(f, w) for w, f in line]
                line_w = sum(line_widths) + sp_w * (len(line) - 1)
                x = x1 + 2
                y = y_start + li * lh
                for wi, (word, wfont) in enumerate(line):
                    draw.text((x, y), word, font=wfont, fill=text_color + (255,))
                    x += _text_width(wfont, word)
                    if wi < len(line) - 1:
                        x += sp_w

        rendered_bboxes.append(bbox)

    return image


# ── Main pipeline ────────────────────────────────────────────────────────────

async def process_image(image_path: Path, output_dir: Path, language: str, lang_config: dict):
    """Process a single image: extract → translate → repair → render → save."""
    target_script = lang_config["script"]
    name = image_path.stem
    logger.info(f"\n{'='*60}")
    logger.info(f"  Processing: {image_path.name}")
    logger.info(f"{'='*60}")

    # Load image
    image_bytes = image_path.read_bytes()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img_w, img_h = image.size
    logger.info(f"  Image size: {img_w}x{img_h}")

    # Step 1: Extract text blocks via Gemini Vision
    logger.info(f"  Step 1: Extracting text blocks via Gemini Vision...")
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    blocks = await extract_text_blocks_from_image(image_bytes, mime_type=mime)
    if not blocks:
        logger.warning(f"  No text blocks found — copying image as-is")
        out = image.convert("RGB")
        out.save(output_dir / f"{name}_hindi{image_path.suffix}")
        return

    # Filter: only blocks with translatable text
    translatable_blocks = []
    for b in blocks:
        if should_translate(b.get("text", "")):
            translatable_blocks.append(b)
    logger.info(f"  Translatable blocks: {len(translatable_blocks)} / {len(blocks)}")

    if not translatable_blocks:
        logger.warning(f"  No translatable text found — copying image as-is")
        out = image.convert("RGB")
        out.save(output_dir / f"{name}_hindi{image_path.suffix}")
        return

    # Step 2: Translate
    logger.info(f"  Step 2: Translating {len(translatable_blocks)} block(s) to {language}...")
    translations = await translate_blocks(translatable_blocks, language, target_script)
    logger.info(f"  Translated: {len(translations)} block(s)")

    # Step 2b: Residual English check + repair
    logger.info(f"  Step 2b: Checking for residual English...")
    translations = await repair_residuals(translatable_blocks, translations, language, target_script)

    # Step 3: Render
    logger.info(f"  Step 3: Rendering translated text on image...")
    result = render_translated_image(image, translatable_blocks, translations, language)

    # Save
    out_path = output_dir / f"{name}_hindi{image_path.suffix}"
    result.convert("RGB").save(out_path, quality=95)
    logger.info(f"  ✅ Saved: {out_path}")


async def main():
    # Collect images sorted by name
    images = sorted(
        [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=lambda f: f.name,
    )

    if not images:
        print(f"ERROR: No images found in {INPUT_DIR}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  PDF Translation Service — Image Mode (FIXED)")
    print(f"  Input:  {INPUT_DIR} ({len(images)} image(s))")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Target language: {TARGET_LANGUAGE}")
    print(f"{'='*60}\n")

    # Init Gemini + fonts
    init_gemini_client()
    await ensure_fonts_available()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lang_config = LANGUAGE_CONFIG[TARGET_LANGUAGE]

    # Process images sequentially (rate-limited)
    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {img_path.name}")
        try:
            await process_image(img_path, OUTPUT_DIR, TARGET_LANGUAGE, lang_config)
        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"  All done! Output in: {OUTPUT_DIR}")
    print(f"{'='*60}")
    output_files = list(OUTPUT_DIR.glob("*_hindi.*"))
    for f in sorted(output_files):
        print(f"    {f.name} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    asyncio.run(main())