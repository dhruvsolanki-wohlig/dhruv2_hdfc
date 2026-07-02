import asyncio
import re
from collections import Counter
from typing import Optional

import pymupdf

from .config import logger
from .extraction import (
    apply_proportional_styles,
    detect_block_alignment,
    get_block_style_segments,
    get_block_text,
    is_residual_english,
    process_page_extraction,
    residual_translation_report,
    should_translate,
)
from .fonts import (
    _get_latin_pil_font,
    get_font_archive,
    get_font_css,
    get_latin_pymupdf_font,
    get_pil_font,
    get_pymupdf_font,
)
from .gemini import translate_batch_with_gemini
from .utils import clean_symbol_text, hex_from_int, is_bold, rect_to_list


# ── License / registration code patterns ─────────────────────────────────────
# Covers: CIN (L65110MH2000PLC128245), IRDAI UIN (101N005V03), IRDA reg numbers.
# These tokens are kept verbatim (not translated).
_LICENSE_RE = re.compile(
    r"\b(?:"
    r"[A-Z]{1,5}\d{5,}\w+"          # CIN-style: L65110MH2000PLC128245
    r"|"
    r"\d{3}[A-Z]\d{3}[A-Z]\d{2}"    # UIN-style: 101N005V03
    r"|"
    r"IRDAI?/[A-Z]+/\w+"             # IRDAI/HLI/... format
    r")\b",
    re.IGNORECASE,
)


def _is_code_only_block(text: str) -> bool:
    """True when a block is essentially just license/registration identifiers with
    no real prose to translate — e.g. a standalone 'CIN: L65110MH2000PLC128245'
    line. A legal DISCLAIMER that merely CONTAINS such a code returns False: it must
    be translated (the codes/URLs/names inside it are preserved by the prompt, not
    by skipping the whole paragraph)."""
    masked = _LICENSE_RE.sub(" ", text)
    words = re.findall("[A-Za-zऀ-෿]{3,}", masked)
    real = [w for w in words if not (w.isupper() and len(w) <= 5)]  # drop ALLCAPS acronyms
    return len(real) < 3


def get_image_rects(page) -> list[pymupdf.Rect]:
    """Get bounding boxes of all images and large colored backgrounds."""
    rects = []
    try:
        for info in page.get_image_info():
            r = pymupdf.Rect(info["bbox"])
            if r.get_area() > 0:
                rects.append(r)
    except Exception:
        pass
    try:
        page_area = page.rect.get_area()
        for d in page.get_drawings():
            fill = d.get("fill")
            if fill and d.get("rect"):
                r = pymupdf.Rect(d["rect"])
                area = r.get_area()
                if area > page_area * 0.15:
                    if any(c < 0.9 for c in fill[:3]):
                        rects.append(r)
    except Exception:
        pass
    return rects


def block_overlaps_image(block_bbox, image_rects: list[pymupdf.Rect]) -> bool:
    for img_rect in image_rects:
        if block_bbox.intersects(img_rect):
            overlap = block_bbox & img_rect
            block_area = block_bbox.get_area()
            if block_area > 0 and overlap.get_area() / block_area > 0.3:
                return True
    return False


def get_effective_insert_bbox(block: dict, image_rects: list[pymupdf.Rect]) -> pymupdf.Rect:
    """Compute the effective insertion bbox for a translated block.

    Adjustments (applied in order):
    1. First-line indent (L=1 only): when the block's first line starts significantly
       to the right of block.x0 (e.g. a short label with leading whitespace), use the
       first line's x0 so the translated text lands at the original text position.
    2. Left-side image push: when an image starts at or before block.x0, push x0
       past the image so inserted text doesn't render under the image.
    3. Image-gap detection (multi-line): when an image fills the horizontal gap between
       block.x0 and the first line's actual text start (the design places text AFTER the
       logo), use first_line.x0. This restores the original visual position after
       redact+insert without the z-order collision from using block.x0.
    """
    block_bbox = block["bbox"]
    effective_x0 = block_bbox.x0
    adjusted_by_line = False

    lines = block.get("lines", [])

    # Compute first-line x0 once (used for both L=1 indent and image-gap detection).
    first_line_x0 = None
    if lines:
        first_spans = lines[0].get("spans", [])
        if first_spans:
            first_line_x0 = min(s["bbox"][0] for s in first_spans)

    # First-line indent: only for single-line blocks. Multi-line blocks (bullet
    # lists, paragraphs) may indent only their first line — using that x0 for the
    # whole block's bbox would clip later lines and break text wrapping.
    if len(lines) == 1 and first_line_x0 is not None:
        if first_line_x0 > block_bbox.x0 + 8:
            effective_x0 = first_line_x0
            adjusted_by_line = True

    if not adjusted_by_line:
        for img in image_rects:
            if not img.intersects(block_bbox):
                continue
            if img.x0 <= effective_x0 + 5:
                # Image starts at or before the block's current left edge — push past it.
                effective_x0 = max(effective_x0, img.x1 + 3)
            elif (
                first_line_x0 is not None
                and block_bbox.x0 < img.x0 <= first_line_x0
            ):
                # Image fills the gap between block.x0 and where the text actually
                # starts (logo-before-text layout). Original text was rendered AFTER
                # the logo; translated text must land at the same position.
                effective_x0 = max(effective_x0, first_line_x0)
                adjusted_by_line = True
                break

    if effective_x0 >= block_bbox.x1 - 20 or effective_x0 == block_bbox.x0:
        return block_bbox
    return pymupdf.Rect(effective_x0, block_bbox.y0, block_bbox.x1, block_bbox.y1)


def build_span_html(parsed_spans: list[dict], scale: float = 1.0) -> str:
    """Build HTML with individual <span> tags preserving per-span styling."""
    html_parts = []
    for span in parsed_spans:
        text = span.get("text", "")
        if not text:
            continue
        color = span.get("color_hex", "#000000")
        weight = "bold" if span.get("bold") else "normal"
        style_italic = "italic" if span.get("italic") else "normal"
        size = (span.get("size") or 10) * scale

        text = clean_symbol_text(text)

        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace("\n", "<br>")

        html_parts.append(
            f'<span style="color:{color}; font-weight:{weight}; '
            f'font-style:{style_italic}; font-size:{size:.1f}px;">{text}</span>'
        )
    return "".join(html_parts)


_DEVA_LO, _DEVA_HI = 0x0900, 0x0DFF  # all Indic blocks


def _line_glyph_colors_01(line: dict) -> list:
    """Distinct glyph colors (RGB 0-1) used by a line's spans — so the inpaint
    mask catches every color in a multi-coloured line, not just the dominant one."""
    seen = set()
    out = []
    for s in line.get("spans", []):
        if not (s.get("text") or "").strip():
            continue
        hexc = (s.get("color_hex") or "#000000").lstrip("#")
        if hexc in seen:
            continue
        seen.add(hexc)
        try:
            out.append((int(hexc[0:2], 16) / 255.0, int(hexc[2:4], 16) / 255.0, int(hexc[4:6], 16) / 255.0))
        except Exception:
            continue
    return out or [(0.0, 0.0, 0.0)]


# Policy/insurance keywords that indicate REAL text, not a decorative badge.
# If any of these appear in the block's text (case-insensitive), the badge
# detector returns False so the text gets translated normally.
_POLICY_BADGE_KEYWORDS = frozenset({
    "policy", "month", "from", "year", "term", "premium", "benefit",
    "option", "plan", "age", "entry", "maturity", "sum", "assured",
    "death", "survival", "income", "lumpsum", "bonus", "rider",
    "guaranteed", "deferment", "payment", "eligibility", "minimum",
    "maximum", "annual", "monthly", "half", "yearly",
})


def _is_styled_badge(page, bbox: pymupdf.Rect, glyph_colors: list, max_font: float,
                     block_text: str = "") -> bool:
    """True for tiny decorative micro-text on a small styled/curved colored emblem
    (e.g. a circular "NEW" ribbon). A rectangular redact + patch can't match a
    non-rectangular colored shape, so such text is left untouched (original
    preserved) — same policy as rotated text and logos.

    Identified by: very small font (<7pt — table cells/body are ≥8pt), small/narrow
    bbox, on a SATURATED colored background with no surrounding white. Validated to
    flag only the emblem and never table cells / headings / body / fine print.

    WHITELIST: if the block's text contains any policy/insurance keyword, it is
    real content (not decoration) and must NOT be skipped — even if it sits on a
    colored badge.  This prevents false positives like "POLICY MONTH 1st from*".
    """
    # Keyword whitelist check — real text is never a decorative badge
    if block_text:
        text_lower = block_text.lower()
        if any(kw in text_lower for kw in _POLICY_BADGE_KEYWORDS):
            return False

    if max_font >= 7.0 or bbox.width >= 45 or bbox.height >= 35:
        return False
    try:
        import numpy as np
    except Exception:
        return False
    r = (bbox + (-2, -2, 2, 2)) & page.rect
    if r.is_empty:
        return False
    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=r, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        flat = arr.reshape(-1, 3).astype(np.int16)
        bg = np.ones(len(flat), bool)
        for c in glyph_colors:
            cc = np.array([int(x * 255) for x in c])
            bg &= np.abs(flat - cc).max(1) > 60  # exclude glyph pixels
        b = flat[bg]
        if len(b) < 10:
            return False
        sat = b.max(1) - b.min(1)
        bright = b.mean(1)
        return float((sat > 60).mean()) > 0.6 and float((bright > 225).mean()) < 0.15
    except Exception:
        return False


# Render scale for the reconstructed background; matches _pil_insert_text SCALE so
# a clip crop maps 1:1 to the text canvas width.
_BG_SCALE = 3


def _insert_invisible_text_layer(page, clip: pymupdf.Rect, final_lines: list, language: str):
    """Write an INVISIBLE (render_mode=3) Unicode text layer over the visible PIL
    image, so the translated Indic text is selectable / searchable / copyable.

    The visible glyphs come from the PIL image (correct OpenType shaping). PyMuPDF
    can't shape Indic conjuncts as live text — but for an invisible layer the
    visual shaping is irrelevant; only the underlying Unicode matters, and that
    round-trips correctly through TextWriter + an embedded Noto CID font (verified:
    extract == input, search hits, 0 visible pixels). Positions follow the same
    line breaks as the image so selection highlights line up reasonably.
    """
    if clip.is_empty or not final_lines:
        return
    script_font = get_pymupdf_font(language)
    latin_font = get_latin_pymupdf_font()
    if script_font is None and latin_font is None:
        return

    def _font_for(word: str):
        if any(_DEVA_LO <= ord(c) <= _DEVA_HI for c in word):
            return script_font or latin_font
        return latin_font or script_font

    n = len(final_lines)
    line_h = clip.height / n
    tw = pymupdf.TextWriter(page.rect)
    wrote = False
    for i, line_words in enumerate(final_lines):
        words = [w["text"] for (w, _f) in line_words if w.get("text", "").strip()]
        if not words:
            continue
        text = " ".join(words)
        # Size to fit this line's width within the clip (cap to the line slot).
        fontsize = max(4.0, min(line_h * 0.85, 16.0))
        f0 = _font_for(text)
        try:
            tl = f0.text_length(text, fontsize=fontsize)
            if tl > clip.width - 2 and tl > 0:
                fontsize = max(3.0, fontsize * (clip.width - 2) / tl)
        except Exception:
            pass
        baseline = clip.y0 + i * line_h + line_h * 0.78
        x = clip.x0 + 1.0
        for wi, word in enumerate(words):
            wtext = word if wi == len(words) - 1 else word + " "
            f = _font_for(word)
            if f is None:
                continue
            try:
                tw.append((x, baseline), wtext, font=f, fontsize=fontsize)
                x += f.text_length(wtext, fontsize=fontsize)
                wrote = True
            except Exception:
                # Skip a word the font can't encode (rare symbol) — keep the rest.
                try:
                    x += f.text_length(wtext, fontsize=fontsize)
                except Exception:
                    x += fontsize * len(wtext) * 0.5
    if wrote:
        try:
            tw.write_text(page, render_mode=3)  # 3 = invisible (no fill/stroke)
        except Exception as e:
            logger.warning(f"  [InvisibleText] write_text failed at {clip}: {e}")


def _pil_insert_text(
    page,
    bbox: pymupdf.Rect,
    parsed_spans: list[dict],
    language: str,
    align: str = "left",
    clean_bg=None,
) -> bool:
    """Render translated Indic text via PIL for correct OpenType shaping.

    Preserves PER-WORD styling: each word keeps the color, weight, and size of
    the span it came from (via proportional style mapping), so a single
    highlighted word inside a sentence stays its own color in the translation.
    Renders with Noto fonts via PIL/FreeType (proper GSUB/GPOS shaping) and
    inserts as a transparent PNG. Shrinks all sizes uniformly until the text
    fits the bbox. Falls back gracefully to False on any error so the htmlbox
    path can retry.
    """
    try:
        from PIL import Image, ImageDraw
        import io
        import re as _re
    except ImportError:
        return False

    # ── Flatten spans into a per-character style array ────────────────────────
    # Word tokens are then cut from the combined text; a word takes the majority
    # style of its characters, so span boundaries that fall mid-word (rare after
    # word-boundary snapping in apply_proportional_styles) cannot split a word.
    chars: list[str] = []
    char_styles: list[tuple] = []  # (size, color_hex, bold)
    for s in parsed_spans:
        text = clean_symbol_text((s.get("text", "") or "").replace("\n", " "))
        if not text:
            continue
        st = (float(s.get("size") or 10), s.get("color_hex") or "#000000", bool(s.get("bold")))
        chars.extend(text)
        char_styles.extend([st] * len(text))
        # NOTE: do NOT insert a space between spans. The spans are contiguous slices
        # of one translated string (real spaces already inside the slices), so
        # joining directly keeps words whole. An inserted space could split a
        # Devanagari cluster at a style boundary (e.g. "विकल्प" → "विकल" + "्प"),
        # leaving a fragment that starts with a virama/matra → a stray detached mark.
    full_text = "".join(chars)
    if not full_text.strip():
        return False

    words: list[dict] = []
    for m in _re.finditer(r"\S+", full_text):
        size, color_hex, bold = Counter(char_styles[m.start():m.end()]).most_common(1)[0][0]
        try:
            color = (int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16))
        except Exception:
            color = (0, 0, 0)
        words.append({"text": m.group(), "size": size, "color": color, "bold": bold})
    if not words:
        return False

    base_size = max(w["size"] for w in words)

    clip = bbox & page.rect
    if clip.is_empty:
        return False

    SCALE = 3
    w_px = max(4, int(clip.width * SCALE))
    h_px = max(4, int(clip.height * SCALE))

    _DEVA_LOW, _DEVA_HIGH = 0x0900, 0x0DFF  # covers all Indic blocks

    def _is_indic(text):
        return any(_DEVA_LOW <= ord(c) <= _DEVA_HIGH for c in text)

    # Font cache for this call: (indic, bold, size_px) → FreeTypeFont.
    # Indic words use the script's Noto font; everything else (hashtags, kept
    # English, numerics) uses the Latin fallback — NotoSansDevanagari etc. have
    # no Latin glyphs and would render tofu boxes.
    font_cache: dict = {}

    def _font_for(word, shrink):
        size_px = max(4, int(word["size"] * SCALE * shrink))
        key = (_is_indic(word["text"]), word["bold"], size_px)
        f = font_cache.get(key)
        if f is None:
            f = (
                get_pil_font(language, size_px, bold=word["bold"])
                if key[0] else _get_latin_pil_font(size_px, bold=word["bold"])
            )
            font_cache[key] = f
        return f

    def _width(text, font):
        try:
            bb = font.getbbox(text)
            return bb[2] - bb[0]
        except Exception:
            return len(text) * int(base_size * SCALE) // 2

    def _layout(shrink):
        """Wrap styled words into lines at this shrink factor.
        Returns (lines, line_height, space_width); lines = [[(word, font)], …]."""
        sized = [(w, _font_for(w, shrink)) for w in words]
        ref_font = max((f for _, f in sized), key=lambda f: getattr(f, "size", 0))
        try:
            sp_w = ref_font.getbbox(" ")[2] - ref_font.getbbox(" ")[0]
        except Exception:
            sp_w = max(2, int(base_size * SCALE * shrink) // 3)

        lines, cur, cur_w = [], [], 0
        for w, f in sized:
            ww = _width(w["text"], f)
            needed = (cur_w + sp_w + ww) if cur else ww
            if needed <= w_px - 4 or not cur:
                cur.append((w, f))
                cur_w = needed
            else:
                lines.append(cur)
                cur = [(w, f)]
                cur_w = ww
        if cur:
            lines.append(cur)

        lh = 0
        for _, f in sized:
            try:
                lh = max(lh, f.getbbox("Ag")[3])
            except Exception:
                lh = max(lh, getattr(f, "size", int(base_size * SCALE)))
        return lines, lh + SCALE, sp_w

    def _one_line(shrink):
        """Single-line (no-wrap) layout at this shrink: (words+fonts, total_w, lh, sp_w)."""
        sized = [(w, _font_for(w, shrink)) for w in words]
        ref_font = max((f for _, f in sized), key=lambda f: getattr(f, "size", 0))
        try:
            sp_w = ref_font.getbbox(" ")[2] - ref_font.getbbox(" ")[0]
        except Exception:
            sp_w = max(2, int(base_size * SCALE * shrink) // 3)
        total = sum(_width(w["text"], f) for w, f in sized) + sp_w * max(0, len(sized) - 1)
        lh = 0
        for _, f in sized:
            try:
                lh = max(lh, f.getbbox("Ag")[3])
            except Exception:
                lh = max(lh, getattr(f, "size", int(base_size * SCALE)))
        return sized, total, lh + SCALE, sp_w

    # Is this a SINGLE-LINE source box (heading / label / table cell)? If the box
    # is only ~one line tall, wrapping the (usually wider) translation to multiple
    # lines forces an unreadably small font — keep it on ONE line and just shrink
    # the font to fit the width instead. Tall boxes (paragraphs) keep wrapping.
    _, _, lh1, _ = _one_line(1.0)
    single_line_box = h_px <= 1.6 * lh1

    final_lines = final_lh = final_sp_w = None
    if single_line_box:
        s = 1.0
        while s >= 0.30:
            sized, total, lh, sp_w = _one_line(round(s, 2))
            if total <= w_px - 4:
                final_lines, final_lh, final_sp_w = [sized], lh, sp_w
                break
            s -= 0.05
        if final_lines is None:  # still too wide even tiny → smallest one-line
            sized, total, lh, sp_w = _one_line(0.30)
            final_lines, final_lh, final_sp_w = [sized], lh, sp_w

    if final_lines is None:
        shrink = 1.0
        while shrink >= 0.34:
            lines, lh, sp_w = _layout(round(shrink, 2))
            if lh * len(lines) <= h_px:
                final_lines, final_lh, final_sp_w = lines, lh, sp_w
                break
            shrink -= 0.07
        if final_lines is None:
            final_lines, final_lh, final_sp_w = _layout(0.34)

    # TRANSPARENT glyphs-only canvas: only the text pixels are opaque. The erase
    # now removes the source text with NO fill (fill=False), so the REAL page
    # background — gradient, photo, colored cell, dark bar — is intact underneath
    # this region. We draw only the glyphs and let that true background show
    # through (no opaque patch → no seam, no reconstruction). The `clean_bg`
    # parameter is retained for signature compatibility but unused.
    # Canvas is at least as tall as the text so a single line whose glyph height
    # exceeds a very short source box is never clipped — insert_image then scales
    # the canvas into the (short) bbox (slight squash, better than clipping).
    total_h = final_lh * len(final_lines)
    canvas_h = max(h_px, total_h)
    img = Image.new("RGBA", (w_px, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y_start = max(0, (canvas_h - total_h) // 2)

    def _build_runs(line_words):
        """Merge consecutive words sharing the same font AND color into one run.

        Rendering a whole run as one string lets FreeType/HarfBuzz shape it
        naturally (correct inter-word spacing, no per-word boundary glitches).
        Words with a different font or color (e.g. one highlighted word) get
        their own run so their style is preserved exactly.
        """
        runs: list[list] = []  # [text, font, color]
        for w, f in line_words:
            if runs and runs[-1][1] is f and runs[-1][2] == w["color"]:
                runs[-1][0] = runs[-1][0] + " " + w["text"]
            else:
                runs.append([w["text"], f, w["color"]])
        return runs

    for i, line_words in enumerate(final_lines):
        y = y_start + i * final_lh
        runs = _build_runs(line_words)

        # Compute total line width for alignment
        run_widths = [_width(text, font) for text, font, _ in runs]
        line_w = sum(run_widths) + final_sp_w * (len(runs) - 1)

        if align == "center":
            x = max(0, (w_px - line_w) // 2)
        elif align == "right":
            x = max(0, w_px - line_w - 4)
        else:
            x = 4

        for ri, (run_text, font, color) in enumerate(runs):
            draw.text((x, y), run_text, font=font, fill=color)
            x += run_widths[ri]
            if ri < len(runs) - 1:
                x += final_sp_w

    try:
        img_rgb = img.convert("RGB")
        img_mask = img.split()[3]  # alpha → stencil mask: only glyph pixels paint
        buf_rgb = io.BytesIO(); img_rgb.save(buf_rgb, format="PNG")
        buf_mask = io.BytesIO(); img_mask.save(buf_mask, format="PNG")
        page.insert_image(
            clip,
            stream=buf_rgb.getvalue(),
            mask=buf_mask.getvalue(),
            keep_proportion=False,
        )
        # Add the invisible selectable/searchable Unicode layer over the image.
        _insert_invisible_text_layer(page, clip, final_lines, language)
        return True
    except Exception as e:
        logger.warning(f"  [PIL] insert_image failed at {clip}: {e}")
        return False


def insert_translated_text(
    page, bbox, parsed_spans: list[dict], base_css: str,
    archive=None, min_scale: float = 0.55, align: str = "left",
    language: Optional[str] = None, clean_bg=None,
):
    """Insert translated text into the page with progressive overflow handling.

    For complex Indic scripts (Devanagari, Tamil, etc.), PyMuPDF's htmlbox
    renderer cannot correctly shape OpenType conjuncts/ligatures. The PIL path
    is tried first: it uses FreeType with the Noto font's GSUB/GPOS tables,
    producing properly shaped glyphs (e.g. 'पर्यंत' instead of 'पǾयत').
    If PIL rendering fails for any reason, we fall through to insert_htmlbox.

    `clean_bg` is the reconstructed background (gradient/texture preserved) used
    behind the glyphs so the patch matches the original background exactly.
    """
    from .config import COMPLEX_SCRIPTS, LANGUAGE_CONFIG, TEXT_RENDER_MODE

    if language:
        script = LANGUAGE_CONFIG.get(language, {}).get("script", "")
        if script in COMPLEX_SCRIPTS:
            # Opt-in vector renderer (crisp glyph outlines); falls back to raster.
            if TEXT_RENDER_MODE == "vector":
                try:
                    from .vector_text import vector_insert_text
                    if vector_insert_text(page, bbox, parsed_spans, language, align, clean_bg=clean_bg):
                        return True
                except Exception as e:
                    logger.warning(f"  [Vector] insert failed ({type(e).__name__}: {str(e)[:100]}) — raster fallback")
            if _pil_insert_text(page, bbox, parsed_spans, language, align, clean_bg=clean_bg):
                return True

    all_steps = [
        (1.0, 1.2), (0.85, 1.2), (0.70, 1.15),
        (0.70, 1.0), (0.60, 1.0),
    ]
    attempts = [(scale, lh) for scale, lh in all_steps if scale >= min_scale]

    for scale, line_height in attempts:
        html_text = build_span_html(parsed_spans, scale=scale)
        css = base_css + f" * {{ background: transparent; line-height: {line_height}; text-align: {align}; }}"
        try:
            kwargs = {"css": css}
            if archive:
                kwargs["archive"] = archive
            result = page.insert_htmlbox(bbox, html_text, **kwargs)
            if isinstance(result, (float, int)):
                if result >= 0:
                    return True
            elif isinstance(result, tuple):
                if result[0] >= 0:
                    return True
        except Exception:
            continue

    try:
        html_text = build_span_html(parsed_spans, scale=min_scale)
        css = base_css + f" * {{ background: transparent; line-height: 1.0; text-align: {align}; }}"
        kwargs = {"css": css}
        if archive:
            kwargs["archive"] = archive
        result = page.insert_htmlbox(bbox, html_text, **kwargs)
        overflowed = (isinstance(result, (float, int)) and result < 0) or (
            isinstance(result, tuple) and result[0] < 0
        )
        if overflowed:
            logger.warning(f"  [Layout] Overflow at min_scale={min_scale}, bbox={bbox}")
            return False
    except Exception as e:
        logger.error(f"Failed to insert text at min scale {min_scale}: {e}")
        return False
    return True


async def translate_page_text_mode(
    page, page_num, target_language, target_script,
    ws_manager, job_id, doc_index, total_pages,
    inventory: Optional[dict] = None,
) -> dict:
    """TEXT MODE: Translate a single page via extract → translate → redact → insert.

    Returns a per-page report: failed_pages, completeness metrics, needs_review,
    and residual_bboxes (Rect list) for the review artifact.

    `inventory` is a pre-extracted page snapshot from extract_page_inventory().
    When provided, the extraction step is skipped — blocks and table regions are
    reused from the shared cache rather than re-extracted per language.
    """
    loop = asyncio.get_event_loop()

    if inventory is not None:
        blocks = inventory["blocks"]
        table_regions = inventory["tables"]
        logger.info(
            f"  [{target_language}] Page {page_num + 1}/{total_pages}: "
            f"{len(blocks)} block(s), {len(table_regions)} table(s) [from cache]"
        )
    else:
        logger.info(f"  [{target_language}] Page {page_num + 1}/{total_pages}: extracting...")
        blocks, table_regions = await loop.run_in_executor(None, process_page_extraction, page)

    if not blocks:
        logger.info(f"  [{target_language}] Page {page_num + 1}: no text, skipping")
        await ws_manager.broadcast(job_id, {
            "type": "page:progress", "docIndex": doc_index,
            "language": target_language, "page": page_num + 1,
            "totalPages": total_pages, "phase": "done",
        })
        return {"failed_pages": [], "pct_translated": 1.0, "residual_count": 0,
                "tofu_count": 0, "translatable_blocks": 0, "needs_review": False,
                "residual_bboxes": []}

    if inventory is None:
        logger.info(f"  [{target_language}] Page {page_num + 1}: {len(blocks)} block(s), {len(table_regions)} table(s)")
    failed_pages = []

    blocks_for_translation = []
    blocks_metadata = []

    for i, block in enumerate(blocks):
        block_text = get_block_text(block)

        if not should_translate(block_text):
            blocks_metadata.append({"translate": False})
            continue

        # Skip blocks that are ONLY a license/registration code (CIN, UIN, IRDAI).
        # A disclaimer paragraph that merely contains a code is still translated —
        # the code itself is preserved by the translation prompt.
        if _LICENSE_RE.search(block_text) and _is_code_only_block(block_text):
            blocks_metadata.append({"translate": False})
            continue

        # Leave decorative micro-text on styled/curved colored emblems untouched
        # (rectangular redact+patch can't match a non-rectangular badge shape).
        _mf = max(
            (s["size"] for ln in block["lines"] for s in ln["spans"] if s.get("text", "").strip()),
            default=0,
        )
        _gc = []
        for ln in block["lines"]:
            _gc += _line_glyph_colors_01(ln)
        if _is_styled_badge(page, block["bbox"], _gc, _mf, block_text=block_text):
            logger.info(
                f"  [{target_language}] Page {page_num + 1}: leaving styled badge "
                f"untouched — {block_text[:24]!r}"
            )
            blocks_metadata.append({"translate": False})
            continue

        style_segments = get_block_style_segments(block)
        blocks_for_translation.append({"text": block_text, "block_index": len(blocks_metadata)})
        blocks_metadata.append({
            "translate": True,
            "style_segments": style_segments,
            "original_length": len(block_text),
            "translation_index": len(blocks_for_translation) - 1,
        })

    page_context = "\n".join(b["text"] for b in blocks_for_translation)

    translations = []
    if blocks_for_translation:
        try:
            translations = await translate_batch_with_gemini(
                blocks_for_translation, target_language, target_script,
                page_context=page_context,
            )
        except Exception as e:
            logger.error(f"Translation failed page {page_num + 1}: {e}")
            failed_pages.append(page_num + 1)
            translations = [b["text"] for b in blocks_for_translation]

    # ── Completeness verification + bounded repair (on translation strings) ──
    # Verify the translator's OUTPUT strings, not a re-extraction of the rendered
    # PDF: PyMuPDF can't round-trip Indic conjuncts written via insert_htmlbox
    # (वर्ष re-extracts as 'व�ष'), so re-reading the output gives false signals.
    # Use the language-specific handler's residual check (isolated per language).
    from .languages import get_language_handler
    _lang_handler = get_language_handler(target_language)

    src_texts = [b["text"] for b in blocks_for_translation]
    residual_idx = [
        i for i, t in enumerate(translations)
        if should_translate(src_texts[i]) and _lang_handler.is_residual(src_texts[i], t or "")
    ]
    if residual_idx:
        logger.info(
            f"  [{target_language}] Page {page_num + 1}: "
            f"{len(residual_idx)} block(s) left in English — re-translating"
        )
        repair_payload = [{"text": src_texts[i]} for i in residual_idx]
        try:
            retry = await translate_batch_with_gemini(
                repair_payload, target_language, target_script, page_context=page_context
            )
            for j, i in enumerate(residual_idx):
                if j < len(retry) and retry[j] and not _lang_handler.is_residual(src_texts[i], retry[j]):
                    translations[i] = retry[j]
        except Exception as e:
            logger.warning(f"  [{target_language}] Page {page_num + 1}: residual repair failed ({e})")

    translated_spans_per_block = []
    translated_block_indices: set[int] = set()

    for block_idx, meta in enumerate(blocks_metadata):
        if not meta.get("translate"):
            translated_spans_per_block.append([])
        else:
            idx = meta["translation_index"]
            translated_text = (
                translations[idx] if idx < len(translations)
                else blocks_for_translation[idx]["text"]
            )
            styled_spans = apply_proportional_styles(
                translated_text, meta["style_segments"], meta["original_length"],
            )
            translated_spans_per_block.append(styled_spans)
            translated_block_indices.add(block_idx)

    # ── Erase + text insertion ────────────────────────────────────────────────
    font_css = get_font_css(target_language)
    archive = get_font_archive()

    # Image rects must be captured BEFORE any patches are inserted — the patches
    # are images themselves and would otherwise shift the insert positions.
    image_rects = get_image_rects(page)

    # PRIMARY ERASE — "like Word": delete ONLY the source-text glyph operators from
    # the content stream and reveal whatever was underneath. In a native PDF the
    # background (vector fills, gradients, photos, colored cells) is a SEPARATE
    # object below the text, so removing the text with NO fill exposes the true
    # background pixel-perfect — no detected-colour fill, no inpainting, no patch,
    # and zero residue (we delete glyph objects, not pixels). images/graphics are
    # kept so only text is removed.
    clean_bg = None  # text renderers draw straight on the revealed real background

    # Identify non-translated blocks that OVERLAP or are ADJACENT to (within 8px
    # vertically) a translated block's row.  These are blocks we chose NOT to
    # translate (pure numbers, codes like "6 8 10 12 15") but whose original
    # glyphs sit in the same visual row as a translated block.  Translated Indic
    # text is wider per glyph and can overflow its source bbox horizontally,
    # painting over adjacent numbers in the same row.  We redact these too and
    # re-insert their original text AFTER the translated blocks are rendered.
    overlap_block_indices: list[int] = []
    translated_bboxes = [pymupdf.Rect(blocks[i]["bbox"]) for i in translated_block_indices]
    for i, block in enumerate(blocks):
        if i in translated_block_indices:
            continue
        b_bbox = pymupdf.Rect(block["bbox"])
        for t_bbox in translated_bboxes:
            # Direct overlap
            if b_bbox.intersects(t_bbox):
                overlap_block_indices.append(i)
                break
            # Same-row adjacency: vertical gap ≤ 8px (text overflow zone)
            # and horizontal overlap OR the non-translated block is within the
            # same horizontal band as the translated block's row.
            v_gap = min(abs(b_bbox.y0 - t_bbox.y1), abs(t_bbox.y0 - b_bbox.y1))
            if v_gap <= 8:
                # Check if they share the same horizontal row (any x overlap
                # OR the number sits between two translated blocks in the same row)
                h_overlap = min(b_bbox.x1, t_bbox.x1) - max(b_bbox.x0, t_bbox.x0)
                if h_overlap > 0:
                    overlap_block_indices.append(i)
                    break
                # Also check: is the number between this translated block and
                # another translated block in the same row? (numbers between
                # two translated cells)
                if b_bbox.x0 >= t_bbox.x0 - 5:
                    # Check if there's another translated block to the right
                    for t2_bbox in translated_bboxes:
                        if (t2_bbox is not t_bbox and
                            abs(t2_bbox.y0 - t_bbox.y0) < 10 and
                            t2_bbox.x1 >= b_bbox.x0):
                            overlap_block_indices.append(i)
                            break
                    if i in overlap_block_indices:
                        break

    # Redact translated blocks AND overlapping non-translated blocks.
    # For overlapping blocks we use fill=True (white) because their text
    # will be re-inserted — the fill ensures the old glyphs are fully erased
    # even when apply_redactions is conservative with TJ operators.
    for i, block in enumerate(blocks):
        if i in translated_block_indices:
            for line in block["lines"]:
                page.add_redact_annot(line["bbox"], fill=False)
        elif i in overlap_block_indices:
            for line in block["lines"]:
                page.add_redact_annot(line["bbox"], fill=False)
    page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_NONE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
    )

    # Insert translated loose blocks as selectable PDF text at the original
    # coordinates. Each insert is isolated: one block failing must not drop the
    # rest of the page's translations.
    #
    # OVERLAP PREVENTION: blocks are sorted top-to-bottom, left-to-right before
    # insertion so that adjacent blocks don't overwrite each other.  Insert
    # bboxes are clamped to the page rect so text can't render off-page.  A
    # collision check skips blocks whose bbox overlaps a previously rendered
    # block by >60% — those would smear the earlier text.

    page_rect = page.rect

    def _clamp_to_page(rect: pymupdf.Rect) -> pymupdf.Rect:
        """Clamp an insert rect to the page bounds so text never renders off-page."""
        x0 = max(0, min(rect.x0, page_rect.x1 - 4))
        y0 = max(0, min(rect.y0, page_rect.y1 - 4))
        x1 = max(x0 + 4, min(rect.x1, page_rect.x1))
        y1 = max(y0 + 4, min(rect.y1, page_rect.y1))
        return pymupdf.Rect(x0, y0, x1, y1)

    def _overlap_ratio(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
        """How much of rect A overlaps rect B (0.0 – 1.0)."""
        if not a.intersects(b):
            return 0.0
        inter = a & b
        a_area = a.get_area()
        if a_area <= 0:
            return 0.0
        return inter.get_area() / a_area

    # Sort translated block indices by position: top-to-bottom, then left-to-right
    sorted_indices = sorted(
        translated_block_indices,
        key=lambda i: (blocks[i]["bbox"].y0, blocks[i]["bbox"].x0),
    )

    rendered_rects: list[pymupdf.Rect] = []
    insert_failures = 0
    for i in sorted_indices:
        block = blocks[i]
        parsed_spans = translated_spans_per_block[i] if i < len(translated_spans_per_block) else None
        if not parsed_spans or not any(s.get("text", "").strip() for s in parsed_spans):
            continue
        align = detect_block_alignment(block)
        if align == "left" and block["bbox"].width < 80:
            block_cx = (block["bbox"].x0 + block["bbox"].x1) / 2
            column_matches = sum(
                1 for other in blocks
                if other is not block
                and abs((other["bbox"].x0 + other["bbox"].x1) / 2 - block_cx) < 4
                and abs(other["bbox"].y0 - block["bbox"].y0) > 5
            )
            if column_matches >= 1:
                align = "center"
        dominant_size = max((s.get("size", 10) for s in parsed_spans), default=10)
        min_scale = 0.75 if dominant_size < 9 else 0.55
        insert_bbox = get_effective_insert_bbox(block, image_rects)

        # Clamp insert bbox to page bounds
        insert_bbox = _clamp_to_page(insert_bbox)

        # Skip if this block's insert area overlaps a previously rendered block
        # by >60% — rendering would smear the earlier text.
        skip = False
        for prev_rect in rendered_rects:
            if _overlap_ratio(insert_bbox, prev_rect) > 0.60:
                logger.warning(
                    f"  [{target_language}] Page {page_num + 1}: block {i} overlaps "
                    f"a previously rendered block — skipping to avoid smear"
                )
                insert_failures += 1
                skip = True
                break
        if skip:
            continue

        try:
            ok = insert_translated_text(
                page, insert_bbox, parsed_spans, font_css, archive,
                min_scale=min_scale, align=align, language=target_language,
                clean_bg=clean_bg,
            )
            if ok is False:
                insert_failures += 1
                logger.warning(
                    f"  [{target_language}] Page {page_num + 1}: block {i} did not "
                    f"fit/insert (overflow) — flagged for review"
                )
            else:
                rendered_rects.append(insert_bbox)
        except Exception as e:
            insert_failures += 1
            logger.error(
                f"  [{target_language}] Page {page_num + 1}: block {i} insert raised "
                f"({type(e).__name__}: {str(e)[:120]}) — continuing"
            )

    # ── Re-insert non-translated blocks that were redacted because they
    #    overlapped a translated block (e.g. "6 8 10 12 15" in a table header).
    #    These keep their original text — we just need to put them back at their
    #    original positions so they're not lost under the translated text.
    for i in overlap_block_indices:
        block = blocks[i]
        orig_text = "".join(
            span["text"] for line in block["lines"] for span in line["spans"]
        ).strip()
        if not orig_text:
            continue
        # Build parsed_spans from the original block's spans (preserve sizes/colors)
        orig_spans = []
        for line in block["lines"]:
            for span in line["spans"]:
                srgb = span.get("color", 0)
                # PyMuPDF color int → hex string
                r = int((srgb >> 16) & 0xFF)
                g = int((srgb >> 8) & 0xFF)
                b = int(srgb & 0xFF)
                color_hex = f"#{r:02x}{g:02x}{b:02x}"
                flags = span.get("flags", 0)
                orig_spans.append({
                    "text": span["text"],
                    "size": span["size"],
                    "color_hex": color_hex,
                    "bold": bool(flags & 2**4),   # bit 4 = bold
                    "italic": bool(flags & 2**1),  # bit 1 = italic
                })
        orig_bbox = _clamp_to_page(pymupdf.Rect(block["bbox"]))
        try:
            ok = insert_translated_text(
                page, orig_bbox, orig_spans, font_css, archive,
                min_scale=0.55, align=detect_block_alignment(block),
                language=None,  # Latin — no Indic shaping needed
                clean_bg=clean_bg,
            )
            if ok:
                rendered_rects.append(orig_bbox)
            else:
                logger.warning(
                    f"  [{target_language}] Page {page_num + 1}: overlapping "
                    f"non-translated block {i} re-insert failed — content lost"
                )
        except Exception as e:
            logger.error(
                f"  [{target_language}] Page {page_num + 1}: overlapping "
                f"non-translated block {i} re-insert raised "
                f"({type(e).__name__}: {str(e)[:120]}) — continuing"
            )

    # ── Post-insert verification ────────────────────────────────────────────
    # Check that SOMETHING was rendered: either extractable text (vector/htmlbox
    # path) or new images (PIL raster path inserts transparent PNGs).
    # The PIL path inserts Hindi text as an image — get_text() won't find it,
    # but get_image_info() will show new images in the text regions.
    if translated_block_indices:
        post_text = page.get_text().strip()
        post_images = page.get_image_info()
        post_drawings = len(page.get_drawings())

        # The invisible TextWriter layer should produce some extractable chars
        # even for the PIL path. But if it fails (e.g. font encoding issue),
        # check for images too.  Vector glyph outlines (uharfbuzz path) show
        # up as filled drawings — count those too so the check doesn't cry
        # wolf when the visible text is vector-rendered but the invisible
        # Unicode layer didn't round-trip through PyMuPDF's text extractor.
        if len(post_text) == 0 and len(post_images) == 0 and post_drawings < 100 and len(translated_block_indices) > 0:
            insert_failures = len(translated_block_indices)
            logger.error(
                f"  [{target_language}] Page {page_num + 1}: POST-INSERT CHECK "
                f"FAILED — 0 chars, 0 images, {post_drawings} drawings after "
                f"inserting {len(translated_block_indices)} block(s). All inserts "
                f"may have failed silently (off-page bbox or transparent render)."
            )
        elif len(post_text) == 0 and len(translated_block_indices) > 0:
            if post_drawings >= 100:
                # Vector renderer (uharfbuzz) drew glyph outlines — text is
                # visually present but the invisible Unicode layer didn't
                # extract.  Log as info, not error.
                logger.info(
                    f"  [{target_language}] Page {page_num + 1}: {post_drawings} "
                    f"vector glyph(s) on page (vector path — text is in drawing layer)"
                )
            else:
                # Text is image-only (PIL raster path) — this is expected when
                # uharfbuzz is missing. Log as info, not error.
                logger.info(
                    f"  [{target_language}] Page {page_num + 1}: {len(post_images)} "
                    f"image(s) on page (PIL raster path — text is in image layer)"
                )

    # ── Completeness score (from clean translation strings) ─────────────────
    # Pair each translated cell with its SOURCE cell (not itself) so legitimately
    # kept English (codes, brand islands) isn't counted as a residual — same basis
    # as the repair logic.
    block_pairs = list(zip(src_texts, translations))
    report = residual_translation_report(block_pairs)
    needs_review = (
        report["residual_count"] > 0
        or report["tofu_count"] > 0
        or report["pct_translated"] < 0.995
        or insert_failures > 0  # a block that overflowed / failed to insert
    )

    # Build bbox list for residual loose blocks (used by review PDF overlay).
    n_block_pairs = len(block_pairs)
    residual_bboxes = []
    for idx in report.get("residual_idx", []):
        if idx < n_block_pairs and idx < len(blocks_for_translation):
            block_pos = blocks_for_translation[idx].get("block_index", -1)
            if 0 <= block_pos < len(blocks):
                residual_bboxes.append(blocks[block_pos]["bbox"])

    logger.info(
        f"  [{target_language}] Page {page_num + 1}: DONE — "
        f"pct_translated={report['pct_translated'] * 100:.1f}% "
        f"residual={report['residual_count']} tofu={report['tofu_count']} "
        f"needs_review={needs_review}"
    )
    await ws_manager.broadcast(job_id, {
        "type": "page:progress", "docIndex": doc_index,
        "language": target_language, "page": page_num + 1,
        "totalPages": total_pages, "phase": "done",
        "pctTranslated": round(report["pct_translated"] * 100, 1),
        "residualBlocks": report["residual_count"],
        "tofuBlocks": report["tofu_count"],
        "needsReview": needs_review,
    })
    return {
        "failed_pages": failed_pages,
        "pct_translated": report["pct_translated"],
        "residual_count": report["residual_count"],
        "tofu_count": report["tofu_count"],
        "translatable_blocks": report["translatable"],
        "needs_review": needs_review,
        "residual_bboxes": residual_bboxes,
    }
