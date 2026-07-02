"""Vector glyph-outline text insertion (alternative to the raster PIL path).

This is a SEPARATE, opt-in renderer kept alongside the default raster renderer
(`_pil_insert_text` in text_mode.py). Switch with TEXT_RENDER_MODE=vector.

How it differs from the raster path:
  - Raster: PIL/FreeType renders the translated text to a PNG, pasted as an image.
  - Vector: HarfBuzz shapes the text → fontTools gives each glyph's outline →
    the outlines are drawn as FILLED VECTOR PATHS directly on the page.

Benefits: crisp at any zoom / print resolution, no raster text patch (so flat
backgrounds need no patch at all — glyphs draw straight over the redaction fill),
much smaller output. Correct Indic conjuncts (HarfBuzz) and letter-holes
(even-odd fill). Fully local (uharfbuzz + fontTools) → data-residency safe.

Selectability is still provided by the invisible TextWriter layer (shared with the
raster path), since vector paths themselves are not selectable.
"""
from collections import Counter
from pathlib import Path
from typing import Optional

import pymupdf

from .config import FONTS_DIR, LANGUAGE_CONFIG, logger

_DEVA_LO, _DEVA_HI = 0x0900, 0x0DFF  # all Indic blocks
_LATIN_REG = "NotoSans-Regular.ttf"
_LATIN_BOLD = "NotoSans-Bold.ttf"

# ── lazy, cached shaping / outline machinery ──────────────────────────────────
_hb_cache: dict = {}
_tt_cache: dict = {}
_contour_cache: dict = {}
_FlattenPenCls = None


def _flatten_pen_cls():
    """Build the curve-flattening pen class lazily (needs fontTools at runtime)."""
    global _FlattenPenCls
    if _FlattenPenCls is None:
        from fontTools.pens.basePen import BasePen

        class _FlattenPen(BasePen):
            """Decompose a glyph outline into flat polygon contours (font units).

            BasePen handles TrueType implied on-curve points and calls these with
            single segments; we sample curves into short line runs."""

            def __init__(self, glyphSet, steps=10):
                super().__init__(glyphSet)
                self.contours = []
                self.cur = None
                self.steps = steps

            def _moveTo(self, p):
                self.cur = [(p[0], p[1])]

            def _lineTo(self, p):
                self.cur.append((p[0], p[1]))

            def _curveToOne(self, p1, p2, p3):
                x0, y0 = self.cur[-1]
                for i in range(1, self.steps + 1):
                    t = i / self.steps
                    m = 1 - t
                    self.cur.append((
                        m**3 * x0 + 3 * m * m * t * p1[0] + 3 * m * t * t * p2[0] + t**3 * p3[0],
                        m**3 * y0 + 3 * m * m * t * p1[1] + 3 * m * t * t * p2[1] + t**3 * p3[1],
                    ))

            def _qCurveToOne(self, p1, p2):
                x0, y0 = self.cur[-1]
                for i in range(1, self.steps + 1):
                    t = i / self.steps
                    m = 1 - t
                    self.cur.append((
                        m * m * x0 + 2 * m * t * p1[0] + t * t * p2[0],
                        m * m * y0 + 2 * m * t * p1[1] + t * t * p2[1],
                    ))

            def _closePath(self):
                if self.cur and len(self.cur) > 2:
                    self.contours.append(self.cur)
                self.cur = None

            _endPath = _closePath

        _FlattenPenCls = _FlattenPen
    return _FlattenPenCls


def _hb(fp: str):
    """Cached (HarfBuzz font scaled to font units, units-per-em)."""
    if fp not in _hb_cache:
        import uharfbuzz as hb
        blob = hb.Blob.from_file_path(fp)
        face = hb.Face(blob)
        font = hb.Font(face)
        font.scale = (face.upem, face.upem)
        _hb_cache[fp] = (font, face.upem)
    return _hb_cache[fp]


def _tt(fp: str):
    """Cached (TTFont, glyphSet, glyphOrder, hhea ascent/descent in font units)."""
    if fp not in _tt_cache:
        from fontTools.ttLib import TTFont
        t = TTFont(fp)
        try:
            asc, desc = t["hhea"].ascent, t["hhea"].descent
        except Exception:
            _, upem = _hb(fp)
            asc, desc = int(upem * 0.8), int(-upem * 0.2)
        _tt_cache[fp] = (t, t.getGlyphSet(), t.getGlyphOrder(), asc, desc)
    return _tt_cache[fp]


# macOS/Linux system Latin fonts, tried only if the bundled Noto Latin is absent
# (dev safety net; production should have NotoSans-*.ttf in FONTS_DIR).
_SYSTEM_LATIN = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _first_existing(*cands):
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    return None


def _font_path(word: str, language: str, bold: bool) -> Optional[str]:
    """Pick the font file for a word: script font for Indic, Latin font otherwise.

    A non-Indic word (URL, kept English, digits, symbols) must NEVER fall back to
    the Devanagari font — that font has no Latin glyphs, so it renders tofu boxes
    (this is exactly what made "www.hdfclife.com" show as boxes). So for non-Indic
    we exhaust the Latin Noto + system Latin fonts before anything else.
    """
    cfg = LANGUAGE_CONFIG.get(language, {})
    fdir = Path(FONTS_DIR)
    script_reg = str(fdir / cfg["regular"]) if cfg.get("regular") and (fdir / cfg["regular"]).exists() else None
    script_bold = str(fdir / cfg["bold"]) if cfg.get("bold") and (fdir / cfg["bold"]).exists() else None
    latin_reg = _first_existing(str(fdir / _LATIN_REG), *_SYSTEM_LATIN)
    latin_bold = _first_existing(str(fdir / _LATIN_BOLD), latin_reg)

    if any(_DEVA_LO <= ord(c) <= _DEVA_HI for c in word):
        return (script_bold if bold else script_reg) or script_reg or latin_reg
    # Non-Indic → Latin only (never the Devanagari font).
    return (latin_bold if bold else latin_reg) or latin_reg or script_reg


def _word_width_pt(fp: str, text: str, size: float) -> float:
    import uharfbuzz as hb
    font, upem = _hb(fp)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    adv = sum(p.x_advance for p in buf.glyph_positions)
    return adv * size / upem


def _vmetrics_pt(fp: str, size: float):
    _, glyph_upem = _hb(fp)
    _, _, _, asc, desc = _tt(fp)
    sc = size / glyph_upem
    return asc * sc, (asc - desc) * sc  # (ascent, line_height base)


def _contours(fp: str, gid: int):
    key = (fp, gid)
    if key not in _contour_cache:
        _, gset, gorder, _, _ = _tt(fp)
        pen = _flatten_pen_cls()(gset)
        try:
            gset[gorder[gid]].draw(pen)
        except Exception:
            pass
        _contour_cache[key] = pen.contours
    return _contour_cache[key]


def _draw_word(shape, fp: str, text: str, size: float, ox: float, baseline: float, color01):
    """Draw a word's shaped glyph outlines (filled) starting at (ox, baseline).
    Returns the x-advance in points."""
    import uharfbuzz as hb
    font, upem = _hb(fp)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    sc = size / upem
    penx = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        gx, gy = penx + pos.x_offset, pos.y_offset
        drew = False
        for c in _contours(fp, info.codepoint):
            pts = [pymupdf.Point(ox + (gx + px) * sc, baseline - (gy + py) * sc) for px, py in c]
            if len(pts) >= 2:
                shape.draw_polyline(pts)
                drew = True
        if drew:  # even-odd per glyph → letter holes (counters) render correctly
            shape.finish(color=None, fill=color01, even_odd=True, closePath=True)
        penx += pos.x_advance
    return penx * sc


def _hex01(h: str):
    h = (h or "#000000").lstrip("#")
    try:
        return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)
    except Exception:
        return (0.0, 0.0, 0.0)


def _segment_runs(text: str, language: str, bold: bool):
    """Split a token into runs of consistent script, each paired with a font that
    actually has the glyphs.

    A token can mix scripts — digits/punctuation/Latin attached to Indic text, or
    an Indic span joined to a Latin one ("…ওয়েবসাইটwww.hdfclife.com"). Routing the
    WHOLE token to the script font boxes its ASCII chars, because Noto Bengali /
    Gujarati / Telugu / Malayalam carry NO Latin/digit/punctuation glyphs. Splitting
    at Indic↔non-Indic boundaries lets each run use its correct font. The split is
    safe for shaping: ASCII is never part of an Indic syllable cluster, and ZWJ/ZWNJ
    (the only joiners that matter) stay attached to the current run.
    """
    runs = []
    cur: list = []
    cur_indic = None
    for ch in text:
        cp = ord(ch)
        if cp in (0x200C, 0x200D) and cur:  # ZWJ/ZWNJ — keep with current run
            cur.append(ch)
            continue
        ind = _DEVA_LO <= cp <= _DEVA_HI
        if cur_indic is None or ind == cur_indic:
            cur.append(ch)
            cur_indic = ind
        else:
            runs.append("".join(cur))
            cur = [ch]
            cur_indic = ind
    if cur:
        runs.append("".join(cur))
    return [(r, _font_path(r, language, bold)) for r in runs if r]


def _word_width_runs(runs, size: float) -> float:
    return sum(_word_width_pt(fp, t, size) for (t, fp) in runs if fp)


def _draw_runs(shape, runs, size: float, ox: float, baseline: float, color01) -> float:
    """Draw a token's runs consecutively (each with its own font). Returns advance."""
    x = ox
    for (t, fp) in runs:
        if not fp:
            continue
        try:
            x += _draw_word(shape, fp, t, size, x, baseline, color01)
        except Exception:
            x += _word_width_pt(fp, t, size)
    return x - ox


def _spans_to_words(parsed_spans, language: str):
    """Flatten styled spans into per-word tokens (text, size, color01, bold, fontpath).
    Mirrors the raster path's tokenization so colours/sizes match per word."""
    import re
    from .utils import clean_symbol_text
    chars, styles = [], []
    for s in parsed_spans:
        text = clean_symbol_text((s.get("text", "") or "").replace("\n", " "))
        if not text:
            continue
        st = (float(s.get("size") or 10), s.get("color_hex") or "#000000", bool(s.get("bold")))
        chars.extend(text)
        styles.extend([st] * len(text))
        # NOTE: do NOT insert a space between spans. The spans are contiguous
        # slices of one translated string (real spaces already inside the slices),
        # so joining directly keeps words whole. Inserting a space could split a
        # Devanagari cluster across a style boundary (e.g. "विकल्प" → "विकल" + "्प"),
        # leaving a fragment that starts with a virama/matra → a stray detached mark.
    full = "".join(chars)
    words = []
    for m in re.finditer(r"\S+", full):
        size, color_hex, bold = Counter(styles[m.start():m.end()]).most_common(1)[0][0]
        wtext = m.group()
        # Per-run fonts: a mixed-script token (Indic + digits/punctuation/Latin) uses
        # the script font for its Indic runs and the Latin font for its ASCII runs,
        # so ASCII never boxes under a script font that lacks those glyphs.
        runs = _segment_runs(wtext, language, bold)
        if not runs:
            continue
        rep_fp = next((fp for _, fp in runs if fp), None)
        if not rep_fp:
            continue
        words.append({"text": wtext, "size": size, "color": _hex01(color_hex),
                      "bold": bold, "runs": runs, "fp": rep_fp})
    return words


def vector_insert_text(page, bbox, parsed_spans, language, align="left", clean_bg=None) -> bool:
    """Insert translated text as crisp filled VECTOR glyph outlines.

    Mirrors `_pil_insert_text`: per-word colours/sizes, single-line fit for short
    boxes else wrap, alignment, vertical centering, and the invisible selectable
    layer. Returns True on success, False so the caller can fall back to raster.
    """
    clip = bbox & page.rect
    if clip.is_empty:
        return False
    words = _spans_to_words(parsed_spans, language)
    if not words:
        return False

    W = clip.width
    H = clip.height
    pad = 1.5
    base_size = max(w["size"] for w in words)
    ref_fp = max(words, key=lambda w: w["size"])["fp"]

    def space_w(size):
        try:
            return _word_width_pt(ref_fp, " ", size) or size * 0.3
        except Exception:
            return size * 0.3

    def one_line_width(shrink):
        sp = space_w(base_size * shrink)
        tot = 0.0
        for i, w in enumerate(words):
            tot += _word_width_runs(w["runs"], w["size"] * shrink)
            if i:
                tot += sp
        return tot

    def line_height(shrink):
        _, lh = _vmetrics_pt(ref_fp, base_size * shrink)
        return lh

    # Single-line box (heading/label/cell) → keep one line, shrink to width.
    single_line = H <= 1.6 * line_height(1.0)
    final = None  # (lines, shrink) where lines = list of list of word dicts
    if single_line:
        shrink = 1.0
        while shrink >= 0.30:
            if one_line_width(shrink) <= W - 2 * pad:
                final = ([words], shrink)
                break
            shrink -= 0.05
        if final is None:
            final = ([words], 0.30)
    else:
        shrink = 1.0
        while shrink >= 0.34 and final is None:
            sp = space_w(base_size * shrink)
            lines, cur, cur_w = [], [], 0.0
            for w in words:
                ww = _word_width_runs(w["runs"], w["size"] * shrink)
                need = (cur_w + sp + ww) if cur else ww
                if need <= W - 2 * pad or not cur:
                    cur.append(w); cur_w = need
                else:
                    lines.append(cur); cur = [w]; cur_w = ww
            if cur:
                lines.append(cur)
            if line_height(shrink) * len(lines) <= H:
                final = (lines, shrink)
            else:
                shrink -= 0.07
        if final is None:
            # smallest: wrap at 0.34 and accept overflow (clipped by page only)
            sp = space_w(base_size * 0.34)
            lines, cur, cur_w = [], [], 0.0
            for w in words:
                ww = _word_width_runs(w["runs"], w["size"] * 0.34)
                need = (cur_w + sp + ww) if cur else ww
                if need <= W - 2 * pad or not cur:
                    cur.append(w); cur_w = need
                else:
                    lines.append(cur); cur = [w]; cur_w = ww
            if cur:
                lines.append(cur)
            final = (lines, 0.34)

    lines, shrink = final
    lh = line_height(shrink)
    asc, _ = _vmetrics_pt(ref_fp, base_size * shrink)
    sp = space_w(base_size * shrink)
    total_h = lh * len(lines)
    y_start = clip.y0 + max(0.0, (H - total_h) / 2.0)

    # No background patch: the erase removed the source text with fill=False, so the
    # REAL page background (gradient/photo/cell) is intact underneath — we draw the
    # glyph outlines straight onto it. (`clean_bg` is unused, kept for signature
    # compatibility.)
    shape = page.new_shape()
    drew_any = False
    for li, line in enumerate(lines):
        # line width for alignment
        widths = [_word_width_runs(w["runs"], w["size"] * shrink) for w in line]
        line_w = sum(widths) + sp * (len(line) - 1)
        if align == "center":
            x = clip.x0 + max(pad, (W - line_w) / 2.0)
        elif align == "right":
            x = clip.x1 - line_w - pad
        else:
            x = clip.x0 + pad
        baseline = y_start + li * lh + asc
        for wi, w in enumerate(line):
            try:
                _draw_runs(shape, w["runs"], w["size"] * shrink, x, baseline, w["color"])
                drew_any = True
            except Exception:
                pass
            x += widths[wi] + (sp if wi < len(line) - 1 else 0)
    try:
        shape.commit()
    except Exception as e:
        logger.warning(f"  [Vector] shape.commit failed at {clip}: {e}")
        return False
    if not drew_any:
        return False

    # Invisible selectable/searchable Unicode layer (shared with the raster path).
    try:
        from .text_mode import _insert_invisible_text_layer
        _insert_invisible_text_layer(page, clip, [[(w, None) for w in ln] for ln in lines], language)
    except Exception:
        pass
    return True
