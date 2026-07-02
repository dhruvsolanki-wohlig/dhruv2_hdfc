import re

import pymupdf

from .config import logger
from .utils import hex_from_int, is_bold, is_italic, should_translate


# ── Full text extraction with metadata ───────────────────────────────────────

def extract_text_blocks_with_metadata(page) -> list[dict]:
    """Extract text from page with full per-span metadata."""
    pymupdf.TOOLS.set_small_glyph_heights(True)
    data = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
    blocks = []

    for block in data["blocks"]:
        if block["type"] != 0:
            continue

        block_info = {"bbox": pymupdf.Rect(block["bbox"]), "lines": []}

        for line in block["lines"]:
            # Skip rotated / non-horizontal text (dir != ~(1,0)). These are
            # decorative badges, ribbons, and watermarks (e.g. a rotated red "NEW"
            # ribbon). Axis-aligned redaction + reinsertion can't reproduce a
            # rotated element without smearing it, so leave the original untouched
            # — same policy as logos and license marks.
            _dir = line.get("dir", (1.0, 0.0))
            if abs(_dir[1]) > 0.08 or _dir[0] < 0.92:
                continue

            line_info = {"bbox": pymupdf.Rect(line["bbox"]), "spans": []}

            prev_span_rect = None
            for span in line["spans"]:
                span_rect = pymupdf.Rect(span["bbox"])
                text = span["text"]

                if prev_span_rect and text:
                    gap = span_rect.x0 - prev_span_rect.x1
                    if gap > 2:
                        prev_text = line_info["spans"][-1]["text"] if line_info["spans"] else ""
                        if prev_text and not prev_text.endswith(" ") and not text.startswith(" "):
                            line_info["spans"][-1]["text"] += " "

                span_info = {
                    "text": text,
                    "bbox": span_rect,
                    "origin": span["origin"],
                    "font": span["font"],
                    "size": span["size"],
                    "color": span["color"],
                    "color_hex": hex_from_int(span["color"]),
                    "bold": is_bold(span["font"]),
                    "italic": is_italic(span["font"]),
                    "flags": span["flags"],
                }
                line_info["spans"].append(span_info)
                if text:
                    prev_span_rect = span_rect

            if line_info["spans"]:
                block_info["lines"].append(line_info)

        if block_info["lines"]:
            blocks.append(block_info)

    return blocks


def _split_line_by_span_gaps(
    block: dict, line: dict, gap_threshold: float = 20.0,
) -> list[dict]:
    """Split a single-line block into sub-blocks when spans have large horizontal gaps."""
    spans = [s for s in line["spans"] if s["text"].strip()]
    if len(spans) <= 1:
        return [block]

    sorted_spans = sorted(spans, key=lambda s: s["bbox"].x0)

    clusters: list[list[dict]] = [[sorted_spans[0]]]
    for span in sorted_spans[1:]:
        prev_span = clusters[-1][-1]
        gap = span["bbox"].x0 - prev_span["bbox"].x1
        if gap >= gap_threshold:
            clusters.append([span])
        else:
            clusters[-1].append(span)

    if len(clusters) <= 1:
        return [block]

    sub_blocks = []
    for cluster_spans in clusters:
        bbox = pymupdf.Rect(cluster_spans[0]["bbox"])
        for s in cluster_spans[1:]:
            bbox |= s["bbox"]
        sub_blocks.append({
            "bbox": bbox,
            "lines": [{"bbox": pymupdf.Rect(bbox), "spans": cluster_spans}],
        })
    return sub_blocks


def split_scattered_block(block: dict) -> list[dict]:
    """Split a block into sub-blocks if it spans multiple table cells."""
    lines = block["lines"]

    if len(lines) == 1:
        max_size = max(
            (s["size"] for s in lines[0]["spans"] if s["text"].strip()),
            default=0,
        )
        if max_size >= 9:
            sub_blocks = _split_line_by_span_gaps(block, lines[0])
            if len(sub_blocks) > 1:
                return sub_blocks
        return [block]

    if not lines:
        return [block]

    block_width = block["bbox"].width
    if block_width < 1:
        return [block]

    max_line_width = max(line["bbox"].width for line in lines)
    # Only split when lines are genuinely scattered: each individual line is much
    # narrower than the full block span (ratio ≥ 3.0). This catches the 3-column
    # feature-icon rows without over-splitting dense paragraphs.
    if max_line_width < 1 or block_width / max_line_width < 3.0:
        return [block]

    sorted_lines = sorted(lines, key=lambda l: l["bbox"].x0)
    clusters: list[list[dict]] = [[sorted_lines[0]]]
    for line in sorted_lines[1:]:
        cluster_x = clusters[-1][0]["bbox"].x0
        if abs(line["bbox"].x0 - cluster_x) < 30:
            clusters[-1].append(line)
        else:
            clusters.append([line])

    if len(clusters) <= 1:
        return [block]

    sub_blocks = []
    for cluster_lines in clusters:
        cluster_lines.sort(key=lambda l: l["bbox"].y0)
        bbox = pymupdf.Rect(cluster_lines[0]["bbox"])
        for line in cluster_lines[1:]:
            bbox |= line["bbox"]
        sub_blocks.append({"bbox": bbox, "lines": cluster_lines})
    return sub_blocks


def split_non_translatable_lines(block: dict) -> list[dict]:
    """Split a block if it mixes non-translatable lines (numbers) with translatable text."""
    lines = block["lines"]
    if len(lines) <= 1:
        return [block]

    line_translatable = []
    for line in lines:
        line_text = "".join(s["text"] for s in line["spans"]).strip()
        line_translatable.append(should_translate(line_text))

    if all(line_translatable) or not any(line_translatable):
        return [block]

    groups: list[list[dict]] = []
    current_type = line_translatable[0]
    current_group = [lines[0]]
    for i in range(1, len(lines)):
        if line_translatable[i] == current_type:
            current_group.append(lines[i])
        else:
            groups.append(current_group)
            current_type = line_translatable[i]
            current_group = [lines[i]]
    groups.append(current_group)

    if len(groups) <= 1:
        return [block]

    sub_blocks = []
    for group_lines in groups:
        bbox = pymupdf.Rect(group_lines[0]["bbox"])
        for line in group_lines[1:]:
            bbox |= line["bbox"]
        sub_blocks.append({"bbox": bbox, "lines": group_lines})
    return sub_blocks


def get_block_text(block: dict) -> str:
    """Get full block text with smart line joining."""
    block_lines = block["lines"]
    line_texts = ["".join(s["text"] for s in ln["spans"]) for ln in block_lines]
    if len(line_texts) <= 1:
        return line_texts[0] if line_texts else ""

    block_width = block["bbox"].width
    parts = [line_texts[0]]
    for li in range(1, len(line_texts)):
        prev_line_width = block_lines[li - 1]["bbox"].width
        if block_width > 0 and prev_line_width / block_width > 0.75:
            parts.append(" ")
        else:
            parts.append("\n")
        parts.append(line_texts[li])
    return "".join(parts)


def get_block_style_segments(block: dict) -> list[dict]:
    """Get style segments with character ranges for proportional style mapping."""
    segments = []
    char_pos = 0

    for line_idx, line in enumerate(block["lines"]):
        if line_idx > 0:
            char_pos += 1

        for span in line["spans"]:
            text = span["text"]
            if not text:
                continue
            style_key = (span["size"], span["color_hex"], span["bold"], span.get("italic", False))
            seg_end = char_pos + len(text)

            if segments:
                prev = segments[-1]
                prev_key = (prev["size"], prev["color_hex"], prev["bold"], prev["italic"])
                if prev_key == style_key:
                    prev["end"] = seg_end
                    char_pos = seg_end
                    continue

            segments.append({
                "start": char_pos,
                "end": seg_end,
                "size": span["size"],
                "color_hex": span["color_hex"],
                "bold": span["bold"],
                "italic": span.get("italic", False),
            })
            char_pos = seg_end

    return segments


def _find_word_boundary(text: str, pos: int, direction: int = 1, max_search: int = 20) -> int:
    """Find the nearest word boundary (space or newline) from pos."""
    best = pos
    for offset in range(1, max_search + 1):
        check = pos + (offset * direction)
        if check < 0 or check >= len(text):
            break
        if text[check] in (' ', '\n'):
            best = check + (1 if direction == 1 else 0)
            break
    return best


def apply_proportional_styles(
    translated_text: str,
    style_segments: list[dict],
    original_length: int,
) -> list[dict]:
    """Map original style segments proportionally onto translated text."""
    if not style_segments or not translated_text:
        return [{
            "text": translated_text or "",
            "size": 10, "color_hex": "#000000", "bold": False, "italic": False,
        }]

    if len(style_segments) == 1:
        seg = style_segments[0]
        return [{
            "text": translated_text,
            "size": seg["size"], "color_hex": seg["color_hex"],
            "bold": seg["bold"], "italic": seg["italic"],
        }]

    # If all segments share the same visual style, apply it to the whole block.
    # Proportional slicing on uniform-style blocks causes bold/color to bleed
    # into the wrong portion when the translated text is much longer.
    unique_styles = {
        (round(s["size"], 1), s["color_hex"], s["bold"], s["italic"])
        for s in style_segments
    }
    if len(unique_styles) == 1:
        seg = style_segments[0]
        return [{
            "text": translated_text,
            "size": seg["size"], "color_hex": seg["color_hex"],
            "bold": seg["bold"], "italic": seg["italic"],
        }]

    if original_length == 0:
        seg = style_segments[0]
        return [{
            "text": translated_text,
            "size": seg["size"], "color_hex": seg["color_hex"],
            "bold": seg["bold"], "italic": seg["italic"],
        }]

    trans_len = len(translated_text)
    result_spans = []
    current_pos = 0

    for i, seg in enumerate(style_segments):
        if i == len(style_segments) - 1:
            span_text = translated_text[current_pos:]
        else:
            proportion = seg["end"] / original_length
            raw_end = int(proportion * trans_len)

            seg_len = seg["end"] - seg.get("start", 0)
            if seg_len <= 3:
                best_pos = raw_end
            else:
                best_pos = _find_word_boundary(translated_text, raw_end, direction=1)
                if best_pos - raw_end > 20:
                    best_pos = _find_word_boundary(translated_text, raw_end, direction=-1)

            best_pos = max(current_pos, min(best_pos, trans_len))
            span_text = translated_text[current_pos:best_pos]
            current_pos = best_pos

        if span_text:
            result_spans.append({
                "text": span_text,
                "size": seg["size"], "color_hex": seg["color_hex"],
                "bold": seg["bold"], "italic": seg["italic"],
            })

    if not result_spans:
        seg = style_segments[0]
        result_spans.append({
            "text": translated_text,
            "size": seg["size"], "color_hex": seg["color_hex"],
            "bold": seg["bold"], "italic": seg["italic"],
        })

    return result_spans


def detect_block_alignment(block: dict) -> str:
    """Infer text alignment from line positions relative to block bbox."""
    bbox = block["bbox"]
    block_width = bbox.width
    if block_width < 1:
        return "left"
    lines = block.get("lines", [])
    if not lines:
        return "left"

    left_margins = []
    right_margins = []
    for line in lines:
        lr = line["bbox"]
        left_margins.append(lr.x0 - bbox.x0)
        right_margins.append(bbox.x1 - lr.x1)

    avg_left = sum(left_margins) / len(left_margins)
    avg_right = sum(right_margins) / len(right_margins)

    threshold = block_width * 0.10
    if avg_left > threshold and avg_right > threshold:
        if abs(avg_left - avg_right) < block_width * 0.15:
            return "center"
    if avg_left > avg_right + threshold:
        return "right"
    return "left"


# ── Table detection ───────────────────────────────────────────────────────────

def detect_table_regions(page) -> list[dict]:
    """Detect tables on the page with quality filters."""
    tables = []
    try:
        found_tables = page.find_tables()
        for table in found_tables:
            rows = table.rows if hasattr(table, "rows") else []
            if len(rows) < 3:
                continue
            try:
                extracted = table.extract()
                total_cells = sum(len(row) for row in extracted)
                empty_cells = sum(
                    1 for row in extracted for cell in row
                    if cell is None or not str(cell).strip()
                )
                if total_cells > 0 and empty_cells / total_cells > 0.5:
                    continue
            except Exception:
                pass

            table_info = {
                "bbox": pymupdf.Rect(table.bbox),
                "cells": [],
                "table_obj": table,
            }
            if hasattr(table, "cells") and table.cells:
                for cell in table.cells:
                    if cell:
                        table_info["cells"].append(pymupdf.Rect(cell))
            tables.append(table_info)
    except Exception as e:
        logger.debug(f"Table detection: {e}")
    return tables


def split_block_into_columns(block: dict, gap: float = 38.0) -> list[dict]:
    """Split a block into separate column sub-blocks when its spans cluster into
    groups with a large horizontal gap between them.

    Table rows are frequently grouped by PyMuPDF into one block that spans a label
    cell AND its value cell(s) — e.g. "Min. Age at Maturity" (left, on a dark cell)
    + "18 years" / "23 years" (right, on light cells). Placing one translated run
    across that whole row puts the label text in the wrong cell (and the wrong
    background colour). Clustering spans by x-gap restores one placement unit per
    cell, so each lands at its own coordinates over its own background.

    Conservative: only splits on gaps ≥ `gap` pts (~0.5"), so normal word spacing
    and justified paragraphs are never split. Spans within a cluster are regrouped
    into lines by y so multi-line cells stay intact.
    """
    spans = [s for ln in block["lines"] for s in ln["spans"] if s["text"].strip()]
    if len(spans) < 2:
        return [block]

    spans_sorted = sorted(spans, key=lambda s: s["bbox"].x0)
    clusters: list[list[dict]] = [[spans_sorted[0]]]
    cluster_max_x1 = spans_sorted[0]["bbox"].x1
    for s in spans_sorted[1:]:
        if s["bbox"].x0 - cluster_max_x1 > gap:
            clusters.append([s])
            cluster_max_x1 = s["bbox"].x1
        else:
            clusters[-1].append(s)
            cluster_max_x1 = max(cluster_max_x1, s["bbox"].x1)

    if len(clusters) <= 1:
        return [block]

    out = []
    for cl in clusters:
        # Regroup this cluster's spans into lines by y (spans within ~4pt share a line).
        line_groups: list[dict] = []
        for s in sorted(cl, key=lambda s: (round(s["bbox"].y0, 1), s["bbox"].x0)):
            placed = False
            for lg in line_groups:
                if abs(lg["y"] - s["bbox"].y0) < 4:
                    lg["spans"].append(s)
                    placed = True
                    break
            if not placed:
                line_groups.append({"y": s["bbox"].y0, "spans": [s]})

        line_objs = []
        for lg in sorted(line_groups, key=lambda l: l["y"]):
            sp = sorted(lg["spans"], key=lambda s: s["bbox"].x0)
            bb = pymupdf.Rect(sp[0]["bbox"])
            for s in sp[1:]:
                bb |= s["bbox"]
            line_objs.append({"bbox": bb, "spans": sp})

        bb = pymupdf.Rect(line_objs[0]["bbox"])
        for lo in line_objs[1:]:
            bb |= lo["bbox"]
        out.append({"bbox": bb, "lines": line_objs})
    return out


def split_logo_indented_lines(block: dict, page) -> list[dict]:
    """Split a block whose FIRST line(s) are pushed right by a logo/image while
    later lines start at the true left margin.

    PyMuPDF groups a heading sitting to the RIGHT of a logo together with the
    full-width subtitle BELOW the logo into one block — e.g.
        line0: "Sampoorna Jeevan"  (x0=100, indented past the HDFC Life logo)
        line1: "An Individual Non-Linked … Plan"  (x0=37, full left, below the logo)
    Because the logo image fills the gap left of the first line, the per-block
    insert places the WHOLE block at the first line's indented x — so the subtitle
    renders to the RIGHT of the heading instead of dropping to the left margin
    below it. Splitting the indented leading line(s) from the full-width remainder
    lets each land at its own left edge.

    Fires ONLY when a later line starts ≥25pt to the LEFT of the first line AND an
    image/large drawing occupies the first line's indent gap — so ordinary
    paragraphs and first-line indents are never split.
    """
    lines = block.get("lines", [])
    if len(lines) < 2:
        return [block]
    first_x0 = lines[0]["bbox"].x0
    rest_min = min(ln["bbox"].x0 for ln in lines[1:])
    if first_x0 - rest_min < 25:
        return [block]

    fl = lines[0]["bbox"]
    gap = pymupdf.Rect(rest_min, fl.y0, first_x0, fl.y1)
    has_logo = False
    try:
        page_area = page.rect.get_area()
        for info in page.get_image_info():
            if pymupdf.Rect(info["bbox"]).intersects(gap):
                has_logo = True
                break
        if not has_logo:
            for d in page.get_drawings():
                r = d.get("rect")
                if r and pymupdf.Rect(r).intersects(gap) and pymupdf.Rect(r).get_area() < 0.5 * page_area:
                    has_logo = True
                    break
    except Exception:
        return [block]
    if not has_logo:
        return [block]

    # Leading lines at the indented level → head; the first line that drops to the
    # left margin (and everything after) → the full-width remainder.
    head, rest = [], []
    for ln in lines:
        if not rest and ln["bbox"].x0 >= first_x0 - 10:
            head.append(ln)
        else:
            rest.append(ln)
    if not head or not rest:
        return [block]

    def _mk(lns):
        bb = pymupdf.Rect(lns[0]["bbox"])
        for l in lns[1:]:
            bb |= l["bbox"]
        return {"bbox": bb, "lines": lns}

    return [_mk(head), _mk(rest)]


def process_page_extraction(page) -> tuple[list[dict], list[dict]]:
    """Extract text blocks and table regions. Splits scattered and mixed blocks."""
    raw_blocks = extract_text_blocks_with_metadata(page)
    blocks = []
    for block in raw_blocks:
        for sub in split_scattered_block(block):
            for sub_l in split_logo_indented_lines(sub, page):
                for sub2 in split_non_translatable_lines(sub_l):
                    blocks.extend(split_block_into_columns(sub2))
    table_regions = detect_table_regions(page)
    return blocks, table_regions


# ── Completeness verification ─────────────────────────────────────────────────

# Acronyms / codes / brand marks that are legitimately kept in Latin script and
# must NOT be counted as untranslated residual English.
_KEEP_LATIN_TOKENS = {
    "hdfc", "sbi", "lic", "irdai", "irda", "uin", "arn", "cin", "plc", "mh",
    "mc", "nav", "ulip", "tpa", "emi", "pan", "kyc", "nri", "gst", "cagr",
    "nps", "sip", "ppf", "fd", "rd", "tds", "pin", "ifsc", "upi", "otp",
    "rs", "inr", "usd",
}
_ROMAN_NUMERALS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii",
}
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+")
_DOMAIN_RE = re.compile(r"\b\S+\.(?:com|in|org|net|co|gov|edu)\b", re.IGNORECASE)
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
# Brand anchors: a short Latin run anchored by one of these is a kept brand
# island ("HDFC Life"), not a translation failure.
_BRAND_ANCHORS = ("hdfc", "sbi", "lic", "irdai", "irda", "click 2 protect")


def is_residual_english(text: str) -> bool:
    """True if `text` is a real, translatable English phrase the translator left
    untranslated — as opposed to text that legitimately stays in Latin script.

    Legitimately-Latin (returns False): brand names ("HDFC Life"), all-caps codes
    (CIN/UIN/IRDAI), roman numerals, URLs, emails, domains, registration numbers,
    and any block already predominantly in the target (non-Latin) script — where
    leftover Latin is just a brand island. Errs toward NOT flagging codes/brands
    so the repair pass never tries to "translate" e.g. IRDAI.
    """
    if not text or not text.strip():
        return False
    masked = _URL_RE.sub(" ", text)
    masked = _EMAIL_RE.sub(" ", masked)
    masked = _DOMAIN_RE.sub(" ", masked)

    real_words = []
    for w in _LATIN_WORD_RE.findall(masked):
        wl = w.lower()
        if len(w) < 3:
            continue  # st, ii, Rs, single letters
        if wl in _ROMAN_NUMERALS or wl in _KEEP_LATIN_TOKENS:
            continue
        if w.isupper() and len(w) <= 5:
            continue  # short all-caps acronym/code (CIN, IRDAI, ARN, PLC)
        real_words.append(w)
    if not real_words:
        return False

    # Any real English word that survived is a residual — EXCEPT a SHORT
    # brand/entity name (≤3 real words anchored by a brand, e.g. "HDFC Life" as a
    # kept island inside an otherwise-translated line).
    #
    # Notes on rejected heuristics:
    #  - Pure registration codes (CIN L65110…, UIN 101N158V06) need no special
    #    case: their tokens are all-caps/short or alphanumeric and never count as
    #    "real words". A block-level code exemption was tried and removed — it
    #    wrongly cleared long English legal/address disclaimers that merely
    #    contain a code (e.g. the footer entity name + registered address).
    #  - A script-ratio (Latin vs target) test was also removed: it exempted
    #    lines like "…75 வேரியண்ட்: 28 years" where one common word is a residual.
    low = masked.lower()
    if len(real_words) <= 3 and any(anchor in low for anchor in _BRAND_ANCHORS):
        return False
    return True


def extract_page_inventory(page) -> dict:
    """Extract and serialize all page data for reuse across language processors.

    Calling process_page_extraction (text extraction + table detection) once and
    caching the result eliminates M× redundant I/O when translating to M languages.
    The table_obj references are serialized to plain geometry data (rows_data / grid)
    so the inventory is safe to hand off to processors that open independent
    pymupdf.Document copies.
    """
    blocks, table_regions = process_page_extraction(page)

    serialized_tables = []
    for tr in table_regions:
        try:
            grid = tr["table_obj"].extract()
        except Exception:
            grid = None

        rows_data: list[list] = []
        try:
            rows = tr["table_obj"].rows if hasattr(tr["table_obj"], "rows") else []
            for row in rows:
                rows_data.append(list(row.cells) if hasattr(row, "cells") else [])
        except Exception:
            pass

        serialized_tables.append({
            "bbox": tr["bbox"],
            "cells": tr["cells"],
            "grid": grid,
            "rows_data": rows_data,
        })

    return {"blocks": blocks, "tables": serialized_tables}


def residual_translation_report(pairs: list[tuple]) -> dict:
    """Measure translation completeness from (source, translated) STRING pairs.

    Verifying the translator's output strings — not a re-extraction of the
    rendered PDF — is deliberate: PyMuPDF cannot reliably round-trip complex
    Indic conjuncts written via insert_htmlbox (e.g. वर्ष re-extracts as 'व�ष'),
    so re-reading the output PDF produces false residual/tofu signals. The
    translation strings are clean Unicode and are the correct source of truth.

    Returns the indices of source blocks whose translation is still English
    (so the caller can repair them), plus completeness counts.
    """
    translatable = 0
    residual_idx = []
    tofu = 0
    for i, (src, tgt) in enumerate(pairs):
        if not should_translate(src or ""):
            continue
        translatable += 1
        tgt = tgt or ""
        if "�" in tgt:
            tofu += 1
        if is_residual_english(tgt):
            residual_idx.append(i)
    pct = 1.0 if translatable == 0 else 1.0 - len(residual_idx) / translatable
    return {
        "translatable": translatable,
        "residual_idx": residual_idx,
        "residual_count": len(residual_idx),
        "tofu_count": tofu,
        "pct_translated": pct,
    }
