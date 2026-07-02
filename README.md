# HDFC PDF Translation Service

## Problem Statement

HDFC Life produces insurance brochures (PDF) in English. These need to be translated into 11 Indian languages while **preserving the exact visual design** — colored gradient backgrounds, tables with merged cells, logos, brand colors, circular badges, and decorative elements.

### Why Traditional Approaches Fail

| Approach | Problem |
|---|---|
| **LaTeX** | LaTeX is a document *authoring* system. It rebuilds pages from semantic markup (`\section{}`, `\begin{table}`). It cannot preserve the original visual layout — gradients, photos, logos, pixel-precise positioning. Translating a designed brochure via LaTeX means *redesigning the entire brochure from scratch*. |
| **Manual translation** | 11 languages × multiple brochures = hundreds of hours. Not scalable. |
| **Google Translate API** | Translates text but doesn't place it back into the PDF layout. |
| **OCR + reflow** | Destroys the original design. Text reflows differently in Indic scripts (wider glyphs, different line breaks). |

### The Core Challenge

The PDF has text layered **on top of** backgrounds (blue headers, gradient cells, photos). To translate:
1. You must **remove** the original English text without damaging the background underneath
2. **Insert** the translated Indic text at the exact same coordinates
3. The Indic text must be **properly shaped** (Devanagari conjuncts like वर्ष, not broken fragments)
4. The result must be **selectable/copyable** (not just a flat image)

---

## Early Solution (Before Changes)

### Architecture

The pipeline was already built with these components:

```
PDF → PyMuPDF extraction → Gemini translation → Redact English → Insert Hindi → Output PDF
```

### How It Worked

1. **Extraction** (`extraction.py`): PyMuPDF extracts text blocks with bbox coordinates, font sizes, colors, and style metadata. Blocks are split/merged via `process_page_extraction()` which handles scattered blocks, column splits, and logo-indented lines.

2. **Translation** (`gemini.py`): Text blocks are batched (20 per chunk) and sent to Google Gemini 3.5 Flash with a structured prompt containing:
   - Transliteration rules (HDFC → एचडीएफसी)
   - Fit constraints (translation length ≈ source length × 1.1)
   - Legal/disclaimer rules (keep company names, CIN, UIN, addresses in English)
   - Terminology glossary (locked insurance terms)
   - Response schema (JSON with `translations` array)

3. **Redaction** (`text_mode.py`): `page.add_redact_annot()` with `fill=False` removes only the text glyph operators from the PDF content stream, revealing the real background (vector fills, gradients, photos) underneath — like Word's "reveal formatting."

4. **Insertion**: Translated text is inserted at the original block's bbox coordinates using one of two renderers:
   - **Vector renderer** (`vector_text.py`): Uses `uharfbuzz` for OpenType shaping + `fontTools` for glyph outlines. Draws filled vector paths directly on the page. Crisp at any zoom.
   - **PIL raster renderer** (`text_mode.py`): Falls back to PIL/FreeType → transparent PNG → `page.insert_image()`. Lower quality, not selectable.
   - **htmlbox fallback**: PyMuPDF's `insert_htmlbox()` for Latin scripts. Can't shape Indic conjuncts.

5. **Invisible text layer**: An invisible (render_mode=3) Unicode text layer is written via `TextWriter` on top of the visible glyphs, making the Hindi text selectable/searchable/copyable.

### Known Issues Before Changes

| Issue | Impact |
|---|---|
| **`uharfbuzz` not installed** | Vector renderer never ran. Everything fell back to PIL raster (transparent PNG images). Hindi text was visible but not selectable. |
| **Output filename hardcoded** | `run_translation.py` always saved as `Cutouts_Page_1_hindi.pdf` regardless of input filename. Every run overwrote the previous output. |
| **No rate limiting** | Gemini free tier (20 req/day) exhausted quickly. No delay between pages or chunks. |
| **Post-insert false alarm** | Verification checked `get_text()` chars and `get_image_info()` only. When vector renderer drew 1600+ glyph outlines (drawings, not text/images), the check reported "POST-INSERT CHECK FAILED — 0 chars and 0 images" even though the page rendered correctly. |
| **Gemini free tier quota** | 429 RESOURCE_EXHAUSTED on every call after 2-3 chunks. Daily limit (20 requests) too low for a 59-block page. |
| **Numbers overlapping with translated text** | Table header numbers (6, 8, 10, 12, 15) were not translated (pure digits) and not redacted. The Hindi header text "प्रीमियम भुगतान अवधि (वर्ष)" was inserted on top of the numbers, making them unreadable. |
| **Legal text false-flagged as residual** | Rule 7 in the Gemini prompt says "keep company names, CIN, addresses in English" for legal/disclaimer text. But `is_residual_english()` saw the many English words (Insurance, Company, Limited, Registered, Office, etc.) and flagged the block as "untranslated." The repair pass sent it back to Gemini, got the same result, and gave up → reported 62.5% completeness. |
| **Invisible text layer broken on some pages** | `render_mode=3` + CID fonts (Noto Sans Devanagari) on short pages → `get_text()` returns empty. Hindi is visually rendered (vector glyphs) but not selectable. |
| **TJ operator not removed by redaction** | `apply_redactions(fill=False)` doesn't remove text in TJ array operators. Original numbers stayed in the content stream even after redaction. |
| **API key env var mismatch** | Code reads `GOOGLE_API_KEY` but `.env` had `GEMINI_API_KEY`. No fallback. |

---

## Changes Made

### Fix 1: Output Filename — Use Actual Input Filename

**File:** `run_translation.py`

**Before:**
```python
output_path = os.path.join(LOCAL_OUTPUT_DIR, f"Cutouts_Page_1_{lang_suffix}.pdf")
```

**After:**
```python
stem = Path(pdf_path).stem
output_path = os.path.join(LOCAL_OUTPUT_DIR, f"{stem}_{lang_suffix}.pdf")
```

**Why:** Every run was saving as `Cutouts_Page_1_hindi.pdf` regardless of input. Running `Cutouts_Page_2.pdf` would overwrite the Page 1 output.

---

### Fix 2: Rate Limiting Between Pages

**File:** `run_translation.py`

**Added:**
```python
if page_num > 0:
    logger.info(f"  Waiting 15s before page {page_num + 1} (rate limit)...")
    await asyncio.sleep(15)
```

**Why:** Gemini free tier has a daily request limit. Without delay between pages, all chunks fire rapidly and exhaust the quota before the document finishes.

---

### Fix 3: Post-Insert Verification — Count Vector Drawings

**File:** `pdf_translation_service/text_mode.py`

**Before:**
```python
if len(post_text) == 0 and len(post_images) == 0 and len(translated_block_indices) > 0:
    # FALSE ALARM: reports "FAILED" even when 1600 vector glyphs were drawn
    logger.error("POST-INSERT CHECK FAILED — 0 chars and 0 images...")
elif len(post_text) == 0 and len(translated_block_indices) > 0:
    logger.info("PIL raster path — text is in image layer")
```

**After:**
```python
post_drawings = len(page.get_drawings())

if len(post_text) == 0 and len(post_images) == 0 and post_drawings < 100 and ...:
    # Only flag as failure if there are NO drawings either
    logger.error("POST-INSERT CHECK FAILED — 0 chars, 0 images, {post_drawings} drawings...")
elif len(post_text) == 0 and len(translated_block_indices) > 0:
    if post_drawings >= 100:
        # Vector renderer drew glyph outlines — text IS there
        logger.info(f"{post_drawings} vector glyph(s) on page (vector path)")
    else:
        logger.info(f"{len(post_images)} image(s) on page (PIL raster path)")
```

**Why:** When `uharfbuzz` is installed, the vector renderer draws Hindi text as filled glyph outlines (vector drawings). These show up as `page.get_drawings()` entries, not as `get_text()` chars or `get_image_info()` images. The old check only looked at text and images, so it falsely reported failure on correctly-rendered pages.

---

### Fix 4: Overlapping Non-Translated Blocks — Redact and Re-insert

**File:** `pdf_translation_service/text_mode.py`

**Problem:** Table header numbers (6, 8, 10, 12, 15) are pure digits → `should_translate()` returns `False` → not added to `translated_block_indices` → not redacted → original glyphs stay in the PDF content stream. When the Hindi header text is inserted at the same coordinates, it paints **on top of** the numbers, making them unreadable.

**Solution:** Three-part fix:

**Part A — Detect adjacent non-translated blocks:**
```python
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
        # Same-row adjacency: within 8px vertically, horizontal overlap
        # or number sits between two translated blocks in the same row
        v_gap = min(abs(b_bbox.y0 - t_bbox.y1), abs(t_bbox.y0 - b_bbox.y1))
        if v_gap <= 8:
            h_overlap = min(b_bbox.x1, t_bbox.x1) - max(b_bbox.x0, t_bbox.x0)
            if h_overlap > 0:
                overlap_block_indices.append(i)
                break
            # Check if number is between two translated cells
            if b_bbox.x0 >= t_bbox.x0 - 5:
                for t2_bbox in translated_bboxes:
                    if (t2_bbox is not t_bbox and
                        abs(t2_bbox.y0 - t_bbox.y0) < 10 and
                        t2_bbox.x1 >= b_bbox.x0):
                        overlap_block_indices.append(i)
                        break
                if i in overlap_block_indices:
                    break
```

**Part B — Redact both translated AND overlapping blocks:**
```python
for i, block in enumerate(blocks):
    if i in translated_block_indices:
        for line in block["lines"]:
            page.add_redact_annot(line["bbox"], fill=False)
    elif i in overlap_block_indices:
        for line in block["lines"]:
            page.add_redact_annot(line["bbox"], fill=False)
page.apply_redactions(...)
```

**Part C — Re-insert non-translated blocks at their original positions AFTER all translated blocks:**
```python
for i in overlap_block_indices:
    block = blocks[i]
    # Build parsed_spans from original block's spans (preserve sizes/colors)
    orig_spans = []
    for line in block["lines"]:
        for span in line["spans"]:
            srgb = span.get("color", 0)
            r = int((srgb >> 16) & 0xFF)
            g = int((srgb >> 8) & 0xFF)
            b = int(srgb & 0xFF)
            color_hex = f"#{r:02x}{g:02x}{b:02x}"
            flags = span.get("flags", 0)
            orig_spans.append({
                "text": span["text"],
                "size": span["size"],
                "color_hex": color_hex,
                "bold": bool(flags & 2**4),
                "italic": bool(flags & 2**1),
            })
    orig_bbox = _clamp_to_page(pymupdf.Rect(block["bbox"]))
    insert_translated_text(
        page, orig_bbox, orig_spans, font_css, archive,
        min_scale=0.55, align=detect_block_alignment(block),
        language=None,  # Latin — no Indic shaping needed
        clean_bg=clean_bg,
    )
```

**Why:** The numbers (6, 8, 10, 12, 15) are in the same visual row as the translated header "प्रीमियम भुगतान अवधि (वर्ष)". The translated Hindi text overflows its source bbox (Indic glyphs are wider) and covers the numbers. By redacting the numbers too and re-inserting them after all translated text is rendered, the numbers paint on top of any Hindi text that overflowed into their area.

---

### Fix 5: API Key Fallback — GEMINI_API_KEY

**File:** `pdf_translation_service/gemini.py`

**Before:**
```python
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
```

**After:**
```python
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
```

**Why:** The `.env` file had `GEMINI_API_KEY` but the code only read `GOOGLE_API_KEY`. When `GOOGLE_API_KEY` wasn't set, the client failed to initialize. The fallback lets either env var work.

---

### Fix 6: uharfbuzz Installation

**Command:**
```bash
pip3 install uharfbuzz
```

**Why:** The vector renderer (`vector_text.py`) requires `uharfbuzz` for OpenType shaping (GSUB/GPOS tables). Without it, `vector_insert_text()` raises `ModuleNotFoundError: No module named 'uharfbuzz'` and falls back to PIL raster (transparent PNG images — lower quality, not selectable, washes out in some PDF viewers). With uharfbuzz installed, Hindi text renders as crisp vector glyph outlines at any zoom level.

---

## How to Run

### Prerequisites

```bash
pip3 install uharfbuzz pymupdf Pillow httpx python-dotenv tenacity fonttools
```

### API Key Setup

Get a valid Gemini API key from https://aistudio.google.com/apikey and set it in `.env`:

```
GEMINI_API_KEY=AIza...your-key...
```

### Translate a PDF

```bash
cd ~/Documents/dhruv2_hdfc
python3 run_translation.py Cutouts_Page_2.pdf Hindi
```

### Verify Output

```bash
python3 -c "
import pymupdf
d = pymupdf.open('hindi/Cutouts_Page_2_hindi.pdf')
for i in range(len(d)):
    t = d[i].get_text()
    imgs = d[i].get_image_info()
    draws = len(d[i].get_drawings())
    print(f'Page {i+1}: {len(t)} chars, {len(imgs)} images, {draws} drawings')
    print(t[:300])
"
```

---

## Known Remaining Issues

| # | Issue | Impact | Status |
|---|---|---|---|
| 1 | **Legal text false-flagged as residual** — `is_residual_english()` flags correctly-translated legal/disclaimer blocks because they contain many kept English words (company name, CIN, address per Rule 7). Repair pass gets same result from Gemini → gives up → reports 62.5%. | Affects all languages. Completeness score understated. | **Not fixed** |
| 2 | **Invisible text layer broken on some pages** — `render_mode=3` + CID fonts → `get_text()` returns empty. Hindi is visually rendered (vector glyphs) but not selectable/copyable. | Affects all Indic languages on certain page layouts. | **Not fixed** |
| 3 | **TJ operator not removed by redaction** — `apply_redactions(fill=False)` doesn't remove TJ array text. Original numbers stay in content stream. | Numbers may appear twice (original + re-inserted). | **Not fixed** |
| 4 | **Fit budget too tight for Tamil/Malayalam** — Budget is `len(src) * 1.1`. Tamil and Malayalam produce longer translations → truncation or unreadable shrinking. | Affects Tamil, Malayalam specifically. | **Not fixed** |
| 5 | **No glossary for 9 Indic languages** — Only Hindi has locked terminology. Others translate freely → inconsistent across runs. | Affects all non-Hindi Indic languages. | **Not fixed** |
| 6 | **English not in LANGUAGE_CONFIG** — Passing "English" as target language will crash. | English translation not supported. | **Not fixed** |

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Input PDF   │────▶│  Extraction   │────▶│  Gemini     │────▶│  Redaction   │────▶│  Insertion   │
│  (English)   │     │  (PyMuPDF)    │     │  Translation│     │  (fill=False)│     │  (Vector/PIL)│
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘     └──────┬───────┘
                     │                    │                    │                      │
                     │  Blocks + bbox     │  Hindi text        │  Erase English        │  Draw Hindi
                     │  + styles +        │  per block         │  glyphs only          │  at same coords
                     │  table detection   │                    │  (keep background)    │  + invisible
                     │                    │                    │                      │  text layer
                     ▼                    ▼                    ▼                      ▼
              process_page_          translate_batch_     add_redact_annot()     vector_insert_text()
              extraction()           with_gemini()        apply_redactions()     _pil_insert_text()
                                                                                   insert_htmlbox()
```

### Key Files

| File | Purpose |
|---|---|
| `run_translation.py` | CLI entry point — loads PDF, runs translation, saves output |
| `config.py` | Language config, fonts, glossary, Gemini model, concurrency |
| `gemini.py` | Gemini API client, prompt builder, batch translation with retry + split |
| `extraction.py` | Block extraction, table detection, `should_translate()`, `is_residual_english()` |
| `text_mode.py` | Main pipeline: extract → translate → redact → insert → verify |
| `vector_text.py` | Vector renderer — HarfBuzz shaping + fontTools glyph outlines |
| `fonts.py` | Font loading, PIL/PyMuPDF font caches, Noto font auto-download |
| `utils.py` | Utilities — `should_translate()`, GCP upload |
| `orchestrator.py` | Multi-language orchestration, WebSocket progress broadcasting |
| `api.py` | FastAPI server endpoints |
| `debug.py` | Review PDF generation (side-by-side original + translated with red borders) |

---

## Supported Languages

| Language | Script | Font | Complex Script |
|---|---|---|---|
| Hindi | Devanagari | Noto Sans Devanagari | ✅ |
| Marathi | Devanagari | Noto Sans Devanagari | ✅ |
| Gujarati | Gujarati | Noto Sans Gujarati | ✅ |
| Kannada | Kannada | Noto Sans Kannada | ✅ |
| Tamil | Tamil | Noto Sans Tamil | ✅ |
| Telugu | Telugu | Noto Sans Telugu | ✅ |
| Bengali | Bengali | Noto Sans Bengali | ✅ |
| Assamese | Bengali | Noto Sans Bengali | ✅ |
| Malayalam | Malayalam | Noto Sans Malayalam | ✅ |
| Punjabi | Gurmukhi | Noto Sans Gurmukhi | ✅ |
| Odia | Odia | Noto Sans Oriya | ✅ |
| English | Latin | Noto Sans | ❌ (not yet configured) |

---

## Translation Prompt Rules (gemini.py)

The Gemini prompt enforces these rules:

1. **Transliterate** brand names, product names, proper nouns phonetically (HDFC Life → एचडीएफसी लाइफ)
2. **Preserve numbers** (digits, %, ₹, dates) but translate words around them (28 years → 28 वर्ष)
3. **Skip** pure numbers/symbols — return unchanged
4. **Preserve line breaks** exactly
5. **Preserve spaces** between items
6. **Fit constraint** — keep translation within ~1.1× source length (Indic text is wider per glyph)
7. **Legal/disclaimer text** — fully translate the prose, but keep company names, CIN/UIN/IRDAI codes, URLs, emails, phone numbers, and addresses in English