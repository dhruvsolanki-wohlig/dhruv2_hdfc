import asyncio
from pathlib import Path
from typing import Optional

import pymupdf

from . import database as _db
from .config import logger
from .extraction import detect_table_regions, extract_text_blocks_with_metadata
from .utils import rect_to_list, upload_to_gcp_with_retry


def generate_debug_pdf_and_metadata(pdf_bytes: bytes) -> tuple[bytes, list[dict]]:
    """Generate a debug PDF with colored bounding boxes and extraction metadata."""
    COLOR_BLOCK = (1.0, 0.0, 0.0)
    COLOR_LINE = (0.0, 0.0, 1.0)
    COLOR_SPAN = (0.0, 0.67, 0.0)
    COLOR_TABLE = (1.0, 0.55, 0.0)
    COLOR_CELL = (0.545, 0.0, 1.0)
    LABEL_FONT_SIZE = 6

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    all_pages_metadata = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_meta = {
            "pageNumber": page_num + 1,
            "width": round(page.rect.width, 2),
            "height": round(page.rect.height, 2),
            "blocks": [],
            "tables": [],
        }

        blocks = extract_text_blocks_with_metadata(page)
        for block_idx, block in enumerate(blocks):
            block_rect = block["bbox"]
            block_label = f"B{block_idx}"
            page.draw_rect(block_rect, color=COLOR_BLOCK, width=1.5)
            page.insert_text(
                (block_rect.x0 + 1, block_rect.y0 + LABEL_FONT_SIZE + 1),
                block_label, fontsize=LABEL_FONT_SIZE, color=COLOR_BLOCK,
            )

            block_text_parts = []
            block_meta = {
                "index": block_idx, "label": block_label,
                "bbox": rect_to_list(block_rect), "lines": [],
            }

            for line_idx, line in enumerate(block["lines"]):
                line_rect = line["bbox"]
                line_label = f"B{block_idx}L{line_idx}"
                page.draw_rect(line_rect, color=COLOR_LINE, width=0.8)
                line_meta = {"index": line_idx, "label": line_label, "bbox": rect_to_list(line_rect), "spans": []}

                for span_idx, span in enumerate(line["spans"]):
                    span_rect = span["bbox"]
                    span_label = f"B{block_idx}L{line_idx}S{span_idx}"
                    page.draw_rect(span_rect, color=COLOR_SPAN, width=0.5)
                    line_meta["spans"].append({
                        "index": span_idx, "label": span_label, "bbox": rect_to_list(span_rect),
                        "text": span["text"], "font": span["font"], "size": span["size"],
                        "color": span["color"], "color_hex": span["color_hex"],
                        "bold": span["bold"], "italic": span["italic"],
                    })
                    block_text_parts.append(span["text"])

                line_meta["spanCount"] = len(line_meta["spans"])
                block_meta["lines"].append(line_meta)

            block_meta["lineCount"] = len(block_meta["lines"])
            block_meta["spanCount"] = sum(len(l["spans"]) for l in block_meta["lines"])
            block_meta["fullText"] = "".join(block_text_parts)
            page_meta["blocks"].append(block_meta)

        table_regions = detect_table_regions(page)
        for table_idx, table in enumerate(table_regions):
            table_rect = table["bbox"]
            table_label = f"T{table_idx}"
            page.draw_rect(table_rect, color=COLOR_TABLE, width=2.0)
            page.insert_text(
                (table_rect.x0 + 1, table_rect.y0 + LABEL_FONT_SIZE + 1),
                table_label, fontsize=LABEL_FONT_SIZE, color=COLOR_TABLE,
            )
            table_meta = {
                "index": table_idx, "label": table_label,
                "bbox": rect_to_list(table_rect), "cells": [], "extractedData": None,
            }
            for cell_idx, cell_rect in enumerate(table["cells"]):
                cell_label = f"T{table_idx}C{cell_idx}"
                page.draw_rect(cell_rect, color=COLOR_CELL, width=0.7)
                table_meta["cells"].append({"index": cell_idx, "label": cell_label, "bbox": rect_to_list(cell_rect)})
            try:
                extracted = table["table_obj"].extract()
                if extracted:
                    table_meta["extractedData"] = [[cell or "" for cell in row] for row in extracted]
            except Exception:
                pass
            page_meta["tables"].append(table_meta)

        page_meta["blockCount"] = len(page_meta["blocks"])
        page_meta["tableCount"] = len(page_meta["tables"])
        all_pages_metadata.append(page_meta)

    debug_pdf_bytes = doc.tobytes(deflate=True, garbage=3)
    doc.close()
    return debug_pdf_bytes, all_pages_metadata


async def generate_and_upload_debug_pdf(pdf_bytes: bytes, original_filename: str, job_id: str):
    try:
        loop = asyncio.get_event_loop()
        debug_pdf_bytes, extraction_metadata = await loop.run_in_executor(
            None, generate_debug_pdf_and_metadata, pdf_bytes
        )
        debug_filename = f"{Path(original_filename).stem}_debug.pdf"
        debug_pdf_url = await upload_to_gcp_with_retry(debug_pdf_bytes, debug_filename)
        logger.info(f"Debug PDF uploaded for job {job_id}: {debug_pdf_url}")
        await _db.db_client.update_debug_data(job_id, debug_pdf_url, extraction_metadata)
    except Exception as e:
        logger.error(f"Failed to generate debug PDF for job {job_id}: {e}", exc_info=True)


def generate_comparison_pdf(
    original_bytes: bytes,
    translated_bytes: bytes,
    page_review_data: list[dict],
) -> bytes:
    """Side-by-side comparison PDF: original (left) vs translated (right).

    `page_review_data` is a list of per-page dicts:
        {"page_num": 0-indexed int,
         "needs_review": bool,
         "residual_bboxes": list[pymupdf.Rect]}

    Pages where needs_review=True get a red border; residual block bboxes are
    overlaid in red on the translated column so reviewers can pinpoint un-translated
    text without reading both versions word-by-word.
    """
    orig_doc = pymupdf.open(stream=original_bytes, filetype="pdf")
    trans_doc = pymupdf.open(stream=translated_bytes, filetype="pdf")
    review_doc = pymupdf.open()

    HEADER_H = 14  # pixels reserved for column labels above each page
    GAP = 8        # gap between the two columns

    for info in page_review_data:
        page_num = info["page_num"]
        needs_review = info.get("needs_review", False)
        residual_bboxes = info.get("residual_bboxes", [])

        if page_num >= len(orig_doc) or page_num >= len(trans_doc):
            continue

        orig_page = orig_doc[page_num]
        pw, ph = orig_page.rect.width, orig_page.rect.height

        review_page = review_doc.new_page(width=pw * 2 + GAP, height=ph + HEADER_H)

        # ── Left column: original ──────────────────────────────────────────
        orig_rect = pymupdf.Rect(0, HEADER_H, pw, ph + HEADER_H)
        review_page.show_pdf_page(orig_rect, orig_doc, page_num)
        review_page.insert_text((2, HEADER_H - 2), f"P{page_num + 1}  ORIGINAL", fontsize=7, color=(0.2, 0.2, 0.7))

        # ── Right column: translated ───────────────────────────────────────
        tx0 = pw + GAP
        trans_rect = pymupdf.Rect(tx0, HEADER_H, tx0 + pw, ph + HEADER_H)
        review_page.show_pdf_page(trans_rect, trans_doc, page_num)

        label_color = (0.8, 0.1, 0.1) if needs_review else (0.1, 0.55, 0.1)
        status_str = "REVIEW" if needs_review else "OK"
        review_page.insert_text(
            (tx0 + 2, HEADER_H - 2),
            f"P{page_num + 1}  TRANSLATED — {status_str}",
            fontsize=7, color=label_color,
        )

        if needs_review:
            review_page.draw_rect(
                pymupdf.Rect(tx0, HEADER_H, tx0 + pw, ph + HEADER_H),
                color=(1, 0, 0), width=0.5,
            )

        for bbox in residual_bboxes:
            shifted = pymupdf.Rect(
                bbox.x0 + tx0, bbox.y0 + HEADER_H,
                bbox.x1 + tx0, bbox.y1 + HEADER_H,
            )
            review_page.draw_rect(shifted, color=(1, 0, 0), width=1.5)

    result = review_doc.tobytes(deflate=True, garbage=3)
    orig_doc.close()
    trans_doc.close()
    review_doc.close()
    return result


async def generate_and_upload_review_pdf(
    original_bytes: bytes,
    translated_bytes: bytes,
    page_review_data: list[dict],
    original_filename: str,
    language: str,
    job_id: str,
) -> Optional[str]:
    """Generate and upload the per-language side-by-side review PDF.

    Returns the CDN URL or None if generation/upload fails (non-fatal).
    """
    try:
        loop = asyncio.get_event_loop()
        review_bytes = await loop.run_in_executor(
            None, generate_comparison_pdf, original_bytes, translated_bytes, page_review_data
        )
        lang_suffix = language.lower().replace(" ", "_")
        review_filename = f"{Path(original_filename).stem}_{lang_suffix}_review.pdf"
        url = await upload_to_gcp_with_retry(review_bytes, review_filename)
        logger.info(f"[{language}] Review PDF uploaded: {url}")
        return url
    except Exception as e:
        logger.error(f"[{language}] Failed to generate review PDF for job {job_id}: {e}")
        return None
