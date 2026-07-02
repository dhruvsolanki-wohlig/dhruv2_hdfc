import asyncio
import os
from pathlib import Path

import pymupdf
from fastapi import WebSocket, WebSocketDisconnect

from . import database as _db
from .config import DOC_SEMAPHORE, LANG_SEMAPHORE, LOCAL_OUTPUT_DIR, logger
from .debug import generate_and_upload_debug_pdf, generate_and_upload_review_pdf
from .extraction import extract_page_inventory
from .text_mode import translate_page_text_mode
from .utils import upload_to_gcp_with_retry


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        logger.info(f"WebSocket connected for job {job_id}")

    async def disconnect(self, job_id: str, websocket: WebSocket):
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast(self, job_id: str, message: dict):
        connections = self.active_connections.get(job_id, [])
        disconnected = []
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            if job_id in self.active_connections and ws in self.active_connections[job_id]:
                self.active_connections[job_id].remove(ws)


ws_manager = ConnectionManager()


async def process_document_language_text_mode(
    pdf_bytes, target_language, job_id, original_filename, doc_index, ws_manager,
    page_inventories=None,
) -> dict:
    """TEXT MODE: Process a document for a single language using redact-and-insert.

    `page_inventories` is an optional list of pre-extracted per-page snapshots
    (from extract_page_inventory).  When provided, each page skips re-extraction
    and reuses the shared cache — critical for multi-language jobs where M languages
    would otherwise each re-extract the same N pages.
    """
    from .config import LANGUAGE_CONFIG
    lang_config = LANGUAGE_CONFIG.get(target_language)
    if not lang_config:
        result = {"language": target_language, "url": None, "success": False,
                  "error": f"Unsupported language: {target_language}", "failedPages": []}
        await _db.db_client.update_translated_doc(job_id, result)
        return result

    target_script = lang_config["script"]

    try:
        async with LANG_SEMAPHORE:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            all_failed_pages = []

            logger.info(f"[{target_language}] TEXT MODE: '{original_filename}' ({total_pages} pages)")

            review_pages = []
            pct_values = []
            page_review_data = []  # accumulated for the review PDF

            for page_num in range(total_pages):
                page = doc[page_num]
                inventory = (
                    page_inventories[page_num]
                    if page_inventories and page_num < len(page_inventories)
                    else None
                )
                try:
                    page_res = await translate_page_text_mode(
                        page, page_num, target_language, target_script,
                        ws_manager, job_id, doc_index, total_pages,
                        inventory=inventory,
                    )
                    all_failed_pages.extend(page_res["failed_pages"])
                    pct_values.append(page_res["pct_translated"])
                    if page_res["needs_review"]:
                        review_pages.append(page_num + 1)
                    page_review_data.append({
                        "page_num": page_num,
                        "needs_review": page_res["needs_review"],
                        "residual_bboxes": page_res.get("residual_bboxes", []),
                    })
                except Exception as e:
                    logger.error(f"Error on page {page_num + 1} ({target_language}): {e}")
                    all_failed_pages.append(page_num + 1)
                    review_pages.append(page_num + 1)
                    page_review_data.append({
                        "page_num": page_num,
                        "needs_review": True,
                        "residual_bboxes": [],
                    })

            doc_pct = round((sum(pct_values) / len(pct_values) * 100), 1) if pct_values else 100.0
            needs_review = bool(review_pages)
            logger.info(
                f"[{target_language}] Completeness: {doc_pct}% translated, "
                f"{len(review_pages)} page(s) need review: {review_pages or '—'}"
            )

            doc.subset_fonts()
            output_bytes = doc.tobytes(deflate=True, garbage=3)
            doc.close()

            lang_suffix = target_language.lower().replace(" ", "_")
            output_filename = f"{Path(original_filename).stem}_{lang_suffix}.pdf"

            try:
                local_path = os.path.join(LOCAL_OUTPUT_DIR, output_filename)
                with open(local_path, "wb") as f:
                    f.write(output_bytes)
                logger.info(f"[{target_language}] Saved locally: {local_path}")
                print(f"\n{'='*60}\n📁 [{target_language}] LOCAL: {local_path}\n{'='*60}\n", flush=True)
            except Exception as _le:
                logger.warning(f"[{target_language}] Local save failed: {_le}")

            cdn_url = await upload_to_gcp_with_retry(output_bytes, output_filename)
            logger.info(f"[{target_language}] Upload: {cdn_url}")
            print(f"\n{'='*60}\n✅ [{target_language}] OUTPUT: {cdn_url}\n{'='*60}\n", flush=True)

            # Side-by-side review PDF (original | translated, residuals highlighted).
            review_pdf_url = await generate_and_upload_review_pdf(
                original_bytes=pdf_bytes,
                translated_bytes=output_bytes,
                page_review_data=page_review_data,
                original_filename=original_filename,
                language=target_language,
                job_id=job_id,
            )

            result = {"language": target_language, "url": cdn_url, "success": True,
                      "error": None, "failedPages": all_failed_pages,
                      "pctTranslated": doc_pct, "needsReview": needs_review,
                      "reviewPages": review_pages, "reviewPdfUrl": review_pdf_url}
            await _db.db_client.update_translated_doc(job_id, result)
            await ws_manager.broadcast(job_id, {
                "type": "doc:language:done", "docIndex": doc_index,
                "language": target_language, "url": cdn_url, "success": True,
                "pctTranslated": doc_pct, "needsReview": needs_review,
                "reviewPages": review_pages, "reviewPdfUrl": review_pdf_url,
            })
            return result

    except Exception as e:
        logger.error(f"Failed {target_language} for {original_filename}: {e}")
        result = {"language": target_language, "url": None, "success": False,
                  "error": str(e), "failedPages": []}
        try:
            await _db.db_client.update_translated_doc(job_id, result)
        except Exception:
            pass
        await ws_manager.broadcast(job_id, {
            "type": "doc:language:failed", "docIndex": doc_index,
            "language": target_language, "error": str(e),
        })
        return result


async def process_single_document(
    pdf_bytes, original_filename, job_id, languages, doc_index, ws_manager,
):
    """Process a single PDF document for all requested languages in parallel (text mode)."""
    async with DOC_SEMAPHORE:
        loop = asyncio.get_event_loop()

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)

        # Extract all pages once; each language processor reuses this inventory
        # instead of re-extracting (eliminates M× redundant I/O).
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
        doc.close()
        logger.info(
            f"Pre-extracted {len(page_inventories)} page(s) "
            f"(shared across {len(languages)} language(s))"
        )

        _lang_names = [l["targetLanguage"] if isinstance(l, dict) else l for l in languages]
        logger.info(f"=== [TEXT MODE] '{original_filename}' ({page_count} pages) -> {', '.join(_lang_names)} ===")

        await _db.db_client.update_job_page_count(job_id, page_count)
        await ws_manager.broadcast(job_id, {
            "type": "doc:started", "docIndex": doc_index,
            "fileName": original_filename, "pageCount": page_count,
        })

        await generate_and_upload_debug_pdf(pdf_bytes, original_filename, job_id)

        def _make_task(lang):
            lang_name = lang["targetLanguage"] if isinstance(lang, dict) else lang
            return process_document_language_text_mode(
                pdf_bytes, lang_name, job_id, original_filename, doc_index, ws_manager,
                page_inventories=page_inventories,
            )

        tasks = [_make_task(lang) for lang in languages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_success = True
        error_info = {"title": None, "pageNumber": None}
        for r in results:
            if isinstance(r, Exception):
                all_success = False
                error_info = {"title": str(r), "pageNumber": None}
            elif isinstance(r, dict) and not r.get("success"):
                all_success = False
                error_info = {"title": r.get("error"), "pageNumber": None}

        status = "success" if all_success else "failed"
        logger.info(f"Job {job_id} finalized: {status}")
        await _db.db_client.finalize_job(job_id, status, error_info if not all_success else None)
        await ws_manager.broadcast(job_id, {"type": "job:completed", "jobId": job_id, "status": status})


async def process_all_jobs(jobs: list[dict], ws_manager: "ConnectionManager"):
    try:
        logger.info(f"===== Starting PDF translation: {len(jobs)} document(s) =====")
        for j in jobs:
            lang_names = [l["targetLanguage"] if isinstance(l, dict) else l for l in j['languages']]
            logger.info(f"  - {j['original_filename']} -> {', '.join(lang_names)}")

        if jobs:
            first_job = jobs[0]
            await ws_manager.broadcast(first_job["job_id"], {
                "type": "job:started", "jobId": first_job["job_id"],
                "totalDocs": len(jobs), "totalLanguages": len(first_job["languages"]),
            })

        tasks = [
            process_single_document(
                pdf_bytes=job["pdf_bytes"],
                original_filename=job["original_filename"],
                job_id=job["job_id"],
                languages=job["languages"],
                doc_index=job["doc_index"],
                ws_manager=ws_manager,
            )
            for job in jobs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Document {i} raised: {r}", exc_info=r)

        logger.info(f"===== All {len(jobs)} document(s) complete =====")
    except Exception as e:
        logger.error(f"FATAL: process_all_jobs crashed: {e}", exc_info=True)
