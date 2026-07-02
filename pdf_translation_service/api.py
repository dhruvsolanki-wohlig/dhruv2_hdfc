import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import database as _db
from .auth import authenticate
from .config import CDN_URL, LANGUAGE_CONFIG, PDF_SERVICE_PORT, logger
from .database import Database
from .fonts import ensure_fonts_available
from .gemini import init_gemini_client
from .orchestrator import (
    ConnectionManager,
    process_all_jobs,
    ws_manager,
)
from .utils import is_image_file, upload_to_gcp_with_retry


# ── Pydantic models ───────────────────────────────────────────────────────────

class TranslateJSONRequest(BaseModel):
    documentUrls: list[str]
    languages: list[str]


class JobResponse(BaseModel):
    jobId: str
    fileName: str
    status: str
    targetLanguages: list[str]
    websocketUrl: str


class TranslateResponse(BaseModel):
    success: bool
    jobs: list[JobResponse]


# ── Serialization helper ──────────────────────────────────────────────────────

def serialize_mongo_doc(doc) -> dict:
    if doc is None:
        return None
    result = {}
    for key, value in doc.items():
        if key == "_id":
            result[key] = str(value)
        elif isinstance(value, datetime):
            result[key] = value.isoformat() + "Z"
        elif isinstance(value, list):
            result[key] = [serialize_mongo_doc(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, dict):
            result[key] = serialize_mongo_doc(value)
        else:
            result[key] = value
    return result


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _db.db_client = Database()
    await _db.db_client.connect()
    init_gemini_client()
    await ensure_fonts_available()

    try:
        result = await _db.db_client.documentconverters.update_many(
            {"status": "processing"},
            {
                "$set": {
                    "status": "failed",
                    "error": {"title": "Server restarted — job was interrupted", "pageNumber": None},
                    "updatedAt": datetime.utcnow(),
                },
                "$inc": {"__v": 1},
            },
        )
        if result.modified_count > 0:
            logger.warning(f"Marked {result.modified_count} stale 'processing' job(s) as failed on startup")
    except Exception as e:
        logger.error(f"Failed to clean up stale jobs on startup: {e}")

    logger.info(f"PDF Translation Service v2 started on port {PDF_SERVICE_PORT}")
    yield
    await _db.db_client.close()
    logger.info("PDF Translation Service stopped")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="PDF Translation Service",
    description="Translate PDF documents to Indian languages — redact-and-insert text mode",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "pdf-translation", "version": "2.0.0"}


@app.post("/api/translate-pdf", status_code=202)
async def translate_pdf(request: Request):
    """Translate PDF documents to Indian languages."""
    user = await authenticate(request)
    user_id = str(user["_id"])
    org_id = str(user.get("defaultOrganizationId", ""))

    content_type = request.headers.get("content-type", "")
    target_languages = []
    documents = []

    if "multipart/form-data" in content_type:
        form = await request.form()
        languages_str = form.get("languages")
        if not languages_str:
            raise HTTPException(400, "languages field is required")
        try:
            target_languages = json.loads(languages_str)
        except json.JSONDecodeError:
            raise HTTPException(400, "languages must be a valid JSON array")

        for key in form:
            field = form[key]
            if hasattr(field, "read"):
                file_bytes = await field.read()
                filename = field.filename or "document.pdf"
                if is_image_file(filename):
                    raise HTTPException(400, f"Image inputs are not supported: {filename}. Upload a PDF.")
                documents.append({"filename": filename, "pdf_bytes": file_bytes})

        if not documents:
            raise HTTPException(400, "No files uploaded")

    elif "application/json" in content_type:
        body = await request.json()
        target_languages = body.get("languages", [])
        document_urls = body.get("documentUrls", [])

        if not document_urls:
            raise HTTPException(400, "documentUrls is required")
        if not target_languages:
            raise HTTPException(400, "languages is required")

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            for url in document_urls:
                filename = url.rsplit("/", 1)[-1] or "document.pdf"
                if is_image_file(filename):
                    raise HTTPException(400, f"Image inputs are not supported: {filename}. Provide a PDF URL.")
                resp = await client.get(url)
                if resp.status_code != 200:
                    raise HTTPException(400, f"Failed to download from {url}: HTTP {resp.status_code}")
                documents.append({"filename": filename, "pdf_bytes": resp.content})
    else:
        raise HTTPException(400, "Content-Type must be multipart/form-data or application/json")

    if not target_languages:
        raise HTTPException(400, "No target languages provided")
    for lang in target_languages:
        lang_name = lang["targetLanguage"] if isinstance(lang, dict) else lang
        if lang_name not in LANGUAGE_CONFIG:
            supported = ", ".join(sorted(LANGUAGE_CONFIG.keys()))
            raise HTTPException(400, f"Unsupported language: {lang_name}. Supported: {supported}")

    jobs_response = []
    jobs_to_process = []

    for doc_index, doc in enumerate(documents):
        job_id = f"doc-translate-{uuid.uuid4().hex[:8]}"

        try:
            original_url = await upload_to_gcp_with_retry(doc["pdf_bytes"], doc["filename"], mime_type="application/pdf")
        except Exception as e:
            raise HTTPException(500, f"Failed to upload file: {e}")

        job_data = {
            "userId": user_id,
            "organizationId": org_id,
            "jobId": job_id,
            "originalDocUrl": original_url,
            "originalFileName": doc["filename"],
            "targetLanguages": target_languages,
            "status": "processing",
            "error": {"title": None, "pageNumber": None},
            "pageCount": 0,
            "isDeleted": False,
            "deletedAt": None,
            "translatedDocs": [],
        }
        await _db.db_client.create_job(job_data)

        host = request.headers.get("host", f"localhost:{PDF_SERVICE_PORT}")
        scheme = "wss" if request.url.scheme == "https" else "ws"

        jobs_response.append({
            "jobId": job_id,
            "fileName": doc["filename"],
            "status": "processing",
            "targetLanguages": target_languages,
            "websocketUrl": f"{scheme}://{host}/ws/{job_id}",
        })

        jobs_to_process.append({
            "pdf_bytes": doc["pdf_bytes"],
            "original_filename": doc["filename"],
            "job_id": job_id,
            "languages": target_languages,
            "doc_index": doc_index,
        })

    logger.info(f"Accepted {len(documents)} doc(s) for translation. Launching background...")
    asyncio.create_task(process_all_jobs(jobs_to_process, ws_manager))

    return {"success": True, "jobs": jobs_response}


@app.get("/api/translate-pdf/{job_id}")
async def get_job_status(job_id: str, request: Request):
    user = await authenticate(request)
    job = await _db.db_client.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if str(job.get("userId")) != str(user["_id"]):
        raise HTTPException(403, "Access denied")
    if job.get("translatedDocs"):
        job["translatedDocs"] = [d for d in job["translatedDocs"] if not d.get("isDeleted")]
    return {"success": True, "job": serialize_mongo_doc(job)}


@app.get("/api/pdf-translation")
async def get_translation_history(
    request: Request, page: int = 1, limit: int = 21, search: str = "", status: str = ""
):
    """Get paginated translation history for the authenticated user."""
    user = await authenticate(request)
    user_id = str(user["_id"])

    if limit > 100:
        limit = 100
    if page < 1:
        page = 1

    result = await _db.db_client.get_user_jobs(user_id, page, limit, search, status)
    for job in result["jobs"]:
        if job.get("translatedDocs"):
            job["translatedDocs"] = [d for d in job["translatedDocs"] if not d.get("isDeleted")]
    serialized_jobs = [serialize_mongo_doc(job) for job in result["jobs"]]

    return {"success": True, "jobs": serialized_jobs, "pagination": result["pagination"]}


@app.get("/api/pdf-translation/download")
async def download_file(request: Request, url: str, filename: str = ""):
    """Proxy download to avoid CORS issues with CDN files."""
    await authenticate(request)
    if not (url.startswith(CDN_URL) or url.startswith("https://storage.googleapis.com/pocketstudio/")):
        raise HTTPException(400, "Invalid download URL")

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise HTTPException(502, f"Failed to fetch file: HTTP {resp.status_code}")

    content_type = resp.headers.get("content-type", "application/octet-stream")
    dl_name = filename or url.rsplit("/", 1)[-1]
    headers = {
        "Content-Disposition": f'attachment; filename="{dl_name}"',
        "Content-Type": content_type,
    }
    return StreamingResponse(iter([resp.content]), headers=headers, media_type=content_type)


@app.delete("/api/pdf-translation/{job_id}/language/{language}")
async def delete_translated_language(job_id: str, language: str, request: Request):
    """Remove a single translated language from a job."""
    user = await authenticate(request)
    job = await _db.db_client.get_job(job_id)
    if not job:
        from bson import ObjectId
        try:
            job = await _db.db_client.documentconverters.find_one({"_id": ObjectId(job_id)})
        except Exception:
            job = await _db.db_client.documentconverters.find_one({"_id": job_id})
    if not job:
        raise HTTPException(404, "Job not found")
    if str(job.get("userId")) != str(user["_id"]):
        raise HTTPException(403, "Access denied")
    actual_job_id = job.get("jobId", job_id)
    await _db.db_client.remove_translated_doc(actual_job_id, language)
    return {"success": True, "message": f"Language '{language}' removed"}


@app.delete("/api/pdf-translation/{job_id}")
async def delete_translation_job(job_id: str, request: Request):
    """Soft delete a translation job."""
    user = await authenticate(request)
    job = await _db.db_client.get_job(job_id)
    if not job:
        from bson import ObjectId
        try:
            job = await _db.db_client.documentconverters.find_one({"_id": ObjectId(job_id)})
        except Exception:
            job = await _db.db_client.documentconverters.find_one({"_id": job_id})
    if not job:
        raise HTTPException(404, "Job not found")
    if str(job.get("userId")) != str(user["_id"]):
        raise HTTPException(403, "Access denied")
    actual_job_id = job.get("jobId", job_id)
    await _db.db_client.soft_delete_job(actual_job_id)
    return {"success": True, "message": "Job deleted"}


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(job_id, websocket)


if __name__ == "__main__":
    uvicorn.run(
        "pdf_translation_service:app",
        host="0.0.0.0",
        port=PDF_SERVICE_PORT,
        reload=False,
        log_level="info",
    )
