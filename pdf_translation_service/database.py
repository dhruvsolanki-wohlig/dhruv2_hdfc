from datetime import datetime
from typing import Optional

import certifi
from motor.motor_asyncio import AsyncIOMotorClient

from .config import MONGO_URL, logger

db_client: Optional["Database"] = None


class Database:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None

    async def connect(self):
        self.client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
        db_name = "framework"
        if MONGO_URL and "/" in MONGO_URL:
            parts = MONGO_URL.rstrip("/").rsplit("/", 1)
            if len(parts) > 1:
                potential_db = parts[1].split("?")[0]
                if potential_db:
                    db_name = potential_db
        self.db = self.client[db_name]
        logger.info(f"Connected to MongoDB database: {db_name}")

    async def close(self):
        if self.client:
            self.client.close()

    @property
    def users(self):
        return self.db["users"]

    @property
    def documentconverters(self):
        return self.db["documentconverters"]

    async def create_job(self, job_data: dict) -> str:
        job_data["createdAt"] = datetime.utcnow()
        job_data["updatedAt"] = datetime.utcnow()
        job_data["__v"] = 1
        result = await self.documentconverters.insert_one(job_data)
        return str(result.inserted_id)

    async def update_translated_doc(self, job_id: str, language_result: dict):
        await self.documentconverters.update_one(
            {"jobId": job_id},
            {
                "$push": {"translatedDocs": language_result},
                "$set": {"updatedAt": datetime.utcnow()},
            },
        )

    async def update_job_page_count(self, job_id: str, page_count: int):
        await self.documentconverters.update_one(
            {"jobId": job_id},
            {"$set": {"pageCount": page_count, "updatedAt": datetime.utcnow()}},
        )

    async def finalize_job(self, job_id: str, status: str, error: dict = None):
        update: dict = {"status": status, "updatedAt": datetime.utcnow()}
        if error:
            update["error"] = error
        await self.documentconverters.update_one(
            {"jobId": job_id},
            {"$set": update, "$inc": {"__v": 1}},
        )

    async def update_debug_data(self, job_id: str, debug_pdf_url: str, extraction_metadata: list[dict]):
        await self.documentconverters.update_one(
            {"jobId": job_id},
            {
                "$set": {
                    "debugPdfUrl": debug_pdf_url,
                    "extractionMetadata": extraction_metadata,
                    "updatedAt": datetime.utcnow(),
                }
            },
        )

    async def get_job(self, job_id: str) -> Optional[dict]:
        return await self.documentconverters.find_one({"jobId": job_id})

    async def soft_delete_job(self, job_id: str):
        await self.documentconverters.update_one(
            {"jobId": job_id},
            {"$set": {"isDeleted": True, "deletedAt": datetime.utcnow(), "updatedAt": datetime.utcnow()}},
        )

    async def remove_translated_doc(self, job_id: str, language: str):
        await self.documentconverters.update_one(
            {"jobId": job_id, "translatedDocs.language": language},
            {
                "$set": {
                    "translatedDocs.$.isDeleted": True,
                    "translatedDocs.$.deletedAt": datetime.utcnow(),
                    "updatedAt": datetime.utcnow(),
                },
            },
        )

    async def get_user_jobs(
        self, user_id: str, page: int = 1, limit: int = 21, search: str = "", status: str = ""
    ) -> dict:
        query = {"userId": user_id, "isDeleted": False}
        if search:
            query["originalFileName"] = {"$regex": search, "$options": "i"}
        if status:
            query["status"] = status

        total = await self.documentconverters.count_documents(query)
        skip = (page - 1) * limit
        cursor = self.documentconverters.find(query).sort("createdAt", -1).skip(skip).limit(limit)
        jobs = await cursor.to_list(length=limit)

        return {
            "jobs": jobs,
            "pagination": {
                "total": total,
                "page": page,
                "limit": limit,
                "hasNext": skip + limit < total,
            },
        }