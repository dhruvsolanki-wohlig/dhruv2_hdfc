import jwt as pyjwt
from fastapi import HTTPException, Request

from . import database as _db
from .config import JWT_SECRET_KEY


async def authenticate(request: Request) -> dict:
    """Validate JWT token and return the authenticated user document."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")

    token = auth_header[7:]
    try:
        decoded = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        user_data = decoded.get("data", {})
        user_id = user_data.get("_id") or user_data.get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        user = await _db.db_client.users.find_one({"_id": user_id})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        jwt_version = decoded.get("tokenVersion", 0)
        user_version = user.get("tokenVersion", 0)
        if jwt_version != user_version:
            raise HTTPException(status_code=401, detail="Token expired - version mismatch")

        return user

    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
