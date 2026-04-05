"""
Authentication routes with planted bugs.

BUGS PLANTED:
- BUG_AUTH_01 (hard): Auth tokens are not user-scoped — any valid token works for any user's resources
- BUG_AUTH_02 (medium): Login with empty password succeeds (missing validation)
"""

import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException
from typing import Optional

from ..database import Database
from ..models import LoginRequest, LoginResponse

router = APIRouter(prefix="/auth", tags=["auth"])

_db: Database | None = None


def set_db(db: Database):
    global _db
    _db = db


def get_db() -> Database:
    return _db


def get_current_user(authorization: Optional[str] = Header(None)) -> dict | None:
    """Extract user from auth token.

    BUG_AUTH_01: Returns the token's user but doesn't enforce ownership anywhere.
    The routes that use this don't check if the resource belongs to the user.
    """
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "")
    db = get_db()
    rows = db.execute(
        "SELECT u.id, u.username, u.role FROM auth_tokens t JOIN users u ON t.user_id = u.id WHERE t.token = ?",
        (token,),
    )
    if not rows:
        return None
    return rows[0]


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    db = get_db()

    # BUG_AUTH_02: Empty password check is missing — empty password matches hash
    # Should validate: if not req.password: raise HTTPException(400, ...)
    rows = db.execute(
        "SELECT id, username, role, password_hash FROM users WHERE username = ?",
        (req.username,),
    )
    if not rows:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = rows[0]
    # BUG_AUTH_02 continued: Only checks username, not password properly
    # In a real system we'd verify the password hash
    # Here we just check if password is non-empty... but we don't!
    # Any password (including empty string) works as long as username exists.

    token = str(uuid.uuid4())
    expires = datetime.utcnow() + timedelta(hours=24)
    db.execute_insert(
        "INSERT INTO auth_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user["id"], expires.isoformat()),
    )

    return LoginResponse(
        token=token,
        user_id=user["id"],
        username=user["username"],
        role=user["role"],
    )
