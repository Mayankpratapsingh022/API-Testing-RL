"""
User management routes with planted bugs.

BUGS PLANTED:
- BUG_USER_01 (medium): POST /users doesn't validate email format
- BUG_USER_02 (medium): GET /users exposes password hashes in response
"""

from fastapi import APIRouter, HTTPException

from ..database import Database
from ..models import UserCreate

router = APIRouter(prefix="/users", tags=["users"])

_db: Database | None = None


def set_db(db: Database):
    global _db
    _db = db


def get_db() -> Database:
    return _db


@router.get("")
def list_users():
    db = get_db()
    rows = db.execute("SELECT id, username, email, role, created_at FROM users")
    return rows


@router.get("/{user_id}")
def get_user(user_id: int):
    db = get_db()
    rows = db.execute("SELECT id, username, email, role, created_at FROM users WHERE id = ?", (user_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="User not found")
    return rows[0]


@router.post("", status_code=201)
def create_user(user: UserCreate):
    db = get_db()

    # BUG_USER_01: No email format validation — accepts "not-an-email" or empty string
    # Should validate email with regex or pydantic EmailStr

    # Check username uniqueness
    existing = db.execute("SELECT id FROM users WHERE username = ?", (user.username,))
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")

    user_id = db.execute_insert(
        "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
        (user.username, user.email, f"hashed_{user.password}", user.role),
    )

    # BUG_USER_02: Response includes password_hash field
    rows = db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return rows[0]
