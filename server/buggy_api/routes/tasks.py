"""
Task CRUD routes with planted bugs.

BUGS PLANTED:
- BUG_TASK_01 (easy):   GET /tasks/{id} returns 200 with null body for non-existent task (should be 404)
- BUG_TASK_02 (easy):   POST /tasks with missing required 'title' returns 500 instead of 400/422
- BUG_TASK_03 (easy):   GET /tasks?page=-1 returns 200 instead of 400
- BUG_TASK_04 (medium):  PUT /tasks/{id} doesn't validate assignee_email format
- BUG_TASK_05 (medium):  DELETE /tasks/{id} returns 200 even for non-existent task (should be 404)
- BUG_TASK_06 (medium):  GET /tasks?limit=999999 has no pagination cap (potential DoS)
- BUG_TASK_07 (hard):    GET /tasks/{id} of another user's task returns data (BOLA/IDOR vulnerability)
- BUG_TASK_08 (hard):    POST /tasks with very long title (>5000 chars) causes 500 (no input length validation)
- BUG_TASK_09 (hard):    POST /tasks with SQL injection payload in title doesn't sanitize (uses parameterized
                         queries so no actual injection, but the input is stored verbatim — a content injection)
- BUG_TASK_10 (hard):    No rate limiting — rapid sequential requests all succeed
"""

from fastapi import APIRouter, HTTPException, Header, Query
from typing import Optional

from ..database import Database
from ..models import TaskCreate, TaskUpdate

router = APIRouter(prefix="/tasks", tags=["tasks"])

_db: Database | None = None

# Simple in-memory cache for BUG demonstration
_cache: dict[int, dict] = {}


def set_db(db: Database):
    global _db, _cache
    _db = db
    _cache = {}


def get_db() -> Database:
    return _db


@router.get("")
def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    sort: Optional[str] = Query(None, description="Sort field"),
    page: Optional[int] = Query(None, description="Page number"),
    limit: Optional[int] = Query(None, description="Items per page"),
    authorization: Optional[str] = Header(None),
):
    db = get_db()

    # BUG_TASK_03: No validation for negative page numbers
    # Should check: if page is not None and page < 1: raise HTTPException(400, ...)

    # BUG_TASK_06: No cap on limit — agent can request limit=999999
    # Should cap at e.g. 100

    query = "SELECT * FROM tasks WHERE 1=1"
    params = []

    if status:
        query += " AND status = ?"
        params.append(status)
    if priority:
        query += " AND priority = ?"
        params.append(priority)

    if sort:
        allowed_sorts = ["created_at", "updated_at", "title", "priority", "status"]
        if sort in allowed_sorts:
            query += f" ORDER BY {sort}"
        else:
            query += " ORDER BY created_at"
    else:
        query += " ORDER BY created_at DESC"

    if limit is not None:
        # BUG_TASK_06: No upper bound check on limit
        query += " LIMIT ?"
        params.append(limit)
    else:
        query += " LIMIT 20"

    if page is not None and limit is not None:
        # BUG_TASK_03: Allows negative offset — page=-1 with limit=10 gives offset=-10
        offset = (page - 1) * limit
        query += " OFFSET ?"
        params.append(offset)

    rows = db.execute(query, tuple(params))
    return rows


@router.get("/{task_id}")
def get_task(
    task_id: int,
    authorization: Optional[str] = Header(None),
):
    db = get_db()

    # Check cache first (used later for stale cache bug)
    if task_id in _cache:
        return _cache[task_id]

    rows = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))

    # BUG_TASK_01: Returns 200 with null instead of 404
    if not rows:
        return None  # Should be: raise HTTPException(status_code=404, detail="Task not found")

    task = rows[0]

    # BUG_TASK_07: No ownership check — any authenticated user can see any task
    # Should check: if user and task["owner_id"] != user["id"]: raise HTTPException(403)

    # Cache the result
    _cache[task_id] = task
    return task


@router.post("/create", status_code=201)
def create_task_internal(
    task: TaskCreate,
    authorization: Optional[str] = Header(None),
):
    """Internal create — used by the raw handler after parsing."""
    db = get_db()

    # BUG_TASK_08: No title length validation
    # Should check: if len(task.title) > 200: raise HTTPException(400, ...)

    # BUG_TASK_09: No content sanitization — SQL injection payloads stored verbatim
    # While parameterized queries prevent actual SQL injection, the content
    # is stored and returned as-is, which is a content injection / XSS vector

    # Determine owner — default to user 1 if no auth
    owner_id = 1
    if authorization:
        token = authorization.replace("Bearer ", "")
        token_rows = db.execute(
            "SELECT user_id FROM auth_tokens WHERE token = ?", (token,)
        )
        if token_rows:
            owner_id = token_rows[0]["user_id"]

    task_id = db.execute_insert(
        "INSERT INTO tasks (title, description, status, priority, assignee_email, owner_id) VALUES (?, ?, ?, ?, ?, ?)",
        (task.title, task.description, task.status, task.priority, task.assignee_email, owner_id),
    )

    rows = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    result = rows[0]
    _cache[task_id] = result
    return result


@router.put("/{task_id}")
def update_task(
    task_id: int,
    task: TaskUpdate,
    authorization: Optional[str] = Header(None),
):
    db = get_db()

    existing = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="Task not found")

    # BUG_TASK_04: No email format validation on assignee_email
    # Should validate if task.assignee_email is provided

    # BUG_TASK_07: No ownership check on update either
    updates = []
    params = []
    for field_name in ["title", "description", "status", "priority", "assignee_email"]:
        value = getattr(task, field_name, None)
        if value is not None:
            updates.append(f"{field_name} = ?")
            params.append(value)

    if updates:
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(task_id)
        db.execute_update(
            f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )

    rows = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    result = rows[0]
    _cache[task_id] = result
    return result


@router.delete("/{task_id}")
def delete_task(
    task_id: int,
    authorization: Optional[str] = Header(None),
):
    db = get_db()

    # BUG_TASK_05: No existence check — returns 200 even for non-existent tasks
    # Should check existence first and return 404
    db.execute_update("DELETE FROM tasks WHERE id = ?", (task_id,))

    # Note: cache is NOT cleared — this enables stale cache detection
    # (BUG_TASK_01 variant: deleted task still returned from cache)

    return {"message": "Task deleted", "id": task_id}
