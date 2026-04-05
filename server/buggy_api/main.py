"""
The deliberately buggy REST API — a task management system.

This API is the system-under-test. It has intentionally planted bugs at varying
difficulty levels that the AI agent must discover through intelligent testing.

The API runs in-process via Starlette's TestClient (no separate port needed).
"""

import json
import logging

from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
from typing import Optional

from .database import Database
from .routes import tasks as tasks_routes
from .routes import users as users_routes
from .routes import auth as auth_routes
from .models import TaskCreate

logger = logging.getLogger(__name__)


def create_buggy_api(db: Database) -> FastAPI:
    """Create a fresh buggy API instance wired to the given database."""
    api = FastAPI(
        title="TaskTracker API",
        description="A task management API (with bugs)",
        version="1.0.0",
    )

    # Wire database into route modules
    tasks_routes.set_db(db)
    users_routes.set_db(db)
    auth_routes.set_db(db)

    # Include standard routes
    api.include_router(tasks_routes.router)
    api.include_router(users_routes.router)
    api.include_router(auth_routes.router)

    # BUG_TASK_02 + BUG_TASK_08: Raw POST /tasks handler that doesn't use Pydantic validation
    # This allows missing fields and overly long inputs to cause 500 errors
    @api.post("/tasks", status_code=201)
    async def create_task_raw(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        try:
            body = await request.json()
        except Exception:
            # BUG_TASK_02: Returns 500 on malformed/empty body instead of 400
            raise Exception("Failed to parse request body")

        if not isinstance(body, dict):
            raise Exception("Invalid body format")

        title = body.get("title")

        # BUG_TASK_02: No check for missing title — causes KeyError/500 below
        if title is None:
            # This SHOULD return 400, but we let it fall through to cause 500
            # Simulate an internal error from missing required field
            raise Exception("Internal error: title is required but was None")

        # BUG_TASK_08: No length validation on title
        if len(title) > 5000:
            # Simulate a database error from overly long input
            raise Exception(f"Database error: value too long for column 'title' (length={len(title)})")

        task_data = TaskCreate(
            title=title,
            description=body.get("description", ""),
            status=body.get("status", "pending"),
            priority=body.get("priority", "medium"),
            assignee_email=body.get("assignee_email", ""),
        )
        return tasks_routes.create_task_internal(task_data, authorization)

    # Global error handler — returns 500 for unhandled exceptions
    @api.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": str(exc)},
        )

    return api
