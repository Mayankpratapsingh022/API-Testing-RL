"""
FastAPI application for the API Testing Environment.

Endpoints:
    - POST /reset:  Reset the environment
    - POST /step:   Execute an action
    - GET  /state:  Get current environment state
    - GET  /schema: Get action/observation schemas
    - WS   /ws:     WebSocket endpoint for persistent sessions
    - GET  /        Info page

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import os
import logging

try:
    from openenv.core.env_server.http_server import create_app
    from ..models import APITestAction, APITestObservation
    from .environment import APITestEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from models import APITestAction, APITestObservation
    from server.environment import APITestEnvironment

logger = logging.getLogger(__name__)

app = create_app(
    APITestEnvironment,
    APITestAction,
    APITestObservation,
    env_name="api_testing_env",
    max_concurrent_envs=int(os.environ.get("MAX_ENVS", "1")),
)


@app.get("/")
async def root():
    return {
        "name": "API Testing Environment",
        "description": "An OpenEnv RL environment where an AI agent learns to test REST APIs intelligently",
        "tasks": ["basic_validation", "edge_cases", "security_workflows"],
        "docs": "/docs",
        "schema": "/schema",
    }


@app.get("/tasks")
async def list_tasks():
    """List available tasks with descriptions."""
    from .environment import TASKS
    return {
        task_id: {
            "description": task["description"],
            "difficulty": task["difficulty"],
            "max_steps": task["max_steps"],
            "total_bugs": task["total_bugs"],
        }
        for task_id, task in TASKS.items()
    }


def main(host: str = "0.0.0.0", port: int = None):
    """Entry point for `uv run server`."""
    import uvicorn

    if port is None:
        port = int(os.environ.get("PORT", "8000"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="API Testing Environment server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
