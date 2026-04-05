"""
In-memory SQLite database for the buggy API.
Supports reset between episodes with DOMAIN RANDOMIZATION —
each seed produces different users, tasks, and data distributions
so that every training episode is unique.
"""

import random
import sqlite3
import threading
from contextlib import contextmanager

# Name pools for randomized seed data
FIRST_NAMES = [
    "alice", "bob", "charlie", "diana", "ethan", "fiona", "george", "hannah",
    "ivan", "julia", "kevin", "luna", "mike", "nina", "oscar", "priya",
    "quinn", "ravi", "sara", "tom", "uma", "victor", "wendy", "xander",
]
DOMAINS = ["example.com", "company.org", "startup.io", "work.dev", "test.net"]
TASK_TITLES = [
    "Setup CI/CD pipeline", "Write unit tests", "Fix login page CSS",
    "Database migration", "API documentation", "Refactor auth module",
    "Add rate limiting", "Setup monitoring", "Fix memory leak",
    "Update dependencies", "Add logging middleware", "Create admin panel",
    "Implement caching", "Fix CORS issues", "Add input validation",
    "Setup Docker compose", "Write integration tests", "Fix date parsing bug",
    "Add search functionality", "Implement pagination", "Setup SSL certs",
    "Add webhook support", "Fix timezone handling", "Create backup script",
    "Optimize database queries", "Add email notifications", "Fix file upload",
    "Implement user roles", "Add audit logging", "Setup load balancer",
]
TASK_DESCRIPTIONS = [
    "Configure GitHub Actions for automated deployment",
    "Add tests for the auth module endpoints",
    "Button alignment issue on mobile devices",
    "Migrate from SQLite to PostgreSQL",
    "Document all REST endpoints with examples",
    "Break down the monolithic auth into smaller services",
    "Prevent API abuse with request throttling",
    "Setup Grafana dashboards for key metrics",
    "Memory usage grows unbounded after 1000 requests",
    "Several packages have critical CVEs",
    "Add structured JSON logging to all routes",
    "Build an admin dashboard for user management",
    "Add Redis caching layer for frequent queries",
    "Frontend gets blocked by CORS policy",
    "Sanitize user inputs to prevent injection",
]
STATUSES = ["pending", "in_progress", "done"]
PRIORITIES = ["low", "medium", "high"]


class Database:
    """Thread-safe in-memory SQLite database that can be reset between episodes.

    When a seed is provided, the database is populated with deterministically
    randomized data — different users, tasks, and distributions each time.
    This prevents the agent from memorizing a single fixed dataset.
    """

    def __init__(self, seed: int | None = None):
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._seed = seed
        self.initialize()

    def initialize(self):
        """Create a fresh database with schema and seed data."""
        with self._lock:
            if self._conn:
                self._conn.close()
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._create_schema()
            self._seed_data()

    def _create_schema(self):
        cursor = self._conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                priority TEXT DEFAULT 'medium',
                assignee_email TEXT DEFAULT '',
                owner_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (owner_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS auth_tokens (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
        self._conn.commit()

    def _seed_data(self):
        """Seed the database with randomized data based on the seed.

        With seed=None, uses a fixed default dataset (for manual testing).
        With a seed, generates random users/tasks so every episode differs.
        """
        rng = random.Random(self._seed)
        cursor = self._conn.cursor()

        if self._seed is None:
            # Default fixed data for manual testing / Gradio UI
            cursor.executescript("""
                INSERT INTO users (username, email, password_hash, role) VALUES
                    ('alice', 'alice@example.com', 'hashed_password123', 'admin'),
                    ('bob', 'bob@example.com', 'hashed_password123', 'user'),
                    ('charlie', 'charlie@example.com', 'hashed_password123', 'user');

                INSERT INTO tasks (title, description, status, priority, assignee_email, owner_id) VALUES
                    ('Setup CI/CD pipeline', 'Configure GitHub Actions', 'in_progress', 'high', 'alice@example.com', 1),
                    ('Write unit tests', 'Add tests for auth module', 'pending', 'medium', 'bob@example.com', 2),
                    ('Fix login page CSS', 'Button alignment issue', 'done', 'low', 'charlie@example.com', 3),
                    ('Database migration', 'Migrate to PostgreSQL', 'pending', 'high', 'alice@example.com', 1),
                    ('API documentation', 'Document all endpoints', 'in_progress', 'medium', 'bob@example.com', 2);
            """)
        else:
            # Randomized data — different every episode
            # Pick 3-5 users from the name pool
            num_users = rng.randint(3, 5)
            user_names = rng.sample(FIRST_NAMES, num_users)
            domain = rng.choice(DOMAINS)

            # First user is always admin, rest are regular users
            for i, name in enumerate(user_names):
                role = "admin" if i == 0 else "user"
                email = f"{name}@{domain}"
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                    (name, email, f"hashed_password_{rng.randint(100, 999)}", role),
                )

            # Pick 4-8 tasks with random assignments
            num_tasks = rng.randint(4, 8)
            task_titles = rng.sample(TASK_TITLES, min(num_tasks, len(TASK_TITLES)))
            task_descs = rng.sample(TASK_DESCRIPTIONS, min(num_tasks, len(TASK_DESCRIPTIONS)))

            for i in range(num_tasks):
                owner_id = rng.randint(1, num_users)
                assignee_id = rng.randint(1, num_users)
                assignee_email = f"{user_names[assignee_id - 1]}@{domain}"
                cursor.execute(
                    "INSERT INTO tasks (title, description, status, priority, assignee_email, owner_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        task_titles[i % len(task_titles)],
                        task_descs[i % len(task_descs)] if i < len(task_descs) else "",
                        rng.choice(STATUSES),
                        rng.choice(PRIORITIES),
                        assignee_email,
                        owner_id,
                    ),
                )

        self._conn.commit()

    @property
    def user_names(self) -> list[str]:
        """Get usernames in the database (for the agent's observation)."""
        rows = self.execute("SELECT username FROM users ORDER BY id")
        return [r["username"] for r in rows]

    @contextmanager
    def get_cursor(self):
        with self._lock:
            cursor = self._conn.cursor()
            try:
                yield cursor
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def execute(self, query: str, params: tuple = ()) -> list[dict]:
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if cursor.description:
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            return []

    def execute_insert(self, query: str, params: tuple = ()) -> int:
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.lastrowid

    def execute_update(self, query: str, params: tuple = ()) -> int:
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount
