"""
Baseline agents for the API Testing Environment.

Three agents of increasing sophistication:
1. RandomAgent     — Picks random endpoints/methods (lower bound)
2. SequentialAgent — Systematically tests each endpoint in order
3. SmartAgent      — Chains requests and probes for known bug patterns
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import APITestAction, HTTPMethod


class RandomAgent:
    """Randomly picks endpoints and methods. Baseline for comparison."""

    name = "random"

    ENDPOINTS = ["/tasks", "/tasks/1", "/tasks/2", "/tasks/999", "/users", "/users/1", "/auth/login"]
    METHODS = ["GET", "POST", "PUT", "DELETE"]

    def act(self, observation: dict) -> APITestAction:
        method = random.choice(self.METHODS)
        endpoint = random.choice(self.ENDPOINTS)
        body = None
        headers = {}

        if method == "POST" and endpoint == "/tasks":
            body = {"title": f"Random task {random.randint(1, 100)}"}
        elif method == "POST" and endpoint == "/auth/login":
            body = {"username": random.choice(["alice", "bob"]), "password": "pass"}
        elif method == "POST" and endpoint == "/users":
            body = {"username": f"user{random.randint(100, 999)}", "email": "test@test.com", "password": "pass"}
        elif method == "PUT":
            endpoint = f"/tasks/{random.randint(1, 5)}"
            body = {"title": "Updated"}

        return APITestAction(
            method=HTTPMethod(method) if method in ("GET", "POST", "PUT", "DELETE") else HTTPMethod.GET,
            endpoint=endpoint,
            headers=headers,
            body=body,
        )


class SequentialAgent:
    """Systematically tests each endpoint with valid requests."""

    name = "sequential"

    def __init__(self):
        self.step = 0

    def act(self, observation: dict) -> APITestAction:
        self.step += 1
        actions = self._get_action_sequence()
        idx = min(self.step - 1, len(actions) - 1)
        return actions[idx]

    def _get_action_sequence(self) -> list[APITestAction]:
        return [
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks", expected_status=200),
            APITestAction(method=HTTPMethod.GET, endpoint="/users", expected_status=200),
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks/1", expected_status=200),
            APITestAction(method=HTTPMethod.GET, endpoint="/users/1", expected_status=200),
            APITestAction(method=HTTPMethod.POST, endpoint="/auth/login",
                          body={"username": "alice", "password": "password123"}, expected_status=200),
            APITestAction(method=HTTPMethod.POST, endpoint="/tasks",
                          body={"title": "Test Task", "description": "Created by baseline"}, expected_status=201),
            APITestAction(method=HTTPMethod.POST, endpoint="/users",
                          body={"username": "testuser", "email": "test@example.com", "password": "test123"},
                          expected_status=201),
            APITestAction(method=HTTPMethod.PUT, endpoint="/tasks/1",
                          body={"title": "Updated Task"}, expected_status=200),
            APITestAction(method=HTTPMethod.DELETE, endpoint="/tasks/5", expected_status=200),
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks/999999", expected_status=404),
            APITestAction(method=HTTPMethod.POST, endpoint="/tasks",
                          body={"description": "No title"}, expected_status=400),
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks",
                          query_params={"page": -1, "limit": 10}, expected_status=400),
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks",
                          query_params={"status": "done"}, expected_status=200),
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks",
                          query_params={"sort": "title"}, expected_status=200),
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks/2", expected_status=200),
        ]


class SmartAgent:
    """Heuristic agent that chains requests and probes for bugs."""

    name = "smart"

    def __init__(self):
        self.step = 0
        self.auth_tokens = {}
        self.created_ids = []

    def act(self, observation: dict) -> APITestAction:
        self.step += 1

        if isinstance(observation, dict):
            self.auth_tokens = observation.get("auth_tokens", self.auth_tokens)
            ids = observation.get("known_resource_ids", {})
            for rtype, id_list in ids.items():
                for rid in id_list:
                    if rid not in self.created_ids:
                        self.created_ids.append(rid)

        actions = self._get_smart_sequence()
        idx = min(self.step - 1, len(actions) - 1)
        return actions[idx]

    def _get_smart_sequence(self) -> list[APITestAction]:
        alice_token = self.auth_tokens.get("alice", "")
        bob_token = self.auth_tokens.get("bob", "")
        alice_auth = {"Authorization": f"Bearer {alice_token}"} if alice_token else {}
        bob_auth = {"Authorization": f"Bearer {bob_token}"} if bob_token else {}

        return [
            # Phase 1: Discovery
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks", expected_status=200),
            APITestAction(method=HTTPMethod.GET, endpoint="/users", expected_status=200),
            # Phase 2: Authentication
            APITestAction(method=HTTPMethod.POST, endpoint="/auth/login",
                          body={"username": "alice", "password": "password123"}, expected_status=200),
            APITestAction(method=HTTPMethod.POST, endpoint="/auth/login",
                          body={"username": "bob", "password": "password123"}, expected_status=200),
            # Phase 3: CRUD with auth
            APITestAction(method=HTTPMethod.POST, endpoint="/tasks",
                          body={"title": "Alice's task", "description": "Test"},
                          headers=alice_auth, expected_status=201),
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks/1", headers=alice_auth, expected_status=200),
            # Phase 4: Easy bugs
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks/999999", expected_status=404),
            APITestAction(method=HTTPMethod.POST, endpoint="/tasks",
                          body={"description": "no title"}, expected_status=400),
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks",
                          query_params={"page": -1, "limit": 10}, expected_status=400),
            # Phase 5: Medium bugs
            APITestAction(method=HTTPMethod.PUT, endpoint="/tasks/1",
                          body={"assignee_email": "not-an-email"}, expected_status=422),
            APITestAction(method=HTTPMethod.DELETE, endpoint="/tasks/99999", expected_status=404),
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks",
                          query_params={"limit": 999999}, expected_status=200),
            # Phase 6: User bugs
            APITestAction(method=HTTPMethod.POST, endpoint="/users",
                          body={"username": "baduser", "email": "invalid-email", "password": "test"},
                          expected_status=422),
            APITestAction(method=HTTPMethod.POST, endpoint="/auth/login",
                          body={"username": "alice", "password": ""}, expected_status=401),
            # Phase 7: BOLA
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks/1",
                          headers=bob_auth, expected_status=403),
            # Phase 8: Injection
            APITestAction(method=HTTPMethod.POST, endpoint="/tasks",
                          body={"title": "test'; DROP TABLE tasks;--"}, expected_status=201),
            APITestAction(method=HTTPMethod.POST, endpoint="/tasks",
                          body={"title": "A" * 6000}, expected_status=400),
            # Phase 9: Cross-user modification
            APITestAction(method=HTTPMethod.PUT, endpoint="/tasks/1",
                          body={"title": "Bob modified Alice's task"},
                          headers=bob_auth, expected_status=403),
            # Phase 10: State consistency
            APITestAction(method=HTTPMethod.POST, endpoint="/tasks",
                          body={"title": "Ephemeral task"}, expected_status=201),
            APITestAction(method=HTTPMethod.DELETE, endpoint="/tasks/6", expected_status=200),
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks/6", expected_status=404),
            # Phase 11: Coverage
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks",
                          query_params={"status": "done"}, expected_status=200),
            APITestAction(method=HTTPMethod.GET, endpoint="/tasks",
                          query_params={"sort": "title"}, expected_status=200),
            APITestAction(method=HTTPMethod.GET, endpoint="/users/2", expected_status=200),
            # Phase 12: Password hash check
            APITestAction(method=HTTPMethod.POST, endpoint="/users",
                          body={"username": "newuser2", "email": "valid@email.com", "password": "pass"},
                          expected_status=201),
        ]


AGENTS = {
    "random": RandomAgent,
    "sequential": SequentialAgent,
    "smart": SmartAgent,
}
