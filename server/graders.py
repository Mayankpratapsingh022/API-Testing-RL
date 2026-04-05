"""
Task-specific grading logic.

Each task has a grader that computes a final score (0.0 - 1.0)
based on what the agent accomplished during the episode.
"""

from dataclasses import dataclass


@dataclass
class GradeResult:
    score: float
    breakdown: dict[str, float]
    feedback: str


class TaskGrader:
    """Computes final scores for each task based on episode performance."""

    @staticmethod
    def grade(
        task_id: str,
        bugs_found: set[str],
        coverage_pct: float,
        endpoints_tested: int,
        total_endpoints: int,
        method_endpoint_pairs: int,
        status_codes_seen: set[int],
        action_history: list[dict],
        created_resources: dict[str, list],
    ) -> GradeResult:
        if task_id == "basic_validation":
            return TaskGrader._grade_basic(
                bugs_found, coverage_pct, endpoints_tested, total_endpoints,
                method_endpoint_pairs, status_codes_seen, action_history, created_resources,
            )
        elif task_id == "edge_cases":
            return TaskGrader._grade_edge_cases(
                bugs_found, coverage_pct, endpoints_tested, method_endpoint_pairs,
                status_codes_seen, action_history, created_resources,
            )
        elif task_id == "security_workflows":
            return TaskGrader._grade_security(
                bugs_found, coverage_pct, action_history, created_resources,
            )
        return GradeResult(score=0.0, breakdown={}, feedback="Unknown task")

    @staticmethod
    def _grade_basic(
        bugs_found, coverage_pct, endpoints_tested, total_endpoints,
        method_endpoint_pairs, status_codes_seen, action_history, created_resources,
    ) -> GradeResult:
        breakdown = {}

        # 0.25: Test all GET endpoints
        get_endpoints = {
            h.get("endpoint") for h in action_history
            if h.get("method", "").upper() == "GET"
        }
        get_score = min(len(get_endpoints) / 4, 1.0) * 0.25
        breakdown["get_coverage"] = round(get_score, 3)

        # 0.20: Test POST with valid data
        post_success = sum(
            1 for h in action_history
            if h.get("method", "").upper() == "POST" and h.get("response_status") == 201
        )
        post_score = min(post_success / 2, 1.0) * 0.20
        breakdown["post_testing"] = round(post_score, 3)

        # 0.15: Test PUT/DELETE
        put_delete = sum(
            1 for h in action_history
            if h.get("method", "").upper() in ("PUT", "DELETE")
        )
        pd_score = min(put_delete / 2, 1.0) * 0.15
        breakdown["put_delete"] = round(pd_score, 3)

        # 0.20: Bug discovery (easy bugs: TASK_01, TASK_02, TASK_03)
        easy_bugs = {"BUG_TASK_01", "BUG_TASK_02", "BUG_TASK_03"}
        found_easy = len(bugs_found & easy_bugs)
        bug_score = min(found_easy / 2, 1.0) * 0.20
        breakdown["bugs_found"] = round(bug_score, 3)

        # 0.20: Response schema validation (status codes variety)
        schema_score = min(len(status_codes_seen) / 4, 1.0) * 0.20
        breakdown["schema_validation"] = round(schema_score, 3)

        score = sum(breakdown.values())
        feedback_parts = []
        if get_score > 0:
            feedback_parts.append(f"GET coverage: {len(get_endpoints)} endpoints")
        if post_success > 0:
            feedback_parts.append(f"POST success: {post_success}")
        if found_easy > 0:
            feedback_parts.append(f"Bugs found: {found_easy}/{len(easy_bugs)}")

        return GradeResult(
            score=round(min(score, 1.0), 4),
            breakdown=breakdown,
            feedback="; ".join(feedback_parts) if feedback_parts else "No significant progress",
        )

    @staticmethod
    def _grade_edge_cases(
        bugs_found, coverage_pct, endpoints_tested, method_endpoint_pairs,
        status_codes_seen, action_history, created_resources,
    ) -> GradeResult:
        breakdown = {}

        # 0.15: Missing required fields testing
        missing_field_tests = sum(
            1 for h in action_history
            if h.get("method", "").upper() == "POST"
            and h.get("body") is not None
            and isinstance(h.get("body"), dict)
            and not h["body"].get("title")
        )
        breakdown["missing_fields"] = round(min(missing_field_tests / 2, 1.0) * 0.15, 3)

        # 0.15: Invalid data type testing
        invalid_tests = sum(
            1 for h in action_history
            if h.get("body") and isinstance(h.get("body"), dict)
            and any(
                isinstance(v, (list, bool)) or v == ""
                for v in h["body"].values()
            )
        )
        breakdown["invalid_types"] = round(min(invalid_tests / 2, 1.0) * 0.15, 3)

        # 0.15: Boundary value testing (negative pages, huge limits, long strings)
        boundary_tests = 0
        for h in action_history:
            qp = h.get("query_params", {})
            if qp.get("page") is not None and int(str(qp.get("page", 1))) < 1:
                boundary_tests += 1
            if qp.get("limit") is not None and int(str(qp.get("limit", 10))) > 100:
                boundary_tests += 1
        breakdown["boundary_values"] = round(min(boundary_tests / 2, 1.0) * 0.15, 3)

        # 0.15: Non-existent resource testing
        nonexistent_tests = sum(
            1 for h in action_history
            if h.get("method", "").upper() in ("GET", "DELETE", "PUT")
            and "/999" in h.get("endpoint", "")
        )
        breakdown["nonexistent_resources"] = round(min(nonexistent_tests / 2, 1.0) * 0.15, 3)

        # 0.20: Bug discovery (medium bugs)
        medium_bugs = {
            "BUG_TASK_04", "BUG_TASK_05", "BUG_TASK_06",
            "BUG_USER_01", "BUG_USER_02", "BUG_AUTH_02",
        }
        all_relevant = medium_bugs | {"BUG_TASK_01", "BUG_TASK_02", "BUG_TASK_03"}
        found_relevant = len(bugs_found & all_relevant)
        breakdown["bugs_found"] = round(min(found_relevant / 3, 1.0) * 0.20, 3)

        # 0.20: Dependency chaining (create → read → update → delete)
        chain_score = 0.0
        if any(h.get("method") == "POST" and h.get("response_status") == 201 for h in action_history):
            chain_score += 0.25
        if created_resources.get("tasks"):
            task_ids = created_resources["tasks"]
            for tid in task_ids:
                gets = [h for h in action_history if h.get("endpoint") == f"/tasks/{tid}" and h.get("method") == "GET"]
                puts = [h for h in action_history if h.get("endpoint") == f"/tasks/{tid}" and h.get("method") == "PUT"]
                deletes = [h for h in action_history if h.get("endpoint") == f"/tasks/{tid}" and h.get("method") == "DELETE"]
                if gets:
                    chain_score += 0.25
                if puts:
                    chain_score += 0.25
                if deletes:
                    chain_score += 0.25
                break  # Only need one complete chain
        breakdown["dependency_chaining"] = round(min(chain_score, 1.0) * 0.20, 3)

        score = sum(breakdown.values())
        return GradeResult(
            score=round(min(score, 1.0), 4),
            breakdown=breakdown,
            feedback=f"Edge cases: {found_relevant} bugs found, chain score {chain_score:.0%}",
        )

    @staticmethod
    def _grade_security(
        bugs_found, coverage_pct, action_history, created_resources,
    ) -> GradeResult:
        breakdown = {}

        # 0.20: Cross-user authorization testing
        cross_user = False
        login_users = set()
        for h in action_history:
            if h.get("endpoint") == "/auth/login" and h.get("response_status") == 200:
                body = h.get("body", {})
                if body:
                    login_users.add(body.get("username"))
        cross_user = len(login_users) >= 2
        breakdown["cross_user_auth"] = 0.20 if cross_user else 0.0

        # 0.20: Injection pattern testing
        injection_attempted = sum(
            1 for h in action_history
            if h.get("body") and isinstance(h.get("body"), dict)
            and any(
                pattern.lower() in str(h["body"]).lower()
                for pattern in ["DROP TABLE", "<script>", "OR 1=1", "UNION SELECT", "'; --"]
            )
        )
        breakdown["injection_testing"] = round(min(injection_attempted / 2, 1.0) * 0.20, 3)

        # 0.20: Multi-step state consistency
        # Check if agent did: create → delete → re-fetch (stale cache test)
        consistency_tests = 0
        for i, h in enumerate(action_history):
            if h.get("method") == "DELETE" and "/tasks/" in h.get("endpoint", ""):
                # Check if agent re-fetched the same resource after deleting
                deleted_endpoint = h["endpoint"]
                for j in range(i + 1, len(action_history)):
                    if action_history[j].get("endpoint") == deleted_endpoint and action_history[j].get("method") == "GET":
                        consistency_tests += 1
                        break
        breakdown["state_consistency"] = round(min(consistency_tests, 1.0) * 0.20, 3)

        # 0.20: Security bug discovery
        security_bugs = {"BUG_TASK_07", "BUG_AUTH_01", "BUG_TASK_08", "BUG_TASK_09"}
        found_security = len(bugs_found & security_bugs)
        breakdown["security_bugs"] = round(min(found_security / 2, 1.0) * 0.20, 3)

        # 0.20: Complete workflow coverage
        workflow_coverage = min(coverage_pct / 80, 1.0)  # 80% coverage = full score
        breakdown["workflow_coverage"] = round(workflow_coverage * 0.20, 3)

        score = sum(breakdown.values())
        return GradeResult(
            score=round(min(score, 1.0), 4),
            breakdown=breakdown,
            feedback=f"Security: {found_security} security bugs, {len(login_users)} users tested, {injection_attempted} injection attempts",
        )
