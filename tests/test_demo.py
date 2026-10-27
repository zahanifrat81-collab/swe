import unittest
import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from bookwyrm_adaptive.demo import create_app
from bookwyrm_adaptive.demo_support import (
    INCOMPLETE_ANSWER, MISCONCEPTION_ANSWER, SUPPORTED_ANSWER, TRANSFER_ANSWER,
)
from bookwyrm_adaptive.models import Analysis
from bookwyrm_adaptive.engine import AnalysisUnavailable


BASE_URL = "http://127.0.0.1:8000"


class DemoBrowserTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(mode="scripted")
        self.client = TestClient(self.app, base_url=BASE_URL)
        self.client.__enter__()
        self.bootstrap = self.client.get("/api/demo/session")
        self.state = self.bootstrap.json()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def post(self, path, body=None, client=None, token=None):
        return (client or self.client).post(
            f"/api/demo/{path}", json=body,
            headers={"X-Demo-CSRF": token or self.state["csrf_token"]},
        )

    def start(self):
        response = self.post("start")
        self.assertEqual(response.status_code, 200, response.text)
        self.state = response.json()
        return self.state

    def answer(self, text):
        reply = self.state["reply"]
        submission = dict(turn_id=reply["turn_id"], answer=text, expected_revision=reply["revision"])
        response = self.post("answer", submission)
        self.assertEqual(response.status_code, 200, response.text)
        self.state = response.json()
        return submission, self.state

    def test_bootstrap_does_not_reveal_opening_and_cookie_is_http_only(self):
        self.assertEqual(self.state["mode"], "scripted")
        self.assertFalse(self.state["started"])
        self.assertIsNone(self.state["reply"])
        self.assertIsNone(self.state["session_id"])
        self.assertEqual(self.state["transcript"], [])
        self.assertIn("script", self.state["notice"].lower())
        self.assertIn("bookwyrm_demo", self.client.cookies)
        self.assertIn("httponly", self.bootstrap.headers["set-cookie"].lower())
        self.assertIn("samesite=strict", self.bootstrap.headers["set-cookie"].lower())
        response = self.client.get("/api/demo/session")
        # A repeated GET preserves the browser's CSRF identity and pre-start state.
        self.assertEqual(response.json()["csrf_token"], self.state["csrf_token"])

    def test_start_is_idempotent_and_refresh_restores_conversation(self):
        state = self.start()
        self.assertTrue(state["started"])
        self.assertEqual(state["reply"]["revision"], 0)
        self.assertTrue(state["reply"]["turn_id"])
        self.assertEqual(len(state["transcript"]), 1)
        self.assertEqual(state["transcript"][0]["role"], "assistant")
        again = self.post("start").json()
        self.assertEqual(again["session_id"], state["session_id"])
        self.assertEqual(again["reply"], state["reply"])
        refreshed = self.client.get("/api/demo/session").json()
        self.assertEqual(refreshed["transcript"], state["transcript"])

    def test_scripted_misconception_probe_transfer_and_completion(self):
        self.start()
        _, probe = self.answer(MISCONCEPTION_ANSWER)
        self.assertEqual(probe["reply"]["question"]["kind"], "misconception_probe")
        self.assertIn(MISCONCEPTION_ANSWER, probe["reply"]["question"]["text"])
        self.assertFalse(probe["complete"])
        _, transfer = self.answer(SUPPORTED_ANSWER)
        self.assertEqual(transfer["reply"]["question"]["kind"], "transfer")
        self.assertIn(TRANSFER_ANSWER, transfer["examples"])
        _, completed = self.answer(TRANSFER_ANSWER)
        self.assertTrue(completed["complete"])
        self.assertEqual(completed["reply"]["question"]["kind"], "complete")
        self.assertIsNone(completed["reply"]["turn_id"])
        self.assertEqual(completed["examples"], [])
        self.assertEqual(sum(t["role"] == "student" for t in completed["transcript"]), 3)

    def test_arbitrary_text_uses_uncertain_clarification_without_faux_semantic_detection(self):
        self.start()
        text = "<img src=x onerror=alert(1)> This answer was typed by the browser user."
        _, state = self.answer(text)
        self.assertEqual(state["reply"]["question"]["kind"], "clarify")
        self.assertEqual(state["transcript"][1], {"role": "student", "text": text})
        self.assertIn(text, state["reply"]["question"]["text"])

    def test_incomplete_example_gets_clarification(self):
        self.start()
        _, state = self.answer(INCOMPLETE_ANSWER)
        self.assertEqual(state["reply"]["question"]["kind"], "clarify")

    def test_original_passage_example_does_not_complete_a_transfer_question(self):
        self.start()
        _, transfer = self.answer(SUPPORTED_ANSWER)
        self.assertEqual(transfer["reply"]["question"]["kind"], "transfer")
        _, incomplete = self.answer(SUPPORTED_ANSWER)
        self.assertFalse(incomplete["complete"])
        self.assertEqual(incomplete["reply"]["question"]["kind"], "clarify")

    def test_retry_returns_same_receipt_without_duplicate_transcript(self):
        self.start()
        submission, first = self.answer(MISCONCEPTION_ANSWER)
        response = self.post("answer", submission)
        self.assertEqual(response.status_code, 200, response.text)
        second = response.json()
        self.assertEqual(second["reply"], first["reply"])
        self.assertEqual(second["transcript"], first["transcript"])
        self.assertEqual(sum(t["role"] == "student" for t in second["transcript"]), 1)

    def test_changed_retry_or_old_revision_cannot_overwrite_answer(self):
        self.start()
        submission, first = self.answer(MISCONCEPTION_ANSWER)
        changed = self.post("answer", submission | {"answer": "Another answer"})
        self.assertEqual(changed.status_code, 409)
        stale = self.post("answer", dict(
            turn_id=first["reply"]["turn_id"], answer=SUPPORTED_ANSWER, expected_revision=0,
        ))
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(self.client.get("/api/demo/session").json()["transcript"], first["transcript"])

    def test_csrf_required_and_tokens_bound_to_browser_cookie(self):
        self.assertEqual(self.client.post("/api/demo/start").status_code, 403)
        self.assertEqual(self.post("start", token="wrong-token").status_code, 403)
        # One TestClient owns application lifespan; the second represents an
        # independent browser without starting/stopping the same app twice.
        other = TestClient(self.app, base_url=BASE_URL)
        try:
            other_state = other.get("/api/demo/session").json()
            self.assertNotEqual(other_state["csrf_token"], self.state["csrf_token"])
            self.assertEqual(self.post("start", client=other).status_code, 403)
        finally:
            other.close()
        self.assertFalse(self.client.get("/api/demo/session").json()["started"])

    def test_browsers_have_independent_conversations(self):
        first = self.start()
        other = TestClient(self.app, base_url=BASE_URL)
        try:
            other_state = other.get("/api/demo/session").json()
            second = self.post("start", client=other, token=other_state["csrf_token"]).json()
            self.assertNotEqual(second["session_id"], first["session_id"])
            self.assertNotEqual(second["reply"]["turn_id"], first["reply"]["turn_id"])
            attack = dict(turn_id=first["reply"]["turn_id"], answer=SUPPORTED_ANSWER, expected_revision=0)
            self.assertEqual(self.post("answer", attack, client=other, token=other_state["csrf_token"]).status_code, 409)
        finally:
            other.close()
        self.assertEqual(self.client.get("/api/demo/session").json()["transcript"], first["transcript"])

    def test_reset_clears_only_this_browser_and_invalidates_old_pending_question(self):
        previous = self.start()
        old_submission = dict(turn_id=previous["reply"]["turn_id"], answer=MISCONCEPTION_ANSWER, expected_revision=0)
        response = self.post("reset")
        self.assertEqual(response.status_code, 200)
        self.state = response.json()
        self.assertFalse(self.state["started"])
        self.assertEqual(self.state["transcript"], [])
        self.assertIsNone(self.state["reply"])
        restarted = self.start()
        self.assertNotEqual(restarted["session_id"], previous["session_id"])
        self.assertEqual(self.post("answer", old_submission).status_code, 409)

    def test_public_session_never_exposes_private_diagnostics(self):
        self.start()
        _, state = self.answer(MISCONCEPTION_ANSWER)
        private_keys = {"analysis", "confidence", "reason", "misconception_id", "needs_teacher_review"}
        def inspect(value):
            if isinstance(value, dict):
                self.assertTrue(private_keys.isdisjoint(value.keys()), value)
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
        inspect(state)

    def test_student_cannot_override_trusted_context(self):
        self.start()
        reply = self.state["reply"]
        response = self.post("answer", dict(
            turn_id=reply["turn_id"], answer="My answer", expected_revision=0, mode="summative",
        ))
        self.assertEqual(response.status_code, 422)

    def test_cross_origin_and_malformed_origin_are_rejected(self):
        for origin in ("https://outside.example", "http://[", "null"):
            with self.subTest(origin=origin):
                response = self.client.post("/api/demo/start", headers={
                    "X-Demo-CSRF": self.state["csrf_token"], "Origin": origin,
                })
                self.assertEqual(response.status_code, 403)
        allowed = self.client.post("/api/demo/start", headers={
            "X-Demo-CSRF": self.state["csrf_token"], "Origin": BASE_URL,
        })
        self.assertEqual(allowed.status_code, 200)

    def test_page_assets_and_response_headers(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Bookwyrm", response.text)
        self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.client.get("/assets/style.css").status_code, 200)
        self.assertEqual(self.client.get("/assets/app.js").status_code, 200)

    def test_missing_browser_cookie_requires_fresh_bootstrap(self):
        fresh = TestClient(self.app, base_url=BASE_URL)
        try:
            response = fresh.post("/api/demo/start", headers={"X-Demo-CSRF": self.state["csrf_token"]})
            self.assertEqual(response.status_code, 401)
        finally:
            fresh.close()


class DemoConfigurationTests(unittest.TestCase):
    def test_capacity_is_bounded_without_evicting_an_active_browser(self):
        app = create_app(mode="scripted", max_sessions=1)
        with TestClient(app, base_url=BASE_URL) as first:
            second = TestClient(app, base_url=BASE_URL)
            try:
                original = first.get("/api/demo/session")
                self.assertEqual(original.status_code, 200)
                self.assertEqual(second.get("/api/demo/session").status_code, 503)
                self.assertEqual(first.get("/api/demo/session").json()["csrf_token"], original.json()["csrf_token"])
            finally:
                second.close()

    def test_live_mode_can_use_injected_analyzer_without_a_network_call(self):
        class Analyzer:
            calls = 0
            async def analyze(self, context, outcome):
                self.calls += 1
                return Analysis(status="supported", confidence=0.95,
                                student_quote=context.turns[-1].answer, misconception_id=None,
                                passage_ids=outcome.passage_ids)
        analyzer = Analyzer()
        app = create_app(mode="live", analyzer=analyzer)
        with TestClient(app, base_url=BASE_URL) as client:
            initial = client.get("/api/demo/session").json()
            headers = {"X-Demo-CSRF": initial["csrf_token"]}
            started = client.post("/api/demo/start", headers=headers).json()
            self.assertEqual(started["mode"], "live")
            answer = client.post("/api/demo/answer", headers=headers, json=dict(
                turn_id=started["reply"]["turn_id"], answer="My interpretation", expected_revision=0,
            ))
            self.assertEqual(answer.status_code, 200, answer.text)
            self.assertEqual(answer.json()["reply"]["question"]["kind"], "transfer")
        self.assertEqual(analyzer.calls, 1)

    def test_live_provider_failure_falls_back_without_exposing_internal_error(self):
        class UnavailableAnalyzer:
            async def analyze(self, *_):
                raise AnalysisUnavailable("private provider details")
        app = create_app(mode="live", analyzer=UnavailableAnalyzer())
        with TestClient(app, base_url=BASE_URL) as client:
            initial = client.get("/api/demo/session").json()
            headers = {"X-Demo-CSRF": initial["csrf_token"]}
            started = client.post("/api/demo/start", headers=headers).json()
            response = client.post("/api/demo/answer", headers=headers, json=dict(
                turn_id=started["reply"]["turn_id"], answer="My own interpretation", expected_revision=0,
            ))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"]["question"]["kind"], "clarify")
        self.assertIn("could not analyze", response.json()["notice"])
        self.assertNotIn("private provider details", response.text)

    def test_invalid_mode_capacity_or_missing_live_configuration_fails_early(self):
        for settings in ({"mode": "invented"}, {"mode": "scripted", "max_sessions": 0},
                         {"mode": "scripted", "max_sessions": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                create_app(**settings)
        with patch.dict(os.environ, {"BOOKWYRM_MODEL_API_KEY": "", "BOOKWYRM_MODEL": ""}):
            with self.assertRaises(ValueError):
                create_app(mode="live")


if __name__ == "__main__":
    unittest.main()
