import json
import unittest

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from bookwyrm_adaptive.api import (
    SessionConflict, SessionUnavailable, StudentReply, create_router,
)
from bookwyrm_adaptive.engine import AdaptiveEngine, AnalysisUnavailable
from bookwyrm_adaptive.models import Analysis, Context
from bookwyrm_adaptive.provider import OpenAIAnalyzer


def context(**overrides):
    data = dict(
        session_id="session-1", assignment_id="reading-1", mode="summative",
        outcomes=[dict(
            id="satire", version="v1", description="Analyze how irony critiques power",
            level="analyze", passage_ids=["p1"],
            clarification_question="How does the narrator's wording shape the reader's view of the mayor?",
            misconceptions=[dict(id="literal-praise", description="Treats ironic praise as literal approval",
                                 probe="How does that interpretation fit the contrast between the mayor's words and actions?")],
            transfer_tasks=[dict(id="new-irony", scenario="A fictional principal calls a locked library a triumph of access.",
                                 question="How does the contrast shape the message, and how is this like the assigned passage?")],
        )],
        passages=[dict(id="p1", text="The mayor praised his generous shelter while closing its doors to the hungry.")],
        turns=[dict(id="turn-1", outcome_id="satire", question_kind="coverage",
                    question="How is the mayor portrayed?", answer="The narrator approves of the mayor.")],
    )
    data.update(overrides)
    return Context.model_validate(data)


def analysis(status="possible_misconception", **overrides):
    data = dict(status=status, confidence=0.9, student_quote="approves of the mayor",
                misconception_id="literal-praise" if status == "possible_misconception" else None,
                passage_ids=["p1"])
    data.update(overrides)
    return Analysis(**data)


class StubAnalyzer:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def analyze(self, *_):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def decide(self, result=None, ctx=None):
        return await AdaptiveEngine(StubAnalyzer(result or analysis())).next_question(ctx or context())

    async def test_misconception_is_probed_not_corrected(self):
        result = await self.decide()
        self.assertEqual(result.student.kind, "misconception_probe")
        self.assertIn("approves of the mayor", result.student.text)
        self.assertNotIn("Treats ironic praise", result.student.text)
        self.assertEqual(result.grounding_passage_ids, ["p1"])
        self.assertFalse(result.needs_teacher_review)

    async def test_incomplete_answer_gets_clarification(self):
        result = await self.decide(analysis("incomplete"))
        self.assertEqual(result.student.kind, "clarify")

    async def test_supported_answer_gets_same_outcome_transfer(self):
        result = await self.decide(analysis("supported"))
        self.assertEqual(result.student.kind, "transfer")
        self.assertEqual(result.student.outcome_id, "satire")
        self.assertIn("fictional principal", result.student.text)
        self.assertEqual(result.transfer_id, "new-irony")

    async def test_low_confidence_is_not_a_misconception_label(self):
        result = await self.decide(analysis(confidence=0.2))
        self.assertTrue(result.needs_teacher_review)
        self.assertEqual(result.student.kind, "clarify")
        self.assertIsNone(result.misconception_id)

    async def test_bad_model_evidence_falls_back(self):
        for a in [analysis(student_quote="fabricated"), analysis(passage_ids=["another-assignment"]),
                  analysis(misconception_id="invented"), analysis("supported", misconception_id="literal-praise")]:
            with self.subTest(a=a):
                result = await self.decide(a)
                self.assertTrue(result.needs_teacher_review)
                self.assertIsNone(result.analysis)
                self.assertEqual(result.student.kind, "clarify")

    async def test_provider_failure_does_not_invent_diagnosis(self):
        result = await self.decide(AnalysisUnavailable())
        self.assertTrue(result.needs_teacher_review)
        self.assertEqual(result.student.kind, "clarify")

    async def test_expired_session_issues_no_question_or_model_call(self):
        provider = StubAnalyzer(analysis())
        result = await AdaptiveEngine(provider).next_question(context(time_expired=True))
        self.assertEqual(result.student.kind, "complete")
        self.assertIsNone(result.student.text)
        self.assertEqual(provider.calls, 0)

    async def test_never_probed_outcome_takes_priority(self):
        ctx = context()
        other = ctx.outcomes[0].model_copy(update={"id": "second"})
        ctx = context(outcomes=[ctx.outcomes[0].model_dump(), other.model_dump()])
        result = await self.decide(ctx=ctx)
        self.assertEqual(result.student.kind, "coverage")
        self.assertEqual(result.student.outcome_id, "second")

    async def test_repeated_probe_requests_teacher_review(self):
        ctx = context()
        turn = ctx.turns[0].model_copy(update={"misconception_id": "literal-praise", "question_kind": "misconception_probe"})
        result = await self.decide(ctx=context(turns=[turn.model_dump()]))
        self.assertEqual(result.student.kind, "clarify")
        self.assertTrue(result.needs_teacher_review)

    async def test_transfer_is_not_repeated(self):
        ctx = context()
        turn = ctx.turns[0].model_copy(update={"transfer_id": "new-irony", "question_kind": "transfer"})
        result = await self.decide(analysis("supported"), context(turns=[turn.model_dump()]))
        self.assertEqual(result.student.kind, "clarify")

    async def test_only_main_engine_coverage_ends_review(self):
        result = await self.decide(analysis("incomplete"), context(sufficient_outcome_ids=["satire"]))
        self.assertEqual(result.student.kind, "complete")

    def test_invalid_server_references_rejected(self):
        with self.assertRaises(ValidationError):
            context(passages=[dict(id="wrong", text="Wrong passage")])


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_adapter_serialization_and_parse(self):
        def handle(request):
            body = json.loads(request.content)
            self.assertEqual(body["response_format"]["type"], "json_schema")
            payload = json.loads(body["messages"][1]["content"])
            self.assertNotIn("session_id", payload)
            self.assertEqual(payload["passages"][0]["id"], "p1")
            return httpx.Response(200, json={"choices": [{"message": {"content": analysis().model_dump_json()}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            ctx = context()
            result = await OpenAIAnalyzer(client, "test-only-key", "configured-model").analyze(ctx, ctx.outcomes[0])
            self.assertEqual(result.status, "possible_misconception")

    async def test_http_error_and_refusal_become_unavailable(self):
        for response in [httpx.Response(429), httpx.Response(200, json={"choices": [{"message": {"refusal": "declined"}}]}),
                         httpx.Response(200, json={"choices": []})]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
                ctx = context()
                with self.assertRaises(AnalysisUnavailable):
                    await OpenAIAnalyzer(client, "test-key", "model").analyze(ctx, ctx.outcomes[0])


class TestHost:
    """Test-only main-project adapter; never used as production persistence."""
    def __init__(self):
        self.private_decision = None

    async def advance(self, principal, session_id, submission, decide):
        if session_id != "session-1" or principal != "student-1":
            raise SessionUnavailable()
        if submission.expected_revision != 0:
            raise SessionConflict()
        ctx = context()
        turn = ctx.turns[0].model_copy(update={"id": submission.turn_id, "answer": submission.answer})
        self.private_decision = await decide(context(turns=[turn.model_dump()]))
        return StudentReply(revision=1, question=self.private_decision.student)


class ApiTests(unittest.TestCase):
    def setUp(self):
        async def principal(authorization: str | None = Header(default=None)):
            if authorization != "test-session":
                raise HTTPException(401)
            return "student-1"
        self.host = TestHost()
        app = FastAPI()
        app.include_router(create_router(AdaptiveEngine(StubAnalyzer(analysis())), self.host, principal))
        self.client = TestClient(app)
        self.body = dict(turn_id="turn-1", answer="The narrator approves of the mayor.", expected_revision=0)

    def post(self, session="session-1", **kwargs):
        return self.client.post(f"/api/conversations/{session}/adaptive-answer",
                                headers={"Authorization": "test-session"}, json=self.body | kwargs)

    def test_main_service_integration_and_student_privacy(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["question"]["kind"], "misconception_probe")
        self.assertEqual(set(response.json()), {"revision", "question"})
        self.assertNotIn("confidence", response.text)
        self.assertIsNotNone(self.host.private_decision.analysis)

    def test_authentication_required(self):
        response = self.client.post("/api/conversations/session-1/adaptive-answer", json=self.body)
        self.assertEqual(response.status_code, 401)

    def test_unknown_session_and_stale_revision(self):
        self.assertEqual(self.post(session="someone-else").status_code, 404)
        self.assertEqual(self.post(expected_revision=99).status_code, 409)

    def test_student_cannot_override_mode_coverage_or_timer(self):
        for field, value in [("mode", "formative"), ("time_expired", False), ("sufficient_outcome_ids", ["satire"])]:
            self.assertEqual(self.post(**{field: value}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
