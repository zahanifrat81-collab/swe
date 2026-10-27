import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from bookwyrm_adaptive.api import StudentReply, SubmitAnswer, create_router
from bookwyrm_adaptive.engine import AdaptiveEngine, AnalysisUnavailable
from bookwyrm_adaptive.host import SessionProcessingUnavailable
from bookwyrm_adaptive.models import Decision, StudentQuestion, Turn
from test_adaptive import StubAnalyzer, analysis, context


def two_outcomes():
    first = context().outcomes[0]
    second = first.model_copy(update={"id": "second"})
    return context(outcomes=[first.model_dump(), second.model_dump()])


class ProgressionRegressions(unittest.IsolatedAsyncioTestCase):
    async def test_misconception_not_discarded_by_outcome_switch(self):
        ctx = two_outcomes()
        engine = AdaptiveEngine(StubAnalyzer(analysis()))
        probe = await engine.next_question(ctx)
        self.assertEqual(probe.student.kind, "misconception_probe")
        self.assertEqual(probe.student.outcome_id, "satire")
        answered = Turn(id="t2", outcome_id="satire", question_kind=probe.student.kind,
                        question=probe.student.text, answer=ctx.turns[-1].answer,
                        misconception_id=probe.misconception_id, question_level=probe.question_level)
        ctx = context(outcomes=[o.model_dump() for o in ctx.outcomes],
                      turns=[t.model_dump() for t in [*ctx.turns, answered]])
        switch = await engine.next_question(ctx)
        self.assertEqual(switch.student.kind, "coverage")
        self.assertEqual(switch.student.outcome_id, "second")
        self.assertTrue(switch.needs_teacher_review)
        self.assertEqual(switch.analysis_turn_id, "t2")
        self.assertEqual(switch.analysis_outcome_id, "satire")
        self.assertIn(analysis().student_quote, switch.student.text)

    async def test_one_transfer_then_visit_remaining_outcomes(self):
        ctx = two_outcomes()
        engine = AdaptiveEngine(StubAnalyzer(analysis("supported")))
        transfer = await engine.next_question(ctx)
        self.assertEqual(transfer.student.kind, "transfer")
        answered = Turn(id="t2", outcome_id="satire", question_kind="transfer",
                        question=transfer.student.text, answer=ctx.turns[-1].answer,
                        transfer_id=transfer.transfer_id, question_level=transfer.question_level)
        switched = await engine.next_question(context(
            outcomes=[o.model_dump() for o in ctx.outcomes],
            turns=[t.model_dump() for t in [*ctx.turns, answered]],
        ))
        self.assertEqual(switched.student.outcome_id, "second")

    async def test_provider_failure_still_references_student_answer(self):
        result = await AdaptiveEngine(StubAnalyzer(AnalysisUnavailable())).next_question(two_outcomes())
        self.assertIn("The narrator approves of the mayor.", result.student.text)
        self.assertEqual(result.student.outcome_id, "second")
        self.assertTrue(result.needs_teacher_review)

    async def test_steering_chooses_least_probed_insufficient_outcome(self):
        ctx = two_outcomes()
        original = ctx.turns[0]
        turns = [original.model_copy(update={"id": f"t{i}"}) for i in range(3)]
        turns.insert(1, original.model_copy(update={"id": "other", "outcome_id": "second"}))
        result = await AdaptiveEngine(StubAnalyzer(analysis("incomplete"))).next_question(context(
            outcomes=[o.model_dump() for o in ctx.outcomes], turns=[t.model_dump() for t in turns],
        ))
        self.assertEqual(result.student.outcome_id, "second")

    async def test_budget_exhaustion_requests_review_without_claiming_mastery(self):
        ctx = context()
        turns = [ctx.turns[0].model_copy(update={"id": f"t{i}"}) for i in range(5)]
        result = await AdaptiveEngine(StubAnalyzer(analysis("incomplete"))).next_question(
            context(turns=[t.model_dump() for t in turns]))
        self.assertEqual(result.student.kind, "complete")
        self.assertEqual(result.reason, "question_budget_exhausted")
        self.assertTrue(result.needs_teacher_review)

    async def test_unseen_outcome_not_skipped_when_current_budget_exhausted(self):
        ctx = two_outcomes()
        turns = [ctx.turns[0].model_copy(update={"id": f"t{i}"}) for i in range(5)]
        result = await AdaptiveEngine(StubAnalyzer(analysis())).next_question(context(
            outcomes=[o.model_dump() for o in ctx.outcomes], turns=[t.model_dump() for t in turns],
        ))
        self.assertEqual(result.student.kind, "coverage")
        self.assertEqual(result.student.outcome_id, "second")

    async def test_sufficient_tag_does_not_replace_higher_order_question(self):
        ctx = context()
        outcome = ctx.outcomes[0].model_copy(update={"transfer_tasks": []})
        turn = ctx.turns[0].model_copy(update={"question_level": "remember"})
        result = await AdaptiveEngine(StubAnalyzer(analysis("supported"))).next_question(context(
            outcomes=[outcome.model_dump()], turns=[turn.model_dump()], sufficient_outcome_ids=["satire"],
        ))
        self.assertEqual(result.reason, "higher_order_coverage")
        self.assertEqual(result.question_level, "analyze")

    async def test_unmet_higher_order_coverage_at_budget_limit_requires_review(self):
        ctx = context()
        outcome = ctx.outcomes[0].model_copy(update={"transfer_tasks": []})
        turns = [ctx.turns[0].model_copy(update={"id": f"t{i}", "question_level": None}) for i in range(5)]
        result = await AdaptiveEngine(StubAnalyzer(analysis("supported"))).next_question(context(
            outcomes=[outcome.model_dump()], turns=[t.model_dump() for t in turns],
            sufficient_outcome_ids=["satire"],
        ))
        self.assertEqual(result.student.kind, "complete")
        self.assertEqual(result.reason, "question_budget_exhausted")
        self.assertTrue(result.needs_teacher_review)

    async def test_transfer_does_not_raise_recall_outcome_to_apply(self):
        ctx = context()
        outcome = ctx.outcomes[0].model_copy(update={"level": "remember"})
        result = await AdaptiveEngine(StubAnalyzer(analysis("supported"))).next_question(
            context(outcomes=[outcome.model_dump()]))
        self.assertEqual(result.student.kind, "clarify")
        self.assertEqual(result.question_level, "remember")

    async def test_catalog_can_offer_different_clarification_questions(self):
        ctx = context()
        primary = ctx.outcomes[0].clarification_question
        outcome = ctx.outcomes[0].model_copy(update={"follow_up_questions": ["How does the shelter's closure affect your interpretation?"]})
        turn = ctx.turns[0].model_copy(update={"question": primary})
        result = await AdaptiveEngine(StubAnalyzer(analysis("incomplete"))).next_question(context(
            outcomes=[outcome.model_dump()], turns=[turn.model_dump()],
        ))
        self.assertIn(outcome.follow_up_questions[0], result.student.text)

    async def test_composed_transfer_question_can_be_answered_at_allowed_input_sizes(self):
        ctx = context()
        task = ctx.outcomes[0].transfer_tasks[0].model_copy(update={"scenario": "S" * 12000, "question": "Q" * 12000})
        outcome = ctx.outcomes[0].model_copy(update={"transfer_tasks": [task]})
        result = await AdaptiveEngine(StubAnalyzer(analysis("supported"))).next_question(
            context(outcomes=[outcome.model_dump()]))
        answered = Turn(id="t2", outcome_id="satire", question_kind="transfer", transfer_id=task.id,
                        question=result.student.text, answer="My reasoning")
        self.assertGreater(len(answered.question), 24000)


class BoundaryRegressions(unittest.TestCase):
    def test_invalid_question_catalog_metadata_rejected_before_persistence(self):
        for kind, kwargs in [("transfer", {}), ("misconception_probe", {}),
                             ("clarify", {"transfer_id": "task"}),
                             ("complete", {"misconception_id": "m1"})]:
            question = StudentQuestion(kind=kind, outcome_id=None if kind == "complete" else "o1",
                                       text=None if kind == "complete" else "Why?")
            with self.subTest(kind=kind), self.assertRaises(ValidationError):
                Decision(student=question, reason="invalid", **kwargs)

    def test_response_requires_server_issued_pending_id(self):
        with self.assertRaises(ValidationError):
            StudentReply(revision=1, question=StudentQuestion(kind="clarify", outcome_id="o1", text="Why?"))

    def test_boolean_revision_is_not_integer_revision(self):
        with self.assertRaises(ValidationError):
            SubmitAnswer(turn_id="t1", answer="Answer", expected_revision=True)

    def test_timeout_api_returns_retryable_error_without_internal_details(self):
        class UnavailableHost:
            async def advance(self, *_):
                raise SessionProcessingUnavailable("internal details must stay private")
        app = FastAPI()
        app.include_router(create_router(AdaptiveEngine(StubAnalyzer(analysis())), UnavailableHost(), lambda: "student"))
        with TestClient(app) as client:
            response = client.post("/api/conversations/s1/adaptive-answer", json={
                "turn_id": "t1", "answer": "My answer", "expected_revision": 0,
            })
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("internal details", response.text)


if __name__ == "__main__":
    unittest.main()
