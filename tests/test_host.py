import asyncio
import copy
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from bookwyrm_adaptive.api import SessionConflict, SessionUnavailable, SubmitAnswer
from bookwyrm_adaptive.host import (
    SessionProcessingUnavailable, SessionSnapshot, StoredSubmission,
    TransactionalConversationHost,
)
from bookwyrm_adaptive.models import Decision, StudentQuestion
from test_adaptive import context


NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)


def snapshot(**overrides):
    ctx = context()
    values = dict(
        session_id=ctx.session_id, assignment_id=ctx.assignment_id, mode=ctx.mode,
        revision=0, outcomes=tuple(ctx.outcomes), passages=tuple(ctx.passages), turns=(),
        pending_turn_id="server-turn-1",
        pending_question=Decision(student=StudentQuestion(
            kind="coverage", outcome_id="satire", text="How is the mayor portrayed?"),
            reason="opening_question"),
        deadline=NOW + timedelta(minutes=10),
    )
    return SessionSnapshot(**(values | overrides))


class MemoryStore:
    """Transactional TEST store. Production stores must use cross-process locks."""

    def __init__(self, state=None):
        self.state = state or snapshot()
        self.submissions = {}
        self.records = []
        self.outbox = set()
        self.lock = asyncio.Lock()
        self.fail_commit = False

    @asynccontextmanager
    async def transaction(self, session_id):
        async with self.lock:
            before = copy.deepcopy((self.state, self.submissions, self.records, self.outbox))
            try:
                yield MemoryTransaction(self, session_id)
                if self.fail_commit:
                    raise RuntimeError("simulated database commit failure")
            except BaseException:
                self.state, self.submissions, self.records, self.outbox = before
                raise


class MemoryTransaction:
    def __init__(self, store, session_id):
        self.store = store
        self.session_id = session_id

    async def load_for_principal(self, principal):
        if principal != "student-1" or self.session_id != self.store.state.session_id:
            return None
        return self.store.state

    async def find_submission(self, turn_id):
        return self.store.submissions.get(turn_id)

    async def commit(self, record):
        store = self.store
        if record.expected_revision != store.state.revision:
            raise SessionConflict()
        store.records.append(record)
        store.submissions[record.submission.turn_id] = StoredSubmission(record.submission, record.reply)
        store.state = replace(
            store.state, revision=record.reply.revision,
            turns=(*store.state.turns, record.answered_turn),
            pending_turn_id=record.next_turn_id, pending_question=None if record.complete else record.decision,
            active=not record.complete,
        )
        if record.complete:
            store.outbox.add(record.session_id)


class HostTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.host = TransactionalConversationHost(self.store, now=lambda: NOW, new_turn_id=lambda: "server-turn-2")
        self.submission = SubmitAnswer(turn_id="server-turn-1", answer="The narrator approves of the mayor.", expected_revision=0)
        self.calls = 0

    async def decide(self, ctx):
        self.calls += 1
        return Decision(student=StudentQuestion(kind="clarify", outcome_id="satire", text="What supports that reading?"),
                        reason="test_follow_up", needs_teacher_review=True, grounding_passage_ids=["p1"])

    async def submit(self, **kwargs):
        return await self.host.advance("student-1", "session-1", self.submission, kwargs.get("decide", self.decide))

    async def test_pending_question_metadata_and_private_decision_are_saved(self):
        reply = await self.submit()
        self.assertEqual(reply.revision, 1)
        self.assertEqual(reply.turn_id, "server-turn-2")
        record = self.store.records[0]
        self.assertEqual(record.answered_turn.question, "How is the mayor portrayed?")
        self.assertEqual(record.answered_turn.question_kind, "coverage")
        self.assertEqual(record.answered_turn.outcome_id, "satire")
        self.assertTrue(record.decision.needs_teacher_review)
        self.assertNotIn("needs_teacher_review", reply.model_dump())

    async def test_issued_probe_metadata_survives_answer_submission(self):
        pending = Decision(student=StudentQuestion(
            kind="misconception_probe", outcome_id="satire", text="How does the contrast fit your interpretation?"),
            reason="probe", misconception_id="literal-praise", question_level="analyze")
        self.store.state = snapshot(pending_question=pending)
        await self.submit()
        answered = self.store.records[0].answered_turn
        self.assertEqual(answered.question_kind, "misconception_probe")
        self.assertEqual(answered.misconception_id, "literal-praise")
        self.assertEqual(answered.question_level, "analyze")

    async def test_invalid_next_outcome_or_grounding_rolls_back(self):
        for outcome_id, passage_ids in (("unknown-outcome", ["p1"]), ("satire", ["another-assignment"]), ("satire", [])):
            async def invalid(_, outcome=outcome_id, passages=passage_ids):
                return Decision(student=StudentQuestion(kind="clarify", outcome_id=outcome, text="Why?"),
                                reason="invalid", grounding_passage_ids=passages)
            with self.assertRaises(ValueError):
                await self.submit(decide=invalid)
        self.assertEqual(self.store.state.revision, 0)
        self.assertEqual(self.store.records, [])

    async def test_retry_returns_committed_reply_without_new_decision(self):
        first = await self.submit()
        second = await self.submit()
        self.assertEqual(first, second)
        self.assertEqual(self.calls, 1)
        self.assertEqual(len(self.store.records), 1)

    async def test_reused_turn_id_with_changed_answer_or_revision_conflicts(self):
        await self.submit()
        for change in ({"answer": "Changed"}, {"expected_revision": 1}):
            altered = self.submission.model_copy(update=change)
            with self.assertRaises(SessionConflict):
                await self.host.advance("student-1", "session-1", altered, self.decide)

    async def test_authorization_checked_before_replay(self):
        await self.submit()
        with self.assertRaises(SessionUnavailable):
            await self.host.advance("other-student", "session-1", self.submission, self.decide)

    async def test_unknown_or_future_turn_ids_rejected(self):
        for turn_id in ("client-invented-id", "server-turn-2"):
            with self.assertRaises(SessionConflict):
                await self.host.advance("student-1", "session-1", self.submission.model_copy(update={"turn_id": turn_id}), self.decide)
        self.assertEqual(self.calls, 0)

    async def test_stale_revision_rejected(self):
        self.store.state = replace(self.store.state, revision=1)
        with self.assertRaises(SessionConflict):
            await self.submit()

    async def test_ineligible_sessions_rejected_without_model(self):
        for flag in ("ready", "started", "active", "ai_allowed", "attempt_allowed"):
            with self.subTest(flag=flag):
                self.store.state = snapshot(**{flag: False})
                with self.assertRaises(SessionUnavailable):
                    await self.submit()
        self.assertEqual(self.calls, 0)

    async def test_expired_in_flight_answer_saved_and_evaluation_enqueued_once(self):
        self.store.state = snapshot(deadline=NOW)
        reply = await self.submit()
        self.assertEqual(reply.question.kind, "complete")
        self.assertIsNone(reply.turn_id)
        self.assertEqual(self.calls, 0)
        self.assertEqual(len(self.store.state.turns), 1)
        self.assertEqual(self.store.outbox, {"session-1"})
        self.assertEqual(await self.submit(), reply)
        self.assertEqual(len(self.store.records), 1)

    async def test_deadline_rechecked_after_model(self):
        times = iter((NOW, NOW + timedelta(hours=1)))
        self.host.now = lambda: next(times)
        reply = await self.submit()
        self.assertEqual(reply.question.kind, "complete")
        self.assertIsNone(reply.turn_id)
        self.assertTrue(self.store.records[0].decision.needs_teacher_review)
        self.assertEqual(self.store.outbox, {"session-1"})

    async def test_prepare_sees_latest_answer_and_refreshes_model_context_only(self):
        seen = []
        async def prepare(ctx):
            self.assertEqual(ctx.turns[-1].answer, self.submission.answer)
            return ctx.model_copy(update={
                "sufficient_outcome_ids": frozenset({"satire"}),
                "turns": [ctx.turns[-1].model_copy(update={"answer": "Redacted model-only answer"})],
            })
        async def decide(ctx):
            seen.append(ctx)
            return await self.decide(ctx)
        self.host.prepare_context = prepare
        await self.submit(decide=decide)
        self.assertEqual(seen[0].sufficient_outcome_ids, frozenset({"satire"}))
        self.assertEqual(seen[0].turns[-1].answer, "Redacted model-only answer")
        self.assertEqual(self.store.records[0].answered_turn.answer, self.submission.answer)

    async def test_deadline_after_preparation_skips_model(self):
        async def prepare(ctx):
            self.host.now = lambda: NOW + timedelta(hours=1)
            return ctx
        self.host.prepare_context = prepare
        reply = await self.submit()
        self.assertEqual(reply.question.kind, "complete")
        self.assertEqual(self.calls, 0)

    async def test_prepare_cannot_change_server_question_or_pinned_versions(self):
        for change in (
            {"mode": "formative"}, {"assignment_id": "other-reading"},
            {"outcomes": [snapshot().outcomes[0].model_copy(update={"version": "other-version"})]},
            {"turns": [context().turns[0].model_copy(update={"id": "invented-id"})]},
        ):
            async def prepare(ctx, updates=change):
                return ctx.model_copy(update=updates)
            self.host.prepare_context = prepare
            with self.assertRaises(ValueError):
                await self.submit()
        self.assertEqual(self.store.records, [])

    async def test_concurrent_duplicate_answers_commit_once(self):
        replies = await asyncio.gather(self.submit(), self.submit())
        self.assertEqual(replies[0], replies[1])
        self.assertEqual(self.calls, 1)
        self.assertEqual(len(self.store.records), 1)

    async def test_concurrent_different_answers_cannot_overwrite(self):
        other = self.submission.model_copy(update={"answer": "Another answer"})
        results = await asyncio.gather(self.submit(), self.host.advance("student-1", "session-1", other, self.decide), return_exceptions=True)
        self.assertEqual(sum(isinstance(r, SessionConflict) for r in results), 1)
        self.assertEqual(len(self.store.records), 1)

    async def test_decision_error_rolls_back_and_can_retry(self):
        async def fail(_):
            raise RuntimeError("model error")
        with self.assertRaises(RuntimeError):
            await self.submit(decide=fail)
        self.assertEqual(self.store.state.revision, 0)
        self.assertEqual(self.store.records, [])
        await self.submit()
        self.assertEqual(self.store.state.revision, 1)

    async def test_transaction_exit_failure_does_not_return_uncommitted_reply(self):
        self.store.fail_commit = True
        with self.assertRaises(RuntimeError):
            await self.submit()
        self.assertEqual(self.store.state.revision, 0)
        self.assertEqual(self.store.records, [])
        self.assertEqual(self.store.submissions, {})

    async def test_bounded_decision_timeout_rolls_back(self):
        self.host.decision_timeout = 0.001
        async def slow(_):
            await asyncio.Event().wait()
        with self.assertRaises(SessionProcessingUnavailable):
            await self.submit(decide=slow)
        self.assertEqual(self.store.state.revision, 0)
        self.assertEqual(self.store.records, [])

    def test_naive_deadline_rejected(self):
        with self.assertRaises(ValueError):
            snapshot(deadline=NOW.replace(tzinfo=None))
