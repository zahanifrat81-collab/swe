"""Transactional adapter for the main Bookwyrm conversation service.

The host implements the answer lifecycle; the store maps it to the application's
existing database. A store transaction must serialize writes to a session across
workers (for example PostgreSQL SELECT ... FOR UPDATE), not just use a Python lock.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Protocol
from uuid import uuid4

from .api import SessionConflict, SessionUnavailable, StudentReply, SubmitAnswer
from .models import Context, Decision, Outcome, Passage, StudentQuestion, Turn


class SessionProcessingUnavailable(Exception):
    """The decision timed out; the transaction must roll back for a safe retry."""


@dataclass(frozen=True)
class SessionSnapshot:
    """Pinned, trusted state loaded by the main project's store.

    `pending_turn_id` is issued by the SERVER when it commits a question. The
    initial question must likewise receive a server-issued ID at Start. `turns`
    contains answered turns only. The store must apply assignment-scoped retrieval
    and the main service's privacy/guardrail policy before returning model context.
    """

    session_id: str
    assignment_id: str
    mode: Literal["formative", "summative"]
    revision: int
    outcomes: tuple[Outcome, ...]
    passages: tuple[Passage, ...]
    turns: tuple[Turn, ...]
    pending_turn_id: str | None
    pending_question: Decision | None
    ready: bool = True
    started: bool = True
    active: bool = True
    ai_allowed: bool = True
    attempt_allowed: bool = True
    deadline: datetime | None = None
    sufficient_outcome_ids: frozenset[str] = frozenset()

    def __post_init__(self):
        if self.revision < 0:
            raise ValueError("Session revision cannot be negative")
        if self.deadline is not None and self.deadline.utcoffset() is None:
            raise ValueError("Session deadline must be timezone-aware")


@dataclass(frozen=True)
class StoredSubmission:
    submission: SubmitAnswer
    reply: StudentReply


@dataclass(frozen=True)
class AdvanceRecord:
    """All state the store must save in ONE transaction before returning.

    Persist the answered turn, private decision, exact submission/reply (for
    retries), and next pending turn ID/question at reply.revision. `complete`
    closes the session and adds a durable, idempotent outcome-evaluation outbox
    entry in the SAME transaction. It never publishes a grade.
    """

    session_id: str
    expected_revision: int
    submission: SubmitAnswer
    answered_turn: Turn
    decision: Decision
    reply: StudentReply
    next_turn_id: str | None
    complete: bool


class ConversationTransaction(Protocol):
    async def load_for_principal(self, principal: object) -> SessionSnapshot | None:
        """Return state only if principal still owns/enrolls in this session.

        Return None for absent or inaccessible sessions, including revoked class
        membership. Completed sessions must remain loadable for authorized replay.
        """
        ...

    async def find_submission(self, turn_id: str) -> StoredSubmission | None:
        """Look up an already committed submission in this locked session."""
        ...

    async def commit(self, record: AdvanceRecord) -> None:
        """Atomically save all record fields using expected_revision as a CAS.

        Raise SessionConflict on revision mismatch. Uniquely constrain session +
        turn ID and the evaluation outbox key. Keep decisions teacher-private.
        Returning is not enough: the surrounding transaction must commit durably
        on successful exit and roll back on exceptions/cancellation.
        """
        ...


class ConversationStore(Protocol):
    def transaction(self, session_id: str) -> AbstractAsyncContextManager[ConversationTransaction]:
        """Lock one session across processes until commit/rollback on exit.

        Configure bounded DB lock/statement timeouts in the implementation. No
        network model calls should outlive the host's decision timeout.
        """
        ...


class TransactionalConversationHost:
    """Concrete implementation of api.ConversationHost using the main DB store."""

    def __init__(
        self,
        store: ConversationStore,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        new_turn_id: Callable[[], str] = lambda: uuid4().hex,
        decision_timeout: float = 5.0,
        prepare_context: Callable[[Context], Awaitable[Context]] | None = None,
    ):
        if decision_timeout <= 0:
            raise ValueError("Decision timeout must be positive")
        self.store = store
        self.now = now
        self.new_turn_id = new_turn_id
        self.decision_timeout = decision_timeout
        # Connect the main service's latest-answer privacy, guardrail, retrieval
        # and coverage pipeline here. It shares the bounded decision budget.
        self.prepare_context = prepare_context

    def _expired(self, deadline: datetime | None) -> bool:
        now = self.now()
        if now.utcoffset() is None:
            raise ValueError("Clock must return a timezone-aware datetime")
        return deadline is not None and now >= deadline

    @staticmethod
    def _validate_prepared(original: Context, prepared: Context) -> None:
        if (prepared.session_id, prepared.assignment_id, prepared.mode) != (
            original.session_id, original.assignment_id, original.mode,
        ):
            raise ValueError("Preparation cannot change session, assignment or mode")
        if [(o.id, o.version) for o in prepared.outcomes] != [(o.id, o.version) for o in original.outcomes]:
            raise ValueError("Preparation cannot change pinned outcome identities or versions")
        # Allow answer redaction and assignment-scoped retrieval updates while
        # preserving the server-issued question identity and ordering. The store
        # persists the ORIGINAL answer, not the redacted model-only view.
        def identity(turn):
            return (turn.id, turn.outcome_id, turn.question_kind, turn.question,
                    turn.misconception_id, turn.transfer_id, turn.question_level)
        if [identity(t) for t in prepared.turns] != [identity(t) for t in original.turns]:
            raise ValueError("Preparation cannot change issued question metadata")

    async def advance(
        self,
        principal: object,
        session_id: str,
        submission: SubmitAnswer,
        decide: Callable[[Context], Awaitable[Decision]],
    ) -> StudentReply:
        async with self.store.transaction(session_id) as transaction:
            snapshot = await transaction.load_for_principal(principal)
            if snapshot is None or snapshot.session_id != session_id:
                raise SessionUnavailable()

            # Authorize before replay. Replays may follow a completed session and
            # carry an older revision, so deduplicate before active/revision checks.
            saved = await transaction.find_submission(submission.turn_id)
            if saved is not None:
                if saved.submission != submission:
                    raise SessionConflict("Turn ID reused with a different submission")
                return saved.reply

            if not all((snapshot.ready, snapshot.started, snapshot.active,
                        snapshot.ai_allowed, snapshot.attempt_allowed)):
                raise SessionUnavailable()
            if submission.expected_revision != snapshot.revision:
                raise SessionConflict("Stale session revision")
            if snapshot.pending_turn_id != submission.turn_id or snapshot.pending_question is None:
                raise SessionConflict("Answer does not match the pending question")

            pending = snapshot.pending_question
            question = pending.student
            if question.kind == "complete" or question.outcome_id is None or question.text is None:
                raise SessionConflict("No answerable question is pending")
            turn = Turn(
                id=submission.turn_id, outcome_id=question.outcome_id,
                question_kind=question.kind, question=question.text,
                answer=submission.answer, misconception_id=pending.misconception_id,
                transfer_id=pending.transfer_id, question_level=pending.question_level,
            )
            context = Context(
                session_id=session_id, assignment_id=snapshot.assignment_id,
                mode=snapshot.mode, outcomes=list(snapshot.outcomes),
                passages=list(snapshot.passages), turns=[*snapshot.turns, turn],
                sufficient_outcome_ids=snapshot.sufficient_outcome_ids,
                time_expired=self._expired(snapshot.deadline),
            )
            if context.time_expired:
                # Accept the answer to the already-issued question at timeout.
                # There is no reason to call the model after the deadline.
                decision = Decision(student=StudentQuestion(kind="complete"), reason="server_deadline")
            else:
                try:
                    async with asyncio.timeout(self.decision_timeout):
                        if self.prepare_context is not None:
                            prepared = await self.prepare_context(context)
                            prepared = Context.model_validate(prepared.model_dump())
                            self._validate_prepared(context, prepared)
                            context = prepared
                        if self.prepare_context is not None and self._expired(snapshot.deadline):
                            decision = Decision(student=StudentQuestion(kind="complete"), reason="server_deadline")
                        else:
                            decision = await decide(context)
                except TimeoutError:
                    raise SessionProcessingUnavailable("Decision timed out; nothing committed") from None
                # Revalidate even if a custom decide callback constructed a model
                # with model_copy/model_construct, which bypass field validators.
                decision = Decision.model_validate(decision.model_dump())
                if decision.student.kind != "complete":
                    target = next((o for o in context.outcomes if o.id == decision.student.outcome_id), None)
                    if target is None:
                        raise ValueError("Decision references an unknown outcome")
                    if not decision.grounding_passage_ids or not set(decision.grounding_passage_ids) <= set(target.passage_ids):
                        raise ValueError("Decision is not grounded in the target outcome")
                    if decision.misconception_id is not None and decision.misconception_id not in {m.id for m in target.misconceptions}:
                        raise ValueError("Decision references an unknown misconception")
                    if decision.transfer_id is not None and decision.transfer_id not in {t.id for t in target.transfer_tasks}:
                        raise ValueError("Decision references an unknown transfer task")
                # A question generated while time remained must not be issued if
                # the deadline passed during retrieval/model generation.
                if self._expired(snapshot.deadline):
                    decision = decision.model_copy(update={
                        "student": StudentQuestion(kind="complete"),
                        "reason": "server_deadline",
                        "misconception_id": None, "transfer_id": None, "question_level": None,
                        "grounding_passage_ids": [],
                    })

            complete = decision.student.kind == "complete"
            next_turn_id = None if complete else self.new_turn_id()
            reply = StudentReply(
                revision=snapshot.revision + 1, question=decision.student,
                turn_id=next_turn_id,
            )
            record = AdvanceRecord(
                session_id=session_id, expected_revision=snapshot.revision,
                submission=submission, answered_turn=turn, decision=decision,
                reply=reply, next_turn_id=next_turn_id, complete=complete,
            )
            await transaction.commit(record)
        # Exiting the store's context must complete the database commit BEFORE a
        # student sees this reply. A connection failure then permits safe replay.
        return reply
