from typing import Annotated, Awaitable, Callable, Protocol

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field, model_validator

from .engine import AdaptiveEngine
from .models import Context, Decision, Identifier, Model, StudentQuestion, Text


class SubmitAnswer(Model):
    turn_id: Identifier
    answer: Text
    expected_revision: Annotated[int, Field(ge=0, strict=True)]


class StudentReply(Model):
    revision: Annotated[int, Field(ge=0)]
    question: StudentQuestion
    turn_id: Identifier | None = None

    @model_validator(mode="after")
    def pending_turn(self):
        if (self.question.kind == "complete") != (self.turn_id is None):
            raise ValueError("Only active questions require a pending turn ID")
        return self


class SessionConflict(Exception):
    pass


class SessionUnavailable(Exception):
    pass


class ConversationHost(Protocol):
    async def advance(
        self, principal: object, session_id: str, submission: SubmitAnswer,
        decide: Callable[[Context], Awaitable[Decision]],
    ) -> StudentReply:
        """Main-project transaction boundary. Implement in its conversation service.

        1. Authorize principal's ownership/enrollment; hide inaccessible sessions.
        2. Lock session, check ready/started/active, opt-out, attempt limit, and
           expected_revision; deduplicate by turn_id (same ID + body returns the
           saved reply, different body conflicts). Enforce the main timer policy:
           permit the in-flight answer at timeout, but issue no new question.
        3. Save answer; load pinned outcome version, teacher question catalogs,
           assignment-scoped retrieval, and transcript from PostgreSQL. Apply
           existing guardrails and pseudonymization BEFORE invoking decide.
           Student body must never supply mode, outcomes, time, or coverage.
        4. Call decide(context), using a bounded session lock or reservation with
           compare-and-swap; recompute deadline before committing the question.
        5. Persist Decision privately and next question atomically, increment
           revision; on complete enqueue main outcome evaluation, NOT a grade
           publication. Commit before returning StudentReply (student fields only).

        Roll back failures. Retry stale revisions via SessionConflict. The host
        owns durable state; this extension does not create a second session store.
        """
        ...


def create_router(
    engine: AdaptiveEngine,
    host: ConversationHost,
    current_principal: Callable,
) -> APIRouter:
    """Mount in the existing FastAPI app using its auth dependency and service."""
    # Deferred because host imports API data types; create_router runs after
    # module initialization when the main app wires its dependencies.
    from .host import SessionProcessingUnavailable

    router = APIRouter(prefix="/api/conversations", tags=["adaptive-follow-ups"])

    @router.post("/{session_id}/adaptive-answer", response_model=StudentReply)
    async def answer(
        session_id: Identifier, submission: SubmitAnswer,
        principal: object = Depends(current_principal),
    ):
        try:
            return await host.advance(principal, session_id, submission, engine.next_question)
        except SessionUnavailable:
            raise HTTPException(404, "Conversation not available") from None
        except SessionConflict:
            raise HTTPException(409, "Conversation changed; reload before retrying") from None
        except SessionProcessingUnavailable:
            raise HTTPException(503, "Conversation temporarily unavailable; retry the same answer") from None

    return router
