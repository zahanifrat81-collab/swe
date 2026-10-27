"""Run: python -m examples.run_conversation

This SCRIPTED fixture demonstrates integration, not real AI assessment. The
in-memory store is deliberately non-production: it loses state on exit, has no
cross-worker locking, and compares a fixture principal instead of authenticating.
Replace it with the main application's authenticated PostgreSQL store, retrieval,
guardrails and evaluator. No API key, model calls, web server or grades are used.
"""

import asyncio
import copy
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from bookwyrm_adaptive.api import SessionConflict, StudentReply, SubmitAnswer
from bookwyrm_adaptive.engine import AdaptiveEngine
from bookwyrm_adaptive.host import (
    SessionSnapshot, StoredSubmission, TransactionalConversationHost,
)
from bookwyrm_adaptive.models import Analysis, Decision, Outcome, Passage, StudentQuestion

ANSWERS = (
    "The narrator approves of the mayor.",
    "The praise is ironic because the mayor closes the shelter.",
    "The principal's phrase is ironic because a locked library prevents access.",
)


class ScriptedDemoAnalyzer:
    """Exactly three predetermined fixtures; this does NOT detect misconceptions."""

    async def analyze(self, context, outcome):
        answer = context.turns[-1].answer
        if answer not in ANSWERS:
            raise ValueError("This scripted example only accepts its fixture answers")
        misconception = answer == ANSWERS[0]
        return Analysis(
            status="possible_misconception" if misconception else "supported",
            confidence=0.95, student_quote=answer,
            misconception_id="literal-praise" if misconception else None,
            passage_ids=["passage-1"],
        )


class DemoMemoryStore:
    """Non-production fixture implementing the host's transactional store API."""

    def __init__(self, state):
        self.state = state
        self.submissions = {}
        self.records = []
        self.evaluation_outbox = set()
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def transaction(self, session_id):
        async with self.lock:
            previous = copy.deepcopy((self.state, self.submissions, self.records, self.evaluation_outbox))
            try:
                yield DemoTransaction(self, session_id)
            except BaseException:
                self.state, self.submissions, self.records, self.evaluation_outbox = previous
                raise


class DemoTransaction:
    def __init__(self, store, session_id):
        self.store, self.session_id = store, session_id

    async def load_for_principal(self, principal):
        if principal == "demo-student" and self.session_id == self.store.state.session_id:
            return self.store.state
        return None

    async def find_submission(self, turn_id):
        return self.store.submissions.get(turn_id)

    async def commit(self, record):
        if record.expected_revision != self.store.state.revision:
            raise SessionConflict()
        self.store.submissions[record.submission.turn_id] = StoredSubmission(record.submission, record.reply)
        self.store.records.append(record)
        self.store.state = replace(
            self.store.state, revision=record.reply.revision,
            turns=(*self.store.state.turns, record.answered_turn),
            pending_turn_id=record.next_turn_id,
            pending_question=None if record.complete else record.decision,
            active=not record.complete,
        )
        if record.complete:
            self.store.evaluation_outbox.add(record.session_id)


async def run_demo():
    outcome = Outcome(
        id="satire", version="teacher-version-1", level="analyze",
        description="Analyze how ironic praise critiques a public official's actions",
        passage_ids=["passage-1"],
        clarification_question="How does the narrator's wording shape the reader's view of the mayor?",
        misconceptions=[{
            "id": "literal-praise", "description": "Treats ironic praise as literal approval",
            "probe": "How does your interpretation fit the contrast between the mayor's words and actions?",
        }],
        transfer_tasks=[{
            "id": "school-irony", "scenario": "A fictional principal calls a locked library a triumph of access.",
            "question": "How does this contrast shape the message, and how is it like the assigned passage?",
        }],
    )
    opening = Decision(
        student=StudentQuestion(kind="coverage", outcome_id=outcome.id,
                                text="What does the narrator's praise suggest about the mayor, and why?"),
        reason="teacher_approved_opening", question_level="analyze", grounding_passage_ids=["passage-1"],
    )
    store = DemoMemoryStore(SessionSnapshot(
        session_id="demo-session", assignment_id="demo-reading", mode="summative", revision=0,
        outcomes=(outcome,), passages=(Passage(id="passage-1", text=(
            "The mayor praised his generous shelter while closing its doors to the hungry."
        )),), turns=(), pending_turn_id="server-question-1", pending_question=opening,
        deadline=datetime.now(timezone.utc) + timedelta(minutes=10),
    ))
    engine = AdaptiveEngine(ScriptedDemoAnalyzer())
    async def main_project_prepare(context):
        # SCRIPTED stand-in for the MAIN project's outcome coverage service. The
        # extension never awards mastery or grades. Here the final fixture answer
        # to the transfer task provides the demo's explicit sufficiency signal.
        if context.turns[-1].question_kind == "transfer" and context.turns[-1].answer == ANSWERS[2]:
            context = context.model_copy(update={"sufficient_outcome_ids": frozenset({outcome.id})})
        return context

    host = TransactionalConversationHost(store, prepare_context=main_project_prepare)

    print("SCRIPTED DEMO: no real AI, production authentication, database, or grades.\n")
    reply = StudentReply(revision=0, question=opening.student, turn_id="server-question-1")
    for answer in ANSWERS:
        print(f"Bookwyrm: {reply.question.text}\nStudent: {answer}\n")
        submission = SubmitAnswer(turn_id=reply.turn_id, answer=answer, expected_revision=reply.revision)
        reply = await host.advance("demo-student", "demo-session", submission, engine.next_question)

    # Verify that the real host/engine wiring followed the intended contribution.
    assert [record.decision.student.kind for record in store.records] == [
        "misconception_probe", "transfer", "complete",
    ]
    assert reply.question.kind == "complete"
    assert store.evaluation_outbox == {"demo-session"}
    # A network retry replays the committed reply, without duplicate evaluation.
    assert await host.advance("demo-student", "demo-session", submission, engine.next_question) == reply
    assert len(store.records) == 3
    print("Conversation complete: 3 answers saved; one main-evaluator handoff queued.")
    print("Summative grade publication remains with the main project and teacher approval.")
    return store


if __name__ == "__main__":
    asyncio.run(run_demo())
