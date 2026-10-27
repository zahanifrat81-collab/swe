"""Fixtures for the local browser demo, not production assessment or persistence."""

import asyncio
import copy
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import uuid4

from .api import SessionConflict
from .host import SessionSnapshot, StoredSubmission
from .models import Analysis, Decision, Outcome, Passage, StudentQuestion

PASSAGE = Passage(
    id="passage-1",
    text=(
        'The mayor stood beside the shelter and declared, "No town cares more for its people. '
        'Our generosity is unmatched." Behind him, a worker fastened a chain across the entrance. '
        'Families waited in the rain while the mayor waved to the cameras. '
        'What a splendid welcome, the narrator observed.'
    ),
)
MISCONCEPTION_ANSWER = "The narrator approves of the mayor because he says the town is generous."
SUPPORTED_ANSWER = "The praise is ironic: the mayor claims generosity while families are locked out in the rain."
INCOMPLETE_ANSWER = "The mayor talks about the shelter."
TRANSFER_ANSWER = "Calling the locked library a triumph of access is ironic because the locked doors prevent access."
TRANSFER_INCOMPLETE = "The principal is talking about a library."

OUTCOME = Outcome(
    id="irony", version="demo-v1", level="analyze", passage_ids=[PASSAGE.id],
    description="Analyze how ironic praise contrasts an official's claims with their actions.",
    clarification_question="How does the contrast between the speech and the closed entrance shape your interpretation?",
    follow_up_questions=[
        "Which detail about the waiting families supports your interpretation of the narrator's tone?",
        "How would the narrator's words sound different if the shelter doors were open?",
    ],
    misconceptions=[{
        "id": "literal-praise",
        "description": "Treats the narrator's ironic praise as literal approval of the mayor.",
        "probe": "How does your interpretation fit the contrast between the mayor's speech and the families waiting outside?",
    }],
    transfer_tasks=[{
        "id": "library-access",
        "scenario": 'A fictional principal calls a locked library "a triumph of access" while students wait outside.',
        "question": "How does the contrast shape the message? Connect your reasoning to the assigned passage.",
    }],
)
OPENING = Decision(
    student=StudentQuestion(
        kind="coverage", outcome_id=OUTCOME.id,
        text="What does the narrator's final remark suggest about the mayor? Use a detail from the passage to explain.",
    ),
    reason="demo_opening", question_level="analyze", grounding_passage_ids=[PASSAGE.id],
)


class ScriptedAnalyzer:
    """Exact example lookup, deliberately NOT a semantic/keyword AI detector."""

    async def analyze(self, context, outcome):
        answer = context.turns[-1].answer
        transfer = context.turns[-1].question_kind == "transfer"
        if transfer and answer == TRANSFER_ANSWER:
            status, misconception, confidence = "supported", None, 0.95
        elif transfer and answer in (MISCONCEPTION_ANSWER, SUPPORTED_ANSWER, INCOMPLETE_ANSWER, TRANSFER_INCOMPLETE):
            status, misconception, confidence = "incomplete", None, 0.95
        elif not transfer and answer == MISCONCEPTION_ANSWER:
            status, misconception, confidence = "possible_misconception", "literal-praise", 0.95
        elif not transfer and answer == SUPPORTED_ANSWER:
            status, misconception, confidence = "supported", None, 0.95
        elif answer in (INCOMPLETE_ANSWER, TRANSFER_INCOMPLETE):
            status, misconception, confidence = "incomplete", None, 0.95
        else:
            status, misconception, confidence = "uncertain", None, 0.2
        return Analysis(
            status=status, confidence=confidence, misconception_id=misconception,
            student_quote=answer[:240], passage_ids=[PASSAGE.id],
        )


class MemoryStore:
    """One browser's demo session. Memory only, one process; not a DB adapter."""

    def __init__(self, principal: str):
        self.principal = principal
        self.state = SessionSnapshot(
            session_id=uuid4().hex, assignment_id="irony-demo", mode="formative", revision=0,
            outcomes=(OUTCOME,), passages=(PASSAGE,), turns=(),
            pending_turn_id=uuid4().hex, pending_question=OPENING,
            # Untimed local practice; no main-project summative policy implied.
            deadline=None,
        )
        self.records = []
        self.submissions = {}
        self.outbox = set()
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def transaction(self, session_id):
        async with self.lock:
            before = copy.deepcopy((self.state, self.records, self.submissions, self.outbox))
            try:
                yield MemoryTransaction(self, session_id)
            except BaseException:
                self.state, self.records, self.submissions, self.outbox = before
                raise


class MemoryTransaction:
    def __init__(self, store, session_id):
        self.store, self.session_id = store, session_id

    async def load_for_principal(self, principal):
        if principal == self.store.principal and self.session_id == self.store.state.session_id:
            return self.store.state
        return None

    async def find_submission(self, turn_id):
        return self.store.submissions.get(turn_id)

    async def commit(self, record):
        if self.store.state.revision != record.expected_revision:
            raise SessionConflict()
        self.store.records.append(record)
        self.store.submissions[record.submission.turn_id] = StoredSubmission(record.submission, record.reply)
        self.store.state = replace(
            self.store.state, revision=record.reply.revision,
            turns=(*self.store.state.turns, record.answered_turn),
            pending_question=None if record.complete else record.decision,
            pending_turn_id=record.next_turn_id, active=not record.complete,
        )
        if record.complete:
            self.store.outbox.add(record.session_id)


def examples_for(store: MemoryStore | None) -> list[str]:
    if store is None or not store.state.active:
        return []
    if store.state.pending_question.student.kind == "transfer":
        return [TRANSFER_ANSWER, TRANSFER_INCOMPLETE]
    return [MISCONCEPTION_ANSWER, SUPPORTED_ANSWER, INCOMPLETE_ANSWER]
