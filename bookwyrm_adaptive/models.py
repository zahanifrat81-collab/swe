from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1, max_length=12000, pattern=r"\S")]
QuestionText = Annotated[str, Field(min_length=1, max_length=25000, pattern=r"\S")]
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
Level = Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]
Kind = Literal["clarify", "misconception_probe", "transfer", "coverage", "complete"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Passage(Model):
    id: Identifier
    text: Text


class Misconception(Model):
    id: Identifier
    description: Text
    # Teacher-approved question, not an explanation of the correct answer.
    probe: Text


class TransferTask(Model):
    id: Identifier
    # Both scenario and question must be teacher-approved at this outcome's level.
    scenario: Text
    question: Text


class Outcome(Model):
    id: Identifier
    version: Identifier
    description: Text
    level: Level
    passage_ids: Annotated[list[Identifier], Field(min_length=1, max_length=30)]
    clarification_question: Text
    follow_up_questions: Annotated[list[Text], Field(max_length=10)] = []
    misconceptions: Annotated[list[Misconception], Field(max_length=20)] = []
    transfer_tasks: Annotated[list[TransferTask], Field(max_length=20)] = []

    @model_validator(mode="after")
    def unique_ids(self):
        for items in (self.misconceptions, self.transfer_tasks):
            if len({item.id for item in items}) != len(items):
                raise ValueError("Duplicate catalog IDs")
        return self


class Turn(Model):
    id: Identifier
    outcome_id: Identifier
    question_kind: Kind
    question: QuestionText
    answer: Text
    misconception_id: Identifier | None = None
    transfer_id: Identifier | None = None
    # Set by the question author, not inferred from student vocabulary.
    question_level: Level | None = None

    @model_validator(mode="after")
    def question_metadata(self):
        if self.question_kind == "complete":
            raise ValueError("A completion marker is not an answered question")
        if (self.question_kind == "misconception_probe") != (self.misconception_id is not None):
            raise ValueError("Only misconception probes require misconception_id")
        if (self.question_kind == "transfer") != (self.transfer_id is not None):
            raise ValueError("Only transfer questions require transfer_id")
        return self


class Context(Model):
    """Trusted server snapshot; never accept this model as a student request body."""

    session_id: Identifier
    assignment_id: Identifier
    mode: Literal["formative", "summative"]
    outcomes: Annotated[list[Outcome], Field(min_length=1, max_length=6)]
    passages: Annotated[list[Passage], Field(min_length=1, max_length=100)]
    turns: Annotated[list[Turn], Field(min_length=1, max_length=200)]
    # Owned by the main coverage engine. "Sufficient" is not a grade.
    sufficient_outcome_ids: frozenset[Identifier] = frozenset()
    time_expired: bool = False

    @model_validator(mode="after")
    def references(self):
        outcomes = {o.id: o for o in self.outcomes}
        passages = {p.id for p in self.passages}
        if len(outcomes) != len(self.outcomes) or len(passages) != len(self.passages):
            raise ValueError("Duplicate outcome or passage IDs")
        if len({t.id for t in self.turns}) != len(self.turns):
            raise ValueError("Duplicate turn IDs")
        if not self.sufficient_outcome_ids <= outcomes.keys():
            raise ValueError("Unknown covered outcome")
        for outcome in self.outcomes:
            if not set(outcome.passage_ids) <= passages:
                raise ValueError("Outcome references missing retrieved passages")
        for turn in self.turns:
            if turn.outcome_id not in outcomes:
                raise ValueError("Turn references unknown outcome")
            outcome = outcomes[turn.outcome_id]
            if turn.misconception_id and turn.misconception_id not in {m.id for m in outcome.misconceptions}:
                raise ValueError("Unknown turn misconception")
            if turn.transfer_id and turn.transfer_id not in {t.id for t in outcome.transfer_tasks}:
                raise ValueError("Unknown turn transfer task")
        return self


class Analysis(Model):
    status: Literal["supported", "incomplete", "possible_misconception", "uncertain"]
    confidence: float = Field(ge=0, le=1)
    # Exact, short substring of the latest answer, validated again by the engine.
    student_quote: Annotated[str, Field(max_length=240)]
    misconception_id: Identifier | None
    passage_ids: Annotated[list[Identifier], Field(min_length=1, max_length=30)]


class StudentQuestion(Model):
    kind: Kind
    outcome_id: Identifier | None = None
    text: QuestionText | None = None

    @model_validator(mode="after")
    def question_or_completion(self):
        if self.kind == "complete":
            if self.outcome_id is not None or self.text is not None:
                raise ValueError("Completion cannot carry a question")
        elif self.outcome_id is None or not self.text or not self.text.strip():
            raise ValueError("A follow-up requires an outcome and question text")
        return self


class Decision(Model):
    student: StudentQuestion
    # Store privately for the teacher/evaluator; the student API excludes these.
    analysis: Analysis | None = None
    needs_teacher_review: bool = False
    reason: str
    misconception_id: Identifier | None = None
    transfer_id: Identifier | None = None
    grounding_passage_ids: list[Identifier] = []
    analysis_turn_id: Identifier | None = None
    analysis_outcome_id: Identifier | None = None
    question_level: Level | None = None

    @model_validator(mode="after")
    def consistent_metadata(self):
        kind = self.student.kind
        if (kind == "misconception_probe") != (self.misconception_id is not None):
            raise ValueError("Only misconception probes require misconception_id")
        if (kind == "transfer") != (self.transfer_id is not None):
            raise ValueError("Only transfer questions require transfer_id")
        if kind == "complete" and self.question_level is not None:
            raise ValueError("Completion cannot carry question level metadata")
        if self.analysis is not None and (self.analysis_turn_id is None or self.analysis_outcome_id is None):
            raise ValueError("Private analysis must identify its source turn and outcome")
        return self
