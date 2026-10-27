from typing import Protocol

from .models import Analysis, Context, Decision, Outcome, StudentQuestion


class Analyzer(Protocol):
    async def analyze(self, context: Context, outcome: Outcome) -> Analysis: ...


class AnalysisUnavailable(Exception):
    """Provider timeout, invalid output, or unavailable model."""


class AdaptiveEngine:
    def __init__(self, analyzer: Analyzer, confidence_threshold: float = 0.75):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between zero and one")
        self.analyzer = analyzer
        self.threshold = confidence_threshold

    async def next_question(self, context: Context) -> Decision:
        if context.time_expired:
            return Decision(student=StudentQuestion(kind="complete"), reason="server_deadline")
        latest = context.turns[-1]
        outcome = next(o for o in context.outcomes if o.id == latest.outcome_id)
        analysis = None
        review = False
        try:
            analysis = await self.analyzer.analyze(context, outcome)
            allowed = set(outcome.passage_ids)
            if not set(analysis.passage_ids) <= allowed:
                raise AnalysisUnavailable("Ungrounded passage reference")
            if not analysis.student_quote.strip() or analysis.student_quote not in latest.answer:
                raise AnalysisUnavailable("Fabricated evidence")
            if analysis.status == "possible_misconception":
                if analysis.misconception_id not in {m.id for m in outcome.misconceptions}:
                    raise AnalysisUnavailable("Unknown misconception")
            elif analysis.misconception_id is not None:
                raise AnalysisUnavailable("Inconsistent analysis")
            review = analysis.confidence < self.threshold or analysis.status == "uncertain"
        except AnalysisUnavailable:
            review = True
            analysis = None

        def question(kind, target, text, reason, **kwargs):
            return Decision(
                student=StudentQuestion(kind=kind, outcome_id=target.id, text=text),
                analysis=analysis, needs_teacher_review=review, reason=reason,
                grounding_passage_ids=target.passage_ids, **kwargs,
            )

        # Prefer a never-probed outcome before using optional diagnostic turns.
        seen = {t.outcome_id for t in context.turns}
        for target in context.outcomes:
            if target.id not in seen:
                return question("coverage", target, target.clarification_question, "unprobed_outcome")

        if not review and analysis and analysis.status == "possible_misconception":
            misconception = next(m for m in outcome.misconceptions if m.id == analysis.misconception_id)
            already_probed = any(
                t.outcome_id == outcome.id and t.misconception_id == misconception.id
                for t in context.turns
            )
            if not already_probed:
                return question(
                    "misconception_probe", outcome,
                    f'You said, “{analysis.student_quote}”\n{misconception.probe}',
                    "possible_misconception_not_confirmed", misconception_id=misconception.id,
                )
            review = True  # A repeated hypothesis requires human attention, not a correction loop.

        # Transfer keeps the same teacher-authored outcome and level. It is not
        # inferred from vocabulary matches or used to award a grade.
        if not review and analysis and analysis.status == "supported":
            used = {t.transfer_id for t in context.turns if t.outcome_id == outcome.id}
            if not used - {None} and outcome.transfer_tasks:
                task = outcome.transfer_tasks[0]
                return question(
                    "transfer", outcome,
                    f'You said, “{analysis.student_quote}”\n'
                    f'New situation: {task.scenario}\n{task.question}',
                    "check_learning_transfer", transfer_id=task.id,
                )

        for target in context.outcomes:
            if target.id not in context.sufficient_outcome_ids:
                prefix = ""
                if target.id == outcome.id and analysis:
                    prefix = f'You said, “{analysis.student_quote}”\n'
                return question("clarify", target, prefix + target.clarification_question,
                                "gather_evidence_for_main_evaluator")
        return Decision(student=StudentQuestion(kind="complete"), analysis=analysis,
                        needs_teacher_review=review, reason="main_engine_coverage_sufficient")
