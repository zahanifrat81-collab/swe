from typing import Protocol

from .models import Analysis, Context, Decision, Kind, Outcome, StudentQuestion

HIGHER_LEVELS = frozenset({"analyze", "evaluate", "create"})
TRANSFER_LEVELS = frozenset({"apply", "analyze", "evaluate", "create"})


class Analyzer(Protocol):
    async def analyze(self, context: Context, outcome: Outcome) -> Analysis: ...


class AnalysisUnavailable(Exception):
    """Provider timeout, invalid output, or unavailable model."""


class AdaptiveEngine:
    def __init__(
        self, analyzer: Analyzer, confidence_threshold: float = 0.75,
        *, max_questions_per_outcome: int = 5,
    ):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between zero and one")
        if type(max_questions_per_outcome) is not int or not 2 <= max_questions_per_outcome <= 30:
            raise ValueError("Question budget must be between two and thirty per outcome")
        self.analyzer = analyzer
        self.threshold = confidence_threshold
        self.max_questions = max_questions_per_outcome

    async def next_question(self, context: Context) -> Decision:
        if context.time_expired:
            return Decision(student=StudentQuestion(kind="complete"), reason="server_deadline")
        latest = context.turns[-1]
        outcome = next(o for o in context.outcomes if o.id == latest.outcome_id)
        analysis = None
        review = False
        try:
            analysis = await self.analyzer.analyze(context, outcome)
            if not set(analysis.passage_ids) <= set(outcome.passage_ids):
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

        provenance = dict(
            analysis=analysis, analysis_turn_id=latest.id,
            analysis_outcome_id=outcome.id,
        )
        # Even a fallback or outcome switch must build on the previous answer.
        # This exact excerpt is rendered as plain text, never interpreted as code.
        quote = analysis.student_quote if analysis else latest.answer.strip()[:160]

        def question(kind: Kind, target: Outcome, text: str, reason: str, **kwargs) -> Decision:
            bridge = "Let's connect that reasoning to another part of the reading.\n" if target.id != outcome.id else ""
            return Decision(
                student=StudentQuestion(kind=kind, outcome_id=target.id,
                                        text=f'You said, “{quote}”\n{bridge}{text}'),
                **provenance, needs_teacher_review=review, reason=reason,
                grounding_passage_ids=target.passage_ids,
                question_level=target.level, **kwargs,
            )

        counts = {o.id: sum(t.outcome_id == o.id for t in context.turns) for o in context.outcomes}
        def clarification(target: Outcome) -> str:
            catalog = [target.clarification_question, *target.follow_up_questions]
            return min(catalog, key=lambda text: sum(
                t.outcome_id == target.id and t.question.endswith(text) for t in context.turns
            ))

        unseen = [o for o in context.outcomes if counts[o.id] == 0]
        in_budget = counts[outcome.id] < self.max_questions
        # One immediate diagnostic/transfer turn preserves the just-detected
        # opportunity. After it, visit unprobed outcomes so they cannot starve.
        allow_extension = in_budget and (
            not unseen or latest.question_kind not in {"misconception_probe", "transfer"}
        )

        if not review and analysis and analysis.status == "possible_misconception":
            misconception = next(m for m in outcome.misconceptions if m.id == analysis.misconception_id)
            already_probed = any(
                t.outcome_id == outcome.id and t.misconception_id == misconception.id
                for t in context.turns
            )
            if not already_probed and allow_extension:
                return question("misconception_probe", outcome, misconception.probe,
                                "possible_misconception_not_confirmed",
                                misconception_id=misconception.id)
            # A repeated/budget-limited hypothesis is not silently confirmed.
            review = True

        # Transfer asks for APPLICATION, not greater difficulty for recall-only
        # outcomes. A teacher must opt in by authoring a task at Apply or higher.
        if not review and analysis and analysis.status == "supported" and allow_extension:
            used = any(t.outcome_id == outcome.id and t.transfer_id for t in context.turns)
            if not used and outcome.transfer_tasks and outcome.level in TRANSFER_LEVELS:
                task = outcome.transfer_tasks[0]
                return question("transfer", outcome, f'New situation: {task.scenario}\n{task.question}',
                                "check_learning_transfer", transfer_id=task.id)

        if unseen:
            target = unseen[0]
            return question("coverage", target, clarification(target), "unprobed_outcome")

        # Higher-order coverage cannot be inferred from merely tagging an outcome
        # Analyze+. Catalog questions carry that teacher-assigned level into Turn.
        unmet_higher_order = [target for target in context.outcomes
                             if target.level in HIGHER_LEVELS and not any(
                                 t.outcome_id == target.id and t.question_level in HIGHER_LEVELS
                                 for t in context.turns)]
        for target in unmet_higher_order:
            if counts[target.id] < self.max_questions:
                return question("coverage", target, clarification(target), "higher_order_coverage")

        insufficient = [o for o in context.outcomes if o.id not in context.sufficient_outcome_ids]
        candidates = [o for o in insufficient if counts[o.id] < self.max_questions]
        if candidates:
            # Round-robin by least evidence collected, rather than repeating the
            # first unsatisfied outcome until time runs out.
            target = min(candidates, key=lambda o: counts[o.id])
            return question("clarify", target, clarification(target),
                            "gather_evidence_for_main_evaluator")

        # The main evaluator owns sufficiency. When evidence is still missing at
        # the finite question budget, stop safely and request human review.
        review = review or bool(insufficient) or bool(unmet_higher_order) or analysis is None or analysis.status != "supported"
        return Decision(
            student=StudentQuestion(kind="complete"), **provenance,
            needs_teacher_review=review,
            reason="question_budget_exhausted" if insufficient or unmet_higher_order else "main_engine_coverage_sufficient",
        )
