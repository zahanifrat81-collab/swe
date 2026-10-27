# Bookwyrm: misconception probes and adaptive learning transfer

This is Ifrat's conversation extension, not a replacement for the full application.
It follows Introduction.pdf US-22/23/24 and the second PDF's proposed extensions:
ask a targeted question about a **possible** misconception before correcting it,
then test the same concept in a teacher-approved new situation. The second PDF
describes these as proposed ideas, not previously implemented features.

## Main-project flow

Teacher assignment + versioned Bloom outcome + approved question catalogs
→ reader/checkpoints → Ready → Start/server timer → existing opening question
→ student answer → main auth/session/guardrail service → assigned-text retrieval
→ **this module's analysis and follow-up decision** → persist transcript/question
→ existing Next.js conversation view → repeat → existing outcome evaluator
→ teacher approval → feedback/gradebook.

Only the bold stage and its integration client are implemented here. The repository
was empty, so actual main-project imports, schema, authentication and retrieval
implementations were not available. `ConversationHost` is the explicit connection
contract, not a claim that a nonexistent main application has been integrated.

## Install and test

```sh
cd /workspace/swe
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover branching, transfer, unsupported model evidence, bounded misconception
probes, provider failures, server deadlines, outcome coverage, API authentication,
student response privacy and the host integration contract. Provider tests use an
HTTP transport fixture; they do not establish model accuracy or real API access.

## Connect to the main FastAPI app

```python
from bookwyrm_adaptive.api import create_router
from bookwyrm_adaptive.engine import AdaptiveEngine
from bookwyrm_adaptive.provider import OpenAIAnalyzer

# Reuse the main application's lifespan-managed httpx.AsyncClient and configured
# model credential. Do not put credentials in source control or browser code.
analyzer = OpenAIAnalyzer(shared_http_client, settings.model_api_key, settings.model_name)
engine = AdaptiveEngine(analyzer)
app.include_router(create_router(engine, conversation_service, current_student))
```

The three host objects above belong to your team's application. Implement
`ConversationHost.advance` in that application's conversation service following
the method's transaction contract. It loads trusted `Context`, invokes the supplied
`decide` callback, stores the private `Decision`, and returns only `StudentReply`.
Pin outcome/catalog versions at session start. Keep authenticated ownership,
CSRF protection, idempotency, deadlines, accommodations, attempt limits, opt-out,
retrieval filtering and PostgreSQL transactions in that service. Never expose a
route accepting `Context` directly from a student. Validate teacher catalogs when
publishing assignments; probe and clarification questions must be open-ended,
grounded, appropriate to the tagged Bloom level, and contain no answer hints.

Use `frontend/adaptive-client.ts` from the existing Next.js conversation component;
configure its existing same-origin backend proxy for `/api/conversations/*`.
Keep the turn ID unchanged when retrying the same answer. On 409 reload the saved
conversation; do not silently overwrite a newer turn. Render question text as plain
React text, announce new questions accessibly, and preserve the existing timer.
On `complete`, switch to the existing review-submitted view; do not publish grades.

## Behavior and boundaries

- Model analysis uses the latest answer, teacher outcome, recent same-outcome turns
  and assignment-scoped passages. It classifies supported/incomplete/possible
  misconception/uncertain reasoning; keyword matching is not used as diagnosis.
- Model output must reference actual passage IDs, an approved misconception ID,
  and an exact quote from the student's answer. Invalid output or low confidence
  requests teacher review and uses a teacher-authored clarification question.
- A never-probed outcome takes priority over extra diagnostic turns. The main
  engine remains responsible for evidence sufficiency and higher-order coverage.
- A possible misconception gets at most one targeted probe per outcome/catalog ID
  in this session. A repeated hypothesis requests teacher review.
- Supported reasoning can trigger one new-situation task per outcome. Scenarios
  are teacher-approved and tied to the same cognitive level rather than generated
  with an uncontrolled increase in difficulty.
- Both modes use questions without corrective hints. Summative mode never exposes
  analysis, labels, confidence, passage text, or teacher review details to students.
- No grades or mastery levels are calculated. Forward private evidence to the
  existing evaluator; teacher approval remains necessary for summative grades.
- Teacher-review flags and private evidence belong in the main database under its
  education-record retention/access rules. Do not log raw model responses or keys.

`OpenAIAnalyzer` calls the structured-output Chat Completions API using an explicitly
configured model that supports JSON schema. You can instead reuse the AI lead's
existing model by implementing the `Analyzer` protocol. Its errors must raise
`AnalysisUnavailable`. Configure egress and the model credential in the main
deployment's secure settings. No live credential was supplied or requested here.

Prompt instructions and schema validation do not prove semantic accuracy or defeat
all prompt injection. The main project's guardrails remain required. Real-model
misconception accuracy, bias, red-team checks and the US-22 30-student/5-second
latency target need validation with your team's deployed service and human-rated
examples; the included fixture-based tests do not establish those results.
