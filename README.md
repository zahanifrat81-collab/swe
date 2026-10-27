# Bookwyrm adaptive conversation contribution

Misconception-focused follow-ups and learning-transfer checks for Bookwyrm's
FastAPI conversation engine, with a Next.js client helper. This implements Ifrat's
proposed extensions in **CEN 5011 (4).pdf**, following the main flow and US-22/23/24
in **Introduction.pdf**. It does not implement the other team members' portals,
reader, authentication, grading engine, or gradebook.

## Open the interactive browser demo

### Run from VS Code

Open the repository's **`swe` folder** in VS Code (the folder containing
`pyproject.toml`). Install Microsoft's **Python** and **Python Debugger**
extensions when prompted. Complete the virtual-environment installation below
once, then use **Run → Start Debugging** and select
**Bookwyrm: Run browser demo**. The saved `.vscode/launch.json` launches the module
using this checkout's `.venv` interpreter. You can also set breakpoints in the
Python files to follow the request flow. Stop an already-running demo with
**Control + C** in its terminal before launching another server on port 8000.

### Run from the terminal

From your existing `swe` checkout on macOS/Linux:

```sh
git pull --ff-only origin main
.venv/bin/python -m pip install -e '.[demo]'
.venv/bin/python -m bookwyrm_adaptive.demo
```

Open **http://127.0.0.1:8000** in your browser. Keep the terminal running; press
**Control + C** to stop the server. If port 8000 is busy, add `--port 8001` and open
the same address with port 8001.

Read the passage, check **I've read the passage**, then click **Start conversation**.
Use the example that says the narrator approves of the mayor, then send it to see
the misconception probe. Next use the example explaining the ironic praise to
reach a new-situation transfer question. Send the library example to finish. You
can also type your own response and start over to explore different branches.

**Default mode is scripted, not live AI.** Only the exact example answers trigger
predetermined classifications; other text requests clarification. The page
identifies this mode before you begin. It calls the same adaptive engine and
transactional host as the integration module. The local harness provides one
fixed passage, cookie-isolated sessions, memory-only storage and a demonstration
coverage check. It does not assign grades, run real teacher evaluation, enforce
the main reader's checkpoints, or provide production accounts. Sessions clear
when the server restarts or expire after one hour of inactivity. Use one server
process; do not deploy this local harness as the main application.

The demo's small HTML/CSS/JavaScript interface needs no Node installation or
separate frontend server. The Next.js integration helper remains available for
your team's eventual main application.

### Optional live model

Configure `BOOKWYRM_DEMO_MODE=live`, `BOOKWYRM_MODEL_API_KEY`, and `BOOKWYRM_MODEL`
in the **server process environment** before launching the same command. Select
a model that supports structured JSON responses. Credentials stay server-side;
never add keys to the browser, Git, or chat. This command does not automatically
load a `.env` file. In live mode, your typed answers are sent to the configured AI
provider and the page discloses that before Start. No example classifications are
used. Model failure falls back to clarification. Demonstration coverage accepts
a supported transfer only with sufficient confidence and validated quote/passage
references; this is not a mastery rating or grade. Real accuracy and latency still
require your team's validation.

## Where it connects

Teacher assignment + versioned learning outcomes + approved question catalogs
→ reader/checkpoints → Ready → Start/server timer → opening question
→ student answer → **authenticated conversation host → main retrieval/guardrails
→ misconception analysis → adaptive follow-up or transfer question**
→ saved transcript → next answer → existing outcome evaluator
→ teacher approval → feedback/gradebook.

The main project's source was not present in the original repository. The code
therefore provides an explicit integration API and a concrete transactional host;
production authentication, database mappings, retrieval and grading must be
supplied by the team's main application. It does not claim an untested connection
to that application.

## Run and verify

```sh
cd /workspace/swe
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m examples.run_conversation
```

The example wires the real engine through the real host. Its analyzer and store
are explicitly scripted, in-memory fixtures; it demonstrates misconception → probe
→ corrected reasoning → new-situation transfer → evaluation handoff, without
pretending that a fixture is a live AI model or production database.

With Node 22.18+ (CI uses Node 24):

```sh
node --test frontend/adaptive-client.test.mjs
npx --yes --package typescript@5.9.3 tsc --noEmit --strict --target ES2022 --module esnext --lib ES2022,DOM frontend/adaptive-client.ts
```

`.github/workflows/ci.yml` runs backend/browser-API tests on Python 3.11/3.12, the
terminal demonstration, client tests, and strict TypeScript checking. Provider tests use HTTP fixtures;
they do not establish real API access, model accuracy, or production latency.

## Main-project integration

### 1. Wire the engine and host

```python
from bookwyrm_adaptive.api import create_router
from bookwyrm_adaptive.engine import AdaptiveEngine
from bookwyrm_adaptive.host import TransactionalConversationHost
from bookwyrm_adaptive.provider import OpenAIAnalyzer

# These objects come from the main application, not from this extension:
# app, shared_http_client, settings, main_conversation_store,
# prepare_conversation_context, current_student.
analyzer = OpenAIAnalyzer(
    shared_http_client, settings.model_api_key, settings.model_name,
)
engine = AdaptiveEngine(analyzer, max_questions_per_outcome=5)
host = TransactionalConversationHost(
    main_conversation_store, prepare_context=prepare_conversation_context,
)
app.include_router(create_router(engine, host, current_student))
```

Reuse the main application's lifespan-managed `httpx.AsyncClient`, secure model
configuration and authentication dependency. The model must support structured
JSON output. `temperature` is omitted by default for compatibility with reasoning
models; set it only for a model that accepts it. A different provider can implement
`Analyzer.analyze(context, outcome)` and raise `AnalysisUnavailable` for expected
provider failures.

### 2. Map the database transaction

Implement the small `ConversationStore` / `ConversationTransaction` interfaces in
[host.py](bookwyrm_adaptive/host.py) using your team's PostgreSQL tables:

- `transaction(session_id)` must acquire a session row lock across all workers,
  with bounded lock/statement timeouts, then commit durably on successful exit or
  roll back on errors/cancellation. An in-process lock is insufficient in production.
- `load_for_principal(principal)` returns a pinned `SessionSnapshot` only after
  checking ownership and current class enrollment. Load completed sessions too,
  so an authorized retry can retrieve its committed receipt. Include the actual
  server-issued pending question ID and its private `Decision` metadata.
- `find_submission(turn_id)` reads the existing session's receipt, if present.
- `commit(record)` atomically saves the original answer, private decision,
  next pending question/ID, incremented revision and exact submission/reply receipt.
  Check `expected_revision` and uniquely constrain `(session_id, turn_id)`.
  On completion, close the conversation and insert one idempotent evaluation
  outbox entry in the same transaction. Do not publish a grade.

The host implements the lifecycle: authorization before replay, exact duplicate
matching, session eligibility, pending-ID and revision checks, deadline checks
before and after preparation/model work, bounded execution and next-ID issuance.
The database adapter supplies real persistence and cross-worker serialization.
Tests verify this behavior with a transactional fixture, not a deployed PostgreSQL
instance. Keep lock contention and model latency in the main load test.

### 3. Prepare the latest answer's trusted context

`prepare_context(context)` is an async hook into the main AI pipeline. It receives
the latest answer and runs assignment-scoped retrieval, the existing guardrails,
privacy/redaction policy, and main coverage evaluation before this extension's
analyzer runs. Return a validated `Context` with any updated passage selection and
`sufficient_outcome_ids`. Preserve session/assignment/mode, the pinned outcomes,
and issued question/turn identities. Keep redaction provenance if the main model
uses sanitized transcript quotes. The original answer remains in the private
transaction record. The hook runs under the host's decision deadline.

Without the hook, the host uses its supplied trusted snapshot directly. That is
useful for tests and for applications which already prepare context upstream; it
does not silently implement the main project's retrieval, privacy or guardrails.

### 4. Connect the existing conversation view

Use `frontend/adaptive-client.ts` after the existing Start action. Configure the
main Next.js app's same-origin backend proxy and CSRF validation for the route.

```ts
const reply = await submitAdaptiveAnswer(
  sessionId, pendingTurnId, answer, revision, csrfToken,
);
// Replace revision and pendingTurnId with reply.revision and reply.turn_id.
// Render reply.question.text as plain React text, never HTML or Markdown.
// On "complete", use the main app's submitted-review view, not a grade screen.
```

The API is `POST /api/conversations/{session_id}/adaptive-answer`. Students send
only `{turn_id, answer, expected_revision}`. They cannot set the outcome, mode,
time, catalogs, passages, or coverage. The response contains only
`{revision, turn_id, question}`. A complete response has a null pending ID and
null question text/outcome ID. Question `kind` is UI routing metadata, not a
confirmed diagnostic label; the misconception ID, analysis, confidence, private
reason, and teacher-review flag are not returned.

Retry the **same** turn ID/body/revision after a network failure or HTTP 503. A
committed duplicate returns its original receipt without another model call. On
409 reload the saved conversation; do not overwrite a newer answer. Preserve
unsent text on all errors. Keep the existing server timer, extended-time
accommodations, accessibility controls and transcript UI.

## Adaptive behavior

- The analyzer distinguishes supported reasoning, incomplete reasoning, a
  possible catalog misconception, and uncertainty. It does not grade grammar,
  dialect, spelling or typing speed. It never awards a grade.
- The engine rejects invented quotes, unrelated passage IDs and nonexistent
  misconception IDs. Low confidence, model failures and unresolved repeated
  hypotheses request teacher review. A possible misconception is not confirmed
  as fact or immediately corrected.
- One immediate targeted probe or transfer question is allowed before switching
  to an unprobed outcome. After that extension, unprobed outcomes take priority.
  Evidence retains the analyzed turn/outcome IDs even when the next question
  targets a different outcome.
- Every follow-up includes an exact excerpt from the previous answer, including
  fallback and coverage-switch questions. Questions use teacher-approved,
  passage-grounded catalogs. Optional `follow_up_questions` allow varied prompts.
- Transfer requires a teacher-authored new situation at Apply, Analyze, Evaluate
  or Create, with one task per outcome/session. Recall-only outcomes are not
  silently made more difficult. Transfer responses return through the same loop.
- Generated questions retain the teacher-assigned cognitive level. Analyze+
  outcomes need at least one higher-order question; a main-engine sufficiency
  flag alone cannot replace it. The teacher must verify that catalog wording
  actually matches its assigned level; this is not semantically proved by metadata.
- Insufficient outcomes receive questions in order of least evidence collected.
  A finite per-outcome budget prevents indefinite clarification loops. Exhausted
  evidence or higher-order requirements close with teacher review, not mastery.
  No outcome is left unprobed unless the server deadline ends the session.
- Both modes ask open-ended questions without corrective hints. Keep formative
  hints/practice policy in the main service; summative answer-giving is forbidden.

## Limits of the validation

The runnable example and automated tests exercise the extension and integration
contracts. They do not prove production authentication, a real database mapping,
model accuracy, fairness, or resistance to all prompt injection. Run the main
project's human-rated misconception/transfer examples, bias and red-team suites,
and US-22's 30-student/5-second latency check with the actual deployed model and
application. No live model call or production database test has been claimed here.
