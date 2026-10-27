"""Local interactive demo: python -m bookwyrm_adaptive.demo

This is an untimed, memory-only practice harness for the contribution, not the
main Bookwyrm application. Default scripted mode makes no model/network calls.
"""

import argparse
import asyncio
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .api import SessionConflict, SessionUnavailable, StudentReply, SubmitAnswer
from .demo_support import PASSAGE, MemoryStore, ScriptedAnalyzer, examples_for
from .engine import AdaptiveEngine, AnalysisUnavailable
from .host import SessionProcessingUnavailable, TransactionalConversationHost
from .models import StudentQuestion
from .provider import OpenAIAnalyzer

ASSETS = Path(__file__).with_name("demo_assets")
COOKIE = "bookwyrm_demo"
SESSION_TTL = 3600


@dataclass
class BrowserSession:
    principal: str
    csrf_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    touched: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    store: MemoryStore | None = None


class BrowserRegistry:
    def __init__(self, max_sessions):
        self.max_sessions = max_sessions
        self.sessions: dict[str, BrowserSession] = {}

    def find(self, token):
        now = time.monotonic()
        for key in [key for key, session in self.sessions.items() if now - session.touched > SESSION_TTL]:
            del self.sessions[key]
        session = self.sessions.get(token)
        if session is not None:
            session.touched = now
        return session

    def create(self):
        if len(self.sessions) >= self.max_sessions:
            raise HTTPException(503, "The local demo is at capacity. Restart it to clear demo sessions.")
        session = BrowserSession(principal=secrets.token_urlsafe(32))
        self.sessions[session.principal] = session
        return session


def _payload(browser, mode):
    store = browser.store
    transcript = []
    reply = None
    complete = False
    if store is not None:
        for turn in store.state.turns:
            transcript.extend([
                {"role": "assistant", "text": turn.question},
                {"role": "student", "text": turn.answer},
            ])
        complete = not store.state.active
        if complete:
            reply = StudentReply(revision=store.state.revision, question=StudentQuestion(kind="complete"))
        else:
            pending = store.state.pending_question.student
            transcript.append({"role": "assistant", "text": pending.text})
            reply = StudentReply(revision=store.state.revision, question=pending,
                                 turn_id=store.state.pending_turn_id)
    notice = (
        "Scripted practice: example answers demonstrate the branches. Other text requests clarification. No live AI or grades."
        if mode == "scripted" else
        "Live-model practice: answers are sent to the configured model provider. No grades are assigned."
    )
    if complete:
        notice = "Practice complete. No grade was assigned. In the main project, the evaluator and teacher review follow this step."
    elif mode == "live" and store is not None and store.records and store.records[-1].decision.analysis is None:
        notice = "The model could not analyze the last answer. The demo asked a clarification question; no grade was assigned."
    return {
        "mode": mode, "started": store is not None, "csrf_token": browser.csrf_token,
        "session_id": store.state.session_id if store else None,
        "passage": {"title": "The mayor's welcome", "text": PASSAGE.text},
        # Topic only; the hidden opening question and assessment wording are not
        # returned before Start. This is not the main project's reader portal.
        "outcome": {"label": "Irony and perspective", "level": "Analyze"},
        "transcript": transcript, "reply": reply.model_dump(mode="json") if reply else None,
        "examples": examples_for(store) if mode == "scripted" else [],
        "complete": complete, "notice": notice,
    }


class _RecordedAnalyzer:
    def __init__(self, result):
        self.result = result

    async def analyze(self, *_):
        if isinstance(self.result, AnalysisUnavailable):
            raise self.result
        return self.result


async def _demo_decision(analyzer, context):
    """One analysis per answer; demonstration coverage is explicitly not a grade."""
    outcome = next(o for o in context.outcomes if o.id == context.turns[-1].outcome_id)
    try:
        result = await analyzer.analyze(context, outcome)
    except AnalysisUnavailable as exc:
        result = exc
    # This miniature coverage stand-in only accepts a supported transfer with
    # validated provenance. The main project supplies its own coverage service.
    if not isinstance(result, AnalysisUnavailable):
        valid_transfer = (
            context.turns[-1].question_kind == "transfer" and result.status == "supported"
            and result.confidence >= 0.75 and result.misconception_id is None
            and bool(result.student_quote.strip())
            and result.student_quote in context.turns[-1].answer
            and bool(result.passage_ids) and set(result.passage_ids) <= set(outcome.passage_ids)
        )
        if valid_transfer:
            context = context.model_copy(update={
                "sufficient_outcome_ids": context.sufficient_outcome_ids | {outcome.id},
            })
    return await AdaptiveEngine(_RecordedAnalyzer(result)).next_question(context)


def create_app(*, mode=None, analyzer=None, max_sessions=128):
    mode = mode or os.environ.get("BOOKWYRM_DEMO_MODE", "scripted")
    if mode not in {"scripted", "live"}:
        raise ValueError("BOOKWYRM_DEMO_MODE must be scripted or live")
    if type(max_sessions) is not int or max_sessions < 1:
        raise ValueError("max_sessions must be a positive integer")
    key = os.environ.get("BOOKWYRM_MODEL_API_KEY", "") if mode == "live" else ""
    model = os.environ.get("BOOKWYRM_MODEL", "") if mode == "live" else ""
    if mode == "live" and analyzer is None and (not key.strip() or not model.strip()):
        raise ValueError("Live mode requires BOOKWYRM_MODEL_API_KEY and BOOKWYRM_MODEL in the server environment")

    registry = BrowserRegistry(max_sessions)

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient() as client:
            app.state.analyzer = analyzer or (ScriptedAnalyzer() if mode == "scripted" else OpenAIAnalyzer(client, key, model))
            yield
        registry.sessions.clear()

    app = FastAPI(title="Bookwyrm local practice demo", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])

    @app.middleware("http")
    async def local_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        return response

    async def current_browser(request: Request):
        browser = registry.find(request.cookies.get(COOKIE))
        if browser is None:
            raise HTTPException(401, "Demo session expired. Reload the page to start again.")
        csrf = request.headers.get("X-Demo-CSRF", "")
        if not csrf.isascii() or not secrets.compare_digest(csrf, browser.csrf_token):
            raise HTTPException(403, "Reload the page before submitting this demo request.")
        origin = request.headers.get("Origin")
        if origin:
            try:
                parsed = urlsplit(origin)
            except ValueError:
                raise HTTPException(403, "Demo requests must come from the same page.") from None
            if parsed.scheme not in {"http", "https"} or parsed.netloc != request.url.netloc:
                raise HTTPException(403, "Demo requests must come from the same page.")
        return browser

    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(ASSETS / "index.html")

    @app.get("/api/demo/session")
    async def session(request: Request, response: Response):
        browser = registry.find(request.cookies.get(COOKIE))
        if browser is None:
            browser = registry.create()
        response.set_cookie(COOKIE, browser.principal, httponly=True, samesite="strict", max_age=SESSION_TTL)
        async with browser.lock:
            return _payload(browser, mode)

    @app.post("/api/demo/start")
    async def start(browser=Depends(current_browser)):
        async with browser.lock:
            if browser.store is None:
                browser.store = MemoryStore(browser.principal)
            return _payload(browser, mode)

    @app.post("/api/demo/reset")
    async def reset(browser=Depends(current_browser)):
        async with browser.lock:
            browser.store = None
            return _payload(browser, mode)

    @app.post("/api/demo/answer")
    async def answer(submission: SubmitAnswer, browser=Depends(current_browser)):
        async with browser.lock:
            store = browser.store
            if store is None:
                raise HTTPException(409, "Start the practice review before answering.")
            host = TransactionalConversationHost(store)

            async def decide(context):
                return await _demo_decision(app.state.analyzer, context)

            try:
                await host.advance(browser.principal, store.state.session_id, submission, decide)
            except SessionConflict:
                raise HTTPException(409, "The conversation changed. Reload its saved state before retrying.") from None
            except SessionUnavailable:
                raise HTTPException(409, "This practice session is no longer active. Start a new review.") from None
            except SessionProcessingUnavailable:
                raise HTTPException(503, "Analysis timed out. Retry the same answer; it has not been committed.") from None
            return _payload(browser, mode)

    app.mount("/assets", StaticFiles(directory=ASSETS), name="demo-assets")
    return app


def main():
    parser = argparse.ArgumentParser(description="Run the local Bookwyrm practice demo")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    try:
        import uvicorn
    except ImportError:
        parser.exit(1, "Install the demo server first: python -m pip install -e '.[demo]'\n")
    try:
        app = create_app()
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
    # Always loopback-bound and one worker: this is not a deployment command.
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
