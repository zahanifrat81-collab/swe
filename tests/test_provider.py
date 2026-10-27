import asyncio
import json
import unittest

import httpx

from bookwyrm_adaptive.engine import AnalysisUnavailable
from bookwyrm_adaptive.models import Analysis, Context
from bookwyrm_adaptive.provider import OpenAIAnalyzer


def fixture():
    return Context(
        session_id="s1", assignment_id="a1", mode="summative",
        outcomes=[dict(id="o1", version="v1", description="Analyze the text", level="analyze",
                       passage_ids=["p1"], clarification_question="What evidence supports your interpretation?")],
        passages=[dict(id="p1", text="The mayor praised the shelter while closing its doors.")],
        turns=[dict(id="t1", outcome_id="o1", question_kind="coverage", question="What does this mean?",
                    answer="The praise seems ironic.")],
    )


def result_body():
    analysis = Analysis(status="supported", confidence=0.9, student_quote="seems ironic",
                        misconception_id=None, passage_ids=["p1"])
    return {"choices": [{"finish_reason": "stop", "message": {"content": analysis.model_dump_json()}}]}


class ProviderBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_temperature_omitted_for_reasoning_models(self):
        def handle(request):
            payload = json.loads(request.content)
            self.assertNotIn("temperature", payload)
            return httpx.Response(200, json=result_body())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            ctx = fixture()
            result = await OpenAIAnalyzer(client, "fixture-key", "configured-model").analyze(ctx, ctx.outcomes[0])
            self.assertEqual(result.status, "supported")

    async def test_temperature_can_be_opted_in(self):
        def handle(request):
            self.assertEqual(json.loads(request.content)["temperature"], 0)
            return httpx.Response(200, json=result_body())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            ctx = fixture()
            await OpenAIAnalyzer(client, "fixture-key", "configured-model", temperature=0).analyze(ctx, ctx.outcomes[0])

    async def test_total_timeout_interrupts_slow_provider(self):
        interrupted = asyncio.Event()

        async def slow(_):
            try:
                await asyncio.Event().wait()
            finally:
                interrupted.set()
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
            ctx = fixture()
            analyzer = OpenAIAnalyzer(client, "fixture-key", "model", total_timeout_seconds=0.01)
            with self.assertRaises(AnalysisUnavailable):
                await analyzer.analyze(ctx, ctx.outcomes[0])
            self.assertTrue(interrupted.is_set())

    async def test_caller_cancellation_is_not_hidden(self):
        started = asyncio.Event()

        async def slow(_):
            started.set()
            await asyncio.Event().wait()
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
            ctx = fixture()
            task = asyncio.create_task(OpenAIAnalyzer(client, "fixture-key", "model").analyze(ctx, ctx.outcomes[0]))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_oversized_input_does_not_send_partial_evidence(self):
        calls = []
        def handle(request):
            calls.append(request)
            return httpx.Response(200, json=result_body())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            ctx = fixture()
            with self.assertRaises(AnalysisUnavailable):
                await OpenAIAnalyzer(client, "fixture-key", "model", max_input_chars=10).analyze(ctx, ctx.outcomes[0])
            self.assertEqual(calls, [])

    async def test_malformed_envelopes_and_incomplete_output(self):
        bodies = [[], None, {"choices": None}, {"choices": [None]},
                  {"choices": [{"message": []}]}, {"choices": [{"message": {"content": None}}]},
                  {"choices": [{"finish_reason": "length", "message": result_body()["choices"][0]["message"]}]},
                  {"choices": [{"message": {"refusal": "declined"}}]},
                  {"choices": [{"message": {"content": "not json"}}]}]
        for body in bodies:
            with self.subTest(body=body):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as client:
                    ctx = fixture()
                    with self.assertRaises(AnalysisUnavailable):
                        await OpenAIAnalyzer(client, "fixture-key", "model").analyze(ctx, ctx.outcomes[0])

    async def test_http_failures_suppress_sensitive_exception_chain(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(401))) as client:
            ctx = fixture()
            with self.assertRaises(AnalysisUnavailable) as caught:
                await OpenAIAnalyzer(client, "secret-fixture", "model").analyze(ctx, ctx.outcomes[0])
            self.assertTrue(caught.exception.__suppress_context__)
            self.assertNotIn("secret-fixture", str(caught.exception))

    async def test_invalid_provider_configuration_fails_early(self):
        async with httpx.AsyncClient() as client:
            for options in [dict(total_timeout_seconds=0), dict(total_timeout_seconds=float("inf")),
                            dict(max_input_chars=0), dict(temperature=5)]:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    OpenAIAnalyzer(client, "fixture-key", "model", **options)


if __name__ == "__main__":
    unittest.main()
