// Node 22.18+ can strip TypeScript types: node --test frontend/adaptive-client.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { AdaptiveRequestError, parseAdaptiveReply, submitAdaptiveAnswer } from "./adaptive-client.ts";

const validReply = () => ({
  revision: 1,
  turn_id: "next-turn",
  question: { kind: "clarify", outcome_id: "o1", text: "What evidence supports that?" },
});

test("accepts validated questions and completion; discards private extras", () => {
  assert.deepEqual(parseAdaptiveReply({ ...validReply(), analysis: { confidence: 1 } }), validReply());
  const complete = { revision: 2, turn_id: null, question: { kind: "complete", outcome_id: null, text: null } };
  assert.deepEqual(parseAdaptiveReply(complete), complete);
});

test("rejects malformed replies before exposing them to the view", () => {
  for (const value of [null, [], {}, { ...validReply(), revision: -1 },
    { ...validReply(), revision: 1.2 }, { ...validReply(), turn_id: null },
    { ...validReply(), question: { kind: "invented", outcome_id: "o1", text: "Question?" } },
    { ...validReply(), question: { kind: "clarify", outcome_id: "o1", text: " " } },
    { ...validReply(), question: { kind: "complete", outcome_id: null, text: null } }]) {
    assert.throws(() => parseAdaptiveReply(value));
  }
});

test("sends only the answer, pending turn and revision, preserving CSRF and credentials", async () => {
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    assert.equal(url, "/api/conversations/session-1/adaptive-answer");
    assert.equal(options.credentials, "same-origin");
    assert.equal(options.headers["X-CSRF-Token"], "csrf");
    assert.deepEqual(JSON.parse(options.body), { turn_id: "pending-turn", answer: "My answer", expected_revision: 0 });
    return Response.json(validReply());
  };
  try {
    assert.deepEqual(await submitAdaptiveAnswer("session-1", "pending-turn", "My answer", 0, "csrf"), validReply());
  } finally { globalThis.fetch = previousFetch; }
});

test("distinguishes expired authentication, stale turns and rate limiting", async () => {
  const previousFetch = globalThis.fetch;
  try {
    for (const status of [401, 403, 404, 409, 422, 429, 503]) {
      globalThis.fetch = async () => new Response(null, { status });
      await assert.rejects(
        submitAdaptiveAnswer("session-1", "pending-turn", "My answer", 0, "csrf"),
        (error) => error instanceof AdaptiveRequestError && error.status === status,
      );
    }
  } finally { globalThis.fetch = previousFetch; }
});

test("rejects unexpected revisions and unreadable replies", async () => {
  const previousFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => Response.json({ ...validReply(), revision: 5 });
    await assert.rejects(submitAdaptiveAnswer("s1", "t1", "My answer", 0, "csrf"), /Conversation changed/);
    globalThis.fetch = async () => new Response("not json");
    await assert.rejects(submitAdaptiveAnswer("s1", "t1", "My answer", 0, "csrf"), /unreadable reply/);
  } finally { globalThis.fetch = previousFetch; }
});

test("invalid local input does not invoke fetch", async () => {
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => { assert.fail("must not call fetch"); };
  try {
    await assert.rejects(submitAdaptiveAnswer("s1", "t1", " ", 0, "csrf"));
    await assert.rejects(submitAdaptiveAnswer("s1", "t1", "answer", -1, "csrf"));
    await assert.rejects(submitAdaptiveAnswer("bad/path", "t1", "answer", 0, "csrf"));
  } finally { globalThis.fetch = previousFetch; }
});
