// Exercise the browser adapter without a network or a live model. This small DOM
// fixture records text/controls; real layout is checked in the served browser.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(
  new URL("../bookwyrm_adaptive/demo_assets/app.js", import.meta.url),
  "utf8",
);
const markup = readFileSync(
  new URL("../bookwyrm_adaptive/demo_assets/index.html", import.meta.url),
  "utf8",
);

class Element {
  constructor() {
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.value = "";
    this.textContent = "";
    this.className = "";
    this.hidden = false;
    this.checked = false;
    this.disabled = false;
    this.readOnly = false;
    this.scrollHeight = 100;
    this.classList = { toggle() {} };
  }
  set innerHTML(_) {
    throw new Error("Untrusted content must not be interpreted as HTML");
  }
  append(...children) {
    this.children.push(...children);
  }
  replaceChildren(...children) {
    this.children = children;
  }
  setAttribute(name, value) {
    this.attributes[name] = value;
  }
  removeAttribute(name) {
    delete this.attributes[name];
  }
  addEventListener(name, callback) {
    (this.listeners[name] ??= []).push(callback);
  }
  querySelectorAll() {
    return this.children;
  }
  querySelector() {
    return null;
  }
  focus() {
    this.focused = true;
  }
  async emit(name) {
    for (const listener of this.listeners[name] ?? [])
      await listener({ preventDefault() {} });
  }
}

const initial = () => ({
  mode: "scripted",
  started: false,
  csrf_token: "fixture-csrf",
  session_id: null,
  passage: {
    title: "The mayor",
    text: "The mayor praised his shelter while closing its doors.",
  },
  outcome: { label: "Analyze ironic praise", level: "Analyze" },
  transcript: [],
  reply: null,
  examples: [],
  complete: false,
  notice: "Memory-only fixture",
});
const active = (revision = 0) => ({
  ...initial(),
  started: true,
  session_id: "fixture-session",
  transcript: [{ role: "assistant", text: "What does the narrator mean?" }],
  reply: {
    revision,
    turn_id: `server-question-${revision}`,
    question: {
      kind: "coverage",
      outcome_id: "irony",
      text: "What does the narrator mean?",
    },
  },
  examples: ["The narrator approves of the mayor."],
});
const complete = (answer) => ({
  ...active(),
  complete: true,
  examples: [],
  transcript: [...active().transcript, { role: "student", text: answer }],
  reply: {
    revision: 1,
    turn_id: null,
    question: { kind: "complete", outcome_id: null, text: null },
  },
});
const response = (body, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => body,
});

async function load(fetch) {
  const elements = new Map(
    [...markup.matchAll(/id="([^"]+)"/g)].map((match) => [
      match[1],
      new Element(),
    ]),
  );
  const document = {
    getElementById: (id) => elements.get(id),
    createElement: () => new Element(),
  };
  vm.runInNewContext(source, { document, fetch, console });
  await new Promise(setImmediate);
  return (id) => elements.get(id);
}

test("reading gate and example buttons do not automatically submit an answer", async () => {
  const calls = [];
  const $ = await load(async (url, options) => {
    calls.push({ url, options });
    return response(url === "/api/demo/session" ? initial() : active());
  });
  assert.equal($("start-button").disabled, true);
  $("ready-checkbox").checked = true;
  await $("ready-checkbox").emit("change");
  assert.equal($("start-button").disabled, false);
  await $("start-button").emit("click");
  assert.equal(calls[1].url, "/api/demo/start");
  assert.equal(calls[1].options.headers["X-Demo-CSRF"], "fixture-csrf");
  assert.equal($("welcome").hidden, true);
  assert.equal($("answer-form").hidden, false);
  await $("examples").children[0].emit("click");
  assert.equal($("answer-input").value, "The narrator approves of the mayor.");
  assert.equal(calls.length, 2);
  assert.equal($("answer-input").focused, true);
});

test("an ambiguous connection failure retries the exact pending answer and revision", async () => {
  const bodies = [];
  const answer = '<img src=x onerror="alert(1)">';
  const $ = await load(async (url, options) => {
    if (url === "/api/demo/session") return response(active());
    bodies.push(JSON.parse(options.body));
    if (bodies.length === 1) throw new TypeError("Connection interrupted");
    return response(complete(answer));
  });
  $("answer-input").value = answer;
  await $("answer-form").emit("submit");
  assert.equal($("answer-input").value, answer);
  assert.equal($("answer-input").readOnly, true);
  assert.match($("send-button").textContent, /Retry/);
  // Even an external script changing the input cannot mutate the retry receipt.
  $("answer-input").value = "A changed answer";
  await $("answer-form").emit("submit");
  assert.deepEqual(bodies, [
    { turn_id: "server-question-0", answer, expected_revision: 0 },
    { turn_id: "server-question-0", answer, expected_revision: 0 },
  ]);
  assert.equal($("completion").hidden, false);
  assert.equal($("answer-form").hidden, true);
  const studentMessage = $("transcript").children[1];
  assert.equal(studentMessage.children[1].children[1].textContent, answer);
});

test("a conflict reloads the authoritative question while preserving typed text", async () => {
  let reads = 0;
  const sent = [];
  const $ = await load(async (url, options) => {
    if (url === "/api/demo/session") return response(active(reads++));
    sent.push(JSON.parse(options.body));
    return sent.length === 1
      ? response({}, 409)
      : response(complete("My answer"));
  });
  $("answer-input").value = "My answer";
  await $("answer-form").emit("submit");
  assert.equal(reads, 2);
  assert.equal($("answer-input").value, "My answer");
  assert.equal($("answer-input").readOnly, false);
  assert.match($("error-box").textContent, /preserved/);
  await $("answer-form").emit("submit");
  assert.equal(sent[1].turn_id, "server-question-1");
  assert.equal(sent[1].expected_revision, 1);
});

test("server errors keep private response details out of the page", async () => {
  const $ = await load(async (url) =>
    url === "/api/demo/session"
      ? response(active())
      : response({ detail: "private model token or traceback" }, 500),
  );
  $("answer-input").value = "My answer";
  await $("answer-form").emit("submit");
  assert.doesNotMatch($("error-box").textContent, /private model/);
  assert.equal($("answer-input").value, "My answer");
  assert.equal($("answer-input").readOnly, true);
});
