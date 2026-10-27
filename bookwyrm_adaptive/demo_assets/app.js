/* The demo uses the existing adaptive engine via its local FastAPI adapter.
 * All dynamic content is plain text. Do not render model or student text as HTML.
 */
const $ = (id) => document.getElementById(id);
const elements = {
  error: $("error-box"),
  activity: $("activity"),
  welcome: $("welcome"),
  transcript: $("transcript"),
  form: $("answer-form"),
  answer: $("answer-input"),
  send: $("send-button"),
  start: $("start-button"),
  ready: $("ready-checkbox"),
  reset: $("reset-button"),
  examples: $("examples"),
  examplesPanel: $("examples-panel"),
  completion: $("completion"),
};

let session = null;
let busy = false;
// Keep the exact body after ambiguous failures. Retrying must not change the
// server-issued turn ID, revision or answer if the first request committed.
let pendingSubmission = null;

class RequestError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function validSession(value) {
  if (
    !value ||
    typeof value !== "object" ||
    !["scripted", "live"].includes(value.mode) ||
    typeof value.csrf_token !== "string" ||
    !value.csrf_token ||
    typeof value.started !== "boolean" ||
    typeof value.complete !== "boolean" ||
    !value.passage ||
    typeof value.passage.title !== "string" ||
    typeof value.passage.text !== "string" ||
    !value.outcome ||
    typeof value.outcome.label !== "string" ||
    typeof value.outcome.level !== "string" ||
    !Array.isArray(value.transcript) ||
    !Array.isArray(value.examples) ||
    !value.examples.every((text) => typeof text === "string") ||
    !value.transcript.every(
      (turn) =>
        turn &&
        ["assistant", "student"].includes(turn.role) &&
        typeof turn.text === "string",
    )
  ) {
    throw new Error(
      "The demo returned an unreadable conversation. Refresh the page to reconnect.",
    );
  }
  if (
    value.started &&
    (!value.reply ||
      !Number.isSafeInteger(value.reply.revision) ||
      value.reply.revision < 0 ||
      !value.reply.question ||
      (value.complete
        ? value.reply.question.kind !== "complete"
        : typeof value.reply.turn_id !== "string" ||
          typeof value.reply.question.text !== "string"))
  ) {
    throw new Error(
      "The demo returned an incomplete conversation. Keep your answer and refresh the page.",
    );
  }
  return value;
}

async function request(path, { method = "GET", body } = {}) {
  const headers = {};
  if (method === "POST") headers["X-Demo-CSRF"] = session?.csrf_token ?? "";
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, {
    method,
    headers,
    credentials: "same-origin",
    cache: "no-store",
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok) {
    const messages = {
      401: "Your demo session expired. Keep a copy of your answer, then refresh the page to reconnect.",
      403: "Your demo session changed. Refresh the page to reconnect; keep a copy of your answer first.",
      404: "The demo session is no longer available. Start over to try a new conversation.",
      409: "The saved conversation changed. Your typed answer has been preserved; check the latest question before sending.",
      422: "That response could not be accepted. Use between 1 and 12,000 characters.",
      503: "The demo is taking longer than expected. Retry the same answer in a moment.",
    };
    throw new RequestError(
      messages[response.status] ??
        "The request could not finish. Your answer is preserved; try again.",
      response.status,
    );
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(
      "The demo returned an unreadable reply. Your answer is preserved; try again.",
    );
  }
  return validSession(payload);
}

function showError(message = "") {
  elements.error.textContent = message;
  elements.error.hidden = !message;
}

function updateControls() {
  const active = session?.started && !session.complete;
  elements.start.disabled = busy || !session || !elements.ready.checked;
  elements.ready.disabled = busy || !session;
  elements.reset.disabled = busy || !session;
  elements.answer.readOnly = busy || pendingSubmission !== null;
  elements.send.disabled =
    busy || !active || (!pendingSubmission && !elements.answer.value.trim());
  elements.send.textContent = busy
    ? "Thinking…"
    : pendingSubmission
      ? "Retry response ↗"
      : "Send response ↗";
  for (const button of elements.examples.querySelectorAll("button")) {
    button.disabled = busy || pendingSubmission !== null;
  }
  elements.form.setAttribute("aria-busy", String(busy));
}

function setBusy(value, message = "") {
  busy = value;
  elements.activity.textContent = message;
  updateControls();
}

function createMessage(turn) {
  const container = document.createElement("article");
  container.className = `message ${turn.role}`;
  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = turn.role === "assistant" ? "b" : "You";
  const content = document.createElement("div");
  content.className = "message-content";
  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = turn.role === "assistant" ? "Bookwyrm" : "Your response";
  const text = document.createElement("p");
  text.className = "message-text";
  text.textContent = turn.text;
  if (turn.role === "assistant") text.tabIndex = -1;
  content.append(label, text);
  container.append(avatar, content);
  return container;
}

function render(nextSession, { focus = false } = {}) {
  session = nextSession;
  $("reading-title").textContent = session.passage.title;
  $("passage-text").textContent = session.passage.text;
  $("outcome-label").textContent = session.outcome.label;
  $("outcome-level").textContent = session.outcome.level;
  $("mode-title").textContent =
    session.mode === "scripted"
      ? "Scripted demo · no live AI"
      : "Live model · reading-review demo";
  $("mode-description").textContent =
    session.mode === "scripted"
      ? "Example answers demonstrate the adaptive branches. Other text asks for clarification; it is not semantically assessed. No grades are issued."
      : "Your answers are sent to the configured AI provider for analysis. Its interpretation may need teacher review. This demo does not issue grades.";
  if (typeof session.notice === "string" && session.notice)
    $("storage-notice").textContent = session.notice;

  const stage = session.complete ? 2 : session.started ? 1 : 0;
  [$("step-read"), $("step-review"), $("step-complete")].forEach(
    (step, index) => {
      step.classList.toggle("current", index === stage);
      step.classList.toggle("done", index < stage);
      if (index === stage) step.setAttribute("aria-current", "step");
      else step.removeAttribute("aria-current");
    },
  );
  elements.welcome.hidden = session.started;
  elements.transcript.hidden = !session.started;
  elements.form.hidden = !session.started || session.complete;
  elements.completion.hidden = !session.complete;

  // Render an authoritative server transcript. Plain text is important even for
  // seemingly harmless fixtures: live model and student content is untrusted.
  elements.transcript.replaceChildren(...session.transcript.map(createMessage));
  elements.transcript.scrollTop = elements.transcript.scrollHeight;
  elements.examples.replaceChildren();
  session.examples.forEach((answer) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "example-button";
    button.textContent = `Use example: ${answer}`;
    button.addEventListener("click", () => {
      if (busy || pendingSubmission) return;
      elements.answer.value = answer;
      elements.answer.focus();
      updateControls();
    });
    elements.examples.append(button);
  });
  elements.examplesPanel.hidden = session.examples.length === 0;
  updateControls();
  if (focus) {
    if (session.complete) elements.completion.focus();
    else
      elements.transcript
        .querySelector(".message.assistant:last-child .message-text")
        ?.focus();
  }
}

async function startConversation() {
  if (busy || !elements.ready.checked) return;
  showError();
  setBusy(true, "Opening your reading conversation…");
  try {
    const next = await request("/api/demo/start", { method: "POST" });
    pendingSubmission = null;
    elements.answer.value = "";
    render(next, { focus: true });
  } catch (error) {
    showError(error.message || "The conversation could not start. Try again.");
  } finally {
    setBusy(false);
  }
}

async function resetConversation() {
  if (busy || !session) return;
  showError();
  setBusy(true, "Preparing a new conversation…");
  try {
    const next = await request("/api/demo/reset", { method: "POST" });
    pendingSubmission = null;
    elements.answer.value = "";
    elements.ready.checked = false;
    render(next);
    $("conversation-title").focus();
  } catch (error) {
    showError(
      error.message ||
        "The conversation could not reset. Your current answer is preserved.",
    );
  } finally {
    setBusy(false);
  }
}

async function submitAnswer(event) {
  event.preventDefault();
  if (busy || !session?.started || session.complete) return;
  const answer = elements.answer.value;
  if (
    !pendingSubmission &&
    (!answer.trim() || Array.from(answer).length > 12000)
  ) {
    showError("Write a response between 1 and 12,000 characters.");
    return;
  }
  if (!pendingSubmission)
    pendingSubmission = {
      turn_id: session.reply.turn_id,
      answer,
      expected_revision: session.reply.revision,
    };
  showError();
  setBusy(
    true,
    session.mode === "scripted"
      ? "Following your response…"
      : "Considering your response…",
  );
  try {
    const next = await request("/api/demo/answer", {
      method: "POST",
      body: pendingSubmission,
    });
    pendingSubmission = null;
    elements.answer.value = "";
    render(next, { focus: true });
  } catch (error) {
    if (error.status === 409) {
      try {
        render(await request("/api/demo/session"));
        pendingSubmission = null;
      } catch {
        /* Preserve the original submission if reconciliation fails. */
      }
    } else if (error.status === 422) {
      // Validation did not commit. Allow the user to correct the rejected input.
      pendingSubmission = null;
    }
    showError(
      error.message ||
        "The connection was interrupted. Your answer is preserved; retry the same response.",
    );
  } finally {
    setBusy(false);
  }
}

elements.ready.addEventListener("change", updateControls);
elements.answer.addEventListener("input", updateControls);
elements.start.addEventListener("click", startConversation);
elements.reset.addEventListener("click", resetConversation);
elements.form.addEventListener("submit", submitAnswer);
elements.answer.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    if (!elements.send.disabled) elements.form.requestSubmit();
  }
});

async function initialize() {
  setBusy(true, "Connecting to your local demo…");
  try {
    render(await request("/api/demo/session"));
  } catch (error) {
    $("mode-title").textContent = "Demo connection unavailable";
    $("mode-description").textContent =
      "Check that the demo server is running, then refresh this page.";
    showError(
      error.message ||
        "The demo could not load. Refresh the page to try again.",
    );
  } finally {
    setBusy(false);
  }
}

initialize();
