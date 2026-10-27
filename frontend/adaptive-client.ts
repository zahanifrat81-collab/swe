/** Use from Bookwyrm's existing Next.js conversation view after Start.
 * Render question.text as text (React interpolation), never as HTML/Markdown.
 * Keep the existing timer, accessibility controls, and transcript UI.
 */
export type AdaptiveReply = {
  revision: number;
  /** Next pending, server-issued question ID. Null once the review is complete. */
  turn_id: string | null;
  question: {
    kind: "clarify" | "misconception_probe" | "transfer" | "coverage" | "complete";
    outcome_id: string | null;
    text: string | null;
  };
};

export class AdaptiveRequestError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "AdaptiveRequestError";
    this.status = status;
  }
}

const identifier = /^[A-Za-z0-9_.:-]{1,128}$/;
const questionKinds = new Set(["clarify", "misconception_probe", "transfer", "coverage", "complete"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Validate the API response instead of trusting a compile-time type assertion. */
export function parseAdaptiveReply(value: unknown): AdaptiveReply {
  if (!isRecord(value) || !Number.isSafeInteger(value.revision) || (value.revision as number) < 0
      || !isRecord(value.question)) {
    throw new Error("The conversation server returned an invalid reply. Keep your answer and reload.");
  }
  const question = value.question;
  if (typeof question.kind !== "string" || !questionKinds.has(question.kind)) {
    throw new Error("The conversation server returned an invalid question.");
  }
  if (question.kind === "complete") {
    if (value.turn_id !== null || question.outcome_id !== null || question.text !== null) {
      throw new Error("The conversation server returned an invalid completion reply.");
    }
  } else if (typeof value.turn_id !== "string" || !identifier.test(value.turn_id)
      || typeof question.outcome_id !== "string" || !identifier.test(question.outcome_id)
      || typeof question.text !== "string" || !question.text.trim()) {
    throw new Error("The conversation server returned an invalid next question.");
  }
  // Pick known fields explicitly; never surface private diagnostic extras.
  return {
    revision: value.revision as number,
    turn_id: value.turn_id as string | null,
    question: {
      kind: question.kind as AdaptiveReply["question"]["kind"],
      outcome_id: question.outcome_id as string | null,
      text: question.text as string | null,
    },
  };
}

export async function submitAdaptiveAnswer(
  sessionId: string,
  turnId: string,
  answer: string,
  revision: number,
  csrfToken: string,
  signal?: AbortSignal,
): Promise<AdaptiveReply> {
  if (!identifier.test(sessionId) || !identifier.test(turnId)
      || !Number.isSafeInteger(revision) || revision < 0 || !answer.trim()
      || Array.from(answer).length > 12000) {
    throw new Error("The answer or conversation details are invalid. Keep your answer and reload.");
  }
  const response = await fetch(
    `/api/conversations/${encodeURIComponent(sessionId)}/adaptive-answer`,
    {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
      body: JSON.stringify({ turn_id: turnId, answer, expected_revision: revision }),
      signal,
    },
  );
  if (!response.ok) {
    const messages: Record<number, string> = {
      401: "Your session expired. Sign in again, then reload your saved conversation.",
      403: "You cannot submit this answer. Reload your saved conversation and check your access.",
      404: "This conversation is not available. Keep your answer and return to your assignments.",
      409: "Conversation changed. Reload your saved conversation before retrying.",
      422: "The answer could not be accepted. Keep it and reload your saved conversation.",
      429: "Too many requests. Keep your answer and wait before retrying.",
    };
    throw new AdaptiveRequestError(messages[response.status]
      ?? "Your answer could not be submitted. Keep it and try again.", response.status);
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The conversation server returned an unreadable reply. Keep your answer and reload.");
  }
  const reply = parseAdaptiveReply(payload);
  if (reply.revision !== revision + 1) {
    throw new Error("Conversation changed. Reload your saved conversation before retrying.");
  }
  return reply;
}
