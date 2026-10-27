/** Use from Bookwyrm's existing Next.js conversation view after Start.
 * Render question.text as text (React interpolation), never as HTML/Markdown.
 * Keep the existing timer, accessibility controls, and transcript UI.
 */
export type AdaptiveReply = {
  revision: number;
  question: {
    kind: "clarify" | "misconception_probe" | "transfer" | "coverage" | "complete";
    outcome_id: string | null;
    text: string | null;
  };
};

export async function submitAdaptiveAnswer(
  sessionId: string,
  turnId: string,
  answer: string,
  revision: number,
  csrfToken: string,
  signal?: AbortSignal,
): Promise<AdaptiveReply> {
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
    throw new Error(response.status === 409
      ? "Conversation changed. Reload your saved conversation before retrying."
      : "Your answer could not be submitted. Keep it and try again.");
  }
  return response.json() as Promise<AdaptiveReply>;
}
