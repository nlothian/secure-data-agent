/**
 * One-shot, tool-less local Gemma turn for side tasks (code summaries,
 * conversation compaction). Renders a proper Gemma 4 `<|turn>` prompt with no
 * tool declarations and no shared chat history, then returns only the
 * visible body text (any thought channel is stripped).
 */
import { ensureLoaded, generate } from './llmService';
import { renderConversationForGemma } from './toolPrompt';
import { createSplitterState, feedSplitter, flushSplitter } from './thinkingChannelSplitter';

export async function generatePlainTurn(args: {
  modelId: string;
  system: string;
  user: string;
  signal?: AbortSignal;
}): Promise<string> {
  const { modelId, system, user, signal } = args;
  await ensureLoaded(modelId);
  const prompt = renderConversationForGemma(system, [{ role: 'user', content: user }], [], false);
  const raw = await generate({
    prompt,
    signal,
    onToken: () => {
      // Discarded — only the final text is used.
    },
  });
  // `generate` resolves with partial text on abort; a truncated summary must
  // never be committed by a caller, so surface the abort instead.
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

  const state = createSplitterState('outside');
  const body: string[] = [];
  for (const e of [...feedSplitter(state, raw), ...flushSplitter(state)]) {
    if (e.kind === 'body') body.push(e.text);
  }
  return body.join('').trim();
}
