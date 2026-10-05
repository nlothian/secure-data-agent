/**
 * One-shot, tool-less local-model turn for side tasks (code summaries,
 * conversation compaction). Renders a proper prompt in the model's own chat
 * template (`promptFormat.ts`) with no tool declarations and no shared chat
 * history, then returns only the visible body text (any thought channel is
 * stripped).
 */
import { ensureLoaded, generate } from './llmService';
import { getLocalGemmaModel } from './models';
import { getPromptFormat } from './promptFormat';
import { escapeForQwenPrompt } from './qwenPrompt';
import { createSplitterState, feedSplitter, flushSplitter } from './thinkingChannelSplitter';

export async function generatePlainTurn(args: {
  modelId: string;
  system: string;
  user: string;
  signal?: AbortSignal;
}): Promise<string> {
  const { modelId, system, user, signal } = args;
  await ensureLoaded(modelId);
  const fmt = getPromptFormat(getLocalGemmaModel(modelId)?.family ?? 'gemma');
  // Callers defang Gemma's control tokens (`escapeForToolPrompt`); a Qwen
  // prompt also needs its own ChatML / tool tags defanged.
  const userText = fmt.family === 'qwen' ? escapeForQwenPrompt(user) : user;
  const prompt = fmt.render(system, [{ role: 'user', content: userText }], [], false);
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

  const state = createSplitterState('outside', fmt.markers);
  const body: string[] = [];
  for (const e of [...feedSplitter(state, raw), ...flushSplitter(state)]) {
    if (e.kind === 'body') body.push(e.text);
  }
  return body.join('').trim();
}
