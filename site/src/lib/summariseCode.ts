/**
 * Summarises a python/SQL snippet for the ExplainerPanel.
 *
 * IMPORTANT: This call is intentionally separate from the agent's chat
 * conversation. It must NOT be appended to the chat history, must NOT pass
 * any tool definitions, and must NOT share `streamChat`'s `conv` array. If
 * the model saw its own summaries on later turns it could be confused into
 * acting on them, and we would also pay for those tokens on every subsequent
 * tool round-trip. The only thing we share with the chat is the endpoint URL,
 * API key, and active model from `LLMConfig`.
 *
 * For the in-browser Gemma provider we run a one-shot, tool-less Gemma turn
 * (`generatePlainTurn`) for the same isolation reason.
 */

import { callLLM } from './llm';
import { isLocalGemmaEndpoint } from '../types/llm';
import type { LLMConfig } from '../types/llm';

export type SummaryLanguage = 'python' | 'sql' | 'react' | 'subagent';

const SYSTEM_PROMPT =
  'You explain short code snippets in plain English for a non-technical reader. ' +
  'Reply with up to 3 sentences. Do not include code, markdown, or preamble.';

function buildUserPrompt(language: SummaryLanguage, code: string): string {
  if (language === 'subagent') {
    return (
      'Summarise these instructions for a sub-agent into up to 3 sentences of English. ' +
      'Describe what the sub-agent is being asked to do, not how it is phrased.\n\n' +
      code
    );
  }
  const label =
    language === 'python' ? 'Python' : language === 'sql' ? 'SQL' : 'React (TSX)';
  const fence = language === 'react' ? 'tsx' : language;
  return (
    `Summarise this ${label} code into up to 3 sentences of English. ` +
    `Describe what the code does, not how it is written.\n\n` +
    '```' +
    fence +
    '\n' +
    code +
    '\n```'
  );
}

export async function summariseCode(
  language: SummaryLanguage,
  code: string,
  config: LLMConfig,
  signal?: AbortSignal,
): Promise<string> {
  const endpoint = config.activeEndpoint;
  if (!endpoint) {
    throw new Error('No LLM endpoint selected.');
  }

  const userPrompt = buildUserPrompt(language, code);

  if (isLocalGemmaEndpoint(endpoint)) {
    return summariseWithLocalGemma(config, userPrompt, signal);
  }

  // `callLLM` issues a single, history-less request with no tools — exactly
  // what we need to keep this call out of the agent's chat context.
  const result = await callLLM(config, SYSTEM_PROMPT, userPrompt);
  if (signal?.aborted) {
    throw new DOMException('Summary aborted', 'AbortError');
  }
  return result.trim();
}

async function summariseWithLocalGemma(
  config: LLMConfig,
  userPrompt: string,
  signal?: AbortSignal,
): Promise<string> {
  const { resolveActiveLocalModelIdOrDefault } = await import('./localLlm/models');
  const { generatePlainTurn } = await import('./localLlm/oneShot');
  const { escapeForToolPrompt } = await import('./localLlm/toolPrompt');

  // Same active-model resolution as the chat path, so summaries use the
  // model the user selected.
  const modelId = resolveActiveLocalModelIdOrDefault(config);

  // Plain system + user turn — no tools available and no shared history to
  // lean on. The snippet is escaped so literal control tokens in source code
  // (`<turn|>`, `<|tool_call>`, `<bos>`, …) can't end the user turn early.
  return generatePlainTurn({
    modelId,
    system: SYSTEM_PROMPT,
    user: escapeForToolPrompt(userPrompt),
    signal,
  });
}
