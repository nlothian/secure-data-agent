/**
 * Stored chat history → the turns a fresh ZEOS chat run replays
 * (`ChatRun.import_history`), with the ring each arrives on:
 *
 * - a user message → `chat.user`, TRUSTED (2);
 * - a tool result → `tools.results`, EXTERNAL (3), or `tools.results.trusted`
 *   (2) when `ZEOS_TRUSTED_RESULTS` names the call (a bundled skill card) and
 *   the turn did not record it at ring 3 (`ChatTrust.toolRings`);
 * - past assistant text → `chat.history.trusted` (2) when the turn recorded
 *   that it was written at integrity 2, else `chat.history` (3). A turn with
 *   no record (another model wrote it, or an older build) is untrusted.
 *
 * Assistant content is the canonical Gemma wire format; `importHistoryForQwen`
 * converts it, splitting a turn at each tool call as the Qwen template does.
 */
import { escapeForQwenPrompt, importHistoryForQwen } from '../localLlm/qwenPrompt';
import type { StreamChatMessage } from '../streamChat';
import { isTrustedToolResult } from './zeosToolClasses';

export const TRUSTED = 2;
export const EXTERNAL = 3;

export interface ZeosImportTurn {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  /** Assistant turns: the integrity it is replayed at (2 trusted, 3 untrusted). */
  integrity?: number;
  /** Tool turns: the tool whose result this is. */
  toolName?: string;
  /** Tool turns: replayed on `tools.results.trusted` (ring 2). */
  trusted?: boolean;
}

/** The ring a turn arrives on. */
export function ringOfImportTurn(turn: ZeosImportTurn): number {
  if (turn.role === 'user') return TRUSTED;
  if (turn.role === 'tool') return turn.trusted ? TRUSTED : EXTERNAL;
  return (turn.integrity ?? EXTERNAL) <= TRUSTED ? TRUSTED : EXTERNAL;
}

/** Tool results are delivered with Qwen's structural tags defanged, live and replayed alike. */
export function toolResultForZeos(resultStr: string): string {
  return resultStr.trim() ? escapeForQwenPrompt(resultStr) : '(empty result)';
}

/**
 * User messages too, live and replayed alike. The model worker tokenizes
 * content without special tokens, so a typed `<|im_start|>` is never the real
 * marker; defanging it as well means a message pasted from somewhere else
 * cannot look like a turn boundary or a tool response to the model either.
 */
export function userTextForZeos(text: string): string {
  return escapeForQwenPrompt(text);
}

function argsOf(json: string | undefined): Record<string, unknown> | undefined {
  if (!json) return {};
  try {
    const v: unknown = JSON.parse(json);
    return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined;
  } catch {
    return undefined;
  }
}

export function buildZeosImport(
  messages: readonly Pick<StreamChatMessage, 'role' | 'content' | 'trust'>[],
): ZeosImportTurn[] {
  const out: ZeosImportTurn[] = [];
  for (const m of messages) {
    if (m.role === 'system') continue;
    if (m.role === 'user') {
      if (m.content.trim()) out.push({ role: 'user', text: userTextForZeos(m.content) });
      continue;
    }
    // The ZEOS integrity is only ever 2 or 3; anything better is still 2.
    const integrity = m.trust ? Math.max(TRUSTED, m.trust.integrity) : EXTERNAL;
    let toolIndex = 0;
    for (const seg of importHistoryForQwen([{ role: 'assistant', content: m.content }])) {
      if (seg.role === 'tool') {
        const recorded = m.trust?.toolRings?.[toolIndex++];
        const trusted =
          seg.toolName !== undefined &&
          isTrustedToolResult(seg.toolName, argsOf(seg.toolArgsJson)) &&
          recorded !== EXTERNAL;
        out.push({
          role: 'tool',
          text: toolResultForZeos(seg.content),
          toolName: seg.toolName,
          ...(trusted ? { trusted: true } : {}),
        });
      } else if (seg.content.trim()) {
        out.push({ role: 'assistant', text: seg.content, integrity });
      }
    }
  }
  // A replay begins with the user's words.
  const first = out.findIndex((t) => t.role === 'user');
  return first === -1 ? [] : out.slice(first);
}
