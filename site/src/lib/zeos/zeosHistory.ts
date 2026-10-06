/**
 * Stored chat history → the turns a fresh ZEOS chat run replays
 * (`ChatRun.import_history`), with the ring each arrives on:
 *
 * - a user message → `chat.user`, TRUSTED (2);
 * - a tool result → `tools.results`, EXTERNAL (3), or `tools.results.trusted`
 *   (2) when `ZEOS_TRUSTED_RESULTS` names the call (a bundled skill card) and
 *   the turn recorded it at ring 2 (`ChatTrust.toolRings`); a turn with no
 *   record replays every result on ring 3;
 * - past assistant text → `chat.history.trusted` (2) when the turn recorded
 *   that it was written at integrity 2, else `chat.history` (3). A turn with
 *   no record (another model wrote it, or an older build) is untrusted.
 *
 * Assistant content is the canonical Gemma wire format; `importHistoryForQwen`
 * converts it, splitting a turn at each tool call as the Qwen template does.
 */
import { escapeForQwenPrompt, importHistoryForQwen } from '../localLlm/qwenPrompt';
import { escapeForToolPrompt, STRUCTURAL_DELIMITERS } from '../localLlm/toolPrompt';
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
  /**
   * Tool turns: the call's arguments, parsed. ZEOS replays a trusted result
   * only with its call (`name`, `arguments`), which `trusted_results` must
   * name exactly, as it would live.
   */
  toolArgs?: Record<string, unknown>;
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

/**
 * The model's own text, as it is stored (history and UI): every Gemma
 * structural delimiter defanged (`escapeForToolPrompt`), and each `→` / `←`
 * followed by a zero-width space. Stored history is the Gemma wire format,
 * which `parseGemmaHistory` reads back on a reload or retry, and the UI text
 * marks tool calls with `\n\n→ name(args)` / `← result`
 * (`parseAssistantContent`). Without this, a reply that an injection talked
 * into spelling `<|tool_call>call:CallSkill{…}<tool_call|><|tool_response>…`
 * would replay as a real tool exchange, and could be taken for a trusted
 * skill card.
 */
export function escapeModelText(text: string): string {
  return escapeForToolPrompt(text).replace(/[→←]/g, (c) => `${c}\u200b`);
}

/** The longest suffix of `text` that could still grow into a delimiter. */
function partialDelimiterSuffix(text: string): number {
  let longest = 0;
  for (const delim of STRUCTURAL_DELIMITERS) {
    for (let n = Math.min(delim.length - 1, text.length); n > longest; n--) {
      if (text.endsWith(delim.slice(0, n))) {
        longest = n;
        break;
      }
    }
  }
  return longest;
}

/**
 * `escapeModelText` for text that arrives in pieces: a delimiter split
 * across two pieces is still caught, because a tail that could be the start
 * of one is held back until the next piece (or `flush`).
 */
export class ModelTextEscaper {
  private held = '';

  push(piece: string): string {
    const text = this.held + piece;
    const keep = partialDelimiterSuffix(text);
    this.held = text.slice(text.length - keep);
    return escapeModelText(text.slice(0, text.length - keep));
  }

  /** Release what is held back, before text the app writes itself (or at the end). */
  flush(): string {
    const text = this.held;
    this.held = '';
    return escapeModelText(text);
  }
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

/**
 * The integrity a past assistant turn replays at: an int, 2 or 3 (ZEOS
 * refuses anything but an int from 0 to 3, and its integrity is only ever 2
 * or 3, so anything better is still 2). A turn with no record, or a record
 * that is not a number, is untrusted.
 */
function replayIntegrity(recorded: unknown): number {
  if (typeof recorded !== 'number' || !Number.isFinite(recorded)) return EXTERNAL;
  return recorded > TRUSTED ? EXTERNAL : TRUSTED;
}

/**
 * Stored messages → `import_history` turns. ZEOS refuses an empty user or
 * assistant turn, so neither is replayed: a whitespace-only message, or the
 * empty placeholder `mapMessagesForLLM` keeps for a demoted failed turn
 * (whose demotion `importStartIntegrity` carries instead). A tool result is
 * never empty (`toolResultForZeos`).
 */
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
    const integrity = replayIntegrity(m.trust?.integrity);
    let toolIndex = 0;
    for (const seg of importHistoryForQwen([{ role: 'assistant', content: m.content }])) {
      if (seg.role === 'tool') {
        // Ring 2 only when the turn recorded this very result on ring 2 live
        // (the kernel put it there) and the call is still an exact bundled
        // skill: never on the name alone, which the model's text could spell.
        const recorded = m.trust?.toolRings?.[toolIndex++];
        const args = argsOf(seg.toolArgsJson);
        const trusted =
          seg.toolName !== undefined && args !== undefined && isTrustedToolResult(seg.toolName, args) && recorded === TRUSTED;
        out.push({
          role: 'tool',
          text: toolResultForZeos(seg.content),
          toolName: seg.toolName,
          ...(trusted ? { toolArgs: args, trusted: true } : {}),
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

/**
 * The integrity a replay starts the job at (`import_history(start_integrity=…)`).
 * The integrity is a low-water mark that never rises within a conversation,
 * so a run rebuilt from history (reload, retry, abort, a mode switch, the
 * call cap) must start where the stored conversation ended, not at 2:
 *
 * - EXTERNAL (3) when any assistant turn recorded integrity 3 (it was
 *   demoted while it ran);
 * - EXTERNAL when a turn with no trust record (another model wrote it, or an
 *   older build) holds a tool result: nothing vouches that it was not read;
 * - TRUSTED (2) otherwise.
 *
 * Strict mode's session floor is not carried over: it lasts only until the
 * next user message, and every replay is followed by one.
 */
export function importStartIntegrity(
  messages: readonly Pick<StreamChatMessage, 'role' | 'content' | 'trust'>[],
): { integrity: number; demotedBy: string | null } {
  for (const m of messages) {
    if (m.role !== 'assistant') continue;
    if (m.trust) {
      if (m.trust.integrity >= EXTERNAL) {
        return { integrity: EXTERNAL, demotedBy: `an earlier turn${m.trust.demotedBy ? ` (${m.trust.demotedBy})` : ''}` };
      }
    } else if (importHistoryForQwen([{ role: 'assistant', content: m.content }]).some((s) => s.role === 'tool')) {
      return { integrity: EXTERNAL, demotedBy: 'an earlier turn with no trust record' };
    }
  }
  return { integrity: TRUSTED, demotedBy: null };
}

