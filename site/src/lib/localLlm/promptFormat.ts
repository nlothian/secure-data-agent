/**
 * Per-family prompt format used by the local agent loop (`streamLocalGemma`)
 * and one-shot side tasks (`oneShot`). Everything that depends on a model's
 * chat template lives behind this interface; the loop itself is shared.
 *
 * Two representations are in play:
 *   - the model's *native* text (what is rendered into the prompt and what
 *     the model streams back): Gemma `<|turn>` / Qwen ChatML;
 *   - the app's *canonical* text (what the UI parses and chat history
 *     stores): Gemma's thought-channel markers and tool-call tokens, for
 *     every family. A Qwen conversation therefore converts stored history on
 *     the way in (`importHistory`) and maps its `<think>` markers to the
 *     canonical ones on the way out (done by the loop).
 */
import type { AgentToolSpec } from '../agentTools';
import type { LocalModelFamily } from './models';
import {
  CHANNEL_OPEN,
  STRING_DELIM,
  TOOL_CALL_OPEN,
  formatToolCallToken,
  parseStreamForToolCall,
  renderConversationForGemma,
  type InternalMessage,
} from './toolPrompt';
import {
  extractQwenPreparingToolCall,
  extractQwenStreamingParam,
  importHistoryForQwen,
  paramTypeLookupFromTools,
  parseQwenStreamForToolCall,
  qwenThinkingPrefix,
  renderConversationForQwen,
} from './qwenPrompt';
import { THINK_CLOSE, THINK_OPEN } from './qwenTokens';
import {
  GEMMA_THINKING_MARKERS,
  type SplitterMode,
  type ThinkingMarkers,
} from './thinkingChannelSplitter';

export interface ParsedToolCall {
  name: string;
  argsJson: string;
  /** The model's exact call text, when the format replays it verbatim. */
  raw?: string;
}

export interface ToolCallParseResult {
  emitText: string;
  toolCall: ParsedToolCall | null;
  rest: string;
}

export type StreamingPaneKind = 'python' | 'sql' | 'react';

export interface PromptFormat {
  family: LocalModelFamily;
  /** Native reasoning markers, for the thinking-channel splitter. */
  markers: ThinkingMarkers;
  render(
    systemPrompt: string,
    conv: InternalMessage[],
    tools: AgentToolSpec[],
    thinkingEnabled: boolean,
  ): string;
  /**
   * How generation iteration `iter` starts: the splitter's initial mode and
   * the native text the rendered prompt already pre-filled for this turn
   * (replayed into the in-progress turn so the next render is byte-exact).
   */
  turnStart(thinkingEnabled: boolean, iter: number): { mode: SplitterMode; convPrefix: string };
  parseStreamForToolCall(buffer: string): ToolCallParseResult;
  /** Native text standing in for a parsed call in the in-progress turn. */
  toolCallForConv(call: ParsedToolCall): string;
  extractPreparingToolCall(buffer: string): { name: string | null } | null;
  extractStreamingCode(buffer: string): { kind: StreamingPaneKind; source: string } | null;
  /** Stored chat history → native conversation messages. */
  importHistory(
    messages: readonly { role: 'user' | 'assistant'; content: string }[],
  ): InternalMessage[];
}

/**
 * Tools whose body has a single "main" code/sql field that we want to stream
 * into the corresponding pane's editor as it's generated.
 */
const STREAMING_FIELDS: Record<string, { kind: StreamingPaneKind; key: string }> = {
  RunPython: { kind: 'python', key: 'code' },
  RunSQL: { kind: 'sql', key: 'sql' },
  RunReact: { kind: 'react', key: 'code' },
};

// ---- Gemma 4 ------------------------------------------------------------------

/**
 * Inspect the held-back tool-call buffer and decide what the throbber should
 * advertise. Returns `null` if the buffer doesn't yet contain a complete
 * `<|tool_call>` opener, `{ name: null }` if the opener is present but the
 * `call:NAME{` prefix isn't parseable yet, and `{ name }` once it is.
 */
function extractGemmaPreparingToolCall(buffer: string): { name: string | null } | null {
  if (!buffer.startsWith(TOOL_CALL_OPEN)) return null;
  const after = buffer.slice(TOOL_CALL_OPEN.length);
  const callPrefix = 'call:';
  if (!after.startsWith(callPrefix)) return { name: null };
  const match = after.slice(callPrefix.length).match(/^([A-Za-z0-9_]+)/);
  if (!match) return { name: null };
  return { name: match[1] };
}

/**
 * Pull the partial source string for a streaming tool-call body: the
 * substring between `<key>:<|"|>` and either the closing `<|"|>` or the
 * current end of the buffer.
 */
function extractGemmaStreamingCode(
  buffer: string,
): { kind: StreamingPaneKind; source: string } | null {
  if (!buffer.startsWith(TOOL_CALL_OPEN)) return null;
  const after = buffer.slice(TOOL_CALL_OPEN.length);
  const callPrefix = 'call:';
  if (!after.startsWith(callPrefix)) return null;
  const rest = after.slice(callPrefix.length);
  const nameMatch = rest.match(/^([A-Za-z0-9_]+)\{/);
  if (!nameMatch) return null;
  const spec = STREAMING_FIELDS[nameMatch[1]];
  if (!spec) return null;
  const body = rest.slice(nameMatch[0].length);
  const opener = `${spec.key}:${STRING_DELIM}`;
  const openerIdx = body.indexOf(opener);
  if (openerIdx === -1) return null;
  const valStart = openerIdx + opener.length;
  const closeIdx = body.indexOf(STRING_DELIM, valStart);
  const source = closeIdx === -1 ? body.slice(valStart) : body.slice(valStart, closeIdx);
  return { kind: spec.kind, source };
}

const GEMMA_FORMAT: PromptFormat = {
  family: 'gemma',
  markers: GEMMA_THINKING_MARKERS,
  render: renderConversationForGemma,
  // The first iteration's prompt ends inside an open thought channel when
  // thinking is on; after a tool response Gemma continues the same turn with
  // no fresh channel open.
  turnStart: (thinkingEnabled, iter) =>
    thinkingEnabled && iter === 0
      ? { mode: 'in-thought', convPrefix: `${CHANNEL_OPEN}thought\n` }
      : { mode: 'outside', convPrefix: '' },
  parseStreamForToolCall,
  toolCallForConv: (call) => formatToolCallToken(call.name, call.argsJson),
  extractPreparingToolCall: extractGemmaPreparingToolCall,
  extractStreamingCode: extractGemmaStreamingCode,
  importHistory: (messages) =>
    messages.map((m) =>
      m.role === 'assistant'
        ? { role: 'assistant' as const, content: m.content }
        : { role: 'user' as const, content: m.content },
    ),
};

// ---- Qwen 3.5 -------------------------------------------------------------------

/** Qwen's reasoning markers. The newline after `<think>` stays thought text. */
export const QWEN_THINKING_MARKERS: ThinkingMarkers = { open: THINK_OPEN, close: THINK_CLOSE };

function qwenFormat(tools: readonly AgentToolSpec[]): PromptFormat {
  const paramType = paramTypeLookupFromTools(tools);
  return {
    family: 'qwen',
    markers: QWEN_THINKING_MARKERS,
    render: renderConversationForQwen,
    // Every Qwen generation (including after a tool response) opens a new
    // assistant turn whose prompt pre-fills `<think>\n` (thinking on) or an
    // empty `<think>` block (thinking off).
    turnStart: (thinkingEnabled) => ({
      mode: thinkingEnabled ? 'in-thought' : 'outside',
      convPrefix: qwenThinkingPrefix(thinkingEnabled),
    }),
    parseStreamForToolCall: (buffer) => parseQwenStreamForToolCall(buffer, paramType),
    toolCallForConv: (call) => call.raw ?? '',
    extractPreparingToolCall: extractQwenPreparingToolCall,
    extractStreamingCode: (buffer) => {
      const p = extractQwenStreamingParam(buffer);
      if (!p) return null;
      const spec = STREAMING_FIELDS[p.tool];
      if (!spec) return null;
      const source = p.param(spec.key);
      return source === null ? null : { kind: spec.kind, source };
    },
    importHistory: importHistoryForQwen,
  };
}

/** `tools` lets the Qwen parser keep `string` parameters verbatim. */
export function getPromptFormat(
  family: LocalModelFamily,
  tools: readonly AgentToolSpec[] = [],
): PromptFormat {
  // ZEOS Qwen speaks Qwen's template; the ZEOS machine frames it kernel-side.
  return family === 'qwen' || family === 'zeos-qwen' ? qwenFormat(tools) : GEMMA_FORMAT;
}
