/**
 * Prompt + parser glue for Qwen 3.5's chat template (ChatML turns, `<think>`
 * reasoning, XML-style `<tool_call>` blocks).
 *
 * Mirrors the `chat_template.jinja` shipped with Qwen/Qwen3.5-* (ONNX exports
 * carry the same file). As with Gemma (`toolPrompt.ts`) we render the prompt
 * ourselves instead of calling `apply_chat_template`, so the streaming parser
 * and the worker's KV-cache reuse see exactly the bytes the model saw.
 *
 *   <|im_start|>system\n# Tools ... <tools>\n{json}\n</tools> ...<|im_end|>\n
 *   <|im_start|>user\n...<|im_end|>\n
 *   <|im_start|>assistant\n<think>\n...\n</think>\n\nbody
 *     <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call><|im_end|>\n
 *   <|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>\n
 *
 * Chat history is stored in the Gemma wire format regardless of the active
 * model (`streamLocalGemma` emits it); `importHistoryForQwen` converts it.
 */

import type { AgentToolSpec } from '../agentTools';
import { stripCompactedMarker } from '../parseAssistantContent';
import { parseGemmaHistory, type InternalMessage } from './toolPrompt';
import {
  FUNCTION_CLOSE,
  FUNCTION_OPEN,
  IM_END,
  IM_START,
  PARAMETER_CLOSE,
  PARAMETER_OPEN,
  QWEN_EMPTY_THINK,
  QWEN_TOOL_CALL_CLOSE,
  QWEN_TOOL_CALL_OPEN,
  QWEN_TOOL_RESPONSE_CLOSE,
  QWEN_TOOL_RESPONSE_OPEN,
  THINK_CLOSE,
  THINK_OPEN,
} from './qwenTokens';

// ---- escaping ---------------------------------------------------------------

const QWEN_STRUCTURAL_DELIMITERS = [
  IM_START,
  IM_END,
  THINK_OPEN,
  THINK_CLOSE,
  QWEN_TOOL_CALL_OPEN,
  QWEN_TOOL_CALL_CLOSE,
  QWEN_TOOL_RESPONSE_OPEN,
  QWEN_TOOL_RESPONSE_CLOSE,
  FUNCTION_OPEN,
  FUNCTION_CLOSE,
  PARAMETER_OPEN,
  PARAMETER_CLOSE,
  '<|endoftext|>',
] as const;

/**
 * Defang every Qwen structural delimiter in tool-result / argument data by
 * inserting a zero-width space after the leading `<` (same scheme and
 * rationale as `escapeForToolPrompt` for Gemma).
 */
export function escapeForQwenPrompt(s: string): string {
  if (!s) return s;
  let out = s;
  for (const delim of QWEN_STRUCTURAL_DELIMITERS) {
    if (out.indexOf(delim) === -1) continue;
    const escaped = `${delim[0]}​${delim.slice(1)}`;
    while (out.indexOf(delim) !== -1) {
      out = out.split(delim).join(escaped);
    }
  }
  return out;
}

// ---- JSON as jinja's `tojson` prints it --------------------------------------

/**
 * `json.dumps(value, ensure_ascii=False)` — Python's default `", "` / `": "`
 * separators, keys in insertion order. This is what the template's
 * `tool | tojson` produces for the `<tools>` block.
 */
export function pyJsonDumps(value: unknown): string {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number') return Number.isFinite(value) ? JSON.stringify(value) : 'NaN';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (Array.isArray(value)) return `[${value.map(pyJsonDumps).join(', ')}]`;
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>).filter(
      ([, v]) => v !== undefined,
    );
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}: ${pyJsonDumps(v)}`).join(', ')}}`;
  }
  return JSON.stringify(String(value));
}

// ---- tool calls ---------------------------------------------------------------

function formatParamValue(value: unknown): string {
  // The template prints strings as-is and everything else via `tojson`.
  if (typeof value === 'string') return escapeForQwenPrompt(value);
  return escapeForQwenPrompt(pyJsonDumps(value));
}

/** Render one call as the template does (without the leading separator). */
export function formatQwenToolCall(name: string, args: Record<string, unknown>): string {
  let out = `${QWEN_TOOL_CALL_OPEN}\n${FUNCTION_OPEN}${name}>\n`;
  for (const [k, v] of Object.entries(args)) {
    out += `${PARAMETER_OPEN}${k}>\n${formatParamValue(v)}\n${PARAMETER_CLOSE}\n`;
  }
  out += `${FUNCTION_CLOSE}\n${QWEN_TOOL_CALL_CLOSE}`;
  return out;
}

function argsObject(argsJson: string): Record<string, unknown> {
  try {
    const v = JSON.parse(argsJson || '{}');
    return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

/**
 * Separator the template puts before a tool call: `\n\n` after non-blank
 * content for the first call, `\n` between calls, nothing otherwise.
 */
function toolCallSeparator(contentSoFar: string, isFirstCall: boolean): string {
  if (!isFirstCall) return '\n';
  return contentSoFar.trim() ? '\n\n' : '';
}

// ---- tool declarations (system turn) -------------------------------------------

const TOOLS_PREAMBLE =
  '# Tools\n\nYou have access to the following functions:\n\n<tools>';

const TOOLS_INSTRUCTIONS =
  '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n' +
  '<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n' +
  '<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n' +
  '</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n' +
  '- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n' +
  '- Required parameters MUST be specified\n' +
  '- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n' +
  '- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n' +
  '</IMPORTANT>';

function formatToolDeclaration(spec: AgentToolSpec): string {
  return pyJsonDumps({
    type: 'function',
    function: {
      name: spec.name,
      description: spec.description,
      parameters: spec.parameters,
    },
  });
}

// ---- prompt rendering --------------------------------------------------------

/**
 * Render a full prompt ending in an open assistant turn.
 *
 * Assistant messages are written verbatim between `<|im_start|>assistant\n`
 * and `<|im_end|>`: history turns arrive already converted
 * (`importHistoryForQwen`, no reasoning — the template drops it for turns
 * before the last user message) and in-progress turns are byte-exact
 * replays of the model's own output, reasoning prefix included, so the
 * worker's KV cache can be reused across tool iterations.
 *
 * Unlike Gemma, Qwen opens a fresh assistant turn after every tool response,
 * so the thinking prefix is added to every generation prompt.
 */
export function renderConversationForQwen(
  systemPrompt: string,
  messages: InternalMessage[],
  tools: AgentToolSpec[] = [],
  thinkingEnabled: boolean = false,
): string {
  let out = '';
  const systemContent = renderQwenSystemContent(systemPrompt, tools);
  if (systemContent) out += `${IM_START}system\n${systemContent}${IM_END}\n`;

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.role === 'user') {
      out += `${IM_START}user\n${msg.content}${IM_END}\n`;
    } else if (msg.role === 'assistant') {
      out += `${IM_START}assistant\n${msg.content}${IM_END}\n`;
    } else {
      if (messages[i - 1]?.role !== 'tool') out += `${IM_START}user`;
      out += `\n${QWEN_TOOL_RESPONSE_OPEN}\n${escapeForQwenPrompt(msg.content)}\n${QWEN_TOOL_RESPONSE_CLOSE}`;
      if (messages[i + 1]?.role !== 'tool') out += `${IM_END}\n`;
    }
  }

  out += `${IM_START}assistant\n${qwenThinkingPrefix(thinkingEnabled)}`;
  return out;
}

/**
 * The text of the system turn (between `<|im_start|>system\n` and
 * `<|im_end|>`): the tool declarations and instructions, then the system
 * prompt. Empty when there is neither. The ZEOS chat uses it as its
 * descriptor body, which the chat machine frames as the system turn.
 */
export function renderQwenSystemContent(
  systemPrompt: string,
  tools: readonly AgentToolSpec[] = [],
): string {
  const system = systemPrompt.trim();
  if (tools.length === 0) return system;
  let out = TOOLS_PREAMBLE;
  for (const tool of tools) out += `\n${formatToolDeclaration(tool)}`;
  out += `\n</tools>${TOOLS_INSTRUCTIONS}`;
  if (system) out += `\n\n${system}`;
  return out;
}

/** What the generation prompt pre-fills after `<|im_start|>assistant\n`. */
export function qwenThinkingPrefix(thinkingEnabled: boolean): string {
  return thinkingEnabled ? `${THINK_OPEN}\n` : QWEN_EMPTY_THINK;
}

// ---- history import ----------------------------------------------------------

/**
 * Convert stored chat history (assistant content in the Gemma wire format,
 * see `parseGemmaHistory`) into Qwen-native messages: each tool call becomes
 * a `<tool_call>` block closing its assistant turn, followed by a `tool`
 * message with the result; text after it starts a new assistant turn.
 */
export function importHistoryForQwen(
  messages: readonly { role: 'user' | 'assistant'; content: string }[],
): InternalMessage[] {
  const out: InternalMessage[] = [];
  for (const m of messages) {
    if (m.role === 'user') {
      out.push({ role: 'user', content: m.content });
      continue;
    }
    let content = '';
    let callsInTurn = 0;
    const flush = (force: boolean): void => {
      if (content || force) out.push({ role: 'assistant', content });
      content = '';
      callsInTurn = 0;
    };
    for (const seg of parseGemmaHistory(stripCompactedMarker(m.content))) {
      if (seg.kind === 'text') {
        if (callsInTurn > 0) flush(false);
        content += seg.text;
        continue;
      }
      content +=
        toolCallSeparator(content, callsInTurn === 0) +
        formatQwenToolCall(seg.name, argsObject(seg.argsJson));
      callsInTurn++;
      if (seg.resultJson !== null) {
        flush(false);
        out.push({ role: 'tool', toolName: seg.name, content: seg.resultJson });
      }
    }
    // Keep an (empty) assistant turn for a message that had no content at
    // all, so user/assistant alternation is preserved.
    const last = out[out.length - 1];
    flush(!last || last.role === 'user');
  }
  return out;
}

// ---- streaming parser for tool calls -----------------------------------------

export interface QwenToolCallChunk {
  name: string;
  argsJson: string;
  /** The exact `<tool_call>…</tool_call>` text the model produced. */
  raw: string;
}

export interface QwenIncrementalParseResult {
  emitText: string;
  toolCall: QwenToolCallChunk | null;
  rest: string;
}

/**
 * JSON-schema type of a tool parameter, or undefined when unknown. Values of
 * `string` parameters are kept verbatim; everything else is JSON-decoded when
 * possible (the same rule as llama.cpp / vLLM's Qwen3-Coder parsers).
 */
export type ParamTypeLookup = (toolName: string, param: string) => string | undefined;

export function paramTypeLookupFromTools(tools: readonly AgentToolSpec[]): ParamTypeLookup {
  const byName = new Map(tools.map((t) => [t.name, t]));
  return (toolName, param) => {
    const params = byName.get(toolName)?.parameters as
      | { properties?: Record<string, { type?: unknown }> }
      | undefined;
    const type = params?.properties?.[param]?.type;
    return typeof type === 'string' ? type.toLowerCase() : undefined;
  };
}

function convertParamValue(raw: string, type: string | undefined): unknown {
  if (type === 'string') return raw;
  const trimmed = raw.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    return raw;
  }
}

function stripOneNewline(s: string): string {
  let v = s;
  if (v.startsWith('\n')) v = v.slice(1);
  if (v.endsWith('\n')) v = v.slice(0, -1);
  return v;
}

/** Parse the inside of a `<tool_call>…</tool_call>` block. */
export function parseQwenToolCallBody(
  inner: string,
  paramType: ParamTypeLookup = () => undefined,
): { name: string; argsJson: string } | null {
  const fnMatch = inner.match(/^\s*<function=([^>\n]+)>([\s\S]*?)<\/function>\s*$/);
  if (!fnMatch) return null;
  const name = fnMatch[1].trim();
  if (!name) return null;
  const body = fnMatch[2];
  const args: Record<string, unknown> = {};
  const re = /<parameter=([^>\n]+)>([\s\S]*?)<\/parameter>/g;
  let last = 0;
  for (let m = re.exec(body); m !== null; m = re.exec(body)) {
    // Only whitespace may sit between parameters.
    if (body.slice(last, m.index).trim()) return null;
    const key = m[1].trim();
    args[key] = convertParamValue(stripOneNewline(m[2]), paramType(name, key));
    last = re.lastIndex;
  }
  if (body.slice(last).trim()) return null;
  return { name, argsJson: JSON.stringify(args) };
}

/**
 * Stateless incremental parser (Qwen counterpart of `parseStreamForToolCall`).
 * Text before a possible `<tool_call>` is safe to show; from the opener on,
 * everything is held until `</tool_call>` arrives. A block that does not
 * parse is released as plain text.
 */
export function parseQwenStreamForToolCall(
  buffer: string,
  paramType?: ParamTypeLookup,
): QwenIncrementalParseResult {
  const openIdx = buffer.indexOf(QWEN_TOOL_CALL_OPEN);
  if (openIdx === -1) {
    const safeEmitLen = Math.max(0, buffer.length - (QWEN_TOOL_CALL_OPEN.length - 1));
    return {
      emitText: buffer.slice(0, safeEmitLen),
      toolCall: null,
      rest: buffer.slice(safeEmitLen),
    };
  }
  const before = buffer.slice(0, openIdx);
  const after = buffer.slice(openIdx + QWEN_TOOL_CALL_OPEN.length);
  const closeIdx = after.indexOf(QWEN_TOOL_CALL_CLOSE);
  if (closeIdx === -1) {
    return { emitText: before, toolCall: null, rest: buffer.slice(openIdx) };
  }
  const inner = after.slice(0, closeIdx);
  const rest = after.slice(closeIdx + QWEN_TOOL_CALL_CLOSE.length);
  const raw = QWEN_TOOL_CALL_OPEN + inner + QWEN_TOOL_CALL_CLOSE;
  const parsed = parseQwenToolCallBody(inner, paramType);
  if (!parsed) return { emitText: before + raw, toolCall: null, rest };
  return { emitText: before, toolCall: { ...parsed, raw }, rest };
}

// ---- live UI hints while a call streams --------------------------------------

/**
 * `null` while the held-back buffer has no `<tool_call>` opener yet,
 * `{ name: null }` once it does, `{ name }` once `<function=NAME>` is complete.
 */
export function extractQwenPreparingToolCall(buffer: string): { name: string | null } | null {
  if (!buffer.startsWith(QWEN_TOOL_CALL_OPEN)) return null;
  const m = buffer.slice(QWEN_TOOL_CALL_OPEN.length).match(/^\s*<function=([A-Za-z0-9_]+)>/);
  return { name: m ? m[1] : null };
}

/**
 * Partial value of the `key` parameter of tool `name` as it streams, or null
 * when the buffer is not (yet) such a call. A trailing fragment that could be
 * the start of `\n</parameter>` is held back.
 */
export function extractQwenStreamingParam(
  buffer: string,
): { tool: string; param: (key: string) => string | null } | null {
  if (!buffer.startsWith(QWEN_TOOL_CALL_OPEN)) return null;
  const after = buffer.slice(QWEN_TOOL_CALL_OPEN.length);
  const fn = after.match(/^\s*<function=([A-Za-z0-9_]+)>/);
  if (!fn) return null;
  const body = after.slice(fn[0].length);
  return {
    tool: fn[1],
    param: (key) => {
      const opener = `${PARAMETER_OPEN}${key}>`;
      const idx = body.indexOf(opener);
      if (idx === -1) return null;
      let value = body.slice(idx + opener.length);
      if (value.startsWith('\n')) value = value.slice(1);
      const closeIdx = value.indexOf(PARAMETER_CLOSE);
      if (closeIdx !== -1) {
        const v = value.slice(0, closeIdx);
        return v.endsWith('\n') ? v.slice(0, -1) : v;
      }
      const tail = `\n${PARAMETER_CLOSE}`;
      for (let n = Math.min(tail.length, value.length); n > 0; n--) {
        if (value.endsWith(tail.slice(0, n))) return value.slice(0, value.length - n);
      }
      return value;
    },
  };
}
