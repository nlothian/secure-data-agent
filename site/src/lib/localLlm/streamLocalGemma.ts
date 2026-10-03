import { runAgentTool } from '../agentTools';
import { isAbortError, type StreamChatOptions } from '../streamChat';
import { clampToolResultSize, estimateResultTokens } from '../toolResultLimits';
import { LOCAL_GEMMA_CONTEXT_WINDOW } from '../contextWindow';
import { compactConversation } from '../compactConversation';
import { COMPACTION_HEADER } from '../autoCompaction';
import * as tokenUsageStore from '../tokenUsageStore';
import type { ChatMessage } from '../../types/chat';
import { resolveActiveLocalModelIdOrDefault } from './models';
import { ensureLoaded, generate } from './llmService';
import type { GenerateStats } from './llmWorkerProtocol';
import {
  formatToolCallToken,
  formatToolResponseToken,
  parseStreamForToolCall,
  renderConversationForGemma,
  CHANNEL_OPEN,
  CHANNEL_CLOSE,
  TOOL_CALL_OPEN,
  STRING_DELIM,
  type InternalMessage,
} from './toolPrompt';
import {
  setLlmPreparingToolCall,
  setStreamingSource,
} from '../executionPanelStore';
import {
  createSplitterState,
  feedSplitter,
  flushSplitter,
  type SplitterEvent,
} from './thinkingChannelSplitter';
import { LOCAL_GEMMA_ENDPOINT, type LLMConfig } from '../../types/llm';

const MAX_TOOL_ITERATIONS = 10;
const THINKING_OPEN_MARKER = `${CHANNEL_OPEN}thought\n`;

/**
 * Inspect the held-back tool-call buffer (`parsed.rest`) and decide what the
 * throbber should advertise. Returns `null` if the buffer doesn't yet contain
 * a complete `<|tool_call>` opener (so we're still in plain-text holdback or
 * idle), `{ name: null }` if the opener is present but the `call:NAME{`
 * prefix isn't parseable yet, and `{ name }` once the name is available.
 */
function extractPreparingToolCall(
  buffer: string,
): { name: string | null } | null {
  if (!buffer.startsWith(TOOL_CALL_OPEN)) return null;
  const after = buffer.slice(TOOL_CALL_OPEN.length);
  const callPrefix = 'call:';
  if (!after.startsWith(callPrefix)) {
    // Opener present but we don't even have `call:` yet — model is still
    // emitting the prefix.
    return { name: null };
  }
  const rest = after.slice(callPrefix.length);
  // Name runs up to `{` (start of body) or whitespace.
  const match = rest.match(/^([A-Za-z0-9_]+)/);
  if (!match) return { name: null };
  return { name: match[1] };
}

/**
 * Tools whose body has a single "main" code/sql field that we want to stream
 * into the corresponding pane's editor as it's generated. The key is the
 * argument name in the Gemma tool-call body (e.g. `code:<|"|>...<|"|>`).
 */
const STREAMING_FIELDS: Record<string, { kind: 'python' | 'sql' | 'react'; key: string }> = {
  RunPython: { kind: 'python', key: 'code' },
  RunSQL: { kind: 'sql', key: 'sql' },
  RunReact: { kind: 'react', key: 'code' },
};

/**
 * Pull the partial source string for a streaming tool-call body. Returns the
 * pane kind and the substring between the opening `<key>:<|"|>` and either
 * the closing `<|"|>` (if it has arrived) or the current end of the buffer.
 * Returns `null` if the body isn't yet shaped like a recognised streaming
 * tool, or the relevant field hasn't started yet.
 */
function extractStreamingCode(
  buffer: string,
): { kind: 'python' | 'sql' | 'react'; source: string } | null {
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

/**
 * Adapt InternalMessage[] for compactConversation by folding tool-role
 * entries into the preceding assistant turn so the summariser sees normal
 * user/assistant alternation.
 */
function convToChatMessagesForCompaction(
  conv: InternalMessage[],
): ChatMessage[] {
  const out: ChatMessage[] = [];
  conv.forEach((m, i) => {
    if (m.role === 'tool') {
      const trailer = `\n[← ${m.toolName ?? 'tool'}: ${m.content}]`;
      const last = out[out.length - 1];
      if (last && last.role === 'assistant') {
        out[out.length - 1] = { ...last, content: last.content + trailer };
      } else {
        out.push({
          id: `inline-compact-${i}`,
          role: 'assistant',
          content: trailer.trimStart(),
          createdAt: 0,
        });
      }
      return;
    }
    out.push({
      id: `inline-compact-${i}`,
      role: m.role,
      content: m.content,
      createdAt: 0,
    });
  });
  return out;
}

interface MidStreamCompactArgs {
  conv: InternalMessage[];
  resultStr: string;
  config: LLMConfig;
  signal?: AbortSignal;
}

/**
 * Returns the summary string and mutates `conv` in place when the projected
 * size of the next prompt would meet maxTokens; returns null otherwise (and
 * for empty summaries, so callers don't fire a no-op UI re-render).
 */
async function maybeCompactBeforeToolResult(
  args: MidStreamCompactArgs,
): Promise<string | null> {
  const { conv, resultStr, config, signal } = args;
  const usage = tokenUsageStore.getSnapshot();
  const currentStep = (usage?.input ?? 0) + (usage?.output ?? 0);
  if (currentStep + estimateResultTokens(resultStr) < LOCAL_GEMMA_CONTEXT_WINDOW) {
    return null;
  }
  const lastUserIdx = conv.findLastIndex((m) => m.role === 'user');
  if (lastUserIdx <= 0) return null;

  const older = conv.slice(0, lastUserIdx);
  const recent = conv.slice(lastUserIdx);
  let summary: string;
  try {
    summary = await compactConversation({
      config,
      toCompact: convToChatMessagesForCompaction(older),
      signal,
    });
  } catch (compactErr) {
    // An abort must end the turn (handled by streamLocalGemma's top-level
    // catch), not fall through to dispatching with an uncompacted history.
    if (isAbortError(compactErr)) throw compactErr;
    console.warn(
      '[streamLocalGemma] Inline pre-tool-result compaction failed:',
      compactErr,
    );
    return null;
  }
  conv.length = 0;
  conv.push(...recent);
  return summary.trim() ? summary : null;
}

export async function streamLocalGemma(opts: StreamChatOptions): Promise<void> {
  const {
    config,
    messages,
    tools,
    toolDispatcher,
    signal,
    onToken,
    onHistoryDelta,
    onDone,
    onError,
    onUsage,
    onMidStreamCompaction,
    onMaxIterationsReached,
  } = opts;
  const dispatch = toolDispatcher ?? runAgentTool;
  const emitHistory = (delta: string): void => {
    if (delta && onHistoryDelta) onHistoryDelta(delta);
  };

  // Track decode-only time and output tokens across iterations so the UI can
  // show a tokens/sec figure that excludes prefill and tool-dispatch latency.
  // The worker measures both per generation (`GenerateStats`).
  let totalOutputTokens = 0;
  let totalDecodeMs = 0;

  const reportUsage = (stats: GenerateStats): void => {
    totalOutputTokens += stats.outputTokens;
    totalDecodeMs += stats.decodeMs;
    if (!onUsage) return;
    const tps =
      totalDecodeMs > 0 ? totalOutputTokens / (totalDecodeMs / 1000) : undefined;
    onUsage({ input: stats.promptTokens, output: stats.outputTokens, tps });
  };

  const modelId = resolveActiveLocalModelIdOrDefault(config);

  const thinkingEnabled = config.thinkingEnabled?.[LOCAL_GEMMA_ENDPOINT] ?? false;

  let accumulatedText = '';
  const emit = (delta: string): void => {
    if (!delta) return;
    accumulatedText += delta;
    onToken(delta);
  };

  let systemPrompt = messages
    .filter((m) => m.role === 'system')
    .map((m) => m.content)
    .join('\n\n')
    .trim();

  const conv: InternalMessage[] = messages
    .filter((m) => m.role !== 'system')
    .map((m) =>
      m.role === 'assistant'
        ? { role: 'assistant' as const, content: m.content }
        : { role: 'user' as const, content: m.content },
    );

  try {
    await ensureLoaded(modelId);

    for (let iter = 0; iter < MAX_TOOL_ITERATIONS; iter++) {
      if (signal?.aborted) {
        onDone(accumulatedText);
        return;
      }

      const prompt = renderConversationForGemma(systemPrompt, conv, tools ?? [], thinkingEnabled);

      // Three parallel buffers:
      //  - `assistantTurnText`: what the UI renders (thought markers + thoughts
      //    + body, but never a stray `<channel|>`).
      //  - `assistantTurnHistory`: thought-free; persists into chat history so
      //    *past* turns re-fed on later user messages follow the template's
      //    bare-past-turn rule.
      //  - `assistantTurnConv`: byte-exact replay of the model's output
      //    (thought markers, thoughts, stray closes, body). It is pushed into
      //    `conv` for the rest of this turn: the template replays reasoning for
      //    messages after the last user turn, and an exact replay is what lets
      //    the worker reuse its KV cache across tool iterations.
      let assistantTurnText = '';
      let assistantTurnHistory = '';
      let assistantTurnConv = '';
      // Body text minus any stray `<channel|>` → UI + persisted history.
      const emitVisible = (text: string): void => {
        const visible = text.split(CHANNEL_CLOSE).join('');
        if (!visible) return;
        assistantTurnText += visible;
        assistantTurnHistory += visible;
        emit(visible);
        emitHistory(visible);
      };
      // Body-only buffer for the existing tool-call streaming parser.
      let toolBuffer = '';
      let pendingToolCall: { name: string; argsJson: string } | null = null;

      // When thinking is on, the very first iteration's prompt ends with an
      // open `<|channel>thought\n`, so the model resumes inside the thought
      // channel. Subsequent iterations after a tool response do NOT add a
      // fresh open, so they start `outside`.
      // Per-iteration controller: aborted by the outer signal, or by us once a
      // complete tool call has been parsed (no point decoding past it).
      const iterCtrl = new AbortController();
      const forwardAbort = (): void => iterCtrl.abort();
      signal?.addEventListener('abort', forwardAbort, { once: true });

      const splitter = createSplitterState(
        thinkingEnabled && iter === 0 ? 'in-thought' : 'outside',
      );
      // For the in-thought start, surface the (already-emitted-in-prompt)
      // open marker to the UI so the parser sees a complete thinking block.
      if (thinkingEnabled && iter === 0) {
        assistantTurnText += THINKING_OPEN_MARKER;
        assistantTurnConv += THINKING_OPEN_MARKER;
        emit(THINKING_OPEN_MARKER);
      }

      const handleEvent = (e: SplitterEvent): void => {
        if (pendingToolCall) return;
        if (e.kind === 'open') {
          assistantTurnText += THINKING_OPEN_MARKER;
          assistantTurnConv += THINKING_OPEN_MARKER;
          emit(THINKING_OPEN_MARKER);
          return;
        }
        if (e.kind === 'close') {
          assistantTurnText += CHANNEL_CLOSE;
          assistantTurnConv += CHANNEL_CLOSE;
          emit(CHANNEL_CLOSE);
          return;
        }
        if (e.kind === 'thought') {
          assistantTurnText += e.text;
          assistantTurnConv += e.text;
          emit(e.text);
          return;
        }
        // body (or a stray `<channel|>` the model emitted outside a thought
        // channel) — route through the tool-call streaming parser so partial
        // `<|tool_call>` prefixes stay held back. The stray close travels
        // through the same buffer so it lands at its true position in the
        // byte-exact `assistantTurnConv` replay; it is stripped from what the
        // UI and persisted history see (`emitVisible`).
        toolBuffer += e.kind === 'stray-close' ? CHANNEL_CLOSE : e.text;
        const parsed = parseStreamForToolCall(toolBuffer);
        if (parsed.emitText) {
          assistantTurnConv += parsed.emitText;
          emitVisible(parsed.emitText);
        }
        toolBuffer = parsed.rest;
        if (parsed.toolCall) {
          pendingToolCall = parsed.toolCall;
          // Once the call is fully parsed, the agent dispatch will switch
          // the relevant pane to pending and the throbber will surface
          // "Running Python" / "Running SQL" instead.
          setLlmPreparingToolCall(null);
          // Stop decoding now — anything after the call is discarded anyway.
          iterCtrl.abort();
        } else {
          setLlmPreparingToolCall(extractPreparingToolCall(toolBuffer));
          const streaming = extractStreamingCode(toolBuffer);
          if (streaming) {
            setStreamingSource(streaming.kind, streaming.source);
          }
        }
      };

      let iterStats = null as GenerateStats | null;
      try {
        await generate({
          prompt,
          signal: iterCtrl.signal,
          onToken: (delta) => {
            if (pendingToolCall) return;
            for (const e of feedSplitter(splitter, delta)) {
              handleEvent(e);
              if (pendingToolCall) break;
            }
          },
          onStats: (s) => {
            iterStats = s;
          },
        });
      } finally {
        signal?.removeEventListener('abort', forwardAbort);
      }

      // Drain the splitter at the end of generation.
      if (!pendingToolCall) {
        for (const e of flushSplitter(splitter)) {
          handleEvent(e);
          if (pendingToolCall) break;
        }
      }

      if (!pendingToolCall) {
        // Drain any remaining tool-buffer tail as plain text — when the
        // stream ends without a tool call, holdback chars are just text.
        if (toolBuffer) {
          assistantTurnConv += toolBuffer;
          emitVisible(toolBuffer);
          toolBuffer = '';
        }
        if (iterStats) reportUsage(iterStats);
        onDone(accumulatedText);
        return;
      }

      if (iterStats) reportUsage(iterStats);

      const tc: { name: string; argsJson: string } = pendingToolCall;
      const toolCallToken = formatToolCallToken(tc.name, tc.argsJson);

      conv.push({
        role: 'assistant',
        content: assistantTurnConv + toolCallToken,
      });
      emitHistory(toolCallToken);

      if (signal?.aborted) {
        onDone(accumulatedText);
        return;
      }

      let inputObj: unknown;
      try {
        inputObj = JSON.parse(tc.argsJson);
      } catch {
        inputObj = {};
      }
      emit(`\n\n→ ${tc.name}(${tc.argsJson || '{}'})\n`);
      const result = await dispatch(tc.name, inputObj, signal);
      const resultStr = clampToolResultSize(tc.name, JSON.stringify(result));

      // Pre-emptive compaction: if appending this tool result would push the
      // next prompt past the context window, summarise older conv entries
      // first. Otherwise the next generate() rejects with
      // `ContextTooLongError` and the tool result is lost.
      const newSummary = await maybeCompactBeforeToolResult({
        conv,
        resultStr,
        config,
        signal,
      });
      if (newSummary) {
        systemPrompt += COMPACTION_HEADER + newSummary;
        onMidStreamCompaction?.({ summary: newSummary });
      }

      emit(`← ${resultStr}\n\n`);
      emitHistory(formatToolResponseToken(tc.name, resultStr));

      conv.push({
        role: 'tool',
        toolName: tc.name,
        content: resultStr,
      });
    }

    emit('\n\nReached max tool iterations');
    onMaxIterationsReached?.();
    onDone(accumulatedText);
  } catch (err) {
    if (isAbortError(err)) {
      onDone(accumulatedText);
      return;
    }
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}
