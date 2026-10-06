import { runAgentTool } from '../agentTools';
import { isAbortError, type StreamChatOptions } from '../streamChat';
import { clampToolResultSize, estimateResultTokens } from '../toolResultLimits';
import { RepeatedCallGuard, repeatedCallNote } from '../repeatedToolCalls';
import { classifyToolCall } from '../zeos/zeosToolClasses';
import { LOCAL_GEMMA_CONTEXT_WINDOW } from '../contextWindow';
import { compactConversation } from '../compactConversation';
import { COMPACTION_HEADER } from '../autoCompaction';
import * as tokenUsageStore from '../tokenUsageStore';
import type { ChatMessage } from '../../types/chat';
import { getLocalGemmaModel, resolveActiveLocalModelIdOrDefault } from './models';
import { ensureLoaded, generate } from './llmService';
import type { GenerateStats } from './llmWorkerProtocol';
import {
  formatToolCallToken,
  formatToolResponseToken,
  CHANNEL_OPEN,
  CHANNEL_CLOSE,
  type InternalMessage,
} from './toolPrompt';
import { getPromptFormat, type ParsedToolCall } from './promptFormat';
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
/**
 * Canonical (Gemma) thought-channel open marker. Whatever the model family,
 * this is what the UI and stored history see; the model's native markers
 * only go into the in-progress turn replay (`assistantTurnConv`).
 */
const THINKING_OPEN_MARKER = `${CHANNEL_OPEN}thought\n`;

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
  const fmt = getPromptFormat(getLocalGemmaModel(modelId)?.family ?? 'gemma', tools ?? []);

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

  const conv: InternalMessage[] = fmt.importHistory(
    messages
      .filter((m) => m.role !== 'system')
      .map((m) => ({
        role: m.role === 'assistant' ? ('assistant' as const) : ('user' as const),
        content: m.content,
      })),
  );

  // Calls made this user message since the last effect: an identical repeat
  // gets a note instead of running again, and still counts as an iteration.
  const repeats = new RepeatedCallGuard();

  try {
    await ensureLoaded(modelId);

    for (let iter = 0; iter < MAX_TOOL_ITERATIONS; iter++) {
      if (signal?.aborted) {
        onDone(accumulatedText);
        return;
      }

      const prompt = fmt.render(systemPrompt, conv, tools ?? [], thinkingEnabled);

      // Three parallel buffers:
      //  - `assistantTurnText`: what the UI renders (thought markers + thoughts
      //    + body, but never a stray `<channel|>`).
      //  - `assistantTurnHistory`: thought-free; persists into chat history so
      //    *past* turns re-fed on later user messages follow the template's
      //    bare-past-turn rule.
      //  - `assistantTurnConv`: byte-exact replay of the model's output in its
      //    native format (thought markers, thoughts, stray closes, body, plus
      //    whatever the prompt pre-filled for this turn). It is pushed into
      //    `conv` for the rest of this turn: the template replays reasoning for
      //    messages after the last user turn, and an exact replay is what lets
      //    the worker reuse its KV cache across tool iterations.
      let assistantTurnText = '';
      let assistantTurnHistory = '';
      let assistantTurnConv = '';
      // Body text minus any stray close marker → UI + persisted history.
      const emitVisible = (text: string): void => {
        const visible = text.split(fmt.markers.close).join('');
        if (!visible) return;
        assistantTurnText += visible;
        assistantTurnHistory += visible;
        emit(visible);
        emitHistory(visible);
      };
      // Body-only buffer for the existing tool-call streaming parser.
      let toolBuffer = '';
      let pendingToolCall: ParsedToolCall | null = null;

      // Per-iteration controller: aborted by the outer signal, or by us once a
      // complete tool call has been parsed (no point decoding past it).
      const iterCtrl = new AbortController();
      const forwardAbort = (): void => iterCtrl.abort();
      signal?.addEventListener('abort', forwardAbort, { once: true });

      // Where this generation starts. Gemma: with thinking on, only the first
      // iteration's prompt ends inside an open thought channel (after a tool
      // response the same turn continues). Qwen: every iteration opens a new
      // assistant turn whose prompt pre-fills a `<think>` prefix.
      const turnStart = fmt.turnStart(thinkingEnabled, iter);
      const splitter = createSplitterState(turnStart.mode, fmt.markers);
      assistantTurnConv += turnStart.convPrefix;
      // For the in-thought start, surface the (already-emitted-in-prompt)
      // open marker to the UI so the parser sees a complete thinking block.
      if (turnStart.mode === 'in-thought') {
        assistantTurnText += THINKING_OPEN_MARKER;
        emit(THINKING_OPEN_MARKER);
      }

      const handleEvent = (e: SplitterEvent): void => {
        if (pendingToolCall) return;
        if (e.kind === 'open') {
          assistantTurnText += THINKING_OPEN_MARKER;
          assistantTurnConv += fmt.markers.open;
          emit(THINKING_OPEN_MARKER);
          return;
        }
        if (e.kind === 'close') {
          assistantTurnText += CHANNEL_CLOSE;
          assistantTurnConv += fmt.markers.close;
          emit(CHANNEL_CLOSE);
          return;
        }
        if (e.kind === 'thought') {
          assistantTurnText += e.text;
          assistantTurnConv += e.text;
          emit(e.text);
          return;
        }
        // body (or a stray close marker the model emitted outside a thought
        // channel) — route through the tool-call streaming parser so partial
        // tool-call openers stay held back. The stray close travels
        // through the same buffer so it lands at its true position in the
        // byte-exact `assistantTurnConv` replay; it is stripped from what the
        // UI and persisted history see (`emitVisible`).
        toolBuffer += e.kind === 'stray-close' ? fmt.markers.close : e.text;
        const parsed = fmt.parseStreamForToolCall(toolBuffer);
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
          setLlmPreparingToolCall(fmt.extractPreparingToolCall(toolBuffer));
          const streaming = fmt.extractStreamingCode(toolBuffer);
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

      const tc: ParsedToolCall = pendingToolCall;

      conv.push({
        role: 'assistant',
        content: assistantTurnConv + fmt.toolCallForConv(tc),
      });
      // History always stores the canonical (Gemma) token.
      emitHistory(formatToolCallToken(tc.name, tc.argsJson));

      if (signal?.aborted) {
        onDone(accumulatedText);
        return;
      }

      let inputObj: unknown;
      let parsedArgs = true;
      try {
        inputObj = JSON.parse(tc.argsJson);
      } catch {
        inputObj = {};
        parsedArgs = false;
      }
      emit(`\n\n→ ${tc.name}(${tc.argsJson || '{}'})\n`);
      // Unparseable arguments are keyed by their text, so two different
      // malformed calls are not taken for the same `{}`.
      const keyArgs = parsedArgs ? inputObj : tc.argsJson;
      let resultStr: string;
      if (repeats.isRepeat(tc.name, keyArgs)) {
        resultStr = repeatedCallNote(tc.name);
      } else {
        // Anything but a read (by the ZEOS classes; an unknown tool is an
        // effect) may change what a repeated read returns: the record restarts.
        const args = inputObj && typeof inputObj === 'object' ? (inputObj as Record<string, unknown>) : undefined;
        repeats.record(tc.name, keyArgs, classifyToolCall(tc.name, args) !== 'read');
        const result = await dispatch(tc.name, inputObj, signal);
        resultStr = clampToolResultSize(tc.name, JSON.stringify(result));
      }

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
