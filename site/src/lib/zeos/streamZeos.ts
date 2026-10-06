/**
 * The agent loop for ZEOS Qwen 4B: the same callback contract as
 * `streamLocalGemma` (UI text through `onToken`, canonical Gemma history
 * through `onHistoryDelta`, tool calls run by `runAgentTool`, abort through
 * `signal`), with the model driven by the ZEOS kernel instead of the
 * transformers.js worker.
 *
 * One ZEOS `ChatRun` holds one conversation and lives across turns. It is
 * keyed by the conversation it has seen (system prompt, thinking flag, and
 * every prior message), so the next message of the same chat goes to the same
 * run, while a new chat, a reload, a retry or an edit opens a fresh run and
 * replays the stored history into it (`buildZeosImport`, with each turn's
 * ring). An aborted or failed turn drops the run; the next message replays.
 *
 * A turn: `send_user`, then step the kernel in small batches so tokens reach
 * the UI as the model decodes them. A `tool_call` event (the machine put the
 * call on `tools.read` or `tools.effect`) is drained, run with
 * `runAgentTool`, clamped and delivered on `tools.results` (ring 3), or on
 * `tools.results.trusted` (ring 2) for a result the app wrote itself
 * (`ZEOS_TRUSTED_RESULTS`: a bundled skill card). A `spoof` event (the
 * kernel's alarm on a result spelling a kernel frame) marks that result in
 * the chat (`ChatTrust.toolSpoofs`) and the journal. With "Mask tool choice"
 * on (`config.zeosMaskToolChoice`, ZEOS `mask_tool_choice`), the model writes
 * each tool's name with the ring-3 tool output hidden; a call whose name was
 * chosen that way is marked in the chat (`ChatTrust.toolMasked`) and the
 * journal (`ui.masked`). An
 * `approval_required` event (the kernel refused a write to `tools.effect` for
 * privilege) shows the approval card: Approve runs the call under the user's
 * authority and delivers its result, Deny delivers a refusal. The turn ends
 * when the job waits on `chat.user` again.
 */
import { getFeatures } from '../agentFeaturesStore';
import { runAgentTool, type AgentToolSpec } from '../agentTools';
import { setLlmPreparingToolCall, setStreamingSource } from '../executionPanelStore';
import { getLocalGemmaModel, resolveActiveLocalModelIdOrDefault } from '../localLlm/models';
import { noteZeosModuleLoaded } from '../localLlm/engineLifecycle';
import { getPromptFormat, type ParsedToolCall, type PromptFormat } from '../localLlm/promptFormat';
import { renderQwenSystemContent } from '../localLlm/qwenPrompt';
import {
  createSplitterState,
  feedSplitter,
  flushSplitter,
  type SplitterEvent,
  type SplitterState,
} from '../localLlm/thinkingChannelSplitter';
import {
  CHANNEL_CLOSE,
  CHANNEL_OPEN,
  formatToolCallToken,
  formatToolResponseToken,
} from '../localLlm/toolPrompt';
import { isAbortError, type StreamChatMessage, type StreamChatOptions } from '../streamChat';
import { clampToolResultSize } from '../toolResultLimits';
import type { ChatTrust } from '../../types/chat';
import { LOCAL_GEMMA_ENDPOINT } from '../../types/llm';
import { QWEN_SAMPLING } from '../localLlm/llmWorkerProtocol';
import type { AttentionLog } from './attentionProbe';
import {
  startKernelChatEngine,
  type ZeosSampling,
  type ZeosChatEngine,
  type ZeosChatRun,
  type ZeosGateMode,
  type ZeosEvent,
  type ZeosSegment,
} from './zeosChatEngine';
import {
  buildZeosImport,
  EXTERNAL,
  importStartIntegrity,
  ModelTextEscaper,
  toolResultForZeos,
  TRUSTED,
  userTextForZeos,
} from './zeosHistory';
import * as store from './zeosSessionStore';
import {
  dispatchForZeos,
  systemHasCompaction,
  ZeosCompactedConversationError,
  zeosSystemPrompt,
  type ZeosCallHow,
} from './zeosPrompt';
import {
  isTrustedToolResult,
  paramTypesFromTools,
  ZEOS_TOOL_CLASSES,
  ZEOS_TRUSTED_RESULTS,
  zeosAgentTools,
} from './zeosToolClasses';

/** Kernel ticks per `step` call: small, so tokens stream. */
const STEP_TICKS = 8;
/**
 * `KernelConfig.theta_read` for this model: the mass a ring-3 segment must
 * get over one 16-token block (16 decode steps, so out of 16) for the job to
 * count as having read it. Measured on the real model (CLAUDE.md, "ZEOS Qwen
 * 4B"): a tool result being read takes 1.5-5.9 a block at its peak, while a
 * resident result nobody is using takes 0.6 at most. The kernel default, 0.2,
 * sits inside that background, so it demoted on any long tool result.
 * Dev override: localStorage `gda.zeos.thetaRead`.
 */
export const ZEOS_THETA_READ = 1.0;
/**
 * Qwen 3.5's recommended non-thinking sampling, as `QWEN_SAMPLING` for the
 * transformers.js Qwen, but seeded by the kernel (ZEOS `Sampling`). Dev
 * override: localStorage `gda.zeos.sampling` = `greedy`.
 */
export const ZEOS_SAMPLING: ZeosSampling = {
  temperature: QWEN_SAMPLING.temperature,
  topK: QWEN_SAMPLING.top_k,
};

function devSetting(key: string): string | null {
  if (!import.meta.env.DEV) return null;
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function thetaRead(): number {
  const raw = devSetting('gda.zeos.thetaRead');
  const v = raw === null ? NaN : Number(raw);
  return Number.isFinite(v) && v > 0 ? v : ZEOS_THETA_READ;
}

function sampling(): ZeosSampling | null {
  return devSetting('gda.zeos.sampling') === 'greedy' ? null : ZEOS_SAMPLING;
}

/** A fresh seed per run, so a retry can differ; `gda.zeos.seed` pins it (dev). */
function runSeed(): number {
  const raw = devSetting('gda.zeos.seed');
  if (raw !== null && /^\d+$/.test(raw)) return Number(raw);
  return Math.floor(Math.random() * 2 ** 31);
}
/** Tool calls one turn may make before the loop stops (as streamLocalGemma). */
const MAX_TOOL_CALLS = 10;
/** What the model reads when the user declines (ZEOS `DEFAULT_REFUSAL`). */
export const ZEOS_REFUSAL = 'The user declined this tool call. It was not run.';
const THINKING_OPEN_MARKER = `${CHANNEL_OPEN}thought\n`;
/** The pipes a tool result arrives on (ZEOS `ChatPipes.result_pipes`). */
export const RESULTS_PIPE = 'tools.results';
export const TRUSTED_RESULTS_PIPE = 'tools.results.trusted';
const isResultPipe = (pipe: string | null): boolean => pipe === RESULTS_PIPE || pipe === TRUSTED_RESULTS_PIPE;

// ---- engine and session ------------------------------------------------------

// A switch to another model disposes this engine (engineLifecycle.ts).
noteZeosModuleLoaded();

let enginePromise: Promise<ZeosChatEngine> | null = null;
let engineOverride: (() => Promise<ZeosChatEngine>) | null = null;
/** The start in progress: aborting it terminates both workers (`disposeZeos`). */
let engineStart: AbortController | null = null;
/** The engine `enginePromise` resolved to, while it lives. */
let liveEngine: ZeosChatEngine | null = null;
/**
 * After a Stop: settles once the kernel has finished the call Stop
 * interrupted, or the watchdog has reset it. The next message waits for it.
 */
let settling: Promise<void> | null = null;

/**
 * How long the kernel may take to finish the call a Stop interrupted before
 * the engine is reset. A decode `step` batch takes well under a second; only
 * a prefill runs longer (the first turn's is ~33 s on an M1 Max, in one model
 * call). Stop drops the run, so the next message replays the history into a
 * fresh one and that prefill is thrown away anyway: restarting the engine
 * (~5 s with the files in the OS cache) is cheaper than waiting it out.
 */
export const ZEOS_ABORT_WATCHDOG_MS = 10_000;

/**
 * How long Stop lets a `step` batch already in flight finish before it
 * interrupts the kernel call. A decode batch (`STEP_TICKS` ticks, ~72 ms each
 * on an M1 Max) ends well within it, and its events are read as usual, so a
 * demotion in that batch is saved with the message. Only a long prefill is
 * cut short; then the batch's events are never seen. Either way the turn
 * ends early, so it is saved as demoted if anything untrusted was in the run
 * (`assumeCutShortDemoted`).
 */
export const ZEOS_STOP_GRACE_MS = 2_000;

/**
 * Consecutive `step` batches with no events (other than `waiting`) while the
 * job is not waiting on `chat.user`, after which the engine counts as stuck:
 * the job waits on a delivery this loop will never make, or ticks without
 * doing anything. It is disposed like a crashed one (`engineDied`), so the
 * turn ends with an error instead of looping forever. A prefill is one model
 * call inside one tick, so a working kernel never comes near this.
 */
export const ZEOS_MAX_IDLE_STEPS = 64;

interface Session {
  run: ZeosChatRun;
  key: string;
  /** Segment id → what it is, for "Demoted by …". */
  segments: Map<number, string>;
  /** Tool results so far in the conversation (for "result #n"). */
  toolCount: number;
  integrity: number;
  demotedBy: string | null;
  /**
   * Something on ring 3 (a tool result or refusal, a replayed untrusted turn)
   * is in the run, or has been delivered to it, so a step could demote. Set
   * on delivery, not only when the arrival is seen: a step that is cut short
   * may have taken it in unseen.
   */
  untrusted: boolean;
}

let session: Session | null = null;

function defaultModel() {
  const m = getLocalGemmaModel('zeos-qwen3.5-4b');
  if (!m) throw new Error('ZEOS Qwen 4B is not available in this build (local-models dev mode only).');
  return m;
}

function engine(): Promise<ZeosChatEngine> {
  if (!enginePromise) {
    store.setStatus('starting', 'Starting the ZEOS kernel');
    const ctrl = new AbortController();
    engineStart = ctrl;
    const start =
      engineOverride ??
      (() =>
        startKernelChatEngine(defaultModel(), {
          signal: ctrl.signal,
          onStatus: (text) => store.setStatus('starting', text),
          onProgress: (p) => {
            if (p.phase === 'session') return store.setStatus('starting', 'Loading model onto GPU');
            const done = typeof p.bytes === 'number' ? p.bytes : 0;
            const total = typeof p.bytes_total === 'number' ? p.bytes_total : 0;
            const pct = total > 0 ? ` · ${Math.round((100 * done) / total)}%` : '';
            if (p.phase === 'download') store.setStatus('starting', `Loading model${pct}`);
          },
        }));
    const started: Promise<ZeosChatEngine> = start().then(
      (e) => {
        if (ctrl.signal.aborted) {
          // `disposeZeos` ran while it started.
          const reason = ctrl.signal.reason instanceof Error ? ctrl.signal.reason : new Error('ZEOS unloaded');
          e.dispose?.(reason);
          throw reason;
        }
        if (engineStart === ctrl) engineStart = null;
        liveEngine = e;
        e.onDispose?.((reason) => engineDied(e, reason));
        store.setStatus('ready', '', { backend: e.backend });
        return e;
      },
      (err: unknown) => {
        if (enginePromise === started) enginePromise = null;
        if (engineStart === ctrl) engineStart = null;
        if (!ctrl.signal.aborted) store.setStatus('error', err instanceof Error ? err.message : String(err));
        throw err;
      },
    );
    enginePromise = started;
  }
  return enginePromise;
}

/** The engine, once any Stop before this has settled (see `settling`). */
async function readyEngine(): Promise<ZeosChatEngine> {
  while (settling) await settling;
  return engine();
}

/**
 * Forget engine `e` and its run, so the next message starts a fresh one. A
 * pending approval card would otherwise wait on a dead run forever: it is
 * cancelled, so the turn behind it ends with `reason` (not a Deny).
 */
function forgetEngine(e: ZeosChatEngine, reason: Error): boolean {
  if (liveEngine !== e) return false;
  liveEngine = null;
  enginePromise = null;
  session = null;
  store.resetConversation(reason);
  return true;
}

/**
 * The engine died on its own (a worker or model-thread crash, a model call
 * that timed out, Pyodide's fatal error). The call in flight, if any, has
 * already rejected with `reason`, so a streaming turn ends with that error;
 * here the status says so and the next message starts a fresh engine.
 */
function engineDied(e: ZeosChatEngine, reason: Error): void {
  if (!forgetEngine(e, reason)) return;
  console.error('[zeos] the engine stopped:', reason);
  store.setStatus('error', `ZEOS stopped: ${reason.message}. Your next message restarts it.`, {
    backend: null,
  });
}

/**
 * Stop: reject the kernel call the turn is waiting on at once (a prefill can
 * hold one call for tens of seconds), then wait for the kernel to finish it in
 * the background. If it has not within `ZEOS_ABORT_WATCHDOG_MS`, reset the
 * engine.
 */
function interruptForStop(): void {
  const e = liveEngine;
  if (!e?.interrupt) return;
  e.interrupt(new DOMException('Aborted', 'AbortError'));
  if (!e.ping) return;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const watchdog = new Promise<'timeout'>((resolve) => {
    timer = setTimeout(() => resolve('timeout'), ZEOS_ABORT_WATCHDOG_MS);
  });
  const done: Promise<void> = Promise.race([
    e.ping().then(
      () => 'settled' as const,
      () => 'settled' as const,
    ),
    watchdog,
  ]).then((outcome) => {
    clearTimeout(timer);
    if (settling === done) settling = null;
    const restarted = new Error('ZEOS was restarted after Stop');
    if (outcome === 'timeout' && forgetEngine(e, restarted)) {
      console.warn(`[zeos] the kernel did not settle within ${ZEOS_ABORT_WATCHDOG_MS} ms of Stop; restarting it`);
      e.dispose?.(restarted);
      store.setStatus('idle', '', { backend: null });
    }
  });
  settling = done;
}

/** Start the kernel and model thread ahead of the first message (model picker, boot). */
export async function warmZeos(): Promise<void> {
  await readyEngine();
}

/**
 * Unload ZEOS Qwen 4B: terminate the kernel worker and the model thread, which
 * frees the model's GPU memory (~2.8 GB), and abort a start in progress. A
 * pending approval is cancelled and a turn in flight ends with an error. The
 * next message (or `warmZeos`) starts it again. Called when the chat switches
 * to another model (src/lib/localLlm/engineLifecycle.ts).
 */
export function disposeZeos(why = 'model switched'): void {
  const reason = new Error(`ZEOS Qwen 4B was unloaded (${why})`);
  engineStart?.abort(reason);
  engineStart = null;
  const e = liveEngine;
  const wasLoaded = e !== null || enginePromise !== null;
  if (e) forgetEngine(e, reason);
  enginePromise = null;
  session = null;
  settling = null;
  store.resetConversation(reason);
  e?.dispose?.(reason);
  if (wasLoaded) store.setStatus('idle', '', { backend: null });
}

/** Whether a ZEOS engine is running or starting (tests, the lifecycle). */
export function isZeosLoaded(): boolean {
  return liveEngine !== null || enginePromise !== null;
}

/** `p`, or an AbortError as soon as `signal` aborts (`p` runs on). */
function untilAborted<T>(p: Promise<T>, signal: AbortSignal | undefined): Promise<T> {
  if (!signal) return p;
  return new Promise<T>((resolve, reject) => {
    const onAbort = (): void => reject(new DOMException('Aborted', 'AbortError'));
    if (signal.aborted) return onAbort();
    signal.addEventListener('abort', onAbort, { once: true });
    p.then(resolve, reject).finally(() => signal.removeEventListener('abort', onAbort));
  });
}

async function dropSession(): Promise<void> {
  const s = session;
  session = null;
  if (s) await s.run.close().catch((err) => console.warn('closing the ZEOS run failed:', err));
}

/** Tests: run against a scripted engine, and forget any session. */
export async function __setZeosEngineForTests(
  factory: (() => Promise<ZeosChatEngine>) | null,
): Promise<void> {
  await dropSession();
  engineOverride = factory;
  enginePromise = null;
  engineStart = null;
  liveEngine = null;
  settling = null;
}

/**
 * What a run was opened for. Anything that changes it -- the past, the system
 * prompt, thinking, the gate mode, tool-choice masking -- opens a fresh run
 * and replays history, so a mode switch mid-chat applies from the next
 * message.
 */
function conversationKey(
  system: string,
  thinking: boolean,
  gateMode: ZeosGateMode,
  maskToolChoice: boolean,
  turns: readonly StreamChatMessage[],
): string {
  return JSON.stringify([system, thinking, gateMode, maskToolChoice, turns.map((m) => [m.role, m.content])]);
}

function segmentLabel(seg: Pick<ZeosSegment, 'pipe' | 'segment'>, s: Session): string {
  const known = s.segments.get(seg.segment);
  if (known) return known;
  if (seg.pipe.startsWith('chat.history')) return 'an earlier assistant turn';
  if (seg.pipe === 'chat.user') return 'a user message';
  return `${seg.pipe} segment ${seg.segment}`;
}

// ---- the UI side of one turn ---------------------------------------------------

/**
 * Turns the model's tokens into the UI's and history's canonical text, as
 * `streamLocalGemma` does for Qwen: `<think>` → the Gemma thought channel,
 * tool-call text held back until it parses, then replaced by `→ name(args)` /
 * `← result` markers in the UI and Gemma tool tokens in history. The model's
 * own text is escaped on the way (`ModelTextEscaper`), so it can never spell
 * those markers or tokens; only the app writes them.
 */
export class ZeosTurnText {
  private splitter!: SplitterState;
  private toolBuffer = '';
  private readonly uiText = new ModelTextEscaper();
  private readonly historyText = new ModelTextEscaper();
  /** The call the parser closed in the current assistant turn, if any. */
  parsedCall: ParsedToolCall | null = null;

  constructor(
    private readonly fmt: PromptFormat,
    private readonly thinking: boolean,
    private readonly emit: (ui: string) => void,
    private readonly emitHistory: (history: string) => void,
  ) {
    this.startAssistantTurn();
  }

  /** Model text for the UI. */
  private modelUi(text: string): void {
    this.emit(this.uiText.push(text));
  }

  /** Text the app writes into the UI (markers): the model's held-back tail goes first. */
  private appUi(text: string): void {
    this.emit(this.uiText.flush() + text);
  }

  private appHistory(text: string): void {
    this.emitHistory(this.historyText.flush() + text);
  }

  /** Each Qwen assistant turn (the first, and one after every tool response). */
  startAssistantTurn(): void {
    const start = this.fmt.turnStart(this.thinking, 0);
    this.splitter = createSplitterState(start.mode, this.fmt.markers);
    this.toolBuffer = '';
    this.parsedCall = null;
    if (start.mode === 'in-thought') this.appUi(THINKING_OPEN_MARKER);
  }

  token(text: string): void {
    for (const e of feedSplitter(this.splitter, text)) this.handle(e);
  }

  private visible(text: string): void {
    const shown = text.split(this.fmt.markers.close).join('');
    if (!shown) return;
    this.modelUi(shown);
    this.emitHistory(this.historyText.push(shown));
  }

  private handle(e: SplitterEvent): void {
    if (this.parsedCall) return;
    if (e.kind === 'open') return this.appUi(THINKING_OPEN_MARKER);
    if (e.kind === 'close') return this.appUi(CHANNEL_CLOSE);
    if (e.kind === 'thought') return this.modelUi(e.text);
    this.toolBuffer += e.kind === 'stray-close' ? this.fmt.markers.close : e.text;
    const parsed = this.fmt.parseStreamForToolCall(this.toolBuffer);
    if (parsed.emitText) this.visible(parsed.emitText);
    this.toolBuffer = parsed.rest;
    if (parsed.toolCall) {
      this.parsedCall = parsed.toolCall;
      setLlmPreparingToolCall(null);
    } else {
      setLlmPreparingToolCall(this.fmt.extractPreparingToolCall(this.toolBuffer));
      const streaming = this.fmt.extractStreamingCode(this.toolBuffer);
      if (streaming) setStreamingSource(streaming.kind, streaming.source);
    }
  }

  /** The machine wrote (or tried to write) a call: whatever is held back was its text. */
  callClosed(): void {
    for (const e of flushSplitter(this.splitter)) this.handle(e);
    this.toolBuffer = '';
    setLlmPreparingToolCall(null);
  }

  /** A call and its result, in both canonical forms; then the next assistant turn. */
  toolExchange(name: string, argsJson: string, resultStr: string): void {
    this.appHistory(formatToolCallToken(name, argsJson));
    this.appUi(`\n\n→ ${name}(${argsJson || '{}'})\n`);
    this.appUi(`← ${resultStr}\n\n`);
    this.appHistory(formatToolResponseToken(name, resultStr));
    this.startAssistantTurn();
  }

  /** Release the model text the escapers hold back (the turn was cut short). */
  flushHeld(): void {
    this.appUi('');
    this.appHistory('');
  }

  /** The turn ended in a reply: release anything held back as text. */
  finish(): void {
    for (const e of flushSplitter(this.splitter)) this.handle(e);
    if (!this.parsedCall && this.toolBuffer) {
      this.visible(this.toolBuffer);
      this.toolBuffer = '';
    }
    this.flushHeld();
  }
}

/** Why the kernel refused, prefixed with the gate mode so the two read side by side. */
export function refusalReason(
  e: Pick<Extract<ZeosEvent, { type: 'approval_required' }>, 'integrity' | 'session_floor' | 'name' | 'fault' | 'detail'>,
  gateMode: ZeosGateMode,
  demotedBy: string | null,
): string {
  if (e.integrity >= EXTERNAL) {
    return `${gateMode}: demoted${demotedBy ? ` by ${demotedBy}` : ''}, so ${e.name} needs your approval.`;
  }
  if ((e.session_floor ?? TRUSTED) >= EXTERNAL) {
    return `${gateMode}: read tool output this turn, so ${e.name} needs your approval.`;
  }
  return `${gateMode}: the kernel refused ${e.name} (${e.fault}: ${e.detail}).`;
}

// ---- the loop ----------------------------------------------------------------------

export async function streamZeos(opts: StreamChatOptions): Promise<void> {
  const { config, messages, signal, onToken, onHistoryDelta, onDone, onError, onUsage } = opts;
  const thinking = config.thinkingEnabled?.[LOCAL_GEMMA_ENDPOINT] ?? false;
  const gateMode: ZeosGateMode = config.zeosAttentionOnly ? 'attention' : 'strict';
  const maskToolChoice = config.zeosMaskToolChoice ?? false;
  const tools: AgentToolSpec[] = zeosAgentTools(opts.tools ?? []);
  const features = { ...getFeatures(), runSubAgent: false };
  const baseDispatch =
    opts.toolDispatcher ?? ((name: string, input: unknown, sig?: AbortSignal) => runAgentTool(name, input, sig, features));
  const dispatch = (name: string, input: unknown, how: ZeosCallHow, sig?: AbortSignal): Promise<unknown> =>
    dispatchForZeos(name, input, how, (n, i) => Promise.resolve(baseDispatch(n, i, sig)));
  const fmt = getPromptFormat('zeos-qwen', tools);

  let accumulated = '';
  const emit = (delta: string): void => {
    if (!delta) return;
    accumulated += delta;
    onToken(delta);
  };
  let history = '';
  const emitHistory = (delta: string): void => {
    if (!delta) return;
    history += delta;
    onHistoryDelta?.(delta);
  };

  const system = messages
    .filter((m) => m.role === 'system')
    .map((m) => m.content)
    .join('\n\n')
    .trim();
  // A chat compacted under another model cannot run here (its summary would
  // enter the trusted system prompt, and the dropped turns' trust is gone):
  // refuse before any engine or run is touched. The chat refuses it first
  // (`isCompactedConversation`); this is the second layer.
  if (systemHasCompaction(system)) {
    onError(new ZeosCompactedConversationError());
    return;
  }
  const turns = messages.filter((m) => m.role !== 'system');
  const last = turns[turns.length - 1];
  if (!last || last.role !== 'user') {
    onError(new Error('streamZeos: the conversation must end with a user message.'));
    return;
  }
  // ZEOS refuses an empty user message (nothing would arrive, and the job
  // would wait on for it), so say so here rather than as a kernel error.
  const userText = userTextForZeos(last.content);
  if (!userText) {
    onError(new Error('streamZeos: the message is empty.'));
    return;
  }
  const prior = turns.slice(0, -1);
  const key = conversationKey(system, thinking, gateMode, maskToolChoice, prior);

  const toolRings: number[] = [];
  /** Indices into `toolRings` of results the kernel raised a spoof alarm on. */
  const toolSpoofs: number[] = [];
  /** Indices into `toolRings` of calls whose name was chosen masked. */
  const toolMasked: number[] = [];
  const reportTrust = (s: Session): void => {
    const trust: ChatTrust = {
      integrity: s.integrity,
      ring: s.integrity,
      toolRings: [...toolRings],
      ...(toolSpoofs.length > 0 ? { toolSpoofs: [...toolSpoofs] } : {}),
      ...(toolMasked.length > 0 ? { toolMasked: [...toolMasked] } : {}),
      ...(s.demotedBy ? { demotedBy: s.demotedBy } : {}),
    };
    opts.onTrust?.(trust);
  };

  let outputTokens = 0;
  let firstTokenAt: number | null = null;
  let lastTokenAt = 0;
  /** The turn in progress, so every exit path can save its trust (T5). */
  let live: { s: Session; text: ZeosTurnText } | null = null;
  /** A `step` call is waiting for the kernel. */
  let stepping = false;
  let graceTimer: ReturnType<typeof setTimeout> | undefined;
  const onStop = (): void => {
    if (!stepping) return interruptForStop();
    graceTimer = setTimeout(() => {
      if (stepping) interruptForStop();
    }, ZEOS_STOP_GRACE_MS);
  };

  signal?.addEventListener('abort', onStop, { once: true });
  try {
    const eng = await untilAborted(readyEngine(), signal);
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

    if (!session || session.key !== key) {
      await dropSession();
      store.resetConversation();
      store.setTrust({ gateMode, maskToolChoice });
      const seed = runSeed();
      const run = await eng.open({
        systemPrompt: renderQwenSystemContent(zeosSystemPrompt(system, features), tools),
        gateMode,
        toolClasses: ZEOS_TOOL_CLASSES,
        trustedResults: ZEOS_TRUSTED_RESULTS,
        paramTypes: paramTypesFromTools(tools),
        thinking,
        sampling: sampling(),
        seed,
        thetaRead: thetaRead(),
        maskToolChoice,
      });
      const s: Session = {
        run,
        key,
        segments: new Map(),
        toolCount: 0,
        integrity: TRUSTED,
        demotedBy: null,
        untrusted: false,
      };
      session = s;
      const imported = buildZeosImport(prior);
      if (imported.length > 0) {
        // Start where the stored conversation ended: a demotion is never forgotten.
        const start = importStartIntegrity(prior);
        const events = await run.importHistory(imported, start.integrity);
        s.integrity = start.integrity;
        s.demotedBy = start.demotedBy;
        // Each tools.results arrival is the next imported tool turn.
        const toolTurns = imported.filter((t) => t.role === 'tool');
        for (const e of events) {
          if (e.type === 'arrived' && e.ring >= EXTERNAL) s.untrusted = true;
          if (e.type === 'arrived' && isResultPipe(e.pipe)) {
            const t = toolTurns[s.toolCount];
            s.toolCount += 1;
            s.segments.set(e.segment, `${t?.toolName ?? 'tool'} result #${s.toolCount}`);
          }
        }
      }
      if (import.meta.env.DEV) store.appendJournal(await run.journalLines());
      store.setTrust({ integrity: s.integrity, sessionFloor: null, demotedBy: s.demotedBy });
    }
    const s = session!;

    const turnStart = performance.now();
    await s.run.sendUser(userText);
    const text = new ZeosTurnText(fmt, thinking, emit, emitHistory);
    live = { s, text };
    let calls = 0;
    let sessionFloor: number | null = null;
    /** Batches in a row that moved nothing (`ZEOS_MAX_IDLE_STEPS`). */
    let idleSteps = 0;

    /** A demotion is saved with the message at once, so a Stop right after it cannot lose it. */
    const applyDemotion = (e: Extract<ZeosEvent, { type: 'demoted' }>): void => {
      s.integrity = Math.max(s.integrity, e.to_integrity);
      s.demotedBy = e.because.map((seg) => segmentLabel(seg, s)).join(', ') || s.demotedBy;
      store.setTrust({ integrity: s.integrity, sessionFloor, demotedBy: s.demotedBy });
      reportTrust(s);
    };

    /**
     * Count a call the model made, whatever happens to it next (run, refused
     * by the kernel, or waiting for approval), and end the turn past the cap,
     * before any approval card: refused calls cannot loop without limit (C7).
     */
    const overCap = async (): Promise<boolean> => {
      if (++calls <= MAX_TOOL_CALLS) return false;
      text.flushHeld();
      emit('\n\nReached max tool iterations');
      opts.onMaxIterationsReached?.();
      // The turn ends early: its last steps' attention was never judged.
      assumeCutShortDemoted(s);
      reportTrust(s);
      live = null;
      await dropSession();
      onDone(accumulated);
      return true;
    };
    /** What the next tools.results arrival is, for "Demoted by …". */
    let pendingResultLabel: string | null = null;
    /** What the latest arrival was, for a spoof alarm on it. */
    let lastResultLabel: string | null = null;

    /** A call whose name was chosen masked: its result will be the next `toolRings` entry. */
    const noteMasked = (e: { name: string; name_masked?: boolean; name_hidden?: number[] }): void => {
      if (!e.name_masked) return;
      toolMasked.push(toolRings.length);
      store.noteMasked({ name: e.name, hidden: (e.name_hidden ?? []).map((id) => s.segments.get(id) ?? `segment ${id}`) });
    };

    const runCall = async (
      name: string,
      args: Record<string, unknown>,
      how: ZeosCallHow,
    ): Promise<void> => {
      logToolCall({ name, args, how });
      const argsJson = JSON.stringify(args);
      const result = await dispatch(name, args, how, signal);
      const resultStr = clampToolResultSize(name, JSON.stringify(result));
      text.toolExchange(name, argsJson, resultStr);
      s.toolCount += 1;
      pendingResultLabel = `${name} result #${s.toolCount}`;
      const trusted = isTrustedToolResult(name, args);
      // Ring 3 from the moment it is handed over, before its arrival is seen.
      if (!trusted) s.untrusted = true;
      await s.run.deliverToolResult(toolResultForZeos(resultStr), trusted);
    };

    for (;;) {
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      let events: ZeosEvent[];
      stepping = true;
      try {
        events = await s.run.step(STEP_TICKS);
      } finally {
        stepping = false;
      }
      let changed = false;
      for (const [i, e] of events.entries()) {
        if (signal?.aborted) {
          // The rest of the batch may still say the job was demoted; keep that.
          for (const d of events.slice(i)) {
            if (d.type === 'arrived' && isResultPipe(d.pipe) && pendingResultLabel) {
              s.segments.set(d.segment, pendingResultLabel);
            }
            if (d.type === 'demoted') applyDemotion(d);
          }
          throw new DOMException('Aborted', 'AbortError');
        }
        switch (e.type) {
          case 'token': {
            // `text` may be '': a token that is part of a character the next
            // one completes (ZEOS reads a turn as bytes). It still counts.
            const now = performance.now();
            if (firstTokenAt === null) firstTokenAt = now;
            lastTokenAt = now;
            outputTokens += 1;
            text.token(e.text);
            break;
          }
          case 'arrived':
            if (e.ring >= EXTERNAL) s.untrusted = true;
            // The kernel's rule: a read sets the floor to the pipe's ring, except
            // that in attention mode tool results and history do not set it, and
            // a trusted result (declared `session_floor: false`) never does.
            if (e.pipe === 'chat.user' || (gateMode === 'strict' && e.pipe !== TRUSTED_RESULTS_PIPE)) {
              sessionFloor = e.ring;
            }
            if (isResultPipe(e.pipe)) {
              toolRings.push(e.ring);
              if (pendingResultLabel) s.segments.set(e.segment, pendingResultLabel);
              lastResultLabel = pendingResultLabel;
              pendingResultLabel = null;
              changed = true;
            }
            break;
          case 'demoted':
            applyDemotion(e);
            break;
          case 'tool_call': {
            text.callClosed();
            noteMasked(e);
            await s.run.drain(e.sink);
            if (await overCap()) return;
            await runCall(e.name, e.arguments, e.sink === 'tools.effect' ? 'effect' : 'read');
            break;
          }
          case 'approval_required': {
            text.callClosed();
            noteMasked(e);
            sessionFloor = e.session_floor;
            store.setTrust({ integrity: e.integrity, sessionFloor: e.session_floor, demotedBy: s.demotedBy });
            if (await overCap()) return;
            const approved = await store.requestApproval(
              {
                call: e.call,
                name: e.name,
                args: e.arguments,
                reason: refusalReason(e, gateMode, s.demotedBy),
                integrity: e.integrity,
                sessionFloor: e.session_floor,
                effectiveIntegrity: e.effective_integrity,
              },
              signal,
            );
            if (approved) {
              await runCall(e.name, e.arguments, 'approved');
            } else {
              logToolCall({ name: e.name, args: e.arguments, how: 'denied' });
              const refusal = JSON.stringify({ error: ZEOS_REFUSAL });
              text.toolExchange(e.name, JSON.stringify(e.arguments), refusal);
              s.toolCount += 1;
              pendingResultLabel = `${e.name} refusal #${s.toolCount}`;
              s.untrusted = true;
              await s.run.deliverRefusal(ZEOS_REFUSAL);
            }
            break;
          }
          case 'tool_refused': {
            text.callClosed();
            noteMasked(e);
            if (await overCap()) return;
            const refusal = JSON.stringify({ error: `The kernel refused this call: ${e.detail}` });
            text.toolExchange(e.name, JSON.stringify(e.arguments), refusal);
            s.toolCount += 1;
            pendingResultLabel = `${e.name} refusal #${s.toolCount}`;
            s.untrusted = true;
            await s.run.deliverRefusal(`The kernel refused this call (${e.fault}): ${e.detail}`);
            break;
          }
          case 'reply':
            text.finish();
            break;
          case 'spoof':
            // The kernel alarms right after the read, so it is about the latest result.
            console.warn('[zeos] spoof:', e);
            if (isResultPipe(e.pipe) && toolRings.length > 0) {
              const index = toolRings.length - 1;
              if (!toolSpoofs.includes(index)) toolSpoofs.push(index);
              store.noteSpoof({ pipe: e.pipe, detail: e.detail, label: lastResultLabel });
              changed = true;
            } else {
              store.noteSpoof({ pipe: e.pipe, detail: e.detail, label: null });
            }
            break;
          case 'fault':
            console.warn(`[zeos] ${e.type}:`, e);
            break;
          case 'waiting':
            break;
        }
      }
      if (changed) {
        store.setTrust({ integrity: s.integrity, sessionFloor, demotedBy: s.demotedBy });
        reportTrust(s);
      }
      if (import.meta.env.DEV) store.appendJournal(await s.run.journalLines());
      const waiting = await s.run.waitingOn();
      if (waiting === 'chat.user') break;
      idleSteps = events.some((e) => e.type !== 'waiting') ? 0 : idleSteps + 1;
      if (idleSteps >= ZEOS_MAX_IDLE_STEPS) {
        const stuck = new Error(
          `ZEOS made no progress in ${idleSteps} steps while the job waited on ${waiting ?? 'nothing'}`,
        );
        // As a crash: the status says so, and the next message starts a fresh engine.
        eng.dispose?.(stuck);
        engineDied(eng, stuck);
        throw stuck;
      }
    }

    text.finish();
    live = null;
    store.setTrust({ integrity: s.integrity, sessionFloor, demotedBy: s.demotedBy });
    reportTrust(s);
    // The run now holds exactly the conversation the next request will send.
    s.key = conversationKey(system, thinking, gateMode, maskToolChoice, [
      ...prior,
      last,
      { role: 'assistant', content: history || accumulated },
    ]);
    if (import.meta.env.DEV) {
      const decodeMs = firstTokenAt === null ? 0 : lastTokenAt - firstTokenAt;
      const stats = {
        firstTokenMs: firstTokenAt === null ? null : Math.round(firstTokenAt - turnStart),
        outputTokens,
        decodeTps: decodeMs > 0 ? Number(((outputTokens - 1) / (decodeMs / 1000)).toFixed(2)) : null,
        turnMs: Math.round(performance.now() - turnStart),
        toolCalls: calls,
      };
      console.debug(`[zeos] turn stats ${JSON.stringify(stats)}`);
      await recordAttention(s.run);
    }
    if (onUsage) {
      const decodeMs = firstTokenAt === null ? 0 : lastTokenAt - firstTokenAt;
      onUsage({
        input: 0,
        output: outputTokens,
        tps: decodeMs > 0 ? outputTokens / (decodeMs / 1000) : undefined,
      });
    }
    onDone(accumulated);
  } catch (err) {
    setLlmPreparingToolCall(null);
    // Stop, abort or error mid-turn: save what the turn knew of its trust
    // (a demotion included) with whatever text it kept. The kernel judges a
    // 16-step block's attention only at its end or at a write, so the tokens
    // decoded since were streamed and stored unjudged: with anything
    // untrusted in the run, the turn is saved as demoted (N4).
    if (live) {
      live.text.flushHeld();
      assumeCutShortDemoted(live.s);
      reportTrust(live.s);
      live = null;
    }
    // Mid-turn, the run is waiting somewhere only this loop knew about; the
    // next message replays the stored history into a fresh one instead. Not
    // awaited: after a Stop the close queues behind the interrupted call.
    void dropSession();
    if (isAbortError(err)) {
      onDone(accumulated);
      return;
    }
    onError(err instanceof Error ? err : new Error(String(err)));
  } finally {
    clearTimeout(graceTimer);
    signal?.removeEventListener('abort', onStop);
  }
}

/**
 * A turn that ends any way but its reply (Stop, abort, an error, a crash, a
 * stall, a dispose, the call cap) may have been influenced by what it read
 * without the kernel saying so: a `step` batch the loop never saw the end of
 * may have demoted the job (its events are gone), and the kernel judges the
 * attention of the open 16-step block only at its end or at a write, so the
 * tokens decoded since were never judged. If anything untrusted was in the
 * run, save the turn as demoted rather than risk storing influenced text as
 * trusted (T5, N4); with nothing on ring 3 there was nothing to demote it. A
 * reply needs none of this: `<|im_end|>` is a write, so the kernel judges it.
 */
function assumeCutShortDemoted(s: Session): void {
  if (s.integrity >= EXTERNAL || !s.untrusted) return;
  s.integrity = EXTERNAL;
  s.demotedBy = 'a turn that was cut short (its last steps were never judged)';
  if (session === s) store.setTrust({ integrity: s.integrity, demotedBy: s.demotedBy });
}

/**
 * Dev: with the attention probe on, keep each turn's measurements on
 * `window.__zeosAttention` (an array of logs) and print the block maxima per
 * ring-3 segment.
 */
async function recordAttention(run: ZeosChatRun): Promise<void> {
  if (devSetting('gda.zeos.attentionLog') === null || !run.attentionLog) return;
  const log = await run.attentionLog().catch((err) => {
    console.warn('[zeos] attention probe read failed:', err);
    return null;
  });
  if (!log) return;
  const w = window as unknown as { __zeosAttention?: AttentionLog[] };
  (w.__zeosAttention ??= []).push(log);
  const maxBySeg: Record<string, number> = {};
  for (const b of log.blocks) {
    for (const [id, m] of Object.entries(b.segs)) {
      if (log.segments[id]?.ring === EXTERNAL) maxBySeg[id] = Math.max(maxBySeg[id] ?? 0, m);
    }
  }
  console.debug(
    `[zeos] attention ${JSON.stringify({ steps: log.steps.length, blocks: log.blocks.length, ring3BlockMax: maxBySeg })}`,
  );
}

/**
 * One tool call the loop settled: run from `tools.read` or `tools.effect`
 * (the kernel let it land), run on the user's approval after the kernel
 * refused it, or denied.
 */
export interface ZeosToolLogEntry {
  name: string;
  args: Record<string, unknown>;
  how: ZeosCallHow | 'denied';
}

/** Dev/e2e: every settled tool call, on `window.__zeosToolLog`. */
function logToolCall(entry: ZeosToolLogEntry): void {
  if (!import.meta.env.DEV) return;
  const w = globalThis as { __zeosToolLog?: ZeosToolLogEntry[] };
  (w.__zeosToolLog ??= []).push(entry);
}

/** Which model id the chat is on, for callers that only have the config. */
export function isZeosActive(config: StreamChatOptions['config']): boolean {
  return getLocalGemmaModel(resolveActiveLocalModelIdOrDefault(config))?.family === 'zeos-qwen';
}
