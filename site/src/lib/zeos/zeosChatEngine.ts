/**
 * The ZEOS chat as `streamZeos` drives it: one kernel worker + one model
 * thread per page (`startZeos`), and a `ChatRun` (ZEOS
 * `zeos_coop_count_web.chat`) per conversation, reached through the kernel
 * worker's generic RPC. `streamZeos` only sees the `ZeosChatEngine` /
 * `ZeosChatRun` interfaces, so its tests swap in a scripted engine.
 */
import type { LocalGemmaModel } from '../localLlm/models';
import { assertCrossOriginIsolated, modelRef, startZeos, type ZeosHandle, type ZeosKernel } from './zeosHost';
import { createZeosLoadProgress, zeosModelThreadFor } from './zeosModelWorker';
import type { ToolClassEntry, TrustedResultRule } from './zeosToolClasses';
import type { ZeosImportTurn } from './zeosHistory';
import { installAttentionProbe, readAttentionLog, type AttentionLog } from './attentionProbe';
import { detectWebGpu } from '../localLlm/webgpu';

/** A segment as `ChatRun.segment_info` reports it. */
export interface ZeosSegment {
  segment: number;
  pipe: string;
  principal: string;
  tag: string;
  ring: number;
  integrity: number;
  tokens: number;
  resident: boolean;
  injected_at: number;
}

interface CallFields {
  call: number;
  name: string;
  arguments: Record<string, unknown>;
  sink: string;
  /** Where the job reads the answer: `tools.results`, or `tools.results.trusted`. */
  results: string;
  /** The name was chosen with the ring-3 deliveries hidden (`mask_tool_choice`). */
  name_masked?: boolean;
  /** The segment ids hidden while the name was chosen. */
  name_hidden?: number[];
}

interface RefusalFields extends CallFields {
  fault: string;
  detail: string;
  integrity: number;
  effective_integrity: number | null;
  session_floor: number | null;
}

/** `ChatRun.step` events (see ZEOS chat.py). */
export type ZeosEvent =
  | { type: 'token'; text: string }
  | ({ type: 'tool_call' } & CallFields)
  | ({ type: 'approval_required' } & RefusalFields)
  | ({ type: 'tool_refused' } & RefusalFields)
  | { type: 'reply'; text: string; reasoning: string | null; raw: string }
  | { type: 'arrived'; pipe: string; segment: number; ring: number; integrity: number }
  | { type: 'demoted'; from_integrity: number; to_integrity: number; because: ZeosSegment[] }
  | { type: 'spoof'; pipe: string | null; detail: string }
  | { type: 'fault'; fault: string; detail: string; pipe: string | null }
  | { type: 'waiting'; pipe: string };

/**
 * How a tool result gates effects (ZEOS `open_chat(gate_mode=…)`). `strict`:
 * reading one sets the session floor to 3 until the next user message, so
 * any effect after it needs approval. `attention`: only the integrity
 * watermark (measured attention past theta_read) refuses effects.
 */
export type ZeosGateMode = 'strict' | 'attention';

/**
 * Seeded sampling (ZEOS `Sampling`): every step draws `u` from a
 * `random.Random` seeded by the run's seed, so a run replays exactly. `null`
 * is greedy.
 */
export interface ZeosSampling {
  temperature: number;
  topK: number;
}

export interface ZeosChatOpenOptions {
  systemPrompt: string;
  gateMode: ZeosGateMode;
  toolClasses: Readonly<Record<string, ToolClassEntry>>;
  /** Calls whose results the app wrote itself (`open_chat(trusted_results=…)`). */
  trustedResults?: Readonly<Record<string, TrustedResultRule>>;
  paramTypes: Record<string, Record<string, string>>;
  thinking: boolean;
  /** `null` (the default) is greedy. */
  sampling?: ZeosSampling | null;
  /** The kernel's seed (`KernelConfig.seed`), which also seeds sampling. */
  seed?: number;
  /** `KernelConfig.theta_read`: a tool result's mass over one block that demotes. */
  thetaRead?: number;
  /** Hide ring-3 deliveries while the model writes a tool's name (`open_chat(mask_tool_choice=…)`). */
  maskToolChoice?: boolean;
}

export interface ZeosChatRun {
  /**
   * Replay stored turns into a fresh run (ZEOS `import_history`).
   * `startIntegrity` is the job's integrity before the replay: 3 when the
   * stored conversation had been demoted, so the replay cannot forget it.
   */
  importHistory(turns: readonly ZeosImportTurn[], startIntegrity?: number): Promise<ZeosEvent[]>;
  sendUser(text: string): Promise<void>;
  step(ticks: number): Promise<ZeosEvent[]>;
  waitingOn(): Promise<string | null>;
  drain(pipe: string): Promise<string[]>;
  /** `trusted`: on `tools.results.trusted` (ring 2); it must match the call's `results`. */
  deliverToolResult(text: string, trusted?: boolean): Promise<void>;
  deliverRefusal(text?: string): Promise<void>;
  journalLines(): Promise<string[]>;
  /** Dev measurement: per-step and per-block attention on segments (./attentionProbe.ts). */
  attentionLog?(): Promise<AttentionLog>;
  close(): Promise<void>;
}

export interface ZeosChatEngine {
  backend: string;
  stub: boolean;
  open(opts: ZeosChatOpenOptions): Promise<ZeosChatRun>;
  /** Terminate the kernel worker and the model thread (frees the model's GPU memory). Idempotent. */
  dispose?(reason?: Error): void;
  /** `listener` runs once when the engine dies: a crash, a fatal call, or `dispose`. */
  onDispose?(listener: (reason: Error) => void): () => void;
  /** Reject every call still waiting for its reply, now (Stop); the calls run on. */
  interrupt?(reason: Error): void;
  /** Resolves once every call sent before it has finished. */
  ping?(): Promise<void>;
}

/**
 * How long one synchronous model call may block the kernel before the
 * channel gives up (ZEOS `SyncModelWorker` `timeoutMs`; its default is 600 s).
 * The longest legitimate call is one prefill: the first turn's ~6.8k-token
 * prompt takes ~33 s on an M1 Max (~210 tok/s), and one `decodeStep` may
 * carry all of it. 180 s leaves ~5x that, room for a slower GPU or a longer
 * replayed history (a 16k-token replay is ~80 s at that rate), while a wedged
 * model thread is reported in 3 minutes instead of 10. A timeout disposes the
 * engine (`isFatalKernelError`), so the next message starts a fresh one.
 * Stop does not wait for it: `interrupt` rejects the pending call at once.
 */
export const ZEOS_MODEL_CALL_TIMEOUT_MS = 180_000;

class KernelChatRun implements ZeosChatRun {
  constructor(
    private readonly kernel: ZeosKernel,
    private readonly run: ZeosHandle,
  ) {}

  private m<T>(method: string, ...args: unknown[]): Promise<T> {
    return this.kernel.callMethod<T>(this.run, method, args);
  }

  importHistory(turns: readonly ZeosImportTurn[], startIntegrity = 2) {
    // `trusted` is always a real bool and `integrity` an int from 0 to 3 (ZEOS
    // refuses anything else). A trusted result carries its call, `name` and
    // `arguments`, which the kernel checks against the trusted-results table
    // exactly; one without its call replays on ring 3.
    const entries = turns.map((t) => {
      if (t.role === 'assistant') return { role: t.role, text: t.text, integrity: t.integrity ?? 3 };
      if (t.role !== 'tool') return { role: t.role, text: t.text };
      if (t.trusted === true && t.toolName !== undefined && t.toolArgs !== undefined) {
        return { role: t.role, text: t.text, trusted: true, name: t.toolName, arguments: t.toolArgs };
      }
      return { role: t.role, text: t.text, trusted: false };
    });
    // Only a demoted start is passed, so a wheel without `start_integrity`
    // still replays an undemoted conversation.
    return this.kernel.callMethod<ZeosEvent[]>(
      this.run,
      'import_history',
      [entries],
      startIntegrity > 2 ? { start_integrity: startIntegrity } : undefined,
    );
  }
  sendUser(text: string) {
    return this.m<void>('send_user', text);
  }
  step(ticks: number) {
    return this.m<ZeosEvent[]>('step', ticks);
  }
  waitingOn() {
    return this.m<string | null>('waiting_on');
  }
  drain(pipe: string) {
    return this.m<string[]>('drain', pipe);
  }
  deliverToolResult(text: string, trusted = false) {
    return this.kernel.callMethod<void>(this.run, 'deliver_tool_result', [text], { trusted });
  }
  deliverRefusal(text?: string) {
    return text === undefined ? this.m<void>('deliver_refusal') : this.m<void>('deliver_refusal', text);
  }
  journalLines() {
    return this.m<string[]>('journal_lines');
  }
  attentionLog() {
    return readAttentionLog(this.kernel, this.run);
  }
  async close() {
    try {
      await this.m<void>('close');
    } finally {
      await this.kernel.release(this.run).catch(() => undefined);
    }
  }
}

export interface StartEngineHooks {
  onStatus?: (text: string) => void;
  onProgress?: (progress: Record<string, unknown>) => void;
  /** Aborting stops the start and terminates both workers (`disposeZeos` mid-start). */
  signal?: AbortSignal;
}

/** Boot the kernel worker and the model thread for `model`, in parallel. */
export async function startKernelChatEngine(
  model: LocalGemmaModel,
  hooks: StartEngineHooks = {},
): Promise<ZeosChatEngine> {
  // Production does not send COOP/COEP yet (CLAUDE.md, "Cross-origin
  // isolation"): say so before anything downloads, rather than hang.
  assertCrossOriginIsolated();
  const thread = zeosModelThreadFor(model);
  if (!thread.stub) {
    // No WebAssembly fallback for this export: fail before booting Pyodide.
    const gpu = await detectWebGpu();
    if (!gpu.supported) {
      throw new Error(
        `${model.label} needs WebGPU, which is not available here: ${gpu.reason ?? 'no adapter'}. ` +
          'It has no WebAssembly fallback.',
      );
    }
    if (gpu.f16 === false) {
      throw new Error(`${model.label} needs a GPU with shader-f16 for its q4f16 weights.`);
    }
  }
  const progress = thread.stub ? null : createZeosLoadProgress(model);
  const began = performance.now();
  let started;
  try {
    started = await startZeos({
      signal: hooks.signal,
      onStatus: hooks.onStatus,
      onLog: (stream, text) => {
        if (stream === 'stderr') console.warn('[zeos]', text);
      },
      model: {
        modelWorker: thread.modelWorker,
        init: thread.init,
        onProgress: (p) => {
          progress?.onProgress(p);
          hooks.onProgress?.(p);
        },
        onActivity: import.meta.env.DEV ? recordActivity : undefined,
        timeoutMs: ZEOS_MODEL_CALL_TIMEOUT_MS,
      },
    });
  } finally {
    progress?.done();
  }
  const { kernel, model: attached } = started;
  if (import.meta.env.DEV) {
    console.debug(`[zeos] ${thread.label} ready in ${Math.round(performance.now() - began)} ms`);
    // DevTools: `await __zeosKernel.exec('…')` runs Python in the kernel worker.
    (globalThis as { __zeosKernel?: ZeosKernel }).__zeosKernel = kernel;
  }
  const probe = import.meta.env.DEV && attentionProbeEnabled();
  try {
    if (probe) await installAttentionProbe(kernel);
    if (hooks.signal?.aborted) throw hooks.signal.reason ?? new DOMException('Aborted', 'AbortError');
  } catch (err) {
    kernel.dispose(err instanceof Error ? err : new Error(String(err)));
    throw err;
  }
  return {
    backend: attached.backend,
    stub: thread.stub,
    dispose: (reason) => kernel.dispose(reason),
    onDispose: (listener) => kernel.onDispose(listener),
    interrupt: (reason) => kernel.interrupt(reason),
    ping: async () => {
      await kernel.exec('None');
    },
    async open(opts) {
      const sampling = opts.sampling
        ? await kernel.call<ZeosHandle>('zeos_coop_count_web.chat_machine', 'Sampling', [], {
            temperature: opts.sampling.temperature,
            top_k: opts.sampling.topK,
          })
        : null;
      let run: ZeosHandle;
      try {
        run = await kernel.call<ZeosHandle>(
          'zeos_coop_count_web.chat',
          'open_chat',
          [modelRef(attached.name)],
          {
            tool_classes: opts.toolClasses,
            system_prompt: opts.systemPrompt,
            thinking: opts.thinking,
            param_types: opts.paramTypes,
            gate_mode: opts.gateMode,
            ...(opts.trustedResults ? { trusted_results: opts.trustedResults } : {}),
            ...(sampling ? { sampling } : {}),
            ...(opts.seed !== undefined ? { seed: opts.seed } : {}),
            ...(opts.thetaRead !== undefined ? { theta_read: opts.thetaRead } : {}),
            ...(opts.maskToolChoice ? { mask_tool_choice: true } : {}),
          },
        );
      } finally {
        // Released on an open_chat error too.
        if (sampling) await kernel.release(sampling).catch(() => undefined);
      }
      if (probe) await kernel.call('_zeos_attention_probe', 'attach', [run]);
      return new KernelChatRun(kernel, run);
    },
  };
}

/**
 * Dev: the model thread's per-run timings (`{phase, count, length, ms,
 * track, hidden}` from OptZeosWorker's `onActivity`), on
 * `window.__zeosActivity`, so the time a decode step spends in the graph can
 * be told from the kernel's and the channel's share. `track` numbers the
 * cache the run used (a masked tool name runs on a second one), `hidden` is
 * how many positions its mask hid, and phase `skip` is a hidden run carried
 * past without running it.
 */
function recordActivity(a: Record<string, unknown>): void {
  if (typeof a.ms !== 'number') return;
  const w = globalThis as { __zeosActivity?: Record<string, unknown>[] };
  const log = (w.__zeosActivity ??= []);
  log.push({
    phase: a.phase,
    count: a.count,
    length: a.length,
    track: a.track,
    hidden: a.hidden,
    ms: Math.round(a.ms as number),
    at: Math.round(performance.now()),
  });
  if (log.length > 5000) log.splice(0, log.length - 5000);
}

/** Dev: localStorage `gda.zeos.attentionLog` turns on the attention probe. */
function attentionProbeEnabled(): boolean {
  try {
    return localStorage.getItem('gda.zeos.attentionLog') != null;
  } catch {
    return false;
  }
}
