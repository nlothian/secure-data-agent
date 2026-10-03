/**
 * Main-thread RPC client for the in-browser Gemma 4 provider.
 *
 * All inference runs in `workers/llm.worker.ts` (transformers.js on WebGPU);
 * this module owns the single long-lived worker, mints request ids, routes
 * replies, drives the load-progress throbber, and exposes a small
 * promise/callback API to `streamLocalGemma`, `oneShot`, and friends.
 *
 * The worker is disposable: when it wedges (a hung shard download, a decode
 * that ignores cancel, a script crash) `recycleWorker` terminates it, fails
 * everything in flight, and the next call starts a fresh one.
 */

import { setLocalLlmDownloadProgress } from '../executionPanelStore';
import { getLocalGemmaModel } from './models';
import { allFiles, requiredFiles } from './modelFiles';
import { isModelCached } from './modelCache';
import { createLoadProgressAggregator } from './loadProgress';
import { detectWebGpu } from './webgpu';
import type {
  GenerateStats,
  LlmWorkerIn,
  LlmWorkerOut,
} from './llmWorkerProtocol';

/** No `progress` event for this long during the fetch phase ⇒ the download is hung. */
export const LOAD_STALL_TIMEOUT_MS = 60_000;
/** After a cancel mid-decode, the worker must report `done` within this long or it is recycled. */
export const ABORT_WATCHDOG_MS = 15_000;
/**
 * Same, before the first token: the worker can only act on a cancel between
 * 2048-token prefill chunks, and one chunk late in a 20k prompt takes up to
 * ~33 s on E4B (docs/transformers-js-spike.md §2b).
 */
export const PREFILL_ABORT_WATCHDOG_MS = 60_000;

/** Thrown by `generate` when the rendered prompt does not fit the context window. */
export class ContextTooLongError extends Error {
  override readonly name = 'ContextTooLongError';
  constructor(
    public readonly promptTokens: number,
    public readonly limit: number,
  ) {
    super(
      `Prompt is ${promptTokens} tokens; the local model's context window is ${limit}.`,
    );
  }
}

export function isInputTooLongError(err: unknown): boolean {
  return (
    err instanceof ContextTooLongError ||
    (err instanceof Error && err.name === 'ContextTooLongError')
  );
}

// ---- worker plumbing --------------------------------------------------------

type ReplyHandler = (msg: LlmWorkerOut) => void;

let worker: Worker | null = null;
let nextRequestId = 0;
const pending = new Map<number, ReplyHandler>();

function failAllPending(message: string): void {
  const handlers = [...pending.entries()];
  pending.clear();
  for (const [id, handler] of handlers) {
    handler({ type: 'error', id, code: 'generate-failed', message });
  }
}

/**
 * Terminate the current worker and fail every in-flight request with
 * `reason`. The next `ensureLoaded` / `generate` starts from a fresh worker
 * (and has to load the model again).
 */
function recycleWorker(reason: string): void {
  const w = worker;
  worker = null;
  loadedModelId = null;
  if (w) {
    try {
      w.terminate();
    } catch {
      // ignore
    }
  }
  failAllPending(reason);
}

function getWorker(): Worker {
  if (worker) return worker;
  const w = new Worker(new URL('../../workers/llm.worker.ts', import.meta.url), {
    type: 'module',
  });
  w.onmessage = (ev: MessageEvent<LlmWorkerOut>) => {
    if (worker !== w) return; // late reply from a recycled worker
    const msg = ev.data;
    pending.get(msg.id)?.(msg);
  };
  w.onerror = (ev: ErrorEvent) => {
    // The worker script itself failed (e.g. a bundling/import error).
    ev.preventDefault?.();
    if (worker !== w) return;
    recycleWorker(`Local LLM worker crashed: ${ev.message || 'unknown error'}`);
  };
  worker = w;
  return w;
}

function post(msg: LlmWorkerIn): void {
  getWorker().postMessage(msg);
}

function mintId(): number {
  return ++nextRequestId;
}

// ---- load lifecycle ---------------------------------------------------------

let loadedModelId: string | null = null;
let loadingModelId: string | null = null;
let currentLoadPromise: Promise<void> | null = null;
let currentLoadId = 0;
/** Request id of the most recent `load` posted to the worker. */
let lastPostedLoadRequestId = 0;

export function getLoadedModelId(): string | null {
  return loadedModelId;
}

async function assertWebGpuReady(): Promise<void> {
  const gpu = await detectWebGpu();
  if (!gpu.supported) {
    throw new Error(gpu.reason ?? 'WebGPU is not available in this browser.');
  }
  if (gpu.f16 === false) {
    throw new Error(
      'This GPU does not support shader-f16, which the q4f16 Gemma weights require.',
    );
  }
}

export async function ensureLoaded(modelId: string): Promise<void> {
  if (loadedModelId === modelId) return;
  if (currentLoadPromise && loadingModelId === modelId) {
    await currentLoadPromise;
    return;
  }

  const model = getLocalGemmaModel(modelId);
  if (!model) {
    throw new Error(`Unknown local model: ${modelId}`);
  }

  const loadId = ++currentLoadId;
  // The worker disposes whatever it holds before loading the next model.
  loadedModelId = null;
  loadingModelId = modelId;

  currentLoadPromise = (async (): Promise<void> => {
    let stallTimer: ReturnType<typeof setTimeout> | null = null;
    const clearStall = (): void => {
      if (stallTimer !== null) clearTimeout(stallTimer);
      stallTimer = null;
    };
    try {
      await assertWebGpuReady();
      if (loadId !== currentLoadId) throw new Error('Model load superseded.');
      const fromCache = await isModelCached(model);
      if (loadId !== currentLoadId) throw new Error('Model load superseded.');

      const expected = allFiles(model);
      const required = new Set(requiredFiles(model).map((f) => f.path));
      const doneFiles = new Set<string>();
      let initStarted = false;
      const agg = createLoadProgressAggregator({
        label: model.label,
        fromCache,
        expectedFiles: expected,
        onChange: (s) => {
          if (loadId === currentLoadId) setLocalLlmDownloadProgress(s);
        },
      });
      setLocalLlmDownloadProgress(agg.snapshot());

      const requestId = mintId();
      lastPostedLoadRequestId = requestId;
      await new Promise<void>((resolve, reject) => {
        // Fetch-phase watchdog: re-armed on every progress event, disarmed
        // once every file is in (session init can legitimately be silent).
        const armStall = (): void => {
          clearStall();
          stallTimer = setTimeout(() => {
            stallTimer = null;
            if (loadId !== currentLoadId || initStarted) return;
            recycleWorker(
              `Model download stalled: no progress for ${LOAD_STALL_TIMEOUT_MS / 1000} s.`,
            );
          }, LOAD_STALL_TIMEOUT_MS);
        };

        pending.set(requestId, (msg) => {
          switch (msg.type) {
            case 'progress': {
              if (loadId !== currentLoadId) return;
              agg.onEvent(msg.event);
              if (msg.event.status === 'done' && msg.event.file) {
                doneFiles.add(msg.event.file);
              }
              const allRequiredDone = [...required].every((p) => doneFiles.has(p));
              if (!initStarted && (msg.event.status === 'ready' || allRequiredDone)) {
                initStarted = true;
                clearStall();
                agg.beginInit();
              } else if (!initStarted) {
                armStall();
              }
              return;
            }
            case 'loaded': {
              pending.delete(requestId);
              if (loadId !== currentLoadId) {
                // A newer `load` (if any) already replaces this model inside
                // the worker; only free it explicitly when nothing newer was
                // posted, otherwise the dispose would queue behind — and tear
                // down — the newer load.
                if (lastPostedLoadRequestId === requestId) {
                  post({ type: 'dispose', id: mintId() });
                }
                reject(new Error('Model load superseded.'));
                return;
              }
              if (import.meta.env.DEV) {
                const known = new Set(expected.map((f) => f.path));
                const gaps = msg.info.files.filter((f) => !known.has(f));
                if (gaps.length > 0) {
                  console.warn(
                    `[llmService] ${model.hfRepoId} loaded files missing from modelFiles.json:`,
                    gaps,
                  );
                }
              }
              loadedModelId = modelId;
              resolve();
              return;
            }
            case 'error': {
              pending.delete(requestId);
              const current = loadId === currentLoadId;
              // `load-failed` may mean the worker is wedged (see the
              // unhandledrejection hook in llm.worker.ts): its serialised load
              // chain never advances again, so start over with a fresh worker.
              if (current && msg.code === 'load-failed') recycleWorker(msg.message);
              reject(current ? new Error(msg.message) : new Error('Model load superseded.'));
              return;
            }
            default:
              return;
          }
        });
        post({ type: 'load', id: requestId, hfId: model.hfRepoId });
        armStall();
      });
    } finally {
      clearStall();
      if (loadId === currentLoadId) setLocalLlmDownloadProgress(null);
    }
  })();

  try {
    await currentLoadPromise;
  } finally {
    if (loadId === currentLoadId) {
      currentLoadPromise = null;
      loadingModelId = null;
    }
  }
}

export async function dispose(): Promise<void> {
  ++currentLoadId;
  loadedModelId = null;
  loadingModelId = null;
  currentLoadPromise = null;
  setLocalLlmDownloadProgress(null);
  if (!worker) return;
  const id = mintId();
  await new Promise<void>((resolve) => {
    pending.set(id, () => {
      pending.delete(id);
      resolve();
    });
    post({ type: 'dispose', id });
  });
}

// ---- generation -------------------------------------------------------------

export interface GenerateOptions {
  prompt: string;
  signal?: AbortSignal;
  onToken: (delta: string, done: boolean) => void;
  onStats?: (s: GenerateStats) => void;
}

// One generation at a time: the worker rejects a second `generate` with
// `busy`, so callers queue here. Resolves however the prior one ended.
let pendingGeneration: Promise<void> | null = null;
/** Caller-side abort for the in-flight generation (used by `cancel()`). */
let activeGeneration: { id: number; abort: () => void } | null = null;

/**
 * Stream a completion for an already-rendered Gemma prompt. Aborting via
 * `signal` (or `cancel()`) interrupts decode and resolves with the text
 * produced so far — it never rejects for an abort. If the worker does not
 * acknowledge the cancel within `ABORT_WATCHDOG_MS`, the worker is recycled.
 */
export async function generate(opts: GenerateOptions): Promise<string> {
  const { prompt, signal, onToken, onStats } = opts;
  if (signal?.aborted) return '';

  while (pendingGeneration) {
    try {
      await pendingGeneration;
    } catch {
      // ignore — the prior generation's caller already saw its error
    }
  }
  if (signal?.aborted) return '';

  if (!loadedModelId) {
    throw new Error('Local Gemma model is not loaded. Call ensureLoaded() first.');
  }

  let releaseGate: () => void = () => {};
  const gate = new Promise<void>((resolve) => {
    releaseGate = resolve;
  });
  pendingGeneration = gate;

  const id = mintId();
  let abortWatchdog: ReturnType<typeof setTimeout> | null = null;

  try {
    return await new Promise<string>((resolve, reject) => {
      let aggregated = '';
      let aborted = false;
      let callbackError: unknown = null;
      let sawToken = false;

      const requestStop = (): void => {
        post({ type: 'cancel', id });
        if (abortWatchdog !== null) return;
        const ms = sawToken ? ABORT_WATCHDOG_MS : PREFILL_ABORT_WATCHDOG_MS;
        abortWatchdog = setTimeout(() => {
          abortWatchdog = null;
          if (!pending.has(id)) return;
          recycleWorker(`Local model did not stop within ${ms / 1000} s; restarting it.`);
        }, ms);
      };
      const onAbort = (): void => {
        if (aborted) return;
        aborted = true;
        requestStop();
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      activeGeneration = { id, abort: onAbort };

      pending.set(id, (msg) => {
        switch (msg.type) {
          case 'token': {
            sawToken = true;
            if (aborted || callbackError !== null) return;
            aggregated += msg.text;
            try {
              onToken(msg.text, false);
            } catch (err) {
              callbackError = err;
              requestStop();
            }
            return;
          }
          case 'done': {
            pending.delete(id);
            signal?.removeEventListener('abort', onAbort);
            if (import.meta.env.DEV) {
              // KV-reuse visibility for headed e2e / manual runs: watch
              // `reusedTokens` (0 = miss) and `prefillTokens` per generation.
              console.debug('[llmService] generate stats ' + JSON.stringify(msg.stats));
            }
            if (callbackError !== null) {
              reject(
                callbackError instanceof Error ? callbackError : new Error(String(callbackError)),
              );
              return;
            }
            if (msg.stats.reason === 'interrupted' && !aborted) {
              // Nobody on this side asked to stop — the worker interrupted us
              // to switch or dispose the model. Don't pass truncated text off
              // as a complete answer.
              reject(new Error('Generation interrupted by a model switch'));
              return;
            }
            onStats?.(msg.stats);
            resolve(aggregated);
            return;
          }
          case 'error': {
            pending.delete(id);
            signal?.removeEventListener('abort', onAbort);
            // An ONNX Runtime failure leaves the worker's runtime unusable
            // (even dispose + reload fails in-process): start over with a
            // fresh worker; the next ensureLoaded reloads the model.
            if (
              msg.code === 'generate-failed' &&
              (msg.data as { fatal?: boolean } | undefined)?.fatal === true
            ) {
              recycleWorker(msg.message);
            }
            if (callbackError !== null) {
              reject(
                callbackError instanceof Error ? callbackError : new Error(String(callbackError)),
              );
            } else if (aborted) {
              // Abort semantics: keep the partial text even if the worker had
              // to be recycled to stop it.
              resolve(aggregated);
            } else if (msg.code === 'context-too-long') {
              const d = (msg.data ?? {}) as { promptTokens?: number; limit?: number };
              reject(new ContextTooLongError(d.promptTokens ?? 0, d.limit ?? 0));
            } else {
              reject(new Error(msg.message));
            }
            return;
          }
          default:
            // e.g. `prefill` progress — not surfaced yet.
            return;
        }
      });

      post({ type: 'generate', id, prompt });
    });
  } finally {
    if (abortWatchdog !== null) clearTimeout(abortWatchdog);
    if (activeGeneration?.id === id) activeGeneration = null;
    if (pendingGeneration === gate) pendingGeneration = null;
    releaseGate();
  }
}

/** Interrupt the in-flight generation, if any (same semantics as aborting its signal). */
export function cancel(): void {
  activeGeneration?.abort();
}

/**
 * Token count of `text` (including the leading `<bos>`) under the loaded
 * model's tokenizer, or `null` when no model is loaded or counting failed.
 */
export async function sizeInTokens(text: string): Promise<number | null> {
  if (!loadedModelId) return null;
  const id = mintId();
  return await new Promise<number | null>((resolve) => {
    pending.set(id, (msg) => {
      pending.delete(id);
      resolve(msg.type === 'count' && Number.isFinite(msg.tokens) ? msg.tokens : null);
    });
    post({ type: 'count', id, text });
  });
}
