/**
 * Page-thread host for the ZEOS kernel (a TypeScript port of ZEOS
 * `model_host.js` / `startBrowserModel`, generalised).
 *
 * Two workers, both started here on the page thread:
 *
 * - the **kernel worker** (src/workers/zeosKernel.worker.ts): Pyodide 314 +
 *   the ZEOS wheels, driven by a generic RPC (`call`, `callMethod`, `exec`…);
 * - a **model thread**: any module worker speaking the model-thread protocol
 *   (`ModelThreadMessage` in ./protocol.ts) — the stub
 *   (src/workers/zeosStubModel.worker.ts) today, an OPT ZEOS worker later.
 *
 * The page creates the SharedArrayBuffer + MessageChannel, hands the model
 * thread `{buffer, port1, ...init}` and the kernel worker `{buffer, port2}`.
 * The kernel then calls the model synchronously (Atomics.wait) without the
 * page in the loop. The page starts the model thread rather than the kernel
 * worker because nested workers failed to start in Chromium.
 *
 * Requires a cross-origin isolated page (COOP/COEP; see astro.config.mjs).
 *
 * Typical use:
 *
 *   const kernel = await startZeosKernel({ onEvent });
 *   const model = await kernel.attachModel({ modelWorker: createStubModelWorker, init: { tapes } });
 *   const run = await kernel.call('zeos_coop_count_web.page', 'open_run',
 *     [kernel.caseDir('coop-count-pipe'), 'js'], { worker: modelRef(model.name) });
 *   const lines = await kernel.callMethod(run, 'step');
 *   kernel.dispose();
 */
import { CHANNEL_BYTES } from './vendor/model_channel.js';
import {
  type BootInfo,
  type KernelError,
  type KernelMessage,
  type KernelRequest,
  type ModelThreadMessage,
  type ZeosHandle,
} from './protocol';

export { modelRef, isHandle } from './protocol';
export type { BootInfo, ZeosHandle, ZeosModelRef } from './protocol';

/** A Python exception (or worker failure) surfaced on the page. */
export class ZeosKernelError extends Error {
  readonly pythonType: string;
  readonly traceback?: string;
  constructor(op: string, err: KernelError) {
    super(`${op}: ${err.name}: ${err.message}`);
    this.name = 'ZeosKernelError';
    this.pythonType = err.name;
    this.traceback = err.traceback;
  }
}

export class NotCrossOriginIsolatedError extends Error {
  constructor() {
    super(
      'This page is not cross-origin isolated, so SharedArrayBuffer is unavailable and the ' +
        'ZEOS kernel cannot call its model. Serve it with ' +
        '`Cross-Origin-Opener-Policy: same-origin` and ' +
        '`Cross-Origin-Embedder-Policy: credentialless` (astro.config.mjs server.headers, ' +
        'public/_headers).',
    );
    this.name = 'NotCrossOriginIsolatedError';
  }
}

export function assertCrossOriginIsolated(): void {
  if (!globalThis.crossOriginIsolated || typeof SharedArrayBuffer === 'undefined') {
    throw new NotCrossOriginIsolatedError();
  }
}

/** A model-thread module: a URL to start as a module worker, or a factory. */
export type ModelWorkerSource = string | URL | (() => Worker);

export interface AttachModelOptions {
  modelWorker: ModelWorkerSource;
  /** Name Python sees it under (`modelRef(name)`, `_zeos_rpc.model(name)`). */
  name?: string;
  /** Extra fields for the model thread's first message (model URL, tapes, …). */
  init?: Record<string, unknown>;
  /** Transferables inside `init`. */
  transfer?: Transferable[];
  /** How long one synchronous model call may block the kernel. */
  timeoutMs?: number;
  onProgress?: (progress: Record<string, unknown>) => void;
  onActivity?: (activity: Record<string, unknown>) => void;
  signal?: AbortSignal;
}

export interface StartedModelThread {
  buffer: SharedArrayBuffer;
  /** The kernel's end of the request channel (transferred on attach). */
  port: MessagePort;
  backend: string;
  thread: Worker;
}

export interface AttachedModel {
  name: string;
  backend: string;
  thread: Worker;
}

export interface StartKernelOptions {
  /** Absolute or page-relative URL of `manifest.json`; default `<BASE_URL>zeos/manifest.json`. */
  manifestUrl?: string;
  /** Python `_zeos_rpc.emit(kind, data)`. */
  onEvent?: (kind: string, data: unknown) => void;
  onLog?: (stream: 'stdout' | 'stderr', text: string) => void;
  onStatus?: (text: string) => void;
  signal?: AbortSignal;
  /** Override the kernel worker (tests). */
  kernelWorker?: () => Worker;
}

type Pending = { op: string; resolve: (v: unknown) => void; reject: (e: Error) => void };

export class ZeosKernel {
  readonly worker: Worker;
  boot!: BootInfo;
  private next = 1;
  private pending = new Map<number, Pending>();
  private models = new Map<string, AttachedModel>();
  private disposed: Error | null = null;
  private readonly opts: StartKernelOptions;

  constructor(worker: Worker, opts: StartKernelOptions) {
    this.worker = worker;
    this.opts = opts;
    worker.onmessage = (e: MessageEvent<KernelMessage>) => this.onMessage(e.data);
    worker.onerror = (e: ErrorEvent) => {
      e.preventDefault();
      this.fail(new Error(`ZEOS kernel worker error: ${e.message ?? 'unknown'} (${e.filename ?? '?'}:${e.lineno ?? '?'})`));
    };
    worker.onmessageerror = () => this.fail(new Error('ZEOS kernel worker: message could not be deserialised'));
  }

  private onMessage(msg: KernelMessage) {
    switch (msg.type) {
      case 'reply': {
        const p = this.pending.get(msg.id);
        if (!p) return;
        this.pending.delete(msg.id);
        if (msg.ok) p.resolve(msg.value);
        else p.reject(new ZeosKernelError(p.op, msg.error));
        return;
      }
      case 'event':
        this.opts.onEvent?.(msg.kind, msg.data);
        return;
      case 'log':
        this.opts.onLog?.(msg.stream, msg.text);
        return;
      case 'status':
        this.opts.onStatus?.(msg.text);
        return;
    }
  }

  private fail(err: Error) {
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
    this.dispose(err);
  }

  /** Send one raw request. Prefer the typed helpers below. */
  request<T = unknown>(req: KernelRequest, transfer: Transferable[] = []): Promise<T> {
    if (this.disposed) return Promise.reject(this.disposed);
    const id = this.next++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { op: req.op, resolve: resolve as (v: unknown) => void, reject });
      this.worker.postMessage({ ...req, id }, transfer);
    });
  }

  /** `module.fn(*args, **kwargs)` in Python; `fn` may be dotted. Non-JSON results come back as handles. */
  call<T = unknown>(module: string, fn: string, args: unknown[] = [], kwargs: Record<string, unknown> = {}): Promise<T> {
    return this.request<T>({ op: 'call', module, fn, args, kwargs });
  }

  callMethod<T = unknown>(
    handle: ZeosHandle,
    method: string,
    args: unknown[] = [],
    kwargs: Record<string, unknown> = {},
  ): Promise<T> {
    return this.request<T>({ op: 'callMethod', handle: handle.$handle, method, args, kwargs });
  }

  getAttr<T = unknown>(handle: ZeosHandle, name: string): Promise<T> {
    return this.request<T>({ op: 'getAttr', handle: handle.$handle, name });
  }

  /** Run Python source in a persistent namespace; resolves to the trailing expression's value. */
  exec<T = unknown>(source: string): Promise<T> {
    return this.request<T>({ op: 'exec', source });
  }

  release(handle: ZeosHandle): Promise<void> {
    return this.request<void>({ op: 'release', handle: handle.$handle });
  }

  /** Path of case `name` inside Pyodide's FS. */
  caseDir(name: string): string {
    return `${this.boot.casesDir}/${name}`;
  }

  /** Start a model thread on the page and attach it to the kernel. */
  async attachModel(opts: AttachModelOptions): Promise<AttachedModel> {
    const name = opts.name ?? 'default';
    if (this.models.has(name)) throw new Error(`a model is already attached as ${name}`);
    return this.attachThread(name, await startModelThread(opts), opts.timeoutMs);
  }

  /** Hand an already-started model thread to the kernel. Terminates it on failure. */
  async attachThread(
    name: string,
    started: StartedModelThread,
    timeoutMs?: number,
  ): Promise<AttachedModel> {
    const { buffer, port, backend, thread } = started;
    try {
      await this.request({ op: 'attachModel', name, buffer, port, timeoutMs }, [port]);
    } catch (err) {
      thread.terminate();
      throw err;
    }
    const model = { name, backend, thread };
    this.models.set(name, model);
    return model;
  }

  /** Terminate the kernel worker and every model thread. Idempotent. */
  dispose(reason: Error = new Error('ZEOS kernel disposed')): void {
    if (this.disposed) return;
    this.disposed = reason;
    for (const p of this.pending.values()) p.reject(reason);
    this.pending.clear();
    this.worker.terminate();
    for (const m of this.models.values()) m.thread.terminate();
    this.models.clear();
  }
}

function defaultManifestUrl(): string {
  return new URL(`${import.meta.env.BASE_URL ?? '/'}zeos/manifest.json`, location.href).href;
}

function abortError(signal: AbortSignal): Error {
  return signal.reason instanceof Error ? signal.reason : new DOMException('Aborted', 'AbortError');
}

/** Start the kernel worker and boot it (Pyodide, wheels, cases). */
export async function startZeosKernel(opts: StartKernelOptions = {}): Promise<ZeosKernel> {
  assertCrossOriginIsolated();
  const worker =
    opts.kernelWorker?.() ??
    new Worker(new URL('../../workers/zeosKernel.worker.ts', import.meta.url), { type: 'module' });
  const kernel = new ZeosKernel(worker, opts);
  const onAbort = () => kernel.dispose(abortError(opts.signal!));
  opts.signal?.addEventListener('abort', onAbort, { once: true });
  try {
    if (opts.signal?.aborted) throw abortError(opts.signal);
    kernel.boot = await kernel.request<BootInfo>({
      op: 'boot',
      manifestUrl: new URL(opts.manifestUrl ?? defaultManifestUrl(), location.href).href,
    });
    return kernel;
  } catch (err) {
    kernel.dispose(err as Error);
    throw err;
  } finally {
    opts.signal?.removeEventListener('abort', onAbort);
  }
}

/**
 * Start a model thread and wait for `{ready}`. Returns what the kernel needs
 * to build a `SyncModelWorker` (the shared buffer and the request port).
 */
export async function startModelThread(opts: AttachModelOptions): Promise<StartedModelThread> {
  assertCrossOriginIsolated();
  const src = opts.modelWorker;
  const thread = typeof src === 'function' ? src() : new Worker(src, { type: 'module' });
  const buffer = new SharedArrayBuffer(CHANNEL_BYTES);
  const channel = new MessageChannel();
  let onAbort: (() => void) | undefined;
  try {
    const backend = await new Promise<string>((resolve, reject) => {
      thread.onmessage = (event: MessageEvent<ModelThreadMessage>) => {
        const data = event.data;
        if ('progress' in data) opts.onProgress?.(data.progress);
        else if ('activity' in data) opts.onActivity?.(data.activity);
        else if (data.ready) resolve(data.backend);
        else reject(new Error(`model thread failed to load: ${data.error}`));
      };
      thread.onerror = (event: ErrorEvent) => {
        event.preventDefault();
        reject(
          new Error(
            `model thread failed to start: ${event.message ?? 'no message'} (${event.filename ?? '?'}:${event.lineno ?? '?'})`,
          ),
        );
      };
      if (opts.signal) {
        if (opts.signal.aborted) return reject(abortError(opts.signal));
        onAbort = () => reject(abortError(opts.signal!));
        opts.signal.addEventListener('abort', onAbort, { once: true });
      }
      thread.postMessage({ ...opts.init, buffer, port: channel.port1 }, [channel.port1, ...(opts.transfer ?? [])]);
    });
    // Keep forwarding activity/progress after ready.
    thread.onmessage = (event: MessageEvent<ModelThreadMessage>) => {
      const data = event.data;
      if ('progress' in data) opts.onProgress?.(data.progress);
      else if ('activity' in data) opts.onActivity?.(data.activity);
    };
    return { buffer, port: channel.port2, backend, thread };
  } catch (err) {
    thread.terminate();
    throw err;
  } finally {
    if (onAbort) opts.signal?.removeEventListener('abort', onAbort);
  }
}

/**
 * Kernel + one model, started in parallel (Pyodide boot and model load are
 * both slow). The model is attached as `opts.model.name ?? 'default'`.
 */
export async function startZeos(
  opts: StartKernelOptions & { model: AttachModelOptions },
): Promise<{ kernel: ZeosKernel; model: AttachedModel }> {
  const [kernelResult, threadResult] = await Promise.allSettled([
    startZeosKernel(opts),
    startModelThread({ ...opts.model, signal: opts.model.signal ?? opts.signal }),
  ]);
  if (kernelResult.status === 'rejected' || threadResult.status === 'rejected') {
    if (kernelResult.status === 'fulfilled') kernelResult.value.dispose();
    if (threadResult.status === 'fulfilled') threadResult.value.thread.terminate();
    throw kernelResult.status === 'rejected' ? kernelResult.reason : (threadResult as PromiseRejectedResult).reason;
  }
  const kernel = kernelResult.value;
  try {
    const model = await kernel.attachThread(opts.model.name ?? 'default', threadResult.value, opts.model.timeoutMs);
    return { kernel, model };
  } catch (err) {
    kernel.dispose(err as Error);
    throw err;
  }
}

/** The stub model thread (ZEOS stub_worker.js over the channel), for tests and smoke runs. */
export function createStubModelWorker(): Worker {
  return new Worker(new URL('../../workers/zeosStubModel.worker.ts', import.meta.url), { type: 'module' });
}
