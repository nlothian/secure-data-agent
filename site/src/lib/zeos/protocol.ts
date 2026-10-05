/**
 * Message shapes between the page (zeosHost.ts) and the ZEOS kernel worker
 * (src/workers/zeosKernel.worker.ts), and between the page and a model thread.
 */

/** A Python object the kernel worker holds for the page. */
export interface ZeosHandle {
  $handle: number;
  type: string;
}

/** Marker that resolves, in Python, to the SyncModelWorker attached as `name`. */
export interface ZeosModelRef {
  $model: string;
}

export function modelRef(name = 'default'): ZeosModelRef {
  return { $model: name };
}

export function isHandle(v: unknown): v is ZeosHandle {
  return typeof v === 'object' && v !== null && typeof (v as ZeosHandle).$handle === 'number';
}

export interface BootInfo {
  pyodide: string;
  python: string;
  zeos: { branch: string; commit: string; dirty: boolean };
  wheels: string[];
  cases: string[];
  /** Directory in Pyodide's FS that holds case `name`: `${casesDir}/${name}`. */
  casesDir: string;
  isolated: boolean;
}

export type KernelRequest =
  | { op: 'boot'; manifestUrl: string }
  | { op: 'attachModel'; name: string; buffer: SharedArrayBuffer; port: MessagePort; timeoutMs?: number }
  | { op: 'exec'; source: string }
  | { op: 'call'; module: string; fn: string; args: unknown[]; kwargs: Record<string, unknown> }
  | { op: 'callMethod'; handle: number; method: string; args: unknown[]; kwargs: Record<string, unknown> }
  | { op: 'getAttr'; handle: number; name: string }
  | { op: 'release'; handle: number };

export interface KernelError {
  name: string;
  message: string;
  traceback?: string;
}

export type KernelMessage =
  | { type: 'reply'; id: number; ok: true; value: unknown }
  | { type: 'reply'; id: number; ok: false; error: KernelError }
  /** Python `_zeos_rpc.emit(kind, data)`. */
  | { type: 'event'; kind: string; data: unknown }
  | { type: 'log'; stream: 'stdout' | 'stderr'; text: string }
  | { type: 'status'; text: string };

/**
 * Model-thread protocol (same as ZEOS `model_thread.js`). The first message
 * the thread receives is `{ buffer, port, ...init }` with `port` transferred;
 * every later request arrives on `port` and is answered into `buffer` by
 * `serveChannel`. The thread reports back on its own `postMessage`.
 */
export type ModelThreadMessage =
  | { progress: Record<string, unknown> }
  | { activity: Record<string, unknown> }
  | { ready: true; backend: string }
  | { ready: false; error: string };
