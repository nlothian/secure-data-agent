/// <reference lib="webworker" />
/**
 * The ZEOS kernel worker: Pyodide 314 (separate from the app's Pyodide 0.29
 * RunPython worker in src/lib/pyodide.ts), the `zeos` and
 * `zeos-coop-count-web` wheels, and the case files, all in one module worker.
 * The page drives it through src/lib/zeos/zeosHost.ts.
 *
 * It is a module worker because Pyodide 314 refuses to load in a classic one.
 * The model runs on a separate thread the page starts (a worker started from
 * inside a worker failed to start in Chromium; see ZEOS model_host.js); the
 * page hands over that thread's SharedArrayBuffer + MessagePort with
 * `attachModel`, and Python calls it synchronously through `SyncModelWorker`,
 * blocking in `Atomics.wait`.
 *
 * RPC: each request is `{ id, op, ... }` (see KernelRequest in
 * src/lib/zeos/protocol.ts) and gets one `{ type: 'reply', id, ok, ... }`.
 * Requests run one at a time, in order. Between replies the worker may post
 * `event` (Python `_zeos_rpc.emit`), `log` (stdout/stderr) and `status`
 * messages; they are posted synchronously, so events emitted while a long
 * Python call (e.g. a kernel step that decodes) is running reach the page
 * while it runs.
 */

import { SyncModelWorker } from '../lib/zeos/vendor/model_channel.js';
import RPC_SOURCE from '../lib/zeos/zeos_rpc.py?raw';
import type { KernelMessage, KernelRequest } from '../lib/zeos/protocol';
import { verifySha256 } from '../lib/zeos/verifySha256';

const PYODIDE_VERSION = '314.0.7';
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const CASES_DIR = '/zeos/cases';
const RPC_DIR = '/zeos/py';
const WHEELS_DIR = '/zeos/wheels';

declare const self: DedicatedWorkerGlobalScope;

// Minimal structural types for the parts of Pyodide used here.
interface PyProxyFn {
  (...args: unknown[]): unknown;
}
interface PyodideLike {
  version: string;
  FS: { mkdirTree(p: string): void; writeFile(p: string, data: Uint8Array | string): void };
  setStdout(o: { batched: (s: string) => void }): void;
  setStderr(o: { batched: (s: string) => void }): void;
  loadPackage(names: string | string[]): Promise<unknown>;
  pyimport(name: string): Record<string, PyProxyFn> & { destroy?: () => void };
  runPython(code: string): unknown;
  registerJsModule(name: string, module: object): void;
}

interface Manifest {
  zeos: { branch: string; commit: string; dirty: boolean };
  wheels: { name: string; path: string; sha256: string }[];
  cases: Record<string, string[]>;
}

let pyodide: PyodideLike | null = null;
let rpc: Record<string, PyProxyFn> | null = null;

function post(message: KernelMessage, transfer: Transferable[] = []) {
  self.postMessage(message, transfer);
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status} ${response.statusText}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function boot(manifestUrl: string) {
  if (pyodide) throw new Error('the ZEOS kernel worker is already booted');
  if (!self.crossOriginIsolated) {
    throw new Error(
      'the ZEOS kernel worker is not cross-origin isolated, so SharedArrayBuffer and ' +
        'Atomics.wait are unavailable; serve the site with COOP: same-origin and ' +
        'COEP: credentialless (astro.config.mjs / public/_headers)',
    );
  }
  const manifestResponse = await fetch(manifestUrl);
  if (!manifestResponse.ok) {
    throw new Error(
      `${manifestUrl}: HTTP ${manifestResponse.status}; run \`npm run zeos:sync\` to generate public/zeos/`,
    );
  }
  const manifest = (await manifestResponse.json()) as Manifest;

  post({ type: 'status', text: `loading Pyodide ${PYODIDE_VERSION}` });
  const { loadPyodide } = (await import(/* @vite-ignore */ `${PYODIDE_INDEX_URL}pyodide.mjs`)) as {
    loadPyodide: (o: { indexURL: string }) => Promise<PyodideLike>;
  };
  const py = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });
  py.setStdout({ batched: (text) => post({ type: 'log', stream: 'stdout', text }) });
  py.setStderr({ batched: (text) => post({ type: 'log', stream: 'stderr', text }) });

  post({ type: 'status', text: 'installing the ZEOS wheels' });
  // Each wheel is checked against the manifest's sha256 before it is
  // installed, from the verified bytes in Pyodide's FS (micropip `emfs:`), so
  // what runs is exactly what `zeos:sync` built.
  const wheelBytes = await Promise.all(
    manifest.wheels.map(async (w) => {
      const url = new URL(w.path, manifestUrl).href;
      const bytes = await fetchBytes(url);
      await verifySha256(bytes, w.sha256, w.name);
      return bytes;
    }),
  );
  py.FS.mkdirTree(WHEELS_DIR);
  const wheelPaths = manifest.wheels.map((w, i) => {
    const target = `${WHEELS_DIR}/${w.name}`;
    py.FS.writeFile(target, wheelBytes[i]);
    return `emfs:${target}`;
  });
  await py.loadPackage('micropip');
  const micropip = py.pyimport('micropip');
  await micropip.install(wheelPaths);
  micropip.destroy?.();

  post({ type: 'status', text: 'writing the ZEOS cases' });
  for (const [name, files] of Object.entries(manifest.cases)) {
    for (const file of files) {
      const target = `${CASES_DIR}/${name}/${file}`;
      py.FS.mkdirTree(target.slice(0, target.lastIndexOf('/')));
      py.FS.writeFile(target, await fetchBytes(new URL(`cases/${name}/${file}`, manifestUrl).href));
    }
  }

  py.registerJsModule('_zeos_host', {
    emit: (kind: string, json: string) => post({ type: 'event', kind, data: parseValue(json) }),
  });
  py.FS.mkdirTree(RPC_DIR);
  py.FS.writeFile(`${RPC_DIR}/_zeos_rpc.py`, RPC_SOURCE);
  const python = String(
    py.runPython(`import sys\nsys.path.insert(0, ${JSON.stringify(RPC_DIR)})\nsys.version.split()[0]`),
  );
  rpc = py.pyimport('_zeos_rpc');
  pyodide = py;
  return {
    pyodide: py.version,
    python,
    zeos: manifest.zeos,
    wheels: manifest.wheels.map((w) => w.name),
    cases: Object.keys(manifest.cases),
    casesDir: CASES_DIR,
    isolated: self.crossOriginIsolated,
  };
}

/** JSON from Python, with `{$bytes: base64}` turned into Uint8Array. */
function parseValue(json: string): unknown {
  return JSON.parse(json, (_key, value) => {
    if (value && typeof value === 'object' && typeof value.$bytes === 'string' && Object.keys(value).length === 1) {
      const bin = atob(value.$bytes);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
    return value;
  });
}

/** JSON for Python, with Uint8Array turned into `{$bytes: base64}`. */
function stringifyArgs(value: unknown): string {
  return JSON.stringify(value, (_key, v) => {
    if (v instanceof Uint8Array) {
      let bin = '';
      for (let i = 0; i < v.length; i++) bin += String.fromCharCode(v[i]);
      return { $bytes: btoa(bin) };
    }
    return v;
  });
}

class PythonCallError extends Error {
  traceback?: string;
  constructor(err: { name: string; message: string; traceback?: string }) {
    super(err.message);
    this.name = err.name;
    this.traceback = err.traceback;
  }
}

function unwrap(json: unknown): unknown {
  const reply = parseValue(String(json)) as
    | { ok: true; value: unknown }
    | { ok: false; error: { name: string; message: string; traceback?: string } };
  if (!reply.ok) throw new PythonCallError(reply.error);
  return reply.value;
}

function needRpc(): Record<string, PyProxyFn> {
  if (!rpc) throw new Error('the ZEOS kernel worker has not booted; send `boot` first');
  return rpc;
}

async function handle(req: KernelRequest): Promise<unknown> {
  switch (req.op) {
    case 'boot':
      return boot(req.manifestUrl);
    case 'attachModel': {
      const r = needRpc();
      const { port } = req;
      const worker = new SyncModelWorker(req.buffer, (m) => port.postMessage(m), {
        timeoutMs: req.timeoutMs,
      });
      r.register_model(req.name, worker);
      return { name: req.name, backend: worker.backend };
    }
    case 'exec':
      return unwrap(needRpc().exec_source(req.source));
    case 'call':
      return unwrap(needRpc().call(req.module, req.fn, stringifyArgs(req.args), stringifyArgs(req.kwargs)));
    case 'callMethod':
      return unwrap(
        needRpc().call_method(req.handle, req.method, stringifyArgs(req.args), stringifyArgs(req.kwargs)),
      );
    case 'getAttr':
      return unwrap(needRpc().get_attr(req.handle, req.name));
    case 'release':
      return unwrap(needRpc().release(req.handle));
  }
}

let chain: Promise<void> = Promise.resolve();

self.onmessage = (event: MessageEvent<KernelRequest & { id: number }>) => {
  const req = event.data;
  chain = chain.then(async () => {
    try {
      const value = await handle(req);
      post({ type: 'reply', id: req.id, ok: true, value });
    } catch (err) {
      const e = err as Partial<PythonCallError>;
      post({
        type: 'reply',
        id: req.id,
        ok: false,
        error: {
          name: e?.name ?? 'Error',
          message: e?.message ?? String(err),
          traceback: e?.traceback ?? (err instanceof Error ? err.stack : undefined),
        },
      });
    }
  });
};
