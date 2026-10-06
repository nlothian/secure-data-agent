/// <reference lib="webworker" />
/**
 * A model thread with no model: ZEOS's `stub_worker.js` (fixed vocabulary,
 * plays each descriptor's tape) served over the SharedArrayBuffer channel.
 * It speaks the model-thread protocol in src/lib/zeos/protocol.ts, so it is a
 * drop-in for a real model thread. Used by the smoke test
 * (e2e/zeosKernel.spec.ts) to exercise Pyodide + wheels + Atomics without a
 * multi-GB model. First message: `{ buffer, port, tapes }`, where `tapes` is
 * `zeos_browser.page.tapes_json(caseDir)` parsed.
 */
import '../lib/zeos/vendor/stub_worker.js';
import { serveChannel } from '../lib/zeos/vendor/model_channel.js';

declare const self: DedicatedWorkerGlobalScope;

self.onmessage = (event: MessageEvent) => {
  self.onmessage = null;
  const { buffer, port, tapes } = event.data as {
    buffer: SharedArrayBuffer;
    port: MessagePort;
    tapes: Record<string, string[]>;
  };
  try {
    const worker = globalThis.createStubWorker(tapes ?? {});
    // serveRequest answers `backend` from .backend, which the in-process
    // stub never needed. (`pieces` reads meta.tokenizerSize, a getter on the
    // stub itself; assigning `meta` would throw.)
    worker.backend = 'stub';
    let calls = 0;
    serveChannel(worker, buffer, (handle) => {
      port.onmessage = (m) => {
        calls += 1;
        if (calls % 50 === 0) self.postMessage({ activity: { calls } });
        handle(m.data);
      };
    });
    self.postMessage({ ready: true, backend: 'stub' });
  } catch (err) {
    self.postMessage({ ready: false, error: String((err as Error)?.stack ?? err) });
  }
};
