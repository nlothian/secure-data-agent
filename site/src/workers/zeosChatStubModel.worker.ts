/// <reference lib="webworker" />
/**
 * A model thread for the ZEOS chat machine with no model: the scripted chat
 * stub (src/lib/zeos/scriptedChatModel.ts) served over the SharedArrayBuffer
 * channel. Speaks the model-thread protocol in src/lib/zeos/protocol.ts, so it
 * is a drop-in for the real OPT ZEOS model thread. First message:
 * `{ buffer, port, script }` with `script` a `ChatStubScript`. After that,
 * `{ crash: message }` makes the thread throw an uncaught error, as a real
 * model thread would on a fatal failure (e2e/zeosLifecycle.spec.ts).
 */
import { serveChannel } from '../lib/zeos/vendor/model_channel.js';
import { createScriptedChatModel, type ChatStubScript } from '../lib/zeos/scriptedChatModel';

declare const self: DedicatedWorkerGlobalScope;

self.onmessage = (event: MessageEvent) => {
  self.onmessage = null;
  const { buffer, port, script } = event.data as {
    buffer: SharedArrayBuffer;
    port: MessagePort;
    script: ChatStubScript;
  };
  try {
    const worker = createScriptedChatModel(script ?? { replies: [] });
    serveChannel(worker, buffer, (handle) => {
      port.onmessage = (m) => handle(m.data);
    });
    self.onmessage = (m: MessageEvent) => {
      const crash = (m.data as { crash?: unknown } | null)?.crash;
      if (crash !== undefined) {
        setTimeout(() => {
          throw new Error(String(crash));
        }, 0);
      }
    };
    self.postMessage({ ready: true, backend: worker.backend });
  } catch (err) {
    self.postMessage({ ready: false, error: String((err as Error)?.stack ?? err) });
  }
};
