/// <reference lib="webworker" />
/**
 * The real model thread for ZEOS Qwen 4B: ZEOS `OptZeosWorker` (vendored
 * `opt_zeos_worker.js`) over the OPT+ZEOS export, on WebGPU, served to the
 * kernel worker over the SharedArrayBuffer channel. A site port of ZEOS
 * `packages/zeos-browser/web/model_thread.js`, with the app's own onnxruntime-web (the build
 * transformers.js pins, 1.31.0-dev, which runs the fused LinearAttention /
 * CausalConvWithState ops on WebGPU) and `@huggingface/tokenizers`.
 *
 * First message: `{ buffer, port, source }`, `source` being `{url, cache}`
 * (`ZeosModelSource` in src/lib/zeos/zeosModelWorker.ts): the export's
 * directory, and the `{repo, revision}` to keep its files under in OPFS, or
 * null for no cache. In local-models mode that is
 * `/models/metacognitionai/Qwen3.5-4B-ZEOS-OPT/` with no cache; otherwise
 * the pinned Hub revision, cached. Every file is read through ZEOS
 * `modelReader` (vendored `model_cache.js`), which reads meta.json first,
 * checks the origin's storage room before the first weight file, and checks
 * each file's size and SHA-256 against `meta.files`. OPFS writes need a
 * synchronous access handle, which only a dedicated worker has: this is one.
 *
 * Speaks the model-thread protocol in src/lib/zeos/protocol.ts: `{progress}`
 * while loading (`modelReader`'s, `phase` `'download'`, `'cache'` or
 * `'verify'` with `bytes` / `bytes_total` over the unique files, then
 * `phase: 'session'`), then `{ready: true, backend: 'webgpu'}` or
 * `{ready: false, error}`. After that every request on `port` is answered
 * into `buffer` by `serveChannel`.
 *
 * There is no WebAssembly fallback: a 4 GiB wasm heap cannot hold the 2.4 GB
 * of weights plus activations (ZEOS README, "The OPT+ZEOS worker").
 */
import * as ort from 'onnxruntime-web/webgpu';
import { Tokenizer } from '@huggingface/tokenizers';
import { ortWasmPaths } from '../lib/localLlm/ortWasm';
import { serveChannel } from '../lib/zeos/vendor/model_channel.js';
import { OptZeosWorker, isOptZeosMeta } from '../lib/zeos/vendor/opt_zeos_worker.js';
import {
  IntegrityError,
  QuotaError,
  modelReader,
  type ModelReadProgress,
} from '../lib/zeos/vendor/model_cache.js';
import { dropOtherRevisions, markUsed, opfsStore } from '../lib/zeos/vendor/opfs_store.js';
import type { ZeosModelSource } from '../lib/zeos/zeosModelWorker';

declare const self: DedicatedWorkerGlobalScope;

interface FileInfo {
  bytes: number;
  sha256?: string;
}

async function requireWebGpu(): Promise<void> {
  const gpu = (self.navigator as Navigator & { gpu?: GPU }).gpu;
  const adapter = gpu ? await gpu.requestAdapter() : null;
  if (!adapter) {
    throw new Error(
      'ZEOS Qwen 4B needs WebGPU, and this browser has no WebGPU adapter. ' +
        'It has no WebAssembly fallback (the weights do not fit a 4 GiB wasm heap).',
    );
  }
  if (!adapter.features.has('shader-f16')) {
    throw new Error('ZEOS Qwen 4B needs a WebGPU adapter with shader-f16 for its q4f16 weights.');
  }
}

/**
 * A rejection nothing handles (the channel's reply chain, say) leaves the
 * kernel waiting on a reply that will never come. Report it as a failure:
 * after ready, the page treats `{ready: false}` as a crash and disposes the
 * kernel (zeosHost `watchModelThread`) instead of waiting for the timeout.
 */
self.addEventListener('unhandledrejection', (event: PromiseRejectionEvent) => {
  const reason = event.reason as Error | undefined;
  self.postMessage({ ready: false, error: `unhandled rejection: ${String(reason?.stack ?? reason)}` });
});

/** Minimum spacing between forwarded download progress messages. */
const PROGRESS_EVERY_MS = 100;

self.onmessage = async (event: MessageEvent) => {
  self.onmessage = null;
  const { buffer, port, source } = event.data as {
    buffer: SharedArrayBuffer;
    port: MessagePort;
    source: ZeosModelSource;
  };
  try {
    await requireWebGpu();
    if (ort.env.wasm) {
      ort.env.wasm.wasmPaths = await ortWasmPaths(ort.env.versions.web);
      // One thread, as in llm.worker.ts: the page is cross-origin isolated,
      // so ORT would otherwise start a pthread pool it does not need here.
      ort.env.wasm.numThreads = 1;
    }
    const url = new URL(source.url, self.location.href).href;

    // Forward progress at most every PROGRESS_EVERY_MS, but always the first
    // message of a phase or file and the last of a file, so the page can
    // tell a resumed download's check from the one that ends a download.
    let lastPost = 0;
    let lastKey = '';
    const reader = modelReader({
      url,
      cache: source.cache,
      fetch: self.fetch.bind(self),
      store: source.cache ? opfsStore : null,
      onProgress: (progress: ModelReadProgress) => {
        const key = `${progress.phase}:${progress.file}`;
        const now = performance.now();
        if (key === lastKey && now - lastPost < PROGRESS_EVERY_MS && progress.loaded < progress.total) return;
        lastKey = key;
        lastPost = now;
        self.postMessage({ progress });
      },
    });

    const sizes = (): Record<string, FileInfo> =>
      ((reader.meta() as { files?: Record<string, FileInfo> } | null)?.files ?? {});
    // What has been read, by hash where the export gives one: OptZeosWorker
    // reads a file whose hash it has already read (the decoder's tied second
    // shard) from memory instead.
    const readKeys = new Set<string>();
    const keyOf = (name: string): string => sizes()[name]?.sha256 ?? name;
    let sessionPosted = false;
    const read = async (name: string): Promise<Uint8Array> => {
      const bytes = await reader.read(name);
      readKeys.add(keyOf(name));
      // ORT builds the decoder session once its graph and every data shard
      // are in. The graph file is read before its ~2.4 GB of data, so
      // 'session' waits for the shards too (it used to follow the graph).
      const decoder = (reader.meta() as { decoder?: { file: string; externalData?: string[] } } | null)
        ?.decoder;
      if (!sessionPosted && decoder) {
        const files = [decoder.file, ...(decoder.externalData ?? [])];
        if (files.every((f) => readKeys.has(keyOf(f)))) {
          sessionPosted = true;
          self.postMessage({
            progress: { phase: 'session', bytes: reader.bytesRead(), bytes_total: reader.bytesTotal() },
          });
        }
      }
      return bytes;
    };

    const began = performance.now();
    // Also fails here, before any weights, when the browser cannot store
    // what is left to download (QuotaError).
    await read('meta.json');
    const meta = reader.meta();
    if (!meta || !isOptZeosMeta(meta)) {
      throw new Error(`${url}meta.json does not describe an OPT+ZEOS export`);
    }
    if (source.cache) {
      await markUsed(source.cache).catch((error) => console.warn(`model cache: ${error}`));
    }
    const worker = await OptZeosWorker.load({
      ort,
      Tokenizer,
      read,
      backend: 'webgpu',
      numThreads: 1,
      onActivity: (activity: Record<string, unknown>) => self.postMessage({ activity }),
    });
    serveChannel(worker, buffer, (handle) => {
      port.onmessage = (message) => handle(message.data);
    });
    self.postMessage({ progress: { phase: 'ready', ms: performance.now() - began } });
    self.postMessage({ ready: true, backend: 'webgpu' });
    // Every file of this revision is stored now. Revisions no load has used
    // for a while are removed, best-effort and not awaited: a failure here
    // never touches the load that finished.
    if (source.cache) {
      dropOtherRevisions(source.cache).catch((error) => console.warn(`model cache cleanup: ${error}`));
    }
  } catch (error) {
    // The cache's own failures carry a message meant for the user; the stack
    // would only bury it in the chat.
    const known = error instanceof QuotaError || error instanceof IntegrityError;
    self.postMessage({ ready: false, error: known ? error.message : String((error as Error)?.stack ?? error) });
  }
};
