/// <reference lib="webworker" />
/**
 * The real model thread for ZEOS Qwen 4B: ZEOS `OptZeosWorker` (vendored
 * `opt_zeos_worker.js`) over the OPT+ZEOS export, on WebGPU, served to the
 * kernel worker over the SharedArrayBuffer channel. A site port of ZEOS
 * `packages/zeos-browser/web/model_thread.js`, with the app's own onnxruntime-web (the build
 * transformers.js pins, 1.31.0-dev, which runs the fused LinearAttention /
 * CausalConvWithState ops on WebGPU) and `@huggingface/tokenizers`.
 *
 * First message: `{ buffer, port, modelUrl }`, `modelUrl` the export's
 * directory (`/models/metacognitionai/Qwen3.5-4B-ZEOS-OPT/` in local-models
 * mode). Speaks the model-thread protocol in src/lib/zeos/protocol.ts:
 * `{progress}` while loading (`phase: 'download'` with `bytes` /
 * `bytes_total` over the unique files, then `phase: 'session'`), then
 * `{ready: true, backend: 'webgpu'}` or `{ready: false, error}`. After that
 * every request on `port` is answered into `buffer` by `serveChannel`.
 *
 * There is no WebAssembly fallback: a 4 GiB wasm heap cannot hold the 2.4 GB
 * of weights plus activations (ZEOS README, "The OPT+ZEOS worker").
 */
import * as ort from 'onnxruntime-web/webgpu';
import { Tokenizer } from '@huggingface/tokenizers';
import { ortWasmPaths } from '../lib/localLlm/ortWasm';
import { serveChannel } from '../lib/zeos/vendor/model_channel.js';
import { OptZeosWorker, isOptZeosMeta } from '../lib/zeos/vendor/opt_zeos_worker.js';

declare const self: DedicatedWorkerGlobalScope;

interface FileInfo {
  bytes: number;
  sha256?: string;
}

async function fetchBytes(url: URL, onProgress: (loaded: number, total: number) => void): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url.pathname}: HTTP ${response.status}`);
  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body || total === 0) return new Uint8Array(await response.arrayBuffer());
  const out = new Uint8Array(total);
  const reader = response.body.getReader();
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, loaded);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  return out;
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
  const { buffer, port, modelUrl } = event.data as {
    buffer: SharedArrayBuffer;
    port: MessagePort;
    modelUrl: string;
  };
  try {
    await requireWebGpu();
    if (ort.env.wasm) {
      ort.env.wasm.wasmPaths = await ortWasmPaths(ort.env.versions.web);
      // One thread, as in llm.worker.ts: the page is cross-origin isolated,
      // so ORT would otherwise start a pthread pool it does not need here.
      ort.env.wasm.numThreads = 1;
    }
    const base = new URL(modelUrl, self.location.href);

    let sizes: Record<string, FileInfo> = {};
    let meta: { decoder?: { file: string; externalData?: string[] } } | null = null;
    let metaBytes: Uint8Array | null = null;
    let bytesBefore = 0;
    let lastPost = 0;
    // Files with the same hash (the embedding data and the decoder's second
    // shard) are read once by OptZeosWorker, so they count once.
    const uniqueTotal = (): number => {
      const seen = new Map<string, number>();
      for (const [name, f] of Object.entries(sizes)) seen.set(f.sha256 ?? name, f.bytes);
      let sum = 0;
      for (const b of seen.values()) sum += b;
      return sum;
    };
    // What has been read, by hash where the export gives one: OptZeosWorker
    // reads a file whose hash it has already read (the decoder's tied second
    // shard) from memory instead.
    const readKeys = new Set<string>();
    const keyOf = (name: string): string => sizes[name]?.sha256 ?? name;
    let sessionPosted = false;
    const read = async (name: string): Promise<Uint8Array> => {
      if (name === 'meta.json' && metaBytes !== null) return metaBytes;
      const total = uniqueTotal();
      const bytes = await fetchBytes(new URL(name, base), (loaded, fileTotal) => {
        const now = performance.now();
        if (now - lastPost < PROGRESS_EVERY_MS && loaded < fileTotal) return;
        lastPost = now;
        self.postMessage({
          progress: { phase: 'download', file: name, loaded, total: fileTotal, bytes: bytesBefore + loaded, bytes_total: total },
        });
      });
      if (name === 'meta.json') {
        metaBytes = bytes;
        meta = JSON.parse(new TextDecoder().decode(bytes));
        sizes = (meta as { files?: Record<string, FileInfo> }).files ?? {};
      } else {
        bytesBefore += bytes.byteLength;
      }
      readKeys.add(keyOf(name));
      // ORT builds the decoder session once its graph and every data shard
      // are in. The graph file is read before its ~2.4 GB of data, so
      // 'session' waits for the shards too (it used to follow the graph).
      const decoder = meta?.decoder;
      if (!sessionPosted && decoder) {
        const files = [decoder.file, ...(decoder.externalData ?? [])];
        if (files.every((f) => readKeys.has(keyOf(f)))) {
          sessionPosted = true;
          self.postMessage({ progress: { phase: 'session', bytes: bytesBefore, bytes_total: uniqueTotal() } });
        }
      }
      return bytes;
    };

    const began = performance.now();
    await read('meta.json');
    if (!meta || !isOptZeosMeta(meta)) {
      throw new Error(`${base.pathname}meta.json does not describe an OPT+ZEOS export`);
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
  } catch (error) {
    self.postMessage({ ready: false, error: String((error as Error)?.stack ?? error) });
  }
};
