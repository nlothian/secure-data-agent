/**
 * Where ONNX Runtime loads its wasm glue from: the `.asyncify` build, which
 * ORT picks when WebGPU is available (both `llm.worker.ts` and
 * `zeosOptModel.worker.ts` run on WebGPU).
 *
 * Dev serves it from our own bundle, so dev and e2e need no CDN for it. A
 * production build loads it from jsDelivr instead: the `.wasm` is ~26.9 MB,
 * over Cloudflare Pages' 25 MiB per-file limit. The dev branch's `?url`
 * imports are dynamic so a production build drops them and never emits the
 * file. jsDelivr sends `Cross-Origin-Resource-Policy: cross-origin` and CORS,
 * so it also loads on a cross-origin-isolated page.
 *
 * `version` is the onnxruntime-web the worker actually imported
 * (`ort.env.versions.web`), so the glue always matches the JS.
 */
export async function ortWasmPaths(version: string | undefined): Promise<{ wasm: string; mjs: string }> {
  if (import.meta.env.DEV) {
    const [wasm, mjs] = await Promise.all([
      import('onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url'),
      import('onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url'),
    ]);
    return { wasm: wasm.default, mjs: mjs.default };
  }
  if (!version) throw new Error('onnxruntime-web did not report its version (ort.env.versions.web)');
  const base = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/`;
  return {
    wasm: `${base}ort-wasm-simd-threaded.asyncify.wasm`,
    mjs: `${base}ort-wasm-simd-threaded.asyncify.mjs`,
  };
}
