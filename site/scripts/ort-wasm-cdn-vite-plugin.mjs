import fs from 'node:fs';
import path from 'node:path';

/**
 * Build-only: point ONNX Runtime Web's own `.wasm` fallback at jsDelivr so
 * the file is never emitted into `dist/`.
 *
 * ORT's bundle resolves its wasm with
 * `new URL("ort-wasm-simd-threaded.asyncify.wasm", import.meta.url)` when
 * `env.wasm.wasmPaths` is unset, and Vite emits every such asset whether or
 * not that path runs. The asyncify `.wasm` is ~26.9 MB, over Cloudflare
 * Pages' 25 MiB per-file limit, so the deploy fails. The workers set
 * `wasmPaths` to the same jsDelivr URLs in a production build
 * (src/lib/localLlm/ortWasm.ts); this removes the unused local copy. Dev is
 * untouched and serves ORT from node_modules.
 *
 * The version comes from the package.json of the onnxruntime-web copy being
 * transformed, so the URL always matches the JS that loads it. Registered for
 * both the page and worker bundles (Vite does not apply top-level plugins to
 * workers).
 */
const WASM_URL = /new URL\(\s*(["'])(ort-wasm-[\w.-]+\.wasm)\1\s*,\s*import\.meta\.url\s*\)/g;

function ortVersionFor(id) {
  const marker = `${path.sep}node_modules${path.sep}onnxruntime-web${path.sep}`;
  const at = id.lastIndexOf(marker);
  if (at < 0) return null;
  const pkg = path.join(id.slice(0, at + marker.length), 'package.json');
  return JSON.parse(fs.readFileSync(pkg, 'utf8')).version;
}

export default function ortWasmCdnPlugin() {
  return {
    name: 'ort-wasm-cdn',
    apply: 'build',
    enforce: 'pre',
    transform(code, id) {
      if (!code.includes('ort-wasm-')) return null;
      const version = ortVersionFor(id.split('?')[0]);
      if (!version) return null;
      let changed = false;
      const out = code.replace(WASM_URL, (_m, _q, file) => {
        changed = true;
        return `new URL(${JSON.stringify(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/${file}`)})`;
      });
      return changed ? { code: out, map: null } : null;
    },
  };
}
