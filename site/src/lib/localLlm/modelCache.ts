/**
 * Browser-cache probing for the predefined local models.
 *
 * transformers.js stores every fetched model file in Cache Storage under
 * `env.cacheKey` ('transformers-cache'), keyed by the full Hugging Face
 * resolve URL. We ask that cache directly rather than keeping our own
 * "downloaded" marker in localStorage: Chrome may evict Cache Storage under
 * storage pressure, a marker would then go stale, and the boot-time eager load
 * (which only runs when this returns true) must never kick off a multi-GB
 * download the user did not ask for.
 *
 * A model with `hubSource` (ZEOS Qwen 4B) is not loaded by transformers.js:
 * its model thread keeps the files in OPFS through ZEOS's `opfs_store.js`,
 * keyed `[repo, revision, path]`, so for it we ask that store instead.
 */
import { isLocalModelsMode, isZeosStubMode, type LocalGemmaModel } from './models';
import { hasManifest, requiredFiles } from './modelFiles';
import { modelFiles, type ExpectedFile, type ModelCacheKey } from '../zeos/vendor/model_cache';
import { opfsStore } from '../zeos/vendor/opfs_store';

/** transformers.js `env.cacheKey` default. */
export const TRANSFORMERS_CACHE_NAME = 'transformers-cache';

/** The exact Cache API key transformers.js uses for a Hub file. */
export function hfFileUrl(hfRepoId: string, path: string): string {
  return `https://huggingface.co/${hfRepoId}/resolve/main/${path}`;
}

/** Dev/e2e path served by the local-models Vite plugin. */
export function localFileUrl(hfRepoId: string, path: string): string {
  return `/models/${hfRepoId}/${path}`;
}

async function missingFiles(model: LocalGemmaModel) {
  const cache = await caches.open(TRANSFORMERS_CACHE_NAME);
  const files = requiredFiles(model);
  const hits = await Promise.all(
    files.map((f) => cache.match(hfFileUrl(model.hfRepoId, f.path))),
  );
  return files.filter((_, i) => hits[i] === undefined);
}

/** The parts of an OPT+ZEOS export's `meta.json` that say which files load. */
interface ZeosExportMeta {
  files?: Record<string, ExpectedFile>;
  embedTokens?: { file: string; externalData?: string[] };
  decoder?: { file: string; externalData?: string[] };
}

const ZEOS_META = 'meta.json';

function hasOpfs(): boolean {
  return typeof navigator !== 'undefined' && typeof navigator.storage?.getDirectory === 'function';
}

/** The stored `meta.json` of `source`, or null when it is not stored. */
async function storedZeosMeta(source: ModelCacheKey): Promise<ZeosExportMeta | null> {
  const key = [source.repo, source.revision, ZEOS_META];
  if ((await opfsStore.complete(key)) === null) return null;
  const decoder = new TextDecoder();
  let text = '';
  for await (const chunk of opfsStore.chunks(key, { partial: false })) {
    text += decoder.decode(chunk, { stream: true });
  }
  return JSON.parse(text + decoder.decode()) as ZeosExportMeta;
}

/**
 * The files the ZEOS model thread reads after `meta.json`, as
 * `OptZeosWorker.load` (vendored `opt_zeos_worker.js`) asks for them: the
 * tokenizer, then each graph and its external data. Not `config.json` or
 * `generation_config.json`, which the manifest lists but it never reads.
 */
function zeosLoadedFiles(meta: ZeosExportMeta): [string, ExpectedFile][] {
  const graphs = [meta.embedTokens, meta.decoder];
  if (graphs.some((g) => g === undefined)) throw new Error('meta.json names no graphs');
  const names = [
    'tokenizer.json',
    'tokenizer_config.json',
    ...graphs.flatMap((g) => [g!.file, ...(g!.externalData ?? [])]),
  ];
  return names.map((name) => {
    const expected = meta.files?.[name];
    if (expected === undefined) throw new Error(`meta.json does not list ${name}`);
    return [name, expected];
  });
}

/**
 * Bytes the ZEOS model thread still has to download for `source`, counted as
 * ZEOS's own `modelFiles().missing` counts them: a stored file of the right
 * size is in, a `.part` counts what it holds, and files with the same SHA-256
 * (the embedding's data and the decoder's tied second shard) count once.
 * Without a stored `meta.json` nothing has been read yet, so it is
 * `fullBytes`.
 */
async function opfsMissingBytes(source: ModelCacheKey, fullBytes: number): Promise<number> {
  const meta = await storedZeosMeta(source);
  if (meta === null) return fullBytes;
  const files = modelFiles({ url: 'about:blank', cache: source, fetch, store: opfsStore });
  return files.missing(zeosLoadedFiles(meta));
}

/** Download size of `model` with nothing cached, from the manifest if it has one. */
function fullBytes(model: LocalGemmaModel): number {
  if (!hasManifest(model)) return model.approxBytes;
  return requiredFiles(model).reduce((n, f) => n + f.bytes, 0);
}

/** Nothing is downloaded: local-models dev mode, or the ZEOS stub (no weights). */
function needsNoDownload(model: LocalGemmaModel): boolean {
  return isLocalModelsMode() || (model.family === 'zeos-qwen' && isZeosStubMode());
}

/**
 * True when every file `model` loads is already cached: in the
 * transformers.js cache, or for a model with `hubSource` in its OPFS store
 * (and always when running against the local `models/` folder, where nothing
 * is downloaded). Any error is treated as "not cached".
 */
export async function isModelCached(model: LocalGemmaModel): Promise<boolean> {
  if (needsNoDownload(model)) return true;
  if (model.hubSource) {
    if (!hasOpfs()) return false;
    try {
      return (await opfsMissingBytes(model.hubSource, Infinity)) === 0;
    } catch {
      return false;
    }
  }
  // No manifest: we cannot tell, so never claim it is cached.
  if (!hasManifest(model)) return false;
  if (typeof caches === 'undefined') return false;
  try {
    return (await missingFiles(model)).length === 0;
  } catch {
    return false;
  }
}

/**
 * Bytes still to download for `model`. For a transformers.js model, the
 * manifest's sizes of the files missing from Cache Storage; for a model with
 * `hubSource`, what its OPFS store still lacks. Falls back to the full
 * required size when the store is unavailable or cannot be read.
 */
export async function uncachedBytes(model: LocalGemmaModel): Promise<number> {
  if (needsNoDownload(model)) return 0;
  const all = fullBytes(model);
  if (model.hubSource) {
    if (!hasOpfs()) return all;
    try {
      return await opfsMissingBytes(model.hubSource, all);
    } catch {
      return all;
    }
  }
  if (!hasManifest(model)) return model.approxBytes;
  if (typeof caches === 'undefined') return all;
  try {
    return (await missingFiles(model)).reduce((n, f) => n + f.bytes, 0);
  } catch {
    return all;
  }
}
