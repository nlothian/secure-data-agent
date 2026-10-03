/**
 * Browser-cache probing for the predefined Gemma models.
 *
 * transformers.js stores every fetched model file in Cache Storage under
 * `env.cacheKey` ('transformers-cache'), keyed by the full Hugging Face
 * resolve URL. We ask that cache directly rather than keeping our own
 * "downloaded" marker in localStorage: Chrome may evict Cache Storage under
 * storage pressure, a marker would then go stale, and the boot-time eager load
 * (which only runs when this returns true) must never kick off a multi-GB
 * download the user did not ask for.
 */
import { isLocalModelsMode, type LocalGemmaModel } from './models';
import { requiredFiles } from './modelFiles';

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

/**
 * True when every required file for `model` is already in the transformers.js
 * cache (or when running against the local `models/` folder, where nothing is
 * downloaded). Any error is treated as "not cached".
 */
export async function isModelCached(model: LocalGemmaModel): Promise<boolean> {
  if (isLocalModelsMode()) return true;
  if (typeof caches === 'undefined') return false;
  try {
    return (await missingFiles(model)).length === 0;
  } catch {
    return false;
  }
}

/**
 * Bytes still to download for `model`, based on the static manifest. Falls
 * back to the full required size when Cache Storage is unavailable.
 */
export async function uncachedBytes(model: LocalGemmaModel): Promise<number> {
  if (isLocalModelsMode()) return 0;
  const all = requiredFiles(model).reduce((n, f) => n + f.bytes, 0);
  if (typeof caches === 'undefined') return all;
  try {
    return (await missingFiles(model)).reduce((n, f) => n + f.bytes, 0);
  } catch {
    return all;
  }
}
