// Hand-written types for the vendored ./opfs_store.js (not generated).
import type { ModelCacheKey, ModelStore } from './model_cache';

/** The model cache's store: OPFS, `zeos-model-cache/<repo>/<revision>/<path>`. */
export const opfsStore: ModelStore;

/** How long a revision must go unused before another revision's load removes it. */
export const UNUSED_FOR_MS: number;

/** Record that a load is using `key.revision` now. */
export function markUsed(key: ModelCacheKey, options?: { top?: unknown; now?: number }): Promise<void>;

/** Best-effort removal of other revisions of `key.repo` unused for `UNUSED_FOR_MS`. */
export function dropOtherRevisions(
  key: ModelCacheKey,
  options?: { top?: unknown; now?: number; log?: (message: string) => void },
): Promise<string[]>;

/** What the cache holds: one entry per repo and revision, with its files' sizes. */
export function cachedModels(): Promise<
  {
    repo: string;
    revision: string;
    files: { path: string; bytes: number; partial: boolean }[];
    bytes: number;
  }[]
>;

/** Remove the whole cache. Fails while a download holds a part open (in another tab). */
export function clearModelCache(): Promise<void>;
