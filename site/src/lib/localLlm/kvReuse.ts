/**
 * Pure decisions for KV-cache prefix reuse and chunked prefill in the LLM
 * worker (`workers/llm.worker.ts`). Kept free of transformers.js so it can be
 * unit-tested in Node.
 *
 * transformers.js `DynamicCache` cannot be cropped, so a kept cache is only
 * reusable when it covers a *strict prefix* of the next prompt: generate()
 * then slices `input_ids` to the unseen suffix (`decoder_prepare_inputs_for_
 * generation`) and only that suffix is prefilled.
 */
import { LOCAL_GEMMA_CONTEXT_WINDOW } from '../contextWindow';

/** Never keep a cache covering more positions than this (GPU memory bound). */
export const MAX_KEPT_CACHE_TOKENS = LOCAL_GEMMA_CONTEXT_WINDOW;

/**
 * Number of leading tokens of `newIds` already covered by a cache over
 * `keptIds`, or `null` when the cache cannot be reused. Reuse requires
 * `keptIds` to be non-empty and a strict prefix of `newIds` — an equal-length
 * or shorter prompt (new chat, compaction rewrote history) always falls back,
 * because the suffix to prefill would be empty and the cache can't be cropped.
 */
export function findReusablePrefix(
  keptIds: readonly number[],
  newIds: readonly number[],
): number | null {
  const n = keptIds.length;
  if (n === 0 || newIds.length <= n) return null;
  for (let i = 0; i < n; i++) {
    if (keptIds[i] !== newIds[i]) return null;
  }
  return n;
}

/**
 * Prefix lengths at which to run chunk `generate` calls (each feeds the
 * unseen suffix up to that length into the cache) before the final,
 * streaming `generate`. Starts from `start` already-cached tokens. Empty when
 * the remaining suffix fits in one chunk — the final `generate` prefills it
 * directly. The last end is `total - 1` so the final call still has a
 * non-empty suffix to slice to.
 */
export function prefillChunkEnds(start: number, total: number, chunk: number): number[] {
  const ends: number[] = [];
  if (total - start <= chunk) return ends;
  let end = start;
  while (end < total - 1) {
    end = Math.min(end + chunk, total - 1);
    ends.push(end);
  }
  return ends;
}

/**
 * Token ids a cache covers after a generate over `promptIds` that sampled
 * `sampledIds` (EOS included). The last sampled token is never fed back
 * through the model, so it is not in the cache. `null` when nothing was
 * sampled (the cache state is then not provable from the ids alone).
 */
export function cachedIdsAfterGenerate(
  promptIds: readonly number[],
  sampledIds: readonly number[],
): number[] | null {
  if (sampledIds.length === 0) return null;
  return [...promptIds, ...sampledIds.slice(0, -1)];
}
