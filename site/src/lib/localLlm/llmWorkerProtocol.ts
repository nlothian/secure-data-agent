/**
 * Message protocol between `llmService.ts` (main thread) and
 * `workers/llm.worker.ts` (transformers.js host). Type-only plus a few
 * constants — this module must stay free of runtime imports so both sides can
 * include it without pulling in app code.
 *
 * Every request carries a monotonic `id` minted on the main thread; every
 * reply echoes it so the service can route replies to the right pending
 * promise.
 */

export type LlmWorkerIn =
  | { type: 'load'; id: number; hfId: string }
  | { type: 'generate'; id: number; prompt: string }
  | { type: 'count'; id: number; text: string }
  /** `id` names the in-flight `generate` request to interrupt. */
  | { type: 'cancel'; id: number }
  | { type: 'dispose'; id: number };

/**
 * Raw transformers.js `progress_callback` payload, forwarded untouched. The
 * main thread aggregates per-file events into a single percentage
 * (`loadProgress.ts`); the worker stays dumb.
 */
export interface RawProgressEvent {
  status: string;
  file?: string;
  loaded?: number;
  total?: number;
  progress?: number;
}

export interface LoadedInfo {
  hfId: string;
  /** `generation_config.eos_token_id`, as loaded. */
  eosIds: number[];
  /** Every file name the progress callback reported — lets us assert the
   *  static manifest in `modelFiles.json` is complete. */
  files: string[];
}

export interface GenerateStats {
  /** Prompt length in tokens, including the leading `<bos>`. */
  promptTokens: number;
  /** Sampled tokens, including any EOS and any held-back tool-call text. */
  outputTokens: number;
  /** First sampled token → last sampled token (excludes prefill). */
  decodeMs: number;
  /** Leading prompt tokens served from the KV cache kept by the previous generation (0 = miss). */
  reusedTokens: number;
  /** Prompt tokens actually run through the model (`promptTokens - reusedTokens`). */
  prefillTokens: number;
  reason: 'eos' | 'length' | 'interrupted';
}

export type LlmErrorCode =
  | 'context-too-long' // data: { promptTokens: number; limit: number }
  | 'not-loaded'
  | 'busy'
  | 'load-superseded'
  | 'tokenizer-mismatch'
  | 'load-failed'
  | 'generate-failed';

export type LlmWorkerOut =
  | { type: 'progress'; id: number; event: RawProgressEvent }
  | { type: 'loaded'; id: number; info: LoadedInfo }
  | { type: 'token'; id: number; text: string }
  /** Chunked-prefill progress for a `generate` request: `done` of `total` prompt tokens are in the KV cache. */
  | { type: 'prefill'; id: number; done: number; total: number }
  | { type: 'done'; id: number; text: string; stats: GenerateStats }
  | { type: 'count'; id: number; tokens: number }
  | { type: 'disposed'; id: number }
  /**
   * `data` by code: `context-too-long` → `{ promptTokens, limit }`;
   * `generate-failed` → `{ fatal: true }` when the throw came out of ONNX
   * Runtime (the worker's runtime is then unusable and must be recycled).
   */
  | { type: 'error'; id: number; message: string; code: LlmErrorCode; data?: unknown };

/**
 * Prompts longer than this are prefilled in chunks of this many tokens into a
 * shared KV cache. A single decoder forward over N tokens materialises an fp16
 * `[8 heads, N, N]` attention-score buffer whose byte size overflows ORT's
 * 32-bit size math at N ≈ 16,384 (`SafeIntOnOverflow`), which also poisons the
 * worker's runtime. Chunk C against total length L is safe while
 * C·L < ~2.68e8, i.e. 2048-token chunks are safe to ~131k total.
 * See docs/transformers-js-spike.md §2.
 */
export const PREFILL_CHUNK_TOKENS = 2048;

/**
 * KV-cache prefix reuse across generations. When on, the worker keeps the
 * KV cache of each completed generation together with the exact token ids it
 * covers; if the next prompt strictly extends those ids, only the new suffix
 * is prefilled. The agent loop's prompt grows monotonically (prompt + model
 * output + tool response), so later tool iterations usually hit. Flip to
 * `false` to always prefill from scratch.
 */
export const LLM_KV_REUSE = true;

/**
 * Also keep the cache when the final decode was interrupted by a caller
 * cancel. `streamLocalGemma` cancels as soon as it has parsed a complete tool
 * call, so most tool iterations end `interrupted`; dropping their caches would
 * make reuse almost never hit. The kept ids are still proven against
 * `get_seq_length()`. Interrupts during chunked prefill, errors and model
 * switches always drop the cache.
 */
export const LLM_KV_KEEP_ON_DECODE_INTERRUPT = true;

/**
 * Sampling parameters, carried over from the previous runtime (temperature 0.8,
 * topK 40). There is no seed equivalent in transformers.js, so outputs are no
 * longer deterministic run-to-run.
 */
//
// Passed as `generate()` kwargs, these override the model's own
// `generation_config.json` (`do_sample: true, temperature: 1, top_k: 64,
// top_p: 0.95`) key by key (`_prepare_generation_config`). `top_p: 1` is
// belt-and-braces: transformers.js 4.3.0 never applies `top_p` (the
// `TopPLogitsWarper` push is commented out and `MultinomialSampler` only uses
// `top_k`), but pinning it keeps an upgrade from silently adding the model's
// 0.95 nucleus cut on top of `top_k: 40`.
export const GEMMA_SAMPLING = {
  do_sample: true,
  temperature: 0.8,
  top_k: 40,
  top_p: 1.0,
} as const;
