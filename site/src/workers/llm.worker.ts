/**
 * Local LLM inference worker (transformers.js on WebGPU).
 *
 * Hosts one tokenizer + causal LM at a time (`Gemma4ForCausalLM` or
 * `Qwen3_5ForCausalLM`, by the load request's `family`) and speaks the
 * `LlmWorkerIn` / `LlmWorkerOut` protocol (`llmWorkerProtocol.ts`) with
 * `lib/localLlm/llmService.ts`. Keeping inference off the main thread means
 * prefill no longer freezes the UI.
 *
 * Import discipline: NEVER import `toolPrompt.ts` here — it pulls in
 * `streamChat` → `agentTools` → DuckDB / Pyodide. Token constants come from
 * the leaf modules `gemmaTokens.ts` / `qwenTokens.ts`.
 */

import {
  env,
  AutoTokenizer,
  Gemma4ForCausalLM,
  Qwen3_5ForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,
  DynamicCache,
  ones,
  type Tensor,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type ProgressInfo,
} from '@huggingface/transformers';
import {
  GEMMA_SAMPLING,
  QWEN_SAMPLING,
  LLM_KV_KEEP_ON_DECODE_INTERRUPT,
  LLM_KV_REUSE,
  PREFILL_CHUNK_TOKENS,
  type GenerateStats,
  type LlmErrorCode,
  type LlmModelFamily,
  type LoadedInfo,
  type LlmWorkerIn,
  type LlmWorkerOut,
  type RawProgressEvent,
} from '../lib/localLlm/llmWorkerProtocol';
import { TURN_CLOSE_TOKEN_ID } from '../lib/localLlm/gemmaTokens';
import { IM_END } from '../lib/localLlm/qwenTokens';
import {
  checkGemmaTokenizer,
  checkQwenTokenizer,
  withBosText,
  type TokenizerLike,
} from '../lib/localLlm/gemmaTokenizerCheck';
import { LOCAL_GEMMA_CONTEXT_WINDOW } from '../lib/contextWindow';
import {
  MAX_KEPT_CACHE_TOKENS,
  cachedIdsAfterGenerate,
  findReusablePrefix,
  prefillChunkEnds,
} from '../lib/localLlm/kvReuse';
import { LOCAL_GEMMA_DTYPE, isLocalModelsMode } from '../lib/localLlm/models';
// ONNX Runtime's wasm glue, served from our own bundle. transformers.js
// (src/backends/onnx.js) otherwise points `wasmPaths` at jsDelivr; with WebGPU
// available it picks the `.asyncify` variant, so we ship exactly that one.
// The onnxruntime-web resolved here is the version transformers.js pins.
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';

// transformers.js only reads `wasmPaths` lazily (`ensureWasmLoaded()` at the
// first session creation), so overriding its import-time CDN default here is
// enough.
if (env.backends.onnx.wasm) {
  env.backends.onnx.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl };
  // Single-threaded wasm, as before the site became cross-origin isolated
  // (COOP/COEP, for the ZEOS kernel). Without isolation ORT falls back to one
  // thread; with it ORT defaults to several pthreads, which made Gemma E4B
  // WebGPU generation ~2.5x slower in realModelSql.spec.ts.
  env.backends.onnx.wasm.numThreads = 1;
}

/** Minimum spacing between forwarded per-file `progress` events. */
const PROGRESS_THROTTLE_MS = 100;

/**
 * Wraps a throw that came out of `model.generate` (i.e. ONNX Runtime). The
 * spike showed an ORT run error (e.g. `SafeIntOnOverflow`) leaves the runtime
 * in this worker permanently broken — even dispose + reload fails — so these
 * are reported as fatal and the main thread recycles the worker.
 */
class ModelRunError extends Error {}

async function runModel<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (e) {
    throw new ModelRunError(errMessage(e));
  }
}

/**
 * Work around a transformers.js (4.3) bug in the Qwen-VL family's
 * `prepare_inputs_for_generation`: whenever `past_key_values` is passed it
 * sets `model_inputs.pixel_values = null`, and `generic_text_to_text_forward`
 * then `pick`s that null (pick only skips `undefined`) and calls
 * `encode_image` on the `vision_encoder` session a text-only load never
 * created ("Cannot read properties of undefined (reading 'inputNames')").
 * Upstream only hits that branch on single-token decode steps, which skip
 * the image path; we pass our own `DynamicCache` on multi-token prefill
 * calls, so strip the null before forward sees it.
 */
function dropNullPixelValues(mdl: PreTrainedModel): void {
  const orig = mdl.prepare_inputs_for_generation.bind(mdl);
  mdl.prepare_inputs_for_generation = (...args: Parameters<typeof orig>) => {
    const inputs = orig(...args);
    if (inputs && inputs.pixel_values == null) delete inputs.pixel_values;
    return inputs;
  };
}

/** Gemma: used when `generation_config.json` is missing or carries no EOS ids. */
const FALLBACK_EOS_IDS = [1, TURN_CLOSE_TOKEN_ID];

if (isLocalModelsMode()) {
  // Dev/e2e: weights are served from the repo-root `models/` folder by the
  // local-models Vite plugin. Never touch the Hub or the browser cache.
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = '/models/';
  env.useBrowserCache = false;
} else {
  env.allowLocalModels = false;
  env.allowRemoteModels = true;
}

let model: PreTrainedModel | null = null;
let tokenizer: PreTrainedTokenizer | null = null;
/** Set together with `model`; lets a repeat `load` of the same repo short-circuit. */
let loadedInfo: LoadedInfo | null = null;
/** Family of the loaded model; selects BOS handling and sampling. */
let family: LlmModelFamily = 'gemma';
/** EOS ids from `generation_config`, dropped from the streamed text. */
let eosIds: Set<number> = new Set([TURN_CLOSE_TOKEN_ID]);
/** Serialises load/dispose so two loads never race on the GPU. */
let loading: Promise<void> | null = null;
/** Request id of the most recently *received* `load`; older queued loads are skipped. */
let latestLoadRequestId = 0;
/** Request id of the load whose task is currently running, if any. */
let runningLoadRequestId: number | null = null;
/**
 * The in-flight generation. `done` settles once `model.generate` has
 * returned, so a dispose can wait for `session.run` to finish before freeing
 * the GPU buffers it is using.
 */
let active: { id: number; stop: InterruptableStoppingCriteria; done: Promise<void> } | null =
  null;
/**
 * KV cache retained from the last completed generation, plus the exact token
 * ids it covers (prompt + sampled − the final sampled token, which is never
 * fed back). Owned here: a generation takes it (sets this to null) and either
 * hands back a new one or disposes it. See `kvReuse.ts`.
 */
let kept: { ids: number[]; cache: DynamicCache } | null = null;

async function dropKept(): Promise<void> {
  const k = kept;
  kept = null;
  if (k) await disposeCache(k.cache);
}

async function disposeCache(c: DynamicCache): Promise<void> {
  try {
    await c.dispose();
  } catch {
    // ignore
  }
}

function post(msg: LlmWorkerOut): void {
  (self as unknown as Worker).postMessage(msg);
}

function postError(id: number, code: LlmErrorCode, message: string, data?: unknown): void {
  post({ type: 'error', id, code, message, ...(data === undefined ? {} : { data }) });
}

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function normaliseEos(raw: unknown): number[] {
  if (raw === null || raw === undefined) return [];
  const arr = Array.isArray(raw) ? raw : [raw];
  return arr.map((x) => Number(x)).filter((x) => Number.isFinite(x));
}

function asTokenizerLike(t: PreTrainedTokenizer): TokenizerLike {
  return {
    encode: (text, opts) => t.encode(text, opts),
    decode: (ids, opts) => t.decode(ids, opts),
  };
}

type PromptEncoding = { input_ids: Tensor; attention_mask: Tensor };

/**
 * Tokenise a rendered prompt. Gemma: with exactly one leading `<bos>` — the
 * Gemma 4 tokenizer does not add it itself (its post-processor has no special
 * tokens), so we prepend the literal text, as the official chat template
 * does. Qwen's template has no BOS. Either way, encode without special
 * tokens so the ids are exactly the rendered text.
 */
function encodePrompt(t: PreTrainedTokenizer, text: string): PromptEncoding {
  const prompt = family === 'gemma' ? withBosText(text) : text;
  return t(prompt, { add_special_tokens: false }) as unknown as PromptEncoding;
}

function samplingFor(f: LlmModelFamily) {
  return f === 'qwen' ? QWEN_SAMPLING : GEMMA_SAMPLING;
}

/** Chain `task` after any in-flight load/dispose. */
function serialised(task: () => Promise<void>): Promise<void> {
  const prev = loading ?? Promise.resolve();
  const next = prev.catch(() => {}).then(task);
  loading = next;
  void next.finally(() => {
    if (loading === next) loading = null;
  });
  return next;
}

async function disposeCurrent(): Promise<void> {
  const gen = active;
  if (gen) {
    // Never free sessions under a running `session.run`: interrupt, then wait
    // for `generate` to return (it stops after the current step).
    gen.stop.interrupt();
    await gen.done.catch(() => {});
  }
  // After the generation has settled (it may have just handed back a cache).
  await dropKept();
  const m = model;
  model = null;
  tokenizer = null;
  loadedInfo = null;
  if (m) {
    try {
      await m.dispose();
    } catch {
      // ignore — the session is going away regardless
    }
  }
}

async function disposeQuietly(m: PreTrainedModel | null | undefined): Promise<void> {
  if (!m) return;
  try {
    await m.dispose();
  } catch {
    // ignore
  }
}

async function handleLoad(id: number, hfId: string, nextFamily: LlmModelFamily): Promise<void> {
  latestLoadRequestId = id;
  await serialised(async () => {
    if (id !== latestLoadRequestId) {
      // A newer `load` arrived while this one was queued; don't spend a
      // multi-GB download on a model nobody wants any more.
      postError(id, 'load-superseded', 'Model load superseded.');
      return;
    }
    runningLoadRequestId = id;
    try {
      if (model && loadedInfo?.hfId === hfId) {
        post({ type: 'loaded', id, info: loadedInfo });
        return;
      }
      await disposeCurrent();
      const files = new Set<string>();
      const lastProgressAt = new Map<string, number>();
      const progress_callback = (info: ProgressInfo): void => {
        const i = info as Partial<RawProgressEvent>;
        // transformers.js 4.3 fires `progress_total` (no `file`) tens of
        // thousands of times per load; the main-thread aggregator computes
        // totals itself, so drop it here.
        if (i.status === 'progress_total') return;
        if (typeof i.file === 'string') files.add(i.file);
        // Per-file byte progress fires ~21k times per `_data` file: forward
        // at most one per file per PROGRESS_THROTTLE_MS. Lifecycle events
        // (`initiate`/`download`/`done`/`ready`) always go through.
        if (i.status === 'progress' && typeof i.file === 'string') {
          const now = performance.now();
          const last = lastProgressAt.get(i.file);
          if (last !== undefined && now - last < PROGRESS_THROTTLE_MS) return;
          lastProgressAt.set(i.file, now);
        }
        post({
          type: 'progress',
          id,
          event: {
            status: String(i.status),
            file: i.file,
            loaded: i.loaded,
            total: i.total,
            progress: i.progress,
          },
        });
      };
      // The tokenizer also gets the callback: `tokenizer.json` is a required
      // (~19 MB) file and the main thread waits for every required file's
      // `done` before switching the throbber to the init phase.
      // allSettled, not all: if the tokenizer fails while the model is still
      // loading, we must wait for the model and dispose it rather than leak
      // its GPU sessions.
      // Text-only load: each class, given its `…ForConditionalGeneration`
      // config, loads just `embed_tokens` + `decoder_model_merged`.
      const ModelClass = nextFamily === 'qwen' ? Qwen3_5ForCausalLM : Gemma4ForCausalLM;
      const [tokRes, mdlRes] = await Promise.allSettled([
        AutoTokenizer.from_pretrained(hfId, { progress_callback }),
        ModelClass.from_pretrained(hfId, {
          dtype: LOCAL_GEMMA_DTYPE,
          device: 'webgpu',
          progress_callback,
        }),
      ]);
      if (tokRes.status === 'rejected' || mdlRes.status === 'rejected') {
        if (mdlRes.status === 'fulfilled') await disposeQuietly(mdlRes.value);
        const reason = tokRes.status === 'rejected' ? tokRes.reason : (mdlRes as PromiseRejectedResult).reason;
        postError(id, 'load-failed', errMessage(reason));
        return;
      }
      const tok = tokRes.value;
      const mdl = mdlRes.value;
      if (nextFamily === 'qwen') dropNullPixelValues(mdl);

      let eos = normaliseEos(mdl.generation_config?.eos_token_id);
      if (eos.length === 0) {
        const fallback =
          nextFamily === 'qwen'
            ? tok.encode(IM_END, { add_special_tokens: false }).slice(0, 1)
            : FALLBACK_EOS_IDS;
        console.warn(
          `[llm.worker] ${hfId}: generation_config has no eos_token_id ` +
            `(generation_config.json missing?); falling back to [${fallback.join(', ')}].`,
        );
        eos = fallback;
      }
      const check = nextFamily === 'qwen' ? checkQwenTokenizer : checkGemmaTokenizer;
      const problems = check(asTokenizerLike(tok), eos);
      if (problems.length > 0) {
        await disposeQuietly(mdl);
        postError(id, 'tokenizer-mismatch', `Tokenizer check failed: ${problems.join('; ')}`, {
          problems,
        });
        return;
      }

      model = mdl;
      tokenizer = tok;
      family = nextFamily;
      eosIds = new Set(eos);
      loadedInfo = { hfId, eosIds: eos, files: [...files] };
      post({ type: 'loaded', id, info: loadedInfo });
    } catch (e) {
      postError(id, 'load-failed', errMessage(e));
    } finally {
      if (runningLoadRequestId === id) runningLoadRequestId = null;
    }
  });
}

/**
 * `TextStreamer` that keeps special tokens (the tool-call / channel markers
 * are special tokens and the main thread parses them) but drops EOS ids so
 * `<turn|>` never leaks into the visible text. Also counts every sampled
 * token (EOS included) and times the decode window for `GenerateStats`.
 */
class GemmaStreamer extends TextStreamer {
  outputTokens = 0;
  /** Every sampled id, EOS included, in order — needed to know what the KV cache covers. */
  readonly sampledIds: number[] = [];
  firstTokenAt: number | null = null;
  lastTokenAt: number | null = null;

  constructor(
    tok: PreTrainedTokenizer,
    private readonly eos: ReadonlySet<number>,
    onText: (text: string) => void,
  ) {
    super(tok, {
      skip_prompt: true,
      skip_special_tokens: false,
      decode_kwargs: { skip_special_tokens: false, clean_up_tokenization_spaces: false },
      callback_function: onText,
    });
  }

  override put(value: bigint[][]): void {
    // `next_tokens_are_prompt` is TextStreamer's own flag (streamers.js);
    // the first `put` carries the whole prompt.
    if ((this as unknown as { next_tokens_are_prompt: boolean }).next_tokens_are_prompt) {
      super.put(value);
      return;
    }
    const toks = value[0] ?? [];
    const now = performance.now();
    this.outputTokens += toks.length;
    for (const t of toks) this.sampledIds.push(Number(t));
    if (this.firstTokenAt === null) this.firstTokenAt = now;
    this.lastTokenAt = now;
    const visible = toks.filter((t) => !this.eos.has(Number(t)));
    if (visible.length > 0) super.put([visible]);
  }

  get decodeMs(): number {
    return this.firstTokenAt !== null && this.lastTokenAt !== null
      ? this.lastTokenAt - this.firstTokenAt
      : 0;
  }
}

async function handleGenerate(id: number, prompt: string): Promise<void> {
  const mdl = model;
  const tok = tokenizer;
  if (!mdl || !tok) {
    postError(id, 'not-loaded', 'Local model is not loaded.');
    return;
  }
  if (active) {
    postError(id, 'busy', 'A generation is already in progress.');
    return;
  }
  const stop = new InterruptableStoppingCriteria();
  stop.reset();
  // `runGeneration` never rejects (it posts its own errors), so `done` is a
  // plain "the model is no longer running" signal covering every chunk.
  const work = runGeneration(id, stop, mdl, tok, prompt);
  const gen = { id, stop, done: work };
  active = gen;
  try {
    await work;
  } finally {
    if (active === gen) active = null;
  }
}

async function runGeneration(
  id: number,
  stop: InterruptableStoppingCriteria,
  mdl: PreTrainedModel,
  tok: PreTrainedTokenizer,
  prompt: string,
): Promise<void> {
  // Take ownership of any kept cache: from here on it is either reused (and
  // possibly handed back as the new `kept`) or disposed in `finally`.
  const prev = kept;
  kept = null;
  /** The cache this generation runs on; disposed in `finally` unless kept. */
  let cache: DynamicCache | null = null;
  try {
    const enc = encodePrompt(tok, prompt);
    const promptTokens = enc.input_ids.dims[1];
    const limit = LOCAL_GEMMA_CONTEXT_WINDOW;
    if (promptTokens >= limit) {
      if (prev) await disposeCache(prev.cache);
      postError(
        id,
        'context-too-long',
        `Prompt is ${promptTokens} tokens; the local model's context window is ${limit}.`,
        { promptTokens, limit },
      );
      return;
    }
    const newIds = Array.from(enc.input_ids.data as BigInt64Array, (x) => Number(x));

    // KV prefix reuse: only when the kept cache covers a strict prefix of this
    // prompt *and* its length is still what we recorded.
    let reusedTokens = 0;
    if (prev) {
      const n = LLM_KV_REUSE ? findReusablePrefix(prev.ids, newIds) : null;
      if (n !== null && prev.cache.get_seq_length() === n) {
        cache = prev.cache;
        reusedTokens = n;
      } else {
        if (import.meta.env.DEV) {
          // Diagnostic: where does the new prompt stop being a strict
          // extension of what the model last saw + produced?
          let k = 0;
          while (k < prev.ids.length && k < newIds.length && prev.ids[k] === newIds[k]) k++;
          const dec = (ids: number[]): string =>
            tok.decode(ids, { skip_special_tokens: false });
          console.warn(
            `[llm.worker] KV reuse miss: kept=${prev.ids.length} new=${newIds.length} ` +
              `diverge@${k} (seq_len=${prev.cache.get_seq_length()})\n` +
              `  kept: ${JSON.stringify(dec(prev.ids.slice(Math.max(0, k - 8), k + 8)))}\n` +
              `  new:  ${JSON.stringify(dec(newIds.slice(Math.max(0, k - 8), k + 8)))}`,
          );
        }
        await disposeCache(prev.cache);
      }
    }

    // Chunked prefill (docs/transformers-js-spike.md §2) of whatever the cache
    // does not cover yet. Each call feeds only the unseen suffix of a growing
    // prefix (`decoder_prepare_inputs_for_generation` slices `input_ids` by
    // `past_key_values.get_seq_length()`) and `getPastKeyValues` updates our
    // cache in place. The token each call samples is discarded and never fed
    // back, so the cache covers exactly the prefix. Chunk ends stop one short
    // of the prompt so the final `generate` still has a suffix to slice to.
    const chunkEnds = prefillChunkEnds(reusedTokens, promptTokens, PREFILL_CHUNK_TOKENS);
    // With reuse on, always run on our own cache so it survives the final
    // generate (passing `past_key_values` makes generate keep, not dispose, it).
    if (!cache && (chunkEnds.length > 0 || LLM_KV_REUSE)) cache = new DynamicCache();
    const kv = cache;
    const ids = enc.input_ids;
    for (const prefixLen of chunkEnds) {
      await runModel(() =>
        mdl.generate({
          input_ids: ids.slice(null, [0, prefixLen]),
          // Must cover the full prefix (past + new); otherwise prepare_inputs
          // builds ones(past + input_len), which is too long.
          attention_mask: ones([1, prefixLen]),
          past_key_values: kv,
          max_new_tokens: 1,
          do_sample: false,
        }),
      );
      const seq = kv!.get_seq_length();
      if (seq !== prefixLen) {
        throw new Error(
          `Chunked prefill: KV cache holds ${seq} positions after a ${prefixLen}-token prefix.`,
        );
      }
      post({ type: 'prefill', id, done: prefixLen, total: promptTokens });
      // `interrupt()` is only polled between decode steps, so a long prefill
      // can only be cancelled here, between chunks. The partial cache is
      // dropped (finally).
      if (stop.interrupted) {
        post({
          type: 'done',
          id,
          text: '',
          stats: {
            promptTokens,
            outputTokens: 0,
            decodeMs: 0,
            reusedTokens,
            prefillTokens: promptTokens - reusedTokens,
            reason: 'interrupted',
          },
        });
        return;
      }
    }

    let aggregated = '';
    // The streamer's first `put` is generate's full `input_ids` (taken before
    // prepare_inputs slices it to the unseen suffix), so `skip_prompt` drops
    // the whole prompt whether or not a cache is passed.
    const streamer = new GemmaStreamer(tok, eosIds, (text) => {
      aggregated += text;
      post({ type: 'token', id, text });
    });
    const maxNewTokens = limit - promptTokens;
    await runModel(() =>
      mdl.generate({
        ...enc,
        // Only pass the key when we own a cache: its mere presence makes
        // generate keep (not dispose) it.
        ...(kv ? { past_key_values: kv } : {}),
        max_new_tokens: maxNewTokens,
        ...samplingFor(family),
        streamer,
        stopping_criteria: stop,
      }),
    );

    const reason: GenerateStats['reason'] = stop.interrupted
      ? 'interrupted'
      : streamer.outputTokens >= maxNewTokens
        ? 'length'
        : 'eos';

    // Keep the cache for the next generation only if we can prove exactly
    // which ids it covers.
    if (LLM_KV_REUSE && kv && (reason !== 'interrupted' || LLM_KV_KEEP_ON_DECODE_INTERRUPT)) {
      const covered = cachedIdsAfterGenerate(newIds, streamer.sampledIds);
      const seq = kv.get_seq_length();
      if (covered && covered.length <= MAX_KEPT_CACHE_TOKENS && seq === covered.length) {
        kept = { ids: covered, cache: kv };
        cache = null; // ownership moved to `kept`
      } else if (covered && seq !== covered.length) {
        console.warn(
          `[llm.worker] not keeping KV cache: holds ${seq} positions, expected ${covered.length}.`,
        );
      }
    }

    post({
      type: 'done',
      id,
      text: aggregated,
      stats: {
        promptTokens,
        outputTokens: streamer.outputTokens,
        decodeMs: streamer.decodeMs,
        reusedTokens,
        prefillTokens: promptTokens - reusedTokens,
        reason,
      },
    });
  } catch (e) {
    postError(
      id,
      'generate-failed',
      errMessage(e),
      e instanceof ModelRunError ? { fatal: true } : undefined,
    );
  } finally {
    if (cache) await disposeCache(cache);
  }
}

function handleCount(id: number, text: string): void {
  if (!tokenizer) {
    postError(id, 'not-loaded', 'Local model is not loaded.');
    return;
  }
  try {
    post({ type: 'count', id, tokens: encodePrompt(tokenizer, text).input_ids.dims[1] });
  } catch (e) {
    postError(id, 'generate-failed', errMessage(e));
  }
}

async function handleDispose(id: number): Promise<void> {
  await serialised(disposeCurrent);
  post({ type: 'disposed', id });
}

// transformers.js 4.3.0 `getModelDataFiles` (utils/model-loader.js) wraps each
// external-data fetch in `new Promise(async …)` with no try/catch, so a
// failed `.onnx_data` download rejects nowhere and `from_pretrained` hangs
// forever. Surface it as a load failure; the main thread then recycles this
// (now wedged) worker.
self.addEventListener('unhandledrejection', (ev: PromiseRejectionEvent) => {
  if (runningLoadRequestId === null) return;
  ev.preventDefault();
  postError(runningLoadRequestId, 'load-failed', errMessage(ev.reason));
});

self.onmessage = (ev: MessageEvent<LlmWorkerIn>) => {
  const msg = ev.data;
  switch (msg.type) {
    case 'load':
      void handleLoad(msg.id, msg.hfId, msg.family);
      break;
    case 'generate':
      void handleGenerate(msg.id, msg.prompt);
      break;
    case 'cancel':
      if (active?.id === msg.id) active.stop.interrupt();
      break;
    case 'count':
      handleCount(msg.id, msg.text);
      break;
    case 'dispose':
      void handleDispose(msg.id);
      break;
  }
};
