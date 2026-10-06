import { LOCAL_GEMMA_ENDPOINT, type LLMConfig } from '../../types/llm';

// The `LocalGemma*` names predate non-Gemma local models; they now cover
// every in-browser model the local endpoint can run.
export type LocalGemmaId = 'gemma-4-e2b' | 'gemma-4-e4b' | 'qwen3.5-4b' | 'zeos-qwen3.5-4b';

/**
 * Chat-template family. Selects the prompt renderer / parsers
 * (`promptFormat.ts`) and, in the worker, the model class and tokenizer
 * checks. `zeos-qwen` is Qwen's chat template run under the ZEOS kernel
 * (`src/lib/zeos/streamZeos.ts`), not by the transformers.js worker.
 */
export type LocalModelFamily = 'gemma' | 'qwen' | 'zeos-qwen';

export interface LocalGemmaModel {
  id: LocalGemmaId;
  label: string;
  family: LocalModelFamily;
  /**
   * Hugging Face repo id. transformers.js resolves files under
   * `https://huggingface.co/<hfRepoId>/resolve/main/<file>` (or under
   * `/models/<hfRepoId>/<file>` in local-models dev mode).
   */
  hfRepoId: string;
  /**
   * Approximate total download for the text-only q4f16 file set (decoder +
   * embed_tokens + tokenizer/config JSON). Shown in the size-confirmation
   * dialogs; `models.test.ts` asserts it stays within 1% of the manifest in
   * `modelFiles.json` (when the repo has a manifest entry).
   */
  approxBytes: number;
  notes?: string;
  /**
   * The transformers.js model that side tasks (code summaries, the Explainer's
   * one-shot turns) use while this model is active, for a model the
   * transformers.js worker cannot run itself (`zeos-qwen`).
   */
  sideTaskModelId?: LocalGemmaId;
  /**
   * Where a model that transformers.js does not load (`zeos-qwen`) comes from
   * outside local-models mode: a Hub repo pinned to one commit, read through
   * ZEOS's `modelReader` and kept in OPFS under that key
   * (`src/lib/zeos/vendor/model_cache.js`). `hfRepoId` still names its
   * `/models/` directory and its `modelFiles.json` entry.
   */
  hubSource?: { repo: string; revision: string };
}

/**
 * Weight precision for every predefined model. Exported so the worker and the
 * file manifest agree on the `_q4f16` file-name suffix.
 */
export const LOCAL_GEMMA_DTYPE = 'q4f16' as const;

/**
 * The models offered in every build. ZEOS Qwen 4B loads from `/models/` in
 * local-models dev mode, from the Hub (`hubSource`, kept in OPFS) otherwise,
 * or runs on the scripted stub in ZEOS stub mode.
 */
export const LOCAL_GEMMA_MODELS: readonly LocalGemmaModel[] = [
  {
    id: 'gemma-4-e2b',
    label: 'Gemma 4 E2B',
    family: 'gemma',
    hfRepoId: 'onnx-community/gemma-4-E2B-it-ONNX',
    approxBytes: 3_130_000_000,
    notes: 'Faster and lighter. Recommended default.',
  },
  {
    id: 'gemma-4-e4b',
    label: 'Gemma 4 E4B',
    family: 'gemma',
    hfRepoId: 'onnx-community/gemma-4-E4B-it-ONNX',
    approxBytes: 4_925_000_000,
    notes:
      'Stronger reasoning; needs more GPU memory and gets slow past ~8k tokens of context.',
  },
  {
    id: 'qwen3.5-4b',
    label: 'Qwen 3.5 4B',
    family: 'qwen',
    hfRepoId: 'onnx-community/Qwen3.5-4B-ONNX-OPT',
    approxBytes: 2_820_843_621,
    notes: 'Alibaba Qwen 3.5. Different chat template and tool-call format from Gemma.',
  },
  {
    id: 'zeos-qwen3.5-4b',
    label: 'ZEOS Qwen 4B',
    family: 'zeos-qwen',
    hfRepoId: 'metacognitionai/Qwen3.5-4B-ZEOS-OPT',
    approxBytes: 2_821_280_718,
    notes:
      'Qwen 3.5 4B under the ZEOS kernel: tool results are untrusted (ring 3), and ' +
      'side-effecting tools need your approval once the model has read them. Needs WebGPU ' +
      'with shader-f16.',
    sideTaskModelId: 'qwen3.5-4b',
    // The commit ZEOS's own pages load (`zeos_browser.model_source`
    // HF_REPO / HF_REVISION). A commit, never a branch: the cache keys
    // files by it.
    hubSource: {
      repo: 'nlothian/Qwen3.5-4B-ZEOS-OPT_Q4F16',
      revision: 'f71e07f80aa6d20f66c69575facae9813a053c5e',
    },
  },
];

/**
 * Dev only: the ZEOS model runs on a scripted stub model thread instead of the
 * real weights when localStorage `gda.zeos.stub` holds a stub script (see
 * `src/lib/zeos/zeosModelWorker.ts`). Nothing is downloaded then, so the model
 * counts as cached (`modelCache.ts`).
 */
export const ZEOS_STUB_STORAGE_KEY = 'gda.zeos.stub';

export function isZeosStubMode(): boolean {
  if (!import.meta.env.DEV) return false;
  try {
    return globalThis.localStorage?.getItem(ZEOS_STUB_STORAGE_KEY) != null;
  } catch {
    return false;
  }
}

export function isZeosModel(m: LocalGemmaModel | undefined): boolean {
  return m?.family === 'zeos-qwen';
}

/**
 * The model the transformers.js worker should load for `id`: itself, or its
 * `sideTaskModelId` for a model that worker cannot run.
 */
export function transformersModelIdFor(id: string): string {
  return LOCAL_GEMMA_MODELS.find((m) => m.id === id)?.sideTaskModelId ?? id;
}

/**
 * `config` for a side task (the Explainer's conversation): with the local
 * endpoint's model replaced by its `sideTaskModelId`, so a side task never
 * runs on ZEOS Qwen 4B. That model's kernel holds one conversation (the main
 * chat's); sharing it would close the chat's run, force a ~30 s re-prefill
 * on its next message, and could answer its pending approval. Any other
 * config is returned unchanged.
 */
export function sideTaskConfig(config: LLMConfig): LLMConfig {
  if (config.activeEndpoint !== LOCAL_GEMMA_ENDPOINT) return config;
  const active = getLocalGemmaModel(resolveActiveLocalModelIdOrDefault(config));
  if (!active?.sideTaskModelId) return config;
  return { ...config, models: { ...config.models, [LOCAL_GEMMA_ENDPOINT]: active.sideTaskModelId } };
}

export const DEFAULT_LOCAL_GEMMA_ID: LocalGemmaId = 'gemma-4-e2b';

export function isLocalGemmaId(id: unknown): id is LocalGemmaId {
  return typeof id === 'string' && LOCAL_GEMMA_MODELS.some((m) => m.id === id);
}

export function getLocalGemmaModel(id: string | null | undefined): LocalGemmaModel | undefined {
  if (!id) return undefined;
  return LOCAL_GEMMA_MODELS.find((m) => m.id === id);
}

export function formatGB(bytes: number): string {
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

/**
 * The single source of truth for "which local Gemma model id is active",
 * applying the default fallback. Used by every local-inference entry point
 * (`streamLocalGemma`, `summariseCode`, `compactConversation`) and the
 * boot-time eager-load so they all resolve the same id. A stale id left in
 * localStorage by an older build (e.g. a `custom:<name>` entry from the
 * removed `.task` picker) falls back to the default.
 */
export function resolveActiveLocalModelIdOrDefault(config: LLMConfig): LocalGemmaId {
  const requested = config.models[LOCAL_GEMMA_ENDPOINT];
  return isLocalGemmaId(requested) ? requested : DEFAULT_LOCAL_GEMMA_ID;
}

/**
 * Dev/e2e only: when `PUBLIC_LOCAL_MODELS=1` the Vite dev server serves the
 * gitignored repo-root `models/` folder at `/models/` and the worker points
 * transformers.js there instead of the Hugging Face Hub. Statically replaced
 * by Vite in both the page and worker bundles.
 */
export function isLocalModelsMode(): boolean {
  // Guarded on DEV so a stray `PUBLIC_LOCAL_MODELS=1` in `.env` can never be
  // inlined into a production build (which would point the deployed site at
  // a `/models/` path that does not exist and disable the Hub).
  return import.meta.env.DEV && import.meta.env.PUBLIC_LOCAL_MODELS === '1';
}
