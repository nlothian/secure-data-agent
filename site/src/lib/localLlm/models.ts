import { LOCAL_GEMMA_ENDPOINT, type LLMConfig } from '../../types/llm';

export type LocalGemmaId = 'gemma-4-e2b' | 'gemma-4-e4b';

export interface LocalGemmaModel {
  id: LocalGemmaId;
  label: string;
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
   * `modelFiles.json`.
   */
  approxBytes: number;
  notes?: string;
}

/**
 * Weight precision for both predefined models. Exported so the worker and the
 * file manifest agree on the `_q4f16` file-name suffix.
 */
export const LOCAL_GEMMA_DTYPE = 'q4f16' as const;

export const LOCAL_GEMMA_MODELS: readonly LocalGemmaModel[] = [
  {
    id: 'gemma-4-e2b',
    label: 'Gemma 4 E2B',
    hfRepoId: 'onnx-community/gemma-4-E2B-it-ONNX',
    approxBytes: 3_130_000_000,
    notes: 'Faster and lighter. Recommended default.',
  },
  {
    id: 'gemma-4-e4b',
    label: 'Gemma 4 E4B',
    hfRepoId: 'onnx-community/gemma-4-E4B-it-ONNX',
    approxBytes: 4_925_000_000,
    notes:
      'Stronger reasoning; needs more GPU memory and gets slow past ~8k tokens of context.',
  },
];

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
