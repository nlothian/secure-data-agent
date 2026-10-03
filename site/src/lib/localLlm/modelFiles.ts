/**
 * Static per-repo file manifest for the text-only q4f16 Gemma 4 ONNX exports.
 *
 * Three consumers share the JSON so they cannot drift:
 *   - `modelCache.ts` — "is every required file already in the browser cache?"
 *   - `scripts/fetch-models.mjs` — populates the dev-only `models/` folder
 *   - `e2e/llm/*.spec.ts` — gates the heavyweight LLM suite on the folder
 *
 * "required" is the set transformers.js fetches for
 * `Gemma4ForCausalLM.from_pretrained(repo, {dtype:'q4f16', device:'webgpu'})`
 * plus `AutoTokenizer`; "optional" are files it probes and tolerates missing.
 * The worker reports every file it actually saw (`LoadedInfo.files`) so a
 * manifest gap shows up in dev as a console warning rather than a silent
 * "not cached" answer.
 */
import manifest from './modelFiles.json';
import type { LocalGemmaModel } from './models';

export interface ModelFile {
  path: string;
  bytes: number;
}

interface RepoManifest {
  required: ModelFile[];
  optional: ModelFile[];
}

const MANIFEST = manifest as Record<string, RepoManifest>;

function repoManifest(model: LocalGemmaModel): RepoManifest {
  const m = MANIFEST[model.hfRepoId];
  if (!m) throw new Error(`No file manifest for ${model.hfRepoId}`);
  return m;
}

export function requiredFiles(model: LocalGemmaModel): readonly ModelFile[] {
  return repoManifest(model).required;
}

export function allFiles(model: LocalGemmaModel): readonly ModelFile[] {
  const m = repoManifest(model);
  return [...m.required, ...m.optional];
}

export function totalBytes(model: LocalGemmaModel): number {
  return allFiles(model).reduce((n, f) => n + f.bytes, 0);
}

export function manifestRepoIds(): readonly string[] {
  return Object.keys(MANIFEST);
}
