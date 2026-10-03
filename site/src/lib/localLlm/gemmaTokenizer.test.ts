/**
 * Runs `checkGemmaTokenizer` against the REAL Gemma 4 tokenizers from the
 * gitignored repo-root `models/` folder (populated by
 * `scripts/fetch-models.mjs`). Each repo is skipped — with a warning, not a
 * failure — when its tokenizer files are absent or still downloading,
 * matching the gating style of `e2e/llm/realModelSql.spec.ts`.
 */
import { describe, expect, it } from 'vitest';
import { existsSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import manifest from './modelFiles.json';
import { checkGemmaTokenizer } from './gemmaTokenizerCheck';

const REPOS = ['onnx-community/gemma-4-E2B-it-ONNX', 'onnx-community/gemma-4-E4B-it-ONNX'];
// Test lives in site/src/lib/localLlm/, so the repo-root models/ dir is four
// levels up.
const MODELS_ROOT = fileURLToPath(new URL('../../../../models/', import.meta.url));
const TOKENIZER_FILES = ['tokenizer.json', 'tokenizer_config.json', 'config.json'];
// `generation_config.json` `eos_token_id` for both exports.
const EOS_IDS = [1, 106, 50];

type Manifest = Record<string, { required: { path: string; bytes: number }[] }>;

function missingFiles(repo: string): string[] {
  const required = (manifest as Manifest)[repo]?.required ?? [];
  return TOKENIZER_FILES.filter((p) => {
    const full = `${MODELS_ROOT}${repo}/${p}`;
    const expected = required.find((f) => f.path === p)?.bytes;
    return !existsSync(full) || (expected !== undefined && statSync(full).size !== expected);
  });
}

for (const repo of REPOS) {
  const missing = missingFiles(repo);
  const present = missing.length === 0;
  if (!present) {
    console.warn(
      `[gemmaTokenizer.test] skipping ${repo}: tokenizer files missing or incomplete under ` +
        `${MODELS_ROOT} (${missing.join(', ')}). Run \`npm run models:fetch\` to enable.`,
    );
  }

  describe.skipIf(!present)(`Gemma 4 tokenizer (real files): ${repo}`, () => {
    it('matches the chat-template assumptions', async () => {
      const { env, AutoTokenizer } = await import('@huggingface/transformers');
      env.allowLocalModels = true;
      env.allowRemoteModels = false;
      env.localModelPath = MODELS_ROOT;
      const tok = await AutoTokenizer.from_pretrained(repo);
      const problems = checkGemmaTokenizer(
        {
          encode: (text, opts) => tok.encode(text, opts),
          decode: (ids, opts) => tok.decode(ids, opts),
        },
        EOS_IDS,
      );
      expect(problems).toEqual([]);
    }, 60_000);
  });
}
