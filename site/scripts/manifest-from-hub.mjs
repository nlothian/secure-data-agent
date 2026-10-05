#!/usr/bin/env node
/**
 * Add (or refresh) a repo's entry in `src/lib/localLlm/modelFiles.json` from
 * the Hugging Face Hub's file listing, so a newly chosen ONNX export gets the
 * same exact-size manifest the Gemma repos have.
 *
 *   npm run models:manifest -- onnx-community/Qwen3.5-4B-ONNX
 *
 * Picks the text-only q4f16 file set transformers.js fetches for
 * `<Model>ForCausalLM.from_pretrained(repo, { dtype: 'q4f16' })` + the
 * tokenizer: the config / tokenizer JSON and every
 * `onnx/{embed_tokens,decoder_model_merged}_q4f16.onnx*` shard as required,
 * `chat_template.jinja` as optional. Prints the total so `approxBytes` in
 * `models.ts` can be set to match (models.test.ts checks it within 1%).
 * Sends `Authorization: Bearer $HF_TOKEN` if set.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const manifestPath = path.resolve(__dirname, '..', 'src', 'lib', 'localLlm', 'modelFiles.json');

const REQUIRED_JSON = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'generation_config.json',
];
const OPTIONAL = ['chat_template.jinja'];
const ONNX_RE = /^onnx\/(embed_tokens|decoder_model_merged)_q4f16\.onnx(_data(_\d+)?)?$/;

const repo = process.argv[2];
if (!repo || !repo.includes('/')) {
  console.error('usage: npm run models:manifest -- <owner/repo>');
  process.exit(2);
}

const headers = { Accept: 'application/json' };
if (process.env.HF_TOKEN) headers.Authorization = `Bearer ${process.env.HF_TOKEN}`;
const url = `https://huggingface.co/api/models/${repo}/tree/main?recursive=true`;
const res = await fetch(url, { headers });
if (!res.ok) {
  console.error(`${url}: ${res.status} ${res.statusText}`);
  process.exit(1);
}
const entries = (await res.json()).filter((e) => e.type === 'file');
const sizeOf = (e) => e.lfs?.size ?? e.size;
const byPath = new Map(entries.map((e) => [e.path, e]));

const missing = REQUIRED_JSON.filter((p) => !byPath.has(p));
const onnx = entries.filter((e) => ONNX_RE.test(e.path));
for (const stem of ['embed_tokens', 'decoder_model_merged']) {
  if (!byPath.has(`onnx/${stem}_q4f16.onnx`)) missing.push(`onnx/${stem}_q4f16.onnx`);
}
if (missing.length > 0) {
  console.error(`${repo} is missing expected files: ${missing.join(', ')}`);
  process.exit(1);
}

const file = (p) => ({ path: p, bytes: sizeOf(byPath.get(p)) });
const required = [
  ...REQUIRED_JSON.filter((p) => p !== 'generation_config.json').map(file),
  ...onnx.map((e) => e.path).sort().map(file),
  file('generation_config.json'),
];
const optional = OPTIONAL.filter((p) => byPath.has(p)).map(file);

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
manifest[repo] = { required, optional };
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');

const total = [...required, ...optional].reduce((n, f) => n + f.bytes, 0);
console.log(`wrote ${repo} to ${path.relative(process.cwd(), manifestPath)}`);
for (const f of [...required, ...optional]) console.log(`  ${f.path} ${f.bytes}`);
console.log(`total ${total} bytes — set approxBytes in models.ts to ~${total}`);
