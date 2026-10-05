#!/usr/bin/env node
/**
 * Add (or refresh) a repo's entry in `src/lib/localLlm/modelFiles.json` from
 * the `meta.json` a local export wrote, for a model that is not on the Hub
 * (the ZEOS export, `metacognitionai/Qwen3.5-4B-ZEOS-OPT`).
 *
 *   npm run models:manifest-local -- metacognitionai/Qwen3.5-4B-ZEOS-OPT
 *
 * Reads `<repoRoot>/models/<repo>/meta.json` (`files: {path: {bytes, sha256}}`).
 * Every listed file is required except `chat_template.jinja` (optional);
 * `meta.json` itself is required too, since the ZEOS model worker reads it
 * first. Prints the total so `approxBytes` in `models.ts` can be set to match
 * (models.test.ts checks it within 1%).
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const manifestPath = path.resolve(__dirname, '..', 'src', 'lib', 'localLlm', 'modelFiles.json');
const OPTIONAL = new Set(['chat_template.jinja']);

const repo = process.argv[2];
if (!repo || !repo.includes('/')) {
  console.error('usage: npm run models:manifest-local -- <owner/repo>');
  process.exit(2);
}
const dir = path.join(repoRoot, 'models', repo);
const metaPath = path.join(dir, 'meta.json');
if (!fs.existsSync(metaPath)) {
  console.error(`${metaPath} not found; populate models/${repo} first (npm run models:fetch)`);
  process.exit(1);
}
const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
if (!meta.files || typeof meta.files !== 'object') {
  console.error(`${metaPath} has no \`files\` map`);
  process.exit(1);
}

const entries = Object.entries(meta.files)
  .map(([p, f]) => ({ path: p, bytes: f.bytes }))
  .sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
const required = [
  ...entries.filter((f) => !OPTIONAL.has(f.path)),
  { path: 'meta.json', bytes: fs.statSync(metaPath).size },
];
const optional = entries.filter((f) => OPTIONAL.has(f.path));

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
manifest[repo] = { required, optional };
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');

const total = [...required, ...optional].reduce((n, f) => n + f.bytes, 0);
console.log(`wrote ${repo} to ${path.relative(process.cwd(), manifestPath)}`);
for (const f of [...required, ...optional]) console.log(`  ${f.path} ${f.bytes}`);
console.log(`total ${total} bytes — set approxBytes in models.ts to ~${total}`);
