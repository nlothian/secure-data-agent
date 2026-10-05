#!/usr/bin/env node
/**
 * Populate the gitignored repo-root `models/` folder with the text-only q4f16
 * ONNX files of the local models, laid out exactly as transformers.js expects
 * under `/models/<hfRepoId>/…` when `PUBLIC_LOCAL_MODELS=1`.
 *
 *   npm run models:fetch -- e2b      # Gemma 4 E2B, ≈ 3.1 GB
 *   npm run models:fetch -- e4b      # Gemma 4 E4B, ≈ 4.9 GB
 *   npm run models:fetch -- qwen4b   # Qwen 3.5 4B (needs its manifest entry)
 *   npm run models:fetch -- zeosq4b  # ZEOS Qwen 4B: linked from $ZEOS_REPO, no download
 *   npm run models:fetch -- all      # every alias that has a manifest entry
 *
 * Reads the same manifest the app uses (`src/lib/localLlm/modelFiles.json`),
 * streams each file to `<path>.part` then renames, skips files whose on-disk
 * size already matches, and sends `Authorization: Bearer $HF_TOKEN` if set.
 * Node ≥ 18 (global fetch). Equivalent `hf download` command is in CLAUDE.md.
 */
import fs from 'node:fs';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const modelsRoot = path.resolve(repoRoot, 'models');
const manifestPath = path.resolve(
  __dirname,
  '..',
  'src',
  'lib',
  'localLlm',
  'modelFiles.json',
);

// Keep in sync with `hfRepoId` in src/lib/localLlm/models.ts.
const ALIASES = {
  e2b: 'onnx-community/gemma-4-E2B-it-ONNX',
  e4b: 'onnx-community/gemma-4-E4B-it-ONNX',
  qwen4b: 'onnx-community/Qwen3.5-4B-ONNX-OPT',
  zeosq4b: 'metacognitionai/Qwen3.5-4B-ZEOS-OPT',
};

/**
 * Repos that are not on the Hub: the ZEOS export is built in a ZEOS checkout
 * (`demo/coop-count-web/export/opt_zeos_surgery.py`), so its files are
 * hard-linked (or copied, across filesystems) from there.
 */
const ZEOS_REPO = path.resolve(
  process.env.ZEOS_REPO ?? '/Users/nlothian/dev/github/metacognitionai/zeos-task2-transformers',
);
const LOCAL_SOURCES = {
  'metacognitionai/Qwen3.5-4B-ZEOS-OPT': path.join(
    ZEOS_REPO,
    'demo/coop-count-web/models/Qwen3.5-4B-ZEOS-OPT',
  ),
};

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

function usage() {
  console.error(`usage: npm run models:fetch -- <${Object.keys(ALIASES).join('|')}|all>`);
  process.exit(2);
}

const arg = (process.argv[2] ?? '').toLowerCase();
let repos;
if (arg === 'all') {
  repos = Object.values(ALIASES).filter((repo) => {
    if (manifest[repo]) return true;
    console.warn(`skipping ${repo}: no manifest entry (run \`npm run models:manifest -- ${repo}\`)`);
    return false;
  });
}
else if (ALIASES[arg]) repos = [ALIASES[arg]];
else usage();

function fmtGB(n) {
  return `${(n / 1e9).toFixed(2)} GB`;
}

async function fetchFile(repo, file) {
  const dest = path.join(modelsRoot, repo, file.path);
  const part = dest + '.part';
  if (fs.existsSync(dest) && fs.statSync(dest).size === file.bytes) {
    console.log(`  ✓ ${file.path} (cached)`);
    return;
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const url = `https://huggingface.co/${repo}/resolve/main/${file.path}`;
  const headers = {};
  if (process.env.HF_TOKEN) headers.Authorization = `Bearer ${process.env.HF_TOKEN}`;
  const res = await fetch(url, { headers });
  if (!res.ok || !res.body) {
    throw new Error(`${url}: ${res.status} ${res.statusText}`);
  }
  const total = Number(res.headers.get('content-length') ?? file.bytes);
  let read = 0;
  let lastPct = -1;
  const progress = new TransformStream({
    transform(chunk, controller) {
      read += chunk.byteLength;
      const pct = Math.floor((read / total) * 100);
      if (pct !== lastPct && (pct % 5 === 0 || pct === 100)) {
        lastPct = pct;
        process.stdout.write(`\r  ↓ ${file.path} ${pct}% of ${fmtGB(total)}`);
      }
      controller.enqueue(chunk);
    },
  });
  await pipeline(
    Readable.fromWeb(res.body.pipeThrough(progress)),
    fs.createWriteStream(part),
  );
  process.stdout.write('\n');
  const size = fs.statSync(part).size;
  if (size !== file.bytes) {
    // Never promote a truncated / mismatched download: the e2e gate and the
    // app's cache check both key on exact sizes.
    fs.unlinkSync(part);
    throw new Error(
      `${file.path}: expected ${file.bytes} bytes, got ${size} — ` +
        `retry, or update modelFiles.json if the upstream export changed`,
    );
  }
  fs.renameSync(part, dest);
}

function linkFile(repo, src, file) {
  const from = path.join(src, file.path);
  const dest = path.join(modelsRoot, repo, file.path);
  if (!fs.existsSync(from)) throw new Error(`${from} is missing; build the export in ZEOS first`);
  const size = fs.statSync(from).size;
  if (size !== file.bytes) {
    throw new Error(
      `${from}: ${size} bytes, the manifest says ${file.bytes} — ` +
        `re-run \`npm run models:manifest-local -- ${repo}\` after rebuilding the export`,
    );
  }
  if (fs.existsSync(dest) && fs.statSync(dest).size === file.bytes) {
    console.log(`  ✓ ${file.path} (present)`);
    return;
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.rmSync(dest, { force: true });
  try {
    fs.linkSync(from, dest);
    console.log(`  ⇒ ${file.path} (linked)`);
  } catch {
    fs.copyFileSync(from, dest);
    console.log(`  ⇒ ${file.path} (copied)`);
  }
}

for (const repo of repos) {
  if (LOCAL_SOURCES[repo] && !manifest[repo]) {
    // First run: link meta.json so manifest-from-local can read it.
    const src = LOCAL_SOURCES[repo];
    fs.mkdirSync(path.join(modelsRoot, repo), { recursive: true });
    fs.copyFileSync(path.join(src, 'meta.json'), path.join(modelsRoot, repo, 'meta.json'));
    console.error(
      `no manifest entry for ${repo} — copied its meta.json; run ` +
        `\`npm run models:manifest-local -- ${repo}\`, then fetch again`,
    );
    process.exit(1);
  }
  const m = manifest[repo];
  if (!m) {
    console.error(
      `no manifest entry for ${repo} — run \`npm run models:manifest -- ${repo}\` first`,
    );
    process.exit(1);
  }
  const files = [...m.required, ...m.optional];
  const total = files.reduce((n, f) => n + f.bytes, 0);
  console.log(`${repo} → ${path.join(modelsRoot, repo)} (${fmtGB(total)})`);
  for (const f of files) {
    if (LOCAL_SOURCES[repo]) linkFile(repo, LOCAL_SOURCES[repo], f);
    else await fetchFile(repo, f);
  }
}
console.log('done');
