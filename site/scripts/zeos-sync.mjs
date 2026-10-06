#!/usr/bin/env node
/**
 * Pull what the browser ZEOS kernel needs out of a ZEOS checkout.
 *
 *   ZEOS_REPO=/path/to/zeos npm run zeos:sync
 *
 * `ZEOS_REPO` is required: there is no default checkout, because building
 * the wrong one (a checkout with someone else's uncommitted edits, or a stale
 * branch) is silent. CLAUDE.md names the checkout to sync from.
 *
 * 1. `uv build --wheel` for the `zeos` and `zeos-coop-count-web` packages,
 *    copied to `public/zeos/wheels/<sha256[0:12]>/<wheel>`. The directory is
 *    content-addressed because both wheels keep version 0.1.0 while their code
 *    changes, and micropip needs the URL to end in the real wheel filename.
 * 2. Every case directory under the case roots below, copied to
 *    `public/zeos/cases/<case>/`.
 * 3. The JavaScript modules the site imports, copied into
 *    `src/lib/zeos/vendor/` with a header naming the source commit.
 * 4. `public/zeos/manifest.json`: wheel paths + sha256 (the kernel worker
 *    checks each wheel against it before installing), case file lists, and
 *    the ZEOS origin remote, branch, commit, and whether its tree was dirty.
 *    It is served with the site, so it holds no local paths.
 *    `src/lib/zeos/vendor/SOURCE.json` (committed, not served) also records
 *    the checkout path that was synced.
 *
 * `public/zeos/` is generated and gitignored; `src/lib/zeos/vendor/` is
 * committed (see CLAUDE.md).
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const siteRoot = path.resolve(__dirname, '..');
if (!process.env.ZEOS_REPO) {
  console.error(
    'Set ZEOS_REPO to the ZEOS checkout to build from, e.g.\n' +
      '  ZEOS_REPO=/path/to/zeos npm run zeos:sync\n' +
      '(CLAUDE.md, "ZEOS kernel (browser)", names the checkout this site runs.)',
  );
  process.exit(1);
}
const zeosRepo = path.resolve(process.env.ZEOS_REPO);
const publicOut = path.join(siteRoot, 'public', 'zeos');
const vendorOut = path.join(siteRoot, 'src', 'lib', 'zeos', 'vendor');

const PACKAGES = ['zeos', 'zeos-coop-count-web'];
/** Directories whose immediate subdirectories are cases. Missing ones are skipped. */
const CASE_ROOTS = ['demo/coop-count/cases', 'demo/coop-count-web/cases'];
const WEB = 'demo/coop-count-web/web';
/**
 * JS modules vendored verbatim (plus a header). `opt_zeos_worker.js` is the
 * real model (src/workers/zeosOptModel.worker.ts loads it); it imports
 * `encodePlain` and `sampleToken` from `transformers_worker.js`. Missing
 * optional files are skipped.
 */
const VENDOR = [
  { file: 'frames.js', required: true },
  { file: 'model_channel.js', required: true },
  { file: 'stub_worker.js', required: true },
  { file: 'opt_zeos_worker.js', required: true },
  { file: 'transformers_worker.js', required: true },
];

function git(...args) {
  return execFileSync('git', args, { cwd: zeosRepo, encoding: 'utf8' }).trim();
}

function sha256(file) {
  return createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

function listFiles(dir, base = dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...listFiles(full, base));
    else if (entry.isFile()) out.push(path.relative(base, full).split(path.sep).join('/'));
  }
  return out.sort();
}

if (!fs.existsSync(path.join(zeosRepo, 'pyproject.toml'))) {
  console.error(`ZEOS_REPO=${zeosRepo} is not a ZEOS checkout (no pyproject.toml)`);
  process.exit(1);
}

const commit = git('rev-parse', 'HEAD');
let remote = null;
try {
  remote = git('remote', 'get-url', 'origin') || null;
} catch {
  // No origin: the path alone names the checkout.
}
const branch = git('rev-parse', '--abbrev-ref', 'HEAD');
// uv.lock churn alone does not change what is built.
const dirty = git('status', '--porcelain', '--', 'src', 'demo')
  .split('\n')
  .filter(Boolean);
console.log(`zeos ${branch}@${commit.slice(0, 12)}${dirty.length ? ' (dirty)' : ''}`);

// 1. wheels
fs.rmSync(publicOut, { recursive: true, force: true });
fs.mkdirSync(path.join(publicOut, 'wheels'), { recursive: true });
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'zeos-wheels-'));
for (const pkg of PACKAGES) {
  execFileSync('uv', ['build', '--wheel', '--package', pkg, '--out-dir', tmp], {
    cwd: zeosRepo,
    stdio: 'inherit',
  });
}
const wheels = [];
for (const name of fs.readdirSync(tmp).filter((n) => n.endsWith('.whl')).sort()) {
  const digest = sha256(path.join(tmp, name));
  const rel = `wheels/${digest.slice(0, 12)}/${name}`;
  fs.mkdirSync(path.join(publicOut, path.dirname(rel)), { recursive: true });
  fs.copyFileSync(path.join(tmp, name), path.join(publicOut, rel));
  wheels.push({ name, path: rel, sha256: digest, bytes: fs.statSync(path.join(tmp, name)).size });
}
fs.rmSync(tmp, { recursive: true, force: true });
if (wheels.length !== PACKAGES.length) {
  console.error(`expected ${PACKAGES.length} wheels, got ${wheels.map((w) => w.name)}`);
  process.exit(1);
}

// 2. cases
const cases = {};
for (const root of CASE_ROOTS) {
  const dir = path.join(zeosRepo, root);
  if (!fs.existsSync(dir)) continue;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    if (cases[entry.name]) {
      console.error(`case ${entry.name} appears under two case roots`);
      process.exit(1);
    }
    const src = path.join(dir, entry.name);
    fs.cpSync(src, path.join(publicOut, 'cases', entry.name), { recursive: true });
    cases[entry.name] = listFiles(src);
  }
}

// 3. vendored JS
fs.mkdirSync(vendorOut, { recursive: true });
const vendored = {};
for (const { file, required } of VENDOR) {
  const src = path.join(zeosRepo, WEB, file);
  if (!fs.existsSync(src)) {
    if (required) {
      console.error(`missing ${WEB}/${file}`);
      process.exit(1);
    }
    continue;
  }
  const body = fs.readFileSync(src, 'utf8');
  const header =
    `// Vendored from ZEOS ${WEB}/${file}\n` +
    `// at ${commit}${dirty.length ? ' (with uncommitted changes)' : ''} by site/scripts/zeos-sync.mjs.\n` +
    `// Do not edit here; change it in ZEOS and re-run \`npm run zeos:sync\`.\n`;
  fs.writeFileSync(path.join(vendorOut, file), header + body);
  vendored[file] = createHash('sha256').update(body).digest('hex');
}
fs.writeFileSync(
  path.join(vendorOut, 'SOURCE.json'),
  JSON.stringify(
    { repo: zeosRepo, remote, branch, commit, dirty: dirty.length > 0, files: vendored },
    null,
    2,
  ) + '\n',
);

// 4. manifest
const manifest = {
  // No `repo`: this file is served, and a local path means nothing to a visitor.
  zeos: { remote, branch, commit, dirty: dirty.length > 0 },
  wheels,
  cases,
};
fs.writeFileSync(path.join(publicOut, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');

console.log(
  `wrote public/zeos: ${wheels.length} wheels, ${Object.keys(cases).length} cases; ` +
    `vendored ${Object.keys(vendored).join(', ')}`,
);
