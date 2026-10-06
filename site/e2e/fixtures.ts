import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { test as base, type Route } from '@playwright/test';

export { expect, type Page } from '@playwright/test';

/**
 * Every spec imports `test` / `expect` from here. The one fixture it adds
 * serves jsDelivr (DuckDB-wasm's 34 MB `duckdb-eh.wasm`, Pyodide's ~12 MB)
 * from a cache instead of downloading it again for every test: each URL is
 * fetched once with `route.fetch()`, then kept in memory for the worker and
 * on disk under `node_modules/.cache/e2e-cdn/` across runs. The URLs are
 * versioned, so an entry never goes stale; delete the directory to clear it.
 * `context.route` also sees the requests that Web Workers make (DuckDB's
 * worker, the ZEOS kernel's Pyodide). Only jsDelivr is cached: remote CSVs
 * and every other host go to the network as before. `GDA_E2E_CDN_CACHE=0`
 * turns the cache off.
 */

const CDN = 'https://cdn.jsdelivr.net/**';
const ENABLED = process.env.GDA_E2E_CDN_CACHE !== '0';
const CACHE_DIR = fileURLToPath(new URL('../node_modules/.cache/e2e-cdn/', import.meta.url));

// Hop-by-hop or describing the wire encoding, which `route.fetch()` has
// already undone: the body handed to `route.fulfill` is the decoded one.
const DROP_HEADERS = new Set([
  'connection',
  'content-encoding',
  'content-length',
  'keep-alive',
  'set-cookie',
  'transfer-encoding',
]);

interface Entry {
  status: number;
  headers: Record<string, string>;
  body: Buffer;
}

const memory = new Map<string, Promise<Entry>>();

function diskPaths(url: string): { meta: string; body: string } {
  const key = createHash('sha256').update(url).digest('hex');
  return { meta: path.join(CACHE_DIR, `${key}.json`), body: path.join(CACHE_DIR, `${key}.body`) };
}

function readDisk(url: string): Entry | null {
  const p = diskPaths(url);
  try {
    const meta = JSON.parse(fs.readFileSync(p.meta, 'utf8')) as { url: string; status: number; headers: Record<string, string> };
    if (meta.url !== url) return null;
    return { status: meta.status, headers: meta.headers, body: fs.readFileSync(p.body) };
  } catch {
    return null;
  }
}

/** Write body then meta, each via a rename, so a parallel reader never sees half an entry. */
function writeDisk(url: string, entry: Entry): void {
  const p = diskPaths(url);
  const tmp = `.${process.pid}.${Date.now()}.tmp`;
  try {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(p.body + tmp, entry.body);
    fs.renameSync(p.body + tmp, p.body);
    fs.writeFileSync(p.meta + tmp, JSON.stringify({ url, status: entry.status, headers: entry.headers }));
    fs.renameSync(p.meta + tmp, p.meta);
  } catch {
    // A cache that cannot be written just means the next run downloads again.
  }
}

async function download(route: Route): Promise<Entry> {
  const res = await route.fetch();
  const headers: Record<string, string> = {};
  for (const [k, v] of Object.entries(res.headers())) {
    if (!DROP_HEADERS.has(k)) headers[k] = v;
  }
  return { status: res.status(), headers, body: await res.body() };
}

async function serveFromCache(route: Route): Promise<void> {
  const req = route.request();
  // Ranged or non-GET requests are rare here; let them through untouched.
  if (req.method() !== 'GET' || req.headers()['range']) return route.fallback();
  const url = req.url();
  let pending = memory.get(url);
  if (!pending) {
    const fromDisk = readDisk(url);
    pending = fromDisk
      ? Promise.resolve(fromDisk)
      : download(route).then((entry) => {
          if (entry.status === 200) writeDisk(url, entry);
          else memory.delete(url);
          return entry;
        });
    memory.set(url, pending);
    pending.catch(() => memory.delete(url));
  }
  let entry: Entry;
  try {
    entry = await pending;
  } catch {
    // The download failed (offline, or the page went away): let the browser try.
    return route.fallback().catch(() => undefined);
  }
  await route.fulfill({ status: entry.status, headers: entry.headers, body: entry.body }).catch(() => undefined);
}

export const test = base.extend<{ cdnCache: void }>({
  cdnCache: [
    async ({ context }, use) => {
      if (ENABLED) await context.route(CDN, serveFromCache);
      await use();
    },
    { auto: true },
  ],
});
