import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadEnv } from 'vite';

/**
 * Dev-only static server for the gitignored repo-root `models/` folder.
 *
 * When `PUBLIC_LOCAL_MODELS=1`, `/models/<hfRepoId>/<file>` is served straight
 * from `<repoRoot>/models/<hfRepoId>/<file>` and the LLM worker points
 * transformers.js at `/models/` instead of the Hugging Face Hub. This keeps
 * multi-GB model downloads out of the browser cache during dev and e2e.
 *
 * Why a middleware rather than a `public/models` symlink: `astro build`
 * copies `public/` into `dist/` (following symlinks), and Vite crawls
 * `publicDir` at startup. Why not `server.fs.allow` + `/@fs/` URLs: Vite's
 * transform middleware would turn `config.json` into a JS module instead of
 * returning the raw JSON transformers.js expects.
 */

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Plugin lives at site/scripts/, so repo root is two levels up.
const repoRoot = path.resolve(__dirname, '..', '..');
const modelsRoot = path.resolve(repoRoot, 'models');

function contentTypeFor(filePath) {
  if (filePath.endsWith('.json')) return 'application/json';
  if (filePath.endsWith('.jinja') || filePath.endsWith('.txt')) {
    return 'text/plain; charset=utf-8';
  }
  return 'application/octet-stream';
}

function isEnabled(mode) {
  // Honour both a shell export and a `site/.env` entry. Astro only copies
  // `.env` into `process.env` during `astro build`, so in dev we read the
  // file through Vite's loader to match what `import.meta.env` will see.
  if (process.env.PUBLIC_LOCAL_MODELS === '1') return true;
  try {
    const env = loadEnv(mode ?? 'development', path.resolve(__dirname, '..'), 'PUBLIC_');
    return env.PUBLIC_LOCAL_MODELS === '1';
  } catch {
    return false;
  }
}

export default function localModelsPlugin() {
  let mode = 'development';
  return {
    name: 'gda-local-models',
    apply: 'serve',
    configResolved(config) {
      mode = config.mode;
    },
    configureServer(server) {
      if (!isEnabled(mode)) return;

      if (!fs.existsSync(modelsRoot)) {
        server.config.logger.warn(
          `[local-models] PUBLIC_LOCAL_MODELS=1 but ${modelsRoot} does not exist. ` +
            `Run \`npm run models:fetch -- e2b\` first.`,
        );
      } else {
        server.config.logger.info(
          `[local-models] serving ${modelsRoot} at /models/`,
        );
      }

      server.middlewares.use('/models', (req, res, next) => {
        if (req.method !== 'GET' && req.method !== 'HEAD') {
          next();
          return;
        }
        // `req.url` is relative to the mount point ("/models").
        const rawPath = (req.url ?? '/').split('?')[0];
        let decoded;
        try {
          decoded = decodeURIComponent(rawPath);
        } catch {
          res.statusCode = 400;
          res.end('bad path');
          return;
        }
        const resolved = path.resolve(modelsRoot, '.' + decoded);
        if (!resolved.startsWith(modelsRoot + path.sep)) {
          res.statusCode = 403;
          res.end('forbidden');
          return;
        }
        let stat;
        try {
          stat = fs.statSync(resolved);
        } catch {
          res.statusCode = 404;
          res.end('not found');
          return;
        }
        if (!stat.isFile()) {
          res.statusCode = 404;
          res.end('not found');
          return;
        }
        res.statusCode = 200;
        res.setHeader('Content-Type', contentTypeFor(resolved));
        res.setHeader('Content-Length', String(stat.size));
        // Keep Chrome's HTTP cache from storing multi-GB copies.
        res.setHeader('Cache-Control', 'no-store');
        res.setHeader('Access-Control-Allow-Origin', '*');
        if (req.method === 'HEAD') {
          res.end();
          return;
        }
        const stream = fs.createReadStream(resolved);
        stream.on('error', () => {
          if (!res.headersSent) res.statusCode = 500;
          res.end();
        });
        stream.pipe(res);
      });
    },
  };
}
