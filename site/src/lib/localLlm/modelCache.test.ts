import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// The ZEOS model's OPFS store, in memory: key `repo|revision|path` -> stored
// bytes (complete) or `.part` bytes (partial).
const opfs = vi.hoisted(() => {
  const complete = new Map<string, Uint8Array>();
  const partial = new Map<string, number>();
  const id = (key: string[]) => key.join('|');
  return {
    complete,
    partial,
    id,
    fail: false,
    store: {
      async complete(key: string[]) {
        if (opfs.fail) throw new Error('NotAllowedError');
        return complete.get(id(key))?.byteLength ?? null;
      },
      async partial(key: string[]) {
        return partial.get(id(key)) ?? 0;
      },
      async *chunks(key: string[]) {
        const bytes = complete.get(id(key));
        if (bytes) yield bytes;
      },
    },
  };
});
vi.mock('../zeos/vendor/opfs_store', () => ({ opfsStore: opfs.store }));
import {
  TRANSFORMERS_CACHE_NAME,
  hfFileUrl,
  isModelCached,
  localFileUrl,
  uncachedBytes,
} from './modelCache';
import { getLocalGemmaModel, type LocalGemmaModel } from './models';
import { requiredFiles } from './modelFiles';
import { ZEOS_STUB_STORAGE_KEY } from './models';

const model = getLocalGemmaModel('gemma-4-e2b') as LocalGemmaModel;
const g = globalThis as { caches?: unknown };
const originalCaches = g.caches;

function fakeCaches(present: Set<string>, opts: { openThrows?: boolean } = {}) {
  const opened: string[] = [];
  const api = {
    opened,
    open: async (name: string) => {
      opened.push(name);
      if (opts.openThrows) throw new Error('SecurityError');
      return {
        match: async (k: string) => (present.has(k) ? new Response() : undefined),
      };
    },
  };
  g.caches = api;
  return api;
}

function allRequiredUrls(): Set<string> {
  return new Set(requiredFiles(model).map((f) => hfFileUrl(model.hfRepoId, f.path)));
}

beforeEach(() => {
  vi.stubEnv('PUBLIC_LOCAL_MODELS', '');
});

afterEach(() => {
  vi.unstubAllEnvs();
  if (originalCaches === undefined) delete g.caches;
  else g.caches = originalCaches;
});

describe('URL helpers', () => {
  it('hfFileUrl matches the transformers.js cache key format', () => {
    expect(hfFileUrl('onnx-community/gemma-4-E2B-it-ONNX', 'onnx/x.onnx_data')).toBe(
      'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/x.onnx_data',
    );
  });

  it('localFileUrl points under /models/', () => {
    expect(localFileUrl('onnx-community/gemma-4-E2B-it-ONNX', 'config.json')).toBe(
      '/models/onnx-community/gemma-4-E2B-it-ONNX/config.json',
    );
  });
});

describe('isModelCached', () => {
  it('is true when every required file is cached', async () => {
    const api = fakeCaches(allRequiredUrls());
    expect(await isModelCached(model)).toBe(true);
    expect(api.opened).toEqual([TRANSFORMERS_CACHE_NAME]);
    expect(await uncachedBytes(model)).toBe(0);
  });

  it('is false when one required file is missing', async () => {
    const present = allRequiredUrls();
    const missing = requiredFiles(model).find((f) => f.path.endsWith('.onnx_data'))!;
    present.delete(hfFileUrl(model.hfRepoId, missing.path));
    fakeCaches(present);
    expect(await isModelCached(model)).toBe(false);
    expect(await uncachedBytes(model)).toBe(missing.bytes);
  });

  it('is false when Cache Storage is unavailable', async () => {
    delete g.caches;
    expect(await isModelCached(model)).toBe(false);
  });

  it('is false when caches.open throws', async () => {
    fakeCaches(allRequiredUrls(), { openThrows: true });
    expect(await isModelCached(model)).toBe(false);
  });

  it('is true in local-models mode without touching Cache Storage', async () => {
    vi.stubEnv('PUBLIC_LOCAL_MODELS', '1');
    const api = fakeCaches(new Set());
    expect(await isModelCached(model)).toBe(true);
    expect(api.opened).toEqual([]);
  });

  it('is false for a model with no manifest entry, sized by approxBytes', async () => {
    const unlisted: LocalGemmaModel = { ...model, hfRepoId: 'example/not-in-manifest' };
    // Even an "everything is cached" store must not count: there is no file list.
    const api = fakeCaches(new Set(), {});
    api.open = async () => ({ match: async () => new Response() });
    expect(await isModelCached(unlisted)).toBe(false);
    expect(await uncachedBytes(unlisted)).toBe(unlisted.approxBytes);
  });
});

describe('a model with hubSource (ZEOS Qwen 4B, kept in OPFS)', () => {
  const zeos = getLocalGemmaModel('zeos-qwen3.5-4b') as LocalGemmaModel;
  const src = zeos.hubSource!;
  const key = (path: string) => opfs.id([src.repo, src.revision, path]);
  const MB = 1_000; // kept small: the fake store holds real bytes
  // A small export shaped like the real one: the embedding's data and the
  // decoder's second shard are the same tied matrix (same SHA-256), and
  // config.json is listed but never loaded.
  const files: Record<string, { bytes: number; sha256: string }> = {
    'config.json': { bytes: 3, sha256: 'c' },
    'tokenizer.json': { bytes: 20 * MB, sha256: 't' },
    'tokenizer_config.json': { bytes: 9, sha256: 'tc' },
    'onnx/embed_tokens_q4f16.onnx': { bytes: 1, sha256: 'e' },
    'onnx/embed_tokens_q4f16.onnx_data': { bytes: 300 * MB, sha256: 'tied' },
    'onnx/decoder_zeos_q4f16.onnx': { bytes: 2, sha256: 'd' },
    'onnx/decoder_zeos_q4f16.onnx_data': { bytes: 2000 * MB, sha256: 'd0' },
    'onnx/decoder_zeos_q4f16.onnx_data_1': { bytes: 300 * MB, sha256: 'tied' },
  };
  const meta = {
    embedTokens: { file: 'onnx/embed_tokens_q4f16.onnx', externalData: ['onnx/embed_tokens_q4f16.onnx_data'] },
    decoder: {
      file: 'onnx/decoder_zeos_q4f16.onnx',
      externalData: ['onnx/decoder_zeos_q4f16.onnx_data', 'onnx/decoder_zeos_q4f16.onnx_data_1'],
    },
    files,
  };
  const store = (path: string, bytes = files[path].bytes) =>
    opfs.complete.set(key(path), new Uint8Array(bytes));
  const storeMeta = (m: unknown = meta) =>
    opfs.complete.set(key('meta.json'), new TextEncoder().encode(JSON.stringify(m)));
  /** Everything OptZeosWorker reads, the tied shard stored once under the embedding's name. */
  const storeLoaded = () => {
    storeMeta();
    for (const p of Object.keys(files)) {
      if (p !== 'config.json' && p !== 'onnx/decoder_zeos_q4f16.onnx_data_1') store(p);
    }
  };
  const manifestBytes = requiredFiles(zeos).reduce((n, f) => n + f.bytes, 0);

  beforeEach(() => {
    opfs.complete.clear();
    opfs.partial.clear();
    opfs.fail = false;
    vi.stubGlobal('navigator', { storage: { getDirectory: async () => ({}) } });
    // Cache Storage must never be asked about it.
    fakeCaches(new Set(), { openThrows: true });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('is cached when every file the model thread loads is stored, the tied shard once', async () => {
    storeLoaded();
    expect(await isModelCached(zeos)).toBe(true);
    expect(await uncachedBytes(zeos)).toBe(0);
  });

  it('counts the tied shard stored under either name', async () => {
    storeLoaded();
    opfs.complete.delete(key('onnx/embed_tokens_q4f16.onnx_data'));
    store('onnx/decoder_zeos_q4f16.onnx_data_1');
    expect(await isModelCached(zeos)).toBe(true);
  });

  it('is not cached, at the manifest size, with nothing stored', async () => {
    expect(await isModelCached(zeos)).toBe(false);
    expect(await uncachedBytes(zeos)).toBe(manifestBytes);
  });

  it('is not cached when a file is missing, and counts what it lacks', async () => {
    storeLoaded();
    opfs.complete.delete(key('onnx/decoder_zeos_q4f16.onnx_data'));
    expect(await isModelCached(zeos)).toBe(false);
    expect(await uncachedBytes(zeos)).toBe(2000 * MB);
  });

  it('subtracts what a .part already holds', async () => {
    storeLoaded();
    opfs.complete.delete(key('onnx/decoder_zeos_q4f16.onnx_data'));
    opfs.partial.set(key('onnx/decoder_zeos_q4f16.onnx_data'), 1500 * MB);
    expect(await isModelCached(zeos)).toBe(false);
    expect(await uncachedBytes(zeos)).toBe(500 * MB);
  });

  it('does not count a stored file of the wrong size', async () => {
    storeLoaded();
    store('tokenizer.json', 5);
    expect(await isModelCached(zeos)).toBe(false);
    expect(await uncachedBytes(zeos)).toBe(20 * MB);
  });

  it('does not count files stored under another revision', async () => {
    storeLoaded();
    for (const [k, v] of [...opfs.complete]) {
      opfs.complete.delete(k);
      opfs.complete.set(k.replace(src.revision, '0'.repeat(40)), v);
    }
    expect(await isModelCached(zeos)).toBe(false);
    expect(await uncachedBytes(zeos)).toBe(manifestBytes);
  });

  it('is not cached, at full size, when the store throws or meta.json is unreadable', async () => {
    storeLoaded();
    opfs.fail = true;
    expect(await isModelCached(zeos)).toBe(false);
    expect(await uncachedBytes(zeos)).toBe(manifestBytes);
    opfs.fail = false;
    opfs.complete.set(key('meta.json'), new TextEncoder().encode('{not json'));
    expect(await isModelCached(zeos)).toBe(false);
    expect(await uncachedBytes(zeos)).toBe(manifestBytes);
    storeMeta({ ...meta, files: {} });
    expect(await isModelCached(zeos)).toBe(false);
  });

  it('is not cached when OPFS is unavailable', async () => {
    storeLoaded();
    vi.stubGlobal('navigator', {});
    expect(await isModelCached(zeos)).toBe(false);
    expect(await uncachedBytes(zeos)).toBe(manifestBytes);
  });

  it('needs no download in local-models mode', async () => {
    vi.stubEnv('PUBLIC_LOCAL_MODELS', '1');
    expect(await isModelCached(zeos)).toBe(true);
    expect(await uncachedBytes(zeos)).toBe(0);
  });

  it('needs no download in ZEOS stub mode', async () => {
    const items = new Map([[ZEOS_STUB_STORAGE_KEY, '{"replies":[]}']]);
    vi.stubGlobal('localStorage', { getItem: (k: string) => items.get(k) ?? null });
    expect(await isModelCached(zeos)).toBe(true);
    expect(await uncachedBytes(zeos)).toBe(0);
  });
});
