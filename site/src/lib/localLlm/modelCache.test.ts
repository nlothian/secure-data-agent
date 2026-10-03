import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  TRANSFORMERS_CACHE_NAME,
  hfFileUrl,
  isModelCached,
  localFileUrl,
  uncachedBytes,
} from './modelCache';
import { getLocalGemmaModel, type LocalGemmaModel } from './models';
import { requiredFiles } from './modelFiles';

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
});
