import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LlmWorkerIn, LlmWorkerOut, GenerateStats } from './llmWorkerProtocol';

const setLocalLlmDownloadProgress = vi.fn();

vi.mock('../executionPanelStore', () => ({
  setLocalLlmDownloadProgress: (...args: unknown[]) => setLocalLlmDownloadProgress(...args),
}));
vi.mock('./modelCache', () => ({
  isModelCached: async () => true,
}));
let gpuStatus: { supported: boolean; reason?: string; f16?: boolean } = {
  supported: true,
  f16: true,
};
vi.mock('./webgpu', () => ({
  detectWebGpu: async () => gpuStatus,
}));

class FakeWorker {
  static instances: FakeWorker[] = [];
  posted: LlmWorkerIn[] = [];
  onmessage: ((ev: { data: LlmWorkerOut }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  constructor(
    public url: URL,
    public opts: unknown,
  ) {
    FakeWorker.instances.push(this);
  }
  postMessage(msg: LlmWorkerIn): void {
    this.posted.push(msg);
  }
  terminated = false;
  terminate(): void {
    this.terminated = true;
  }
  emit(msg: LlmWorkerOut): void {
    this.onmessage?.({ data: msg });
  }
  last<T extends LlmWorkerIn['type']>(type: T): Extract<LlmWorkerIn, { type: T }> {
    const found = [...this.posted].reverse().find((m) => m.type === type);
    if (!found) throw new Error(`no ${type} posted; got ${JSON.stringify(this.posted)}`);
    return found as Extract<LlmWorkerIn, { type: T }>;
  }
  all<T extends LlmWorkerIn['type']>(type: T): Extract<LlmWorkerIn, { type: T }>[] {
    return this.posted.filter((m) => m.type === type) as Extract<LlmWorkerIn, { type: T }>[];
  }
}

// Works with real and fake timers alike.
const tick = async () => {
  if (vi.isFakeTimers()) await vi.advanceTimersByTimeAsync(0);
  else await new Promise<void>((r) => setTimeout(r, 0));
};

const STATS: GenerateStats = {
  promptTokens: 12,
  outputTokens: 3,
  decodeMs: 30,
  reusedTokens: 0,
  prefillTokens: 12,
  reason: 'eos',
};

type Svc = typeof import('./llmService');

async function freshService(): Promise<{ svc: Svc; worker: () => FakeWorker }> {
  vi.resetModules();
  const svc = await import('./llmService');
  return {
    svc,
    worker: () => {
      const w = FakeWorker.instances.at(-1);
      if (!w) throw new Error('worker not created');
      return w;
    },
  };
}

async function loadModel(svc: Svc, worker: () => FakeWorker, modelId = 'gemma-4-e2b') {
  const p = svc.ensureLoaded(modelId);
  await tick();
  const w = worker();
  const load = w.last('load');
  w.emit({
    type: 'loaded',
    id: load.id,
    info: { hfId: load.hfId, eosIds: [1, 106], files: ['config.json'] },
  });
  await p;
  return w;
}

const debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});

beforeEach(() => {
  debugSpy.mockClear();
  gpuStatus = { supported: true, f16: true };
  FakeWorker.instances = [];
  setLocalLlmDownloadProgress.mockClear();
  vi.stubGlobal('Worker', FakeWorker);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('llmService.ensureLoaded', () => {
  it('creates one module worker, posts load, and resolves on loaded', async () => {
    const { svc, worker } = await freshService();
    const p = svc.ensureLoaded('gemma-4-e2b');
    await tick();
    const w = worker();
    expect(String(w.url)).toMatch(/workers\/llm\.worker\.ts$/);
    expect(w.opts).toEqual({ type: 'module' });
    const load = w.last('load');
    expect(load.hfId).toBe('onnx-community/gemma-4-E2B-it-ONNX');

    // Initial snapshot is published before any event arrives.
    expect(setLocalLlmDownloadProgress).toHaveBeenCalledWith(
      expect.objectContaining({ label: 'Gemma 4 E2B', pct: 0, fromCache: true, phase: 'fetch' }),
    );

    w.emit({
      type: 'progress',
      id: load.id,
      event: { status: 'progress', file: 'tokenizer.json', loaded: 100, total: 200 },
    });
    // Every required file done → init phase.
    for (const file of [
      'config.json',
      'tokenizer.json',
      'tokenizer_config.json',
      'onnx/embed_tokens_q4f16.onnx',
      'onnx/embed_tokens_q4f16.onnx_data',
      'onnx/decoder_model_merged_q4f16.onnx',
      'onnx/decoder_model_merged_q4f16.onnx_data',
      'generation_config.json',
    ]) {
      w.emit({ type: 'progress', id: load.id, event: { status: 'done', file } });
    }
    expect(setLocalLlmDownloadProgress).toHaveBeenLastCalledWith(
      expect.objectContaining({ pct: 100, phase: 'init' }),
    );

    let resolved = false;
    void p.then(() => (resolved = true));
    await tick();
    expect(resolved).toBe(false);

    w.emit({
      type: 'loaded',
      id: load.id,
      info: { hfId: load.hfId, eosIds: [1, 106], files: ['config.json'] },
    });
    await p;
    expect(svc.getLoadedModelId()).toBe('gemma-4-e2b');
    expect(setLocalLlmDownloadProgress).toHaveBeenLastCalledWith(null);

    // Already loaded → no second load posted.
    await svc.ensureLoaded('gemma-4-e2b');
    expect(w.all('load')).toHaveLength(1);
    expect(FakeWorker.instances).toHaveLength(1);
  });

  it('rejects unknown model ids without touching the worker', async () => {
    const { svc } = await freshService();
    await expect(svc.ensureLoaded('custom:nope')).rejects.toThrow(/Unknown local model/);
    expect(FakeWorker.instances).toHaveLength(0);
  });

  it('rejects a superseded load and frees nothing the newer load still needs', async () => {
    const { svc, worker } = await freshService();
    const first = svc.ensureLoaded('gemma-4-e2b');
    const firstSettled = first.catch((e: unknown) => e);
    await tick();
    const w = worker();
    const firstLoad = w.last('load');

    const second = svc.ensureLoaded('gemma-4-e4b');
    await tick();
    const secondLoad = w.last('load');
    expect(secondLoad.id).not.toBe(firstLoad.id);
    expect(secondLoad.hfId).toBe('onnx-community/gemma-4-E4B-it-ONNX');

    w.emit({
      type: 'loaded',
      id: firstLoad.id,
      info: { hfId: firstLoad.hfId, eosIds: [106], files: [] },
    });
    const err = await firstSettled;
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).message).toBe('Model load superseded.');
    // A newer load is queued in the worker — it replaces the stale model, so
    // no explicit dispose (which would run after, and kill, the newer load).
    expect(w.all('dispose')).toHaveLength(0);

    w.emit({
      type: 'loaded',
      id: secondLoad.id,
      info: { hfId: secondLoad.hfId, eosIds: [106], files: [] },
    });
    await second;
    expect(svc.getLoadedModelId()).toBe('gemma-4-e4b');
  });

  it('rejects with the worker message on load error', async () => {
    const { svc, worker } = await freshService();
    const p = svc.ensureLoaded('gemma-4-e2b');
    await tick();
    const w = worker();
    w.emit({
      type: 'error',
      id: w.last('load').id,
      code: 'tokenizer-mismatch',
      message: 'Tokenizer check failed: nope',
    });
    await expect(p).rejects.toThrow('Tokenizer check failed: nope');
    expect(svc.getLoadedModelId()).toBeNull();
    expect(setLocalLlmDownloadProgress).toHaveBeenLastCalledWith(null);
  });
});

describe('llmService.generate', () => {
  it('streams tokens routed by request id and resolves with text + stats', async () => {
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const deltas: string[] = [];
    const onStats = vi.fn();
    const p = svc.generate({ prompt: 'P', onToken: (d) => deltas.push(d), onStats });
    await tick();
    const gen = w.last('generate');
    expect(gen.prompt).toBe('P');

    w.emit({ type: 'token', id: gen.id, text: 'Hel' });
    w.emit({ type: 'token', id: gen.id + 1000, text: 'IGNORED' });
    w.emit({ type: 'token', id: gen.id, text: 'lo' });
    w.emit({ type: 'done', id: gen.id, text: 'Hello', stats: STATS });

    await expect(p).resolves.toBe('Hello');
    expect(deltas).toEqual(['Hel', 'lo']);
    expect(onStats).toHaveBeenCalledWith(STATS);
    // DEV-only visibility of KV reuse per generation.
    expect(debugSpy).toHaveBeenCalledWith(
      expect.stringMatching(/^\[llmService\] generate stats \{.*"reusedTokens":0.*"prefillTokens":12/),
    );
  });

  it('abort posts cancel and resolves with the partial text', async () => {
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const ctrl = new AbortController();
    const deltas: string[] = [];
    const p = svc.generate({ prompt: 'P', signal: ctrl.signal, onToken: (d) => deltas.push(d) });
    await tick();
    const gen = w.last('generate');
    w.emit({ type: 'token', id: gen.id, text: 'part' });

    ctrl.abort();
    expect(w.last('cancel').id).toBe(gen.id);
    // Tokens that race the cancel are dropped.
    w.emit({ type: 'token', id: gen.id, text: 'LATE' });
    w.emit({
      type: 'done',
      id: gen.id,
      text: 'partLATE',
      stats: { ...STATS, reason: 'interrupted' },
    });
    await expect(p).resolves.toBe('part');
    expect(deltas).toEqual(['part']);
  });

  it('resolves "" without posting when the signal is already aborted', async () => {
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(svc.generate({ prompt: 'P', signal: ctrl.signal, onToken: () => {} })).resolves.toBe(
      '',
    );
    expect(w.all('generate')).toHaveLength(0);
  });

  it('rejects with ContextTooLongError on context-too-long', async () => {
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const p = svc.generate({ prompt: 'P', onToken: () => {} });
    const settled = p.catch((e: unknown) => e);
    await tick();
    w.emit({
      type: 'error',
      id: w.last('generate').id,
      code: 'context-too-long',
      message: 'too long',
      data: { promptTokens: 25_000, limit: 20_000 },
    });
    const err = await settled;
    expect(err).toBeInstanceOf(svc.ContextTooLongError);
    expect(err).toMatchObject({ name: 'ContextTooLongError', promptTokens: 25_000, limit: 20_000 });
    expect(svc.isInputTooLongError(err)).toBe(true);

    const plain = new Error('x');
    plain.name = 'ContextTooLongError';
    expect(svc.isInputTooLongError(plain)).toBe(true);
    expect(svc.isInputTooLongError(new Error('Input is too long'))).toBe(false);
  });

  it('a second generate waits for the first to finish', async () => {
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const a = svc.generate({ prompt: 'A', onToken: () => {} });
    const b = svc.generate({ prompt: 'B', onToken: () => {} });
    await tick();
    expect(w.all('generate').map((g) => g.prompt)).toEqual(['A']);

    const genA = w.last('generate');
    w.emit({ type: 'token', id: genA.id, text: 'a' });
    w.emit({ type: 'done', id: genA.id, text: 'a', stats: STATS });
    await expect(a).resolves.toBe('a');
    await tick();
    expect(w.all('generate').map((g) => g.prompt)).toEqual(['A', 'B']);

    const genB = w.last('generate');
    w.emit({ type: 'token', id: genB.id, text: 'b' });
    w.emit({ type: 'done', id: genB.id, text: 'b', stats: STATS });
    await expect(b).resolves.toBe('b');
  });

  it('an onToken throw cancels decode and rejects once the worker is done', async () => {
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const boom = new Error('boom');
    const p = svc.generate({
      prompt: 'P',
      onToken: () => {
        throw boom;
      },
    });
    const settled = p.catch((e: unknown) => e);
    await tick();
    const gen = w.last('generate');
    w.emit({ type: 'token', id: gen.id, text: 'x' });
    expect(w.last('cancel').id).toBe(gen.id);
    w.emit({ type: 'done', id: gen.id, text: 'x', stats: { ...STATS, reason: 'interrupted' } });
    expect(await settled).toBe(boom);
  });

  it('cancel() interrupts the in-flight generation', async () => {
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const p = svc.generate({ prompt: 'P', onToken: () => {} });
    await tick();
    const gen = w.last('generate');
    svc.cancel();
    expect(w.last('cancel').id).toBe(gen.id);
    w.emit({ type: 'done', id: gen.id, text: '', stats: { ...STATS, reason: 'interrupted' } });
    await expect(p).resolves.toBe('');
  });

  it('throws when no model is loaded', async () => {
    const { svc } = await freshService();
    await expect(svc.generate({ prompt: 'P', onToken: () => {} })).rejects.toThrow(/not loaded/);
  });
});

describe('llmService.sizeInTokens / dispose', () => {
  it('returns null before load and the worker count after', async () => {
    const { svc, worker } = await freshService();
    await expect(svc.sizeInTokens('hi')).resolves.toBeNull();
    const w = await loadModel(svc, worker);
    const p = svc.sizeInTokens('hi');
    const count = w.last('count');
    expect(count.text).toBe('hi');
    w.emit({ type: 'count', id: count.id, tokens: 3 });
    await expect(p).resolves.toBe(3);
  });

  it('dispose posts dispose and clears the loaded id', async () => {
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const p = svc.dispose();
    const d = w.last('dispose');
    w.emit({ type: 'disposed', id: d.id });
    await p;
    expect(svc.getLoadedModelId()).toBeNull();
  });
});

describe('llmService WebGPU gate', () => {
  it('rejects without shader-f16 and clears the throbber', async () => {
    gpuStatus = { supported: true, f16: false };
    const { svc } = await freshService();
    await expect(svc.ensureLoaded('gemma-4-e2b')).rejects.toThrow(/shader-f16/);
    expect(FakeWorker.instances).toHaveLength(0);
    expect(setLocalLlmDownloadProgress).toHaveBeenLastCalledWith(null);
  });

  it('rejects with the detector reason when WebGPU is unavailable', async () => {
    gpuStatus = { supported: false, reason: 'No WebGPU here.' };
    const { svc } = await freshService();
    await expect(svc.ensureLoaded('gemma-4-e2b')).rejects.toThrow('No WebGPU here.');
    expect(FakeWorker.instances).toHaveLength(0);
  });
});

describe('llmService worker recycling', () => {
  it('a fetch-phase stall recycles the worker; the next load starts a fresh one', async () => {
    vi.useFakeTimers();
    const { svc, worker } = await freshService();
    const p = svc.ensureLoaded('gemma-4-e2b');
    const settled = p.catch((e: unknown) => e);
    await tick();
    const w1 = worker();
    const load = w1.last('load');

    // Progress keeps the watchdog quiet…
    await vi.advanceTimersByTimeAsync(svc.LOAD_STALL_TIMEOUT_MS - 1000);
    w1.emit({
      type: 'progress',
      id: load.id,
      event: { status: 'progress', file: 'tokenizer.json', loaded: 1, total: 2 },
    });
    await vi.advanceTimersByTimeAsync(svc.LOAD_STALL_TIMEOUT_MS - 1000);
    expect(w1.terminated).toBe(false);

    // …silence for the full window trips it.
    await vi.advanceTimersByTimeAsync(2000);
    const err = await settled;
    expect((err as Error).message).toMatch(/stalled/);
    expect(w1.terminated).toBe(true);
    expect(svc.getLoadedModelId()).toBeNull();
    expect(setLocalLlmDownloadProgress).toHaveBeenLastCalledWith(null);

    const w2 = await loadModel(svc, worker);
    expect(w2).not.toBe(w1);
    expect(FakeWorker.instances).toHaveLength(2);
    expect(svc.getLoadedModelId()).toBe('gemma-4-e2b');
  });

  it('does not treat the silent session-init phase as a stall', async () => {
    vi.useFakeTimers();
    const { svc, worker } = await freshService();
    const p = svc.ensureLoaded('gemma-4-e2b');
    await tick();
    const w = worker();
    const load = w.last('load');
    for (const file of [
      'config.json',
      'tokenizer.json',
      'tokenizer_config.json',
      'onnx/embed_tokens_q4f16.onnx',
      'onnx/embed_tokens_q4f16.onnx_data',
      'onnx/decoder_model_merged_q4f16.onnx',
      'onnx/decoder_model_merged_q4f16.onnx_data',
      'generation_config.json',
    ]) {
      w.emit({ type: 'progress', id: load.id, event: { status: 'done', file } });
    }
    await vi.advanceTimersByTimeAsync(svc.LOAD_STALL_TIMEOUT_MS * 3);
    expect(w.terminated).toBe(false);
    w.emit({ type: 'loaded', id: load.id, info: { hfId: load.hfId, eosIds: [106], files: [] } });
    await p;
    expect(svc.getLoadedModelId()).toBe('gemma-4-e2b');
  });

  it('load-failed recycles the worker and a later ensureLoaded creates a new one', async () => {
    const { svc, worker } = await freshService();
    const p = svc.ensureLoaded('gemma-4-e2b');
    const settled = p.catch((e: unknown) => e);
    await tick();
    const w1 = worker();
    w1.emit({
      type: 'error',
      id: w1.last('load').id,
      code: 'load-failed',
      message: 'onnx_data fetch failed',
    });
    expect(((await settled) as Error).message).toBe('onnx_data fetch failed');
    expect(w1.terminated).toBe(true);

    const w2 = await loadModel(svc, worker);
    expect(w2).not.toBe(w1);
    expect(w2.all('load')).toHaveLength(1);
  });

  it('tokenizer-mismatch does not recycle the worker', async () => {
    const { svc, worker } = await freshService();
    const p = svc.ensureLoaded('gemma-4-e2b');
    const settled = p.catch((e: unknown) => e);
    await tick();
    const w = worker();
    w.emit({ type: 'error', id: w.last('load').id, code: 'tokenizer-mismatch', message: 'bad' });
    await settled;
    expect(w.terminated).toBe(false);
  });

  it('a superseded reply from the worker rejects the stale load only', async () => {
    const { svc, worker } = await freshService();
    const first = svc.ensureLoaded('gemma-4-e2b').catch((e: unknown) => e);
    await tick();
    const w = worker();
    const firstLoad = w.last('load');
    const second = svc.ensureLoaded('gemma-4-e4b');
    await tick();
    const secondLoad = w.last('load');
    w.emit({ type: 'error', id: firstLoad.id, code: 'load-superseded', message: 'superseded' });
    expect(((await first) as Error).message).toBe('Model load superseded.');
    expect(w.terminated).toBe(false);
    w.emit({ type: 'loaded', id: secondLoad.id, info: { hfId: secondLoad.hfId, eosIds: [106], files: [] } });
    await second;
    expect(svc.getLoadedModelId()).toBe('gemma-4-e4b');
  });
});

describe('llmService generation watchdogs', () => {
  it('recycles the worker when a cancel is not acknowledged, resolving with partial text', async () => {
    vi.useFakeTimers();
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const ctrl = new AbortController();
    const p = svc.generate({ prompt: 'P', signal: ctrl.signal, onToken: () => {} });
    await tick();
    const gen = w.last('generate');
    w.emit({ type: 'token', id: gen.id, text: 'par' });
    ctrl.abort();
    expect(w.last('cancel').id).toBe(gen.id);

    await vi.advanceTimersByTimeAsync(svc.ABORT_WATCHDOG_MS - 1);
    expect(w.terminated).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await expect(p).resolves.toBe('par');
    expect(w.terminated).toBe(true);
    expect(svc.getLoadedModelId()).toBeNull();

    // The gate is released: the next call proceeds (and needs a reload).
    await expect(svc.generate({ prompt: 'Q', onToken: () => {} })).rejects.toThrow(/not loaded/);
  });

  it('gives a cancel during prefill (no token yet) the longer window', async () => {
    vi.useFakeTimers();
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const ctrl = new AbortController();
    const p = svc.generate({ prompt: 'P', signal: ctrl.signal, onToken: () => {} });
    await tick();
    const gen = w.last('generate');
    w.emit({ type: 'prefill', id: gen.id, done: 2048, total: 9000 });
    ctrl.abort();

    // A long prefill chunk outlasts the decode window without being wedged.
    await vi.advanceTimersByTimeAsync(svc.ABORT_WATCHDOG_MS * 2);
    expect(w.terminated).toBe(false);
    w.emit({ type: 'prefill', id: gen.id, done: 4096, total: 9000 });
    w.emit({ type: 'done', id: gen.id, text: '', stats: { ...STATS, reason: 'interrupted' } });
    await expect(p).resolves.toBe('');
    expect(w.terminated).toBe(false);
  });

  it('recycles a worker stuck in prefill after the prefill window', async () => {
    vi.useFakeTimers();
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const ctrl = new AbortController();
    const p = svc.generate({ prompt: 'P', signal: ctrl.signal, onToken: () => {} });
    await tick();
    ctrl.abort();
    await vi.advanceTimersByTimeAsync(svc.PREFILL_ABORT_WATCHDOG_MS);
    await expect(p).resolves.toBe('');
    expect(w.terminated).toBe(true);
  });

  it('a cancel acknowledged in time disarms the watchdog', async () => {
    vi.useFakeTimers();
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const ctrl = new AbortController();
    const p = svc.generate({ prompt: 'P', signal: ctrl.signal, onToken: () => {} });
    await tick();
    const gen = w.last('generate');
    ctrl.abort();
    w.emit({ type: 'done', id: gen.id, text: '', stats: { ...STATS, reason: 'interrupted' } });
    await expect(p).resolves.toBe('');
    await vi.advanceTimersByTimeAsync(svc.ABORT_WATCHDOG_MS * 2);
    expect(w.terminated).toBe(false);
    expect(svc.getLoadedModelId()).toBe('gemma-4-e2b');
  });

  it('rejects an interrupted generation the caller did not abort', async () => {
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const onStats = vi.fn();
    const p = svc.generate({ prompt: 'P', onToken: () => {}, onStats });
    const settled = p.catch((e: unknown) => e);
    await tick();
    const gen = w.last('generate');
    w.emit({ type: 'token', id: gen.id, text: 'trunc' });
    w.emit({ type: 'done', id: gen.id, text: 'trunc', stats: { ...STATS, reason: 'interrupted' } });
    const err = await settled;
    expect((err as Error).message).toBe('Generation interrupted by a model switch');
    expect(onStats).not.toHaveBeenCalled();
  });
});

describe('llmService fatal generate errors', () => {
  it('recycles the worker on a fatal generate-failed; the next ensureLoaded reloads in a new worker', async () => {
    const { svc, worker } = await freshService();
    const w1 = await loadModel(svc, worker);
    const p = svc.generate({ prompt: 'P', onToken: () => {} });
    const settled = p.catch((e: unknown) => e);
    await tick();
    w1.emit({
      type: 'error',
      id: w1.last('generate').id,
      code: 'generate-failed',
      message: 'OrtRun() SafeIntOnOverflow() Integer overflow',
      data: { fatal: true },
    });
    expect(((await settled) as Error).message).toMatch(/SafeIntOnOverflow/);
    expect(w1.terminated).toBe(true);
    expect(svc.getLoadedModelId()).toBeNull();

    const w2 = await loadModel(svc, worker);
    expect(w2).not.toBe(w1);
    expect(w2.all('load')).toHaveLength(1);
    expect(svc.getLoadedModelId()).toBe('gemma-4-e2b');
  });

  it('keeps the worker for non-fatal errors (context-too-long, plain generate-failed)', async () => {
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);

    const a = svc.generate({ prompt: 'P', onToken: () => {} }).catch((e: unknown) => e);
    await tick();
    w.emit({
      type: 'error',
      id: w.last('generate').id,
      code: 'context-too-long',
      message: 'too long',
      data: { promptTokens: 30_000, limit: 20_000 },
    });
    expect(svc.isInputTooLongError(await a)).toBe(true);

    const b = svc.generate({ prompt: 'P', onToken: () => {} }).catch((e: unknown) => e);
    await tick();
    w.emit({ type: 'error', id: w.last('generate').id, code: 'generate-failed', message: 'tokenizer' });
    expect(((await b) as Error).message).toBe('tokenizer');

    expect(w.terminated).toBe(false);
    expect(svc.getLoadedModelId()).toBe('gemma-4-e2b');
  });

  it('ignores prefill progress messages', async () => {
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker);
    const deltas: string[] = [];
    const p = svc.generate({ prompt: 'P', onToken: (d) => deltas.push(d) });
    await tick();
    const gen = w.last('generate');
    w.emit({ type: 'prefill', id: gen.id, done: 2048, total: 5000 });
    w.emit({ type: 'prefill', id: gen.id, done: 4096, total: 5000 });
    w.emit({ type: 'token', id: gen.id, text: 'ok' });
    w.emit({ type: 'done', id: gen.id, text: 'ok', stats: STATS });
    await expect(p).resolves.toBe('ok');
    expect(deltas).toEqual(['ok']);
  });
});

describe('llmService idle unload (side-task model while ZEOS Qwen 4B chats)', () => {
  it('unloadIfIdle terminates an idle worker, but keeps the model it is told to keep', async () => {
    const { svc, worker } = await freshService();
    const w = await loadModel(svc, worker, 'qwen3.5-4b');
    expect(svc.unloadIfIdle('qwen3.5-4b')).toBe(false);
    expect(w.terminated).toBe(false);
    expect(svc.unloadIfIdle('zeos-qwen3.5-4b')).toBe(true);
    expect(w.terminated).toBe(true);
    expect(svc.getLoadedModelId()).toBeNull();
  });

  it('unloadIfIdle leaves a busy worker alone', async () => {
    const { svc, worker } = await freshService();
    void svc.ensureLoaded('gemma-4-e2b').catch(() => undefined);
    await tick();
    expect(svc.unloadIfIdle(null)).toBe(false);
    expect(worker().terminated).toBe(false);
  });

  it('setIdleUnload(ms) unloads after the worker has been idle that long, re-armed by each use', async () => {
    vi.useFakeTimers();
    const { svc, worker } = await freshService();
    svc.setIdleUnload(60_000);
    const w = await loadModel(svc, worker, 'qwen3.5-4b');

    await vi.advanceTimersByTimeAsync(59_000);
    expect(w.terminated).toBe(false);
    // A generation re-arms the timer from its end.
    const gen = svc.generate({ prompt: 'p', onToken: () => {} });
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(w.terminated).toBe(false);
    const g = w.last('generate');
    w.emit({ type: 'done', id: g.id, text: '', stats: STATS });
    await gen;
    await vi.advanceTimersByTimeAsync(59_000);
    expect(w.terminated).toBe(false);
    await vi.advanceTimersByTimeAsync(1_001);
    expect(w.terminated).toBe(true);
    expect(svc.getLoadedModelId()).toBeNull();
  });

  it('setIdleUnload(null) keeps the model loaded', async () => {
    vi.useFakeTimers();
    const { svc, worker } = await freshService();
    svc.setIdleUnload(1_000);
    svc.setIdleUnload(null);
    const w = await loadModel(svc, worker);
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(w.terminated).toBe(false);
    expect(svc.getLoadedModelId()).toBe('gemma-4-e2b');
  });
});
