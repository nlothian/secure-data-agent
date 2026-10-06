import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LocalGemmaModel } from '../localLlm/models';

// startKernelChatEngine with the kernel and the model thread mocked: what it
// hands ZEOS `open_chat` and `ChatRun` methods, and how it refuses to start.

const kernelCalls: { module: string; fn: string; args: unknown[]; kwargs?: Record<string, unknown> }[] = [];
const methodCalls: { method: string; args: unknown[]; kwargs?: Record<string, unknown> }[] = [];

const fakeKernel = {
  call: vi.fn(async (module: string, fn: string, args: unknown[] = [], kwargs?: Record<string, unknown>) => {
    kernelCalls.push({ module, fn, args, kwargs });
    return { handle: fn };
  }),
  callMethod: vi.fn(async (_h: unknown, method: string, args: unknown[] = [], kwargs?: Record<string, unknown>) => {
    methodCalls.push({ method, args, kwargs });
    return [];
  }),
  release: vi.fn(async () => undefined),
};

const startZeos = vi.fn(async () => ({ kernel: fakeKernel, model: { name: 'default', backend: 'stub' } }));

vi.mock('./zeosHost', async (importOriginal) => {
  const real = await importOriginal<typeof import('./zeosHost')>();
  return { ...real, startZeos: (...a: unknown[]) => (startZeos as (...x: unknown[]) => unknown)(...a) };
});
vi.mock('./zeosModelWorker', () => ({
  zeosModelThreadFor: () => ({ stub: true, label: 'stub', modelWorker: () => ({}), init: {} }),
  createZeosLoadProgress: () => ({ onProgress: () => undefined, done: () => undefined }),
}));

const { startKernelChatEngine } = await import('./zeosChatEngine');
const { NotCrossOriginIsolatedError } = await import('./zeosHost');
const { ZEOS_TOOL_CLASSES, ZEOS_TRUSTED_RESULTS } = await import('./zeosToolClasses');

const MODEL = { id: 'zeos-qwen3.5-4b', label: 'ZEOS Qwen 4B' } as LocalGemmaModel;

describe('startKernelChatEngine', () => {
  beforeEach(() => {
    kernelCalls.length = 0;
    methodCalls.length = 0;
    vi.stubGlobal('crossOriginIsolated', true);
    vi.stubGlobal('SharedArrayBuffer', globalThis.SharedArrayBuffer ?? class {});
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    startZeos.mockClear();
  });

  it('refuses at once, with the reason, on a page that is not cross-origin isolated', async () => {
    vi.stubGlobal('crossOriginIsolated', false);
    await expect(startKernelChatEngine(MODEL)).rejects.toBeInstanceOf(NotCrossOriginIsolatedError);
    await expect(startKernelChatEngine(MODEL)).rejects.toThrow(/not cross-origin isolated/);
    expect(startZeos).not.toHaveBeenCalled();
  });

  it('hands open_chat the very tool-class and trusted-results tables', async () => {
    const engine = await startKernelChatEngine(MODEL);
    await engine.open({
      systemPrompt: 'sys',
      gateMode: 'strict',
      toolClasses: ZEOS_TOOL_CLASSES,
      trustedResults: ZEOS_TRUSTED_RESULTS,
      paramTypes: {},
      thinking: false,
    });
    const open = kernelCalls.find((c) => c.fn === 'open_chat')!;
    expect(open.module).toBe('zeos_chat.chat');
    expect(open.kwargs!.tool_classes).toBe(ZEOS_TOOL_CLASSES);
    expect(open.kwargs!.trusted_results).toBe(ZEOS_TRUSTED_RESULTS);
    expect(open.kwargs!.gate_mode).toBe('strict');
  });

  it('releases the Sampling handle when open_chat fails, and when it succeeds', async () => {
    const engine = await startKernelChatEngine(MODEL);
    const opts = {
      systemPrompt: 'sys',
      gateMode: 'strict' as const,
      toolClasses: ZEOS_TOOL_CLASSES,
      paramTypes: {},
      thinking: false,
      sampling: { temperature: 0.7, topK: 20 },
    };
    fakeKernel.call.mockImplementationOnce(async (module: string, fn: string) => {
      kernelCalls.push({ module, fn, args: [] });
      return { handle: 'sampling-1' };
    });
    fakeKernel.call.mockImplementationOnce(async () => {
      throw new Error('ValueError: refusing to run a tree that does not lint');
    });
    fakeKernel.release.mockClear();
    await expect(engine.open(opts)).rejects.toThrow(/does not lint/);
    expect(fakeKernel.release).toHaveBeenCalledWith({ handle: 'sampling-1' });

    fakeKernel.release.mockClear();
    await engine.open(opts);
    expect(fakeKernel.release).toHaveBeenCalledWith({ handle: 'Sampling' });
    expect(fakeKernel.release).toHaveBeenCalledTimes(1);
  });

  it('imports history with a real bool for every tool turn, and the starting integrity', async () => {
    const engine = await startKernelChatEngine(MODEL);
    const run = await engine.open({
      systemPrompt: 'sys',
      gateMode: 'strict',
      toolClasses: ZEOS_TOOL_CLASSES,
      paramTypes: {},
      thinking: false,
    });
    await run.importHistory(
      [
        { role: 'user', text: 'u' },
        { role: 'assistant', text: 'a', integrity: 2 },
        { role: 'tool', text: 'card', toolName: 'CallSkill', toolArgs: { skill: 'sql' }, trusted: true },
        { role: 'tool', text: 'rows', toolName: 'RunSQL' },
        // Trusted without its call: ZEOS would refuse it, so it replays on ring 3.
        { role: 'tool', text: 'orphan', toolName: 'CallSkill', trusted: true },
        { role: 'assistant', text: 'b' },
      ],
      3,
    );
    const imp = methodCalls.find((c) => c.method === 'import_history')!;
    expect(imp.args).toEqual([
      [
        { role: 'user', text: 'u' },
        { role: 'assistant', text: 'a', integrity: 2 },
        { role: 'tool', text: 'card', trusted: true, name: 'CallSkill', arguments: { skill: 'sql' } },
        { role: 'tool', text: 'rows', trusted: false },
        { role: 'tool', text: 'orphan', trusted: false },
        { role: 'assistant', text: 'b', integrity: 3 },
      ],
    ]);
    expect(imp.kwargs).toEqual({ start_integrity: 3 });
  });
});
