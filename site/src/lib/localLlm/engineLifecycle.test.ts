import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LOCAL_GEMMA_ENDPOINT, type LLMConfig } from '../../types/llm';

const llm = {
  setIdleUnload: vi.fn(),
  unloadIfIdle: vi.fn(() => true),
};
const zeos = { disposeZeos: vi.fn() };
vi.mock('./llmService', () => llm);
vi.mock('../zeos/streamZeos', () => zeos);

import {
  __resetEngineLifecycleForTests,
  activeLocalModel,
  beginModelStream,
  getModelSwitchSnapshot,
  modelSwitchBlockedReason,
  noteZeosModuleLoaded,
  releaseUnusedEngines,
  SIDE_TASK_IDLE_UNLOAD_MS,
  subscribeModelSwitch,
} from './engineLifecycle';
import { getLocalGemmaModel, sideTaskConfig } from './models';

const config = (endpoint: string | null, model?: string): LLMConfig =>
  ({
    activeEndpoint: endpoint,
    customEndpoints: [],
    apiKeys: {},
    models: model ? { [endpoint ?? '']: model } : {},
    thinkingEnabled: {},
  }) as LLMConfig;

// The ZEOS model is listed only in local-models or stub mode; build one from
// the plain Qwen entry for the lifecycle, which only reads family and sideTaskModelId.
const zeosModel = {
  ...getLocalGemmaModel('qwen3.5-4b')!,
  id: 'zeos-qwen3.5-4b' as const,
  label: 'ZEOS Qwen 4B',
  family: 'zeos-qwen' as const,
  sideTaskModelId: 'qwen3.5-4b' as const,
};

describe('engineLifecycle', () => {
  beforeEach(() => {
    __resetEngineLifecycleForTests();
    vi.clearAllMocks();
  });
  afterEach(() => __resetEngineLifecycleForTests());

  describe('switch guard', () => {
    it('blocks switching while any reply streams, and names it', () => {
      expect(modelSwitchBlockedReason()).toBeNull();
      const seen: (string | null)[] = [];
      const unsubscribe = subscribeModelSwitch(() => seen.push(getModelSwitchSnapshot()));
      const endChat = beginModelStream('the chat reply');
      expect(modelSwitchBlockedReason()).toMatch(/^Wait for the chat reply to finish, or stop it/);
      const endExplainer = beginModelStream('the Explainer reply');
      endChat();
      expect(modelSwitchBlockedReason()).toMatch(/the Explainer reply/);
      endExplainer();
      endExplainer(); // idempotent
      expect(modelSwitchBlockedReason()).toBeNull();
      expect(seen.at(-1)).toBeNull();
      expect(seen.length).toBe(4);
      unsubscribe();
    });

    it('the snapshot is stable between changes (useSyncExternalStore)', () => {
      const end = beginModelStream('the chat reply');
      expect(getModelSwitchSnapshot()).toBe(getModelSwitchSnapshot());
      end();
    });
  });

  describe('releaseUnusedEngines', () => {
    it('to ZEOS: idle-unloads the transformers.js worker, keeping only the side-task model', async () => {
      await releaseUnusedEngines(zeosModel);
      expect(llm.setIdleUnload).toHaveBeenCalledWith(SIDE_TASK_IDLE_UNLOAD_MS);
      expect(llm.unloadIfIdle).toHaveBeenCalledWith('qwen3.5-4b');
      expect(zeos.disposeZeos).not.toHaveBeenCalled();
    });

    it('away from ZEOS: disposes the ZEOS engine once its module has loaded', async () => {
      const gemma = getLocalGemmaModel('gemma-4-e2b')!;
      await releaseUnusedEngines(gemma);
      expect(zeos.disposeZeos).not.toHaveBeenCalled(); // never loaded: nothing to free
      expect(llm.setIdleUnload).toHaveBeenCalledWith(null);

      noteZeosModuleLoaded();
      await releaseUnusedEngines(gemma);
      expect(zeos.disposeZeos).toHaveBeenCalledWith('switched to Gemma 4 E2B');
      expect(llm.unloadIfIdle).not.toHaveBeenCalled();

      await releaseUnusedEngines(null);
      expect(zeos.disposeZeos).toHaveBeenLastCalledWith('switched to a cloud model');
    });
  });

  it('activeLocalModel is null for a cloud endpoint', () => {
    expect(activeLocalModel(config('https://api.anthropic.com/v1/messages', 'x'))).toBeNull();
    expect(activeLocalModel(config(LOCAL_GEMMA_ENDPOINT, 'qwen3.5-4b'))?.id).toBe('qwen3.5-4b');
  });
});

describe('sideTaskConfig (C4: the Explainer never runs on the ZEOS session)', () => {
  it('leaves models without a side-task model alone', () => {
    const c = config(LOCAL_GEMMA_ENDPOINT, 'gemma-4-e4b');
    expect(sideTaskConfig(c)).toBe(c);
    const cloud = config('https://api.anthropic.com/v1/messages', 'claude');
    expect(sideTaskConfig(cloud)).toBe(cloud);
  });

  it('routes ZEOS Qwen 4B to its side-task model, plain Qwen 3.5 4B', async () => {
    vi.stubEnv('PUBLIC_LOCAL_MODELS', '1');
    vi.resetModules();
    try {
      const m = await import('./models');
      const c = config(LOCAL_GEMMA_ENDPOINT, 'zeos-qwen3.5-4b');
      const side = m.sideTaskConfig(c);
      expect(side.models[LOCAL_GEMMA_ENDPOINT]).toBe('qwen3.5-4b');
      expect(m.getLocalGemmaModel(m.resolveActiveLocalModelIdOrDefault(side))?.family).toBe('qwen');
      expect(c.models[LOCAL_GEMMA_ENDPOINT]).toBe('zeos-qwen3.5-4b'); // not mutated
    } finally {
      vi.unstubAllEnvs();
    }
  });
});
