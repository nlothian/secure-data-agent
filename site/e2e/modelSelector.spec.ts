import { expect, test } from './fixtures';

// Fast ModelSelector coverage (chromium project — never loads a model):
// booting with a predefined model selected must not download anything (no
// Hugging Face requests, no /models/ requests). The dropdown's own test needs
// WebGPU, which headless Chromium lacks, so it lives in the `llm` project
// (e2e/llm/modelDropdown.spec.ts).

// Keep in lockstep with src/types/llm.ts.
const LLM_CONFIG_STORAGE_KEY = 'haw.llm.config.v1';
const LOCAL_GEMMA_ENDPOINT = 'local://gemma';

/** The local endpoint active, with `modelId` saved, or no model saved for null. */
function configJson(modelId: string | null): string {
  return JSON.stringify({
    activeEndpoint: LOCAL_GEMMA_ENDPOINT,
    customEndpoints: [],
    apiKeys: {},
    models: modelId === null ? {} : { [LOCAL_GEMMA_ENDPOINT]: modelId },
    thinkingEnabled: {},
  });
}

async function seedModel(page: import('@playwright/test').Page, modelId: string | null): Promise<void> {
  await page.addInitScript(
    ({ key, cfg }) => {
      if (!sessionStorage.getItem('e2e.modelSelector.seeded')) {
        localStorage.setItem(key, cfg);
        sessionStorage.setItem('e2e.modelSelector.seeded', '1');
      }
    },
    { key: LLM_CONFIG_STORAGE_KEY, cfg: configJson(modelId) },
  );
}

test.describe('ModelSelector — predefined Gemma 4 ONNX models', () => {
  test.beforeEach(async ({ page }) => {
    // Boot with gemma-4-e2b already selected. Init scripts re-run on every
    // navigation, so seed only on the very first load of this page — later
    // reloads must observe what the app persisted.
    await seedModel(page, 'gemma-4-e2b');
  });

  test('boot never downloads a model', async ({ page }) => {
    const modelRequests: string[] = [];
    page.on('request', (req) => {
      const url = req.url();
      if (url.includes('huggingface.co') || /\/models\//.test(url)) {
        modelRequests.push(url);
      }
    });

    await page.goto('/');
    await expect(page.locator('.chat-model-split')).toBeVisible();
    // ChatSidebar's boot-time eager load records its decision; anything but
    // 'load' means it never schedules a model load. (Without WebGPU, as in
    // headless Chromium, it is 'no-webgpu'; with it, 'not-cached'.)
    const eager = await (
      await page.waitForFunction(
        () => (window as unknown as { __gdaEagerLoad?: { id: string; decision: string } }).__gdaEagerLoad,
      )
    ).jsonValue();
    expect(eager?.id).toBe('gemma-4-e2b');
    expect(eager?.decision).not.toBe('load');

    expect(modelRequests).toEqual([]);
  });
});

test.describe('ModelSelector — ZEOS Qwen 4B outside local-models mode', () => {
  test('is kept as the selected model, and boot downloads nothing', async ({ page }) => {
    // The chromium project runs without PUBLIC_LOCAL_MODELS and without the
    // ZEOS stub, as a production build does: the model is listed there too,
    // so a stored selection of it is not reset to the default.
    await seedModel(page, 'zeos-qwen3.5-4b');
    const modelRequests: string[] = [];
    page.on('request', (req) => {
      const url = req.url();
      if (url.includes('huggingface.co') || /\/models\//.test(url)) {
        modelRequests.push(url);
      }
    });

    await page.goto('/');
    await expect(page.locator('.chat-model-label')).toHaveText('ZEOS Qwen 4B');
    const eager = await (
      await page.waitForFunction(
        () => (window as unknown as { __gdaEagerLoad?: { id: string; decision: string } }).__gdaEagerLoad,
      )
    ).jsonValue();
    expect(eager?.id).toBe('zeos-qwen3.5-4b');
    expect(eager?.decision).not.toBe('load');
    expect(modelRequests).toEqual([]);
  });
});

type Gpu = 'none' | 'no-f16' | 'f16';

/**
 * Stand in for what `defaultLocalModelId` reads: `navigator.gpu` (absent, an
 * adapter without shader-f16, or one with it, with buffer limits big enough
 * that the GPU warning banner stays away) and, when `isolated` is false,
 * `crossOriginIsolated` (as in Safari, which lacks COEP credentialless). The
 * adapter is never asked for a device: nothing here loads a model.
 */
async function fakeCapabilities(page: import('@playwright/test').Page, gpu: Gpu, isolated = true) {
  await page.addInitScript(
    ({ gpu, isolated }) => {
      const big = 4 * 1024 ** 3;
      const adapter = {
        features: new Set(gpu === 'f16' ? ['shader-f16'] : []),
        limits: { maxBufferSize: big, maxStorageBufferBindingSize: big },
        requestDevice: () => Promise.reject(new Error('fake adapter')),
      };
      const value =
        gpu === 'none'
          ? undefined
          : { requestAdapter: () => Promise.resolve(adapter), getPreferredCanvasFormat: () => 'bgra8unorm' };
      Object.defineProperty(Navigator.prototype, 'gpu', { configurable: true, get: () => value });
      if (!isolated) Object.defineProperty(window, 'crossOriginIsolated', { configurable: true, get: () => false });
    },
    { gpu, isolated },
  );
}

async function storedLocalModel(page: import('@playwright/test').Page): Promise<string | null> {
  return page.evaluate(
    ({ key, ep }) => {
      const raw = localStorage.getItem(key);
      return raw ? ((JSON.parse(raw) as { models?: Record<string, string> }).models?.[ep] ?? null) : null;
    },
    { key: LLM_CONFIG_STORAGE_KEY, ep: LOCAL_GEMMA_ENDPOINT },
  );
}

async function eagerDecision(page: import('@playwright/test').Page) {
  return (
    await page.waitForFunction(
      () => (window as unknown as { __gdaEagerLoad?: { id: string; decision: string } }).__gdaEagerLoad,
    )
  ).jsonValue();
}

test.describe('ModelSelector — the default local model', () => {
  test('is ZEOS Qwen 4B where it can run, and boot downloads nothing', async ({ page }) => {
    await fakeCapabilities(page, 'f16');
    await seedModel(page, null);
    const modelRequests: string[] = [];
    page.on('request', (req) => {
      const url = req.url();
      if (url.includes('huggingface.co') || /\/models\//.test(url)) modelRequests.push(url);
    });

    await page.goto('/');
    await expect(page.locator('.chat-model-label')).toHaveText('ZEOS Qwen 4B');
    // The default is saved, so the label and every inference path agree.
    await expect.poll(() => storedLocalModel(page)).toBe('zeos-qwen3.5-4b');
    // As for any uncached model: boot never starts the ~2.5 GB download.
    expect(await eagerDecision(page)).toEqual({ id: 'zeos-qwen3.5-4b', decision: 'not-cached' });
    expect(modelRequests).toEqual([]);
  });

  for (const [name, gpu, isolated] of [
    ['without WebGPU', 'none', true],
    ['without shader-f16', 'no-f16', true],
    ['on a page that is not cross-origin isolated', 'f16', false],
  ] as const) {
    test(`falls back to Gemma 4 E2B ${name}`, async ({ page }) => {
      await fakeCapabilities(page, gpu, isolated);
      await seedModel(page, null);
      await page.goto('/');
      await expect(page.locator('.chat-model-label')).toHaveText('Gemma 4 E2B');
      await expect.poll(() => storedLocalModel(page)).toBe('gemma-4-e2b');
      expect((await eagerDecision(page))?.id).toBe('gemma-4-e2b');
    });
  }

  test('never replaces a saved choice', async ({ page }) => {
    await fakeCapabilities(page, 'f16');
    await seedModel(page, 'gemma-4-e4b');
    await page.goto('/');
    await expect(page.locator('.chat-model-label')).toHaveText('Gemma 4 E4B');
    expect(await eagerDecision(page)).toEqual({ id: 'gemma-4-e4b', decision: 'not-cached' });
    expect(await storedLocalModel(page)).toBe('gemma-4-e4b');
  });
});
