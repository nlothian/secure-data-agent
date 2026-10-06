import { expect, test } from './fixtures';

// Fast ModelSelector coverage (chromium project — never loads a model):
// booting with a predefined model selected must not download anything (no
// Hugging Face requests, no /models/ requests). The dropdown's own test needs
// WebGPU, which headless Chromium lacks, so it lives in the `llm` project
// (e2e/llm/modelDropdown.spec.ts).

// Keep in lockstep with src/types/llm.ts.
const LLM_CONFIG_STORAGE_KEY = 'haw.llm.config.v1';
const LOCAL_GEMMA_ENDPOINT = 'local://gemma';

function configJson(modelId: string): string {
  return JSON.stringify({
    activeEndpoint: LOCAL_GEMMA_ENDPOINT,
    customEndpoints: [],
    apiKeys: {},
    models: { [LOCAL_GEMMA_ENDPOINT]: modelId },
    thinkingEnabled: {},
  });
}

async function seedModel(page: import('@playwright/test').Page, modelId: string): Promise<void> {
  await page.addInitScript(
    ({ key, cfg }) => {
      localStorage.setItem('tour.seen', '1');
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
    // Suppress the first-visit onboarding tour (its dialog intercepts
    // clicks) and boot with gemma-4-e2b already selected. Init scripts
    // re-run on every navigation, so seed only on the very first load of
    // this page — later reloads must observe what the app persisted.
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
