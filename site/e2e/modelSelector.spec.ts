import { expect, test } from '@playwright/test';

// Fast ModelSelector coverage (chromium project — never loads a model):
//   - booting with a predefined model selected must not download anything
//     (no Hugging Face requests, no /models/ requests);
//   - the dropdown lists exactly the two predefined Gemma 4 models with
//     their download sizes and no custom-file picker;
//   - choosing an uncached model asks for confirmation and Cancel leaves
//     the persisted selection untouched.

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

async function readActiveModel(
  page: import('@playwright/test').Page,
): Promise<string | null> {
  return page.evaluate(
    ({ key, ep }) => {
      const raw = localStorage.getItem(key);
      if (!raw) return null;
      const cfg = JSON.parse(raw) as { models?: Record<string, string> };
      return cfg.models?.[ep] ?? null;
    },
    { key: LLM_CONFIG_STORAGE_KEY, ep: LOCAL_GEMMA_ENDPOINT },
  );
}

test.describe('ModelSelector — predefined Gemma 4 ONNX models', () => {
  test.beforeEach(async ({ page }) => {
    // Suppress the first-visit onboarding tour (its dialog intercepts
    // clicks) and boot with gemma-4-e2b already selected. Init scripts
    // re-run on every navigation, so seed only on the very first load of
    // this page — later reloads must observe what the app persisted.
    await page.addInitScript(
      ({ key, cfg }) => {
        localStorage.setItem('tour.seen', '1');
        if (!sessionStorage.getItem('e2e.modelSelector.seeded')) {
          localStorage.setItem(key, cfg);
          sessionStorage.setItem('e2e.modelSelector.seeded', '1');
        }
      },
      { key: LLM_CONFIG_STORAGE_KEY, cfg: configJson('gemma-4-e2b') },
    );
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
    // Give any idle-scheduled eager-load effect ample time to (not) fire.
    await page.waitForTimeout(4000);

    expect(modelRequests).toEqual([]);
  });

  test('dropdown lists the two models with sizes and no custom picker', async ({
    page,
  }) => {
    await page.goto('/');
    await expect(page.locator('.chat-model-split')).toBeVisible();

    // The dropdown button is disabled without WebGPU — environmental, skip.
    const gpu = await page.evaluate(async () => {
      const m = await import('/src/lib/localLlm/webgpu.ts');
      return m.detectWebGpu();
    });
    test.skip(
      !gpu.supported,
      `WebGPU unavailable in this browser: ${gpu.reason ?? 'unknown'}`,
    );

    const dropdown = page.locator('[data-tour-id="chat.modelDropdown"]');
    await expect(dropdown).toBeEnabled({ timeout: 30_000 });
    await dropdown.click();

    const popover = page.locator('[data-tour-id="chat.modelPopover"]');
    await expect(popover).toBeVisible();

    const e2b = page.getByRole('menuitem', { name: /Gemma 4 E2B/ });
    const e4b = page.getByRole('menuitem', { name: /Gemma 4 E4B/ });
    await expect(e2b).toBeVisible();
    await expect(e2b).toContainText('3.1 GB');
    await expect(e4b).toBeVisible();
    await expect(e4b).toContainText('4.9 GB');
    await expect(page.getByRole('menuitem', { name: /Qwen 3\.5 4B/ })).toBeVisible();

    // The old "Advanced" custom-file picker section is gone.
    await expect(page.locator('.chat-model-advanced-toggle')).toHaveCount(0);
    await expect(popover.locator('input[type="file"]')).toHaveCount(0);

    await e4b.click();

    // Uncached → a download-size confirmation appears. If this browser
    // profile happens to have E4B cached, the selection commits directly.
    const confirm = page.locator('.chat-model-confirm');
    const confirmShown = await confirm
      .waitFor({ state: 'visible', timeout: 3_000 })
      .then(() => true)
      .catch(() => false);

    if (confirmShown) {
      await expect(confirm).toContainText('4.9 GB');
      await page.locator('.chat-model-cancel').click();
      await expect(confirm).toHaveCount(0);
      expect(await readActiveModel(page)).toBe('gemma-4-e2b');
    } else {
      await expect
        .poll(() => readActiveModel(page), { timeout: 5_000 })
        .toBe('gemma-4-e4b');
    }
  });
});
