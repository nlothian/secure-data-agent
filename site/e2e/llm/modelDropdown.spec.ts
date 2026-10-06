import { expect, test } from '../fixtures';

// The model dropdown, which is disabled without WebGPU, so it runs in the
// `llm` project (headed Chrome with WebGPU) rather than with the headless
// smoke tests, where it always skipped. It never loads a model: it lists the
// models with their sizes and no custom-file picker, then picks one. Without
// local models an uncached pick asks for confirmation and Cancel leaves the
// selection alone; in local-models mode (how `test:llm_tests` serves the app)
// nothing is downloaded, so the pick commits directly.

const LLM_CONFIG_STORAGE_KEY = 'haw.llm.config.v1';
const LOCAL_GEMMA_ENDPOINT = 'local://gemma';

async function readActiveModel(page: import('@playwright/test').Page): Promise<string | null> {
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
  test('dropdown lists the four models with sizes and no custom picker', async ({ page }) => {
    // No model is preselected, so boot does not eager-load one (in
    // local-models mode every model counts as cached).
    await page.goto('/');
    await expect(page.locator('.chat-model-split')).toBeVisible();

    // The dropdown button is disabled without WebGPU — environmental, skip.
    const gpu = await page.evaluate(async () => {
      const m = await import('/src/lib/localLlm/webgpu.ts');
      return m.detectWebGpu();
    });
    test.skip(!gpu.supported, `WebGPU unavailable in this browser: ${gpu.reason ?? 'unknown'}`);
    const before = await readActiveModel(page);

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
    // Offered in every build (from the Hub outside local-models mode).
    const zeos = page.getByRole('menuitem', { name: /ZEOS Qwen 4B/ });
    await expect(zeos).toBeVisible();
    await expect(zeos).toContainText('2.8 GB');
    await expect(popover.getByRole('menuitem')).toHaveCount(4);

    // The old "Advanced" custom-file picker section is gone.
    await expect(page.locator('.chat-model-advanced-toggle')).toHaveCount(0);
    await expect(popover.locator('input[type="file"]')).toHaveCount(0);

    await e4b.click();

    // Uncached → a download-size confirmation appears. In local-models mode,
    // or if this browser profile has E4B cached, the selection commits directly.
    const confirm = page.locator('.chat-model-confirm');
    const confirmShown = await confirm
      .waitFor({ state: 'visible', timeout: 3_000 })
      .then(() => true)
      .catch(() => false);

    if (confirmShown) {
      await expect(confirm).toContainText('4.9 GB');
      await page.locator('.chat-model-cancel').click();
      await expect(confirm).toHaveCount(0);
      expect(await readActiveModel(page)).toBe(before);
    } else {
      await expect.poll(() => readActiveModel(page), { timeout: 5_000 }).toBe('gemma-4-e4b');
    }
  });
});
