import { expect, test, type Page } from './fixtures';

// A first visit (no saved LLM config) selects the default local model by
// itself where one can run, and a send with a model that is not cached asks
// before downloading it. Headless Chromium has no WebGPU, so the first-visit
// pick is seen through the ZEOS dev stub (which `canRunZeos` counts); without
// it the page stays on "Choose model". The send-time dialog uses the real
// ZEOS model id with no stub: nothing is in OPFS, so it is not cached. Its
// engine then fails at once for want of WebGPU, before any download.

const LLM_CONFIG_STORAGE_KEY = 'haw.llm.config.v1';
const LOCAL_GEMMA_ENDPOINT = 'local://gemma';
const STUB_KEY = 'gda.zeos.stub';

async function readConfig(page: Page): Promise<{ activeEndpoint?: string | null; models?: Record<string, string> } | null> {
  return page.evaluate((key) => {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  }, LLM_CONFIG_STORAGE_KEY);
}

test.describe('first visit model', () => {
  test('selects ZEOS Qwen 4B where it can run', async ({ page }) => {
    await page.addInitScript((stubKey) => {
      localStorage.setItem(stubKey, JSON.stringify({ replies: ['Hi.'], attention: 'first' }));
    }, STUB_KEY);
    await page.goto('/');
    await expect(page.locator('.chat-model-label')).toHaveText('ZEOS Qwen 4B');
    const cfg = await readConfig(page);
    expect(cfg?.activeEndpoint).toBe(LOCAL_GEMMA_ENDPOINT);
    expect(cfg?.models?.[LOCAL_GEMMA_ENDPOINT]).toBe('zeos-qwen3.5-4b');
    await expect(page.getByLabel('Chat message')).toBeEnabled();
  });

  test('selects nothing without WebGPU', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.chat-model-split')).toBeVisible();
    const gpu = await page.evaluate(async () => {
      const m = await import('/src/lib/localLlm/webgpu.ts');
      return m.detectWebGpu();
    });
    test.skip(gpu.supported, 'this browser has WebGPU');
    // The pick runs right after the same (cached) WebGPU check.
    await page.evaluate(() => new Promise((r) => setTimeout(r, 200)));
    await expect(page.getByText('Choose model')).toBeVisible();
    expect(await readConfig(page)).toBeNull();
  });

  test('a send asks before downloading an uncached model', async ({ page }) => {
    await page.addInitScript(
      ({ cfgKey, cfg }) => {
        if (!sessionStorage.getItem('e2e.firstVisit.seeded')) {
          localStorage.setItem(cfgKey, cfg);
          sessionStorage.setItem('e2e.firstVisit.seeded', '1');
        }
      },
      {
        cfgKey: LLM_CONFIG_STORAGE_KEY,
        cfg: JSON.stringify({
          activeEndpoint: LOCAL_GEMMA_ENDPOINT,
          customEndpoints: [],
          apiKeys: {},
          models: { [LOCAL_GEMMA_ENDPOINT]: 'zeos-qwen3.5-4b' },
        }),
      },
    );
    await page.goto('/');
    await expect(page.locator('.chat-model-label')).toHaveText('ZEOS Qwen 4B');

    const composer = page.getByLabel('Chat message');
    const play = page.getByRole('button', { name: 'Play' });
    const confirm = page.locator('.chat-model-confirm');

    // Cancel: nothing is sent and the text goes back to the composer.
    await composer.fill('hello');
    await play.click();
    await expect(confirm).toBeVisible();
    await expect(confirm).toContainText('ZEOS Qwen 4B is about');
    await expect(confirm.getByRole('button', { name: 'Download' })).toBeVisible();
    await confirm.getByRole('button', { name: 'Cancel' }).click();
    await expect(confirm).toHaveCount(0);
    await expect(composer).toHaveValue('hello');
    await expect(page.locator('.chat-msg-user')).toHaveCount(0);

    // Download: the message is sent.
    await play.click();
    await expect(confirm).toBeVisible();
    await confirm.getByRole('button', { name: 'Download' }).click();
    await expect(confirm).toHaveCount(0);
    await expect(page.locator('.chat-msg-user')).toHaveCount(1);
    await expect(page.locator('.chat-msg-user').first()).toContainText('hello');

    // Agreed once: the next send does not ask again.
    await expect(composer).toBeEnabled({ timeout: 30_000 });
    await composer.fill('again');
    await expect(play).toBeEnabled({ timeout: 30_000 });
    await play.click();
    await expect(page.locator('.chat-msg-user')).toHaveCount(2);
    await expect(confirm).toHaveCount(0);
  });
});
