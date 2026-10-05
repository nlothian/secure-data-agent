import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

// ZEOS Qwen 4B's engine lifecycle on the scripted chat stub model thread (no
// weights; see zeosChat.spec.ts): a kernel worker crash while idle, a model
// thread crash while an approval card waits, model switching blocked while a
// reply streams, and disposeZeos. Needs `npm run zeos:sync` and network for
// jsDelivr (Pyodide).

const here = path.dirname(fileURLToPath(import.meta.url));
const manifest = path.resolve(here, '..', 'public', 'zeos', 'manifest.json');

const LLM_CONFIG_STORAGE_KEY = 'haw.llm.config.v1';
const LOCAL_GEMMA_ENDPOINT = 'local://gemma';
const STUB_KEY = 'gda.zeos.stub';

const call = (name: string, args: Record<string, string> = {}) =>
  `<tool_call>\n<function=${name}>\n` +
  Object.entries(args)
    .map(([k, v]) => `<parameter=${k}>\n${v}\n</parameter>\n`)
    .join('') +
  `</function>\n</tool_call>`;

async function boot(page: Page, replies: string[]) {
  await page.setViewportSize({ width: 1400, height: 900 });
  await page.addInitScript(
    ({ cfgKey, cfg, stubKey, stub }) => {
      localStorage.setItem('tour.seen', '1');
      if (!sessionStorage.getItem('e2e.zeosLifecycle.seeded')) {
        localStorage.setItem(cfgKey, cfg);
        localStorage.setItem(stubKey, stub);
        localStorage.removeItem('haw.chat.history.v1');
        sessionStorage.setItem('e2e.zeosLifecycle.seeded', '1');
      }
    },
    {
      cfgKey: LLM_CONFIG_STORAGE_KEY,
      cfg: JSON.stringify({
        activeEndpoint: LOCAL_GEMMA_ENDPOINT,
        customEndpoints: [],
        apiKeys: {},
        models: { [LOCAL_GEMMA_ENDPOINT]: 'zeos-qwen3.5-4b' },
        thinkingEnabled: {},
      }),
      stubKey: STUB_KEY,
      stub: JSON.stringify({ replies, attention: 'first' }),
    },
  );
  await page.goto('/');
  await expect(page.locator('.chat-model-label')).toHaveText('ZEOS Qwen 4B');
}

async function send(page: Page, text: string) {
  await page.getByLabel('Chat message').fill(text);
  await page.getByRole('button', { name: 'Play' }).click();
}

const card = (page: Page) => page.locator('.chat-zeos-approval');
const trust = (page: Page) => page.locator('.chat-zeos-trust');
const lastAssistant = (page: Page) => page.locator('.chat-msg-assistant').last();
const modelButton = (page: Page) => page.locator('[data-tour-id="chat.modelDropdown"]');

/** Wait for the engine to be up and expose the dev handle (`window.__zeosKernel`). */
async function waitForKernel(page: Page) {
  await page.waitForFunction(() => (window as unknown as { __zeosKernel?: unknown }).__zeosKernel != null, null, {
    timeout: 120_000,
  });
}

const APPROVAL_REPLIES = [
  `Checking.\n\n${call('ListInputs')}`,
  `Saving a note.\n\n${call('WriteLines', { path: '/scratchpad/lifecycle.txt', content: 'note' })}`,
  'Done with the note.',
];

test.describe('ZEOS Qwen 4B engine lifecycle (scripted stub model)', () => {
  test.skip(!fs.existsSync(manifest), 'public/zeos/manifest.json missing; run `npm run zeos:sync`');
  test.setTimeout(240_000);

  test('a kernel worker crash while idle is reported, and the next message restarts the engine', async ({
    page,
  }) => {
    await boot(page, ['Hello from the stub.']);
    await send(page, 'hi');
    await expect(lastAssistant(page)).toContainText('Hello from the stub.', { timeout: 120_000 });
    await expect(trust(page)).toHaveText('strict: trusted');
    await waitForKernel(page);

    // An uncaught error inside the kernel worker: the page's onerror fires.
    await page.evaluate(() => {
      const k = (window as unknown as { __zeosKernel: { exec(src: string): Promise<unknown> } }).__zeosKernel;
      void k
        .exec(`import js\njs.eval("setTimeout(() => { throw new Error('e2e kernel crash'); }, 0)")`)
        .catch(() => undefined);
    });
    await expect(trust(page)).toHaveText('ZEOS error', { timeout: 30_000 });
    await expect(trust(page)).toHaveAttribute('title', /ZEOS stopped: .*e2e kernel crash.*next message restarts it/);

    // A fresh engine replays the history; the new stub thread plays its script again.
    await send(page, 'hi again');
    await expect(lastAssistant(page)).toContainText('Hello from the stub.', { timeout: 120_000 });
    await expect(trust(page)).toHaveText('strict: trusted');
    await expect(page.locator('.chat-msg-assistant')).toHaveCount(2);
  });

  test('a model thread crash while an approval waits ends the turn with the error and drops the card', async ({
    page,
  }) => {
    await boot(page, APPROVAL_REPLIES);
    await send(page, 'Save a note.');
    await expect(card(page)).toBeVisible({ timeout: 120_000 });
    await waitForKernel(page);

    // While the turn streams (the card is part of it) the model cannot be switched.
    await expect(modelButton(page)).toBeDisabled();
    await expect(modelButton(page)).toHaveAttribute('data-switch-blocked', 'true');
    const switched = await page.evaluate(async (key) => {
      const bridge = await import('/src/lib/tour/bridge.ts');
      bridge.getChatBridge().requestModel('gemma-4-e2b');
      await new Promise((r) => setTimeout(r, 500));
      return JSON.parse(localStorage.getItem(key)!).models['local://gemma'];
    }, LLM_CONFIG_STORAGE_KEY);
    expect(switched).toBe('zeos-qwen3.5-4b');
    await expect(card(page)).toBeVisible();

    await page.evaluate(() => {
      const k = (
        window as unknown as { __zeosKernel: { models: Map<string, { thread: Worker }> } }
      ).__zeosKernel;
      k.models.get('default')!.thread.postMessage({ crash: 'e2e model crash' });
    });
    await expect(card(page)).toHaveCount(0, { timeout: 30_000 });
    // The turn's bubble becomes the error (with Retry).
    const failed = page.locator('.chat-msg-error').last();
    await expect(failed).toContainText('ZEOS model thread default crashed', { timeout: 30_000 });
    await expect(failed).toContainText('e2e model crash');
    await expect(trust(page)).toHaveText('ZEOS error');
    // The turn is over, so switching is allowed again.
    await expect(modelButton(page)).not.toHaveAttribute('data-switch-blocked', 'true');

    // The next message runs on a fresh engine; the effect still waits for approval.
    await send(page, 'Try again.');
    await expect(card(page)).toBeVisible({ timeout: 120_000 });
    await card(page).getByRole('button', { name: 'Deny' }).click();
    await expect(lastAssistant(page)).toContainText('Done with the note.', { timeout: 60_000 });
  });

  test('Stop during an approval returns at once; disposeZeos terminates the workers', async ({ page }) => {
    await boot(page, APPROVAL_REPLIES);
    await send(page, 'Save a note.');
    await expect(card(page)).toBeVisible({ timeout: 120_000 });
    await page.getByRole('button', { name: 'Stop' }).click();
    await expect(card(page)).toHaveCount(0, { timeout: 5_000 });
    await expect(modelButton(page)).not.toHaveAttribute('data-switch-blocked', 'true', { timeout: 5_000 });

    // A switch away from ZEOS (what the model picker runs) unloads it.
    const after = await page.evaluate(async () => {
      const w = window as unknown as { __zeosKernel: { disposedReason: Error | null } };
      const kernel = w.__zeosKernel;
      const lifecycle = await import('/src/lib/localLlm/engineLifecycle.ts');
      await lifecycle.releaseUnusedEngines(null);
      const z = await import('/src/lib/zeos/streamZeos.ts');
      const store = await import('/src/lib/zeos/zeosSessionStore.ts');
      return {
        reason: kernel.disposedReason?.message ?? null,
        loaded: z.isZeosLoaded(),
        status: store.getSnapshot().status,
      };
    });
    expect(after).toEqual({
      reason: 'ZEOS Qwen 4B was unloaded (switched to a cloud model)',
      loaded: false,
      status: 'idle',
    });

    // Still on ZEOS in the config, so the next message starts it again.
    await send(page, 'Again.');
    await expect(card(page)).toBeVisible({ timeout: 120_000 });
    await card(page).getByRole('button', { name: 'Deny' }).click();
    await expect(lastAssistant(page)).toContainText('Done with the note.', { timeout: 60_000 });
  });
});
