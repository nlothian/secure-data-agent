import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

// ZEOS Qwen 4B's agent loop and trust UI, end to end, on the scripted chat
// stub model thread (no weights): the real Pyodide kernel, the real ZEOS chat
// machine and capability checks, the real tools. Stub mode is turned on by
// the `gda.zeos.stub` localStorage script, which also makes the model
// selectable outside local-models mode. Needs `npm run zeos:sync` (public/zeos/
// is generated) and network for jsDelivr (Pyodide).
//
// The script: ListInputs (a read), then WriteLines (an effect), then a reply.
// With attention "first" the stub never attends the tool result, so nothing
// demotes; with "recent" it attends the latest positions, which right after
// the result arrives are mostly the result, so reading it demotes the job.

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

const REPLIES = [
  `Let me check what data is loaded.\n\n${call('ListInputs')}`,
  `Nothing is loaded yet. I will save a note.\n\n${call('WriteLines', {
    path: '/scratchpad/zeos-note.txt',
    content: 'No inputs loaded yet.',
  })}`,
  'Done: I handled the note.',
];

async function boot(
  page: Page,
  { attention, attentionOnly = false }: { attention: 'first' | 'recent'; attentionOnly?: boolean },
) {
  await page.setViewportSize({ width: 1400, height: 900 });
  await page.addInitScript(
    ({ cfgKey, cfg, stubKey, stub }) => {
      localStorage.setItem('tour.seen', '1');
      if (!sessionStorage.getItem('e2e.zeosChat.seeded')) {
        localStorage.setItem(cfgKey, cfg);
        localStorage.setItem(stubKey, stub);
        localStorage.removeItem('haw.chat.history.v1');
        sessionStorage.setItem('e2e.zeosChat.seeded', '1');
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
        ...(attentionOnly ? { zeosAttentionOnly: true } : {}),
      }),
      stubKey: STUB_KEY,
      stub: JSON.stringify({ replies: REPLIES, attention }),
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

test.describe('ZEOS Qwen 4B chat (scripted stub model)', () => {
  test.skip(!fs.existsSync(manifest), 'public/zeos/manifest.json missing; run `npm run zeos:sync`');
  test.setTimeout(180_000);

  test('strict: a read runs, its result is ring 3, and the effect after it waits for approval', async ({
    page,
  }) => {
    await boot(page, { attention: 'first' });
    await expect(page.getByLabel('Attention-only approval')).not.toBeChecked();
    await send(page, 'What data is loaded? Save a note.');

    await expect(card(page)).toBeVisible({ timeout: 120_000 });
    await expect(card(page)).toContainText('Approve WriteLines?');
    await expect(card(page)).toContainText('strict: read tool output this turn');
    await expect(card(page)).toContainText('/scratchpad/zeos-note.txt');
    await expect(trust(page)).toHaveText('strict: read tool output this turn');
    const listInputs = lastAssistant(page).locator('.chat-tool-call', { hasText: 'ListInputs' });
    await expect(listInputs.locator('.chat-ring-badge')).toHaveText('ring 3');
    await page.screenshot({ path: test.info().outputPath('zeos-approval-card.png') });

    await card(page).getByRole('button', { name: 'Approve' }).click();
    await expect(card(page)).toHaveCount(0);
    await expect(lastAssistant(page)).toContainText('Done: I handled the note.', { timeout: 60_000 });
    const writeLines = lastAssistant(page).locator('.chat-tool-call', { hasText: 'WriteLines' });
    await expect(writeLines.locator('.chat-ring-badge')).toHaveText('ring 3');
    await writeLines.locator('.chat-tool-summary').click();
    await expect(writeLines).not.toContainText('declined');

    // Reload: the history and its ring badges persist, and the next message
    // replays it into a fresh run, which still gates the effect.
    await page.reload();
    await expect(
      lastAssistant(page).locator('.chat-tool-call', { hasText: 'ListInputs' }).locator('.chat-ring-badge'),
    ).toHaveText('ring 3');
    await send(page, 'Once more, please.');
    await expect(card(page)).toBeVisible({ timeout: 120_000 });
    await card(page).getByRole('button', { name: 'Deny' }).click();
    await expect(lastAssistant(page)).toContainText('Done: I handled the note.', { timeout: 60_000 });
  });

  test('strict, demoted: the card names the result that demoted the job; Deny delivers a refusal', async ({
    page,
  }) => {
    await boot(page, { attention: 'recent' });
    await send(page, 'What data is loaded? Save a note.');
    await expect(card(page)).toBeVisible({ timeout: 120_000 });
    await expect(card(page)).toContainText('strict: demoted by ListInputs result #1');
    await expect(trust(page)).toHaveText('strict: demoted by ListInputs result #1');

    await card(page).getByRole('button', { name: 'Deny' }).click();
    await expect(lastAssistant(page)).toContainText('Done: I handled the note.', { timeout: 60_000 });
    const writeLines = lastAssistant(page).locator('.chat-tool-call', { hasText: 'WriteLines' });
    await writeLines.locator('.chat-tool-summary').click();
    await expect(writeLines).toContainText('The user declined this tool call');
    await expect(trust(page)).toHaveText('strict: demoted by ListInputs result #1');
  });

  test('attention-only (toggled in the UI): the same flow runs the effect without a card', async ({
    page,
  }) => {
    await boot(page, { attention: 'first' });
    const toggle = page.getByLabel('Attention-only approval');
    await toggle.check();
    await expect(toggle).toBeChecked();
    expect(
      await page.evaluate((k) => JSON.parse(localStorage.getItem(k)!).zeosAttentionOnly, LLM_CONFIG_STORAGE_KEY),
    ).toBe(true);

    await send(page, 'What data is loaded? Save a note.');
    await expect(lastAssistant(page)).toContainText('Done: I handled the note.', { timeout: 120_000 });
    await expect(card(page)).toHaveCount(0);
    await expect(trust(page)).toHaveText('attention: trusted');
    const writeLines = lastAssistant(page).locator('.chat-tool-call', { hasText: 'WriteLines' });
    await expect(writeLines.locator('.chat-ring-badge')).toHaveText('ring 3');
  });

  test('attention-only, demoted: the effect still waits for approval', async ({ page }) => {
    await boot(page, { attention: 'recent', attentionOnly: true });
    await send(page, 'What data is loaded? Save a note.');
    await expect(card(page)).toBeVisible({ timeout: 120_000 });
    await expect(card(page)).toContainText('attention: demoted by ListInputs result #1');
    await expect(trust(page)).toHaveText('attention: demoted by ListInputs result #1');
    await card(page).getByRole('button', { name: 'Approve' }).click();
    await expect(lastAssistant(page)).toContainText('Done: I handled the note.', { timeout: 60_000 });
  });
});
