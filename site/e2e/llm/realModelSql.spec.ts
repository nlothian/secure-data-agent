import { dispatchLoadData, resolveLocalUrl } from '../helpers/loadData';
import { expect, test } from '@playwright/test';

import { existsSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// Heavyweight end-to-end LLM suite (runs only via `npm run test:llm_tests`,
// the dir-scoped `llm` Playwright project). Loads a real local Gemma 4 ONNX
// model into WebGPU via transformers.js and drives a real generation:
//
//   1. pick the model in the ModelSelector dropdown and await ensureLoaded()
//   2. load /tour-data/train.csv as the DuckDB table `train`
//   3. ask the model to write+run SQL and assert a result grid is rendered
//
// playwright.config.ts starts a dedicated dev server on :4322 with
// PUBLIC_LOCAL_MODELS=1, so the weights are served from the repo-root
// models/ folder (scripts/local-models-vite-plugin.mjs) and never fetched
// from the Hugging Face Hub.
//
// Gated on the model files being present AND WebGPU being usable. Missing
// files / absent WebGPU is a SKIP, not a failure — consistent with the
// project stance that e2e is environmentally red, not a regression signal.

type ModelId = 'gemma-4-e2b' | 'gemma-4-e4b' | 'qwen3.5-4b';

// Kept in lockstep with src/lib/localLlm/models.ts. That module reads
// import.meta.env, so it is not imported from Node here.
const MODELS: Record<ModelId, { repo: string; label: string; fetchArg: string }> = {
  'gemma-4-e2b': {
    repo: 'onnx-community/gemma-4-E2B-it-ONNX',
    label: 'Gemma 4 E2B',
    fetchArg: 'e2b',
  },
  'gemma-4-e4b': {
    repo: 'onnx-community/gemma-4-E4B-it-ONNX',
    label: 'Gemma 4 E4B',
    fetchArg: 'e4b',
  },
  'qwen3.5-4b': {
    repo: 'onnx-community/Qwen3.5-4B-ONNX',
    label: 'Qwen 3.5 4B',
    fetchArg: 'qwen4b',
  },
};

const MODEL_ID = (process.env.GDA_E2E_MODEL ?? 'gemma-4-e4b') as ModelId;
const MODEL = MODELS[MODEL_ID];
if (!MODEL) {
  throw new Error(
    `GDA_E2E_MODEL=${MODEL_ID} is not one of: ${Object.keys(MODELS).join(', ')}`,
  );
}

interface ManifestEntry {
  path: string;
  bytes: number;
}
type Manifest = Record<string, { required: ManifestEntry[]; optional: ManifestEntry[] }>;

const MANIFEST = JSON.parse(
  readFileSync(
    fileURLToPath(
      new URL('../../src/lib/localLlm/modelFiles.json', import.meta.url),
    ),
    'utf8',
  ),
) as Manifest;

// Spec lives in site/e2e/llm/, so the repo-root models/ dir is three levels up.
const MODEL_DIR = fileURLToPath(
  new URL(`../../../models/${MODEL.repo}/`, import.meta.url),
);

// Every required file must exist with its exact manifest byte size. A
// partial (in-progress or interrupted) download therefore skips up front
// rather than failing mid-load deep inside ONNX Runtime.
const MISSING_FILES = (MANIFEST[MODEL.repo]?.required ?? [
  { path: '<manifest entry missing>', bytes: -1 },
]).filter(({ path, bytes }) => {
  const full = MODEL_DIR + path;
  return !existsSync(full) || statSync(full).size !== bytes;
});
const MODEL_EXISTS = MISSING_FILES.length === 0;

if (!MODEL_EXISTS) {
  // Surfaced in the Playwright "list" reporter output before the skip.
  console.warn(
    `\n[realModelSql] SKIPPED: ${MODEL.repo} is missing or incomplete under\n` +
      `  ${MODEL_DIR}\n` +
      `  Missing / wrong-size: ${MISSING_FILES.map((f) => f.path).join(', ')}\n` +
      `  This gated heavyweight spec only runs when the local model's ONNX files\n` +
      `  are present. Fetch them with: cd site && npm run models:fetch -- ${MODEL.fetchArg}\n` +
      `  (select the model with GDA_E2E_MODEL=${Object.keys(MODELS).join('|')}), then\n` +
      `  run: npm run test:llm_tests\n`,
  );
}

const LLM_CONFIG_STORAGE_KEY = 'haw.llm.config.v1';

const PROMPT =
  'Use SQL and show me survived percentage grouped by age groups from ' +
  'train. You will need to create age groups';

test.describe('real local Gemma — writes & runs SQL, renders a result grid', () => {
  test.skip(
    !MODEL_EXISTS,
    `models/${MODEL.repo} missing or incomplete — gated heavyweight LLM spec skipped`,
  );

  test.beforeEach(async ({ page }) => {
    // Suppress the first-visit onboarding tour: its dialog overlays the
    // chat/model UI and intercepts clicks. addInitScript re-runs on reload.
    await page.addInitScript(() =>
      localStorage.setItem('tour.seen', '1'),
    );

    await page.goto('/');
    await expect(page.getByText('Choose model')).toBeVisible();

    // Belt-and-suspenders state reset: drop any persisted LLM config and
    // clear loaded tables / chat history via the same New-chat path the app
    // uses.
    await page.evaluate(async (key) => {
      localStorage.removeItem(key);
      const bridge = await import('/src/lib/tour/bridge.ts');
      bridge.getChatBridge().newChat();
    }, LLM_CONFIG_STORAGE_KEY);
    await page.reload();
    await expect(page.getByText('Choose model')).toBeVisible();
  });

  test('loads train.csv + the real model and renders a SQL result table', async ({
    page,
  }) => {
    // Model load into WebGPU + a multi-turn local-inference agent loop is
    // slow; give generous headroom over the 15-min project timeout.
    test.setTimeout(20 * 60_000);

    // The dev server must have been started with PUBLIC_LOCAL_MODELS=1 —
    // otherwise the worker would resolve weights from the Hugging Face Hub.
    const localMode = await page.evaluate(() =>
      import('/src/lib/localLlm/models.ts').then((m) => m.isLocalModelsMode()),
    );
    expect(
      localMode,
      'isLocalModelsMode() is false in the page: the llm webServer env wiring ' +
        'in playwright.config.ts (PUBLIC_LOCAL_MODELS=1 on :4322) is broken',
    ).toBe(true);

    const requestUrls: string[] = [];
    page.on('request', (req) => requestUrls.push(req.url()));
    // Surface the engine's per-generation stats (DEV-only console.debug in
    // llmService) and any worker warnings in the Playwright output, so a slow
    // or failing run shows prefill/reuse numbers instead of just a timeout.
    page.on('console', (msg) => {
      const text = msg.text();
      if (text.includes('[llmService] generate stats') || text.includes('[llm.worker]')) {
        console.log(`  [browser] ${text.split('\n')[0]}`);
      }
    });

    // WebGPU is mandatory: the model dropdown button is disabled without it
    // and inference cannot run. Absence is environmental → skip, not fail.
    const gpu = await page.evaluate(async () => {
      const m = await import('/src/lib/localLlm/webgpu.ts');
      return m.detectWebGpu();
    });
    test.skip(
      !gpu.supported,
      `WebGPU unavailable in this browser: ${gpu.reason ?? 'unknown'} — ` +
        `environmentally red, not a regression`,
    );

    // --- Pick the model through the real ModelSelector UI ---------------
    const dropdown = page.locator('[data-tour-id="chat.modelDropdown"]');
    await expect(dropdown).toBeEnabled({ timeout: 30_000 });
    await dropdown.click();
    await expect(
      page.locator('[data-tour-id="chat.modelPopover"]'),
    ).toBeVisible();
    await page
      .getByRole('menuitem', { name: new RegExp(MODEL.label) })
      .click();

    // In local-models mode isModelCached() is true, so there is normally no
    // download-size dialog. Accept it if one appears anyway.
    const confirm = page.locator('.chat-model-confirm');
    const confirmShown = await confirm
      .waitFor({ state: 'visible', timeout: 1_500 })
      .then(() => true)
      .catch(() => false);
    if (confirmShown) await page.locator('.chat-model-apply').click();

    // Await the actual transformers.js/WebGPU load deterministically:
    // ensureLoaded() is idempotent and resolves only once the model is fully
    // loaded (joining the load the selection already kicked off).
    await page.evaluate(async (id) => {
      const svc = await import('/src/lib/localLlm/llmService.ts');
      await svc.ensureLoaded(id);
    }, MODEL_ID);

    // Model committed + loaded → the composer is no longer "unconfigured".
    const composer = page.locator('[data-tour-id="chat.messageEntry"]');
    await expect(composer).toBeEnabled({ timeout: 60_000 });

    // --- Load the Titanic CSV as DuckDB table `train` ------------------
    const url = await resolveLocalUrl(page, '/tour-data/train.csv');
    const loaded = await dispatchLoadData(page, url, 'train');
    expect(loaded.error).toBeUndefined();
    expect(loaded).toMatchObject({
      name: 'train',
      source: 'url',
      rowCount: 891,
    });

    // --- Ask the model and let the agent loop run ----------------------
    await composer.fill(PROMPT);
    const play = page.locator('[data-tour-id="chat.playButton"]');
    await expect(play).toBeEnabled();
    await play.click();

    // onPlay sets the tool-debugger to "running" before sending, so RunSQL
    // auto-executes. This pump is a safety net against the rare interleave
    // where a tool reaches the gate before that mode flip commits; play()
    // is idempotent and only resolves a pending gate / re-asserts running.
    const pump = setInterval(() => {
      page
        .evaluate(async () => {
          const dbg = await import('/src/lib/toolDebugger.ts');
          const snap = dbg.getSnapshot();
          if (snap.mode !== 'running' || snap.pending) dbg.play();
        })
        .catch(() => {
          /* page navigating/closing — ignore */
        });
    }, 1500);

    try {
      // Pass condition: the SQL execution-panel grid renders with >=1 data
      // row. RunSQLTool.onRunning auto-switches to the SQL tab, so the grid
      // mounts on its own. We intentionally do NOT assert column names,
      // bucket edges, or percentages — the model is nondeterministic; a
      // non-empty grid proves it produced runnable SQL whose result was
      // published to the panel.
      const grid = page.locator('table.exec-grid');
      await expect(grid).toBeVisible({ timeout: 12 * 60_000 });
      await expect(grid.locator('tbody tr')).not.toHaveCount(0);
    } finally {
      clearInterval(pump);
    }

    // Local-models mode must never touch the Hub.
    const hubRequests = requestUrls.filter((u) => u.includes('huggingface.co'));
    expect(hubRequests, 'requests to huggingface.co in local-models mode').toEqual([]);
  });
});
