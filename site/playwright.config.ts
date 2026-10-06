import { defineConfig, devices } from '@playwright/test';

// Playwright runs the local dev server, exercises the app in chromium, and
// targets /e2e for spec files. Vitest stays unit-only (npm test); these are
// browser-level smoke tests (npm run test:e2e).
//
// `npm run test:llm_tests` sets GDA_E2E_LLM=1 and runs only the `llm`
// project. That suite loads the real local Gemma ONNX files served from the
// repo-root models/ folder, which needs the dev server started with
// PUBLIC_LOCAL_MODELS=1 (see scripts/local-models-vite-plugin.mjs). It gets
// its own port and never reuses an existing server, so a stale dev server
// started without the flag can never silently send it to the Hugging Face Hub.
const LLM = process.env.GDA_E2E_LLM === '1';
// GDA_E2E_PORT moves the server, e.g. to run two worktrees' suites side by side.
const PORT = Number(process.env.GDA_E2E_PORT) || (LLM ? 4322 : 4321);

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'retain-on-failure',
  },
  projects: [
    {
      // Fast browser-level smoke tests. Excludes e2e/llm/ — those load a
      // multi-GB model and need WebGPU, so they run only in the `llm`
      // project via `npm run test:llm_tests`.
      name: 'chromium',
      testIgnore: /e2e[\\/]llm[\\/]/,
      use: { ...devices['Desktop Chrome'] },
    },
    {
      // Heavyweight LLM suite: real local Gemma ONNX files (served from the
      // repo-root models/ folder) + transformers.js WebGPU inference. Headed
      // real Chrome is by far the most reliable WebGPU path on macOS; the
      // suite self-skips when the model files are absent or WebGPU is
      // unavailable (environmentally red, not a regression).
      name: 'llm',
      testMatch: /e2e[\\/]llm[\\/].*\.spec\.ts$/,
      timeout: 15 * 60_000,
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chrome',
        headless: false,
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-angle=metal',
          ],
        },
      },
    },
  ],
  webServer: LLM
    ? {
        command: `npm run dev -- --port ${PORT}`,
        url: `http://localhost:${PORT}`,
        env: { ...process.env, PUBLIC_LOCAL_MODELS: '1' } as Record<
          string,
          string
        >,
        // Never reuse: a server started without PUBLIC_LOCAL_MODELS=1 would
        // make the LLM suite download from the Hub instead of models/.
        reuseExistingServer: false,
        timeout: 60_000,
      }
    : {
        command: `npm run dev -- --port ${PORT}`,
        url: `http://localhost:${PORT}`,
        reuseExistingServer: !process.env.CI,
        timeout: 60_000,
      },
});
