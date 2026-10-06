import { chromium, type FullConfig } from '@playwright/test';

/**
 * Warm the dev server before the first test. Every suite run starts a fresh
 * `astro dev` (a GDA_E2E_PORT or the LLM suite never reuses one), and the
 * first page load makes Vite transform every module the app imports, which
 * takes longer than the specs' 5 s `expect` timeout. Without this the first
 * test of a run failed waiting for the chat sidebar ("Choose model") on an
 * empty <main>, while the same test passed on every later load.
 */
export default async function globalSetup(config: FullConfig): Promise<void> {
  const baseURL = config.projects[0]?.use.baseURL;
  if (!baseURL) return;
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    await page.addInitScript(() => localStorage.setItem('tour.seen', '1'));
    await page.goto(baseURL);
    await page.getByText('Choose model').waitFor({ state: 'visible', timeout: 120_000 });
  } finally {
    await browser.close();
  }
}
