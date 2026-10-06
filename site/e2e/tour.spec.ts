import { expect, test } from './fixtures';

// The tour never starts by itself, not even on a first visit (an empty
// localStorage); the header's Tour button starts it.

test.describe('Tour', () => {
  test('does not start on a first visit, and the Tour button starts it', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('Choose model')).toBeVisible();

    // Wait until the TourOverlay island has hydrated (Astro drops `ssr`
    // then) and React has had a few frames to run its mount effect, which
    // is where a first-visit auto-start would begin the tour.
    await expect(
      page.locator('astro-island[component-url*="TourOverlay"]:not([ssr])'),
    ).toHaveCount(1);
    await page.evaluate(
      () =>
        new Promise<void>((resolve) => {
          let frames = 3;
          const tick = (): void => {
            if (--frames === 0) setTimeout(resolve, 0);
            else requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        }),
    );
    const active = await page.evaluate(async () => {
      const c = await import('/src/lib/tour/controller.ts');
      return c.getActiveDefinition() !== null;
    });
    expect(active).toBe(false);
    const dialog = page.getByRole('dialog', { name: 'Tour step' });
    await expect(dialog).toHaveCount(0);

    // Without a big enough WebGPU buffer (headless Chromium has no WebGPU)
    // the GPU warning bar covers the header; dismiss it to reach the button.
    const smallGpu = await page.evaluate(async () => {
      const m = await import('/src/lib/localLlm/webgpu.ts');
      const s = await m.detectWebGpu();
      return Math.min(s.maxBufferSize ?? 0, s.maxStorageBufferBindingSize ?? Infinity) < 1.5 * 1024 ** 3;
    });
    if (smallGpu) await page.getByRole('alert').getByRole('button', { name: 'Dismiss' }).click();

    await page.getByRole('button', { name: 'Tour', exact: true }).click();
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText('This tour walks through these features');
  });
});
