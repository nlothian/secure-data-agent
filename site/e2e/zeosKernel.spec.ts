import { expect, test } from '@playwright/test';
import { requireZeosSync } from './helpers/zeosSync';

// Smoke test for the ZEOS plumbing: the page is cross-origin isolated, the
// kernel worker boots Pyodide 314 from jsDelivr and micropip-installs the
// wheels from public/zeos/, and the kernel drives the stub model thread
// synchronously over the SharedArrayBuffer/Atomics channel for a few ticks of
// the coop-count-scripted case. Needs network (jsDelivr) and
// `npm run zeos:sync` (public/zeos/ is generated and gitignored).

test.describe('ZEOS kernel worker', () => {
  requireZeosSync(test);

  test('page is cross-origin isolated', async ({ page }) => {
    await page.goto('/');
    expect(await page.evaluate(() => crossOriginIsolated)).toBe(true);
  });

  test('boots, attaches the stub model and steps coop-count-scripted', async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto('/');
    const result = await page.evaluate(async () => {
      const h = await import(/* @vite-ignore */ '/src/lib/zeos/zeosHost.ts');
      const events: [string, unknown][] = [];
      const kernel = await h.startZeosKernel({ onEvent: (k: string, d: unknown) => events.push([k, d]) });
      try {
        const dir = kernel.caseDir('coop-count-scripted');
        const tapes = JSON.parse(await kernel.call('zeos_coop_count_web.page', 'tapes_json', [dir]));
        const model = await kernel.attachModel({
          modelWorker: h.createStubModelWorker,
          init: { tapes },
        });
        const run = await kernel.call('zeos_coop_count_web.page', 'open_run', [dir, 'js'], {
          schedule: true,
          worker: h.modelRef(model.name),
          max_ticks: 200,
        });
        const lines: string[] = [];
        for (let i = 0; i < 6; i++) lines.push(...(await kernel.callMethod(run, 'step')));
        const ticks = await kernel.getAttr(run, 'ticks');
        const transcript = await kernel.callMethod(run, 'transcript_lines');
        const journal: Uint8Array = await kernel.callMethod(run, 'journal_bytes');
        await kernel.release(run);
        const viaExec = await kernel.exec(
          "import _zeos_rpc\n_zeos_rpc.emit('smoke', {'ok': True})\n_zeos_rpc.model().backend",
        );
        let pyError: { name: string; pythonType: string } | null = null;
        try {
          await kernel.callMethod(run, 'step');
        } catch (e) {
          pyError = { name: (e as Error).name, pythonType: (e as { pythonType: string }).pythonType };
        }
        return {
          boot: kernel.boot,
          backend: model.backend,
          decodes: lines.filter((l) => l.includes('"machine.decode"')).length,
          ticks,
          transcript,
          journalBytes: journal.byteLength,
          isBytes: journal instanceof Uint8Array,
          viaExec,
          events,
          pyError,
        };
      } finally {
        kernel.dispose();
      }
    });

    expect(result.boot.isolated).toBe(true);
    expect(result.boot.pyodide).toBe('314.0.7');
    expect(result.boot.cases).toContain('coop-count-scripted');
    expect(result.backend).toBe('stub');
    expect(result.decodes).toBeGreaterThan(0);
    expect(result.ticks).toBe(6);
    expect(result.transcript[0]).toMatch(/counter-a\s+say 1/);
    expect(result.isBytes).toBe(true);
    expect(result.journalBytes).toBeGreaterThan(0);
    expect(result.viaExec).toBe('stub');
    expect(result.events).toEqual([['smoke', { ok: true }]]);
    expect(result.pyError).toEqual({ name: 'ZeosKernelError', pythonType: 'KeyError' });
  });
});
