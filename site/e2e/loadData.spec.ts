import { test, expect } from './fixtures';
import { dispatchLoadData, seedSandbox, warmDuckDB } from './helpers/loadData';

// End-to-end coverage for the LoadData `/input/...` path fix. We seed an OPFS
// directory with a small CSV, install it as the sandbox dir via the test seam
// exported from sandboxStore, and exercise the full runAgentTool dispatch for
// the `/input/...` form. Releases the Step/Play gate via the toolDebugger so
// the gated tool actually completes. The other path forms (bare, `sandbox:`,
// `file://`, `./`) are parseLoadDataInput's vitest cases in
// src/lib/agentTools.test.ts, and dataErrorClear.spec.ts loads a bare path
// through this same dispatch.
//
// Sandbox loads are fast (no network, and beforeEach starts DuckDB first), so
// we tighten the helper's default timeouts here — a regression that pushes
// either phase past 5 s should fail loudly rather than silently soak up the
// 30 s default.

const SANDBOX_TIMEOUTS = { pendingTimeoutMs: 5000, resultTimeoutMs: 5000 };

test.describe('LoadData sandbox-path forms', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for the chat sidebar to mount — that's the signal the React
    // islands have finished hydrating and the dynamic imports we use are
    // wired up.
    await expect(page.getByText('Choose model')).toBeVisible();
    await seedSandbox(page);
    await warmDuckDB(page);
  });

  test('accepts the /input/... form used by ListFiles/ReadLines', async ({
    page,
  }) => {
    const res = await dispatchLoadData(
      page,
      '/input/mini.csv',
      'inputForm',
      SANDBOX_TIMEOUTS,
    );
    expect(res).toMatchObject({
      name: 'inputForm',
      rowCount: 2,
      source: 'sandbox',
      sourcePath: 'mini.csv',
    });
  });
});
