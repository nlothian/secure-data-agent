import { beforeEach, describe, expect, it, vi } from 'vitest';

// recordPythonTable only mutates the loadedTables map + broadcasts via
// cacheRegistry — no DuckDB-wasm (getDuckDB) is reached, so this runs in
// node. duckdb.ts is import-safe here (apache-arrow is the only top-level
// value import; wasm inits lazily in getDuckDB).
beforeEach(() => {
  vi.resetModules();
});

describe('recordPythonTable — RunPython output surfaces in loadedTables', () => {
  it('records the table tagged python/arrow and broadcasts a register', async () => {
    const duck = await import('./duckdb');
    const cr = await import('./cacheRegistry');

    const onRegister = vi.fn();
    cr.registerCache({
      id: 'probe',
      list: () => [],
      invalidateNames: async () => {},
      onRegister,
    });

    const schema = [
      { name: 'g', type: 'VARCHAR' },
      { name: 'n', type: 'BIGINT' },
    ];
    duck.recordPythonTable('py_out', schema, 7);

    expect(duck.listLoadedTables().find((x) => x.name === 'py_out')).toEqual({
      name: 'py_out',
      url: 'RunPython output',
      format: 'arrow',
      schema,
      rowCount: 7,
      source: 'python',
    });
    expect(onRegister).toHaveBeenCalledWith({
      name: 'py_out',
      source: 'python',
    });
  });

  it('overwrites a prior entry of the same name (re-publish)', async () => {
    const duck = await import('./duckdb');

    duck.recordPythonTable('t', [], 1);
    duck.recordPythonTable('t', [{ name: 'a', type: 'INTEGER' }], 9);

    const rows = duck.listLoadedTables().filter((x) => x.name === 't');
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      rowCount: 9,
      schema: [{ name: 'a', type: 'INTEGER' }],
    });
  });
});

describe('createExtensionAutoloadGuard — overlapping sections', () => {
  // A fake connection that keeps the two settings and logs every query; each
  // query settles a macrotask later, so sections interleave as they would on
  // the real worker.
  function fakeConn(initial = { autoload: true, autoinstall: true }) {
    const settings = { ...initial };
    const log: string[] = [];
    let failOn: RegExp | null = null;
    const conn = {
      async query(sql: string) {
        log.push(sql);
        await new Promise((r) => setTimeout(r, 0));
        if (failOn?.test(sql)) throw new Error(`refused: ${sql}`);
        const set = /^SET (autoload|autoinstall)_known_extensions = (true|false)$/.exec(sql);
        if (set) settings[set[1] as 'autoload' | 'autoinstall'] = set[2] === 'true';
        const row = { ...settings };
        return { get: () => ({ toJSON: () => row }) };
      },
    };
    return { conn, settings, log, failOn: (re: RegExp | null) => (failOn = re) };
  }

  function deferred() {
    let resolve!: () => void;
    let reject!: (e: unknown) => void;
    const promise = new Promise<void>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    return { promise, resolve, reject };
  }

  const tick = () => new Promise((r) => setTimeout(r, 5));
  const reads = (log: string[]) => log.filter((q) => q.startsWith('SELECT current_setting')).length;
  const restores = (log: string[]) => log.filter((q) => q.endsWith('= true')).length;

  async function setup(initial?: { autoload: boolean; autoinstall: boolean }) {
    const { createExtensionAutoloadGuard } = await import('./duckdb');
    const fake = fakeConn(initial);
    return { ...fake, guard: createExtensionAutoloadGuard(async () => fake.conn) };
  }

  // Two guarded calls, the second entering while the first's query runs.
  async function overlap(finishFirst: 'a' | 'b', failing: 'a' | 'b' | null) {
    const { guard, settings, log } = await setup();
    const seen: Record<string, boolean[]> = { a: [], b: [] };
    const gates = { a: deferred(), b: deferred() };
    const run = (k: 'a' | 'b') => {
      const p = guard.guarded(async () => {
        seen[k].push(settings.autoload, settings.autoinstall);
        await gates[k].promise;
        seen[k].push(settings.autoload, settings.autoinstall);
        if (failing === k) throw new Error(`${k} failed`);
        return k;
      });
      p.catch(() => {}); // settled below with allSettled
      return p;
    };
    const a = run('a');
    await tick();
    const b = run('b');
    await tick();
    const [first, second] = finishFirst === 'a' ? (['a', 'b'] as const) : (['b', 'a'] as const);
    gates[first].resolve();
    await tick();
    // The first is done; the other still runs with both settings off.
    expect(settings).toEqual({ autoload: false, autoinstall: false });
    expect(restores(log)).toBe(0);
    gates[second].resolve();
    const results = await Promise.allSettled([a, b]);
    expect(seen).toEqual({ a: [false, false, false, false], b: [false, false, false, false] });
    expect(settings).toEqual({ autoload: true, autoinstall: true });
    expect(reads(log)).toBe(1);
    expect(restores(log)).toBe(2); // one SET per setting, once
    return results;
  }

  it('keeps autoloading off for both and restores once, whichever finishes first', async () => {
    for (const first of ['a', 'b'] as const) {
      const results = await overlap(first, null);
      expect(results).toEqual([
        { status: 'fulfilled', value: 'a' },
        { status: 'fulfilled', value: 'b' },
      ]);
    }
  });

  it('keeps autoloading off and restores once when either call throws', async () => {
    for (const first of ['a', 'b'] as const) {
      for (const failing of ['a', 'b'] as const) {
        const results = await overlap(first, failing);
        const failed = results[failing === 'a' ? 0 : 1];
        const ok = results[failing === 'a' ? 1 : 0];
        expect(failed).toEqual({ status: 'rejected', reason: new Error(`${failing} failed`) });
        expect(ok.status).toBe('fulfilled');
      }
    }
  });

  it('puts back settings that were already off', async () => {
    const { guard, settings, log } = await setup({ autoload: false, autoinstall: true });
    await guard.guarded(async () => expect(settings).toEqual({ autoload: false, autoinstall: false }));
    expect(settings).toEqual({ autoload: false, autoinstall: true });
    expect(log.slice(-2)).toEqual(['SET autoload_known_extensions = false', 'SET autoinstall_known_extensions = true']);
  });

  it('does not run fn when the settings cannot be changed, and restores what it changed', async () => {
    const { guard, settings, failOn } = await setup();
    failOn(/^SET autoinstall_known_extensions = false$/);
    const fn = vi.fn(async () => 1);
    await expect(guard.guarded(fn)).rejects.toThrow('refused');
    expect(fn).not.toHaveBeenCalled();
    expect(settings).toEqual({ autoload: true, autoinstall: true });
  });

  it('never overlaps an outside section with a guarded one, in arrival order', async () => {
    const { guard, settings } = await setup();
    const events: string[] = [];
    const g1 = deferred();
    const o1 = deferred();
    const a = guard.guarded(async () => {
      events.push('guard1 in');
      await g1.promise;
      events.push('guard1 out');
    });
    await tick();
    // An approved query arrives while the guard runs: it waits, and so does
    // a guard that arrives after it.
    const b = guard.outside(async () => {
      events.push(`outside in autoload=${settings.autoload}`);
      await o1.promise;
      events.push('outside out');
    });
    const c = guard.guarded(async () => {
      events.push(`guard2 in autoload=${settings.autoload}`);
    });
    await tick();
    expect(events).toEqual(['guard1 in']);
    g1.resolve();
    await tick();
    expect(events).toEqual(['guard1 in', 'guard1 out', 'outside in autoload=true']);
    // SQL in the outside section turns autoloading on; the next guard turns it off again.
    settings.autoload = true;
    o1.resolve();
    await Promise.all([a, b, c]);
    expect(events).toEqual([
      'guard1 in',
      'guard1 out',
      'outside in autoload=true',
      'outside out',
      'guard2 in autoload=false',
    ]);
    expect(settings).toEqual({ autoload: true, autoinstall: true });
  });

  it('lets a guard that arrives while the last one restores wait for the restore', async () => {
    const { guard, settings, log } = await setup();
    const g1 = deferred();
    const a = guard.guarded(() => g1.promise);
    await tick();
    g1.resolve();
    // Let the first guard reach its restore, then arrive.
    await new Promise((r) => setTimeout(r, 0));
    await Promise.resolve();
    const seen: boolean[] = [];
    const b = guard.guarded(async () => {
      seen.push(settings.autoload, settings.autoinstall);
    });
    await Promise.all([a, b]);
    expect(seen).toEqual([false, false]);
    expect(settings).toEqual({ autoload: true, autoinstall: true });
    expect(reads(log)).toBe(2);
  });
});
