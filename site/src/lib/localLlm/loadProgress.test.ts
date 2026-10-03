import { describe, expect, it } from 'vitest';
import { createLoadProgressAggregator, type LoadProgressSnapshot } from './loadProgress';

function setup(expectedFiles = [
  { path: 'a', bytes: 100 },
  { path: 'b', bytes: 300 },
]) {
  const changes: LoadProgressSnapshot[] = [];
  const agg = createLoadProgressAggregator({
    label: 'Gemma 4 E2B',
    fromCache: false,
    expectedFiles,
    onChange: (s) => changes.push(s),
  });
  return { agg, changes };
}

describe('createLoadProgressAggregator', () => {
  it('starts at 0% in the fetch phase', () => {
    const { agg, changes } = setup();
    expect(agg.snapshot()).toEqual({
      label: 'Gemma 4 E2B',
      pct: 0,
      fromCache: false,
      phase: 'fetch',
    });
    expect(changes).toEqual([]);
  });

  it('weights progress by the seeded file sizes', () => {
    const { agg } = setup();
    agg.onEvent({ status: 'progress', file: 'a', loaded: 100, total: 100 });
    expect(agg.snapshot().pct).toBe(25);
    agg.onEvent({ status: 'progress', file: 'b', loaded: 150, total: 300 });
    expect(agg.snapshot().pct).toBe(63); // 250/400 = 62.5
  });

  it('lets a positive event total override the seeded size', () => {
    const { agg } = setup();
    agg.onEvent({ status: 'progress', file: 'b', loaded: 100, total: 100 });
    // a: 0/100, b: 100/100 → 50%
    expect(agg.snapshot().pct).toBe(50);
    agg.onEvent({ status: 'progress', file: 'a', loaded: 50, total: 0 });
    // total 0 ignored → a: 50/100 → 150/200
    expect(agg.snapshot().pct).toBe(75);
  });

  it('registers files it was not told about', () => {
    const { agg } = setup([{ path: 'a', bytes: 100 }]);
    agg.onEvent({ status: 'initiate', file: 'extra', total: 100 });
    agg.onEvent({ status: 'progress', file: 'a', loaded: 100, total: 100 });
    expect(agg.snapshot().pct).toBe(50);
    agg.onEvent({ status: 'progress', file: 'other', loaded: 0, total: 200 });
    expect(agg.snapshot().pct).toBe(50); // monotonic: 100/400 would be 25
  });

  it('treats done as fully loaded', () => {
    const { agg } = setup();
    agg.onEvent({ status: 'done', file: 'b' });
    expect(agg.snapshot().pct).toBe(75);
    agg.onEvent({ status: 'done', file: 'a' });
    expect(agg.snapshot().pct).toBe(100);
  });

  it('never decreases', () => {
    const { agg } = setup();
    agg.onEvent({ status: 'progress', file: 'a', loaded: 100, total: 100 });
    agg.onEvent({ status: 'progress', file: 'a', loaded: 10, total: 100 });
    expect(agg.snapshot().pct).toBe(25);
  });

  it('clamps to 100', () => {
    const { agg } = setup([{ path: 'a', bytes: 100 }]);
    agg.onEvent({ status: 'progress', file: 'a', loaded: 500 });
    expect(agg.snapshot().pct).toBe(100);
  });

  it('beginInit jumps to 100% in the init phase', () => {
    const { agg, changes } = setup();
    agg.onEvent({ status: 'progress', file: 'a', loaded: 100, total: 100 });
    agg.beginInit();
    expect(agg.snapshot()).toMatchObject({ pct: 100, phase: 'init' });
    expect(changes.at(-1)).toMatchObject({ pct: 100, phase: 'init' });
  });

  it('calls onChange only when pct or phase changes', () => {
    const { agg, changes } = setup();
    agg.onEvent({ status: 'initiate', file: 'a' });
    agg.onEvent({ status: 'download', file: 'a' });
    expect(changes).toHaveLength(0);
    agg.onEvent({ status: 'progress', file: 'a', loaded: 100, total: 100 });
    agg.onEvent({ status: 'progress', file: 'a', loaded: 100, total: 100 });
    agg.onEvent({ status: 'done', file: 'a' });
    expect(changes.map((c) => c.pct)).toEqual([25]);
    agg.beginInit();
    agg.beginInit();
    expect(changes.map((c) => [c.pct, c.phase])).toEqual([
      [25, 'fetch'],
      [100, 'init'],
    ]);
  });

  it('passes fromCache through from the caller', () => {
    const agg = createLoadProgressAggregator({
      label: 'x',
      fromCache: true,
      expectedFiles: [],
      onChange: () => {},
    });
    expect(agg.snapshot().fromCache).toBe(true);
  });
});
