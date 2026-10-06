import { describe, expect, it } from 'vitest';
import type { LocalModelDownloadState } from '../executionPanelStore';
import { createZeosLoadTracker } from './loadProgress';

function setup(local = false) {
  const changes: LocalModelDownloadState[] = [];
  const tracker = createZeosLoadTracker({ label: 'ZEOS Qwen 4B', local, onChange: (s) => changes.push(s) });
  return { tracker, changes };
}

const ev = (phase: string, file: string, bytes: number, bytes_total = 1000, loaded = 0, total = 0) => ({
  phase,
  file,
  bytes,
  bytes_total,
  loaded,
  total,
});

describe('createZeosLoadTracker', () => {
  it('ignores meta.json, read before the export size is known', () => {
    const { tracker, changes } = setup();
    tracker.onProgress(ev('download', 'meta.json', 10, 0));
    expect(changes).toEqual([]);
    expect(tracker.snapshot()).toBeNull();
  });

  it('words a Hub download as Downloading and a cache read as Loading', () => {
    const { tracker, changes } = setup();
    tracker.onProgress(ev('cache', 'a', 250));
    expect(tracker.snapshot()).toEqual({ label: 'ZEOS Qwen 4B', pct: 25, fromCache: true, phase: 'fetch' });
    tracker.onProgress(ev('download', 'b', 500));
    expect(tracker.snapshot()).toEqual({ label: 'ZEOS Qwen 4B', pct: 50, fromCache: false, phase: 'fetch' });
    expect(changes).toHaveLength(2);
  });

  it('words a local-models download as Loading', () => {
    const { tracker } = setup(true);
    tracker.onProgress(ev('download', 'a', 100));
    expect(tracker.snapshot()).toMatchObject({ pct: 10, fromCache: true, phase: 'fetch' });
  });

  it('shows a resumed part being hashed as verify', () => {
    const { tracker } = setup();
    tracker.onProgress(ev('cache', 'a', 100));
    tracker.onProgress(ev('verify', 'b', 300));
    expect(tracker.snapshot()).toMatchObject({ pct: 30, fromCache: true, phase: 'verify' });
    tracker.onProgress(ev('download', 'b', 400));
    expect(tracker.snapshot()).toMatchObject({ pct: 40, fromCache: false, phase: 'fetch' });
  });

  it('keeps the download wording for the check that ends a download', () => {
    const { tracker, changes } = setup();
    tracker.onProgress(ev('download', 'a', 500));
    tracker.onProgress(ev('verify', 'a', 500));
    expect(tracker.snapshot()).toMatchObject({ pct: 50, fromCache: false, phase: 'fetch' });
    expect(changes).toHaveLength(1);
  });

  it('never goes backwards (a download restarted from the start)', () => {
    const { tracker } = setup();
    tracker.onProgress(ev('download', 'a', 600));
    tracker.onProgress(ev('download', 'a', 100));
    expect(tracker.snapshot()?.pct).toBe(60);
  });

  it('moves to init on session and ignores anything after', () => {
    const { tracker } = setup();
    tracker.onProgress(ev('cache', 'a', 900));
    tracker.onProgress({ phase: 'session', bytes: 1000, bytes_total: 1000 });
    expect(tracker.snapshot()).toEqual({ label: 'ZEOS Qwen 4B', pct: 100, fromCache: true, phase: 'init' });
    tracker.onProgress(ev('download', 'b', 950));
    tracker.onProgress({ phase: 'ready', ms: 5 });
    expect(tracker.snapshot()?.phase).toBe('init');
  });
});
