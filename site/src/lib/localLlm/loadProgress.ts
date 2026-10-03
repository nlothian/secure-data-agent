/**
 * Pure aggregator turning transformers.js per-file `progress_callback` events
 * into one monotonic percentage for the throbber.
 *
 * transformers.js 4.3.0 emits `initiate`/`download`/`progress`/`done` even on
 * cache hits, so the events cannot tell us whether bytes are actually coming
 * over the network. `fromCache` therefore comes from the caller's
 * `isModelCached` pre-check (`modelCache.ts`).
 */
import type { RawProgressEvent } from './llmWorkerProtocol';

export interface LoadProgressSnapshot {
  label: string;
  pct: number;
  fromCache: boolean;
  phase: 'fetch' | 'init';
}

export interface LoadProgressAggregator {
  onEvent(e: RawProgressEvent): void;
  /** All files fetched; ONNX Runtime is now building the WebGPU session. */
  beginInit(): void;
  snapshot(): LoadProgressSnapshot;
}

export function createLoadProgressAggregator(opts: {
  label: string;
  fromCache: boolean;
  expectedFiles: readonly { path: string; bytes: number }[];
  onChange: (s: LoadProgressSnapshot) => void;
}): LoadProgressAggregator {
  // Keyed by `e.file`: transformers.js passes the repo-relative path
  // (e.g. `onnx/decoder_model_merged_q4f16.onnx_data`), matching the manifest.
  const files = new Map<string, { loaded: number; total: number }>();
  for (const f of opts.expectedFiles) files.set(f.path, { loaded: 0, total: f.bytes });

  let pct = 0;
  let phase: LoadProgressSnapshot['phase'] = 'fetch';

  const snapshot = (): LoadProgressSnapshot => ({
    label: opts.label,
    pct,
    fromCache: opts.fromCache,
    phase,
  });

  const update = (nextPct: number, nextPhase: LoadProgressSnapshot['phase']) => {
    const clamped = Math.max(pct, Math.min(100, Math.max(0, nextPct)));
    if (clamped === pct && nextPhase === phase) return;
    pct = clamped;
    phase = nextPhase;
    opts.onChange(snapshot());
  };

  const recompute = () => {
    let loaded = 0;
    let total = 0;
    for (const f of files.values()) {
      loaded += Math.min(f.loaded, f.total);
      total += f.total;
    }
    if (total <= 0) return;
    update(Math.round((100 * loaded) / total), phase);
  };

  const entry = (e: RawProgressEvent) => {
    if (!e.file) return undefined;
    let f = files.get(e.file);
    if (!f) {
      f = { loaded: 0, total: e.total ?? 0 };
      files.set(e.file, f);
    }
    return f;
  };

  return {
    onEvent(e) {
      if (phase === 'init') return;
      switch (e.status) {
        case 'initiate':
        case 'download':
          entry(e);
          break;
        case 'progress': {
          const f = entry(e);
          if (!f) return;
          if (typeof e.total === 'number' && e.total > 0) f.total = e.total;
          if (typeof e.loaded === 'number') f.loaded = e.loaded;
          recompute();
          break;
        }
        case 'done': {
          const f = e.file ? files.get(e.file) : undefined;
          if (!f) return;
          f.loaded = f.total;
          recompute();
          break;
        }
        default:
          break;
      }
    },
    beginInit() {
      update(100, 'init');
    },
    snapshot,
  };
}
