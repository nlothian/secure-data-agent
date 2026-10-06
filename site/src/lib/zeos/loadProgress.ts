/**
 * Pure mapping from the ZEOS model thread's `{progress}` messages
 * (src/workers/zeosOptModel.worker.ts, which forwards ZEOS `modelReader`'s)
 * to the Throbber's `LocalModelDownloadState`, with one monotonic percentage
 * over `bytes` / `bytes_total` (each unique file counted once).
 *
 * - `cache`: a file read from the OPFS cache → "Loading … · N%".
 * - `download`: a file fetched → "Downloading … · N%"; in local-models mode
 *   the files come from the dev server's disk with no cache, so "Loading".
 * - `verify`: hashing the stored part of an interrupted download before
 *   resuming it → "Verifying … · N%". The `verify` that ends every download
 *   (one event, the hash was taken as it streamed) keeps the download's
 *   wording, so it does not flash once per file.
 * - `session`: every file is in and ONNX Runtime builds the sessions →
 *   "Loading … onto GPU".
 */
import type { LocalModelDownloadState } from '../executionPanelStore';

export interface ZeosLoadTracker {
  onProgress(p: Record<string, unknown>): void;
  snapshot(): LocalModelDownloadState | null;
}

export function createZeosLoadTracker(opts: {
  label: string;
  /** Local-models mode: a `download` is a read from the dev server's disk. */
  local: boolean;
  onChange: (s: LocalModelDownloadState) => void;
}): ZeosLoadTracker {
  let state: LocalModelDownloadState | null = null;
  let last: { phase: unknown; file: unknown } | null = null;

  const set = (next: LocalModelDownloadState) => {
    if (
      state &&
      state.pct === next.pct &&
      state.fromCache === next.fromCache &&
      state.phase === next.phase
    ) {
      return;
    }
    state = next;
    opts.onChange(next);
  };

  return {
    onProgress(p) {
      const prev = last;
      last = { phase: p.phase, file: p.file };
      if (p.phase === 'session') {
        set({ label: opts.label, pct: 100, fromCache: state?.fromCache ?? opts.local, phase: 'init' });
        return;
      }
      if (state?.phase === 'init') return;
      const total = typeof p.bytes_total === 'number' ? p.bytes_total : 0;
      // meta.json is read before the export's size is known.
      if (total <= 0 || typeof p.bytes !== 'number') return;
      const raw = Math.round((100 * p.bytes) / total);
      const pct = Math.max(state?.pct ?? 0, Math.min(100, Math.max(0, raw)));
      let fromCache: boolean;
      let phase: LocalModelDownloadState['phase'] = 'fetch';
      if (p.phase === 'cache') {
        fromCache = true;
      } else if (p.phase === 'download') {
        fromCache = opts.local;
      } else if (p.phase === 'verify') {
        if (prev?.phase === 'download' && prev.file === p.file) {
          // The check that ends a download: keep its wording.
          fromCache = state?.fromCache ?? opts.local;
        } else {
          fromCache = true;
          phase = 'verify';
        }
      } else {
        return;
      }
      set({ label: opts.label, pct, fromCache, phase });
    },
    snapshot: () => state,
  };
}
