import { useSyncExternalStore } from 'react';
import {
  subscribe,
  getSnapshot,
  getServerSnapshot,
  type PaneStatus,
} from '../lib/executionPanelStore';

function isBusy(status: PaneStatus): boolean {
  return status === 'pending' || status === 'running';
}

export default function Throbber() {
  const snap = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const labels: string[] = [];
  if (snap.restoring) {
    // Restore takes precedence: nothing else can run until it clears.
    labels.push('Reconstructing state');
  } else {
    const dl = snap.llm.modelDownload;
    if (dl) {
      if (dl.phase === 'init') {
        labels.push(`Loading ${dl.label} onto GPU`);
      } else {
        const verb = dl.fromCache ? 'Loading' : 'Downloading';
        labels.push(`${verb} ${dl.label} · ${dl.pct}%`);
      }
    }
    if (isBusy(snap.data.status)) labels.push('Loading data');
    if (isBusy(snap.sql.status)) labels.push('Running SQL');
    if (isBusy(snap.python.status)) labels.push('Running Python');
    if (snap.llm.compacting) labels.push('Compacting');
    else if (snap.llm.preparingToolCall && labels.length === 0) {
      const name = snap.llm.preparingToolCall.name;
      labels.push(name ? `Calling ${name}…` : 'Preparing tool call…');
    }
    else if (snap.llm.active && labels.length === 0) labels.push('Thinking');
  }

  if (labels.length === 0) return null;
  const label = labels.join(' · ');

  return (
    <div
      className="throbber"
      data-active="true"
      data-tour-id="chat.throbber"
      role="status"
      aria-live="polite"
    >
      <span className="throbber-spinner" aria-hidden="true" />
      <span className="throbber-label">{label}</span>
    </div>
  );
}
