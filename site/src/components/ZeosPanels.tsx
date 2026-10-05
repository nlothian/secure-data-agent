import { useState, useSyncExternalStore } from 'react';
import * as zeosStore from '../lib/zeos/zeosSessionStore';
import { ChevronRightIcon } from './Icons';

function useZeos(): zeosStore.ZeosSnapshot {
  return useSyncExternalStore(
    zeosStore.subscribe,
    zeosStore.getSnapshot,
    zeosStore.getServerSnapshot,
  );
}

const RING_NAMES: Record<number, string> = {
  0: 'KERNEL',
  1: 'SYSTEM',
  2: 'TRUSTED',
  3: 'EXTERNAL',
};

export function ringName(ring: number): string {
  return RING_NAMES[ring] ?? `ring ${ring}`;
}

/** The ring a tool result arrived on, beside its name in the chat. */
export function RingBadge({ ring }: { ring: number }) {
  return (
    <span
      className="chat-ring-badge"
      data-ring={ring}
      title={`Arrived on ZEOS ring ${ring} (${ringName(ring)})${
        ring >= 3 ? ': untrusted content' : ''
      }`}
    >
      ring {ring}
    </span>
  );
}

/**
 * A tool result that spelled a kernel frame (`<KERNEL>`, `<FAULT …>`, …): the
 * kernel treated it as inert data and raised a spoof alarm.
 */
export function SpoofBadge() {
  return (
    <span
      className="chat-spoof-badge"
      role="img"
      aria-label="Spoof alarm"
      title={
        'This result spells a ZEOS kernel frame (such as <KERNEL> or <FAULT>). ' +
        'The kernel treated it as data, not a notice, and raised a spoof alarm.'
      }
    >
      ⚠ spoof
    </span>
  );
}

/**
 * Chat header: the gate mode, and whether the ZEOS job is still trusted, has
 * read tool output this turn (strict mode: effects need approval until the
 * next message), or was demoted by measured attention, and by what.
 */
export function ZeosTrustIndicator({
  configuredMode,
}: {
  /** The mode the next conversation opens with (config). */
  configuredMode: 'strict' | 'attention';
}) {
  const z = useZeos();
  const open = z.integrity !== null;
  const mode = open ? z.gateMode : configuredMode;
  let label: string;
  let state: 'trusted' | 'floor' | 'demoted' | 'starting' | 'error' | 'idle';
  let title: string;
  if (z.status === 'error') {
    state = 'error';
    label = 'ZEOS error';
    title = z.error ?? 'The ZEOS kernel failed to start.';
  } else if (z.status === 'starting') {
    state = 'starting';
    label = 'ZEOS starting';
    title = z.statusText || 'Starting the ZEOS kernel';
  } else if (open && z.integrity! >= 3) {
    state = 'demoted';
    label = `${mode}: demoted${z.demotedBy ? ` by ${z.demotedBy}` : ''}`;
    title =
      'The model attended untrusted content, so the ZEOS kernel lowered this ' +
      "conversation's integrity to 3. Side-effecting tools need your approval.";
  } else if (open && (z.sessionFloor ?? 2) >= 3) {
    state = 'floor';
    label = `${mode}: read tool output this turn`;
    title =
      'Integrity 2, but the model has read tool output since your last message, so ' +
      'side-effecting tools need your approval until you send another message.';
  } else if (open) {
    state = 'trusted';
    label = `${mode}: trusted`;
    title =
      mode === 'attention'
        ? 'Integrity 2: the model has not measurably attended untrusted content.'
        : 'Integrity 2: the model has only acted on your own messages.';
  } else {
    state = 'idle';
    label = `${mode}: ZEOS`;
    title = 'ZEOS kernel: trust state appears with the first message.';
  }
  if (open && configuredMode !== mode) {
    label += ` (${configuredMode} next)`;
    title += ` The ${configuredMode} mode applies from your next message.`;
  }
  if (z.backend) title += ` (model: ${z.backend})`;
  return (
    <span
      className="chat-zeos-trust"
      data-state={state}
      data-mode={mode}
      title={title}
      role="status"
    >
      <span className="chat-zeos-trust-dot" aria-hidden="true" />
      {label}
    </span>
  );
}

function formatArgs(args: Record<string, unknown>): string {
  return JSON.stringify(args, null, 2);
}

/** The kernel refused a side-effecting call: the user decides. */
export function ZeosApprovalCard() {
  const z = useZeos();
  const pending = z.pending;
  if (!pending) return null;
  return (
    <div className="chat-zeos-approval" role="alertdialog" aria-label="Approve tool call">
      <div className="chat-zeos-approval-head">
        <span className="chat-zeos-approval-title">Approve {pending.name}?</span>
        <span className="chat-zeos-approval-demoted" data-mode={z.gateMode}>
          {pending.integrity >= 3
            ? `${z.gateMode}: demoted${z.demotedBy ? ` by ${z.demotedBy}` : ''}`
            : `${z.gateMode}: read tool output this turn`}
        </span>
      </div>
      <p className="chat-zeos-approval-reason">{pending.reason}</p>
      <pre className="chat-zeos-approval-args">
        <code>{formatArgs(pending.args)}</code>
      </pre>
      <div className="chat-model-confirm-actions">
        <button type="button" className="chat-model-apply" onClick={zeosStore.approve}>
          Approve
        </button>
        <button type="button" className="chat-model-cancel" onClick={zeosStore.deny}>
          Deny
        </button>
      </div>
    </div>
  );
}

/** Dev only: the kernel's journal for this conversation, newest last. */
export function ZeosJournalView() {
  const z = useZeos();
  const [expanded, setExpanded] = useState(false);
  if (!import.meta.env.DEV) return null;
  return (
    <div className="chat-zeos-journal">
      <button
        type="button"
        className="chat-tool-summary"
        data-expanded={expanded ? 'true' : 'false'}
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        <ChevronRightIcon size={14} />
        <span className="chat-tool-name">ZEOS journal</span>
        <span className="chat-zeos-journal-count">
          {z.journal.length} events
          {z.spoofs.length > 0 && ` · ${z.spoofs.length} spoof alarm${z.spoofs.length === 1 ? '' : 's'}`}
        </span>
      </button>
      {expanded && (
        <div className="chat-tool-body">
          <pre>
            <code>{z.journal.slice(-300).join('\n')}</code>
          </pre>
        </div>
      )}
    </div>
  );
}
