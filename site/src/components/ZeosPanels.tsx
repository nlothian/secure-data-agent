import { useEffect, useState, useSyncExternalStore, type MouseEvent } from 'react';
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

/** A tool call whose name the model chose with the ring-3 tool output hidden. */
export function MaskedBadge() {
  return (
    <span
      className="chat-masked-badge"
      title={
        'ZEOS hid the untrusted (ring 3) tool output from the model while it wrote ' +
        "this tool's name, so no tool result chose it. The arguments saw everything."
      }
    >
      name masked
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
  configuredMask,
}: {
  /** The mode the next conversation opens with (config). */
  configuredMode: 'strict' | 'attention';
  /** Whether the next conversation masks tool choice (config). */
  configuredMask?: boolean;
}) {
  const z = useZeos();
  const open = z.integrity !== null;
  const masking = open ? z.maskToolChoice : (configuredMask ?? false);
  const mode = `${open ? z.gateMode : configuredMode}${masking ? '+mask' : ''}`;
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
  const next = `${configuredMode}${configuredMask ? '+mask' : ''}`;
  if (open && next !== mode) {
    label += ` (${next} next)`;
    title += ` The ${next} mode applies from your next message.`;
  }
  if (masking) {
    title += ' Tool names are chosen with the untrusted tool output hidden.';
    if (open && z.masked.length > 0) title += ` ${z.masked.length} so far.`;
  }
  if (z.backend) title += ` (model: ${z.backend})`;
  return (
    <span
      className="chat-zeos-trust"
      data-state={state}
      data-mode={open ? z.gateMode : configuredMode}
      data-masked={masking ? 'true' : 'false'}
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

/**
 * How long a new card's buttons stay disabled: the second click of a
 * double-click on the previous card must not land on this one.
 */
export const APPROVAL_ARM_MS = 400;

/** The kernel refused a side-effecting call: the user decides. */
export function ZeosApprovalCard() {
  const z = useZeos();
  const pending = z.pending;
  if (!pending) return null;
  // A fresh card per call, so nothing (the arming delay included) carries over.
  return <ApprovalCardBody key={pending.id} pending={pending} z={z} />;
}

function ApprovalCardBody({ pending, z }: { pending: zeosStore.ZeosPendingApproval; z: zeosStore.ZeosSnapshot }) {
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setArmed(true), APPROVAL_ARM_MS);
    return () => clearTimeout(t);
  }, []);
  // Each answer names this card's call: a click meant for an earlier card
  // (or a double-click's second click, `detail` 2) settles nothing else.
  const answer = (approved: boolean) => (e: MouseEvent) => {
    if (e.detail > 1) return;
    if (approved) zeosStore.approve(pending.id);
    else zeosStore.deny(pending.id);
  };
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
        <button type="button" className="chat-model-apply" disabled={!armed} onClick={answer(true)}>
          Approve
        </button>
        <button type="button" className="chat-model-cancel" disabled={!armed} onClick={answer(false)}>
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
          {z.masked.length > 0 && ` · ${z.masked.length} name${z.masked.length === 1 ? '' : 's'} masked`}
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
