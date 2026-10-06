/**
 * What the chat UI shows of the ZEOS session: kernel start-up, the job's
 * trust (integrity, session floor, what demoted it), a tool call waiting for
 * the user's approval, and (dev) the journal. Written by `streamZeos`, read by
 * `ZeosTrustIndicator`, `ZeosApprovalCard` and `ZeosJournalView` through
 * `useSyncExternalStore`.
 */

export interface ZeosApproval {
  /** The machine's index of the call within its job. */
  call: number;
  name: string;
  args: Record<string, unknown>;
  /** Why the kernel refused it, in words. */
  reason: string;
  integrity: number;
  sessionFloor: number | null;
  effectiveIntegrity: number | null;
}

/**
 * A call on the approval card. `id` is unique for the page's lifetime, so an
 * answer meant for one card can never settle the next (a double-click on
 * Approve must not approve a call whose arguments were never shown).
 */
export interface ZeosPendingApproval extends ZeosApproval {
  id: number;
}

/** The kernel's spoof alarm: a delivery that spelled a kernel frame (inert data). */
export interface ZeosSpoof {
  pipe: string | null;
  detail: string;
  /** e.g. "ReadLines result #2", when it was a tool result. */
  label: string | null;
}

/** A tool name the model chose with the ring-3 deliveries hidden. */
export interface ZeosMaskedCall {
  name: string;
  /** What was hidden, e.g. "ReadLines result #2". */
  hidden: string[];
}

export interface ZeosSnapshot {
  /** The gate mode of the open conversation (or the configured one before it opens). */
  gateMode: 'strict' | 'attention';
  /** Whether the open conversation masks tool choice (`open_chat(mask_tool_choice=…)`). */
  maskToolChoice: boolean;
  status: 'idle' | 'starting' | 'ready' | 'error';
  statusText: string;
  error: string | null;
  backend: string | null;
  /** The job's integrity watermark; null with no open conversation. */
  integrity: number | null;
  sessionFloor: number | null;
  /** e.g. "ReadLines result #3", or "an earlier assistant turn". */
  demotedBy: string | null;
  /** The call waiting for the user, with the id Approve/Deny must name. */
  pending: ZeosPendingApproval | null;
  /** Journal lines (JSON), newest last, capped. */
  journal: string[];
  /** Spoof alarms in this conversation, oldest first. */
  spoofs: ZeosSpoof[];
  /** Tool names chosen masked in this conversation, oldest first. */
  masked: ZeosMaskedCall[];
}

const JOURNAL_CAP = 2000;

const INITIAL: ZeosSnapshot = {
  gateMode: 'strict',
  maskToolChoice: false,
  status: 'idle',
  statusText: '',
  error: null,
  backend: null,
  integrity: null,
  sessionFloor: null,
  demotedBy: null,
  pending: null,
  journal: [],
  spoofs: [],
  masked: [],
};

let snapshot: ZeosSnapshot = INITIAL;
const listeners = new Set<() => void>();
let decide: ((approved: boolean) => void) | null = null;
let nextApprovalId = 1;

function set(patch: Partial<ZeosSnapshot>): void {
  snapshot = { ...snapshot, ...patch };
  for (const l of listeners) l();
}

export function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getSnapshot(): ZeosSnapshot {
  return snapshot;
}

export function getServerSnapshot(): ZeosSnapshot {
  return INITIAL;
}

export function setStatus(
  status: ZeosSnapshot['status'],
  statusText = '',
  extra: Partial<Pick<ZeosSnapshot, 'error' | 'backend'>> = {},
): void {
  set({ status, statusText, error: status === 'error' ? (extra.error ?? statusText) : null, ...extra });
}

export function setTrust(
  trust: Partial<Pick<ZeosSnapshot, 'integrity' | 'sessionFloor' | 'demotedBy' | 'gateMode' | 'maskToolChoice'>>,
): void {
  set(trust);
}

/** A new conversation: forget the old one's trust and journal. */
export function resetConversation(): void {
  cancelApproval();
  set({ integrity: null, sessionFloor: null, demotedBy: null, journal: [], spoofs: [], masked: [] });
}

/** Record a tool name chosen masked, and add it to the journal as a `ui.masked` line. */
export function noteMasked(call: ZeosMaskedCall): void {
  set({ masked: [...snapshot.masked, call] });
  appendJournal([JSON.stringify({ kind: 'ui.masked', ...call })]);
}

/**
 * Record a spoof alarm, and add it to the journal as a `ui.spoof` line (the
 * kernel's own `fault.raised` line does not say which tool result it was).
 */
export function noteSpoof(spoof: ZeosSpoof): void {
  set({ spoofs: [...snapshot.spoofs, spoof] });
  appendJournal([JSON.stringify({ kind: 'ui.spoof', ...spoof })]);
}

export function appendJournal(lines: readonly string[]): void {
  if (lines.length === 0) return;
  const journal = [...snapshot.journal, ...lines];
  set({ journal: journal.length > JOURNAL_CAP ? journal.slice(-JOURNAL_CAP) : journal });
}

/**
 * Show the approval card and wait for the user. Resolves true on Approve,
 * false on Deny; rejects with an AbortError if `signal` aborts first.
 */
export function requestApproval(approval: ZeosApproval, signal?: AbortSignal): Promise<boolean> {
  cancelApproval();
  return new Promise<boolean>((resolve, reject) => {
    const onAbort = (): void => {
      decide = null;
      set({ pending: null });
      reject(new DOMException('Aborted', 'AbortError'));
    };
    if (signal?.aborted) return onAbort();
    signal?.addEventListener('abort', onAbort, { once: true });
    decide = (approved) => {
      signal?.removeEventListener('abort', onAbort);
      decide = null;
      set({ pending: null });
      resolve(approved);
    };
    set({ pending: { ...approval, id: nextApprovalId++ } });
  });
}

/** Approve the call on the card with this `id`; an answer for any other card does nothing. */
export function approve(id: number): void {
  if (snapshot.pending?.id === id) decide?.(true);
}

/** Deny the call on the card with this `id`; an answer for any other card does nothing. */
export function deny(id: number): void {
  if (snapshot.pending?.id === id) decide?.(false);
}

/** Drop a pending approval without answering (the stream was torn down). */
function cancelApproval(): void {
  if (decide) decide(false);
}

/** Tests. */
export function resetForTests(): void {
  decide = null;
  snapshot = INITIAL;
  for (const l of listeners) l();
}
