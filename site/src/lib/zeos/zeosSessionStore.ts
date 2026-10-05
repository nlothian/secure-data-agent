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

export interface ZeosSnapshot {
  /** The gate mode of the open conversation (or the configured one before it opens). */
  gateMode: 'strict' | 'attention';
  status: 'idle' | 'starting' | 'ready' | 'error';
  statusText: string;
  error: string | null;
  backend: string | null;
  /** The job's integrity watermark; null with no open conversation. */
  integrity: number | null;
  sessionFloor: number | null;
  /** e.g. "ReadLines result #3", or "an earlier assistant turn". */
  demotedBy: string | null;
  pending: ZeosApproval | null;
  /** Journal lines (JSON), newest last, capped. */
  journal: string[];
}

const JOURNAL_CAP = 2000;

const INITIAL: ZeosSnapshot = {
  gateMode: 'strict',
  status: 'idle',
  statusText: '',
  error: null,
  backend: null,
  integrity: null,
  sessionFloor: null,
  demotedBy: null,
  pending: null,
  journal: [],
};

let snapshot: ZeosSnapshot = INITIAL;
const listeners = new Set<() => void>();
let decide: ((approved: boolean) => void) | null = null;

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
  trust: Partial<Pick<ZeosSnapshot, 'integrity' | 'sessionFloor' | 'demotedBy' | 'gateMode'>>,
): void {
  set(trust);
}

/** A new conversation: forget the old one's trust and journal. */
export function resetConversation(): void {
  cancelApproval();
  set({ integrity: null, sessionFloor: null, demotedBy: null, journal: [] });
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
    set({ pending: approval });
  });
}

export function approve(): void {
  decide?.(true);
}

export function deny(): void {
  decide?.(false);
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
