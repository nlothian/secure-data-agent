// Hand-written types for the vendored ./model_channel.js (not generated).

/** Bytes of the SharedArrayBuffer a model thread answers into. */
export const CHANNEL_BYTES: number;
/** What a synchronous call throws while a begun decode step is in flight. */
export const CHANNEL_BUSY: string;

/** What one decode step reports besides its token: positions now cached for
 * the context, and the runs of the graph this step made. */
export interface DecodeStepStats {
  positions: number;
  chunks: number;
  fillMs: number;
}

/** A decode step's options as a model worker receives them. */
export interface DecodeStepOptions {
  allowedBlocks: Uint8Array | null;
  allowedTokens: Uint8Array | null;
  /** Seeded sampling; greedy (argmax) when absent or null. */
  sample?: { temperature: number; topK: number; u: number } | null;
  /** The most positions one prefill run may take, for this step only. */
  maxChunk?: number | null;
  /**
   * Asked before every run of the graph; true ends the step with
   * `{cancelled: true, resident, stats}`. Only a begun step
   * (`SyncModelWorker.beginDecodeStep`) gets one, injected by `serveChannel`.
   */
  shouldStop?: (() => boolean) | null;
}

/** A decode step's options as the kernel side passes them (masks as any
 * array of 0/1, copied into `Uint8Array`s for the channel). */
export interface SyncDecodeStepOptions {
  allowedBlocks?: ArrayLike<number> | null;
  allowedTokens?: ArrayLike<number> | null;
  sample?: DecodeStepOptions['sample'];
  maxChunk?: number | null;
}

/** The method surface a model thread serves (ZEOS `ZeosModelWorker`). */
export interface ZeosModelWorkerLike {
  // `serveRequest` reads `meta.tokenizerSize` to answer `pieces`, and `backend`.
  readonly meta: { tokenizerSize: number };
  backend: string;
  info(): unknown;
  tokenize(text: string): Int32Array | number[];
  piece(tokenId: number): string;
  /**
   * The bytes of a token whose piece is not whole characters (it holds U+FFFD:
   * one byte of an emoji, say). Asked only for such ids, through `serveRequest`'s
   * `partialPieces`; a worker whose pieces are all whole characters never is.
   */
  pieceBytes?(tokenId: number): Uint8Array;
  createContext(jobId: string): void;
  destroyContext(jobId: string): void;
  length(jobId: string): number;
  append(jobId: string, ids: Int32Array | number[]): void;
  truncate(jobId: string, n: number): void;
  fork(parentId: string, childId: string): void;
  decodeStep(jobId: string, opts: DecodeStepOptions): unknown;
}

/**
 * Model side: answer every request arriving through `onMessage` into `buffer`,
 * in order. A begun decode step is given a `shouldStop` that reads the abort
 * slot.
 */
export function serveChannel(
  worker: ZeosModelWorkerLike,
  buffer: SharedArrayBuffer,
  onMessage: (handle: (request: unknown) => void) => void,
): void;

/**
 * Kernel side: the synchronous `ZeosModelWorker` Python calls. Buffer layout:
 * int32 slot 0 the state (idle / waiting / answered), slot 1 the reply's
 * length, slot 2 the abort token and slot 3 the in-flight marker of a begun
 * step (both `requestId + 1`); the reply frame starts at byte 16. A call that
 * times out, or a reply naming another request, leaves the channel unusable:
 * every later call throws `model channel unusable: …`.
 */
export class SyncModelWorker {
  constructor(
    buffer: SharedArrayBuffer,
    post: (message: unknown) => void,
    options?: { timeoutMs?: number },
  );
  readonly backend: string;
  /** Why the channel can no longer be used (a call timed out), or null. */
  readonly broken: string | null;
  /** Whether a begun step's result has not yet been taken by `pollDecode`. */
  readonly inFlight: boolean;
  call(method: string, ...args: unknown[]): unknown;
  info(): unknown;
  piece(tokenId: number): string;
  /** From one `partialPieces` call on first use; a RangeError for an id that is whole characters. */
  pieceBytes(tokenId: number): Uint8Array;
  decodeStep(
    jobId: string,
    opts?: SyncDecodeStepOptions,
  ): unknown;
  /** Post one decode step and return its request id; never waits. */
  beginDecodeStep(
    jobId: string,
    opts?: SyncDecodeStepOptions,
  ): number;
  /** The begun step's answer, or null if it has not landed within `timeoutMs` (or nothing is in flight). */
  pollDecode(
    timeoutMs?: number,
  ):
    | { cancelled: false; tokenId: number; attention: Float32Array | null; resident: number; stats: DecodeStepStats }
    | { cancelled: true; resident: number; stats: DecodeStepStats }
    | null;
  /** Ask the begun step to stop before its next prefill chunk; never waits. */
  cancelDecode(): void;
}
