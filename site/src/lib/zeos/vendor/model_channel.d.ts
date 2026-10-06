// Hand-written types for the vendored ./model_channel.js (not generated).

/** Bytes of the SharedArrayBuffer a model thread answers into. */
export const CHANNEL_BYTES: number;

/** The method surface a model thread serves (ZEOS `ZeosModelWorker`). */
export interface ZeosModelWorkerLike {
  // `serveRequest` reads `meta.tokenizerSize` to answer `pieces`, and `backend`.
  meta: { tokenizerSize: number };
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
  decodeStep(
    jobId: string,
    opts: { allowedBlocks: Uint8Array | null; allowedTokens: Uint8Array | null },
  ): unknown;
}

/** Model side: answer every request arriving through `onMessage` into `buffer`. */
export function serveChannel(
  worker: ZeosModelWorkerLike,
  buffer: SharedArrayBuffer,
  onMessage: (handle: (request: unknown) => void) => void,
): void;

/** Kernel side: the synchronous `ZeosModelWorker` Python calls. */
export class SyncModelWorker {
  constructor(
    buffer: SharedArrayBuffer,
    post: (message: unknown) => void,
    options?: { timeoutMs?: number },
  );
  readonly backend: string;
  call(method: string, ...args: unknown[]): unknown;
  piece(tokenId: number): string;
  /** From one `partialPieces` call on first use; a RangeError for an id that is whole characters. */
  pieceBytes(tokenId: number): Uint8Array;
}
