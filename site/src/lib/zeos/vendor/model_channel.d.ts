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
}
