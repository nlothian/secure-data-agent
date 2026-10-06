// Hand-written types for the vendored ./frames.js (not generated).
export function encodeFrame(header: Record<string, unknown>): Uint8Array;
export function frameLength(bytes: Uint8Array): number;
export function decodeFrame(bytes: Uint8Array): Record<string, unknown>;
/**
 * Answer one request against a worker. Besides the `ZeosModelWorker` methods it
 * answers `pieces` (every piece), `partialPieces` (`[id, bytes]` for every id
 * whose piece holds U+FFFD, which calls `worker.pieceBytes`) and `backend`. A
 * begun `decodeStep` (`request.begun`) gets `shouldStop` added to its options.
 */
export function serveRequest(
  worker: import('./model_channel').ZeosModelWorkerLike,
  request: { id: number; method: string; args: unknown[]; begun?: boolean },
  options?: { shouldStop?: (() => boolean) | null },
): Promise<{ id: number; ok: boolean; value?: unknown; error?: string }>;
