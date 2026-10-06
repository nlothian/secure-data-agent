// Hand-written types for the vendored ./stub_worker.js (not generated).
// The script defines `globalThis.createStubWorker` as a side effect. The stub
// has a `meta` getter (do not assign it) but no `backend`; a page serving it
// over a channel sets `backend`.
export {};
declare global {
  // eslint-disable-next-line no-var
  var createStubWorker: (
    tapes: Record<string, string[]>,
    options?: {
      blockSize?: number;
      terminator?: string;
      /** Pretend costs, with real awaits so a cancel can land between chunks: per
       * prefilled position, and for the final decode. Both 0 by default (a step
       * is then synchronous). */
      positionMs?: number;
      stepMs?: number;
    },
  ) => Omit<import('./model_channel').ZeosModelWorkerLike, 'info'> & { info(): { vocabSize: number } };
}
