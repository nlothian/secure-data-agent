// Hand-written types for the vendored ./stub_worker.js (not generated).
// The script defines `globalThis.createStubWorker` as a side effect.
export {};
declare global {
  // eslint-disable-next-line no-var
  var createStubWorker: (
    tapes: Record<string, string[]>,
    options?: { blockSize?: number; terminator?: string },
  ) => Omit<import('./model_channel').ZeosModelWorkerLike, 'info'> & { info(): { vocabSize: number } };
}
