// Hand-written types for the vendored ./opt_zeos_worker.js (not generated).
import type { DecodeStepOptions, DecodeStepStats, ZeosModelWorkerLike } from './model_channel';

export const SNAPSHOT_EVERY: number;
export const MAX_SNAPSHOTS: number;
export const MAX_TRACKS: number;
/** The shortest hidden run carried past rather than run (`skipHidden`): 16. */
export const MIN_SKIP: number;

/** Throw for a backend other than `webgpu`, or for any key of `unknown`. */
export function refuseOptions(where: string, backend: string, unknown: Record<string, unknown>): void;

/** Whether a `meta.json` describes an OPT+ZEOS export (it names `decoder` and `embedTokens`). */
export function isOptZeosMeta(meta: unknown): boolean;

/**
 * `OptZeosWorker.load`'s options. The worker is WebGPU only; any option not
 * listed here is refused (it throws), not ignored.
 */
export interface OptZeosLoadOptions {
  /** ONNX Runtime Web's WebGPU build. */
  ort: unknown;
  /** The `Tokenizer` class from `@huggingface/tokenizers`. */
  Tokenizer: unknown;
  /** Reads one file of the export, by its path relative to the export directory. */
  read: (name: string) => Uint8Array | Promise<Uint8Array>;
  /** A graph or weights file as bytes, or as a URL ONNX Runtime reads itself (`read` by default). */
  source?: (name: string) => Uint8Array | string | Promise<Uint8Array | string>;
  /** The only backend, and the default; anything else is refused. */
  backend?: 'webgpu';
  sessionOptions?: Record<string, unknown>;
  /** Threads for the kernels ONNX Runtime still runs as WebAssembly beside WebGPU (default 1). */
  numThreads?: number;
  onActivity?: ((activity: Record<string, unknown>) => void) | null;
  snapshotEvery?: number;
  maxSnapshots?: number;
  /** Caches per context, one per mask history: 1 or 2 (the default). */
  maxTracks?: number;
  /** Carry the state past runs of hidden positions without running them (default true). */
  skipHidden?: boolean;
  /** The shortest such run, at least the convolution window (default `MIN_SKIP`). */
  minSkip?: number;
}

/** A decode step's answer: a token, or `cancelled` when `shouldStop` ended it
 * (the positions already run stay cached, `resident` of them). */
export type OptZeosStepResult =
  | { tokenId: number; attention: Float32Array | null; resident: number; stats: DecodeStepStats; cancelled?: undefined }
  | { cancelled: true; resident: number; stats: DecodeStepStats };

export class OptZeosWorker implements ZeosModelWorkerLike {
  static load(options: OptZeosLoadOptions): Promise<OptZeosWorker>;
  meta: { tokenizerSize: number; vocabSize: number; [key: string]: unknown };
  backend: string;
  stats: {
    runs: number;
    positions: number;
    reruns: number;
    rerunPositions: number;
    /** Caches started for a new mask. */
    tracks: number;
    /** Switches back to a kept cache. */
    switches: number;
    /** Hidden positions carried past without a run. */
    skipped: number;
  };
  info(): unknown;
  tokenize(text: string): Int32Array | number[];
  piece(tokenId: number): string;
  /** The byte-level BPE bytes of a token (transformers_worker `pieceBytes`). */
  pieceBytes(tokenId: number): Uint8Array;
  createContext(jobId: string): void;
  destroyContext(jobId: string): void;
  length(jobId: string): number;
  append(jobId: string, ids: Int32Array | number[]): void;
  truncate(jobId: string, n: number): void;
  fork(parentId: string, childId: string): void;
  decodeStep(jobId: string, opts: DecodeStepOptions): Promise<OptZeosStepResult>;
  release(): Promise<void>;
}
