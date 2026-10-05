// Hand-written types for the vendored ./opt_zeos_worker.js (not generated).
import type { ZeosModelWorkerLike } from './model_channel';

export const SNAPSHOT_EVERY: number;
export const MAX_SNAPSHOTS: number;
export const MAX_TRACKS: number;

/** Whether a `meta.json` describes an OPT+ZEOS export (it names `decoder` and `embedTokens`). */
export function isOptZeosMeta(meta: unknown): boolean;

export interface OptZeosLoadOptions {
  /** ONNX Runtime (`onnxruntime-web` or `onnxruntime-node`). */
  ort: unknown;
  /** The `Tokenizer` class from `@huggingface/tokenizers`. */
  Tokenizer: unknown;
  /** Reads one file of the export, by its path relative to the export directory. */
  read: (name: string) => Promise<Uint8Array>;
  backend?: 'webgpu' | 'wasm' | 'cpu';
  sessionOptions?: Record<string, unknown>;
  numThreads?: number;
  onActivity?: ((activity: Record<string, unknown>) => void) | null;
  snapshotEvery?: number;
  maxSnapshots?: number;
  /** Caches per context, one per mask history: 1 or 2 (the default). */
  maxTracks?: number;
  /** Carry the state past runs of hidden positions without running them (default true). */
  skipHidden?: boolean;
}

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
  createContext(jobId: string): void;
  destroyContext(jobId: string): void;
  length(jobId: string): number;
  append(jobId: string, ids: Int32Array | number[]): void;
  truncate(jobId: string, n: number): void;
  fork(parentId: string, childId: string): void;
  decodeStep(
    jobId: string,
    opts: { allowedBlocks: Uint8Array | null; allowedTokens: Uint8Array | null },
  ): Promise<{ tokenId: number; attention: Float32Array | null }>;
  release(): Promise<void>;
}
