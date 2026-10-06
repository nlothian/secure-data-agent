// Hand-written types for the vendored ./transformers_worker.js (not generated).
// The site uses it only through ./opt_zeos_worker.js.
import type { DecodeStepOptions, ZeosModelWorkerLike } from './model_channel';
import type { OptZeosStepResult } from './opt_zeos_worker';

export const SNAPSHOT_EVERY: number;
export function encodePlain(tokenizer: unknown, text: string): number[];
/**
 * The bytes a byte-level BPE token stands for: a token that is part of a
 * character decodes to U+FFFD on its own, and these are its real bytes.
 */
export function pieceBytes(tokenizer: unknown, tokenId: number): Uint8Array;
export function sampleToken(
  logits: Float32Array,
  allowedTokens: Uint8Array | null,
  limit: number,
  sample: { temperature: number; topK: number; u: number },
): number;

/** ZEOS's transformers.js-export model worker (not used by the site). WebGPU only. */
export class TransformersWorker implements ZeosModelWorkerLike {
  /** Any option not listed is refused (it throws), as is a backend other than `webgpu`. */
  static load(options: {
    ort: unknown;
    Tokenizer: unknown;
    read: (name: string) => Uint8Array | Promise<Uint8Array>;
    backend?: 'webgpu';
    numThreads?: number;
    sessionOptions?: Record<string, unknown>;
    onActivity?: ((activity: Record<string, unknown>) => void) | null;
  }): Promise<TransformersWorker>;
  constructor(deps: {
    ort: unknown;
    tokenizer: unknown;
    meta: Record<string, unknown>;
    session: unknown;
    backend: string;
    onActivity?: ((activity: Record<string, unknown>) => void) | null;
  });
  meta: { tokenizerSize: number; [key: string]: unknown };
  backend: string;
  info(): unknown;
  tokenize(text: string): Int32Array | number[];
  piece(tokenId: number): string;
  pieceBytes(tokenId: number): Uint8Array;
  createContext(jobId: string): void;
  destroyContext(jobId: string): void;
  length(jobId: string): number;
  append(jobId: string, ids: Int32Array | number[]): void | Promise<void>;
  truncate(jobId: string, n: number): void;
  fork(parentId: string, childId: string): void;
  decodeStep(jobId: string, opts: DecodeStepOptions): Promise<OptZeosStepResult>;
}
