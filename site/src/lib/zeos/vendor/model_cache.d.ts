// Hand-written types for the vendored ./model_cache.js (not generated).

export const HUB: string;

/** The directory a Hub repo's files resolve under, at one commit. */
export function hubUrl(options: { endpoint?: string; repo: string; revision: string }): string;

/** A download failed its size or hash check; nothing was stored. */
export class IntegrityError extends Error {}

/** The browser would not store the model; nothing partial is left behind. */
export class QuotaError extends Error {}

/** A Hub repo at one commit: the cache key of every file it stores. */
export interface ModelCacheKey {
  repo: string;
  revision: string;
}

/** One file's size and SHA-256, from the export's `meta.json`. */
export interface ExpectedFile {
  bytes: number;
  sha256?: string;
}

/** The storage `modelFiles` / `modelReader` take (`opfsStore` in the browser). */
export interface ModelStore {
  complete(key: string[]): Promise<number | null>;
  partial(key: string[]): Promise<number>;
  chunks(key: string[], options: { partial: boolean }): AsyncIterable<Uint8Array>;
  writer(
    key: string[],
    from: number,
  ): Promise<{
    write(bytes: Uint8Array): void | Promise<void>;
    flush(): void | Promise<void>;
    close(): void | Promise<void>;
  }>;
  commit(key: string[]): Promise<void>;
  discard(key: string[]): Promise<void>;
  space(): Promise<{ usage: number; quota: number } | null>;
}

export interface ModelSourceOptions {
  /** The export's directory (`hubUrl(...)` for the Hub, or a local `/models/...` URL). */
  url: string;
  /** `{repo, revision}` to keep the files in `store`, or null for no cache. */
  cache: ModelCacheKey | null;
  fetch: typeof fetch;
  /** Required when `cache` is set. */
  store: ModelStore | null;
}

/** The model thread's progress message, with bytes counted over the whole export. */
export interface ModelReadProgress {
  /** `download`, `cache` (reading a stored file) or `verify` (hashing a stored part). */
  phase: 'download' | 'cache' | 'verify';
  file: string;
  loaded: number;
  total: number;
  files: number;
  file_index: number;
  bytes: number;
  bytes_total: number;
}

export function modelFiles(options: ModelSourceOptions): {
  read(
    name: string,
    expected?: ExpectedFile,
    onProgress?: (p: { phase: string; loaded: number; total: number }) => void,
  ): Promise<Uint8Array>;
  missing(files: [string, ExpectedFile][]): Promise<number>;
  checkSpace(needed: number): Promise<void>;
};

/**
 * The `read(name)` a worker's `load` takes: reads `meta.json` first, checks
 * every later file against `meta.files` (size and SHA-256), and checks the
 * store's room before the first large file.
 */
export function modelReader(
  options: ModelSourceOptions & { onProgress?: (progress: ModelReadProgress) => void },
): {
  read(name: string): Promise<Uint8Array>;
  meta(): Record<string, unknown> | null;
  bytesRead(): number;
  bytesTotal(): number;
};
