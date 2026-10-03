import { describe, expect, it } from 'vitest';
import {
  cachedIdsAfterGenerate,
  findReusablePrefix,
  MAX_KEPT_CACHE_TOKENS,
  prefillChunkEnds,
} from './kvReuse';
import { LOCAL_GEMMA_CONTEXT_WINDOW } from '../contextWindow';

describe('findReusablePrefix', () => {
  it('reuses an exact strict extension', () => {
    expect(findReusablePrefix([2, 105, 7], [2, 105, 7, 9, 10])).toBe(3);
  });

  it('reuses when the extension is a single token', () => {
    expect(findReusablePrefix([2, 105, 7], [2, 105, 7, 9])).toBe(3);
  });

  it('rejects divergence at index k', () => {
    expect(findReusablePrefix([2, 105, 7, 8], [2, 105, 99, 8, 9])).toBeNull();
    expect(findReusablePrefix([2, 105, 7, 8], [2, 105, 7, 99, 9])).toBeNull();
    expect(findReusablePrefix([2, 105, 7, 8], [3, 105, 7, 8, 9])).toBeNull();
  });

  it('rejects an equal-length prompt, even an identical one', () => {
    expect(findReusablePrefix([2, 105, 7], [2, 105, 7])).toBeNull();
    expect(findReusablePrefix([2, 105, 7], [2, 105, 8])).toBeNull();
  });

  it('rejects a shorter prompt', () => {
    expect(findReusablePrefix([2, 105, 7, 8], [2, 105, 7])).toBeNull();
  });

  it('rejects an empty kept cache', () => {
    expect(findReusablePrefix([], [2, 105, 7])).toBeNull();
    expect(findReusablePrefix([], [])).toBeNull();
  });
});

describe('prefillChunkEnds', () => {
  it('needs no chunk calls when the whole prompt fits in one chunk', () => {
    expect(prefillChunkEnds(0, 2048, 2048)).toEqual([]);
    expect(prefillChunkEnds(0, 10, 2048)).toEqual([]);
  });

  it('chunks a long prompt from scratch, stopping one short of the end', () => {
    expect(prefillChunkEnds(0, 2049, 2048)).toEqual([2048]);
    expect(prefillChunkEnds(0, 5000, 2048)).toEqual([2048, 4096, 4999]);
    expect(prefillChunkEnds(0, 4097, 2048)).toEqual([2048, 4096]);
  });

  it('chunks relative to the reused prefix', () => {
    // 9,000 cached; a 1,500-token suffix fits in the final generate.
    expect(prefillChunkEnds(9000, 10_500, 2048)).toEqual([]);
    // 9,000 cached; a 5,000-token suffix needs chunks at +2048, +4096, end-1.
    expect(prefillChunkEnds(9000, 14_000, 2048)).toEqual([11_048, 13_096, 13_999]);
  });

  it('never produces a chunk longer than the chunk size', () => {
    for (const [start, total] of [
      [0, 19_999],
      [123, 19_999],
      [17_000, 19_999],
    ]) {
      const ends = prefillChunkEnds(start, total, 2048);
      let prev = start;
      for (const e of ends) {
        expect(e - prev).toBeGreaterThan(0);
        expect(e - prev).toBeLessThanOrEqual(2048);
        prev = e;
      }
      // The final generate's suffix is also within one chunk.
      expect(total - prev).toBeLessThanOrEqual(2048);
    }
  });
});

describe('cachedIdsAfterGenerate', () => {
  it('covers the prompt plus every sampled token except the last', () => {
    expect(cachedIdsAfterGenerate([2, 105], [7, 8, 106])).toEqual([2, 105, 7, 8]);
  });

  it('covers just the prompt after a single sampled token', () => {
    expect(cachedIdsAfterGenerate([2, 105], [106])).toEqual([2, 105]);
  });

  it('is null when nothing was sampled', () => {
    expect(cachedIdsAfterGenerate([2, 105], [])).toBeNull();
  });
});

it('caps kept caches at the context window', () => {
  expect(MAX_KEPT_CACHE_TOKENS).toBe(LOCAL_GEMMA_CONTEXT_WINDOW);
});
