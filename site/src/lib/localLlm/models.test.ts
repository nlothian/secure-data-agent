import { describe, expect, it } from 'vitest';
import { LOCAL_GEMMA_ENDPOINT, type LLMConfig } from '../../types/llm';
import {
  DEFAULT_LOCAL_GEMMA_ID,
  LOCAL_GEMMA_MODELS,
  formatGB,
  getLocalGemmaModel,
  isLocalGemmaId,
  resolveActiveLocalModelIdOrDefault,
} from './models';
import { totalBytes } from './modelFiles';

function cfg(localId: string | undefined): LLMConfig {
  return {
    models: localId === undefined ? {} : { [LOCAL_GEMMA_ENDPOINT]: localId },
  } as unknown as LLMConfig;
}

describe('LOCAL_GEMMA_MODELS', () => {
  it('lists exactly the E2B and E4B models', () => {
    expect(LOCAL_GEMMA_MODELS.map((m) => m.id)).toEqual(['gemma-4-e2b', 'gemma-4-e4b']);
    expect(isLocalGemmaId('gemma-4-e2b')).toBe(true);
    expect(isLocalGemmaId('gemma-4-e4b')).toBe(true);
    expect(isLocalGemmaId('custom:foo')).toBe(false);
    expect(getLocalGemmaModel('gemma-4-e4b')?.label).toBe('Gemma 4 E4B');
  });

  it('defaults to a predefined model', () => {
    expect(isLocalGemmaId(DEFAULT_LOCAL_GEMMA_ID)).toBe(true);
  });

  it.each(LOCAL_GEMMA_MODELS.map((m) => [m.id, m] as const))(
    '%s points at an onnx-community Gemma 4 repo',
    (_id, m) => {
      expect(m.hfRepoId).toMatch(/^onnx-community\/gemma-4-E[24]B-it-ONNX$/);
    },
  );

  it.each(LOCAL_GEMMA_MODELS.map((m) => [m.id, m] as const))(
    '%s approxBytes is within 1%% of the file manifest',
    (_id, m) => {
      const actual = totalBytes(m);
      expect(Math.abs(m.approxBytes - actual) / actual).toBeLessThan(0.01);
    },
  );
});

describe('resolveActiveLocalModelIdOrDefault', () => {
  it('passes through a predefined id', () => {
    expect(resolveActiveLocalModelIdOrDefault(cfg('gemma-4-e4b'))).toBe('gemma-4-e4b');
  });

  it('falls back to the default when unset', () => {
    expect(resolveActiveLocalModelIdOrDefault(cfg(undefined))).toBe(DEFAULT_LOCAL_GEMMA_ID);
  });

  it('falls back to the default for a stale custom id', () => {
    expect(resolveActiveLocalModelIdOrDefault(cfg('custom:not-registered'))).toBe(
      DEFAULT_LOCAL_GEMMA_ID,
    );
  });
});

describe('formatGB', () => {
  it('formats decimal gigabytes to one place', () => {
    expect(formatGB(3_130_000_000)).toBe('3.1 GB');
    expect(formatGB(4_925_000_000)).toBe('4.9 GB');
    expect(formatGB(0)).toBe('0.0 GB');
  });
});
