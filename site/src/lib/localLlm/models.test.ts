import { describe, expect, it } from 'vitest';
import { LOCAL_GEMMA_ENDPOINT, type LLMConfig } from '../../types/llm';
import {
  ALL_LOCAL_MODELS,
  DEFAULT_LOCAL_GEMMA_ID,
  LOCAL_GEMMA_MODELS,
  transformersModelIdFor,
  formatGB,
  getLocalGemmaModel,
  isLocalGemmaId,
  resolveActiveLocalModelIdOrDefault,
} from './models';
import { hasManifest, totalBytes } from './modelFiles';

function cfg(localId: string | undefined): LLMConfig {
  return {
    models: localId === undefined ? {} : { [LOCAL_GEMMA_ENDPOINT]: localId },
  } as unknown as LLMConfig;
}

describe('LOCAL_GEMMA_MODELS', () => {
  it('lists the Gemma 4 E2B / E4B and Qwen 3.5 4B models', () => {
    expect(LOCAL_GEMMA_MODELS.map((m) => m.id)).toEqual([
      'gemma-4-e2b',
      'gemma-4-e4b',
      'qwen3.5-4b',
    ]);
    expect(isLocalGemmaId('gemma-4-e2b')).toBe(true);
    expect(isLocalGemmaId('gemma-4-e4b')).toBe(true);
    expect(isLocalGemmaId('qwen3.5-4b')).toBe(true);
    expect(getLocalGemmaModel('qwen3.5-4b')?.family).toBe('qwen');
    expect(isLocalGemmaId('custom:foo')).toBe(false);
    expect(getLocalGemmaModel('gemma-4-e4b')?.label).toBe('Gemma 4 E4B');
  });

  it('lists ZEOS Qwen 4B only in local-models dev mode', () => {
    const zeos = ALL_LOCAL_MODELS.find((m) => m.id === 'zeos-qwen3.5-4b');
    expect(zeos).toMatchObject({
      label: 'ZEOS Qwen 4B',
      family: 'zeos-qwen',
      hfRepoId: 'metacognitionai/Qwen3.5-4B-ZEOS-OPT',
      sideTaskModelId: 'qwen3.5-4b',
    });
    // Vitest runs with neither PUBLIC_LOCAL_MODELS nor a stub script.
    expect(LOCAL_GEMMA_MODELS.map((m) => m.id)).not.toContain('zeos-qwen3.5-4b');
    expect(isLocalGemmaId('zeos-qwen3.5-4b')).toBe(false);
  });

  it('runs side tasks for a ZEOS model on its plain counterpart', () => {
    expect(transformersModelIdFor('zeos-qwen3.5-4b')).toBe('qwen3.5-4b');
    expect(transformersModelIdFor('gemma-4-e2b')).toBe('gemma-4-e2b');
  });

  it('defaults to a predefined model', () => {
    expect(isLocalGemmaId(DEFAULT_LOCAL_GEMMA_ID)).toBe(true);
  });

  it.each(
    LOCAL_GEMMA_MODELS.filter((m) => m.family === 'gemma').map((m) => [m.id, m] as const),
  )('%s points at an onnx-community Gemma 4 repo', (_id, m) => {
    expect(m.hfRepoId).toMatch(/^onnx-community\/gemma-4-E[24]B-it-ONNX$/);
  });

  it.each(ALL_LOCAL_MODELS.filter(hasManifest).map((m) => [m.id, m] as const))(
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
