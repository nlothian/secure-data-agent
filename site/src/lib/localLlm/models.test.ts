import { describe, expect, it } from 'vitest';
import { LOCAL_GEMMA_ENDPOINT, type LLMConfig } from '../../types/llm';
import {
  canRunZeos,
  defaultLocalModelId,
  FALLBACK_LOCAL_GEMMA_ID,
  LOCAL_GEMMA_MODELS,
  PREFERRED_LOCAL_GEMMA_ID,
  transformersModelIdFor,
  formatGB,
  getLocalGemmaModel,
  isLocalGemmaId,
  resolveActiveLocalModelIdOrDefault,
  type ZeosCapabilities,
} from './models';
import { hasManifest, totalBytes } from './modelFiles';

function cfg(localId: string | undefined): LLMConfig {
  return {
    models: localId === undefined ? {} : { [LOCAL_GEMMA_ENDPOINT]: localId },
  } as unknown as LLMConfig;
}

describe('LOCAL_GEMMA_MODELS', () => {
  it('lists the Gemma 4 E2B / E4B, Qwen 3.5 4B and ZEOS Qwen 4B models', () => {
    expect(LOCAL_GEMMA_MODELS.map((m) => m.id)).toEqual([
      'gemma-4-e2b',
      'gemma-4-e4b',
      'qwen3.5-4b',
      'zeos-qwen3.5-4b',
    ]);
    expect(isLocalGemmaId('gemma-4-e2b')).toBe(true);
    expect(isLocalGemmaId('gemma-4-e4b')).toBe(true);
    expect(isLocalGemmaId('qwen3.5-4b')).toBe(true);
    expect(getLocalGemmaModel('qwen3.5-4b')?.family).toBe('qwen');
    expect(isLocalGemmaId('custom:foo')).toBe(false);
    expect(getLocalGemmaModel('gemma-4-e4b')?.label).toBe('Gemma 4 E4B');
  });

  it('lists ZEOS Qwen 4B outside local-models mode, loading it from a pinned Hub commit', () => {
    // Vitest runs with neither PUBLIC_LOCAL_MODELS nor a stub script.
    expect(isLocalGemmaId('zeos-qwen3.5-4b')).toBe(true);
    const zeos = getLocalGemmaModel('zeos-qwen3.5-4b');
    expect(zeos).toMatchObject({
      label: 'ZEOS Qwen 4B',
      family: 'zeos-qwen',
      hfRepoId: 'metacognitionai/Qwen3.5-4B-ZEOS-OPT',
      sideTaskModelId: 'qwen3.5-4b',
      hubSource: { repo: 'nlothian/Qwen3.5-4B-ZEOS-OPT_Q4F16' },
    });
    // The OPFS cache keys files by revision, so it must be a commit, not a branch.
    expect(zeos?.hubSource?.revision).toMatch(/^[0-9a-f]{40}$/);
    expect(zeos?.notes).not.toMatch(/local models only/i);
    expect(resolveActiveLocalModelIdOrDefault(cfg('zeos-qwen3.5-4b'))).toBe('zeos-qwen3.5-4b');
  });

  it('gives only the ZEOS model a hubSource', () => {
    expect(LOCAL_GEMMA_MODELS.filter((m) => m.hubSource).map((m) => m.id)).toEqual([
      'zeos-qwen3.5-4b',
    ]);
  });

  it('runs side tasks for a ZEOS model on its plain counterpart', () => {
    expect(transformersModelIdFor('zeos-qwen3.5-4b')).toBe('qwen3.5-4b');
    expect(transformersModelIdFor('gemma-4-e2b')).toBe('gemma-4-e2b');
  });

  it('defaults to predefined models: ZEOS Qwen 4B, else Gemma 4 E2B', () => {
    expect(PREFERRED_LOCAL_GEMMA_ID).toBe('zeos-qwen3.5-4b');
    expect(FALLBACK_LOCAL_GEMMA_ID).toBe('gemma-4-e2b');
    expect(getLocalGemmaModel(PREFERRED_LOCAL_GEMMA_ID)?.family).toBe('zeos-qwen');
  });

  it.each(
    LOCAL_GEMMA_MODELS.filter((m) => m.family === 'gemma').map((m) => [m.id, m] as const),
  )('%s points at an onnx-community Gemma 4 repo', (_id, m) => {
    expect(m.hfRepoId).toMatch(/^onnx-community\/gemma-4-E[24]B-it-ONNX$/);
  });

  it.each(LOCAL_GEMMA_MODELS.filter(hasManifest).map((m) => [m.id, m] as const))(
    '%s approxBytes is within 1%% of the file manifest',
    (_id, m) => {
      const actual = totalBytes(m);
      expect(Math.abs(m.approxBytes - actual) / actual).toBeLessThan(0.01);
    },
  );
});

const CAPABLE: ZeosCapabilities = {
  crossOriginIsolated: true,
  webGpu: { supported: true, f16: true },
  stub: false,
};

describe('canRunZeos / defaultLocalModelId', () => {
  it('picks ZEOS Qwen 4B on an isolated page with WebGPU and shader-f16', () => {
    expect(canRunZeos(CAPABLE)).toBe(true);
    expect(defaultLocalModelId(CAPABLE)).toBe('zeos-qwen3.5-4b');
  });

  it.each<[string, ZeosCapabilities]>([
    ['not cross-origin isolated (Safari)', { ...CAPABLE, crossOriginIsolated: false }],
    ['no WebGPU', { ...CAPABLE, webGpu: { supported: false, reason: 'none' } }],
    ['no shader-f16', { ...CAPABLE, webGpu: { supported: true, f16: false } }],
    ['shader-f16 unknown', { ...CAPABLE, webGpu: { supported: true } }],
    ['the WebGPU check not finished', { ...CAPABLE, webGpu: null }],
    ['stub mode, not isolated', { crossOriginIsolated: false, webGpu: null, stub: true }],
  ])('falls back to Gemma 4 E2B with %s', (_name, caps) => {
    expect(canRunZeos(caps)).toBe(false);
    expect(defaultLocalModelId(caps)).toBe('gemma-4-e2b');
  });

  it('needs no GPU for the ZEOS stub', () => {
    expect(defaultLocalModelId({ crossOriginIsolated: true, webGpu: null, stub: true })).toBe(
      'zeos-qwen3.5-4b',
    );
  });

  it('reads this environment by default (Vitest: no WebGPU, not isolated)', () => {
    expect(defaultLocalModelId()).toBe('gemma-4-e2b');
  });
});

describe('resolveActiveLocalModelIdOrDefault', () => {
  const INCAPABLE: ZeosCapabilities = { ...CAPABLE, webGpu: null };

  it('passes through a saved predefined id, whatever the browser can run', () => {
    expect(resolveActiveLocalModelIdOrDefault(cfg('gemma-4-e4b'), CAPABLE)).toBe('gemma-4-e4b');
    expect(resolveActiveLocalModelIdOrDefault(cfg('gemma-4-e2b'), CAPABLE)).toBe('gemma-4-e2b');
    expect(resolveActiveLocalModelIdOrDefault(cfg('zeos-qwen3.5-4b'), INCAPABLE)).toBe(
      'zeos-qwen3.5-4b',
    );
  });

  it('uses the capability default when unset', () => {
    expect(resolveActiveLocalModelIdOrDefault(cfg(undefined), CAPABLE)).toBe('zeos-qwen3.5-4b');
    expect(resolveActiveLocalModelIdOrDefault(cfg(undefined), INCAPABLE)).toBe('gemma-4-e2b');
  });

  it('uses the capability default for a stale custom id', () => {
    expect(resolveActiveLocalModelIdOrDefault(cfg('custom:not-registered'), CAPABLE)).toBe(
      'zeos-qwen3.5-4b',
    );
    expect(resolveActiveLocalModelIdOrDefault(cfg('custom:not-registered'), INCAPABLE)).toBe(
      'gemma-4-e2b',
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
