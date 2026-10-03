import { isBrowser } from '../browser';

export interface WebGpuStatus {
  supported: boolean;
  reason?: string;
  /**
   * The adapter's `maxBufferSize` limit in bytes, when an adapter was
   * obtained. ONNX Runtime's WebGPU backend uploads each weight tensor as its
   * own GPU buffer, so the largest Gemma tensor must fit under this limit;
   * browsers that cap it low (Firefox currently pins it at 1 GiB) cannot hold
   * the model. Undefined when no adapter was available.
   */
  maxBufferSize?: number;
  /**
   * The adapter's `maxStorageBufferBindingSize` limit in bytes. Weight
   * tensors are bound as storage buffers, so this caps the largest tensor
   * independently of `maxBufferSize`. Undefined when no adapter was available.
   */
  maxStorageBufferBindingSize?: number;
  /**
   * Whether the adapter exposes `shader-f16`. The q4f16 Gemma weights run
   * fastest (and on some ORT kernels, only) with native f16 shaders.
   */
  f16?: boolean;
}

let cached: WebGpuStatus | null = null;
let inflight: Promise<WebGpuStatus> | null = null;

export function isWebGpuApiPresent(): boolean {
  if (!isBrowser()) return false;
  return typeof (navigator as unknown as { gpu?: unknown }).gpu !== 'undefined';
}

export async function detectWebGpu(): Promise<WebGpuStatus> {
  if (cached) return cached;
  if (inflight) return inflight;

  inflight = (async (): Promise<WebGpuStatus> => {
    if (!isBrowser()) {
      return { supported: false, reason: 'WebGPU is unavailable outside the browser.' };
    }
    const gpu = (navigator as unknown as { gpu?: { requestAdapter: (opts?: unknown) => Promise<unknown> } }).gpu;
    if (!gpu) {
      return {
        supported: false,
        reason: 'WebGPU is not exposed by this browser. Use a recent Chrome or Edge.',
      };
    }
    try {
      const adapter = (await gpu.requestAdapter({ powerPreference: 'high-performance' })) as
        | {
            limits?: { maxBufferSize?: number; maxStorageBufferBindingSize?: number };
            features?: { has(name: string): boolean };
          }
        | null;
      if (!adapter) {
        return {
          supported: false,
          reason: 'WebGPU adapter request returned null. No usable GPU was found.',
        };
      }
      const maxBufferSize =
        typeof adapter.limits?.maxBufferSize === 'number'
          ? adapter.limits.maxBufferSize
          : undefined;
      const maxStorageBufferBindingSize =
        typeof adapter.limits?.maxStorageBufferBindingSize === 'number'
          ? adapter.limits.maxStorageBufferBindingSize
          : undefined;
      const f16 = adapter.features?.has('shader-f16') ?? false;
      return { supported: true, maxBufferSize, maxStorageBufferBindingSize, f16 };
    } catch (err) {
      return {
        supported: false,
        reason: `WebGPU adapter request failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  })();

  try {
    cached = await inflight;
    return cached;
  } finally {
    inflight = null;
  }
}
