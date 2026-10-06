/**
 * DEV-ONLY scratch benchmark (not imported by the app). Drive from DevTools:
 *
 *   const b = await import('/src/dev/llmBench.ts');
 *   await b.load('qwen');            // or 'gemma'
 *   await b.prefill([128, 512, 2048]);
 *   await b.decode(128, 64);
 *   await b.profile(256);            // per-kernel GPU timings, if supported
 *   await b.unload();
 *
 * Runs on the main thread (the page freezes while it runs).
 */
import {
  env,
  AutoTokenizer,
  Gemma4ForCausalLM,
  Qwen3_5ForCausalLM,
  TextStreamer,
  DynamicCache,
  ones,
  Tensor,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from '@huggingface/transformers';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';

type Family = 'gemma' | 'qwen';
const REPOS: Record<Family, string> = {
  gemma: 'onnx-community/gemma-4-E4B-it-ONNX',
  qwen: 'onnx-community/Qwen3.5-4B-ONNX-OPT',
};

if (env.backends.onnx.wasm) env.backends.onnx.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl };
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = '/models/';
env.useBrowserCache = false;

let model: PreTrainedModel | null = null;
let tok: PreTrainedTokenizer | null = null;
let fam: Family = 'gemma';

/** Captured ORT console lines during session creation (node placement). */
export const ortLog: string[] = [];
let traceSink: string[] | null = null;

function dropNullPixelValues(mdl: PreTrainedModel): void {
  const orig = mdl.prepare_inputs_for_generation.bind(mdl);
  mdl.prepare_inputs_for_generation = (...args: Parameters<typeof orig>) => {
    const inputs = orig(...args);
    if (inputs && inputs.pixel_values == null) delete inputs.pixel_values;
    return inputs;
  };
}

export async function load(f: Family, { verbose = true } = {}) {
  await unload();
  fam = f;
  const ortEnv = env.backends.onnx as any;
  const prevLevel = ortEnv.logLevel;
  const origWarn = console.warn, origLog = console.log, origErr = console.error;
  const tap = (orig: (...a: unknown[]) => void) => (...a: unknown[]) => {
    const s = a.map(String).join(' ');
    if (traceSink) { traceSink.push(s); return; }
    if (/ExecutionProvider|placed on|assigned to/i.test(s)) ortLog.push(s);
    orig(...a);
  };
  if (verbose) {
    ortEnv.logLevel = 'verbose';
    console.warn = tap(origWarn); console.log = tap(origLog); console.error = tap(origErr);
  }
  const t0 = performance.now();
  const Cls = f === 'qwen' ? Qwen3_5ForCausalLM : Gemma4ForCausalLM;
  try {
    [tok, model] = await Promise.all([
      AutoTokenizer.from_pretrained(REPOS[f]),
      Cls.from_pretrained(REPOS[f], {
        dtype: 'q4f16',
        device: 'webgpu',
        session_options: verbose ? { logSeverityLevel: 0, logVerbosityLevel: 0 } : {},
      } as any),
    ]);
  } finally {
    ortEnv.logLevel = prevLevel;
    console.warn = origWarn; console.log = origLog; console.error = origErr;
  }
  if (f === 'qwen') dropNullPixelValues(model!);
  return { loadMs: Math.round(performance.now() - t0), placementLines: ortLog.length };
}

export async function unload() {
  if (model) await model.dispose();
  model = null;
  tok = null;
}

/** `n` real-text token ids (a repeated paragraph), no special tokens. */
function ids(n: number): Tensor {
  const para =
    'The Titanic dataset lists passengers with their age, sex, ticket class, fare, ' +
    'cabin and port of embarkation, and whether they survived the sinking in 1912. ' +
    'Analysts often group passengers by age band to compare survival rates. ';
  const one = tok!.encode(para, { add_special_tokens: false });
  const out: bigint[] = [];
  while (out.length < n) for (const t of one) if (out.length < n) out.push(BigInt(t));
  return new Tensor('int64', BigInt64Array.from(out), [1, n]);
}

class Stamper extends TextStreamer {
  stamps: number[] = [];
  constructor(t: PreTrainedTokenizer) {
    super(t, { skip_prompt: true });
  }
  put(value: bigint[][]) {
    this.stamps.push(performance.now());
    // swallow text
    void value;
  }
  end() {}
}

const gpuSync = async () => {
  // Make sure no queued GPU work from a previous run bleeds into timings.
  await new Promise((r) => setTimeout(r, 50));
};

/**
 * Prompt-processing time for each length: one `generate(max_new_tokens=1)`
 * (that is exactly one forward over the whole prompt). Modes:
 *  - plain: no cache passed (upstream's normal path)
 *  - cache: pass a fresh DynamicCache (what our worker does)
 */
export async function prefill(lengths = [128, 512, 2048], { modes = ['plain', 'cache'], warmup = true } = {}) {
  const rows: Record<string, unknown>[] = [];
  if (warmup) await model!.generate({ input_ids: ids(16), attention_mask: ones([1, 16]), max_new_tokens: 1, do_sample: false } as any);
  for (const n of lengths) {
    for (const mode of modes) {
      const input_ids = ids(n);
      const kv = mode === 'cache' ? new DynamicCache() : undefined;
      await gpuSync();
      const t0 = performance.now();
      await model!.generate({
        input_ids,
        attention_mask: ones([1, n]),
        ...(kv ? { past_key_values: kv } : {}),
        max_new_tokens: 1,
        do_sample: false,
      } as any);
      const ms = performance.now() - t0;
      if (kv) await (kv as any).dispose?.();
      rows.push({ family: fam, mode, tokens: n, ms: Math.round(ms), tokPerS: +(n / (ms / 1000)).toFixed(1) });
      console.log('[bench] prefill', JSON.stringify(rows.at(-1)));
    }
  }
  return rows;
}

/** Decode timing: per-token gaps after a prompt of `promptLen` tokens. */
export async function decode(promptLen = 128, newTokens = 64) {
  const st = new Stamper(tok!);
  const t0 = performance.now();
  await model!.generate({
    input_ids: ids(promptLen),
    attention_mask: ones([1, promptLen]),
    max_new_tokens: newTokens,
    min_new_tokens: newTokens,
    do_sample: false,
    streamer: st,
  } as any);
  // stamps[0] is the prompt put (skip_prompt still calls put), then 1 per token.
  const s = st.stamps;
  const firstTok = s[1] - t0;
  const gaps = s.slice(2).map((x, i) => x - s[i + 1]);
  gaps.sort((a, b) => a - b);
  const med = gaps[Math.floor(gaps.length / 2)];
  const mean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
  const row = {
    family: fam,
    promptLen,
    newTokens: s.length - 1,
    firstTokenMs: Math.round(firstTok),
    decodeMedianMs: +med.toFixed(1),
    decodeMeanMs: +mean.toFixed(1),
    p90Ms: +gaps[Math.floor(gaps.length * 0.9)].toFixed(1),
    tokPerS: +(1000 / mean).toFixed(1),
  };
  console.log('[bench] decode', JSON.stringify(row));
  return row;
}

/**
 * Per-kernel GPU timing for one prompt forward of `n` tokens (and one decode
 * step), aggregated by kernel type. Needs the adapter's `timestamp-query`.
 */
export async function profile(n = 256, { decodeSteps = 4 } = {}) {
  const ortEnv = env.backends.onnx as any;
  const events: { kernelType: string; kernelName: string; programName: string; ms: number }[] = [];
  ortEnv.webgpu = ortEnv.webgpu ?? {};
  ortEnv.webgpu.profiling = {
    mode: 'default',
    ondata: (d: any) => {
      events.push({
        kernelType: d.kernelType,
        kernelName: d.kernelName,
        programName: d.programName,
        ms: Number(d.endTime - d.startTime) / 1e6,
      });
    },
  };
  const agg = (label: string, wallMs: number) => {
    const by = new Map<string, { n: number; ms: number }>();
    for (const e of events) {
      const k = e.kernelType;
      const v = by.get(k) ?? { n: 0, ms: 0 };
      v.n++; v.ms += e.ms; by.set(k, v);
    }
    const gpuMs = [...by.values()].reduce((a, b) => a + b.ms, 0);
    const table = [...by.entries()]
      .sort((a, b) => b[1].ms - a[1].ms)
      .map(([k, v]) => ({ kernel: k, count: v.n, gpuMs: +v.ms.toFixed(2), pct: +((100 * v.ms) / Math.max(gpuMs, 1e-9)).toFixed(1) }));
    return { label, family: fam, wallMs: Math.round(wallMs), gpuMs: +gpuMs.toFixed(1), kernels: events.length, table };
  };
  try {
    events.length = 0;
    let t0 = performance.now();
    await model!.generate({ input_ids: ids(n), attention_mask: ones([1, n]), max_new_tokens: 1, do_sample: false } as any);
    const pre = agg(`prefill ${n}`, performance.now() - t0);
    events.length = 0;
    t0 = performance.now();
    await model!.generate({
      input_ids: ids(16), attention_mask: ones([1, 16]),
      max_new_tokens: 1 + decodeSteps, min_new_tokens: 1 + decodeSteps, do_sample: false,
    } as any);
    const dec = agg(`prefill16 + ${decodeSteps} decode`, performance.now() - t0);
    return { pre, dec };
  } finally {
    ortEnv.webgpu.profiling = { mode: 'off' };
  }
}

/**
 * Capture ORT's verbose run log (needs `load(f, { verbose: true })`, which
 * sets session logSeverityLevel 0) for one generate call. Logging inflates
 * absolute times; use the counts and relative shares.
 */
export async function trace(n = 32, newTokens = 1) {
  const lines: string[] = [];
  const origs = { log: console.log, warn: console.warn, error: console.error, info: console.info };
  const tap = (...a: unknown[]) => { lines.push(a.map(String).join(' ')); };
  console.log = console.warn = console.error = console.info = tap as any;
  traceSink = lines; // ORT's emscripten print hooks were bound to load()'s tap
  const t0 = performance.now();
  try {
    await model!.generate({
      input_ids: ids(n), attention_mask: ones([1, n]),
      max_new_tokens: newTokens, min_new_tokens: newTokens, do_sample: false,
    } as any);
  } finally {
    Object.assign(console, origs);
    traceSink = null;
  }
  (globalThis as any).__traceLines = lines;
  return { wallMs: Math.round(performance.now() - t0), lines: lines.length };
}
