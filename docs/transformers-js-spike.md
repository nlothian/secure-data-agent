# transformers.js v4.3.0 feasibility spike (Phase 0)

Date: 2026-10-04. Branch `transformers-js`.
Machine: Apple M1 Max, 32 GB, macOS. Browser: the Claude desktop's embedded
Chromium pane (UA `Chrome/152.0.7977.130`). WebGPU adapter `apple / metal-3`.
Stack: `@huggingface/transformers@4.3.0`, `onnxruntime-web@1.31.0-dev.20260914-8d85527a0`
(WebGPU EP), models `onnx-community/gemma-4-E2B-it-ONNX` and `…E4B-it-ONNX`,
`dtype: 'q4f16'`, text-only (`Gemma4ForCausalLM`), served from the repo `models/`
folder via `PUBLIC_LOCAL_MODELS=1`. Everything ran in a dedicated module Web
Worker. Decoding was greedy (`do_sample: false`) unless stated otherwise.

The harness was a temporary `site/src/pages/spike.astro` page plus
`site/src/workers/spike.worker.ts`. Both were deleted after the run.

## Summary

| # | Item | Result | Verdict |
|---|------|--------|---------|
| 1 | Load E2B text-only in a worker | 3.8–5.0 s from the local dev server (tokenizer about 0.46 s, model 3.2–4.7 s). Only `embed_tokens` and `decoder_model_merged` sessions are created; vision and audio are skipped. `eos_token_id = [1,106,50]`. **The tokenizer does NOT add `<bos>`.** | **GO** |
| 2 | Prefill memory / full logits / 20k window | `generate()` keeps only the last logit (the decoder has a `num_logits_to_keep` input). A direct `forward()` without it returns all logits: `[1,540,262144]` fp16, 283 MB. **A single forward of ≥ about 16.4k tokens fails** with an ORT `SafeIntOnOverflow() Integer overflow`, and **that failure poisons the whole worker**. Chunked prefill (2048-token chunks, shared `DynamicCache`) works at 20k, with TTFT 103 s. | **GO, but chunked prefill is required** |
| 3 | `shader-f16` / limits | `shader-f16` = true. `maxBufferSize` = `maxStorageBufferBindingSize` = 4,294,967,292 (4 GiB − 4). No WebGPU errors during load. | **GO** |
| 4 | Streaming fidelity + interrupt | `<|tool_call>…<|"|>…<tool_call|>` streams verbatim, one `put` per control token. The EOS token's text (`<turn|>` or `<|tool_response>`) **does** reach `put()` and the text callback. `clean_up_tokenization_spaces:false` is honoured; the default is `true`. `interrupt()` returns from `generate` in 2–12 ms; `reset()` works. | **GO** |
| 5 | Cache API | In this browser, `Cache.put` **rejects any body ≥ 256 MiB** (`UnknownError … Unexpected internal error`). Both 1.5 GB `_data` files were silently not cached; transformers.js only logs a warning. Small files are cached under absolute request URLs. OPFS accepted a 1.6 GB file. | **NO-GO for the default browser cache.** Use OPFS through `env.customCache`. |
| 6 | Throughput (E2B) | Decode 31 tok/s at about 100 ctx, 20.6 tok/s at 2k, 9.0 tok/s at 10k, 5.1 tok/s at 20k. TTFT 0.29 s / 5.2 s / 34 s / 103 s. | **Partial.** Usable to about 8k; slow beyond that. |
| 7 | KV reuse (`past_key_values` + longer `input_ids`) | `decoder_prepare_inputs_for_generation` slices `input_ids` to the unseen suffix, and this was verified at runtime by the chunked prefill. `DynamicCache` has **no crop/truncate**. | **GO** for prefix-extension reuse; rollback needs our own code |
| (2b) | E4B (downloaded after all) | Loads in 12.0 s. Decode 16.6 tok/s at 100 ctx, 12.5 tok/s at 2k, 5.2 tok/s at 8k, 2.1 tok/s at 20k. TTFT 43.7 s at 8k and 188 s at 20k. | **Partial.** Viable for short contexts only. |

## 1. Load E2B text-only in a worker

```js
env.allowLocalModels = true; env.allowRemoteModels = false; env.localModelPath = '/models/';
AutoTokenizer.from_pretrained(id, { progress_callback });
Gemma4ForCausalLM.from_pretrained(id, { dtype: 'q4f16', device: 'webgpu', progress_callback });
```

Load times (local dev server, files warm in the OS page cache):

| run | tokenizer | model | total |
|-----|-----------|-------|-------|
| first | 461 ms | 3,641 ms | 4,102 ms |
| reload (new page) | – | 3,215 ms | 3,847 ms |
| with `useBrowserCache: true` (also writing the cache) | – | 4,693 ms | 5,340 ms |

Fetching 3.11 GB from the dev server took about 2.2 s. Session creation took about 1 s.

**Session keys:** `["embed_tokens", "decoder_model_merged"]`. `Gemma4ForCausalLM`
on a checkpoint whose `config.architectures[0]` is `Gemma4ForConditionalGeneration`
is detected as "text-only" (`modeling_utils.js:184-198`), and the session config
then drops the encoders (`models/session_config.js:77-86`):

```js
text_only_sessions: { embed_tokens: 'embed_tokens', decoder_model_merged: 'decoder_model_merged' },
sessions: (config, options, textOnly) => {
    const s = { ...MODEL_SESSION_CONFIG[MODEL_TYPES.ImageAudioTextToText].text_only_sessions };
    if (!textOnly) {
        s['audio_encoder'] = 'audio_encoder';
        s['vision_encoder'] = 'vision_encoder';
```

**Session I/O (E2B):**
- `embed_tokens`: in `input_ids [batch, seq]`. Out `inputs_embeds`, `per_layer_inputs`.
- `decoder_model_merged` inputs: `inputs_embeds [b,s,1536] f32`, `attention_mask [b,total_seq] i64`,
  `position_ids`, `num_logits_to_keep [] i64`, `per_layer_inputs [b,s,35,256] f32`, plus 30
  `past_key_values.{0..14}.{key,value}` fp16 `[b,1,past,256]` (6 of them are `[b,1,past,512]`,
  the global-attention layers).
- `decoder_model_merged` outputs: `logits` plus 30 `present.*`.
- E4B: 48 past inputs (24 KV layers), `[b,2,past,256|512]`, and `per_layer_inputs [b,s,42,256]`.

**`model.generation_config`:** `{ bos_token_id: 2, eos_token_id: [1,106,50], pad_token_id: 0,
do_sample: true, temperature: 1, top_k: 64, top_p: 0.95 }`. The defaults sample, so
pass `do_sample` explicitly. 50 is `<|tool_response>` and 106 is `<turn|>`.

**Progress events (local mode):** every file goes through `initiate → download →
progress* → done`. **`download` is emitted in local mode too.** Files seen:
`tokenizer.json, tokenizer_config.json, config.json, onnx/embed_tokens_q4f16.onnx(+_data),
onnx/decoder_model_merged_q4f16.onnx(+_data), generation_config.json`. `chat_template.jinja`
is not fetched. For E4B, `onnx/decoder_model_merged_q4f16.onnx_data_1` is fetched
automatically (`use_external_data_format: 2` in its `config.json`).

There is also a **new aggregate event `status: 'progress_total'`** (no `file`) with
`{progress, loaded, total, files:{…}}`. It fired **42,613 times** in one 4 s load, and the
per-file `progress` events about 21k times per `_data` file. The engine must throttle
before it posts these to the main thread.

**`<bos>` is not added by the tokenizer.** `tokenizer(text, { add_special_tokens: true })`
produced `[105 (<|turn>), …]`. The `tokenizer.json` post-processor is an empty
`TemplateProcessing` (`"special_tokens": {}`), and `chat_template.jinja:177` emits
`{{- bos_token -}}` itself. The engine must prepend `<bos>` (id 2) itself.
`'<bos>' + prompt` encodes to `[2, 105, …]`. The existing
`gemmaTokenizerCheck.ts` assertion "encode('x') starts with `<bos>`" would fail
against this tokenizer.

`performance.memory` is `null` inside the worker. On the page (main thread) the JS heap
stayed at about 25 MB. Note that transformers.js reads every external-data file fully into
a JS `Uint8Array` before it creates the session (`session_options.externalData`), so the
worker briefly holds about 3.1 GB (E2B) or 4.9 GB (E4B) of ArrayBuffers during load.

## 2. Prefill memory / full logits / chunking

### Full logits?

`generate()` never materialises full logits. `decoder_prepare_inputs_for_generation`
sets `num_logits_to_keep = 1` (`modeling_utils.js:1603`), and the decoder declares that
input. A **direct `model.forward()`** falls back to 0, which means all positions
(`modeling_utils.js:1375-1376`):

```js
// Fallback for non-generation forward calls (e.g. prefill scoring): compute all logits.
setNumLogitsToKeep(self, new_model_inputs, 0n);
```

Measured on E2B with 540 prompt tokens:

| call | logits dims | size | time |
|------|-------------|------|------|
| `forward({input_ids, attention_mask})` | `[1,540,262144]` fp16 | 283 MB (CPU) | 1,691 ms (first run, includes shader compile) |
| `forward({…, num_logits_to_keep: 1})` | `[1,1,262144]` | 0.5 MB | 986 / 947 ms |

So at 20k, full logits would be about 10.5 GB. **Any direct `forward()` must pass
`num_logits_to_keep`.**

### KV cache

All 30 `present.*` tensors are full length (the 512-token sliding window is not
reflected in cache size): 540 tokens = 10.0 MB, which is about 18.5 KB/token on E2B.
At 19,987 positions there were 24 × `[1,1,19987,256]` + 6 × `[1,1,19987,512]` fp16, about 368 MB.
E4B is about 57 KB/token (40 × `[1,2,L,256]` + 8 × `[1,2,L,512]`): about 470 MB at 8k and about 1.15 GB at 20k.

### Monolithic prefill (`generate`, `max_new_tokens: 1`, E2B)

| prompt tokens | TTFT | result |
|---------------|------|--------|
| 540 | 0.95 s | ok |
| 2,052 | 4.5 s | ok |
| 8,156 | 37.3 s | ok |
| 12,020 | 70.8 s | ok |
| 16,220 | 145.0 s | ok |
| 16,500 | fails after 10.7 s | `OrtRun() … safeint.h:17 SafeIntOnOverflow() Integer overflow` |
| 19,244 | fails after 17 s | same |

The boundary matches `8 heads × N² × 2 bytes = 2³²`, which gives N = 16,384. The attention
op materialises an fp16 `[heads, q_len, kv_len]` score buffer, and its byte size overflows
a 32-bit size. It is **not** an out-of-memory error.

**After the overflow the ORT runtime in that worker is permanently broken.** Even a
92-token generate then fails with the same error. `model.dispose()` followed by
`Gemma4ForCausalLM.from_pretrained(...)` in the **same worker** still fails. Only a new
worker (a page reload in the spike) recovers. The engine must either enforce the size bound
or terminate and respawn the worker on this error.

### Chunked prefill (public API only)

Repeated `model.generate({ input_ids: ids[:, :end], attention_mask: ones(end), past_key_values: cache, max_new_tokens: 1 })`
with one shared `new DynamicCache()`, then a final `generate` with the full `input_ids` and
the same cache. **You must pass `attention_mask` with length = full prefix.** If you leave it
out, `prepare_inputs` builds `ones(past + input_len)`, which is too long (`modeling_utils.js:1605-1617`).

| model | prompt | chunks (2048 each, ms) | TTFT | decode |
|-------|--------|------------------------|------|--------|
| E2B | 8,156 | 4,444 / 5,646 / 6,916 + final | **25.5 s** (vs 37.3 s monolithic) | 10.1 tok/s |
| E2B | 10,014 | 4,406 / 5,626 / 6,914 / 8,245 + final | 34.3 s | 9.0 tok/s |
| E2B | 19,972 | 4,472 / 5,640 / 6,921 / 8,245 / 9,563 / 11,437 / 12,865 / 14,576 / 15,594 + final | **103.2 s** | 5.1 tok/s |
| E4B | 8,166 | 8,122 / 9,848 / 11,637 + final | 43.7 s | 5.2 tok/s |

The output was coherent in every run. The cache length after the run was prompt + generated − 1
(for example 8,171 for 8,156 + 16), because the last sampled token is never fed back.

### KV-prefix reuse (item 7)

`models/modeling_utils.js:1600-1645` (Gemma 4 uses this through
`multimodal_text_to_text_prepare_inputs_for_generation`, `:1660-1665`):

```js
const past_length = model_inputs.past_key_values ? model_inputs.past_key_values.get_seq_length() : 0;
setNumLogitsToKeep(self, model_inputs, 1n);
...
if (model_inputs.past_key_values) {
    const { input_ids, attention_mask } = model_inputs;
    if (attention_mask && attention_mask.dims[1] > input_ids.dims[1]) {
        // NOTE: not needed since we only pass the generated tokens to the next forward pass
    }
    // 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens.
    // We can discard input_ids based on the past_length.
    else if (past_length < input_ids.dims[1]) {
        model_inputs.input_ids = input_ids.slice(null, [past_length, null]);
    }
```

So **yes**, `input_ids` is sliced to the unseen suffix. It does not check that the cached
prefix actually matches the new `input_ids`; that check is our job. The cache is kept alive
after `generate` only if you passed it (`modeling_utils.js:1057`):

```js
const keepCacheAlive = 'past_key_values' in kwargs || generation_config.return_dict_in_generate;
```

`DynamicCache` (`src/cache_utils.js:14-82`) exposes only `get_seq_length()`, `update(entries)`
and `dispose()`. There is **no crop or truncate**. `get_seq_length()` reads
`dims.at(-2)` of the first `past_key_values.*` entry. Rolling back, for example to drop a
generated turn, means slicing each `[b,h,L,d]` tensor ourselves. Those tensors are
`gpu-buffer` tensors, so this needs a GPU-side slice or a download, re-upload, or re-prefill.
`DynamicCache`, `InterruptableStoppingCriteria` and `TextStreamer` are all exported from the
package root.

## 3. `shader-f16` / limits

```
adapter.features.has('shader-f16')     true
adapter.limits.maxBufferSize           4294967292
adapter.limits.maxStorageBufferBindingSize 4294967292
adapter.limits.maxComputeWorkgroupStorageSize 32768
adapter.info                           { vendor: 'apple', architecture: 'metal-3' }
features also include: subgroups, subgroup-size-control, timestamp-query
```

There were no WebGPU validation errors during the load of either model. The largest
known single buffers are the 1.52 GB decoder and 1.59 GB embed external-data blobs for E2B,
and 2.07 GB / 2.02 GB for E4B. Both are under 4 GiB. The binding constraint we actually hit
is ORT's 32-bit size computation for the attention scores (§2), not the adapter limit.

## 4. Streaming fidelity

Streamer: `class LoggingStreamer extends TextStreamer` with `{ skip_prompt: true,
skip_special_tokens: false, decode_kwargs: { skip_special_tokens: false,
clean_up_tokenization_spaces: false }, callback_function, token_callback_function }`.

**Tool call (E2B, prompt from `renderConversationForGemma` + `<bos>`, 102 tokens,
`get_weather{location}`).** Raw streamed text, verbatim:

```
<|tool_call>call:get_weather{location:<|"|>Adelaide<|"|>}<tool_call|><|tool_response>
```

Every `put` (id = decoded text): `48=<|tool_call>`, `6639=call`, `236787=:`, `828=get`,
`236779=_`, `19323=weather`, `236782={`, `7125=location`, `236787=:`, `52=<|"|>`,
`4036=Ad`, `46007=elaide`, `52=<|"|>`, `236783=}`, `49=<tool_call|>`, `50=<|tool_response>`.
TTFT was 585 ms. The model stopped on **id 50 `<|tool_response>`**, which is part of
`eos_token_id`. That EOS token **was** passed to `put()` and emitted through
`callback_function`, so the engine must strip a trailing `<|tool_response>` / `<turn|>`.

**Plain answer:** "What is 2+2?" ended with puts `101=<channel|>`, `236812=4`,
`106=<turn|>`. So `<turn|>` also reaches `put()` and the text callback. The prompt is
`put` first (`modeling_utils.js:952-954`), and each sampled token is `put` **before** the
stopping criteria run (`:1021-1027`). That is why EOS is always streamed.

**Control-token path in `TextStreamer.put`** (`generation/streamers.js:118`): a lone
special id flushes the cache and is decoded and emitted immediately. With
`skip_special_tokens: false`, every Gemma control token arrives as its own callback chunk.
Note that `this.decode_kwargs = { skip_special_tokens, ...decode_kwargs, ...kwargs }` (`:88`),
so the top-level `skip_special_tokens` option and `decode_kwargs.skip_special_tokens` are merged.

**"Next tokens are prompt" field:** `next_tokens_are_prompt` (`streamers.js:93`, `:108-110`,
reset to `true` in `end()` at `:172`). It is not `skip_prompt_tokens` or `is_prompt`.

**`clean_up_tokenization_spaces`:** for the ids of `"x = 1 . y , z ! w ? it 's don 't"`:
- `{clean_up_tokenization_spaces:false}` gives `" = 1 . y , z ! w ? it 's don 't"`. This is honoured.
- `{clean_up_tokenization_spaces:true}` and the **default** give `" = 1. y, z! w? it's don 't"`. The default cleans up, so the flag must be passed.

**Interrupt (`InterruptableStoppingCriteria`, passed as `stopping_criteria: [stopper]`):**

| case | interrupt at | `generate` returned after | tokens |
|------|--------------|---------------------------|--------|
| 300 ms into a 400-token essay | 301 ms | **2 ms** | 4 |
| 1,500 ms | 1,500 ms | 12 ms | 43 |
| next call after `reset()` | – | normal (20 tokens) | 20 |

The criteria is checked once per decode step, so **`interrupt()` cannot abort a prefill**.
A 100 s chunked prefill can only be cancelled between chunks, and the engine should check a
flag between chunks.

**Model behaviour (not a transformers.js issue):** with greedy decoding and the
empty-thought prefix, both E2B and E4B often narrate reasoning ("The user wants…", "Here's
a thinking process…") in the answer channel. Sometimes they also emit a stray `<channel|>`
before the real answer or tool call. E4B's tool call came after about 60 tokens of
reasoning and a `<channel|>`. The parser should tolerate this. Sampling with the default
`generation_config` may behave differently; this was not measured.

## 5. Cache API behaviour

Run: fresh page, `env.useBrowserCache = true`, load E2B, then `caches.open('transformers-cache')`.

Console, twice (once per `_data` file):
```
Unable to add response to browser cache: UnknownError: Failed to execute 'put' on 'Cache': Unexpected internal error..
```
The load still succeeded. The error is swallowed (`utils/hub.js:242-246`, "Do not crash if
unable to add to cache (e.g., QuotaExceededError)").

Stored keys (all `cache.match` non-null, blob sizes equal content-length):
```
http://localhost:4330/models/onnx-community/gemma-4-E2B-it-ONNX/tokenizer_config.json   18,807
http://localhost:4330/models/onnx-community/gemma-4-E2B-it-ONNX/tokenizer.json          19,439,251
http://localhost:4330/models/onnx-community/gemma-4-E2B-it-ONNX/config.json             5,549
http://localhost:4330/models/onnx-community/gemma-4-E2B-it-ONNX/onnx/embed_tokens_q4f16.onnx           5,621
http://localhost:4330/models/onnx-community/gemma-4-E2B-it-ONNX/onnx/decoder_model_merged_q4f16.onnx   673,231
http://localhost:4330/models/onnx-community/gemma-4-E2B-it-ONNX/generation_config.json  238
https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/ort-wasm-simd-threaded.asyncify.mjs   53,057
https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/ort-wasm-simd-threaded.asyncify.wasm  26,861,777
```
Both `_data` files are **missing**.

The local-mode key is the local path (`cacheKey = localPath`, `hub.js:305`), which the
Cache API resolves to an absolute URL. The remote key is the HF resolve URL
(`utils/hub.js:139-155`):

```js
const remoteURL = pathJoin(
    env.remoteHost,
    env.remotePathTemplate
        .replaceAll('{model}', path_or_repo_id)
        .replaceAll('{revision}', encodeURIComponent(revision)),
    filename,
);
const proposedCacheKey =
    cache instanceof FileCache ? (revision === 'main' ? requestURL : pathJoin(path_or_repo_id, revision, filename))
        : remoteURL;
```
with `env.js:256-257` `remoteHost: 'https://huggingface.co/'`,
`remotePathTemplate: '{model}/resolve/{revision}/'`. So the remote key is
`https://huggingface.co/<repo>/resolve/main/<file>`. The cache name is `env.cacheKey`
(`'transformers-cache'`, `env.js:273`).

**Size-limit isolation** (main thread, scratch cache, deleted afterwards):

| body | result |
|------|--------|
| `new Response(Uint8Array)` 32 / 64 / 100 / 128 / 192 / 224 / 240 / 255 MiB, 256 MiB − 1 B | ok |
| exactly 268,435,456 B (256 MiB), 256 MiB + 1 B, 512 MiB, 1 GiB, 1.1 GiB | **UnknownError** |
| streamed `ReadableStream` body, 1.1 GiB and 1.6 GiB | **UnknownError** |
| `Blob` body 1.6 GiB | **UnknownError** |
| **OPFS** `createWritable()` 256 MiB / 1.6 GiB | ok (387 ms / 3,054 ms) |

`navigator.storage.estimate()`: quota about 5.9 GB, `persisted() = false`.

**Caveat:** this was measured in the Claude desktop's embedded Chromium 152 pane, not in
stock Chrome. A hard 2²⁸-byte per-entry cap may be specific to this embedder, so it needs
re-checking in stock Chrome before we rely on it either way. The safe design is the same in
both cases: route large files through OPFS (`env.useCustomCache = true; env.customCache =
{ match, put }` backed by OPFS, or keep the existing OPFS layer and hand
transformers.js `ArrayBuffer`s). The ORT wasm comes from jsdelivr at runtime
(`backends/onnx.js:345`) unless `env.backends.onnx.wasm.wasmPaths` is set. For offline use,
or a strict CSP, self-host it.

## 6. Throughput (E2B, greedy, 128 new tokens unless noted)

TTFT is measured from the `generate()` call to the first streamed token. tok/s is
(tokens − 1) divided by the time from the first to the last token.

| context | prefill | TTFT | decode tok/s |
|---------|---------|------|--------------|
| 102 | monolithic | 288 ms | **31.1** |
| 2,062 | monolithic | 5,195 ms | **20.6** |
| 8,156 | chunked 2048 (16 new tokens) | 25.5 s | 10.1 |
| 10,014 | chunked 2048 | 34.3 s | **9.0** |
| 19,972 | chunked 2048 (16 new tokens) | 103.2 s | 5.1 |

Prefill throughput is about 460 tok/s for the first 2k chunk and falls as the past grows
(each 2048-token chunk costs about 1.3 s more than the one before). Decode slows roughly in
proportion to context. Every layer attends over the full-length cache, including the
sliding-window layers, which have full-length KV in this export.

There are no MediaPipe tok/s figures anywhere in `docs/` to compare against, so this is
reported as-is. The app already records `tps` per turn (`streamChat.ts`
`TokenUsageReport.tps`), which is where a side-by-side comparison should come from.

### 2b. E4B (downloaded with `npm run models:fetch -- e4b`, about 8 min)

| measure | value |
|---------|-------|
| load (local) | 11,977 ms (model 10,932 ms) |
| 102 ctx | TTFT 2,860 ms (first generate, includes shader compile), **16.6 tok/s** |
| 2,062 ctx | TTFT 9,801 ms, **12.5 tok/s** |
| 8,166 ctx (chunked) | TTFT 43.7 s, **5.2 tok/s** |
| 19,972 ctx (chunked, 32 new tokens) | chunks 8,165 / 9,826 / 11,614 / 13,459 / 16,025 / 19,193 / 21,305 / 26,281 / 33,526 ms + final. **TTFT 187.9 s, decode 2.06 tok/s.** The late chunks grow superlinearly (likely memory pressure). |
| tool call | correct `<|tool_call>call:get_weather{location:<|"|>…` after reasoning text plus `<channel|>` |

## Implications for the engine

**(a) Is chunked prefill required?** **Yes.** One reason is correctness: a single decoder
forward with `8 × q_len × kv_len × 2 B ≥ 2³²` crashes ORT and poisons the worker.
Monolithic prefill therefore breaks at about 16.4k tokens on both E2B and E4B (both have 8
query heads). The other reason is speed: chunked 8k is 25.5 s versus 37.3 s, and
monolithic 16k takes 145 s. Use chunks of ≤ 2048 tokens. A chunk C against total length L
is safe while `C × L < 2.68 × 10⁸`, so 2048-token chunks are safe up to about 131k total and
1024-token chunks give more headroom. Implement it as repeated
`generate({ input_ids: prefix, attention_mask: ones(prefix_len), past_key_values: cache, max_new_tokens: 1 })`,
or as a direct `forward` loop that always passes `num_logits_to_keep: 1`. Check a cancel
flag between chunks. Also treat any `OrtRun()` error as fatal to the worker: terminate it,
respawn it, and reload the model.

**(b) Threshold for the GPU buffer warning banner.** The adapter's `maxBufferSize` (4 GiB −
4 here) is not the limit we hit. The practical guard is the attention-score size above,
which chunking handles. For the banner, warn when `maxBufferSize` or
`maxStorageBufferBindingSize` is less than about 2.1 GB (the largest single external-data
blob is 2.07 GB, E4B decoder `_data`; E2B's largest is 1.59 GB), or when `shader-f16` is
missing, since q4f16 assumes it. Separately, warn when the selected context window is
above about 8k on E2B or 2k on E4B, because TTFT exceeds about 25–45 s there.

**(c) Is E4B viable?** **Only for short contexts.** It loads fine (12 s locally, 4.9 GB) and
calls tools correctly. Decode is 16.6 tok/s at about 100 ctx and 12.5 tok/s at 2k, but drops
to 5.2 tok/s with a 44 s TTFT at 8k, and to 2.1 tok/s with a 188 s TTFT at 20k. On top of that it holds about 4.9 GB of
transient JS buffers during load, and its KV cache is about 57 KB/token (about 1.15 GB at
20k). Recommendation: keep E2B as the default with a 16k–20k window and offer E4B as an
opt-in with a smaller default window (≤ 8k).

**(d) transformers.js names and behaviours that differ from what we assumed:**
- `<bos>` is **not** added by `add_special_tokens: true` for this export. Prepend it manually, and update `gemmaTokenizerCheck.ts`.
- Progress gains a `status: 'progress_total'` aggregate event (no `file`, has a `files` map), fired tens of thousands of times. `download` fires in local mode too.
- The streamer's prompt flag is `next_tokens_are_prompt`. The top-level `skip_special_tokens` and `decode_kwargs.skip_special_tokens` are merged. `clean_up_tokenization_spaces` defaults to true, so pass `false`.
- EOS tokens (`<turn|>` 106, `<|tool_response>` 50, `<eos>` 1) are delivered to `put()` and the text callback; strip them in the engine.
- The `generation_config` defaults are `do_sample: true, top_k: 64, top_p: 0.95, temperature: 1`.
- `DynamicCache` has no `crop()`. `input_ids` prefix-slicing is automatic, but you must pass a full-length `attention_mask`. The cache survives `generate` only when you pass `past_key_values`.
- A direct `model.forward()` defaults to all-position logits (`num_logits_to_keep` falls back to 0).
- `InterruptableStoppingCriteria` must be passed as `stopping_criteria` (array OK) and is checked only between decode steps.
- The browser cache: large `_data` files fail `Cache.put` here (≥ 256 MiB), and the failure is silent apart from a `logger.warn`. Cache keys for local loads are absolute `http://<host>/models/<repo>/<file>` URLs. Remote keys are `https://huggingface.co/<repo>/resolve/main/<file>`.
- The ORT wasm loads from `cdn.jsdelivr.net` by default.
- `performance.memory` is unavailable in workers.
- Housekeeping, outside this spike's files: `site/src/lib/localLlm/modelFiles.json` lists the E4B `config.json` as 5,549 bytes, but the downloaded file is 5,741 bytes.

## Addendum (orchestrator, 2026-10-04): Cache API limit is partition-specific

Re-tested `cache.put` with buffer-backed `Response` bodies in stock Chrome 154 via
DevTools:

| Context | 300 MiB | 1.6 GB |
|---|---|---|
| Isolated / incognito-style browser context (memory-backed storage) | fails (`UnknownError: Unexpected internal error`) | fails |
| Default profile (disk-backed) | stored and matched | stored and matched (1.6 s) |

So the 256 MiB ceiling observed in item 5 is a property of memory-backed storage
partitions (incognito contexts and the Claude desktop app's embedded browser pane), not
of Chrome's Cache API in general. Decision: keep transformers.js's default Cache API
(`transformers-cache`) for the public site. In dev/e2e the local-models mode bypasses the
cache entirely, which is also what makes the embedded pane usable for manual testing.
transformers.js only warns when a `put` fails, so the worst case on an exotic partition is
a silent re-download next session, never a broken load. Follow-up idea: call
`navigator.storage.persist()` once to reduce eviction risk.
