# Project notes

## Package manager

This project uses **npm**, not pnpm or yarn. The committed lockfile is
`site/package-lock.json`. Use `npm install` / `npm run <script>` from
`site/`. Do not introduce a `pnpm-lock.yaml` or `yarn.lock`. The
pnpm-flavoured comments in `site/astro.config.mjs` (about `.pnpm`
symlink paths and `@codemirror/state` deduping) are leftover context;
the `dedupe` config is harmless under npm and still useful as
documentation, so leave it alone.

## No SSR

This project does not use SSR. Astro is configured for static output and the
app runs entirely in the browser. Treat all React components as client-only:

- Use `client:only="react"` for React islands rather than `client:idle` /
  `client:load` / `client:visible` — there is no value in producing server
  HTML for them, and SSR-only warnings (e.g. `useLayoutEffect does nothing on
  the server`) should be fixed by skipping SSR, not by working around it.
- Browser-only APIs (`window`, `document`, `localStorage`, etc.) can be used
  directly inside React components without `typeof window` guards.

## Local model files for dev/e2e

The repo-root `models/` directory is gitignored. It can hold the local
models' ONNX files (Gemma 4 E2B / E4B, Qwen 3.5 4B) so the dev server serves
them locally instead of the app downloading them from the Hugging Face Hub.
Layout (E4B is the same under `gemma-4-E4B-it-ONNX/`, and its decoder adds a
second `_data_1` shard; Qwen follows the same text-only file set under its
own repo id):

```
models/onnx-community/gemma-4-E2B-it-ONNX/
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
  onnx/embed_tokens_q4f16.onnx
  onnx/embed_tokens_q4f16.onnx_data
  onnx/decoder_model_merged_q4f16.onnx
  onnx/decoder_model_merged_q4f16.onnx_data
  onnx/decoder_model_merged_q4f16.onnx_data_1   (E4B only)
```

The expected files and byte sizes are listed in
`site/src/lib/localLlm/modelFiles.json`.

### Qwen 3.5 4B

The Qwen entry in `site/src/lib/localLlm/models.ts` (`qwen3.5-4b`) uses
`onnx-community/Qwen3.5-4B-ONNX-OPT`. Use the `-OPT` export, not
`onnx-community/Qwen3.5-4B-ONNX`. The plain export runs each
linear-attention (Gated DeltaNet) layer's prefill as an ONNX `Scan`, which
ONNX Runtime Web places on the CPU, with a GPU round-trip per token per
layer. That makes prefill about 13 tok/s, versus ~250 tok/s for Gemma E4B.
The `-OPT` export uses the fused `com.microsoft:LinearAttention` /
`CausalConvWithState` ops, which have WebGPU kernels. The export needs
text-only q4f16 `embed_tokens` and `decoder_model_merged` files, like the
Gemma repos. To switch exports:

1. Set `hfRepoId` in `models.ts`, and the `qwen4b` alias in
   `site/scripts/fetch-models.mjs` and in
   `site/e2e/llm/realModelSql.spec.ts`.
2. Run `cd site && npm run models:manifest -- <repo>` to write its
   `modelFiles.json` entry. Set `approxBytes` to the printed total.
3. Run `npm run models:fetch -- qwen4b`, then `npx vitest run
   src/lib/localLlm/qwenTokenizer.test.ts`. That test checks the hand-written
   Qwen prompt renderer (`qwenPrompt.ts`) against the export's own
   `chat_template.jinja`. It skips while the files are absent.

Until the manifest entry exists, the model still loads, but it is never
reported as cached (no boot-time eager load) and its size is `approxBytes`.

Every model family has its own chat template (`promptFormat.ts`). Chat
history and the UI always use the Gemma wire format; the Qwen format
converts stored history on the way in.

### Fetching and running

1. Populate it with `cd site && npm run models:fetch -- e2b` (or `e4b` /
   `qwen4b` / `all`), or the equivalent Hugging Face CLI command:

   ```sh
   hf download onnx-community/gemma-4-E2B-it-ONNX \
     --include "*.json" "chat_template.jinja" \
       "onnx/embed_tokens_q4f16.onnx*" "onnx/decoder_model_merged_q4f16.onnx*" \
     --local-dir models/onnx-community/gemma-4-E2B-it-ONNX
   ```

2. Run `cd site && PUBLIC_LOCAL_MODELS=1 npm run dev`. The files are served
   at `/models/` by `scripts/local-models-vite-plugin.mjs`. In this mode
   transformers.js runs with `allowRemoteModels=false`, so a missing file
   fails loudly instead of falling back to the Hub, and the browser Cache
   API is bypassed. Astro picks the next free port; check the "Local" line
   in stdout for the URL.
3. Click the chevron next to the model label in the chat sidebar and pick
   the model. There is no download-size dialog in local mode.
4. Wait for the Throbber's "Loading … · N%" / "Loading … onto GPU" status
   to disappear. The chat textarea becomes enabled when the model is ready.

`npm run test:llm_tests` starts its own dev server on :4322 with
`PUBLIC_LOCAL_MODELS=1` (it never reuses an existing server). It skips
unless every required file for the chosen model is present with the right
size. It defaults to E4B; set `GDA_E2E_MODEL=gemma-4-e2b` to use E2B, or
`GDA_E2E_MODEL=qwen3.5-4b` for Qwen.

To exercise a specific tour stage without walking the whole flow, start
a one-stage tour via the controller in DevTools:

```js
const c = await import('/src/lib/tour/controller.ts');
const s = await import('/src/lib/tour/stages/index.ts');
const stage = s.DEFAULT_TOUR.stages.find((x) => x.id === '<stage-id>');
c.startTour({ id: 'jump', stages: [stage] });
```

## Cross-origin isolation (COOP/COEP)

Every page is served with `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: credentialless`: `server.headers` in
`site/astro.config.mjs` for `astro dev` / `astro preview`, and `/*` in
`site/public/_headers` for production. That makes `crossOriginIsolated`
true, which the ZEOS kernel needs for `SharedArrayBuffer` + `Atomics.wait`.
`credentialless` (not `require-corp`) keeps Google Fonts, the Hugging Face
Hub's CDN redirects, jsDelivr (Pyodide, DuckDB-wasm) and remote CSV URLs
working without CORP headers; cross-origin no-cors loads just go without
cookies. Any new cross-origin iframe or popup must cope with this.
Isolation also turns on ONNX Runtime's multi-threaded wasm by default,
which made Gemma generation ~2.5x slower, so `src/workers/llm.worker.ts` pins
`env.backends.onnx.wasm.numThreads = 1` (the pre-isolation behaviour).

## ZEOS kernel (browser)

ZEOS (`zeos-task2-transformers`) runs in its own Pyodide
314 worker, separate from the app's Pyodide 0.29 RunPython worker.

- `cd site && npm run zeos:sync` (env `ZEOS_REPO`, default
  `/Users/nlothian/dev/github/metacognitionai/zeos-task2-transformers`)
  builds the `zeos` and `zeos-coop-count-web` wheels with `uv`, and copies
  them plus the case directories into `site/public/zeos/` with a
  `manifest.json`. `public/zeos/` is **generated and gitignored**; re-run the
  sync after changing ZEOS.
- The same script vendors ZEOS's JS (`frames.js`, `model_channel.js`,
  `stub_worker.js`, and `opt_zeos_worker.js` once it exists) into
  `site/src/lib/zeos/vendor/`, which **is committed** (`SOURCE.json` records
  the ZEOS commit). Do not edit vendored `.js`
  files; the `.d.ts` files there are hand-written.
- `src/workers/zeosKernel.worker.ts` is the kernel worker; its generic RPC
  (`boot`, `call`, `callMethod`, `getAttr`, `exec`, `release`,
  `attachModel`) is glued to Python by `src/lib/zeos/zeos_rpc.py`.
  `src/lib/zeos/zeosHost.ts` (`startZeosKernel`, `startZeos`,
  `kernel.attachModel`) starts it and the model thread on the page thread.
- `e2e/zeosKernel.spec.ts` (in `npm run test:e2e`) boots it with the stub
  model thread and steps `coop-count-scripted`; it skips until
  `npm run zeos:sync` has run, and needs network for jsDelivr.

## ZEOS Qwen 4B (model `zeos-qwen3.5-4b`)

The agent runs under the ZEOS kernel instead of the transformers.js worker
(family `zeos-qwen`; `src/lib/streamChat.ts` routes it to
`src/lib/zeos/streamZeos.ts`). User messages enter on ring 2 (TRUSTED),
tool results on ring 3 (EXTERNAL). Calls go to `tools.read` or
`tools.effect` according to `src/lib/zeos/zeosToolClasses.ts`, which is the
only place that policy lives. `RunSQL` is a read only when its SQL is inline
(`sql`) and read-only, so this model gets an inline-`sql` RunSQL spec. When the
kernel refuses an effect, the chat shows an approval card. RunSubAgent and
compaction are off for this model. Side tasks such as code summaries use
`qwen3.5-4b`.

- **Selecting it.** It is listed only with `PUBLIC_LOCAL_MODELS=1` or in stub
  mode. Its files are not on the Hub. Run `cd site && npm run models:fetch --
  zeosq4b`, which hard-links them from
  `$ZEOS_REPO/demo/coop-count-web/models/Qwen3.5-4B-ZEOS-OPT`. After
  rebuilding the export, run `npm run models:manifest-local --
  metacognitionai/Qwen3.5-4B-ZEOS-OPT`, which rewrites its `modelFiles.json`
  entry from `meta.json`.
- **Real model thread.** It is not wired yet. The one plug point is
  `createRealModelThread` in `src/lib/zeos/zeosModelWorker.ts`.
- **Stub mode (dev only).** Set localStorage `gda.zeos.stub` to
  `{"replies": [...], "attention": "first" | "recent" | "uniform" | "none"}`
  and reload. The scripted chat stub (`src/lib/zeos/scriptedChatModel.ts`)
  then plays the replies; a reply ending in `</tool_call>` is a tool call. With
  `recent`, reading a tool result demotes the job; with `first`, nothing ever
  does. `e2e/zeosChat.spec.ts` uses this.
- **Gate mode.** The "Attention-only approval" checkbox (config
  `zeosAttentionOnly`, shown only for this model) switches ZEOS
  `open_chat(gate_mode=…)`:
  - **Strict, the default.** Reading any tool result needs approval for
    effects until the next user message.
  - **Attention.** Effects need approval only after a measured-attention
    demotion.

  A switch applies from the next message: the conversation key changes, so a
  fresh run replays the history.
- **Runs.** There is one ZEOS `ChatRun` per conversation, keyed by system
  prompt, thinking, gate mode and prior messages. A new chat, reload, retry or
  abort opens a fresh run and replays history (`buildZeosImport`). Past
  assistant turns replay at their recorded `ChatMessage.trust.integrity`;
  turns with no record replay as untrusted (3).
