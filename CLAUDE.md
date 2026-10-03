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

The repo-root `models/` directory is gitignored. It can hold the Gemma 4
ONNX files so the dev server serves them locally instead of the app
downloading them from the Hugging Face Hub. Layout (E4B is the same under
`gemma-4-E4B-it-ONNX/`, and its decoder adds a second `_data_1` shard):

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

1. Populate it with `cd site && npm run models:fetch -- e2b` (or `e4b` /
   `all`), or the equivalent Hugging Face CLI command:

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
size. It defaults to E4B; set `GDA_E2E_MODEL=gemma-4-e2b` to use E2B.

To exercise a specific tour stage without walking the whole flow, start
a one-stage tour via the controller in DevTools:

```js
const c = await import('/src/lib/tour/controller.ts');
const s = await import('/src/lib/tour/stages/index.ts');
const stage = s.DEFAULT_TOUR.stages.find((x) => x.id === '<stage-id>');
c.startTour({ id: 'jump', stages: [stage] });
```
