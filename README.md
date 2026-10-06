# Secure Data Agent

An offline data and coding agent that runs local models (Gemma 4, Qwen 3.5 and ZEOS Qwen 4B) entirely in your browser, with built-in explainability and a ZEOS-kernel mode that gates effectful tools behind trust rings.

It began as [Gemma Data Agent](https://github.com/nlothian/gemma-data-agent), built for the [Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon/). That version is live at [gemma-data-agent.nicklothian.com](https://gemma-data-agent.nicklothian.com/), with an optional guided tour of about 5 minutes that shows all the features.

Chrome or Edge only for now: the model's largest weight tensor needs a single WebGPU buffer of about 1.5 GB, and Firefox caps `maxBufferSize` at 1 GiB.


## Features

- **Chat with a local Gemma 4 model.** Pick `Gemma 4 E2B` (≈3.1 GB download) or `E4B` (≈4.9 GB) — q4f16 ONNX weights from `onnx-community`, run by transformers.js on ONNX Runtime WebGPU, downloaded once from Hugging Face and cached by your browser.
- **Run code in-browser.** Pyodide for Python, DuckDB-WASM for SQL, and an isolated React sandbox for JSX visualizations (three.js, D3 or ReCharts). The model issues tool calls; the user sees the call, the streamed source, and the result.
- **Ask "how does this app work?".** The Explainer Panel uses Gemma to search over the bundled source code, explains how it works and provides clickable links to view the bundled source.
- **Detailed, step by step explanations.** Use the step mode and the agent will pause before each step, and use a separate Gemma thread to explain what the Agent is about to do.
- **Careful context management**. The smaller Gemma models perform worse as the context length increases. The Agent tracks token counts and runs a compaction when required. Sub-agents further protect the main context from lengthy debugging loops, while progressive disclosure via skills means tool instructions are only added to the context when needed.

## How it runs

- [transformers.js](https://huggingface.co/docs/transformers.js) v4 runs inside a Web Worker, so model loading and token generation never block the UI.
- The model is loaded as the text-only `Gemma4ForCausalLM` (no vision/audio towers) with `dtype: "q4f16"` on `device: "webgpu"`, using the [`onnx-community/gemma-4-E2B-it-ONNX`](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX) and [`onnx-community/gemma-4-E4B-it-ONNX`](https://huggingface.co/onnx-community/gemma-4-E4B-it-ONNX) weights.
- Prompts are built with a hand-rendered Gemma 4 chat template that uses the model's native tool-calling tokens.
- Long prompts are prefilled in 2k-token chunks (a single ONNX Runtime pass overflows above ~16k tokens), and the KV cache is kept between tool iterations: when the next prompt is a strict extension of the previous one, only the new tokens are prefilled.
- For development and the end-to-end LLM tests the weights can be served from a local `models/` folder instead of the Hub; see [CLAUDE.md](CLAUDE.md) for the layout and commands.

## ZEOS Qwen 4B

`ZEOS Qwen 4B` is the default local model where the browser can run it (WebGPU with shader-f16 and a cross-origin isolated page, so not Safari); elsewhere the default is `Gemma 4 E2B`. It runs the agent under the [ZEOS](https://github.com/metacognitionai/zeos) kernel. The kernel runs in Pyodide and drives Qwen3.5-4B, using the `-OPT` ONNX export with a key mask and attention output added. Your chat messages enter on a trusted ring (2), while file and tool output enters on the external ring (3). Once the model has read ring-3 content, effectful tools (WriteLines, RunPython, non-read-only SQL, …) need your approval. A toggle in the model dropdown switches between two gate modes:

- **strict:** any tool output read this turn raises the gate.
- **attention-only:** only measured attention to ring-3 content raises it.

A second toggle at the bottom of the model dropdown, **Mask tool choice** (off by default), hides ring-3 content from the model while it writes the name of the tool it calls, so a tool result cannot pick the next tool. The arguments still see everything.

See [CLAUDE.md](CLAUDE.md) for setup.

### Performance

Measured in Chrome on an Apple M1 Max, with the model files already in the OS cache and the same SQL prompt for both models:

| | ZEOS Qwen 4B | Plain Qwen 3.5 4B |
|---|---|---|
| Ready after page load | 5.3 s (Pyodide boots in parallel) | 4.3 s |
| First-turn prompt | 6,790 tokens | 7,185 tokens |
| First-turn prefill | first token at 32.8 s, ~210 tok/s | ~31 s, ~232 tok/s |
| Later turns, first token | ~1.6 s (the run is reused) | — |
| Decode at ~6.8k positions | 72 ms/step, ~14 tok/s | 17–18 tok/s |
| Mask tool choice (off by default), per tool call after a tool result | +1.8–2.2 s at ~8k positions in a bench (22–31% of the call); 2.6–3.9 s a masked call in the SQL e2e (more when content arrives after the first tool result, such as the 1.1k-token skill card, which is then prefilled on both caches) | — |

- **Decode overhead:** most of the gap comes from the graph's measured-attention output, since the graph alone takes 68 ms a step. The kernel, the Atomics channel and the event stream add about 4 ms per token.
- **Model worker alone:** load in 3.8–4.2 s (files cached). Prefill runs at 288–305 tok/s. Decode takes 47–49 ms a step at ~1k positions, about 21 tok/s.
- **Hiding a past span:** this replays from the nearest snapshot, which takes 2.4–2.7 s for 16 positions at 600 in a 1,052-token context.
- **Graph benchmark (ZEOS-OPT vs -OPT):** prefill of 2,048 tokens from empty runs at 326 vs 270 tok/s. Decode at 64 positions takes 45 vs 40 ms, and at ~4.1k positions 56 vs 46 ms.

Read the [complete Kaggle Write Up of the original Gemma version here](https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/gemma-data-agent-litert-and-safety-and-trust).

## Build and Run

```
cd site
npm install
npm run dev
```
