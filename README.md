# Gemma Data Agent

An offline data and coding agent with built-in explainability, built for the [Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon/).

You can try it yourself at the live [Gemma Data Agent](https://gemma-data-agent.nicklothian.com/) website. It starts with an optional guided tour which will take about 5 minutes and shows all the features. 

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

Read the [complete Kaggle Write Up here](https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/gemma-data-agent-litert-and-safety-and-trust).