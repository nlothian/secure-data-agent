import type { TourStage } from '../types';

const welcome: TourStage = {
  id: 'welcome',
  markdown: `# Secure Data Agent

This app uses ZEOS Qwen 4B to show how AI agents work, end to end, and how to keep them safe from prompt injection.

It is a fully functional, fully private agent environment running entirely in your browser, so feel free to experiment.

Key features:
- **Local inference** — Qwen 3.5 4B runs on your GPU via ONNX Runtime on WebGPU. The model is a one-off download (about 2.8 GB) from Hugging Face, kept in your browser's storage for later visits.
- **Prompt-injection defence** — the ZEOS kernel tracks which text came from you and which came from tools. Once the model has read untrusted tool output, any action with side effects (running Python, writing files, loading data) waits for your approval.
- **Tool use** — run Python (via Pyodide), SQL (via DuckDB), and React, all in the browser.
- **Data loading** — load CSV or Parquet files from a sandboxed local folder (via the HTML5 File System Access API) or from remote servers (via DuckDB).
- **Self-explaining code** — the app ships with its own source code, so the built-in AI explainer can show you how any part of it works.

This tour downloads the model, then walks through these features. It takes about 5 minutes, plus the download, depending on your hardware and connection.
`,
  cutouts: [],
  onEnter: [{ action: 'clearAllData' }, { action: 'newChat' }],
  next: 'manual',
};

export default welcome;
