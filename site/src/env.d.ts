/// <reference types="astro/client" />

interface ImportMetaEnv {
  /**
   * Set to `'1'` to serve the repo-root `models/` folder at `/models/` from
   * the Vite dev server and point transformers.js there instead of the
   * Hugging Face Hub. Dev/e2e only; see CLAUDE.md.
   */
  readonly PUBLIC_LOCAL_MODELS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
