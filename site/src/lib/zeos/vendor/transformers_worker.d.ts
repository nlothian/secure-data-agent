// Hand-written types for the vendored ./transformers_worker.js (not generated).
// The site uses it only through ./opt_zeos_worker.js.
export function encodePlain(tokenizer: unknown, text: string): number[];
export function sampleToken(
  logits: Float32Array,
  allowedTokens: Uint8Array | null,
  limit: number,
  sample: { temperature: number; topK: number; u: number },
): number;
