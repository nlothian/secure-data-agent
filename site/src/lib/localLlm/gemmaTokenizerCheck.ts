/**
 * Load-time sanity check that the tokenizer we just loaded agrees with the
 * Gemma 4 chat-template assumptions baked into `toolPrompt.ts` /
 * `gemmaTokens.ts`. Run by the LLM worker right after `from_pretrained`; a
 * non-empty result fails the load with `tokenizer-mismatch` rather than
 * letting the model silently see multi-token "structural" markers.
 *
 * Pure and import-light (no transformers.js) so it can be unit-tested against
 * a fake and reused by the real-tokenizer test.
 */
import {
  BOS_TOKEN_ID,
  STRING_DELIM,
  STRUCTURAL_TOKENS,
  TURN_CLOSE,
  TURN_CLOSE_TOKEN_ID,
} from './gemmaTokens';

/**
 * Literal `<bos>` text. The Gemma 4 ONNX tokenizer's post-processor adds NO
 * special tokens even with `add_special_tokens: true` — the official chat
 * template writes `{{ bos_token }}` as text instead. So the worker prepends
 * this string and encodes with `add_special_tokens: false`, which yields
 * exactly one `<bos>` regardless of the tokenizer's post-processor.
 */
export const BOS_TEXT = '<bos>';

/** `prompt` with a single leading `<bos>` (idempotent). */
export function withBosText(prompt: string): string {
  return prompt.startsWith(BOS_TEXT) ? prompt : BOS_TEXT + prompt;
}

export interface TokenizerLike {
  encode(text: string, opts: { add_special_tokens: boolean }): number[];
  decode(ids: number[], opts: { skip_special_tokens: boolean }): string;
}

function countOf(ids: readonly number[], id: number): number {
  let n = 0;
  for (const x of ids) if (x === id) n++;
  return n;
}

/** Returns a list of human-readable problems; empty means the tokenizer is OK. */
export function checkGemmaTokenizer(t: TokenizerLike, eosIds: number[]): string[] {
  const problems: string[] = [];
  const singleIds = new Map<string, number>();

  for (const tok of STRUCTURAL_TOKENS) {
    const ids = t.encode(tok, { add_special_tokens: false });
    if (ids.length !== 1) {
      problems.push(`${tok} encodes to ${ids.length} ids [${ids.join(',')}], expected 1`);
      continue;
    }
    singleIds.set(tok, ids[0]);
    const back = t.decode(ids, { skip_special_tokens: false });
    if (back !== tok) {
      problems.push(`${tok} (id ${ids[0]}) decodes to ${JSON.stringify(back)}`);
    }
  }

  const turnClose = singleIds.get(TURN_CLOSE);
  if (turnClose !== undefined && turnClose !== TURN_CLOSE_TOKEN_ID) {
    problems.push(`${TURN_CLOSE} is id ${turnClose}, expected ${TURN_CLOSE_TOKEN_ID}`);
  }
  if (!eosIds.includes(TURN_CLOSE_TOKEN_ID)) {
    problems.push(
      `eos_token_id [${eosIds.join(',')}] does not include ${TURN_CLOSE} (${TURN_CLOSE_TOKEN_ID})`,
    );
  }

  const delimId = singleIds.get(STRING_DELIM);
  if (delimId !== undefined) {
    const ids = t.encode(`x${STRING_DELIM}y`, { add_special_tokens: false });
    const n = countOf(ids, delimId);
    if (n !== 1) {
      problems.push(`${STRING_DELIM} appears ${n} times in encode('x${STRING_DELIM}y'), expected 1`);
    }
  }

  // The exact path the worker uses to build prompt ids (see `withBosText`).
  for (const sample of ['x', `${BOS_TEXT}x`]) {
    const ids = t.encode(withBosText(sample), { add_special_tokens: false });
    const label = `encode(withBosText(${JSON.stringify(sample)}))`;
    if (ids[0] !== BOS_TOKEN_ID) {
      problems.push(`${label} starts with ${ids[0]}, expected <bos> (${BOS_TOKEN_ID})`);
    }
    const bosCount = countOf(ids, BOS_TOKEN_ID);
    if (bosCount !== 1) {
      problems.push(`${label} contains <bos> ${bosCount} times, expected 1`);
    }
  }

  return problems;
}
