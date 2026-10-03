/**
 * Gemma 4 chat-template control tokens.
 *
 * Kept in a leaf module with no imports so the LLM Web Worker can use them
 * without dragging in `toolPrompt.ts` (which imports `streamChat` →
 * `agentTools` → DuckDB / Pyodide). `toolPrompt.ts` re-exports everything
 * here, so existing importers are unaffected.
 *
 * Each of these is a single entry in the Gemma 4 tokenizer (verified at model
 * load by `gemmaTokenizerCheck.ts`).
 *
 *   <|turn>{role}\n ... <turn|>\n         turn delimiters
 *   <|tool>declaration:name{...}<tool|>   tool declaration (system block)
 *   <|tool_call>call:name{...}<tool_call|>      model emits to call a tool
 *   <|tool_response>response:name{...}<tool_response|>  we inject the result
 *   <|"|>...<|"|>                         string literal delimiters
 *   <|channel>thought\n...<channel|>      reasoning channel
 */

export const TURN_OPEN = '<|turn>';
export const TURN_CLOSE = '<turn|>';
export const TOOL_DECL_OPEN = '<|tool>';
export const TOOL_DECL_CLOSE = '<tool|>';
export const TOOL_CALL_OPEN = '<|tool_call>';
export const TOOL_CALL_CLOSE = '<tool_call|>';
export const TOOL_RESPONSE_OPEN = '<|tool_response>';
export const TOOL_RESPONSE_CLOSE = '<tool_response|>';
export const STRING_DELIM = '<|"|>';
export const CHANNEL_OPEN = '<|channel>';
export const CHANNEL_CLOSE = '<channel|>';
export const EMPTY_THOUGHT = `${CHANNEL_OPEN}thought\n${CHANNEL_CLOSE}`;

/** Every structural token that must encode to exactly one tokenizer id. */
export const STRUCTURAL_TOKENS: readonly string[] = [
  TURN_OPEN,
  TURN_CLOSE,
  TOOL_DECL_OPEN,
  TOOL_DECL_CLOSE,
  TOOL_CALL_OPEN,
  TOOL_CALL_CLOSE,
  TOOL_RESPONSE_OPEN,
  TOOL_RESPONSE_CLOSE,
  STRING_DELIM,
  CHANNEL_OPEN,
  CHANNEL_CLOSE,
];

/**
 * `<bos>` — must start every prompt. The ONNX export's tokenizer.json has an
 * empty post-processor, so `add_special_tokens: true` does NOT add it; the
 * worker prepends the literal text via `withBosText` in
 * `gemmaTokenizerCheck.ts` and encodes with `add_special_tokens: false`.
 */
export const BOS_TOKEN_ID = 2;
/** `<turn|>` — end of a turn; also listed in the model's `eos_token_id`. */
export const TURN_CLOSE_TOKEN_ID = 106;
