/**
 * Qwen 3.5 chat-template control tokens.
 *
 * Leaf module with no imports so the LLM Web Worker can use it without
 * dragging in `qwenPrompt.ts` (which imports app code). Mirrors
 * `gemmaTokens.ts`.
 *
 *   <|im_start|>{role}\n ... <|im_end|>\n     turn delimiters (ChatML)
 *   <think>\n ... \n</think>\n\n              reasoning block
 *   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
 *   <tool_response>\n ... \n</tool_response>  tool result (inside a user turn)
 *
 * Every entry in `QWEN_SINGLE_TOKENS` is one tokenizer entry (verified at
 * model load by `qwenTokenizerCheck`). The `<function=` / `<parameter=`
 * tags are plain text, not special tokens.
 */

export const IM_START = '<|im_start|>';
export const IM_END = '<|im_end|>';
export const THINK_OPEN = '<think>';
export const THINK_CLOSE = '</think>';
export const QWEN_TOOL_CALL_OPEN = '<tool_call>';
export const QWEN_TOOL_CALL_CLOSE = '</tool_call>';
export const QWEN_TOOL_RESPONSE_OPEN = '<tool_response>';
export const QWEN_TOOL_RESPONSE_CLOSE = '</tool_response>';
export const FUNCTION_OPEN = '<function=';
export const FUNCTION_CLOSE = '</function>';
export const PARAMETER_OPEN = '<parameter=';
export const PARAMETER_CLOSE = '</parameter>';

/** Empty reasoning block the template uses to switch thinking off. */
export const QWEN_EMPTY_THINK = `${THINK_OPEN}\n\n${THINK_CLOSE}\n\n`;

/** Structural tokens that must each encode to exactly one tokenizer id. */
export const QWEN_SINGLE_TOKENS: readonly string[] = [
  IM_START,
  IM_END,
  THINK_OPEN,
  THINK_CLOSE,
  QWEN_TOOL_CALL_OPEN,
  QWEN_TOOL_CALL_CLOSE,
  QWEN_TOOL_RESPONSE_OPEN,
  QWEN_TOOL_RESPONSE_CLOSE,
];
