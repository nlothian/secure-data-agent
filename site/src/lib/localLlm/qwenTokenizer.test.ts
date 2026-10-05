/**
 * Checks the hand-written Qwen prompt format against the REAL Qwen 3.5 ONNX
 * export in the gitignored repo-root `models/` folder: the tokenizer must
 * pass `checkQwenTokenizer`, and `renderConversationForQwen` must produce
 * byte-for-byte what the export's own `chat_template.jinja` renders for the
 * same conversation. Skipped (with a warning) when the files are absent —
 * populate them with `npm run models:fetch -- qwen4b`.
 */
import { describe, expect, it } from 'vitest';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { AgentToolSpec } from '../agentTools';
import { checkQwenTokenizer } from './gemmaTokenizerCheck';
import { getLocalGemmaModel } from './models';
import { importHistoryForQwen, renderConversationForQwen } from './qwenPrompt';
import { formatToolCallToken, formatToolResponseToken } from './toolPrompt';

const REPO = getLocalGemmaModel('qwen3.5-4b')!.hfRepoId;
const MODELS_ROOT = fileURLToPath(new URL('../../../../models/', import.meta.url));
const FILES = ['tokenizer.json', 'tokenizer_config.json', 'config.json'];
const missing = FILES.filter((p) => !existsSync(`${MODELS_ROOT}${REPO}/${p}`));
const present = missing.length === 0;
if (!present) {
  console.warn(
    `[qwenTokenizer.test] skipping ${REPO}: files missing under ${MODELS_ROOT} ` +
      `(${missing.join(', ')}). Run \`npm run models:fetch -- qwen4b\` to enable.`,
  );
}

const TOOL = {
  name: 'RunSQL',
  description: 'Run a SQL query.',
  parameters: {
    type: 'object',
    properties: { sql: { type: 'string', description: 'The query' } },
    required: ['sql'],
  },
} as unknown as AgentToolSpec;

async function loadTokenizer() {
  const { env, AutoTokenizer } = await import('@huggingface/transformers');
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = MODELS_ROOT;
  return AutoTokenizer.from_pretrained(REPO);
}

describe.skipIf(!present)(`Qwen 3.5 tokenizer + template (real files): ${REPO}`, () => {
  it('has single-token structural markers', async () => {
    const tok = await loadTokenizer();
    const imEnd = tok.encode('<|im_end|>', { add_special_tokens: false });
    const problems = checkQwenTokenizer(
      {
        encode: (text, opts) => tok.encode(text, opts),
        decode: (ids, opts) => tok.decode(ids, opts),
      },
      imEnd,
    );
    expect(problems).toEqual([]);
  }, 60_000);

  it.each([false, true])(
    'renders like chat_template.jinja (enable_thinking=%s)',
    async (thinking) => {
      const tok = await loadTokenizer();
      const args = { sql: 'SELECT 1' };
      const result = { rows: [[1]] };
      const expected = tok.apply_chat_template(
        [
          { role: 'system', content: 'Be brief.' },
          { role: 'user', content: 'How many?' },
          {
            role: 'assistant',
            content: 'Checking.',
            tool_calls: [{ type: 'function', function: { name: 'RunSQL', arguments: args } }],
          },
          { role: 'tool', content: JSON.stringify(result) },
          { role: 'assistant', content: 'One.' },
          { role: 'user', content: 'Thanks' },
        ] as never,
        {
          tools: [
            {
              type: 'function',
              function: {
                name: TOOL.name,
                description: TOOL.description,
                parameters: TOOL.parameters,
              },
            },
          ],
          add_generation_prompt: true,
          tokenize: false,
          enable_thinking: thinking,
        } as never,
      ) as unknown as string;

      const stored =
        'Checking.' +
        formatToolCallToken('RunSQL', JSON.stringify(args)) +
        formatToolResponseToken('RunSQL', JSON.stringify(result)) +
        'One.';
      const ours = renderConversationForQwen(
        'Be brief.',
        importHistoryForQwen([
          { role: 'user', content: 'How many?' },
          { role: 'assistant', content: stored },
          { role: 'user', content: 'Thanks' },
        ]),
        [TOOL],
        thinking,
      );
      expect(ours).toBe(expected);
    },
    60_000,
  );
});
