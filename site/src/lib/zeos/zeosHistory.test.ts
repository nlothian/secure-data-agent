import { describe, expect, it } from 'vitest';
import { formatToolCallToken, formatToolResponseToken } from '../localLlm/toolPrompt';
import { buildZeosImport, ringOfImportTurn, toolResultForZeos } from './zeosHistory';

const call = formatToolCallToken('ReadLines', JSON.stringify({ path: '/input/a.csv' }));
const result = formatToolResponseToken('ReadLines', JSON.stringify({ lines: ['a,b', '1,2'] }));

describe('buildZeosImport', () => {
  it('puts user turns on ring 2, tool results on 3, and assistant text at its recorded integrity', () => {
    const turns = buildZeosImport([
      { role: 'system', content: 'ignored' },
      { role: 'user', content: 'what is in a.csv?' },
      {
        role: 'assistant',
        content: `Let me look.${call}${result}Two rows.`,
        trust: { integrity: 3, ring: 3, toolRings: [3] },
      },
      { role: 'user', content: 'thanks' },
      { role: 'assistant', content: 'You are welcome.', trust: { integrity: 2, ring: 2 } },
      { role: 'user', content: 'and from another model?' },
      { role: 'assistant', content: 'No record of trust.' },
    ]);
    expect(turns.map((t) => [t.role, ringOfImportTurn(t)])).toEqual([
      ['user', 2],
      ['assistant', 3],
      ['tool', 3],
      ['assistant', 3],
      ['user', 2],
      ['assistant', 2],
      ['user', 2],
      ['assistant', 3],
    ]);
    expect(turns[1].text).toMatch(/^Let me look\.\n\n<tool_call>\n<function=ReadLines>\n/);
    expect(turns[1].text.endsWith('</tool_call>')).toBe(true);
    expect(turns[2]).toEqual({ role: 'tool', text: JSON.stringify({ lines: ['a,b', '1,2'] }), toolName: 'ReadLines' });
    expect(turns[3].text).toBe('Two rows.');
  });

  it('starts at the first user turn and skips empty text', () => {
    const turns = buildZeosImport([
      { role: 'assistant', content: 'Hello! (greeting before any user turn)' },
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: '' },
      { role: 'user', content: 'again' },
    ]);
    expect(turns.map((t) => t.role)).toEqual(['user', 'user']);
  });

  it('defangs Qwen tags in tool results and never sends an empty one', () => {
    expect(toolResultForZeos('{"x":"<tool_call>"}')).not.toContain('<tool_call>');
    expect(toolResultForZeos('')).toBe('(empty result)');
  });
});
