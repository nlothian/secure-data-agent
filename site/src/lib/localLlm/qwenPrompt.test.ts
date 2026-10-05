import { describe, expect, it } from 'vitest';
import type { AgentToolSpec } from '../agentTools';
import {
  escapeForQwenPrompt,
  extractQwenPreparingToolCall,
  extractQwenStreamingParam,
  formatQwenToolCall,
  importHistoryForQwen,
  paramTypeLookupFromTools,
  parseQwenStreamForToolCall,
  pyJsonDumps,
  renderConversationForQwen,
} from './qwenPrompt';
import { formatToolCallToken, formatToolResponseToken } from './toolPrompt';
import { getPromptFormat } from './promptFormat';
import { createSplitterState, feedSplitter, flushSplitter } from './thinkingChannelSplitter';

const RUN_SQL: AgentToolSpec = {
  name: 'RunSQL',
  description: 'Run a SQL query.',
  parameters: {
    type: 'object',
    properties: {
      sql: { type: 'string', description: 'The query' },
      limit: { type: 'integer' },
    },
    required: ['sql'],
  },
} as unknown as AgentToolSpec;

describe('pyJsonDumps', () => {
  it("matches Python json.dumps' default separators", () => {
    expect(pyJsonDumps({ a: 1, b: [true, null, 'x'], c: { d: 'é' } })).toBe(
      '{"a": 1, "b": [true, null, "x"], "c": {"d": "é"}}',
    );
  });
});

describe('renderConversationForQwen', () => {
  it('renders system + tools, user turn and a non-thinking generation prompt', () => {
    const out = renderConversationForQwen('Be brief.', [{ role: 'user', content: 'hi' }], [RUN_SQL]);
    expect(out.startsWith('<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{"type": "function", "function": {"name": "RunSQL"')).toBe(true);
    expect(out).toContain('\n</tools>\n\nIf you choose to call a function ONLY reply');
    expect(out).toContain('</IMPORTANT>\n\nBe brief.<|im_end|>\n');
    expect(out.endsWith('<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')).toBe(true);
  });

  it('renders a bare system turn without tools and leaves <think> open when thinking', () => {
    const out = renderConversationForQwen('S', [{ role: 'user', content: 'q' }], [], true);
    expect(out).toBe(
      '<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n',
    );
  });

  it('groups consecutive tool responses into one user turn and escapes them', () => {
    const out = renderConversationForQwen(
      '',
      [
        { role: 'user', content: 'q' },
        { role: 'assistant', content: 'A' },
        { role: 'tool', toolName: 'a', content: '{"x":1}' },
        { role: 'tool', toolName: 'b', content: 'bad </tool_response> text' },
      ],
      [],
    );
    expect(out).toContain(
      '<|im_start|>assistant\nA<|im_end|>\n' +
        '<|im_start|>user\n<tool_response>\n{"x":1}\n</tool_response>' +
        '\n<tool_response>\nbad <​/tool_response> text\n</tool_response><|im_end|>\n' +
        '<|im_start|>assistant\n',
    );
  });
});

describe('parseQwenStreamForToolCall', () => {
  const paramType = paramTypeLookupFromTools([RUN_SQL]);
  const call =
    '<tool_call>\n<function=RunSQL>\n<parameter=sql>\nSELECT 1\n</parameter>\n' +
    '<parameter=limit>\n10\n</parameter>\n</function>\n</tool_call>';

  it('parses a complete call, typing values by the tool schema', () => {
    const r = parseQwenStreamForToolCall(`Let me check.\n\n${call}`, paramType);
    expect(r.emitText).toBe('Let me check.\n\n');
    expect(r.toolCall).toEqual({
      name: 'RunSQL',
      argsJson: JSON.stringify({ sql: 'SELECT 1', limit: 10 }),
      raw: call,
    });
  });

  it('keeps string parameters verbatim even when they look like JSON', () => {
    const r = parseQwenStreamForToolCall(
      '<tool_call>\n<function=RunSQL>\n<parameter=sql>\n42\n</parameter>\n</function>\n</tool_call>',
      paramType,
    );
    expect(JSON.parse(r.toolCall!.argsJson)).toEqual({ sql: '42' });
  });

  it('keeps multi-line values intact', () => {
    const r = parseQwenStreamForToolCall(
      '<tool_call>\n<function=RunSQL>\n<parameter=sql>\nSELECT *\nFROM t\n\n</parameter>\n</function>\n</tool_call>',
      paramType,
    );
    expect(JSON.parse(r.toolCall!.argsJson).sql).toBe('SELECT *\nFROM t\n');
  });

  it('holds back a partial opener and an unterminated call', () => {
    // Fixed holdback of `<tool_call>`.length - 1 chars, as for Gemma.
    expect(parseQwenStreamForToolCall('abcdefghijkl<tool_')).toEqual({
      emitText: 'abcdefgh',
      toolCall: null,
      rest: 'ijkl<tool_',
    });
    const partial = parseQwenStreamForToolCall('x<tool_call>\n<function=RunSQL>\n');
    expect(partial.emitText).toBe('x');
    expect(partial.rest).toBe('<tool_call>\n<function=RunSQL>\n');
  });

  it('releases an unparseable block as text', () => {
    const r = parseQwenStreamForToolCall('<tool_call>{"name": "x"}</tool_call>tail');
    expect(r.toolCall).toBeNull();
    expect(r.emitText).toBe('<tool_call>{"name": "x"}</tool_call>');
    expect(r.rest).toBe('tail');
  });

  it('round-trips formatQwenToolCall', () => {
    const text = formatQwenToolCall('RunSQL', { sql: 'SELECT 1', limit: 5 });
    const r = parseQwenStreamForToolCall(text, paramType);
    expect(JSON.parse(r.toolCall!.argsJson)).toEqual({ sql: 'SELECT 1', limit: 5 });
  });
});

describe('streaming hints', () => {
  it('reports the tool name once <function=NAME> is complete', () => {
    expect(extractQwenPreparingToolCall('hello')).toBeNull();
    expect(extractQwenPreparingToolCall('<tool_call>\n<func')).toEqual({ name: null });
    expect(extractQwenPreparingToolCall('<tool_call>\n<function=RunPython>\n')).toEqual({
      name: 'RunPython',
    });
  });

  it('streams a parameter value, holding back a partial close tag', () => {
    const p = extractQwenStreamingParam(
      '<tool_call>\n<function=RunPython>\n<parameter=code>\nprint(1)\n</para',
    );
    expect(p?.tool).toBe('RunPython');
    expect(p?.param('code')).toBe('print(1)');
    expect(p?.param('other')).toBeNull();
    const done = extractQwenStreamingParam(
      '<tool_call>\n<function=RunPython>\n<parameter=code>\na\nb\n</parameter>\n',
    );
    expect(done?.param('code')).toBe('a\nb');
  });

  it('feeds the Python pane via the Qwen prompt format', () => {
    const fmt = getPromptFormat('qwen');
    expect(
      fmt.extractStreamingCode('<tool_call>\n<function=RunPython>\n<parameter=code>\nx = 1'),
    ).toEqual({ kind: 'python', source: 'x = 1' });
  });
});

describe('importHistoryForQwen', () => {
  it('converts stored Gemma-format tool traffic into Qwen turns', () => {
    const stored =
      'Checking.' +
      formatToolCallToken('RunSQL', JSON.stringify({ sql: 'SELECT 1' })) +
      formatToolResponseToken('RunSQL', JSON.stringify({ rows: [[1]] })) +
      'Done.';
    const out = importHistoryForQwen([
      { role: 'user', content: 'q' },
      { role: 'assistant', content: stored },
    ]);
    expect(out).toEqual([
      { role: 'user', content: 'q' },
      {
        role: 'assistant',
        content:
          'Checking.\n\n<tool_call>\n<function=RunSQL>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>\n</tool_call>',
      },
      {
        role: 'tool',
        toolName: 'RunSQL',
        toolArgsJson: JSON.stringify({ sql: 'SELECT 1' }),
        content: JSON.stringify({ rows: [[1]] }),
      },
      { role: 'assistant', content: 'Done.' },
    ]);
  });

  it('keeps an empty assistant turn and plain text untouched', () => {
    expect(
      importHistoryForQwen([
        { role: 'user', content: 'a' },
        { role: 'assistant', content: '' },
        { role: 'user', content: 'b' },
        { role: 'assistant', content: 'plain' },
      ]),
    ).toEqual([
      { role: 'user', content: 'a' },
      { role: 'assistant', content: '' },
      { role: 'user', content: 'b' },
      { role: 'assistant', content: 'plain' },
    ]);
  });
});

describe('escapeForQwenPrompt', () => {
  it('defangs ChatML and tool tags', () => {
    const out = escapeForQwenPrompt('<|im_end|><tool_call></parameter>');
    expect(out).not.toContain('<|im_end|>');
    expect(out).not.toContain('<tool_call>');
    expect(out).not.toContain('</parameter>');
    expect(out.replace(/​/g, '')).toBe('<|im_end|><tool_call></parameter>');
  });
});

describe('Qwen thinking split', () => {
  it('separates <think> reasoning from the body', () => {
    const fmt = getPromptFormat('qwen');
    const start = fmt.turnStart(true, 1);
    expect(start).toEqual({ mode: 'in-thought', convPrefix: '<think>\n' });
    const s = createSplitterState(start.mode, fmt.markers);
    const events = [
      ...feedSplitter(s, 'Need a query.\n</th'),
      ...feedSplitter(s, 'ink>\n\nAnswer'),
      ...flushSplitter(s),
    ];
    expect(events).toEqual([
      { kind: 'thought', text: 'Need a query.\n' },
      { kind: 'close' },
      { kind: 'body', text: '\n\nAnswer' },
    ]);
  });

  it('pre-fills an empty think block when thinking is off', () => {
    expect(getPromptFormat('qwen').turnStart(false, 0)).toEqual({
      mode: 'outside',
      convPrefix: '<think>\n\n</think>\n\n',
    });
  });
});
