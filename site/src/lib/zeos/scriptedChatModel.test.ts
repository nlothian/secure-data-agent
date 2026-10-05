import { describe, expect, it } from 'vitest';
import { createScriptedChatModel } from './scriptedChatModel';

describe('createScriptedChatModel', () => {
  it('plays each reply, ending a turn with <|im_end|> unless it ends in a tool call', () => {
    const w = createScriptedChatModel({ replies: ['<tool_call>\nx</tool_call>', 'Hi'] });
    w.createContext('1');
    w.append('1', [5, 6]);
    const all = new Uint8Array(w.meta.tokenizerSize).fill(1);
    const pieces: string[] = [];
    for (let i = 0; i < 7; i++) {
      const step = w.decodeStep('1', { allowedBlocks: null, allowedTokens: all }) as { tokenId: number };
      pieces.push(w.piece(step.tokenId));
    }
    expect(pieces).toEqual(['<tool_call>', '\n', 'x', '</tool_call>', 'H', 'i', '<|im_end|>']);
    expect(() => w.decodeStep('1', { allowedBlocks: null, allowedTokens: all })).toThrow(/no reply left/);
  });

  it('tokenizes one id per character, reserved tags included, and reports attention by mode', () => {
    const w = createScriptedChatModel({ replies: ['ab'], attention: 'uniform' });
    expect(Array.from(w.tokenize('<|im_end|>')).length).toBe('<|im_end|>'.length);
    w.createContext('j');
    w.append('j', w.tokenize('abcd'));
    const all = new Uint8Array(w.meta.tokenizerSize).fill(1);
    const step = w.decodeStep('j', {
      allowedBlocks: Uint8Array.from([1, 0, 1, 1]),
      allowedTokens: all,
    }) as { attention: Float32Array };
    expect(Array.from(step.attention).map((x) => Math.round(x * 3))).toEqual([1, 0, 1, 1]);
    const mask = new Uint8Array(w.meta.tokenizerSize);
    expect(() => w.decodeStep('j', { allowedBlocks: null, allowedTokens: mask })).toThrow(/mask refuses/);
  });

  it('attends only the latest positions in recent mode', () => {
    const w = createScriptedChatModel({ replies: ['a'], attention: 'recent' });
    w.createContext('j');
    w.append('j', w.tokenize('x'.repeat(100)));
    const all = new Uint8Array(w.meta.tokenizerSize).fill(1);
    const step = w.decodeStep('j', { allowedBlocks: null, allowedTokens: all }) as { attention: Float32Array };
    const hot = Array.from(step.attention).flatMap((x, i) => (x > 0 ? [i] : []));
    expect(hot).toEqual(Array.from({ length: 64 }, (_, k) => 36 + k));
  });
});
