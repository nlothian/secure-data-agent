/**
 * A ZEOS model worker with no model, for the chat machine: the TypeScript
 * twin of ZEOS `tests/chat_workers.py` `ScriptedChatWorker`.
 *
 * Its vocabulary is the reserved pieces of a Qwen3.5 export (`<|im_end|>`,
 * `<tool_call>`, …), one piece per printable ASCII character, a newline and a
 * tab. It tokenizes one id per character (anything else is `<unk>`) and parses
 * no special token, as `encodePlain` does. Its replies are one tape per
 * context: each reply in order, spelled greedily from the longest pieces (so
 * `<tool_call>` is the reserved piece), followed by `<|im_end|>` unless it
 * ends in a tool call. Blocks are one position each.
 *
 * `attention` decides what each step reports attending: `none` (the worker
 * cannot measure, so nothing demotes), `uniform` (every allowed position
 * equally, which after a long system prompt is too thin to demote), `recent`
 * (the last 64 allowed positions equally: right after a tool result arrives
 * that is mostly the result, so reading it demotes the job), or `first` (only
 * the system prompt's first position, so nothing ever demotes).
 */
import type { ZeosModelWorkerLike } from './vendor/model_channel';

export type StubAttention = 'none' | 'uniform' | 'first' | 'recent';

/** Positions `recent` attention spreads over: the latest arrival and a little more. */
const RECENT_WINDOW = 64;

export interface ChatStubScript {
  replies: string[];
  attention?: StubAttention;
}

const SPECIAL = [
  '<pad>',
  '<|endoftext|>',
  '<|im_start|>',
  '<|im_end|>',
  '<tool_call>',
  '</tool_call>',
  '<think>',
  '</think>',
  '<tool_response>',
  '</tool_response>',
  '<unk>',
] as const;
const PAD_ID = 0;
const IM_END_ID = 3;
const UNK_ID = SPECIAL.indexOf('<unk>');
const CONTROL = ['<|endoftext|>', '<|im_start|>', '<|im_end|>'];

function chars(): string[] {
  const out: string[] = [];
  for (let c = 32; c < 127; c++) out.push(String.fromCharCode(c));
  out.push('\n', '\t');
  return out;
}

interface Context {
  ids: number[];
  tape: number[];
}

export interface ScriptedChatModel extends ZeosModelWorkerLike {
  vocab: readonly string[];
  /** The context's ids as text, for tests. */
  text(jobId: string): string;
  spell(text: string): number[];
}

export function createScriptedChatModel(script: ChatStubScript): ScriptedChatModel {
  const vocab: string[] = [...SPECIAL, ...chars()];
  const ids = new Map(vocab.map((p, i) => [p, i]));
  const longest = Math.max(...vocab.map((p) => p.length));
  const attention = script.attention ?? 'none';

  const spell = (text: string): number[] => {
    const out: number[] = [];
    let at = 0;
    while (at < text.length) {
      let matched = false;
      for (let size = Math.min(longest, text.length - at); size > 0; size--) {
        const id = ids.get(text.slice(at, at + size));
        if (id !== undefined && id !== PAD_ID) {
          out.push(id);
          at += size;
          matched = true;
          break;
        }
      }
      if (!matched) throw new Error(`no piece spells ${JSON.stringify(text[at])}`);
    }
    return out;
  };

  const tape: number[] = [];
  for (const reply of script.replies) {
    tape.push(...spell(reply));
    if (!reply.endsWith('</tool_call>')) tape.push(IM_END_ID);
  }
  const contexts = new Map<string, Context>();
  const ctxOf = (jobId: string): Context => {
    const ctx = contexts.get(jobId);
    if (!ctx) throw new Error(`no context ${jobId}`);
    return ctx;
  };

  return {
    vocab,
    meta: { tokenizerSize: vocab.length },
    backend: 'stub-chat',
    spell,
    info: () => ({
      blockSize: 1,
      padId: PAD_ID,
      controlIds: CONTROL.map((p) => ids.get(p)!),
      eosId: IM_END_ID,
      vocabSize: vocab.length,
    }),
    tokenize: (text) => Int32Array.from(text, (c) => ids.get(c) ?? UNK_ID),
    piece: (tokenId) => vocab[tokenId],
    createContext: (jobId) => {
      if (contexts.has(jobId)) throw new Error(`context ${jobId} exists`);
      contexts.set(jobId, { ids: [], tape: [...tape] });
    },
    destroyContext: (jobId) => {
      contexts.delete(jobId);
    },
    length: (jobId) => ctxOf(jobId).ids.length,
    append: (jobId, newIds) => {
      ctxOf(jobId).ids.push(...Array.from(newIds, Number));
    },
    truncate: (jobId, n) => {
      ctxOf(jobId).ids.length = n;
    },
    fork: (parentId, childId) => {
      const parent = ctxOf(parentId);
      contexts.set(childId, { ids: [...parent.ids], tape: [...parent.tape] });
    },
    text: (jobId) => ctxOf(jobId).ids.map((i) => vocab[i]).join(''),
    decodeStep: (jobId, opts) => {
      const ctx = ctxOf(jobId);
      const tokenId = ctx.tape.shift();
      if (tokenId === undefined) throw new Error(`${jobId}: the script has no reply left`);
      if (opts.allowedTokens && !opts.allowedTokens[tokenId]) {
        throw new Error(`the mask refuses the script's next piece ${JSON.stringify(vocab[tokenId])}`);
      }
      const n = ctx.ids.length;
      const blocks = opts.allowedBlocks;
      const allowed = (i: number) => !blocks || blocks[i] !== 0;
      if (attention === 'none') return { tokenId, attention: null };
      const weights = new Float32Array(n);
      if (attention === 'first') {
        for (let i = 0; i < n; i++) {
          if (allowed(i)) {
            weights[i] = 1;
            break;
          }
        }
      } else {
        const picked: number[] = [];
        for (let i = n - 1; i >= 0; i--) {
          if (!allowed(i)) continue;
          picked.push(i);
          if (attention === 'recent' && picked.length === RECENT_WINDOW) break;
        }
        for (const i of picked) weights[i] = 1 / picked.length;
      }
      return { tokenId, attention: weights };
    },
  };
}
