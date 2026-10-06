import { beforeEach, describe, expect, it, vi } from 'vitest';
import { AGENT_TOOLS } from '../agentTools';
import { repeatedCallNote } from '../repeatedToolCalls';
import { LOCAL_GEMMA_ENDPOINT, type LLMConfig } from '../../types/llm';
import { formatToolCallToken } from './toolPrompt';

// The worker is replaced by a script: each `generate` streams the next reply.
const llm = vi.hoisted(() => ({ replies: [] as string[], prompts: [] as string[] }));
vi.mock('./llmService', () => ({
  ensureLoaded: async () => {},
  generate: async (opts: { prompt: string; onToken: (d: string) => void; signal?: AbortSignal }) => {
    llm.prompts.push(opts.prompt);
    const reply = llm.replies.shift() ?? 'Done.';
    for (const piece of reply.match(/[\s\S]{1,4}/g) ?? []) {
      if (opts.signal?.aborted) break;
      opts.onToken(piece);
    }
    return reply;
  },
}));

const { streamLocalGemma } = await import('./streamLocalGemma');

const CONFIG = {
  activeEndpoint: LOCAL_GEMMA_ENDPOINT,
  customEndpoints: [],
  apiKeys: {},
  models: { [LOCAL_GEMMA_ENDPOINT]: 'gemma-4-e2b' },
  thinkingEnabled: {},
} as LLMConfig;

const call = (name: string, args: Record<string, unknown>) => formatToolCallToken(name, JSON.stringify(args));

async function run(replies: string[]) {
  llm.replies = [...replies];
  llm.prompts = [];
  const dispatched: [string, unknown][] = [];
  let ui = '';
  let maxed = false;
  await new Promise<void>((resolve, reject) => {
    void streamLocalGemma({
      config: CONFIG,
      messages: [
        { role: 'system', content: 'You are a data agent.' },
        { role: 'user', content: 'go' },
      ],
      tools: AGENT_TOOLS,
      toolDispatcher: async (name, input) => {
        dispatched.push([name, input]);
        return 'ok';
      },
      onToken: (d) => {
        ui += d;
      },
      onMaxIterationsReached: () => {
        maxed = true;
      },
      onDone: () => resolve(),
      onError: reject,
    });
  });
  return { dispatched, ui, maxed };
}

describe('streamLocalGemma: identical repeated calls', () => {
  beforeEach(() => {
    llm.replies = [];
  });

  it('runs a call once and answers an identical repeat with the note', async () => {
    const c = await run([call('ListInputs', {}), call('ListInputs', {}), 'Done.']);
    expect(c.dispatched).toEqual([['ListInputs', {}]]);
    expect(c.ui).toContain(`← ${repeatedCallNote('ListInputs')}`);
    expect(c.maxed).toBe(false);
  });

  it('compares arguments regardless of key order', async () => {
    const a = { path: '/input', from: 1 };
    const b = { from: 1, path: '/input' };
    const c = await run([call('ReadLines', a), call('ReadLines', b), 'Done.']);
    expect(c.dispatched).toHaveLength(1);
  });

  it('starts the record again after an effect runs', async () => {
    const w = { path: '/scratchpad/q.sql', content: 'SELECT 1' };
    const c = await run([
      call('ListInputs', {}),
      call('WriteLines', w),
      call('ListInputs', {}),
      call('ListInputs', {}),
      'Done.',
    ]);
    expect(c.dispatched.map(([n]) => n)).toEqual(['ListInputs', 'WriteLines', 'ListInputs']);
  });

  it('still counts a repeat as an iteration', async () => {
    const c = await run(Array.from({ length: 12 }, () => call('ListInputs', {})));
    expect(c.maxed).toBe(true);
    expect(c.dispatched).toHaveLength(1);
    expect(c.ui.endsWith('Reached max tool iterations')).toBe(true);
  });
});
