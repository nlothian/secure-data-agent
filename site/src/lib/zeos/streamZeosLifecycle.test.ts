/**
 * The ZEOS engine's lifecycle around `streamZeos`: a crash after ready, Stop
 * while a kernel call is in flight (and the watchdog behind it), and
 * `disposeZeos` (a model switch), including mid-approval and mid-start.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AGENT_TOOLS } from '../agentTools';
import type { StreamChatMessage } from '../streamChat';
import { LOCAL_GEMMA_ENDPOINT, type LLMConfig } from '../../types/llm';
import type { ZeosChatEngine, ZeosChatRun, ZeosEvent } from './zeosChatEngine';
import {
  __setZeosEngineForTests,
  disposeZeos,
  isZeosLoaded,
  streamZeos,
  warmZeos,
  ZEOS_ABORT_WATCHDOG_MS,
} from './streamZeos';
import * as store from './zeosSessionStore';

const CONFIG = {
  activeEndpoint: LOCAL_GEMMA_ENDPOINT,
  customEndpoints: [],
  apiKeys: {},
  models: { [LOCAL_GEMMA_ENDPOINT]: 'zeos-qwen3.5-4b' },
  thinkingEnabled: {},
} as LLMConfig;

const reply = (text: string): ZeosEvent[] => [
  { type: 'token', text },
  { type: 'reply', text, reasoning: null, raw: '' },
  { type: 'waiting', pipe: 'chat.user' },
];

const approval: ZeosEvent = {
  type: 'approval_required',
  call: 0,
  name: 'WriteLines',
  arguments: { path: '/scratchpad/n.txt', content: 'hi' },
  sink: 'tools.effect',
  results: 'tools.results',
  fault: 'privilege_fault',
  detail: 'floor 3',
  integrity: 2,
  effective_integrity: 3,
  session_floor: 3,
};

/** A step that never answers until the test settles or the kernel interrupts it. */
const HANG = Symbol('hang');
type Step = ZeosEvent[] | typeof HANG;

/**
 * A kernel-like engine: every call goes through `call`, which `interrupt`
 * and `die` reject, as ZeosKernel does for its pending requests.
 */
class FakeEngine implements ZeosChatEngine {
  backend = 'fake';
  stub = true;
  disposed: Error | null = null;
  interrupted = 0;
  opened = 0;
  runs: FakeRun[] = [];
  /** Resolves `ping` (the in-flight call finished); null pings never settle. */
  pingSettles = true;
  private pending = new Set<(e: Error) => void>();
  private listeners = new Set<(reason: Error) => void>();
  constructor(private readonly steps: Step[][]) {}

  call<T>(work: () => T | Promise<T>, hang = false): Promise<T> {
    if (this.disposed) return Promise.reject(this.disposed);
    return new Promise<T>((resolve, reject) => {
      const fail = (e: Error) => {
        this.pending.delete(fail);
        reject(e);
      };
      this.pending.add(fail);
      if (hang) return;
      Promise.resolve()
        .then(work)
        .then(
          (v) => {
            if (!this.pending.has(fail)) return;
            this.pending.delete(fail);
            resolve(v);
          },
          (e) => fail(e as Error),
        );
    });
  }

  async open() {
    this.opened += 1;
    const run = new FakeRun(this, this.steps.shift() ?? []);
    this.runs.push(run);
    return run;
  }
  dispose(reason = new Error('disposed')) {
    this.die(reason);
  }
  /** The worker crashed: pending calls reject, listeners run. */
  die(reason: Error) {
    if (this.disposed) return;
    this.disposed = reason;
    for (const fail of [...this.pending]) fail(reason);
    for (const l of [...this.listeners]) l(reason);
  }
  onDispose(listener: (reason: Error) => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
  interrupt(reason: Error) {
    this.interrupted += 1;
    for (const fail of [...this.pending]) fail(reason);
  }
  ping() {
    return this.call(() => undefined, !this.pingSettles);
  }
}

class FakeRun implements ZeosChatRun {
  log: string[] = [];
  waiting: string | null = 'chat.user';
  constructor(
    private readonly engine: FakeEngine,
    private readonly steps: Step[],
  ) {}
  importHistory() {
    this.log.push('importHistory');
    return this.engine.call(() => [] as ZeosEvent[]);
  }
  sendUser() {
    this.log.push('sendUser');
    return this.engine.call(() => {
      this.waiting = null;
    });
  }
  step() {
    const next = this.steps.shift() ?? [];
    this.log.push(next === HANG ? 'step(hang)' : 'step');
    if (next === HANG) return this.engine.call(() => [] as ZeosEvent[], true);
    return this.engine.call(() => {
      for (const e of next) {
        if (e.type === 'waiting') this.waiting = e.pipe;
        if (e.type === 'approval_required' || e.type === 'tool_call') this.waiting = 'tools.results';
      }
      return next;
    });
  }
  waitingOn() {
    return this.engine.call(() => this.waiting);
  }
  drain() {
    return this.engine.call(() => [] as string[]);
  }
  deliverToolResult() {
    this.log.push('deliverToolResult');
    return this.engine.call(() => {
      this.waiting = null;
    });
  }
  deliverRefusal() {
    this.log.push('deliverRefusal');
    return this.engine.call(() => {
      this.waiting = null;
    });
  }
  journalLines() {
    return this.engine.call(() => [] as string[]);
  }
  close() {
    this.log.push('close');
    return this.engine.call(() => undefined);
  }
}

interface Outcome {
  ui: string;
  done: boolean;
  error: Error | null;
}

function send(messages: StreamChatMessage[], signal?: AbortSignal): Promise<Outcome> {
  const out: Outcome = { ui: '', done: false, error: null };
  return new Promise<Outcome>((resolve) => {
    void streamZeos({
      config: CONFIG,
      messages: [{ role: 'system', content: 'You are a data agent.' }, ...messages],
      tools: AGENT_TOOLS,
      toolDispatcher: async () => 'ok',
      signal,
      onToken: (d) => {
        out.ui += d;
      },
      onDone: () => {
        out.done = true;
        resolve(out);
      },
      onError: (e) => {
        out.error = e;
        resolve(out);
      },
    });
  });
}

const until = async (cond: () => boolean, what: string): Promise<void> => {
  for (let i = 0; i < 200; i++) {
    if (cond()) return;
    await new Promise((r) => setTimeout(r, 1));
  }
  throw new Error(`timed out waiting for ${what}`);
};

describe('ZEOS engine lifecycle', () => {
  let engines: FakeEngine[];
  let scripts: Step[][][];
  const factory = vi.fn(async () => {
    const e = new FakeEngine(scripts.shift() ?? []);
    engines.push(e);
    return e;
  });

  beforeEach(async () => {
    store.resetForTests();
    engines = [];
    scripts = [];
    factory.mockClear();
    await __setZeosEngineForTests(factory);
  });
  afterEach(async () => {
    vi.useRealTimers();
    await __setZeosEngineForTests(null);
  });

  it('a crash mid-turn ends the turn with the error, says so, and the next message starts afresh', async () => {
    scripts = [[[HANG]], [[reply('Back.')]]];
    const turn = send([{ role: 'user', content: 'hi' }]);
    await until(() => engines[0]?.runs[0]?.log.includes('step(hang)'), 'the hanging step');
    expect(store.getSnapshot().status).toBe('ready');

    engines[0].die(new Error('ZEOS model thread default crashed: GPU device lost'));
    const out = await turn;
    expect(out.error?.message).toContain('GPU device lost');
    expect(store.getSnapshot().status).toBe('error');
    expect(store.getSnapshot().error).toContain('GPU device lost');
    expect(store.getSnapshot().error).toContain('next message restarts it');
    expect(isZeosLoaded()).toBe(false);

    const again = await send([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: '' },
      { role: 'user', content: 'again' },
    ]);
    expect(again.error).toBeNull();
    expect(again.ui).toBe('Back.');
    expect(factory).toHaveBeenCalledTimes(2);
    expect(store.getSnapshot().status).toBe('ready');
    // The new run got the history replayed.
    expect(engines[1].runs[0].log[0]).toBe('importHistory');
  });

  it('a crash while idle flips the status to error; the next message restarts the engine', async () => {
    scripts = [[[reply('One.')]], [[reply('Two.')]]];
    expect((await send([{ role: 'user', content: 'a' }])).ui).toBe('One.');
    engines[0].die(new Error('ZEOS kernel worker error: out of memory'));
    expect(store.getSnapshot().status).toBe('error');
    const out = await send([
      { role: 'user', content: 'a' },
      { role: 'assistant', content: 'One.' },
      { role: 'user', content: 'b' },
    ]);
    expect(out.ui).toBe('Two.');
    expect(factory).toHaveBeenCalledTimes(2);
  });

  it('Stop returns at once while a kernel call is in flight, and keeps a kernel that settles', async () => {
    scripts = [[[HANG]]];
    const ctrl = new AbortController();
    const turn = send([{ role: 'user', content: 'hi' }], ctrl.signal);
    await until(() => engines[0]?.runs[0]?.log.includes('step(hang)'), 'the hanging step');
    ctrl.abort();
    const out = await turn;
    expect(out.done).toBe(true);
    expect(out.error).toBeNull();
    expect(engines[0].interrupted).toBe(1);
    // The ping settled, so the engine stays.
    await until(() => engines[0].runs[0].log.includes('close'), 'the run to close');
    await new Promise((r) => setTimeout(r, 5));
    expect(engines[0].disposed).toBeNull();
    expect(isZeosLoaded()).toBe(true);
  });

  it('Stop: a kernel that does not settle within the watchdog is reset, and the next message waits for it', async () => {
    vi.useFakeTimers();
    scripts = [[[HANG]], [[reply('Fresh.')]]];
    const ctrl = new AbortController();
    const turn = send([{ role: 'user', content: 'hi' }], ctrl.signal);
    for (let i = 0; i < 50 && !engines[0]?.runs[0]?.log.includes('step(hang)'); i++) {
      await vi.advanceTimersByTimeAsync(0);
    }
    engines[0].pingSettles = false;
    ctrl.abort();
    expect((await turn).done).toBe(true);

    // A message sent right after Stop waits for the watchdog's verdict.
    const next = send([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: '' },
      { role: 'user', content: 'again' },
    ]);
    await vi.advanceTimersByTimeAsync(ZEOS_ABORT_WATCHDOG_MS - 1);
    expect(engines[0].disposed).toBeNull();
    expect(factory).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(2);
    expect(engines[0].disposed?.message).toContain('restarted after Stop');
    vi.useRealTimers();
    const out = await next;
    expect(out.ui).toBe('Fresh.');
    expect(factory).toHaveBeenCalledTimes(2);
  });

  it('disposeZeos mid-approval cancels the card, ends the turn and frees the engine', async () => {
    scripts = [[[[approval]]]];
    const turn = send([{ role: 'user', content: 'save a note' }]);
    await until(() => store.getSnapshot().pending !== null, 'the approval card');

    disposeZeos('switched to Gemma 4 E2B');
    expect(store.getSnapshot().pending).toBeNull();
    const out = await turn;
    expect(out.error?.message).toContain('ZEOS Qwen 4B was unloaded (switched to Gemma 4 E2B)');
    expect(engines[0].disposed).not.toBeNull();
    expect(isZeosLoaded()).toBe(false);
    expect(store.getSnapshot().status).toBe('idle');
    expect(store.getSnapshot().integrity).toBeNull();
  });

  it('disposeZeos while the engine starts disposes it once it arrives', async () => {
    let release!: (e: FakeEngine) => void;
    const late = new FakeEngine([]);
    await __setZeosEngineForTests(() => new Promise<ZeosChatEngine>((r) => (release = r)));
    const warming = warmZeos();
    expect(store.getSnapshot().status).toBe('starting');
    disposeZeos();
    expect(store.getSnapshot().status).toBe('idle');
    release(late);
    await expect(warming).rejects.toThrow(/unloaded/);
    expect(late.disposed).not.toBeNull();
    expect(isZeosLoaded()).toBe(false);
    expect(store.getSnapshot().status).toBe('idle');
  });

  it('disposeZeos with nothing loaded is a no-op', () => {
    disposeZeos();
    expect(store.getSnapshot().status).toBe('idle');
    expect(factory).not.toHaveBeenCalled();
  });
});
