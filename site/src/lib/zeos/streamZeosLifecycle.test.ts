/**
 * The ZEOS engine's lifecycle around `streamZeos`: a crash after ready, Stop
 * while a kernel call is in flight (the grace period, and the watchdog
 * behind it), `disposeZeos` (a model switch), including mid-approval and
 * mid-start, a kernel that stops making progress, and a page that is not
 * cross-origin isolated. On every exit the turn's trust is saved (T5), a
 * pending approval card dies with the engine (T6), and the next run starts
 * demoted when the conversation was (T4).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AGENT_TOOLS } from '../agentTools';
import type { StreamChatMessage } from '../streamChat';
import { LOCAL_GEMMA_ENDPOINT, type LLMConfig } from '../../types/llm';
import type { ChatTrust } from '../../types/chat';
import type { ZeosChatEngine, ZeosChatRun, ZeosEvent } from './zeosChatEngine';
import { NotCrossOriginIsolatedError } from './zeosHost';
import {
  __setZeosEngineForTests,
  disposeZeos,
  isZeosLoaded,
  streamZeos,
  warmZeos,
  ZEOS_ABORT_WATCHDOG_MS,
  ZEOS_MAX_IDLE_STEPS,
  ZEOS_REFUSAL,
  ZEOS_STOP_GRACE_MS,
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

const listInputs: ZeosEvent[] = [
  { type: 'token', text: 'Checking.' },
  { type: 'tool_call', call: 0, name: 'ListInputs', arguments: {}, sink: 'tools.read', results: 'tools.results' },
];
/** The ListInputs result arrives on ring 3: the run now holds something untrusted. */
const arrived: ZeosEvent = { type: 'arrived', pipe: 'tools.results', segment: 7, ring: 3, integrity: 3 };
const demoted: ZeosEvent = {
  type: 'demoted',
  from_integrity: 2,
  to_integrity: 3,
  because: [
    {
      segment: 7,
      pipe: 'tools.results',
      principal: 'tool',
      tag: '',
      ring: 3,
      integrity: 3,
      tokens: 4,
      resident: true,
      injected_at: 0,
    },
  ],
};

/** A step that never answers until the test settles or the kernel interrupts it. */
const HANG = Symbol('hang');
/** A step that answers `events` after `ms` (a decode batch in progress). */
interface Slow {
  ms: number;
  events: ZeosEvent[];
}
type Step = ZeosEvent[] | typeof HANG | Slow;
const isSlow = (s: Step): s is Slow => typeof s === 'object' && !Array.isArray(s);

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

  call<T>(work: () => T | Promise<T>, hang = false, delayMs = 0): Promise<T> {
    if (this.disposed) return Promise.reject(this.disposed);
    return new Promise<T>((resolve, reject) => {
      const fail = (e: Error) => {
        this.pending.delete(fail);
        reject(e);
      };
      this.pending.add(fail);
      if (hang) return;
      (delayMs > 0 ? new Promise((r) => setTimeout(r, delayMs)) : Promise.resolve())
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
  startIntegrity: number | undefined;
  constructor(
    private readonly engine: FakeEngine,
    private readonly steps: Step[],
  ) {}
  importHistory(_turns: unknown, startIntegrity?: number) {
    this.log.push('importHistory');
    this.startIntegrity = startIntegrity;
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
    this.log.push(next === HANG ? 'step(hang)' : isSlow(next) ? 'step(slow)' : 'step');
    if (next === HANG) return this.engine.call(() => [] as ZeosEvent[], true);
    const events = isSlow(next) ? next.events : next;
    return this.engine.call(
      () => {
        for (const e of events) {
          if (e.type === 'waiting') this.waiting = e.pipe;
          if (e.type === 'approval_required' || e.type === 'tool_call') this.waiting = 'tools.results';
        }
        return events;
      },
      false,
      isSlow(next) ? next.ms : 0,
    );
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
  history: string;
  done: boolean;
  error: Error | null;
  trust: ChatTrust[];
}

function send(messages: StreamChatMessage[], signal?: AbortSignal): Promise<Outcome> {
  const out: Outcome = { ui: '', history: '', done: false, error: null, trust: [] };
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
      onHistoryDelta: (d) => {
        out.history += d;
      },
      onTrust: (t) => out.trust.push(t),
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

  /** Under fake timers: let the loop run until `cond`. */
  const tick = async (cond: () => boolean): Promise<void> => {
    for (let i = 0; i < 50 && !cond(); i++) await vi.advanceTimersByTimeAsync(0);
  };

  it('Stop interrupts a step still in flight after the grace period, and keeps a kernel that settles', async () => {
    vi.useFakeTimers();
    scripts = [[[HANG]]];
    const ctrl = new AbortController();
    const turn = send([{ role: 'user', content: 'hi' }], ctrl.signal);
    await tick(() => !!engines[0]?.runs[0]?.log.includes('step(hang)'));
    ctrl.abort();
    await vi.advanceTimersByTimeAsync(ZEOS_STOP_GRACE_MS - 1);
    expect(engines[0].interrupted).toBe(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(engines[0].interrupted).toBe(1);
    const out = await turn;
    expect(out.done).toBe(true);
    expect(out.error).toBeNull();
    // Nothing untrusted was in the run, so the lost batch could not have demoted it.
    expect(out.trust.at(-1)).toMatchObject({ integrity: 2 });
    // The ping settled, so the engine stays.
    await tick(() => engines[0].runs[0].log.includes('close'));
    expect(engines[0].runs[0].log).toContain('close');
    expect(engines[0].disposed).toBeNull();
    expect(isZeosLoaded()).toBe(true);
  });

  it('Stop waits for a decode batch in flight, so a demotion in it is saved with the message (T5)', async () => {
    scripts = [[[listInputs, { ms: 30, events: [arrived, demoted, { type: 'token', text: ' More' }] }]]];
    const ctrl = new AbortController();
    const turn = send([{ role: 'user', content: 'list' }], ctrl.signal);
    await until(() => !!engines[0]?.runs[0]?.log.includes('step(slow)'), 'the slow step');
    ctrl.abort();
    const out = await turn;
    expect(out.done).toBe(true);
    expect(engines[0].interrupted).toBe(0);
    expect(out.trust.at(-1)).toMatchObject({ integrity: 3, demotedBy: 'ListInputs result #1' });
  });

  it('Stop past the grace period saves the turn as demoted when the lost batch could have demoted it (T5)', async () => {
    vi.useFakeTimers();
    scripts = [[[listInputs, [arrived, { type: 'token', text: 'Reading' }], HANG]]];
    const ctrl = new AbortController();
    const turn = send([{ role: 'user', content: 'list' }], ctrl.signal);
    await tick(() => !!engines[0]?.runs[0]?.log.includes('step(hang)'));
    ctrl.abort();
    await vi.advanceTimersByTimeAsync(ZEOS_STOP_GRACE_MS);
    const out = await turn;
    expect(out.done).toBe(true);
    expect(engines[0].interrupted).toBe(1);
    expect(out.trust.at(-1)).toMatchObject({ integrity: 3, ring: 3, toolRings: [3] });
    expect(out.trust.at(-1)?.demotedBy).toContain('cut short');
  });

  it('Stop: a kernel that does not settle within the watchdog is reset, and the next message waits for it', async () => {
    vi.useFakeTimers();
    scripts = [[[HANG]], [[reply('Fresh.')]]];
    const ctrl = new AbortController();
    const turn = send([{ role: 'user', content: 'hi' }], ctrl.signal);
    await tick(() => !!engines[0]?.runs[0]?.log.includes('step(hang)'));
    engines[0].pingSettles = false;
    ctrl.abort();
    await vi.advanceTimersByTimeAsync(ZEOS_STOP_GRACE_MS);
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

  it('a crash mid-step after an untrusted read saves the turn demoted, and the replay starts demoted (T5, T4)', async () => {
    scripts = [[[listInputs, [arrived, { type: 'token', text: 'Reading' }], HANG]], [[reply('Again.')]]];
    const turn = send([{ role: 'user', content: 'list' }]);
    await until(() => !!engines[0]?.runs[0]?.log.includes('step(hang)'), 'the hanging step');
    engines[0].die(new Error('ZEOS model thread default crashed: GPU device lost'));
    const out = await turn;
    expect(out.error?.message).toContain('GPU device lost');
    const trust = out.trust.at(-1)!;
    expect(trust).toMatchObject({ integrity: 3, toolRings: [3] });

    // The chat stores the failed turn as its error text, and replays only its
    // demotion (mapMessagesForLLM): the fresh engine's run starts demoted.
    const again = await send([
      { role: 'user', content: 'list' },
      { role: 'assistant', content: '', trust },
      { role: 'user', content: 'again' },
    ]);
    expect(again.ui).toBe('Again.');
    expect(engines[1].runs[0].startIntegrity).toBe(3);
    expect(again.trust.at(-1)).toMatchObject({ integrity: 3 });
  });

  it('a crash while an approval waits ends the turn with the crash, not a denial, and kills the card (T6)', async () => {
    scripts = [[[listInputs, [arrived, demoted, approval]]], [[[approval], reply('Not saved.')]]];
    const turn = send([{ role: 'user', content: 'save a note' }]);
    await until(() => store.getSnapshot().pending !== null, 'the approval card');
    const staleId = store.getSnapshot().pending!.id;

    engines[0].die(new Error('ZEOS kernel worker error: out of memory'));
    expect(store.getSnapshot().pending).toBeNull();
    const out = await turn;
    expect(out.error?.message).toContain('out of memory');
    // The user never declined: no refusal reached the model, the UI or history.
    expect(engines[0].runs[0].log).not.toContain('deliverRefusal');
    expect(out.ui).not.toContain(ZEOS_REFUSAL);
    expect(out.history).not.toContain(ZEOS_REFUSAL);
    expect(out.trust.at(-1)).toMatchObject({ integrity: 3, demotedBy: 'ListInputs result #1' });

    // The next engine's card has a new id: the old card's late click does nothing.
    const next = send([
      { role: 'user', content: 'save a note' },
      { role: 'assistant', content: '', trust: out.trust.at(-1) },
      { role: 'user', content: 'again' },
    ]);
    await until(() => store.getSnapshot().pending !== null, 'the new card');
    const fresh = store.getSnapshot().pending!;
    expect(fresh.id).not.toBe(staleId);
    store.approve(staleId);
    store.deny(staleId);
    expect(store.getSnapshot().pending?.id).toBe(fresh.id);
    expect(engines[1].runs[0].log).not.toContain('deliverRefusal');
    store.deny(fresh.id);
    const done = await next;
    expect(engines[1].runs[0].log).toContain('deliverRefusal');
    expect(done.ui).toContain('Not saved.');
  });

  it('a kernel that stops making progress ends the turn and is restarted, instead of looping forever', async () => {
    // The script runs out with the job runnable: every later step returns nothing.
    scripts = [[[listInputs, [arrived]]], [[reply('Back.')]]];
    const out = await send([{ role: 'user', content: 'list' }]);
    expect(out.error?.message).toMatch(/no progress in \d+ steps/);
    expect(engines[0].runs[0].log.filter((l) => l === 'step').length).toBe(2 + ZEOS_MAX_IDLE_STEPS);
    expect(engines[0].disposed).not.toBeNull();
    expect(store.getSnapshot().status).toBe('error');
    expect(store.getSnapshot().error).toContain('next message restarts it');
    // Its trust is saved like any other exit's.
    expect(out.trust.at(-1)).toMatchObject({ toolRings: [3] });

    const again = await send([
      { role: 'user', content: 'list' },
      { role: 'assistant', content: '' },
      { role: 'user', content: 'again' },
    ]);
    expect(again.ui).toBe('Back.');
    expect(factory).toHaveBeenCalledTimes(2);
  });

  it('a job blocked on a delivery the loop never makes is a stall too', async () => {
    scripts = [[[[{ type: 'waiting', pipe: 'tools.results' }]]]];
    const out = await send([{ role: 'user', content: 'hi' }]);
    expect(out.error?.message).toMatch(/no progress .* waited on tools\.results/);
    expect(engines[0].disposed).not.toBeNull();
  });

  it('a page that is not cross-origin isolated reports it once per message, with no restart loop (D1)', async () => {
    const notIsolated = vi.fn(async (): Promise<ZeosChatEngine> => {
      throw new NotCrossOriginIsolatedError();
    });
    await __setZeosEngineForTests(notIsolated);
    const out = await send([{ role: 'user', content: 'hi' }]);
    expect(out.error).toBeInstanceOf(NotCrossOriginIsolatedError);
    expect(store.getSnapshot().status).toBe('error');
    expect(store.getSnapshot().error).toBe(out.error!.message);
    expect(store.getSnapshot().error).not.toContain('restarts it');
    expect(isZeosLoaded()).toBe(false);
    // Nothing retries on its own.
    await new Promise((r) => setTimeout(r, 20));
    expect(notIsolated).toHaveBeenCalledTimes(1);
    // The next message tries once more, and fails the same way.
    const again = await send([{ role: 'user', content: 'hi' }]);
    expect(again.error).toBeInstanceOf(NotCrossOriginIsolatedError);
    expect(notIsolated).toHaveBeenCalledTimes(2);
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
    // Unloading is not the user declining (T6), and the turn's trust is saved (T5).
    expect(engines[0].runs[0].log).not.toContain('deliverRefusal');
    expect(out.ui).not.toContain(ZEOS_REFUSAL);
    expect(out.trust.at(-1)).toMatchObject({ integrity: 2 });
  });

  it('disposeZeos mid-step after an untrusted read saves the turn as demoted (T5)', async () => {
    scripts = [[[listInputs, [arrived, { type: 'token', text: 'Reading' }], HANG]]];
    const turn = send([{ role: 'user', content: 'list' }]);
    await until(() => !!engines[0]?.runs[0]?.log.includes('step(hang)'), 'the hanging step');
    disposeZeos('switched to Gemma 4 E2B');
    const out = await turn;
    expect(out.error?.message).toContain('unloaded');
    expect(out.trust.at(-1)).toMatchObject({ integrity: 3, toolRings: [3] });
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
