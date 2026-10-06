/**
 * ZeosKernel's failure handling with fake workers: interrupt (Stop), fatal
 * replies, a model thread that crashes after ready, and startZeos failing
 * fast on the first rejection.
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { isFatalKernelError, startZeos, startZeosKernel, ZeosKernel } from './zeosHost';

type Listener = ((event: { data: unknown }) => void) | null;

class FakeWorker {
  posted: Record<string, unknown>[] = [];
  terminated = false;
  onmessage: Listener = null;
  onerror: ((event: Partial<ErrorEvent> & { preventDefault(): void }) => void) | null = null;
  onmessageerror: (() => void) | null = null;
  postMessage(msg: Record<string, unknown>): void {
    this.posted.push(msg);
  }
  terminate(): void {
    this.terminated = true;
  }
  emit(data: unknown): void {
    this.onmessage?.({ data });
  }
  crash(message: string): void {
    this.onerror?.({ message, filename: 'worker.js', lineno: 1, preventDefault() {} });
  }
  /** Reply to the latest request with id. */
  reply(
    msg: { ok: true; value: unknown } | { ok: false; error: { name: string; message: string } },
    id = this.lastId(),
  ): void {
    this.emit({ type: 'reply', id, ...msg });
  }
  lastId(): number {
    return this.posted.at(-1)!.id as number;
  }
}

const asWorker = (w: FakeWorker) => w as unknown as Worker;
const tick = () => new Promise<void>((r) => setTimeout(r, 0));

function kernelWith(worker: FakeWorker): ZeosKernel {
  const k = new ZeosKernel(asWorker(worker), {});
  return k;
}

describe('zeosHost failure handling', () => {
  const g = globalThis as { crossOriginIsolated?: boolean; location?: unknown };
  const saved = { coi: g.crossOriginIsolated, location: g.location };
  beforeAll(() => {
    g.crossOriginIsolated = true;
    g.location ??= { href: 'http://localhost/' };
  });
  afterAll(() => {
    g.crossOriginIsolated = saved.coi;
    if (saved.location === undefined) delete g.location;
  });

  it('interrupt rejects pending calls at once, keeps the worker, and drops the late reply', async () => {
    const w = new FakeWorker();
    const k = kernelWith(w);
    const call = k.exec('slow()');
    const id = w.lastId();
    k.interrupt(new DOMException('Aborted', 'AbortError'));
    await expect(call).rejects.toThrow('Aborted');
    expect(w.terminated).toBe(false);
    expect(k.busy).toBe(false);
    w.reply({ ok: true, value: 1 }, id); // ignored
    const next = k.exec('None');
    w.reply({ ok: true, value: null });
    await expect(next).resolves.toBeNull();
  });

  it('a channel timeout is fatal: the call rejects and the kernel disposes itself', async () => {
    const w = new FakeWorker();
    const k = kernelWith(w);
    const reasons: Error[] = [];
    k.onDispose((r) => reasons.push(r));
    const call = k.exec('step()');
    w.reply({
      ok: false,
      error: { name: 'Error', message: 'model worker did not answer decodeStep within 180000 ms' },
    });
    await expect(call).rejects.toThrow('did not answer decodeStep');
    expect(w.terminated).toBe(true);
    expect(reasons).toHaveLength(1);
    expect(k.disposedReason?.message).toContain('did not answer');
    await expect(k.exec('None')).rejects.toThrow('did not answer');
  });

  it('an ordinary Python error is not fatal', async () => {
    const w = new FakeWorker();
    const k = kernelWith(w);
    const call = k.exec('1/0');
    w.reply({ ok: false, error: { name: 'ZeroDivisionError', message: 'division by zero' } });
    await expect(call).rejects.toThrow('ZeroDivisionError');
    expect(w.terminated).toBe(false);
    expect(isFatalKernelError({ name: 'PythonError', message: 'Pyodide already fatally failed' })).toBe(true);
    expect(
      isFatalKernelError({
        name: 'PythonError',
        message: 'model channel unusable: decodeStep (request 7) timed out after 180000 ms',
      }),
    ).toBe(true);
  });

  it('a kernel worker crash rejects the pending call and fires onDispose', async () => {
    const w = new FakeWorker();
    const k = kernelWith(w);
    const reasons: string[] = [];
    k.onDispose((r) => reasons.push(r.message));
    const call = k.exec('step()');
    w.crash('RangeError: out of memory');
    await expect(call).rejects.toThrow('out of memory');
    expect(reasons[0]).toContain('ZEOS kernel worker error');
  });

  it('a model thread that crashes after attach disposes the kernel, so the blocked call fails now', async () => {
    const w = new FakeWorker();
    const k = kernelWith(w);
    const thread = new FakeWorker();
    const attach = k.attachThread('default', {
      buffer: new SharedArrayBuffer(16),
      port: new MessageChannel().port2,
      backend: 'webgpu',
      thread: asWorker(thread),
    });
    w.reply({ ok: true, value: { name: 'default', backend: 'webgpu' } });
    await attach;
    const step = k.exec('step()');
    thread.crash('GPU device was lost');
    await expect(step).rejects.toThrow(/model thread default crashed: GPU device was lost/);
    expect(w.terminated).toBe(true);
    expect(thread.terminated).toBe(true);
  });

  it('a late {ready: false} from the model thread is a crash too', async () => {
    const w = new FakeWorker();
    const k = kernelWith(w);
    const thread = new FakeWorker();
    const seen: unknown[] = [];
    thread.onmessage = (e) => seen.push(e.data);
    const attach = k.attachThread('default', {
      buffer: new SharedArrayBuffer(16),
      port: new MessageChannel().port2,
      backend: 'webgpu',
      thread: asWorker(thread),
    });
    w.reply({ ok: true, value: {} });
    await attach;
    thread.emit({ activity: { ms: 1 } });
    expect(seen).toEqual([{ activity: { ms: 1 } }]);
    thread.emit({ ready: false, error: 'unhandled rejection: boom' });
    expect(k.disposedReason?.message).toContain('unhandled rejection: boom');
  });

  it('startZeos fails as soon as the kernel fails, without waiting for the model, and stops the model thread', async () => {
    const kernelWorker = new FakeWorker();
    const thread = new FakeWorker();
    const started = startZeos({
      manifestUrl: 'http://localhost/zeos/manifest.json',
      kernelWorker: () => asWorker(kernelWorker),
      model: { modelWorker: () => asWorker(thread) },
    });
    await tick();
    expect(thread.posted).toHaveLength(1); // the model thread got its init and is "loading"
    kernelWorker.reply({ ok: false, error: { name: 'Error', message: 'manifest.json: HTTP 404' } });
    await expect(started).rejects.toThrow('HTTP 404');
    expect(thread.terminated).toBe(true);
    expect(kernelWorker.terminated).toBe(true);
  });

  it('startZeos fails as soon as the model thread fails, and disposes the booting kernel', async () => {
    const kernelWorker = new FakeWorker();
    const thread = new FakeWorker();
    const started = startZeos({
      manifestUrl: 'http://localhost/zeos/manifest.json',
      kernelWorker: () => asWorker(kernelWorker),
      model: { modelWorker: () => asWorker(thread) },
    });
    await tick();
    thread.emit({ ready: false, error: 'no WebGPU adapter' });
    await expect(started).rejects.toThrow('no WebGPU adapter');
    expect(kernelWorker.terminated).toBe(true);
    expect(thread.terminated).toBe(true);
  });

  it('startZeosKernel boots through the fake worker', async () => {
    const w = new FakeWorker();
    const booting = startZeosKernel({ manifestUrl: 'http://localhost/m.json', kernelWorker: () => asWorker(w) });
    await tick();
    w.reply({ ok: true, value: { casesDir: '/zeos/cases' } });
    const k = await booting;
    expect(k.caseDir('x')).toBe('/zeos/cases/x');
  });
});
