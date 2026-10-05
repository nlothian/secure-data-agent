/**
 * The ZEOS chat as `streamZeos` drives it: one kernel worker + one model
 * thread per page (`startZeos`), and a `ChatRun` (ZEOS
 * `zeos_coop_count_web.chat`) per conversation, reached through the kernel
 * worker's generic RPC. `streamZeos` only sees the `ZeosChatEngine` /
 * `ZeosChatRun` interfaces, so its tests swap in a scripted engine.
 */
import type { LocalGemmaModel } from '../localLlm/models';
import { modelRef, startZeos, type ZeosHandle, type ZeosKernel } from './zeosHost';
import { zeosModelThreadFor } from './zeosModelWorker';
import type { ToolClassEntry } from './zeosToolClasses';
import type { ZeosImportTurn } from './zeosHistory';

/** A segment as `ChatRun.segment_info` reports it. */
export interface ZeosSegment {
  segment: number;
  pipe: string;
  principal: string;
  tag: string;
  ring: number;
  integrity: number;
  tokens: number;
  resident: boolean;
  injected_at: number;
}

interface CallFields {
  call: number;
  name: string;
  arguments: Record<string, unknown>;
  sink: string;
}

interface RefusalFields extends CallFields {
  fault: string;
  detail: string;
  integrity: number;
  effective_integrity: number | null;
  session_floor: number | null;
}

/** `ChatRun.step` events (see ZEOS chat.py). */
export type ZeosEvent =
  | { type: 'token'; text: string }
  | ({ type: 'tool_call' } & CallFields)
  | ({ type: 'approval_required' } & RefusalFields)
  | ({ type: 'tool_refused' } & RefusalFields)
  | { type: 'reply'; text: string; reasoning: string | null; raw: string }
  | { type: 'arrived'; pipe: string; segment: number; ring: number; integrity: number }
  | { type: 'demoted'; from_integrity: number; to_integrity: number; because: ZeosSegment[] }
  | { type: 'spoof'; pipe: string | null; detail: string }
  | { type: 'fault'; fault: string; detail: string; pipe: string | null }
  | { type: 'waiting'; pipe: string };

/**
 * How a tool result gates effects (ZEOS `open_chat(gate_mode=…)`). `strict`:
 * reading one sets the session floor to 3 until the next user message, so
 * any effect after it needs approval. `attention`: only the integrity
 * watermark (measured attention past theta_read) refuses effects.
 */
export type ZeosGateMode = 'strict' | 'attention';

export interface ZeosChatOpenOptions {
  systemPrompt: string;
  gateMode: ZeosGateMode;
  toolClasses: Readonly<Record<string, ToolClassEntry>>;
  paramTypes: Record<string, Record<string, string>>;
  thinking: boolean;
}

export interface ZeosChatRun {
  importHistory(turns: readonly ZeosImportTurn[]): Promise<ZeosEvent[]>;
  sendUser(text: string): Promise<void>;
  step(ticks: number): Promise<ZeosEvent[]>;
  waitingOn(): Promise<string | null>;
  drain(pipe: string): Promise<string[]>;
  deliverToolResult(text: string): Promise<void>;
  deliverRefusal(text?: string): Promise<void>;
  journalLines(): Promise<string[]>;
  close(): Promise<void>;
}

export interface ZeosChatEngine {
  backend: string;
  stub: boolean;
  open(opts: ZeosChatOpenOptions): Promise<ZeosChatRun>;
}

class KernelChatRun implements ZeosChatRun {
  constructor(
    private readonly kernel: ZeosKernel,
    private readonly run: ZeosHandle,
  ) {}

  private m<T>(method: string, ...args: unknown[]): Promise<T> {
    return this.kernel.callMethod<T>(this.run, method, args);
  }

  importHistory(turns: readonly ZeosImportTurn[]) {
    return this.m<ZeosEvent[]>(
      'import_history',
      turns.map((t) =>
        t.role === 'assistant'
          ? { role: t.role, text: t.text, integrity: t.integrity ?? 3 }
          : { role: t.role, text: t.text },
      ),
    );
  }
  sendUser(text: string) {
    return this.m<void>('send_user', text);
  }
  step(ticks: number) {
    return this.m<ZeosEvent[]>('step', ticks);
  }
  waitingOn() {
    return this.m<string | null>('waiting_on');
  }
  drain(pipe: string) {
    return this.m<string[]>('drain', pipe);
  }
  deliverToolResult(text: string) {
    return this.m<void>('deliver_tool_result', text);
  }
  deliverRefusal(text?: string) {
    return text === undefined ? this.m<void>('deliver_refusal') : this.m<void>('deliver_refusal', text);
  }
  journalLines() {
    return this.m<string[]>('journal_lines');
  }
  async close() {
    try {
      await this.m<void>('close');
    } finally {
      await this.kernel.release(this.run).catch(() => undefined);
    }
  }
}

export interface StartEngineHooks {
  onStatus?: (text: string) => void;
  onProgress?: (progress: Record<string, unknown>) => void;
}

/** Boot the kernel worker and the model thread for `model`, in parallel. */
export async function startKernelChatEngine(
  model: LocalGemmaModel,
  hooks: StartEngineHooks = {},
): Promise<ZeosChatEngine> {
  const thread = zeosModelThreadFor(model);
  const { kernel, model: attached } = await startZeos({
    onStatus: hooks.onStatus,
    onLog: (stream, text) => {
      if (stream === 'stderr') console.warn('[zeos]', text);
    },
    model: {
      modelWorker: thread.modelWorker,
      init: thread.init,
      onProgress: hooks.onProgress,
    },
  });
  return {
    backend: attached.backend,
    stub: thread.stub,
    async open(opts) {
      const run = await kernel.call<ZeosHandle>(
        'zeos_coop_count_web.chat',
        'open_chat',
        [modelRef(attached.name)],
        {
          tool_classes: opts.toolClasses,
          system_prompt: opts.systemPrompt,
          thinking: opts.thinking,
          param_types: opts.paramTypes,
          gate_mode: opts.gateMode,
        },
      );
      return new KernelChatRun(kernel, run);
    },
  };
}
