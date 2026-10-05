/**
 * The single plug point for the ZEOS chat's model thread.
 *
 * `zeosModelThreadFor(model)` decides which model thread `streamZeos` starts
 * next to the kernel worker (`zeosHost.startZeos({ model })`):
 *
 * - **Stub mode** (dev only): localStorage `gda.zeos.stub` holds a
 *   `ChatStubScript` (`{replies, attention}`), and the scripted chat stub
 *   (src/workers/zeosChatStubModel.worker.ts) plays it. The e2e suite and
 *   manual UI work use this; no weights are loaded.
 * - **Real model**: `createRealModelThread` below. It is where the OPT ZEOS
 *   model thread (ZEOS `web/opt_zeos_worker.js`, vendored by
 *   `npm run zeos:sync`) gets wired: return its module worker and the `init`
 *   its first message needs. It must answer the channel's `pieces` and
 *   `backend` calls (`serveRequest` reads `meta.tokenizerSize` and
 *   `backend`) and post `{progress}` / `{ready}` per the model-thread
 *   protocol (`ModelThreadMessage` in ./protocol.ts).
 */
import {
  isLocalModelsMode,
  isZeosStubMode,
  ZEOS_STUB_STORAGE_KEY,
  type LocalGemmaModel,
} from '../localLlm/models';
import type { AttachModelOptions } from './zeosHost';
import type { ChatStubScript } from './scriptedChatModel';

export interface ZeosModelThread {
  modelWorker: AttachModelOptions['modelWorker'];
  init: Record<string, unknown>;
  /** Shown in the trust indicator's tooltip. */
  label: string;
  stub: boolean;
}

export function readStubScript(): ChatStubScript | null {
  if (!isZeosStubMode()) return null;
  try {
    const raw = localStorage.getItem(ZEOS_STUB_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<ChatStubScript>;
    return {
      replies: Array.isArray(parsed.replies) ? parsed.replies.map(String) : [],
      attention: parsed.attention ?? 'recent',
    };
  } catch (err) {
    console.warn(`${ZEOS_STUB_STORAGE_KEY} is not a valid stub script:`, err);
    return { replies: [], attention: 'recent' };
  }
}

function createStubThread(script: ChatStubScript): ZeosModelThread {
  return {
    modelWorker: () =>
      new Worker(new URL('../../workers/zeosChatStubModel.worker.ts', import.meta.url), {
        type: 'module',
      }),
    init: { script },
    label: 'scripted stub model',
    stub: true,
  };
}

/** B1/C1: wire the OPT ZEOS model thread here. */
function createRealModelThread(model: LocalGemmaModel): ZeosModelThread {
  throw new Error(
    `The ${model.label} model thread (ZEOS opt_zeos_worker.js) is not wired up yet. ` +
      (isLocalModelsMode()
        ? `Its files would be served from /models/${model.hfRepoId}/. `
        : 'It needs PUBLIC_LOCAL_MODELS=1. ') +
      `For the scripted stub, set localStorage['${ZEOS_STUB_STORAGE_KEY}'] to ` +
      '{"replies": [...], "attention": "recent"} and reload.',
  );
}

export function zeosModelThreadFor(model: LocalGemmaModel): ZeosModelThread {
  const script = readStubScript();
  return script ? createStubThread(script) : createRealModelThread(model);
}
