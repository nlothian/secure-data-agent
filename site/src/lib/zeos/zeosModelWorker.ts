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
 * - **Real model**: `createRealModelThread` below, the OPT ZEOS model thread
 *   (src/workers/zeosOptModel.worker.ts over ZEOS `web/opt_zeos_worker.js`,
 *   vendored by `npm run zeos:sync`) on WebGPU. It answers the channel's
 *   `pieces` and `backend` calls and posts `{progress}` / `{ready}` per the
 *   model-thread protocol (`ModelThreadMessage` in ./protocol.ts).
 */
import {
  isLocalModelsMode,
  isZeosStubMode,
  ZEOS_STUB_STORAGE_KEY,
  type LocalGemmaModel,
} from '../localLlm/models';
import {
  createLoadProgressAggregator,
  type LoadProgressAggregator,
  type LoadProgressSnapshot,
} from '../localLlm/loadProgress';
import { setLocalLlmDownloadProgress } from '../executionPanelStore';
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

/**
 * The OPT+ZEOS model thread (src/workers/zeosOptModel.worker.ts), loading the
 * export from `/models/<hfRepoId>/`. Its files are not on the Hub, so this
 * needs local-models mode (`PUBLIC_LOCAL_MODELS=1`).
 */
function createRealModelThread(model: LocalGemmaModel): ZeosModelThread {
  if (!isLocalModelsMode()) {
    throw new Error(
      `${model.label} is only served from the local models/ folder: start the dev server ` +
        `with PUBLIC_LOCAL_MODELS=1 (and run \`npm run models:fetch -- zeosq4b\`). ` +
        `For the scripted stub, set localStorage['${ZEOS_STUB_STORAGE_KEY}'] to ` +
        '{"replies": [...], "attention": "recent"} and reload.',
    );
  }
  return {
    modelWorker: () =>
      new Worker(new URL('../../workers/zeosOptModel.worker.ts', import.meta.url), {
        type: 'module',
      }),
    init: { modelUrl: `${import.meta.env.BASE_URL ?? '/'}models/${model.hfRepoId}/` },
    label: `${model.label} (OPT+ZEOS, WebGPU)`,
    stub: false,
  };
}

/**
 * Feed the model thread's `{progress}` messages to the Throbber, as
 * llmService does for the transformers.js models: one aggregate over the
 * thread's own byte count (`bytes` / `bytes_total` cover each unique file
 * once; the tied embedding is read once for two manifest entries), then
 * "Loading … onto GPU" while ONNX Runtime builds the sessions.
 */
export function createZeosLoadProgress(model: LocalGemmaModel): {
  onProgress: (p: Record<string, unknown>) => void;
  done: () => void;
} {
  let agg: LoadProgressAggregator | null = null;
  const publish = (s: LoadProgressSnapshot) => setLocalLlmDownloadProgress(s);
  return {
    onProgress(p) {
      const total = typeof p.bytes_total === 'number' ? p.bytes_total : 0;
      if (!agg && total > 0) {
        agg = createLoadProgressAggregator({
          label: model.label,
          // Local-models mode: the files come from disk, not the network.
          fromCache: true,
          expectedFiles: [{ path: 'model', bytes: total }],
          onChange: publish,
        });
        publish(agg.snapshot());
      }
      if (!agg) return;
      if (p.phase === 'download' && typeof p.bytes === 'number') {
        agg.onEvent({ status: 'progress', file: 'model', loaded: p.bytes, total });
      } else if (p.phase === 'session') {
        agg.beginInit();
      }
    },
    done() {
      setLocalLlmDownloadProgress(null);
    },
  };
}

export function zeosModelThreadFor(model: LocalGemmaModel): ZeosModelThread {
  const script = readStubScript();
  return script ? createStubThread(script) : createRealModelThread(model);
}
