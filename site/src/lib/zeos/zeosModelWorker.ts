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
 *   model-thread protocol (`ModelThreadMessage` in ./protocol.ts). Its init
 *   carries the export's source: `/models/<hfRepoId>/` with no cache in
 *   local-models mode, otherwise the model's pinned Hub revision
 *   (`hubSource`), kept in OPFS.
 */
import {
  isLocalModelsMode,
  isZeosStubMode,
  ZEOS_STUB_STORAGE_KEY,
  type LocalGemmaModel,
} from '../localLlm/models';
import { setLocalLlmDownloadProgress } from '../executionPanelStore';
import { createZeosLoadTracker } from './loadProgress';
import { hubUrl, type ModelCacheKey } from './vendor/model_cache.js';
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
 * Where the model thread reads the export (`init.source`): the directory URL,
 * and the `{repo, revision}` to keep its files under in OPFS, or null for no
 * cache. See ZEOS `model_cache.js`.
 */
export interface ZeosModelSource {
  url: string;
  cache: ModelCacheKey | null;
}

/**
 * Local-models mode reads `/models/<hfRepoId>/` from the dev server with no
 * cache (a missing file fails loudly, as for the transformers.js models);
 * otherwise the pinned Hub revision in `model.hubSource`, cached in OPFS.
 */
export function zeosModelSource(model: LocalGemmaModel): ZeosModelSource {
  if (isLocalModelsMode()) {
    return { url: `${import.meta.env.BASE_URL ?? '/'}models/${model.hfRepoId}/`, cache: null };
  }
  if (!model.hubSource) {
    throw new Error(
      `${model.label} has no Hub source, so it is only served from the local models/ folder: ` +
        'start the dev server with PUBLIC_LOCAL_MODELS=1 (and run `npm run models:fetch -- zeosq4b`). ' +
        `For the scripted stub, set localStorage['${ZEOS_STUB_STORAGE_KEY}'] to ` +
        '{"replies": [...], "attention": "recent"} and reload.',
    );
  }
  const { repo, revision } = model.hubSource;
  return { url: hubUrl({ repo, revision }), cache: { repo, revision } };
}

/** The OPT+ZEOS model thread (src/workers/zeosOptModel.worker.ts). */
function createRealModelThread(model: LocalGemmaModel): ZeosModelThread {
  const source = zeosModelSource(model);
  return {
    modelWorker: () =>
      new Worker(new URL('../../workers/zeosOptModel.worker.ts', import.meta.url), {
        type: 'module',
      }),
    init: { source },
    label: `${model.label} (OPT+ZEOS, WebGPU)`,
    stub: false,
  };
}

/**
 * Feed the model thread's `{progress}` messages to the Throbber, as
 * llmService does for the transformers.js models: one percentage over the
 * thread's own byte count (`bytes` / `bytes_total` cover each unique file
 * once; the tied embedding is read once for two manifest entries), worded
 * by phase (./loadProgress.ts: "Downloading", "Loading" from the OPFS cache
 * or the local models/ folder, "Verifying" a resumed download), then
 * "Loading … onto GPU" while ONNX Runtime builds the sessions.
 */
export function createZeosLoadProgress(model: LocalGemmaModel): {
  onProgress: (p: Record<string, unknown>) => void;
  done: () => void;
} {
  const tracker = createZeosLoadTracker({
    label: model.label,
    local: isLocalModelsMode(),
    onChange: setLocalLlmDownloadProgress,
  });
  return {
    onProgress: tracker.onProgress,
    done() {
      setLocalLlmDownloadProgress(null);
    },
  };
}

export function zeosModelThreadFor(model: LocalGemmaModel): ZeosModelThread {
  const script = readStubScript();
  return script ? createStubThread(script) : createRealModelThread(model);
}
