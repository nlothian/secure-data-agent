/**
 * Which in-browser engines stay loaded for the chat's model, and the switch
 * guard while a reply streams.
 *
 * Two engines can hold a model on the GPU: the transformers.js worker
 * (`llmService`: Gemma, Qwen 3.5) and the ZEOS kernel + model thread
 * (`streamZeos`: ZEOS Qwen 4B, ~2.8 GB). Switching models must free the one
 * the new model does not use:
 *
 * - **To ZEOS Qwen 4B**: the transformers.js worker is unloaded now if it
 *   holds anything but the side-task model (`sideTaskModelId`, `qwen3.5-4b`),
 *   and from then on whenever it has been idle for
 *   `SIDE_TASK_IDLE_UNLOAD_MS`. Side tasks (code summaries, the Explainer)
 *   cannot share the ZEOS model: it is a different export, run by
 *   OptZeosWorker in its own thread, and its single-conversation kernel run
 *   would have to be closed and re-prefilled (~30 s) for every side task (C4).
 *   So a side task loads plain `qwen3.5-4b` (~4 s from the OS cache), and both
 *   4B models sit on the GPU only while side tasks are running, plus the
 *   idle window, instead of for the rest of the session.
 * - **To any other model or endpoint**: `disposeZeos()` terminates the ZEOS
 *   workers. The transformers.js worker keeps its model, as before.
 *
 * Switching is refused while a chat reply streams (`modelSwitchBlockedReason`):
 * unloading the engine under a turn would end it with an error, or strand
 * its approval card.
 */
import { LOCAL_GEMMA_ENDPOINT, type LLMConfig } from '../../types/llm';
import { getLocalGemmaModel, resolveActiveLocalModelIdOrDefault, type LocalGemmaModel } from './models';

/** How long the side-task model stays loaded after its last use while ZEOS is the chat model. */
export const SIDE_TASK_IDLE_UNLOAD_MS = 60_000;

// ---- streams that block a switch ---------------------------------------------

const streams = new Map<number, string>();
let nextStream = 0;
const listeners = new Set<() => void>();
let snapshot: string | null = null;

function publish(): void {
  const first = streams.values().next();
  snapshot = first.done ? null : first.value;
  for (const l of listeners) l();
}

/**
 * Mark a reply as streaming (`what` names it, e.g. "the chat reply"); returns
 * the function that marks it done. Idempotent.
 */
export function beginModelStream(what: string): () => void {
  const id = ++nextStream;
  streams.set(id, what);
  publish();
  return () => {
    if (streams.delete(id)) publish();
  };
}

/** Why the model cannot be switched right now, or null when it can. */
export function modelSwitchBlockedReason(): string | null {
  return snapshot === null ? null : `Wait for ${snapshot} to finish, or stop it, before switching models.`;
}

export function subscribeModelSwitch(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getModelSwitchSnapshot(): string | null {
  return modelSwitchBlockedReason();
}

export function getModelSwitchServerSnapshot(): string | null {
  return null;
}

// ---- releasing engines ---------------------------------------------------------

/** The local model `config` chats with, or null for a cloud endpoint. */
export function activeLocalModel(config: LLMConfig): LocalGemmaModel | null {
  if (config.activeEndpoint !== LOCAL_GEMMA_ENDPOINT) return null;
  return getLocalGemmaModel(resolveActiveLocalModelIdOrDefault(config)) ?? null;
}

/**
 * Free whatever the chat's model (`model`, null for a cloud endpoint) does
 * not use. Safe to call repeatedly; it only imports the ZEOS module when it
 * may have been loaded.
 */
export async function releaseUnusedEngines(model: LocalGemmaModel | null): Promise<void> {
  const llm = await import('./llmService');
  if (model?.family === 'zeos-qwen') {
    llm.setIdleUnload(SIDE_TASK_IDLE_UNLOAD_MS);
    llm.unloadIfIdle(model.sideTaskModelId ?? null);
    return;
  }
  llm.setIdleUnload(null);
  if (zeosModuleLoaded) {
    const { disposeZeos } = await import('../zeos/streamZeos');
    disposeZeos(model ? `switched to ${model.label}` : 'switched to a cloud model');
  }
}

let zeosModuleLoaded = false;

/** Called by streamZeos when it loads, so a switch away knows to dispose it. */
export function noteZeosModuleLoaded(): void {
  zeosModuleLoaded = true;
}

/** Tests. */
export function __resetEngineLifecycleForTests(): void {
  streams.clear();
  snapshot = null;
  zeosModuleLoaded = false;
}
