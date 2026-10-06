import { useCallback, useRef, useState } from 'react';
import useLLMConfig from './useLLMConfig';
import { LOCAL_GEMMA_ENDPOINT } from '../types/llm';
import {
  getLocalGemmaModel,
  type LocalGemmaId,
  type LocalGemmaModel,
} from '../lib/localLlm/models';
import { isModelCached, uncachedBytes } from '../lib/localLlm/modelCache';
import { hasDownloadConsent, noteDownloadConsent } from '../lib/localLlm/downloadConsent';
import { modelSwitchBlockedReason, releaseUnusedEngines } from '../lib/localLlm/engineLifecycle';

export type SwitcherState =
  | { phase: 'idle' }
  | { phase: 'checking'; modelId: LocalGemmaId }
  /**
   * `bytes`: what is still to download (a resumed download counts what it
   * holds). `forSend`: asked by `confirmDownload` for the active model before
   * a send, rather than by a switch.
   */
  | { phase: 'confirm'; model: LocalGemmaModel; bytes: number; forSend?: boolean };

export interface UseLocalGemmaSwitcherOptions {
  // When true, apply() also triggers ensureLoaded(modelId) so the download
  // starts immediately (chat-header dropdown). When false, apply() only
  // commits config; the caller is responsible for triggering the load
  // (Settings pane defers it to overlay close).
  loadOnApply: boolean;
}

export interface UseLocalGemmaSwitcherResult {
  state: SwitcherState;
  request: (modelId: LocalGemmaId) => void;
  apply: () => void;
  cancel: () => void;
  /**
   * Before a send with `modelId` active: resolves true at once when the model
   * is cached or its download was already agreed to in this page load,
   * otherwise shows the size dialog and resolves with the user's answer.
   */
  confirmDownload: (modelId: LocalGemmaId) => Promise<boolean>;
}

export default function useLocalGemmaSwitcher(
  opts: UseLocalGemmaSwitcherOptions,
): UseLocalGemmaSwitcherResult {
  const { config, setActiveEndpoint, setModel } = useLLMConfig();
  const [state, setState] = useState<SwitcherState>({ phase: 'idle' });
  // The pending `confirmDownload` promise's resolver, while its dialog shows.
  const sendAnswerRef = useRef<((ok: boolean) => void) | null>(null);
  const answerSend = useCallback((ok: boolean): void => {
    const resolve = sendAnswerRef.current;
    sendAnswerRef.current = null;
    resolve?.(ok);
  }, []);

  const commit = useCallback(
    (modelId: LocalGemmaId): void => {
      noteDownloadConsent(modelId);
      setActiveEndpoint(LOCAL_GEMMA_ENDPOINT);
      setModel(LOCAL_GEMMA_ENDPOINT, modelId);
      const model = getLocalGemmaModel(modelId) ?? null;
      // Free the engine the new model does not use before loading it, so the
      // two never sit on the GPU together.
      const released = releaseUnusedEngines(model).catch((err) =>
        console.error('Failed to unload the previous model:', err),
      );
      if (opts.loadOnApply) {
        void (async () => {
          try {
            await released;
            if (model?.family === 'zeos-qwen') {
              const { warmZeos } = await import('../lib/zeos/streamZeos');
              await warmZeos();
              return;
            }
            const { ensureLoaded } = await import('../lib/localLlm/llmService');
            await ensureLoaded(modelId);
          } catch (err) {
            console.error('Failed to load local Gemma model:', err);
          }
        })();
      }
    },
    [opts.loadOnApply, setActiveEndpoint, setModel],
  );

  const request = useCallback(
    (modelId: LocalGemmaId): void => {
      const model = getLocalGemmaModel(modelId);
      if (!model) return;
      const blocked = modelSwitchBlockedReason();
      if (blocked) {
        console.warn(`Model switch to ${model.label} refused: ${blocked}`);
        setState({ phase: 'idle' });
        return;
      }
      const isAlreadyActive =
        config.activeEndpoint === LOCAL_GEMMA_ENDPOINT &&
        config.models[LOCAL_GEMMA_ENDPOINT] === modelId;
      if (isAlreadyActive) {
        setState({ phase: 'idle' });
        return;
      }
      // A switch replaces a send's dialog: that send is declined.
      answerSend(false);
      setState({ phase: 'checking', modelId });
      void (async () => {
        try {
          const cached = await isModelCached(model);
          if (cached) {
            commit(modelId);
            setState({ phase: 'idle' });
          } else {
            setState({ phase: 'confirm', model, bytes: await uncachedBytes(model) });
          }
        } catch {
          // Fall back to confirm on cache-check failure so the user is still
          // warned about the download size before we hit the network.
          setState({ phase: 'confirm', model, bytes: model.approxBytes });
        }
      })();
    },
    [answerSend, commit, config.activeEndpoint, config.models],
  );

  const apply = useCallback((): void => {
    if (state.phase !== 'confirm') return;
    if (state.forSend) {
      // The model is already the active one; the send loads it.
      noteDownloadConsent(state.model.id);
      setState({ phase: 'idle' });
      answerSend(true);
      return;
    }
    if (modelSwitchBlockedReason()) return;
    const modelId = state.model.id;
    setState({ phase: 'idle' });
    commit(modelId);
  }, [answerSend, commit, state]);

  const cancel = useCallback((): void => {
    setState({ phase: 'idle' });
    answerSend(false);
  }, [answerSend]);

  const confirmDownload = useCallback(
    async (modelId: LocalGemmaId): Promise<boolean> => {
      if (hasDownloadConsent(modelId)) return true;
      const model = getLocalGemmaModel(modelId);
      if (!model) return true;
      let bytes = model.approxBytes;
      try {
        if (await isModelCached(model)) {
          noteDownloadConsent(modelId);
          return true;
        }
        bytes = await uncachedBytes(model);
      } catch {
        // Ask with the approximate size.
      }
      answerSend(false);
      return new Promise<boolean>((resolve) => {
        sendAnswerRef.current = resolve;
        setState({ phase: 'confirm', model, bytes, forSend: true });
      });
    },
    [answerSend],
  );

  return { state, request, apply, cancel, confirmDownload };
}
