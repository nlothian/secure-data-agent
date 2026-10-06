import { useEffect, useRef, useState } from 'react';
import useLLMConfig, { isFirstVisitLLMConfig } from '../hooks/useLLMConfig';
import useLocalGemmaSwitcher from '../hooks/useLocalGemmaSwitcher';
import useModelSwitchBlocked from '../hooks/useModelSwitchBlocked';
import {
  canRunZeos,
  currentZeosCapabilities,
  defaultLocalModelId,
  formatGB,
  getLocalGemmaModel,
  isLocalGemmaId,
  LOCAL_GEMMA_MODELS,
} from '../lib/localLlm/models';
import { detectWebGpu, type WebGpuStatus } from '../lib/localLlm/webgpu';
import { isLocalGemmaEndpoint, LOCAL_GEMMA_ENDPOINT } from '../types/llm';
import { ChevronDownIcon } from './Icons';

export interface ModelSelectorProps {
  onModelMenuOpenChange?: (setter: (open: boolean) => void) => void;
  onRequestModelReady?: (fn: (id: string) => void) => void;
  onConfirmDownloadReady?: (fn: (id: string) => Promise<boolean>) => void;
}

export default function ModelSelector({
  onModelMenuOpenChange,
  onRequestModelReady,
  onConfirmDownloadReady,
}: ModelSelectorProps) {
  const { config, ready, setActiveEndpoint, setModel, setThinkingEnabled, setZeosAttentionOnly, setZeosMaskToolChoice } =
    useLLMConfig();
  const modelSwitcher = useLocalGemmaSwitcher({ loadOnApply: true });
  const switchBlocked = useModelSwitchBlocked();

  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const [gpuStatus, setGpuStatus] = useState<WebGpuStatus | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);

  // Normalise a missing or stale local model id (e.g. a `custom:<name>` entry
  // left in localStorage by the removed custom-model picker) to the default,
  // so the label, active-option highlight and the inference path all agree.
  // The default depends on WebGPU (`defaultLocalModelId`), so wait for that
  // check first. A first visit (no saved config at all) selects the local
  // endpoint and the default model too, where one can run (WebGPU, or the
  // ZEOS dev stub); nothing is downloaded until a send, which asks first
  // (`confirmDownload`).
  const storedLocalId = config.models[LOCAL_GEMMA_ENDPOINT];
  const firstVisit = isFirstVisitLLMConfig(config);
  useEffect(() => {
    if (!ready) return;
    if (!firstVisit) {
      if (config.activeEndpoint !== LOCAL_GEMMA_ENDPOINT) return;
      if (isLocalGemmaId(storedLocalId)) return;
    }
    let cancelled = false;
    void detectWebGpu().then((gpu) => {
      if (cancelled) return;
      if (firstVisit) {
        if (!gpu.supported && !canRunZeos(currentZeosCapabilities())) return;
        setActiveEndpoint(LOCAL_GEMMA_ENDPOINT);
      }
      setModel(LOCAL_GEMMA_ENDPOINT, defaultLocalModelId());
    });
    return () => {
      cancelled = true;
    };
  }, [ready, firstVisit, config.activeEndpoint, storedLocalId, setActiveEndpoint, setModel]);

  useEffect(() => {
    if (!modelMenuOpen) return;
    const onPointerDown = (e: PointerEvent) => {
      if (!modelMenuRef.current?.contains(e.target as Node)) {
        setModelMenuOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setModelMenuOpen(false);
    };
    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [modelMenuOpen]);

  // A reply started streaming: close the menu (its options are refused now).
  useEffect(() => {
    if (switchBlocked) setModelMenuOpen(false);
  }, [switchBlocked]);

  useEffect(() => {
    let cancelled = false;
    detectWebGpu().then((s) => {
      if (!cancelled) setGpuStatus(s);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    onModelMenuOpenChange?.(setModelMenuOpen);
  }, [onModelMenuOpenChange]);

  useEffect(() => {
    onRequestModelReady?.((id) =>
      modelSwitcher.request(
        id as Parameters<typeof modelSwitcher.request>[0],
      ),
    );
  }, [onRequestModelReady, modelSwitcher.request]);

  useEffect(() => {
    onConfirmDownloadReady?.((id) =>
      modelSwitcher.confirmDownload(
        id as Parameters<typeof modelSwitcher.confirmDownload>[0],
      ),
    );
  }, [onConfirmDownloadReady, modelSwitcher.confirmDownload]);

  const ep = config.activeEndpoint;
  const rawModel = ep ? config.models[ep] : '';
  const isLocal = ep ? isLocalGemmaEndpoint(ep) : false;
  const resolvedActive = isLocal ? getLocalGemmaModel(rawModel) : undefined;
  const labelText = isLocal
    ? resolvedActive?.label ?? 'Choose model'
    : rawModel || 'Choose model';
  const isEmpty = !ep || !rawModel || (isLocal && !resolvedActive);
  const webGpuSupported = gpuStatus?.supported === true;
  const webGpuReason = gpuStatus?.reason;
  const pendingConfirm = modelSwitcher.state.phase === 'confirm' ? modelSwitcher.state : null;

  return (
    <>
      <div className="chat-model-split" ref={modelMenuRef}>
        <span
          className={
            'chat-model chat-model-label' + (isEmpty ? ' chat-model-empty' : '')
          }
        >
          {labelText}
        </span>
        <button
          type="button"
          className="chat-iconbtn chat-model-menu-btn"
          data-tour-id="chat.modelDropdown"
          onClick={() => setModelMenuOpen((v) => !v)}
          title={
            !webGpuSupported
              ? webGpuReason ?? 'WebGPU is unavailable.'
              : switchBlocked ?? 'Choose local model'
          }
          aria-label="Choose local model"
          aria-haspopup="menu"
          aria-expanded={modelMenuOpen}
          disabled={!webGpuSupported || switchBlocked !== null}
          data-switch-blocked={switchBlocked !== null ? 'true' : undefined}
        >
          <ChevronDownIcon size={12} />
        </button>
        {modelMenuOpen && (
          <div
            className="chat-model-popover"
            role="menu"
            data-tour-id="chat.modelPopover"
          >
            {LOCAL_GEMMA_MODELS.map((m) => {
              const isActive =
                ep === LOCAL_GEMMA_ENDPOINT && rawModel === m.id;
              return (
                <button
                  key={m.id}
                  type="button"
                  role="menuitem"
                  title={m.notes}
                  className={
                    'chat-model-option' +
                    (isActive ? ' chat-model-option--active' : '')
                  }
                  onClick={() => {
                    modelSwitcher.request(m.id);
                    setModelMenuOpen(false);
                  }}
                >
                  <span className="chat-model-option-main">
                    <img
                      src={m.family === 'zeos-qwen' ? '/zeos-shield.svg' : '/gemma-color.svg'}
                      alt=""
                      aria-hidden="true"
                      className="chat-model-prefix"
                    />
                    <span className="chat-model-option-label">{m.label}</span>
                  </span>
                  <span className="chat-model-size">
                    {formatGB(m.approxBytes)}
                  </span>
                </button>
              );
            })}
            {isLocal && resolvedActive?.family === 'zeos-qwen' && (
              <label
                className="chat-model-popover-toggle"
                title={
                  'Hide untrusted tool output from the model while it writes the name of the ' +
                  'tool it calls, so a tool result cannot choose the next tool. The arguments ' +
                  'still see everything. Each tool call after a tool result costs extra ' +
                  'prefill. Applies from your next message.'
                }
              >
                <input
                  type="checkbox"
                  checked={config.zeosMaskToolChoice ?? false}
                  onChange={(e) => setZeosMaskToolChoice(e.target.checked)}
                  aria-label="Mask tool choice"
                />
                Mask tool choice
              </label>
            )}
          </div>
        )}
        {pendingConfirm && (
          <div className="chat-model-confirm" role="alert">
            <p className="chat-model-confirm-text">
              {pendingConfirm.model.label} is about{' '}
              {formatGB(pendingConfirm.bytes)} to download. It is fetched
              once from Hugging Face and cached in this browser.
            </p>
            <div className="chat-model-confirm-actions">
              <button
                type="button"
                className="chat-model-apply"
                onClick={modelSwitcher.apply}
              >
                {pendingConfirm.forSend ? 'Download' : 'Apply'}
              </button>
              <button
                type="button"
                className="chat-model-cancel"
                onClick={modelSwitcher.cancel}
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
      {ep && config.models[ep] && isLocalGemmaEndpoint(ep) && (
        <label className="chat-thinking-toggle">
          <input
            type="checkbox"
            checked={config.thinkingEnabled?.[LOCAL_GEMMA_ENDPOINT] ?? false}
            onChange={(e) =>
              setThinkingEnabled(LOCAL_GEMMA_ENDPOINT, e.target.checked)
            }
            aria-label="Enable thinking mode"
          />
          Thinking
        </label>
      )}
      {isLocal && resolvedActive?.family === 'zeos-qwen' && (
        <label
          className="chat-thinking-toggle"
          title={
            'Off (strict, default): once the model reads any tool output, side-effecting ' +
            'tools need your approval until your next message. On: they need approval ' +
            'only after the model measurably attends untrusted content. Applies from ' +
            'your next message.'
          }
        >
          <input
            type="checkbox"
            checked={config.zeosAttentionOnly ?? false}
            onChange={(e) => setZeosAttentionOnly(e.target.checked)}
            aria-label="Attention-only approval"
          />
          Attention-only approval
        </label>
      )}
    </>
  );
}
