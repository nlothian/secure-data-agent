import { useEffect, useRef, useState } from 'react';
import useLLMConfig from '../hooks/useLLMConfig';
import useLocalGemmaSwitcher from '../hooks/useLocalGemmaSwitcher';
import {
  DEFAULT_LOCAL_GEMMA_ID,
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
}

export default function ModelSelector({
  onModelMenuOpenChange,
  onRequestModelReady,
}: ModelSelectorProps) {
  const { config, ready, setModel, setThinkingEnabled } = useLLMConfig();
  const modelSwitcher = useLocalGemmaSwitcher({ loadOnApply: true });

  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const [gpuStatus, setGpuStatus] = useState<WebGpuStatus | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);

  // Normalise a stale local model id (e.g. a `custom:<name>` entry left in
  // localStorage by the removed custom-model picker) to the default, so the label,
  // active-option highlight and the inference path all agree.
  const storedLocalId = config.models[LOCAL_GEMMA_ENDPOINT];
  useEffect(() => {
    if (!ready) return;
    if (config.activeEndpoint !== LOCAL_GEMMA_ENDPOINT) return;
    if (isLocalGemmaId(storedLocalId)) return;
    setModel(LOCAL_GEMMA_ENDPOINT, DEFAULT_LOCAL_GEMMA_ID);
  }, [ready, config.activeEndpoint, storedLocalId, setModel]);

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
  const pendingConfirm =
    modelSwitcher.state.phase === 'confirm' ? modelSwitcher.state.model : null;

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
            webGpuSupported
              ? 'Choose local model'
              : webGpuReason ?? 'WebGPU is unavailable.'
          }
          aria-label="Choose local model"
          aria-haspopup="menu"
          aria-expanded={modelMenuOpen}
          disabled={!webGpuSupported}
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
                      src="/gemma-color.svg"
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
          </div>
        )}
        {pendingConfirm && (
          <div className="chat-model-confirm" role="alert">
            <p className="chat-model-confirm-text">
              {pendingConfirm.label} is about{' '}
              {formatGB(pendingConfirm.approxBytes)} to download. It is fetched
              once from Hugging Face and cached in this browser.
            </p>
            <div className="chat-model-confirm-actions">
              <button
                type="button"
                className="chat-model-apply"
                onClick={modelSwitcher.apply}
              >
                Apply
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
            aria-label="Enable Gemma thinking mode"
          />
          Thinking
        </label>
      )}
    </>
  );
}
