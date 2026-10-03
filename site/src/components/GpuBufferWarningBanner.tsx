import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

import { CloseIcon } from './Icons';
import { detectWebGpu } from '../lib/localLlm/webgpu';

/**
 * ONNX Runtime's WebGPU backend binds each weight tensor as its own storage
 * buffer, so the binding limit has to fit the single largest tensor rather
 * than the whole model. For the q4f16 Gemma 4 exports that is the
 * per-layer-input embedding table, roughly 1.2–1.4 GB. Firefox currently pins
 * `maxBufferSize` at exactly 1 GiB, which is too small; Chrome and Edge expose
 * the full adapter limit (typically several GB).
 *
 * Provisional: the exact floor is to be finalised by the transformers.js
 * spike. Tune this single constant if the requirement changes.
 */
const REQUIRED_GPU_BUFFER_BYTES = 1.5 * 1024 ** 3;
const REQUIRED_LABEL = '1.5 GB';

// Above the tour overlay (SpotlightOverlay svg z 80, tour card z 84) and the
// compaction preview (z 95) so a "this browser can't run the app" warning is
// never obscured by app chrome.
const BANNER_Z_INDEX = 2000;

function formatGiB(bytes: number | undefined): string {
  if (typeof bytes !== 'number' || bytes <= 0) return 'none';
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

/**
 * Full-width bar warning that the current browser's WebGPU buffer limit is
 * too small to run Gemma Data Agent. Renders nothing until WebGPU detection
 * resolves, and nothing when the adapter reports a large enough buffer.
 */
export default function GpuBufferWarningBanner(): JSX.Element | null {
  const [effectiveLimit, setEffectiveLimit] = useState<number | undefined>(undefined);
  const [resolved, setResolved] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    detectWebGpu().then((status) => {
      if (cancelled) return;
      // A tensor must fit in one buffer AND one storage binding.
      setEffectiveLimit(
        Math.min(
          status.maxBufferSize ?? 0,
          status.maxStorageBufferBindingSize ?? Infinity,
        ),
      );
      setResolved(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Don't flash the banner before detection completes.
  if (!resolved) return null;
  if (dismissed) return null;

  const ok =
    typeof effectiveLimit === 'number' && effectiveLimit >= REQUIRED_GPU_BUFFER_BYTES;
  if (ok) return null;

  if (typeof document === 'undefined') return null;

  const barStyle: React.CSSProperties = {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    zIndex: BANNER_Z_INDEX,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 'var(--s-3, 12px)',
    // Extra right padding so the centred message never runs under the
    // absolutely-positioned close button.
    padding: 'var(--s-3, 12px) calc(var(--s-4, 16px) + 36px)',
    background: 'var(--danger-500, #e53935)',
    color: '#fff',
    fontFamily: 'var(--font-sans)',
    fontSize: '14px',
    lineHeight: 1.5,
    textAlign: 'center',
    boxShadow: 'var(--el-2, 0 2px 4px rgba(15, 20, 25, 0.05))',
    pointerEvents: 'auto',
  };

  return createPortal(
    <div style={barStyle} role="alert">
      <span aria-hidden="true" style={{ fontSize: '16px' }}>
        ⚠
      </span>
      <span>
        <strong>Gemma Data Agent won&rsquo;t run in this browser.</strong> Its
        GPU buffer limit is {formatGiB(effectiveLimit)}; the model&rsquo;s
        largest weight tensor needs at least {REQUIRED_LABEL} in one buffer. Use
        Google Chrome or Edge.
      </span>
      <button
        type="button"
        onClick={() => setDismissed(true)}
        aria-label="Dismiss"
        title="Dismiss"
        style={{
          position: 'absolute',
          top: '50%',
          right: 'var(--s-3, 12px)',
          transform: 'translateY(-50%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 4,
          border: 'none',
          background: 'transparent',
          color: 'inherit',
          cursor: 'pointer',
          borderRadius: 6,
          lineHeight: 0,
        }}
      >
        <CloseIcon size={20} />
      </button>
    </div>,
    document.body,
  );
}
