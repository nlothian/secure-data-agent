import type { TourStage } from '../types';

const PROMPT = 'Show me how the LoadData function runs';

const explainerSourcecode: TourStage = {
  id: 'explainer-sourcecode',
  markdown:
    "Now we'll use the **Explainer** to navigate the codebase. We'll maximize the Explainer pane and ask it about the LoadData function. The Explainer has its own LLM conversation with read-only `GrepCodebase` / `ReadLines` / `HighlightSourcecode` tools — answers cite real lines via `@sourcecode:` links.\n\nThe Explainer runs on plain Qwen 3.5 4B, since the ZEOS kernel keeps its one conversation for the chat. If that model isn't downloaded yet, its first answer waits for a second download of about 2.8 GB.",
  cutouts: ['exec.explainerPanel'],
  onEnter: [
    { action: 'pressExplainerExpand' },
    { action: 'typeExplainerMessage', params: { text: PROMPT }, delayMs: 300 },
    { action: 'sendExplainerMessage', delayMs: 400 },
    { action: 'waitForExplainerIdle', params: { timeoutMs: 600000 } },
  ],
  next: 'auto-after-actions',
};

export { PROMPT };
export default explainerSourcecode;
