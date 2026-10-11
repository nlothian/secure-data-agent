import type { TourStage } from '../types';

const pickModel: TourStage = {
  id: 'pick-model',
  markdown: `Pick a model from the dropdown here. We'll use **ZEOS Qwen 4B**: Qwen 3.5 4B running under the ZEOS kernel, entirely in your browser with no server in the loop.

Click **Download** to fetch it (about 2.8 GB, once). It is kept in your browser's storage, so later visits load it from disk. If it is already downloaded, there is nothing to confirm.

You can move on while it downloads; the chat waits for the model to be ready.`,
  cutouts: ['chat.modelDropdown', 'chat.modelConfirm'],
  onEnter: [
    { action: 'toggleModelDropdown', params: { open: true } },
    // The download dialog shows where the open menu is, as after a click.
    { action: 'toggleModelDropdown', params: { open: false }, delayMs: 1500 },
    { action: 'selectModel', params: { modelId: 'zeos-qwen3.5-4b' } },
  ],
  next: 'manual',
};

export default pickModel;
