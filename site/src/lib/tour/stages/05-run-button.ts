import type { TourStage } from '../types';

const runButton: TourStage = {
  id: 'run-button',
  markdown:
    "Press **Play** next to Step to run the agent on your message. Here we'll press it for you. If the model isn't downloaded yet, you'll be asked to download it first.",
  cutouts: ['chat.playButton', 'chat.modelConfirm'],
  onEnter: [{ action: 'pressPlayButton', delayMs: 2000 }],
  next: 'auto-after-actions',
};

export default runButton;
