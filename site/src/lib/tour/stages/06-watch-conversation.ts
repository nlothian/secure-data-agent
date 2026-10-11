import type { TourStage } from '../types';

const watchConversation: TourStage = {
  id: 'watch-conversation',
  markdown:
    `Watch the conversation as the agent replies. Note that you can inspect the system prompt in the drop-down at the top.

The first message takes longer: the model finishes downloading and loading onto your GPU, then reads the whole system prompt. Later turns start in a second or two.

We'll wait here while it streams — click **Next** when you're ready to move on.`,
  cutouts: ['chat.conversation', 'chat.throbber', 'chat.modelConfirm'],
  // The first send can include the model's download.
  onEnter: [{ action: 'waitForLlmIdle', params: { timeoutMs: 300000 } }],
  next: 'manual',
};

export default watchConversation;
