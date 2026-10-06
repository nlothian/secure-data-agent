export const CHAT_HISTORY_STORAGE_KEY = 'haw.chat.history.v1';

export type ChatRole = 'user' | 'assistant' | 'system';

/**
 * How far the ZEOS kernel trusted an assistant turn (ZEOS Qwen only). Rings
 * and integrities are ZEOS's: 0 most trusted, 2 TRUSTED (what the user
 * types), 3 EXTERNAL (tool output).
 */
export interface ChatTrust {
  /** The job's integrity when the turn ended; the turn is replayed at this ring. */
  integrity: number;
  /** The ring the turn's text is replayed on after a reload (`chat.history[.trusted]`). */
  ring: number;
  /**
   * The ring each tool result of the turn arrived on, in order: 3, or 2 for a
   * result the app wrote itself (a bundled skill card). History import
   * replays a result on ring 2 only when it is recorded here at 2.
   */
  toolRings?: number[];
  /** Indices into `toolRings` of results that spelled a kernel frame (ZEOS spoof alarm). */
  toolSpoofs?: number[];
  /**
   * Indices into `toolRings` of calls whose tool name the model chose with the
   * ring-3 tool output hidden (ZEOS mask on demand).
   */
  toolMasked?: number[];
  /** What demoted the job, if anything, e.g. "ReadLines result #3". */
  demotedBy?: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  /** UI-facing text. Includes thinking blocks and `→/←` tool-call markers. */
  content: string;
  /**
   * Replay-facing text used when this message is fed back to the model on a
   * later turn. For local-Gemma assistant turns this contains body text plus
   * proper `<|tool_call>` / `<|tool_response>` tokens — the format the model
   * was trained on. Absent on user messages and on cloud-API assistant turns
   * (where `content` is already the right shape).
   */
  historyContent?: string;
  /**
   * When set, this row replaces a span of older messages in the UI as a
   * collapsible "Compacted" block. `content` holds the summary text. The
   * marker is filtered out of `messages[]` at request time and its summary
   * is appended to the system prompt instead — see `sendPrompt`.
   */
  kind?: 'compaction';
  createdAt: number;
  error?: boolean;
  /**
   * Set on an assistant message when the agent loop stopped because the tool
   * iteration budget was exhausted. The UI renders a "Continue" button on
   * that bubble; clicking it submits "Continue" as a new user turn.
   */
  maxIterationsReached?: boolean;
  /** ZEOS Qwen assistant turns: the kernel's trust in the turn. */
  trust?: ChatTrust;
}

export interface ChatHistory {
  messages: ChatMessage[];
}

export const EMPTY_CHAT_HISTORY: ChatHistory = { messages: [] };
