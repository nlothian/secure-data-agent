/**
 * Prompt text this model sees differently from the others.
 *
 * - RunSQL takes its query inline as `sql` (so the ZEOS machine can classify
 *   a read-only query as a read; see zeosToolClasses.ts), so the system
 *   prompt's SQL section and the `sql` skill card describe that instead of
 *   the WriteLines + RunSQL(path) workflow. Otherwise the model writes every
 *   query to a file first, and WriteLines is an effect that needs approval
 *   once it has read tool output.
 * - base.md's "the execution tools take a `path`" paragraph leaves RunSQL
 *   out and says it takes inline SQL (`AgentPromptOptions.inlineSql`).
 * - The shared prompt writes skills as `CallSkill('sql')`. Qwen 3.5 4B then
 *   sometimes emits `<function=CallSkill('sql')>`, a tool name the machine
 *   does not know, so it is an effect and needs approval. Here they read
 *   `CallSkill({"skill":"sql"})`, like the prompt's other examples.
 */
import { buildAgentSystemPrompt, type AgentPromptFeatures, type AgentPromptOptions } from '../agentTools';
import { COMPACTION_HEADER } from '../autoCompaction';
import type { ChatMessage } from '../../types/chat';
import runSqlMd from '../../prompts/agent/runSql.md?raw';
import zeosRunSqlMd from '../../prompts/zeos/runSql.md?raw';
import zeosSqlSkillMd from '../../prompts/zeos/SqlSkill.md?raw';

const CALL_SKILL_SHORTHAND = /CallSkill\('([A-Za-z0-9_-]+)'\)/g;

/** `CallSkill('x')` → `CallSkill({"skill":"x"})`. */
export function spellOutCallSkill(text: string): string {
  return text.replace(CALL_SKILL_SHORTHAND, (_m, name: string) => `CallSkill({"skill":"${name}"})`);
}

/**
 * How this model's agent prompt differs: RunSQL's section is the inline-`sql`
 * one, and base.md says RunSQL takes inline SQL rather than a path. The
 * prompt is composed with these parts, not edited after the fact, so a change
 * to the shared prompt files cannot silently skip the swap.
 */
export const ZEOS_PROMPT_OPTIONS: AgentPromptOptions = {
  inlineSql: true,
  toolPrompts: { RunSQL: zeosRunSqlMd },
};

/** The agent system prompt for this model, built from `features`. */
export function zeosAgentSystemPrompt(features: AgentPromptFeatures): string {
  return spellOutCallSkill(buildAgentSystemPrompt(features, ZEOS_PROMPT_OPTIONS));
}

/**
 * What the chat says when a compacted conversation is sent to this model
 * (re-review finding N3).
 */
export const ZEOS_COMPACTED_REFUSAL =
  'This conversation was compacted under another model, so it cannot be continued with ' +
  'ZEOS Qwen 4B: the summary of the compacted turns would enter its trusted system prompt, ' +
  'and their trust record would be lost. Start a new chat to use ZEOS Qwen 4B, or switch ' +
  'back to the other model to continue this one.';

/** A compacted conversation reached this model; nothing was sent to it. */
export class ZeosCompactedConversationError extends Error {
  constructor() {
    super(ZEOS_COMPACTED_REFUSAL);
    this.name = 'ZeosCompactedConversationError';
  }
}

const COMPACTION_HEADING = COMPACTION_HEADER.trim();

/**
 * Whether a chat holds a compaction: a `kind: 'compaction'` message (the
 * summary another model wrote of the turns it dropped). Compaction is off
 * while this model is selected, but a chat compacted under another model can
 * be switched to it. Such a chat cannot run under ZEOS: the dropped turns are
 * not replayed (so a demotion among them is forgotten), and the summary,
 * written from untrusted content, would be appended to the system prompt,
 * which ZEOS treats as trusted.
 */
export function isCompactedConversation(messages: readonly Pick<ChatMessage, 'kind'>[]): boolean {
  return messages.some((m) => m.kind === 'compaction');
}

/** Whether a system prompt carries a compaction summary (`buildCompactionContext`'s heading). */
export function systemHasCompaction(system: string): boolean {
  return system.includes(COMPACTION_HEADING);
}

/**
 * The system prompt as this model gets it. The chat sends the shared agent
 * prompt for `features`, which is rebuilt as `zeosAgentSystemPrompt`. Nothing
 * may follow it: the only thing the chat ever appends is a compaction
 * summary, and a summary in any position throws
 * `ZeosCompactedConversationError`, so untrusted text can never reach the
 * trusted descriptor this way. Any other prompt is kept, with CallSkill
 * spelled out, unless it carries the shared RunSQL section: that would be the
 * agent prompt for other features, and the model would be told the
 * WriteLines + RunSQL(path) workflow, so it throws instead.
 */
export function zeosSystemPrompt(system: string, features: AgentPromptFeatures): string {
  if (systemHasCompaction(system)) throw new ZeosCompactedConversationError();
  const shared = buildAgentSystemPrompt(features);
  if (system.startsWith(shared)) {
    if (system.slice(shared.length).trim()) {
      throw new Error(
        'zeosSystemPrompt: text follows the agent prompt; ZEOS Qwen 4B takes the agent prompt alone.',
      );
    }
    return zeosAgentSystemPrompt(features);
  }
  if (system.includes(runSqlMd.trim())) {
    throw new Error(
      'zeosSystemPrompt: the system prompt has the shared RunSQL section but is not the agent ' +
        'prompt for these features, so the inline-sql version cannot be put in its place.',
    );
  }
  return spellOutCallSkill(system);
}

/** Run DuckDB work with extension autoloading off (`withoutExtensionAutoload`). */
export type ReadSqlGuard = <T>(fn: () => Promise<T>) => Promise<T>;

const guardWithDuckDb: ReadSqlGuard = async (fn) =>
  (await import('../duckdb')).withoutExtensionAutoload(fn);

/** Run DuckDB work that must not overlap a guarded read (`outsideExtensionAutoloadGuard`). */
const outsideGuardWithDuckDb: ReadSqlGuard = async (fn) =>
  (await import('../duckdb')).outsideExtensionAutoloadGuard(fn);

/**
 * How the kernel let a call run: from `tools.read` (no approval needed),
 * from `tools.effect` (the job was trusted enough to write it), or on the
 * user's approval after it refused (`ZeosToolLogEntry['how']`).
 */
export type ZeosCallHow = 'read' | 'effect' | 'approved';

/**
 * Run a tool call for this model: `CallSkill({"skill":"sql"})` returns the
 * inline-`sql` card, and every skill card has its `CallSkill('x')` spelled
 * out. A RunSQL the kernel ran as a read (`how === 'read'`: it landed on
 * `tools.read` with no approval) runs under `guardRead`, DuckDB with
 * extension autoloading off. That is keyed on what the kernel did, never on
 * classifying the SQL again here: where Python's `re` and `RegExp` disagree
 * (re-review finding N2), the kernel's verdict is the one that counted. Any
 * other RunSQL, and LoadData, runs under `outsideGuard`: never while a
 * guarded read is running, so it neither runs with autoloading switched off
 * under it nor changes the settings under a guarded query. Anything else
 * goes to `dispatch` unchanged.
 */
export async function dispatchForZeos(
  name: string,
  args: unknown,
  how: ZeosCallHow,
  dispatch: (name: string, args: unknown) => Promise<unknown>,
  guardRead: ReadSqlGuard = guardWithDuckDb,
  outsideGuard: ReadSqlGuard = outsideGuardWithDuckDb,
): Promise<unknown> {
  if (name === 'RunSQL' && how === 'read') return guardRead(() => dispatch(name, args));
  if (name === 'RunSQL' || name === 'LoadData') return outsideGuard(() => dispatch(name, args));
  if (name !== 'CallSkill') return dispatch(name, args);
  if ((args as { skill?: unknown } | null)?.skill === 'sql') return spellOutCallSkill(zeosSqlSkillMd.trim());
  const result = await dispatch(name, args);
  return typeof result === 'string' ? spellOutCallSkill(result) : result;
}
