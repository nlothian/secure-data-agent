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
import runSqlMd from '../../prompts/agent/runSql.md?raw';
import zeosRunSqlMd from '../../prompts/zeos/runSql.md?raw';
import zeosSqlSkillMd from '../../prompts/zeos/SqlSkill.md?raw';
import { classifyToolCall } from './zeosToolClasses';

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
 * The system prompt as this model gets it. The chat sends the shared agent
 * prompt for `features` plus anything after it (a compaction summary); that
 * prefix is rebuilt as `zeosAgentSystemPrompt`. Any other prompt is kept, with
 * CallSkill spelled out, unless it carries the shared RunSQL section: that
 * would be the agent prompt for other features, and the model would be told
 * the WriteLines + RunSQL(path) workflow, so it throws instead.
 */
export function zeosSystemPrompt(system: string, features: AgentPromptFeatures): string {
  const shared = buildAgentSystemPrompt(features);
  if (system.startsWith(shared)) {
    return zeosAgentSystemPrompt(features) + spellOutCallSkill(system.slice(shared.length));
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

/**
 * Run a tool call for this model: `CallSkill({"skill":"sql"})` returns the
 * inline-`sql` card, and every skill card has its `CallSkill('x')` spelled
 * out. A RunSQL the classifier calls a read (it ran with no approval) runs
 * under `guardRead`, DuckDB with extension autoloading off. Anything else
 * goes to `dispatch` unchanged.
 */
export async function dispatchForZeos(
  name: string,
  args: unknown,
  dispatch: (name: string, args: unknown) => Promise<unknown>,
  guardRead: ReadSqlGuard = guardWithDuckDb,
): Promise<unknown> {
  if (name === 'RunSQL') {
    const fields = args && typeof args === 'object' ? (args as Record<string, unknown>) : undefined;
    if (classifyToolCall(name, fields) === 'read') return guardRead(() => dispatch(name, args));
  }
  if (name !== 'CallSkill') return dispatch(name, args);
  if ((args as { skill?: unknown } | null)?.skill === 'sql') return spellOutCallSkill(zeosSqlSkillMd.trim());
  const result = await dispatch(name, args);
  return typeof result === 'string' ? spellOutCallSkill(result) : result;
}
