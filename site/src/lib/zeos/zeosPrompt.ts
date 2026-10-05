/**
 * Prompt text this model sees differently from the others.
 *
 * - RunSQL takes its query inline as `sql` (so the ZEOS machine can classify
 *   a read-only query as a read; see zeosToolClasses.ts), so the system
 *   prompt's SQL section and the `sql` skill card describe that instead of
 *   the WriteLines + RunSQL(path) workflow. Otherwise the model writes every
 *   query to a file first, and WriteLines is an effect that needs approval
 *   once it has read tool output.
 * - The shared prompt writes skills as `CallSkill('sql')`. Qwen 3.5 4B then
 *   sometimes emits `<function=CallSkill('sql')>`, a tool name the machine
 *   does not know, so it is an effect and needs approval. Here they read
 *   `CallSkill({"skill":"sql"})`, like the prompt's other examples.
 */
import runSqlMd from '../../prompts/agent/runSql.md?raw';
import zeosRunSqlMd from '../../prompts/zeos/runSql.md?raw';
import zeosSqlSkillMd from '../../prompts/zeos/SqlSkill.md?raw';

const CALL_SKILL_SHORTHAND = /CallSkill\('([A-Za-z0-9_-]+)'\)/g;

/** `CallSkill('x')` → `CallSkill({"skill":"x"})`. */
export function spellOutCallSkill(text: string): string {
  return text.replace(CALL_SKILL_SHORTHAND, (_m, name: string) => `CallSkill({"skill":"${name}"})`);
}

/** The agent system prompt as this model gets it (see the module comment). */
export function zeosSystemPrompt(system: string): string {
  const from = runSqlMd.trim();
  const swapped = system.includes(from) ? system.replace(from, zeosRunSqlMd.trim()) : system;
  return spellOutCallSkill(swapped);
}

/**
 * Run a tool call for this model: `CallSkill({"skill":"sql"})` returns the
 * inline-`sql` card, and every skill card has its `CallSkill('x')` spelled
 * out. Anything else goes to `dispatch` unchanged.
 */
export async function dispatchForZeos(
  name: string,
  args: unknown,
  dispatch: (name: string, args: unknown) => Promise<unknown>,
): Promise<unknown> {
  if (name !== 'CallSkill') return dispatch(name, args);
  if ((args as { skill?: unknown } | null)?.skill === 'sql') return spellOutCallSkill(zeosSqlSkillMd.trim());
  const result = await dispatch(name, args);
  return typeof result === 'string' ? spellOutCallSkill(result) : result;
}
