import { describe, expect, it } from 'vitest';
import { buildAgentSystemPrompt } from '../agentTools';
import runSqlMd from '../../prompts/agent/runSql.md?raw';
import zeosRunSqlMd from '../../prompts/zeos/runSql.md?raw';
import { dispatchForZeos, spellOutCallSkill, zeosAgentSystemPrompt, zeosSystemPrompt } from './zeosPrompt';

describe('zeosPrompt', () => {
  const FEATURE_SETS = [
    { runSql: true },
    { runSql: true, runPython: true, runReact: true, fileTools: true, dataLoading: true },
    { runSql: true, runPython: true, fileTools: true },
  ];

  it.each(FEATURE_SETS)('composes the inline-sql prompt for %j', (features) => {
    const p = zeosAgentSystemPrompt(features);
    expect(p).toContain(spellOutCallSkill(zeosRunSqlMd.trim()));
    expect(p).not.toContain(runSqlMd.trim());
    expect(p).toContain('RunSQL({"sql": "SELECT …"})');
    expect(p).not.toContain('`WriteLines` + `RunSQL(path)` workflow');
    expect(p).not.toMatch(/CallSkill\('/);
    expect(p).toContain('CallSkill({"skill":"sql"})');
    // base.md: RunSQL is not among the tools that take a path.
    expect(p).toContain('`RunSQL` is the exception: it takes its query inline as `sql`');
    expect(p).not.toMatch(/The execution tools \([^)]*`RunSQL`[^)]*\) take a `path`/);
    expect(p).not.toMatch(/\.\{[^}]*sql[^}]*\}/);
  });

  it('leaves the shared prompt unchanged for other models', () => {
    const shared = buildAgentSystemPrompt({ runSql: true, runPython: true });
    expect(shared).toContain(runSqlMd.trim());
    expect(shared).not.toContain('{{');
    expect(shared).not.toContain('is the exception');
    expect(shared).toMatch(/The execution tools \(`RunPython` and `RunSQL`\) take a `path`/);
  });

  it('rebuilds the agent prompt the chat sends, keeping what follows it', () => {
    const features = { runSql: true, runPython: true };
    const base = buildAgentSystemPrompt(features);
    expect(base).toContain("CallSkill('sql')");
    const tail = "\n\n## Earlier\nUse CallSkill('sql').";
    expect(zeosSystemPrompt(base + tail, features)).toBe(
      zeosAgentSystemPrompt(features) + '\n\n## Earlier\nUse CallSkill({"skill":"sql"}).',
    );
  });

  it('refuses the shared SQL section it cannot replace', () => {
    const other = buildAgentSystemPrompt({ runSql: true, runPython: true });
    expect(() => zeosSystemPrompt(other, { runSql: true })).toThrow(/shared RunSQL section/);
    expect(() => buildAgentSystemPrompt({}, { toolPrompts: { NoSuchTool: 'x' } })).toThrow(/NoSuchTool/);
  });

  it('leaves a prompt without the SQL section alone, apart from CallSkill', () => {
    expect(zeosSystemPrompt('plain text', { runSql: true })).toBe('plain text');
    expect(spellOutCallSkill("see CallSkill('data-loading') first")).toBe(
      'see CallSkill({"skill":"data-loading"}) first',
    );
  });

  it('answers the sql skill itself and rewrites other skill cards', async () => {
    const calls: string[] = [];
    const dispatch = async (name: string, args: unknown) => {
      calls.push(`${name}:${JSON.stringify(args)}`);
      return name === 'CallSkill' ? "Use CallSkill('sql') next." : { ok: true };
    };
    const sql = await dispatchForZeos('CallSkill', { skill: 'sql' }, dispatch);
    expect(sql).toContain('RunSQL(sql)');
    expect(calls).toEqual([]);
    expect(await dispatchForZeos('CallSkill', { skill: 'data-loading' }, dispatch)).toBe(
      'Use CallSkill({"skill":"sql"}) next.',
    );
    expect(await dispatchForZeos('ListInputs', {}, dispatch)).toEqual({ ok: true });
  });
});
