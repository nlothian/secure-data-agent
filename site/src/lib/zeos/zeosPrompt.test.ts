import { describe, expect, it } from 'vitest';
import { buildAgentSystemPrompt } from '../agentTools';
import { dispatchForZeos, spellOutCallSkill, zeosSystemPrompt } from './zeosPrompt';

describe('zeosPrompt', () => {
  it('swaps the SQL section for the inline-sql one and spells out CallSkill', () => {
    const base = buildAgentSystemPrompt({ runSql: true });
    expect(base).toContain("CallSkill('sql')");
    const p = zeosSystemPrompt(base);
    expect(p).toContain('RunSQL({"sql": "SELECT …"})');
    expect(p).not.toContain('`WriteLines` + `RunSQL(path)` workflow');
    expect(p).not.toMatch(/CallSkill\('/);
    expect(p).toContain('CallSkill({"skill":"sql"})');
  });

  it('leaves a prompt without the SQL section alone, apart from CallSkill', () => {
    expect(zeosSystemPrompt('plain text')).toBe('plain text');
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
