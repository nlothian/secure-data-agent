import { spawnSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';
import { AGENT_TOOLS } from '../agentTools';
import {
  READ_ONLY_SQL_PATTERN,
  ZEOS_TOOL_CLASSES,
  classifyToolCall,
  isReadOnlySql,
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
  "SELECT * FROM read_csv('/input/a.csv')",
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
  "SELECT ';'",
  'EXPLAIN ANALYZE CREATE TABLE t AS SELECT 1',
  'WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x',
  "SELECT * FROM query('CREATE TABLE t (a INT)')",
  "EXPORT DATABASE '/scratchpad/db'",
  'BEGIN TRANSACTION',
  '-- a comment\nSELECT 1',
  'SELECT * INTO t2 FROM t',
];

describe('isReadOnlySql', () => {
  it.each(READ_ONLY)('reads: %j', (sql) => {
    expect(isReadOnlySql(sql)).toBe(true);
  });
  it.each(NOT_READ_ONLY)('not read-only: %j', (sql) => {
    expect(isReadOnlySql(sql)).toBe(false);
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
// reaches the same verdicts, so the UI never disagrees with the kernel.
const python = spawnSync('python3', ['--version']).status === 0;
describe.skipIf(!python)('the pattern under Python re', () => {
  it('agrees with RegExp on every case', () => {
    const cases = [...READ_ONLY, ...NOT_READ_ONLY];
    const script =
      'import json, re, sys\n' +
      'data = json.load(sys.stdin)\n' +
      'p = re.compile(data["pattern"], re.IGNORECASE | re.DOTALL)\n' +
      'print(json.dumps([p.fullmatch(c) is not None for c in data["cases"]]))\n';
    const out = spawnSync('python3', ['-c', script], {
      input: JSON.stringify({ pattern: READ_ONLY_SQL_PATTERN, cases }),
      encoding: 'utf8',
    });
    expect(out.stderr).toBe('');
    expect(JSON.parse(out.stdout)).toEqual(cases.map(isReadOnlySql));
  });
});
