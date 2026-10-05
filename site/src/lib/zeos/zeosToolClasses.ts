/**
 * The ZEOS trust policy for the agent's tools: which calls only read and
 * which have side effects. It is the one place to change that policy.
 *
 * The table is handed to the ZEOS chat machine (`open_chat(tool_classes=…)`),
 * which picks the sink for each call the model writes: `tools.read`
 * (capability `min_integrity: 3`, open to a job that has read untrusted
 * content) or `tools.effect` (`min_integrity: 2`). The kernel's capability
 * check on `tools.effect` is the only gate; this module never approves a
 * call. The TypeScript classifier below runs the same patterns so the UI and
 * the tests see the verdict the machine reaches.
 *
 * `RunSQL` is classified by its arguments: a `read_if` rule makes a call a
 * read only when its arguments are exactly `{sql}` and the SQL is a single
 * read-only statement. The SQL must therefore be inline, so this model gets a
 * RunSQL spec with an `sql` parameter (`zeosAgentTools`); a call by `path`
 * cannot be inspected and is an effect.
 *
 * The patterns stay inside the syntax Python's `re` and JavaScript's
 * `RegExp` share (the machine compiles them with IGNORECASE | DOTALL, here
 * `is`), and use ASCII word boundaries written out, since `\b` differs.
 */
import type { AgentToolSpec } from '../agentTools';

export type ToolClass = 'read' | 'effect';

/** A rule as the ZEOS machine takes it: read iff args are exactly these, each fully matching. */
export interface ReadIfRule {
  read_if: Record<string, string>;
}

export type ToolClassEntry = ToolClass | ReadIfRule;

const WS = '[ \\t\\r\\n\\f\\v]';
const NOT_WORD_BEFORE = '(?<![A-Za-z0-9_])';
const NOT_WORD_AFTER = '(?![A-Za-z0-9_])';

/**
 * Keywords that make a statement something other than a read anywhere they
 * appear (including inside string literals and comments: a false "effect" only
 * asks the user, a false "read" would bypass them). Covers DuckDB's DML, DDL,
 * file and extension access, settings and transactions, and `EXPLAIN ANALYZE`
 * of a write. `QUERY` catches the `query()` table function.
 */
export const SQL_WRITE_KEYWORDS = [
  'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'UPSERT', 'CREATE', 'DROP', 'ALTER',
  'TRUNCATE', 'ATTACH', 'DETACH', 'COPY', 'EXPORT', 'IMPORT', 'INSTALL', 'LOAD',
  'FORCE', 'SET', 'RESET', 'CALL', 'EXECUTE', 'PREPARE', 'DEALLOCATE', 'VACUUM',
  'CHECKPOINT', 'BEGIN', 'COMMIT', 'ROLLBACK', 'ABORT', 'GRANT', 'REVOKE', 'USE',
  'INTO', 'QUERY',
] as const;

/** Statements that only read. */
const READ_STATEMENTS = [
  'SELECT', 'WITH', 'FROM', 'VALUES', 'TABLE', 'DESCRIBE', 'DESC', 'SHOW',
  'SUMMARIZE', 'EXPLAIN',
] as const;

/** PRAGMAs that only report. Any other PRAGMA (they set options) is an effect. */
const READ_PRAGMAS = [
  'table_info', 'show_tables_expanded', 'show_tables', 'show', 'database_list',
  'database_size', 'storage_info', 'version', 'functions', 'platform',
  'metadata_info',
] as const;

/**
 * One read-only SQL statement, optionally ending in `;`: a read statement or a
 * reporting PRAGMA, no other `;` (so no second statement), and no write
 * keyword anywhere.
 */
export const READ_ONLY_SQL_PATTERN =
  `(?!.*${NOT_WORD_BEFORE}(?:${SQL_WRITE_KEYWORDS.join('|')})${NOT_WORD_AFTER})` +
  `${WS}*(?:` +
  `(?:${READ_STATEMENTS.join('|')})${NOT_WORD_AFTER}[^;]*` +
  `|PRAGMA${WS}+(?:${READ_PRAGMAS.join('|')})${NOT_WORD_AFTER}${WS}*(?:\\([^;=]*\\))?${WS}*` +
  `);?${WS}*`;

const READ_ONLY_SQL_RE = new RegExp(`^(?:${READ_ONLY_SQL_PATTERN})$`, 'is');

export function isReadOnlySql(sql: string): boolean {
  return READ_ONLY_SQL_RE.test(sql);
}

/** The table `open_chat` receives. A tool it does not name is an effect. */
export const ZEOS_TOOL_CLASSES: Readonly<Record<string, ToolClassEntry>> = {
  ListFiles: 'read',
  ReadLines: 'read',
  ListInputs: 'read',
  CallSkill: 'read',
  RunSQL: { read_if: { sql: READ_ONLY_SQL_PATTERN } },
  WriteLines: 'effect',
  RunPython: 'effect',
  RunReact: 'effect',
  LoadData: 'effect',
  RunSubAgent: 'effect',
};

/**
 * The class the machine gives a call, by the same rule it applies
 * (`ChatToolMachine.tool_class` in ZEOS chat_machine.py).
 */
export function classifyToolCall(
  name: string,
  args: Record<string, unknown> | undefined,
  table: Readonly<Record<string, ToolClassEntry>> = ZEOS_TOOL_CLASSES,
): ToolClass {
  const entry = table[name];
  if (entry === undefined) return 'effect';
  if (typeof entry === 'string') return entry;
  if (!args) return 'effect';
  const keys = Object.keys(args).sort();
  const params = Object.keys(entry.read_if).sort();
  if (keys.length !== params.length || keys.some((k, i) => k !== params[i])) return 'effect';
  for (const [param, pattern] of Object.entries(entry.read_if)) {
    const value = args[param];
    if (typeof value !== 'string') return 'effect';
    if (!new RegExp(`^(?:${pattern})$`, 'is').test(value)) return 'effect';
  }
  return 'read';
}

/** Tools this model does not get in v1. */
export const ZEOS_DISABLED_TOOLS: ReadonlySet<string> = new Set(['RunSubAgent']);

const RUN_SQL_ZEOS_NOTE =
  'With this model, pass the query inline as `sql` (one statement). A ' +
  'read-only query (SELECT, WITH, DESCRIBE, SHOW, EXPLAIN, SUMMARIZE, a ' +
  'reporting PRAGMA) given only as `sql` runs straight away; anything else, ' +
  'or a query given by `path` or with `register_as`, waits for the user to ' +
  'approve it once you have read tool output. ';

/**
 * The agent's tool specs as this model sees them: RunSubAgent removed, and
 * RunSQL taking its query inline as `sql` (so the machine can classify it).
 */
export function zeosAgentTools(tools: readonly AgentToolSpec[]): AgentToolSpec[] {
  return tools
    .filter((t) => !ZEOS_DISABLED_TOOLS.has(t.name))
    .map((t) => {
      if (t.name !== 'RunSQL') return t;
      const params = t.parameters as {
        properties?: Record<string, unknown>;
        required?: string[];
      };
      return {
        ...t,
        description: RUN_SQL_ZEOS_NOTE + t.description,
        parameters: {
          ...t.parameters,
          properties: {
            sql: {
              type: 'string',
              description: 'The SQL to run (DuckDB dialect). Use this instead of `path`.',
            },
            ...params.properties,
          },
          required: [],
        },
      };
    });
}

/** `param_types` for `open_chat`: each tool parameter's JSON-schema type. */
export function paramTypesFromTools(
  tools: readonly AgentToolSpec[],
): Record<string, Record<string, string>> {
  const out: Record<string, Record<string, string>> = {};
  for (const t of tools) {
    const props = (t.parameters as { properties?: Record<string, { type?: unknown }> }).properties;
    if (!props) continue;
    const types: Record<string, string> = {};
    for (const [k, v] of Object.entries(props)) {
      if (typeof v?.type === 'string') types[k] = v.type;
    }
    out[t.name] = types;
  }
  return out;
}
