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
 *
 * `ZEOS_TRUSTED_RESULTS` is the other half of the policy: which results the
 * app wrote itself, so they arrive on `tools.results.trusted` (ring 2)
 * instead of `tools.results` (ring 3). The machine picks the pipe from the
 * call (`open_chat(trusted_results=…)`), and `streamZeos` delivers with
 * `trusted` from `isTrustedToolResult`; ZEOS refuses a delivery where the two
 * disagree.
 */
import { CALL_SKILL_NAMES, type AgentToolSpec } from '../agentTools';

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
 * Keywords that make a statement something other than a read wherever they
 * appear as a bare word in the SQL's code: DuckDB's DML, DDL, file and
 * extension access, settings and transactions, `EXPLAIN ANALYZE` of a write,
 * and functions with side effects. Inside a string literal, a quoted
 * identifier or a comment they are only text. A bare column or alias with one
 * of these names (`SELECT load FROM t`) is still an effect: a false "effect"
 * only asks the user, a false "read" would bypass them. `QUERY` catches the
 * `query()` table function.
 */
export const SQL_WRITE_KEYWORDS = [
  'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'UPSERT', 'CREATE', 'DROP', 'ALTER',
  'TRUNCATE', 'ATTACH', 'DETACH', 'COPY', 'EXPORT', 'IMPORT', 'INSTALL', 'LOAD',
  'FORCE', 'SET', 'RESET', 'CALL', 'EXECUTE', 'PREPARE', 'DEALLOCATE', 'VACUUM',
  'CHECKPOINT', 'BEGIN', 'COMMIT', 'ROLLBACK', 'ABORT', 'GRANT', 'REVOKE', 'USE',
  'INTO', 'QUERY', 'NEXTVAL', 'SETSEED',
] as const;

/**
 * Statements that only read. Not `PIVOT`: without an `IN` list DuckDB runs it
 * as a `CREATE TYPE` for the pivot values first.
 */
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

// The lexical units of a statement, as DuckDB splits them. Anything this does
// not recognise (a backslash, a `$`, an unterminated quote, a nested comment)
// makes the statement fail to match, so it is an effect.
/** A string literal, `''` escaping a quote. No backslash: in an `E'…'` string it escapes. */
const SQL_STRING = "'(?:[^'\\\\]|'')*'";
/** A quoted identifier, `""` escaping a quote. */
const SQL_QUOTED_ID = '"(?:[^"]|"")*"';
/**
 * A line comment, which (as in DuckDB's Postgres lexer) ends at `\n` or `\r`.
 * It must run to that end: a shorter match would let the rest of the line
 * open a string that hides the code after it.
 */
const SQL_LINE_COMMENT = '--[^\\n\\r]*(?![^\\n\\r])';
/** A block comment with no `/*` inside it (DuckDB may nest them; this never does). */
const SQL_BLOCK_COMMENT = '/\\*(?:[^*/]|\\*(?!/)|/(?!\\*))*\\*/';
/** A whole bare word (identifier, keyword or number) that is not a write keyword. */
const SQL_WORD =
  `${NOT_WORD_BEFORE}(?!(?:${SQL_WRITE_KEYWORDS.join('|')})${NOT_WORD_AFTER})` +
  `[A-Za-z0-9_]+${NOT_WORD_AFTER}`;
/** Any other character but `;`, a quote, `$` or a backslash; `-` and `/` when they open no comment. */
const SQL_OTHER = '[^;\'"A-Za-z0-9_$\\\\/-]|-(?!-)|/(?!\\*)';
const SQL_GAP = `(?:${WS}|${SQL_LINE_COMMENT}|${SQL_BLOCK_COMMENT})`;
const SQL_UNIT = `(?:${SQL_STRING}|${SQL_QUOTED_ID}|${SQL_LINE_COMMENT}|${SQL_BLOCK_COMMENT}|${SQL_WORD}|${SQL_OTHER})`;
/** As `SQL_UNIT`, without `=` or parentheses: a reporting PRAGMA's argument list. */
const SQL_PRAGMA_ARG_UNIT =
  `(?:${SQL_STRING}|${SQL_QUOTED_ID}|${SQL_LINE_COMMENT}|${SQL_BLOCK_COMMENT}|${SQL_WORD}|` +
  '[^;\'"A-Za-z0-9_$\\\\/=()-]|-(?!-)|/(?!\\*))';

/**
 * One read-only SQL statement: leading whitespace and comments, then a read
 * statement or a reporting PRAGMA, no `;` outside a string, quoted identifier
 * or comment except one at the end (followed only by whitespace and
 * comments), and no write keyword as a bare word.
 */
export const READ_ONLY_SQL_PATTERN =
  `${SQL_GAP}*(?:` +
  `(?:${READ_STATEMENTS.join('|')})${NOT_WORD_AFTER}${SQL_UNIT}*` +
  `|PRAGMA${SQL_GAP}+(?:${READ_PRAGMAS.join('|')})${NOT_WORD_AFTER}${SQL_GAP}*` +
  `(?:\\(${SQL_PRAGMA_ARG_UNIT}*\\))?${SQL_GAP}*` +
  `)(?:;${SQL_GAP}*)?`;

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
  return argsMatch(entry.read_if, args) ? 'read' : 'effect';
}

/** A rule as `open_chat(trusted_results=…)` takes it: args exactly these, each fully matching. */
export type TrustedResultRule = Record<string, string>;

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\-]/g, '\\$&');
}

/**
 * Results the app authored, delivered on ring 2. Only `CallSkill` with one of
 * the bundled skill names: its result is a reference card shipped with the
 * app (or, for a name that matches only case-insensitively, the app's own
 * "Unknown skill" error). Anything a tool fetched from data, files, code or
 * the network stays on ring 3.
 */
export const ZEOS_TRUSTED_RESULTS: Readonly<Record<string, TrustedResultRule>> = {
  CallSkill: { skill: `(?:${CALL_SKILL_NAMES.map(escapeRegExp).join('|')})` },
};

function argsMatch(rule: Readonly<Record<string, string>>, args: Record<string, unknown> | undefined): boolean {
  if (!args) return false;
  const keys = Object.keys(args).sort();
  const params = Object.keys(rule).sort();
  if (keys.length !== params.length || keys.some((k, i) => k !== params[i])) return false;
  for (const [param, pattern] of Object.entries(rule)) {
    const value = args[param];
    if (typeof value !== 'string') return false;
    if (!new RegExp(`^(?:${pattern})$`, 'is').test(value)) return false;
  }
  return true;
}

/**
 * Whether a call's result arrives on ring 2, by the rule the machine applies
 * (`ChatToolMachine.results_pipe` in ZEOS chat_machine.py).
 */
export function isTrustedToolResult(
  name: string,
  args: Record<string, unknown> | undefined,
  table: Readonly<Record<string, TrustedResultRule>> = ZEOS_TRUSTED_RESULTS,
): boolean {
  const rule = table[name];
  return rule !== undefined && argsMatch(rule, args);
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
