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
 * read-only statement that reaches no file, URL or extension (it runs with
 * no approval, so it must not be a way to fetch or leak anything; see
 * `SQL_EXTERNAL_WORDS`, and `dispatchForZeos`, which runs every RunSQL the
 * kernel put on `tools.read` with DuckDB's extension autoloading off). The
 * SQL must therefore be inline, so this model gets a
 * RunSQL spec with an `sql` parameter (`zeosAgentTools`); a call by `path`
 * cannot be inspected and is an effect.
 *
 * The patterns stay inside the syntax Python's `re` and JavaScript's
 * `RegExp` share (the machine compiles them with IGNORECASE | DOTALL, here
 * `is`), and use ASCII word boundaries written out, since `\b` differs.
 * Without `re.ASCII`, Python's IGNORECASE still matches `ſ`, `K` (Kelvin),
 * `İ` and `ı` against `[A-Za-z]`, which `RegExp` without `u` never does;
 * ZEOS compiles `read_if` rules with `re.ASCII` so the two agree, and in any
 * case only the kernel's verdict decides what runs.
 *
 * `ZEOS_TRUSTED_RESULTS` is the other half of the policy: which results the
 * app wrote itself, so they arrive on `tools.results.trusted` (ring 2)
 * instead of `tools.results` (ring 3). The machine picks the pipe from the
 * call (`open_chat(trusted_results=…)`, an exact, case-sensitive match on the
 * skill name, not a pattern), and `streamZeos` delivers with
 * `trusted` from `isTrustedToolResult`; ZEOS refuses a delivery where the two
 * disagree.
 */
import { CALL_SKILL_NAMES, type AgentToolSpec } from '../agentTools';
import { LAST_SQL_RESULT_NAME, LLM_SAMPLE_ROWS } from '../duckdb';

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

/**
 * Table functions and other bare words that reach files, the network or
 * another database, as DuckDB names them: the `read_*` readers, `glob`,
 * `sniff_csv`, every `*_scan` (several of which also autoload an extension
 * over the network), the `parquet_*`, `iceberg_*`, `sqlite_*`, `postgres_*`
 * and `mysql_*` families, `st_read*` (`ST_ReadOSM`, …), `getenv`, functions
 * that run SQL from a value (`json_execute_serialized_sql`, `query_table`;
 * `query` is a write keyword), the secrets (`duckdb_secrets`, `which_secret`),
 * the table and pragma functions that change settings or logs (`enable_*`,
 * `disable_*`, `force_checkpoint`, `truncate_duckdb_logs`, `write_log`), and
 * `UESCAPE` (a `U&"…"` Unicode-escaped name could spell any of them; DuckDB
 * 1.4 does not implement those, and `&` before a quote is refused anyway).
 * A read-only query reads tables the
 * app already loaded (LoadData), so it needs none of them; a column that
 * happens to share one of these names makes the query an effect, which only
 * asks the user.
 */
export const SQL_EXTERNAL_WORDS = [
  'read_[A-Za-z0-9_]*', 'glob', 'sniff_csv', '[A-Za-z0-9_]*_scan',
  '(?:parquet|iceberg|sqlite|postgres|mysql)_[A-Za-z0-9_]*', 'st_read[A-Za-z0-9_]*',
  'getenv', 'load_aws_credentials', 'json_execute_serialized_sql', 'query_table',
  'duckdb_secrets', 'which_secret', '(?:enable|disable)_[A-Za-z0-9_]*', 'force_checkpoint',
  'truncate_duckdb_logs', 'write_log', 'UESCAPE',
] as const;

/**
 * File extensions DuckDB (or an extension it may load) reads a table name
 * ending in as a file: a replacement scan turns `FROM 'x.csv'`, `FROM
 * "x.parquet"` and `FROM x.csv` into a file read. Core DuckDB 1.4.3 does so
 * for csv, tsv, parquet, json, jsonl, ndjson and xlsx; the rest are formats
 * extensions read (avro, arrow, spatial's GDAL formats, sqlite, attached
 * databases) or common data files, listed so a later extension cannot turn
 * them into reads.
 */
const SQL_FILE_EXTENSIONS = [
  'csv', 'tsv', 'tbl', 'txt', 'dat', 'parquet', 'json', 'jsonl', 'ndjson', 'geojson', 'xlsx',
  'xls', 'arrow', 'ipc', 'feather', 'avro', 'orc', 'db', 'duckdb', 'sqlite', 'sqlite3', 'shp',
  'gpkg', 'fgb', 'kml', 'gpx', 'osm', 'pbf',
] as const;
/** Compression suffixes DuckDB accepts after a file extension (`x.csv.gz`). */
const SQL_COMPRESSION_EXTENSIONS = ['gz', 'gzip', 'zst', 'zstd', 'bz2', 'xz', 'lz4', 'snappy', 'zip'] as const;

/**
 * Extensions a string or quoted identifier must not end with: DuckDB reads
 * `FROM 'x.csv'` or `FROM "x.parquet"` as a file (a replacement scan), with
 * an optional compression suffix.
 */
const SQL_FILE_SUFFIX =
  `\\.(?:${SQL_FILE_EXTENSIONS.join('|')})` + `(?:\\.(?:${SQL_COMPRESSION_EXTENSIONS.join('|')}))?`;

/**
 * A `.` that qualifies a name must not be followed (past whitespace and
 * comments) by a part, bare or quoted, that is a file extension or a
 * compression suffix: DuckDB joins `catalog.schema.table` back into a path
 * for its replacement scans, so `FROM data.csv`, `FROM "data"."csv"`, `FROM
 * "sub/data".csv`, `FROM data."csv.gz"` and `FROM data.csv.gz` all read a
 * file. Ordinary qualified
 * names (`t.col`, `main.t`) stay reads; a column literally named `csv`
 * reached as `t.csv` is an effect, which only asks the user.
 */
const SQL_FILE_PART = [...SQL_FILE_EXTENSIONS, ...SQL_COMPRESSION_EXTENSIONS].join('|');

// The lexical units of a statement, as DuckDB splits them. Anything this does
// not recognise (a backslash, a `$`, an unterminated quote, a nested comment)
// makes the statement fail to match, so it is an effect.
/** Bare words that make a statement an effect: the write keywords and external access. */
const SQL_FORBIDDEN_WORDS = [...SQL_WRITE_KEYWORDS, ...SQL_EXTERNAL_WORDS].join('|');
/**
 * A line comment, which (as in DuckDB's Postgres lexer) ends at `\n` or `\r`.
 * It must run to that end: a shorter match would let the rest of the line
 * open a string that hides the code after it.
 */
const SQL_LINE_COMMENT = '--[^\\n\\r]*(?![^\\n\\r])';
/** A block comment with no `/*` inside it (DuckDB may nest them; this never does). */
const SQL_BLOCK_COMMENT = '/\\*(?:[^*/]|\\*(?!/)|/(?!\\*))*\\*/';
const SQL_GAP = `(?:${WS}|${SQL_LINE_COMMENT}|${SQL_BLOCK_COMMENT})`;
/**
 * A string literal, `''` escaping a quote. No backslash: in an `E'…'` string
 * it escapes. Not one that names a URL (`://` anywhere: `https://`, `s3://`,
 * `hf://`, …) or a file DuckDB would read in its place. It runs to its real
 * end (`'a''b'` is one string, never `'a'` and `'b'`), and no `&` comes
 * before it (`U&'…'` spells characters by escapes).
 */
const SQL_STRING =
  "(?<!&)'(?!(?:[^'\\\\]|'')*?://)(?!(?:[^'\\\\]|'')*" + SQL_FILE_SUFFIX + "'(?!'))(?:[^'\\\\]|'')*'(?!')";
/**
 * A quoted identifier, `""` escaping a quote; not a URL or a file name, as a
 * string, and likewise whole and with no `&` before it. DuckDB resolves a
 * quoted name like a bare one (`"glob"('*')`, `main."read_csv"(…)`,
 * `"GETENV"('HOME')` all run), so the name must not be a write keyword or an
 * external word either (compared whole and case-insensitively; `""` cannot
 * spell any of them). Nor one that spells a path or a glob (`/`, `\\`, `*`,
 * `?` or `[` anywhere): DuckDB looks for a file by that name (`FROM
 * "sub/x"`), and an alias like `"a/b"` is an effect too, which only asks the
 * user. And it must not be followed, past whitespace and comments, by `(`:
 * that calls it as a function, and a quoted name is not one the model needs
 * for a read.
 */
const SQL_QUOTED_ID =
  `(?<!&)"(?!(?:${SQL_FORBIDDEN_WORDS})"(?!"))` +
  '(?!(?:[^"]|"")*?[/\\\\*?\\[])(?!(?:[^"]|"")*' + SQL_FILE_SUFFIX + '"(?!"))(?:[^"]|"")*"(?!")' +
  `(?!${SQL_GAP}*\\()`;
/** A whole bare word (identifier, keyword or number) that is not a write keyword or external access. */
const SQL_WORD =
  `${NOT_WORD_BEFORE}(?!(?:${SQL_FORBIDDEN_WORDS})${NOT_WORD_AFTER})` +
  `[A-Za-z0-9_]+${NOT_WORD_AFTER}`;
/** A `.` not followed by a file extension as the next part of a name (see `SQL_FILE_PART`). */
const SQL_DOT = `\\.(?!${SQL_GAP}*(?:(?:${SQL_FILE_PART})${NOT_WORD_AFTER}|"(?:${SQL_FILE_PART})(?:"(?!")|\\.)))`;
/**
 * Any other character but `;`, a quote, `$` or a backslash; `-` and `/` when
 * they open no comment, `.` as `SQL_DOT`.
 */
const SQL_OTHER = `[^;'"A-Za-z0-9_$\\\\/.-]|-(?!-)|/(?!\\*)|${SQL_DOT}`;
const SQL_UNIT = `(?:${SQL_STRING}|${SQL_QUOTED_ID}|${SQL_LINE_COMMENT}|${SQL_BLOCK_COMMENT}|${SQL_WORD}|${SQL_OTHER})`;
/** As `SQL_UNIT`, without `=` or parentheses: a reporting PRAGMA's argument list. */
const SQL_PRAGMA_ARG_UNIT =
  `(?:${SQL_STRING}|${SQL_QUOTED_ID}|${SQL_LINE_COMMENT}|${SQL_BLOCK_COMMENT}|${SQL_WORD}|` +
  `[^;'"A-Za-z0-9_$\\\\/.=()-]|-(?!-)|/(?!\\*)|${SQL_DOT})`;

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

/**
 * A rule as `open_chat(trusted_results=…)` takes it: args exactly these keys,
 * each a string equal to one of the listed values. The comparison is exact and
 * case-sensitive, never a pattern (unlike `read_if`).
 */
export type TrustedResultRule = Record<string, readonly string[]>;

/**
 * Results the app authored, delivered on ring 2. Only `CallSkill` with a
 * bundled skill name spelled exactly: its result is a reference card shipped
 * with the app. `SQL`, ` sql` or `sql\n` is not a bundled name, so its result
 * (the "Unknown skill" error, which echoes the name) stays on ring 3, as does
 * anything a tool fetched from data, files, code or the network.
 */
export const ZEOS_TRUSTED_RESULTS: Readonly<Record<string, TrustedResultRule>> = {
  CallSkill: { skill: CALL_SKILL_NAMES },
};

function sameKeys(rule: object, args: Record<string, unknown>): boolean {
  const keys = Object.keys(args).sort();
  const params = Object.keys(rule).sort();
  return keys.length === params.length && keys.every((k, i) => k === params[i]);
}

function exactArgsMatch(rule: TrustedResultRule, args: Record<string, unknown> | undefined): boolean {
  if (!args || !sameKeys(rule, args)) return false;
  return Object.entries(rule).every(([param, values]) => {
    const value = args[param];
    return typeof value === 'string' && values.includes(value);
  });
}

function argsMatch(rule: Readonly<Record<string, string>>, args: Record<string, unknown> | undefined): boolean {
  if (!args || !sameKeys(rule, args)) return false;
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
  return rule !== undefined && exactArgsMatch(rule, args);
}

/** Tools this model does not get in v1. */
export const ZEOS_DISABLED_TOOLS: ReadonlySet<string> = new Set(['RunSubAgent']);

/**
 * RunSQL as this model sees it: the query inline as `sql`. It replaces the
 * shared description (and the `path` parameter's) rather than prefixing it,
 * since the shared text describes the WriteLines + RunSQL(path) workflow and
 * would contradict it.
 */
export const RUN_SQL_ZEOS_DESCRIPTION =
  'Execute one SQL statement in DuckDB-WASM. Pass the query inline as `sql`; ' +
  'do not write a .sql file. A read-only query (SELECT, WITH, DESCRIBE, SHOW, ' +
  'EXPLAIN, SUMMARIZE, a reporting PRAGMA) given only as `sql` runs straight ' +
  'away; anything else, or a query given by `path` or with `register_as`, or ' +
  "one that reads a file or URL itself (read_csv, FROM 'x.csv'), waits for the " +
  'user to approve it once you have read tool output. Query the loaded tables ' +
  'by name. On success returns { columns: [{name, type}], sample_rows: ' +
  'unknown[][], total_rows: number, registered_as: string }. ' +
  `\`sample_rows\` holds at most the first ${LLM_SAMPLE_ROWS} rows and ` +
  '`total_rows` the full count; when `sample_rows` holds all `total_rows` ' +
  'rows you have the whole result, so answer from it. Long string cells in ' +
  '`sample_rows` are truncated. On failure returns { error: string }: fix the ' +
  'query and call RunSQL again. The full result is published to the input ' +
  `registry as \`registered_as\` ("${LAST_SQL_RESULT_NAME}", overwritten by ` +
  'each call), for RunPython to read from `arrow_inputs`. Use SQL ' +
  'aggregations / WHERE / LIMIT for anything that needs more rows than ' +
  '`sample_rows`.';

const RUN_SQL_ZEOS_PATH =
  'Instead of `sql`: a .sql file under /scratchpad or /input. It always ' +
  'needs approval once you have read tool output, so prefer `sql`.';

/**
 * The agent's tool specs as this model sees them: RunSubAgent removed, and
 * RunSQL taking its query inline as `sql` (so the machine can classify it),
 * with a description that says so throughout (`RUN_SQL_ZEOS_DESCRIPTION`).
 */
export function zeosAgentTools(tools: readonly AgentToolSpec[]): AgentToolSpec[] {
  return tools
    .filter((t) => !ZEOS_DISABLED_TOOLS.has(t.name))
    .map((t) => {
      if (t.name !== 'RunSQL') return t;
      const params = t.parameters as {
        properties?: Record<string, Record<string, unknown>>;
        required?: string[];
      };
      const props = params.properties ?? {};
      return {
        ...t,
        description: RUN_SQL_ZEOS_DESCRIPTION,
        parameters: {
          ...t.parameters,
          properties: {
            sql: {
              type: 'string',
              description: 'The SQL to run (DuckDB dialect), one statement.',
            },
            ...props,
            ...(props.path ? { path: { ...props.path, description: RUN_SQL_ZEOS_PATH } } : {}),
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
