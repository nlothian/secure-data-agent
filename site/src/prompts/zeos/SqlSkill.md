# RunSQL reference card

## RunSQL(sql)

Executes one SQL statement against an in-browser DuckDB-WASM database. Pass the query **inline** as `sql`; do not write a `.sql` file:

```
→ RunSQL({"sql":"SELECT \"Sex\", COUNT(*) AS n FROM train GROUP BY \"Sex\""})
```

- One read-only statement (SELECT, WITH, DESCRIBE, SHOW, SUMMARIZE, EXPLAIN) runs straight away. Anything that writes (CREATE, INSERT, COPY, …), `register_as`, or a query by `path` waits for the user's approval once you have read tool output, so prefer a single SELECT.
- On success: `{ columns: [{name, type}], sample_rows: unknown[][], total_rows: number, registered_as: string }`.
- On failure: `{ error: string }`. Fix the query and call `RunSQL` again.
- **`sample_rows` holds at most the first 10 rows; `total_rows` is the full count.** When `sample_rows` holds all `total_rows` rows, you have the whole result: answer from it. Otherwise the user's UI panel shows up to 1000 rows and you don't, so use aggregations for anything that needs the full result.
- **The full Arrow result is always at `arrow_inputs[registered_as]`** — `registered_as` is always `"_last_sql_result"`, overwritten on the next `RunSQL` call.
- Long string cells in `sample_rows` are truncated with a `[truncated, full=N chars]` suffix.

## Querying loaded tables

**Discover before you query — don't guess names.** Call `ListInputs` to see what's loaded. Every `LoadData` of a tabular file (csv / json / parquet / xlsx) creates a DuckDB table named `table_name`, queryable directly (`SELECT * FROM foo`). Use the entry's `schema` for the exact column names.

- Use standard DuckDB SQL syntax.
- **ALWAYS quote column names** — e.g. `"PM10 BAM ug/m3"`; unquoted names with spaces or symbols fail to parse.

## Error symptom → cause

- `"exception_type":"Parser","exception_message":"syntax error at or near` → quote the column names in the query.
- `IO Error: No files found that match the pattern "X"` → you referenced a file path in SQL. RunSQL's DuckDB can't see sandbox files. `LoadData("X", "<table>")` first, then `SELECT … FROM <table>`.
