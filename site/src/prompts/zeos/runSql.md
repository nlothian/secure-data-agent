# SQL queries

Use `RunSQL` for filtering, aggregating, joining, or any analysis you can express in SQL. Don't tell the user to run a SQL query — call `RunSQL` instead.

With this model `RunSQL` takes the query **inline** as `sql`: `RunSQL({"sql": "SELECT …"})`. Do not `WriteLines` a `.sql` file first. A single read-only statement (SELECT, WITH, DESCRIBE, SHOW, SUMMARIZE, EXPLAIN) passed only as `sql` runs straight away; anything that writes, or a query by `path`, needs the user's approval.

You **MUST** call `CallSkill('sql')` BEFORE your first `RunSQL` call. It documents the `_last_sql_result` / `arrow_inputs[...]` bridge, sample-row truncation, and the DuckDB error idioms.
