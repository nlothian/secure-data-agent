/**
 * The repeat-identical-call guard the agent loops share (`streamZeos`,
 * `streamLocalGemma`, the cloud loop in `streamChat`). Within one user
 * message, a tool call identical to one already made (same name, same
 * arguments up to key order) with no effect run in between is not run again:
 * the model gets `repeatedCallNote` as its result instead. Small models
 * otherwise loop on one call until the call cap, e.g. when a result that
 * looked like a refusal made them retry.
 *
 * Any effect that runs clears the record: WriteLines, LoadData or a CREATE can
 * change what a repeated read returns. The note is a constant: it never
 * quotes tool output, and it is not an `{error}` (the base prompt tells the
 * model to surface errors to the user).
 */

/** `value` as JSON with every object's keys sorted, so key order never matters. */
export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map((v) => (v === undefined ? 'null' : canonicalJson(v))).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`).join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
}

/** What makes two calls identical: the name and the canonical arguments (no SQL normalisation). */
export function toolCallKey(name: string, args: unknown): string {
  return `${name}\0${canonicalJson(args ?? {})}`;
}

/** The result a repeated call gets instead of running again (JSON). */
export function repeatedCallNote(name: string): string {
  return JSON.stringify({
    note: `You already ran ${name} with these exact arguments; its result is above. Answer from it, or make a different call.`,
  });
}

/** The calls made since the last effect, for one user message. */
export class RepeatedCallGuard {
  private readonly seen = new Set<string>();

  /** Whether this call was already made since the last effect ran. */
  isRepeat(name: string, args: unknown): boolean {
    return this.seen.has(toolCallKey(name, args));
  }

  /**
   * Record a call. `effectRan`: it was an effect and it ran, so what came
   * before no longer holds; the record restarts with this call alone. A call
   * that did not run (denied) is recorded without clearing anything.
   */
  record(name: string, args: unknown, effectRan = false): void {
    if (effectRan) this.seen.clear();
    this.seen.add(toolCallKey(name, args));
  }
}
