/**
 * Integrity check for the files the ZEOS kernel worker installs: the wheels
 * `npm run zeos:sync` built, against the sha256 it wrote into
 * `public/zeos/manifest.json`.
 */
/** Throw unless `bytes` hash to `expected` (hex sha256). */
export async function verifySha256(bytes: Uint8Array, expected: string, label: string): Promise<void> {
  if (!/^[0-9a-f]{64}$/i.test(expected ?? '')) {
    throw new Error(`${label}: the manifest has no sha256 for it; re-run \`npm run zeos:sync\``);
  }
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as BufferSource));
  const actual = Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
  if (actual !== expected.toLowerCase()) {
    throw new Error(`${label}: sha256 ${actual} does not match the manifest's ${expected}`);
  }
}
