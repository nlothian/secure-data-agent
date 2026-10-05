import { describe, expect, it } from 'vitest';
import { verifySha256 } from './verifySha256';

const bytes = new TextEncoder().encode('abc');
const ABC = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';

describe('verifySha256', () => {
  it('accepts matching bytes (either case)', async () => {
    await expect(verifySha256(bytes, ABC, 'w.whl')).resolves.toBeUndefined();
    await expect(verifySha256(bytes, ABC.toUpperCase(), 'w.whl')).resolves.toBeUndefined();
  });
  it('rejects a mismatch, naming the file', async () => {
    await expect(verifySha256(new TextEncoder().encode('abd'), ABC, 'w.whl')).rejects.toThrow(
      /^w\.whl: sha256 [0-9a-f]{64} does not match/,
    );
  });
  it('rejects a manifest without a hash', async () => {
    await expect(verifySha256(bytes, '', 'w.whl')).rejects.toThrow(/no sha256/);
  });
});
