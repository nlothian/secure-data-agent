import { beforeEach, describe, expect, it } from 'vitest';
import * as store from './zeosSessionStore';

const approval = (call: number, name = 'WriteLines'): store.ZeosApproval => ({
  call,
  name,
  args: { path: `/scratchpad/${call}.txt` },
  reason: 'strict: read tool output this turn',
  integrity: 2,
  sessionFloor: 3,
  effectiveIntegrity: 3,
});

describe('approval answers are bound to a call (T6)', () => {
  beforeEach(() => store.resetForTests());

  it('settles only the card whose id it names', async () => {
    const first = store.requestApproval(approval(0));
    const firstId = store.getSnapshot().pending!.id;
    store.approve(firstId + 1000);
    store.deny(firstId + 1000);
    expect(store.getSnapshot().pending?.id).toBe(firstId);
    store.approve(firstId);
    await expect(first).resolves.toBe(true);
    expect(store.getSnapshot().pending).toBeNull();
  });

  it('a double-click on Approve does not approve the next call', async () => {
    const first = store.requestApproval(approval(0));
    const firstId = store.getSnapshot().pending!.id;
    store.approve(firstId);
    await first;
    // The loop runs the call, and the model's next call needs approval too.
    const second = store.requestApproval(approval(1, 'RunPython'));
    const secondId = store.getSnapshot().pending!.id;
    expect(secondId).not.toBe(firstId);
    store.approve(firstId); // the double-click's second click, for the first card
    expect(store.getSnapshot().pending).toMatchObject({ id: secondId, name: 'RunPython' });
    store.deny(secondId);
    await expect(second).resolves.toBe(false);
  });

  it('ids never repeat, even for the same call index in a fresh run', async () => {
    const a = store.requestApproval(approval(0));
    const idA = store.getSnapshot().pending!.id;
    store.deny(idA);
    await a;
    store.resetConversation();
    const b = store.requestApproval(approval(0));
    const idB = store.getSnapshot().pending!.id;
    expect(idB).not.toBe(idA);
    store.approve(idA);
    expect(store.getSnapshot().pending?.id).toBe(idB);
    store.approve(idB);
    await expect(b).resolves.toBe(true);
  });
});
