/**
 * Dev-only measurement of the attention the kernel acts on, for tuning
 * `theta_read` (see CLAUDE.md, "ZEOS Qwen 4B"). Off unless localStorage
 * `gda.zeos.attentionLog` is set before the engine starts.
 *
 * It installs a small Python module, `_zeos_attention_probe`, in the kernel
 * worker. `attach(run)` wraps the run's `ChatToolMachine.decode` to record
 * each step's measured mass per resident segment, and the kernel's
 * `demote_for_boundary` to record each block boundary's mass per segment and
 * the `theta_read` it was compared with: the number that actually demotes.
 * `take(run)` returns what was recorded since the last call. Nothing here
 * changes what the kernel decides.
 */
import type { ZeosHandle, ZeosKernel } from './zeosHost';

export interface AttentionSegmentInfo {
  pipe: string;
  ring: number;
  integrity: number;
  tokens: number;
}

export interface AttentionLog {
  /** One per decode step: the piece chosen and its mass per segment id. */
  steps: { tok: string; segs: Record<string, number> }[];
  /** One per block boundary: mass per segment over the block, and theta_read. */
  blocks: { segs: Record<string, number>; theta: number; demoted: boolean }[];
  /** Segment id -> what it is. */
  segments: Record<string, AttentionSegmentInfo>;
}

const PROBE_SOURCE = String.raw`
import sys, types
import zeos.core.kernel as _kernel_mod

_src = '''
import zeos.core.kernel as _kernel_mod

_logs = {}
_active = [None]
_orig_demote = _kernel_mod.demote_for_boundary


def _info(table, sid):
    rec = table.get(sid)
    return {
        "pipe": str(rec.provenance.pipe),
        "ring": int(rec.ring),
        "integrity": int(rec.integrity),
        "tokens": int(rec.tokens),
    }


def _demote(current, *, table, mass_this_block, theta_read):
    result = _orig_demote(
        current, table=table, mass_this_block=mass_this_block, theta_read=theta_read
    )
    log = _active[0]
    if log is not None:
        segs = {}
        for sid, mass in mass_this_block.items():
            if mass > 0 and sid in table:
                segs[str(int(sid))] = round(float(mass), 5)
                log["segments"].setdefault(str(int(sid)), _info(table, sid))
        log["blocks"].append(
            {"segs": segs, "theta": float(theta_read), "demoted": bool(result.moved)}
        )
    return result


_kernel_mod.demote_for_boundary = _demote


def attach(run):
    log = {"steps": [], "blocks": [], "segments": {}}
    _logs[id(run)] = log
    _active[0] = log
    machine = run.machine
    orig = machine.decode

    def decode(job, *, allow_control):
        r = orig(job, allow_control=allow_control)
        if r.attention is not None:
            kjob = run.job()
            segs = {}
            for rec in kjob.segments.resident():
                blocks = kjob.segments.blocks_for(rec)
                tot = sum(r.attention.get(b, 0.0) for b in blocks)
                if tot > 0:
                    segs[str(int(rec.id))] = round(float(tot), 5)
                    log["segments"].setdefault(str(int(rec.id)), _info(kjob.segments, rec.id))
            log["steps"].append({"tok": "".join(t.text for t in r.tokens), "segs": segs})
        return r

    machine.decode = decode


def take(run):
    log = _logs.get(id(run))
    if log is None:
        return {"steps": [], "blocks": [], "segments": {}}
    out = {"steps": log["steps"], "blocks": log["blocks"], "segments": dict(log["segments"])}
    log["steps"] = []
    log["blocks"] = []
    return out
'''

if "_zeos_attention_probe" not in sys.modules:
    _m = types.ModuleType("_zeos_attention_probe")
    exec(_src, _m.__dict__)
    sys.modules["_zeos_attention_probe"] = _m
True
`;

export async function installAttentionProbe(kernel: ZeosKernel): Promise<void> {
  await kernel.exec(PROBE_SOURCE);
}

export async function readAttentionLog(kernel: ZeosKernel, run: ZeosHandle): Promise<AttentionLog> {
  return kernel.call<AttentionLog>('_zeos_attention_probe', 'take', [run]);
}
