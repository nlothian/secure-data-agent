import { describe, it, expect } from 'vitest';
import dataPipeline, {
  PROMPT,
  TRAIN_CSV_URL_TEMPLATE,
} from '../stages/10-data-pipeline';
import { DEFAULT_TOUR } from '../stages';
import { ACTION_NAMES } from '../actions';

describe('data-pipeline stage', () => {
  it('is registered in DEFAULT_TOUR', () => {
    expect(DEFAULT_TOUR.stages).toContain(dataPipeline);
  });

  it('uses only known action names', () => {
    const valid = new Set<string>(ACTION_NAMES);
    for (const step of dataPipeline.onEnter ?? []) {
      expect(valid.has(step.action)).toBe(true);
    }
  });

  it('spotlights the conversation, the approval card and the execution panel', () => {
    expect(dataPipeline.cutouts).toEqual(
      expect.arrayContaining(['chat.conversation', 'chat.zeosApproval', 'exec.panel']),
    );
  });

  it('starts from a fresh chat by clearing first', () => {
    const first = (dataPipeline.onEnter ?? [])[0];
    expect(first?.action).toBe('newChat');
  });

  it('enables dataLoading and runPython, not runSubAgent (off under ZEOS), before sending', () => {
    const step = (dataPipeline.onEnter ?? []).find(
      (s) => s.action === 'setEnabledFeatures',
    );
    expect(step).toBeDefined();
    const features = (step!.params as { features: Record<string, boolean> }).features;
    expect(features.dataLoading).toBe(true);
    expect(features.runPython).toBe(true);
    expect(features.runSubAgent).toBe(false);
  });

  it('types the multi-step prompt and then presses play', () => {
    const steps = dataPipeline.onEnter ?? [];
    const typeIdx = steps.findIndex((s) => s.action === 'typeMessage');
    const playIdx = steps.findIndex((s) => s.action === 'pressPlayButton');
    expect(typeIdx).toBeGreaterThan(-1);
    expect(playIdx).toBeGreaterThan(typeIdx);
    const typed = (steps[typeIdx].params as { text: string }).text;
    expect(typed).toBe(PROMPT);
  });

  it('prompt names LoadData and RunPython, the CSV URL, and the Sex feature', () => {
    expect(PROMPT).toContain('LoadData');
    expect(PROMPT).toContain('RunPython');
    expect(PROMPT).not.toContain('RunSubAgent');
    expect(PROMPT).toContain(TRAIN_CSV_URL_TEMPLATE);
    expect(PROMPT).toContain('Sex');
    expect(PROMPT).toMatch(/linear regression/i);
  });

  it('uses a generous wait timeout for the long LLM run', () => {
    const wait = (dataPipeline.onEnter ?? []).find(
      (s) => s.action === 'waitForLlmIdle',
    );
    expect(wait).toBeDefined();
    const t = (wait!.params as { timeoutMs?: number }).timeoutMs;
    expect(typeof t).toBe('number');
    expect(t).toBeGreaterThanOrEqual(120000);
  });

  it('waits for the user to advance after the LLM finishes', () => {
    expect(dataPipeline.next).toBe('manual');
  });
});
