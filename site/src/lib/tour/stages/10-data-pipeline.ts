import { TOUR_DATA_ORIGIN_TOKEN } from '../actions';
import type { TourStage } from '../types';

// Served from site/public/tour-data/. The token is substituted at dispatch
// time — see TOUR_DATA_ORIGIN_TOKEN in ../actions for why it is a placeholder.
const TRAIN_CSV_URL_TEMPLATE = `${TOUR_DATA_ORIGIN_TOKEN}/tour-data/train.csv`;

const PROMPT = `Load the CSV at ${TRAIN_CSV_URL_TEMPLATE} as a table named \`train\` with the LoadData tool.

Then train a linear regression in Python (RunPython) on the \`train\` table, predicting Survived from the \`Sex\` feature only. Print the fitted coefficients and R² score.
`;

const dataPipeline: TourStage = {
  id: 'data-pipeline',
  markdown: `Now we'll run a two-step data pipeline: load a CSV, then train a linear regression on the \`Sex\` feature in Python.

The CSV comes from outside, so ZEOS treats what the model reads from it as **untrusted**. Once the model has read it, running Python needs your say-so: an approval card appears above the message box showing the exact code. Click **Approve** to run it, or **Deny** to refuse.

This is what stops a prompt injection hidden in data from making the agent act on its own.`,
  cutouts: ['chat.conversation', 'chat.zeosApproval', 'chat.throbber', 'exec.panel'],
  onEnter: [
    { action: 'newChat' },
    {
      action: 'setEnabledFeatures',
      params: {
        features: {
          dataLoading: true,
          runSql: false,
          runPython: true,
          runReact: false,
          runSubAgent: false,
          fileTools: true,
        },
      },
    },
    { action: 'typeMessage', params: { text: PROMPT }, delayMs: 200 },
    { action: 'pressPlayButton', delayMs: 600 },
    { action: 'waitForLlmIdle', params: { timeoutMs: 600000 } },
  ],
  next: 'manual',
};

export { TRAIN_CSV_URL_TEMPLATE, PROMPT };
export default dataPipeline;
