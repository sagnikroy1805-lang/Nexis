/** Static demo payloads for /api/drift and /api/models. */
import type { DriftReport, DriftStatus, DriftWindow, ModelLadder } from "../types";
import { MINUTE_MS, parseIso, round, toIso } from "./rng";
import { REFERENCE, REPLAY_END, REPLAY_START } from "./world";

const WARN = 0.1;
const DRIFT = 0.25;

function statusOf(...values: number[]): DriftStatus {
  const m = Math.max(...values);
  return m >= DRIFT ? "drift" : m >= WARN ? "warn" : "ok";
}

// Two-hour windows across the replay. Score PSI spikes at 04:10-06:10 and the
// threshold is recalibrated; feature PSI keeps rising through the afternoon.
const PSI_MAX = [0.03, 0.04, 0.05, 0.04, 0.06, 0.09, 0.13, 0.11, 0.12, 0.16, 0.21, 0.27, 0.31, 0.26, 0.22, 0.24];
const SCORE_PSI = [0.01, 0.02, 0.02, 0.03, 0.05, 0.11, 0.27, 0.07, 0.05, 0.06, 0.09, 0.12, 0.14, 0.1, 0.08, 0.09];
const ALERT_RATE = [
  0.0011, 0.0012, 0.0010, 0.0013, 0.0012, 0.0017, 0.0031, 0.0013, 0.0012, 0.0014, 0.0015, 0.0016, 0.0018,
  0.0015, 0.0013, 0.0014,
];
const TOP_FEATURES = ["src_out_cnt_24h", "log_amount", "dst_in_cnt_24h", "src_secs_since_in", "hour"];

export function driftReport(): DriftReport {
  const start = parseIso(REPLAY_START);
  const end = parseIso(REPLAY_END);
  const windows: DriftWindow[] = PSI_MAX.map((psi, i) => {
    const ws = start + i * 120 * MINUTE_MS;
    const we = Math.min(start + (i + 1) * 120 * MINUTE_MS, end);
    const lead = TOP_FEATURES[(i >= 9 ? 0 : i >= 5 ? 1 : i) % TOP_FEATURES.length];
    const second = TOP_FEATURES[(i + 2) % TOP_FEATURES.length];
    const third = TOP_FEATURES[(i + 3) % TOP_FEATURES.length];
    return {
      start: toIso(ws),
      end: toIso(we),
      psi_max: psi,
      psi_top: {
        [lead]: psi,
        [second]: round(psi * 0.45, 3),
        [third]: round(psi * 0.2, 3),
      },
      score_psi: SCORE_PSI[i],
      alert_rate: ALERT_RATE[i],
      status: statusOf(psi, SCORE_PSI[i]),
    };
  });
  return {
    reference: { ...REFERENCE },
    windows,
    events: [
      { detected_at: "2022-09-10T04:10:00", metric: "score_psi", value: 0.11, level: "warn", action: "none" },
      {
        detected_at: "2022-09-10T06:10:00",
        metric: "score_psi",
        value: 0.27,
        level: "drift",
        action: "recalibrate_threshold",
      },
      { detected_at: "2022-09-10T06:10:00", metric: "psi_max", value: 0.13, level: "warn", action: "none" },
      {
        detected_at: "2022-09-10T16:10:00",
        metric: "psi_max",
        value: 0.27,
        level: "drift",
        action: "queue_retraining_review",
      },
    ],
  };
}

/**
 * The modelling ladder (CLAUDE.md rule 4), rungs 1-4, mean ± std over 5 seeds.
 * Deterministic scorers (majority, rules) have zero seed variance by construction.
 */
export function modelLadder(): ModelLadder {
  return {
    prevalence: 0.00147,
    items: [
      { name: "random", rung: 1, pr_auc: 0.0015, pr_auc_std: 0.0001, roc_auc: 0.501, roc_auc_std: 0.004, recall_at_fpr_1e3: 0.001, recall_at_fpr_1e3_std: 0.001, precision_at_budget: 0.0016, precision_at_budget_std: 0.0012, n_seeds: 5 },
      { name: "majority", rung: 1, pr_auc: 0.00147, pr_auc_std: 0, roc_auc: 0.5, roc_auc_std: 0, recall_at_fpr_1e3: 0, recall_at_fpr_1e3_std: 0, precision_at_budget: 0, precision_at_budget_std: 0, n_seeds: 5 },
      { name: "amount_only", rung: 1, pr_auc: 0.0031, pr_auc_std: 0, roc_auc: 0.583, roc_auc_std: 0, recall_at_fpr_1e3: 0.004, recall_at_fpr_1e3_std: 0, precision_at_budget: 0.006, precision_at_budget_std: 0, n_seeds: 5 },
      { name: "rules", rung: 2, pr_auc: 0.0412, pr_auc_std: 0, roc_auc: 0.712, roc_auc_std: 0, recall_at_fpr_1e3: 0.061, recall_at_fpr_1e3_std: 0, precision_at_budget: 0.084, precision_at_budget_std: 0, n_seeds: 5 },
      { name: "rules_strict", rung: 2, pr_auc: 0.0368, pr_auc_std: 0, roc_auc: 0.664, roc_auc_std: 0, recall_at_fpr_1e3: 0.072, recall_at_fpr_1e3_std: 0, precision_at_budget: 0.091, precision_at_budget_std: 0, n_seeds: 5 },
      { name: "lr_tabular", rung: 3, pr_auc: 0.071, pr_auc_std: 0.004, roc_auc: 0.902, roc_auc_std: 0.003, recall_at_fpr_1e3: 0.118, recall_at_fpr_1e3_std: 0.009, precision_at_budget: 0.142, precision_at_budget_std: 0.011, n_seeds: 5 },
      { name: "rf_tabular", rung: 3, pr_auc: 0.182, pr_auc_std: 0.011, roc_auc: 0.948, roc_auc_std: 0.004, recall_at_fpr_1e3: 0.231, recall_at_fpr_1e3_std: 0.014, precision_at_budget: 0.286, precision_at_budget_std: 0.017, n_seeds: 5 },
      { name: "xgb_tabular", rung: 3, pr_auc: 0.236, pr_auc_std: 0.009, roc_auc: 0.961, roc_auc_std: 0.003, recall_at_fpr_1e3: 0.274, recall_at_fpr_1e3_std: 0.012, precision_at_budget: 0.331, precision_at_budget_std: 0.013, n_seeds: 5 },
      { name: "isolation_forest", rung: 4, pr_auc: 0.0088, pr_auc_std: 0.0021, roc_auc: 0.684, roc_auc_std: 0.019, recall_at_fpr_1e3: 0.012, recall_at_fpr_1e3_std: 0.004, precision_at_budget: 0.017, precision_at_budget_std: 0.005, n_seeds: 5 },
      { name: "lr_behavioural", rung: 4, pr_auc: 0.214, pr_auc_std: 0.006, roc_auc: 0.958, roc_auc_std: 0.002, recall_at_fpr_1e3: 0.262, recall_at_fpr_1e3_std: 0.008, precision_at_budget: 0.309, precision_at_budget_std: 0.01, n_seeds: 5 },
      { name: "rf_behavioural", rung: 4, pr_auc: 0.548, pr_auc_std: 0.014, roc_auc: 0.986, roc_auc_std: 0.002, recall_at_fpr_1e3: 0.461, recall_at_fpr_1e3_std: 0.012, precision_at_budget: 0.588, precision_at_budget_std: 0.015, n_seeds: 5 },
      { name: "xgb_behavioural", rung: 4, pr_auc: 0.61, pr_auc_std: 0.01, roc_auc: 0.99, roc_auc_std: 0.001, recall_at_fpr_1e3: 0.5, recall_at_fpr_1e3_std: 0.011, precision_at_budget: 0.64, precision_at_budget_std: 0.012, n_seeds: 5 },
    ],
  };
}
