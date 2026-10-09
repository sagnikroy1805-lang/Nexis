/**
 * TypeScript shapes for every payload in docs/api_contract.md (v1).
 *
 * Field names mirror the contract exactly (snake_case). Where the contract shows
 * a value only by example, the type is widened defensively and the reason is
 * noted, so the UI never crashes on a legitimate backend response.
 */

/** ISO-8601 timestamp string, e.g. "2022-09-10T04:12:00" (dataset wall-clock, no zone). */
export type IsoTime = string;

/** Account id as it appears in IBM AML, e.g. "011_8000ABC10" (bank_account). */
export type AccountId = string;

/** Transaction id, e.g. "HI-Small:0412337" (dataset:row). */
export type TxId = string;

// ---------------------------------------------------------------------------
// GET /api/health
// ---------------------------------------------------------------------------

export interface Health {
  status: string;
  model_version: string;
  db: string;
}

// ---------------------------------------------------------------------------
// GET /api/summary
// ---------------------------------------------------------------------------

export interface TimeRange {
  start: IsoTime;
  end: IsoTime;
}

export interface SummaryMetrics {
  pr_auc: number;
  pr_auc_std: number;
  prevalence: number;
  recall_at_budget: number;
  precision_at_budget: number;
}

export interface Summary {
  model_version: string;
  dataset: string;
  replay_period: TimeRange;
  transactions_scored: number;
  alerts_total: number;
  alerts_open: number;
  alert_budget: number;
  threshold: number;
  metrics: SummaryMetrics;
}

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------

export type AlertStatus = "open" | "escalated" | "dismissed" | "needs_info";
export const ALERT_STATUSES: readonly AlertStatus[] = ["open", "escalated", "dismissed", "needs_info"];

/** The `status` query value accepted by GET /api/alerts. */
export type AlertStatusFilter = AlertStatus | "all";

/** The `sort` query value accepted by GET /api/alerts. */
export type AlertSort = "risk_score" | "occurred_at";

/** One entry of `top_reasons`; a feature the model weighted for this score. */
export interface Reason {
  feature: string;
  label: string;
  contribution: number;
}

export interface Alert {
  alert_id: number;
  tx_id: TxId;
  occurred_at: IsoTime;
  created_at: IsoTime;
  src_account: AccountId;
  dst_account: AccountId;
  amount_usd: number;
  payment_format: string;
  risk_score: number;
  rank: number;
  status: AlertStatus;
  top_reasons: Reason[];
}

export interface AlertListParams {
  status?: AlertStatusFilter;
  limit?: number;
  offset?: number;
  sort?: AlertSort;
}

export interface AlertList {
  total: number;
  items: Alert[];
}

/** The transaction record embedded in an evidence packet. */
export interface Transaction {
  tx_id: TxId;
  occurred_at: IsoTime;
  src_account: AccountId;
  dst_account: AccountId;
  amount_usd: number;
  amount_paid: number;
  pay_currency: string;
  payment_format: string;
  is_cross_bank: boolean;
  is_cross_currency: boolean;
}

export interface FeatureContribution {
  feature: string;
  label: string;
  /** The feature's value for this transaction. Widened to null for missing values. */
  value: number | null;
  contribution: number;
}

/**
 * Per-account context computed from records before the alert time.
 * The contract shows different fields for src and dst, so every count is
 * optional and the UI renders whichever fields are present.
 */
export interface AccountContext {
  account: AccountId;
  prior_sends?: number;
  prior_receipts?: number;
  distinct_counterparties?: number;
  secs_since_last_receipt?: number | null;
  [extra: string]: string | number | boolean | null | undefined;
}

export interface GraphNode {
  id: AccountId;
  kind: string;
  /** Account-level risk in [0, 1]. Widened to null for accounts with no scored activity. */
  risk: number | null;
}

export interface GraphEvidenceNode extends GraphNode {
  is_focus: boolean;
}

export interface GraphEdge {
  tx_id: TxId;
  src: AccountId;
  dst: AccountId;
  amount_usd: number;
  occurred_at: IsoTime;
  /** Widened to null: records outside the scored replay period carry no score. */
  risk_score: number | null;
  payment_format: string;
}

export interface GraphEvidenceEdge extends GraphEdge {
  important: boolean;
}

export interface GraphEvidence {
  nodes: GraphEvidenceNode[];
  edges: GraphEvidenceEdge[];
}

export interface RingCandidate {
  ring_id: string;
  members: AccountId[];
  score: number;
}

export interface EvidencePacket {
  alert_id: number;
  tx_id: TxId;
  model_version: string;
  risk_score: number;
  threshold: number;
  generated_at: IsoTime;
  /** Never empty (PROJECT_RULES.md rule 6). */
  source_record_ids: TxId[];
  transaction: Transaction;
  feature_contributions: FeatureContribution[];
  rule_hits: string[];
  account_context: {
    src: AccountContext;
    dst: AccountContext;
  };
  graph_evidence: GraphEvidence;
  ring_candidate: RingCandidate | null;
  language_note: string;
}

export interface AlertDetail {
  alert: Alert;
  evidence: EvidencePacket;
}

/** The only disposition values the API accepts. There is no account action. */
export type Disposition = "escalated" | "dismissed" | "needs_info";

export interface DispositionRequest {
  disposition: Disposition;
  note: string;
}

// ---------------------------------------------------------------------------
// POST /api/alerts/{id}/investigate
// ---------------------------------------------------------------------------

export interface InvestigatorClaim {
  text: string;
  source_record_ids: TxId[];
  verified: boolean;
}

export interface InvestigatorResult {
  alert_id: number;
  mode: "llm" | "template";
  /** LLM model id in `llm` mode; the contract does not say what `template` mode sends, so null is allowed. */
  model: string | null;
  summary: string;
  claims: InvestigatorClaim[];
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Accounts
// ---------------------------------------------------------------------------

export interface NetworkParams {
  until?: IsoTime;
  hops?: 1 | 2;
  limit?: number;
}

export interface NetworkNode extends GraphNode {
  is_center: boolean;
}

export interface Network {
  center: AccountId;
  until: IsoTime;
  truncated: boolean;
  nodes: NetworkNode[];
  edges: GraphEdge[];
}

export interface AccountTransactionsParams {
  until?: IsoTime;
  limit?: number;
}

/** Transaction shape plus `risk_score` (null if not scored). */
export interface AccountTransaction extends Transaction {
  risk_score: number | null;
}

export interface AccountTransactions {
  items: AccountTransaction[];
}

// ---------------------------------------------------------------------------
// GET /api/rings
// ---------------------------------------------------------------------------

export interface Ring {
  ring_id: string;
  members: AccountId[];
  n_transactions: number;
  score: number;
  first_seen: IsoTime;
  last_seen: IsoTime;
  tx_ids: TxId[];
}

export interface RingList {
  items: Ring[];
}

// ---------------------------------------------------------------------------
// GET /api/drift
// ---------------------------------------------------------------------------

export type DriftStatus = "ok" | "warn" | "drift";

export interface DriftWindow {
  start: IsoTime;
  end: IsoTime;
  psi_max: number;
  /** Feature name -> PSI, the features with the largest shift in this window. */
  psi_top: Record<string, number>;
  score_psi: number;
  alert_rate: number;
  status: DriftStatus;
}

export interface DriftEvent {
  detected_at: IsoTime;
  metric: string;
  value: number;
  /** `warn | drift` in practice; kept as string so unknown levels still render. */
  level: string;
  action: string;
}

export interface DriftReport {
  reference: TimeRange;
  windows: DriftWindow[];
  events: DriftEvent[];
}

// ---------------------------------------------------------------------------
// GET /api/models
// ---------------------------------------------------------------------------

/**
 * One rung-of-the-ladder row: mean over seeds, with std.
 *
 * The contract example carries `pr_auc_std` only, but says the endpoint is
 * "mean ± std over seeds". The other `*_std` fields follow the same naming
 * convention and are optional; the UI flags any metric that arrives without
 * one (PROJECT_RULES.md rule 3).
 */
export interface ModelRow {
  name: string;
  rung: number;
  pr_auc: number;
  pr_auc_std: number;
  roc_auc: number;
  roc_auc_std?: number;
  recall_at_fpr_1e3: number;
  recall_at_fpr_1e3_std?: number;
  precision_at_budget: number;
  precision_at_budget_std?: number;
  n_seeds: number;
}

export interface ModelLadder {
  items: ModelRow[];
  prevalence: number;
}
