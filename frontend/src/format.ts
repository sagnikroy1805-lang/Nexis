/**
 * Formatting and display labels.
 *
 * Copy rule (CLAUDE.md rule 5): every label here describes what the system
 * observed or what the model weighted. None asserts intent or wrongdoing.
 */
import type { AlertStatus, DriftStatus } from "./types";

const usdFmt = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
const intFmt = new Intl.NumberFormat("en-US");

export function fmtUsd(x: number | null | undefined): string {
  return x === null || x === undefined || Number.isNaN(x) ? "—" : usdFmt.format(x);
}

/** Compact USD for chart labels: $18.2k, $1.2M. */
export function fmtUsdShort(x: number): string {
  const abs = Math.abs(x);
  if (abs >= 1e6) return `$${(x / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `$${(x / 1e3).toFixed(1)}k`;
  return `$${x.toFixed(0)}`;
}

export function fmtInt(x: number | null | undefined): string {
  return x === null || x === undefined ? "—" : intFmt.format(x);
}

/**
 * Dataset timestamps are naive wall-clock strings. Show them as recorded
 * (no time-zone conversion), minute resolution.
 */
export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 16);
}

export function fmtScore(x: number | null | undefined, digits = 4): string {
  return x === null || x === undefined ? "—" : x.toFixed(digits);
}

export function fmtNum(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  if (Number.isInteger(x)) return intFmt.format(x);
  return x.toFixed(digits);
}

export function fmtPct(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined ? "—" : `${(x * 100).toFixed(digits)}%`;
}

/** Signed contribution, e.g. +1.84 / −0.21 (true minus sign). */
export function fmtSigned(x: number, digits = 2): string {
  const s = Math.abs(x).toFixed(digits);
  return x > 0 ? `+${s}` : x < 0 ? `−${s}` : s;
}

export function fmtDuration(secs: number | null | undefined): string {
  if (secs === null || secs === undefined) return "—";
  if (secs < 60) return `${secs} s`;
  if (secs < 3600) return `${Math.round(secs / 60)} min`;
  if (secs < 86_400) return `${(secs / 3600).toFixed(1)} h`;
  return `${(secs / 86_400).toFixed(1)} d`;
}

/** Number of decimals that keeps `std` visible (at least two significant digits). */
export function decimalsFor(std: number, min = 2, max = 5): number {
  if (!std || !Number.isFinite(std)) return min;
  const d = Math.ceil(-Math.log10(Math.abs(std))) + 1;
  return Math.max(min, Math.min(max, d));
}

export function snakeToWords(s: string): string {
  const w = s.replace(/_/g, " ");
  return w.charAt(0).toUpperCase() + w.slice(1);
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

export const STATUS_LABEL: Record<AlertStatus, string> = {
  open: "Open",
  escalated: "Escalated",
  dismissed: "Dismissed",
  needs_info: "Needs info",
};

export const DRIFT_LABEL: Record<DriftStatus, string> = {
  ok: "OK",
  warn: "Warn",
  drift: "Drift",
};

/** Plain-language descriptions of RulesScorer rule names. */
export const RULE_LABEL: Record<string, string> = {
  large_amount: "Amount above the training-period 99th percentile",
  far_above_own_history: "Amount more than 3 std above the sender's own history",
  burst_1h: "Sender's payment count in the prior hour is unusually high",
  fan_in_24h: "Receiver's incoming count in the prior 24h is unusually high",
  pass_through_1h: "Sender received a similar amount within the prior hour",
  new_cross_bank_counterparty: "First payment to a counterparty at another bank",
  cross_currency: "Payment and receipt currencies differ",
};

export const CONTEXT_LABEL: Record<string, string> = {
  prior_sends: "Prior payments sent",
  prior_receipts: "Prior payments received",
  distinct_counterparties: "Distinct counterparties",
  secs_since_last_receipt: "Time since last receipt",
};

/** Drift-response actions. These adjust the model pipeline, never an account. */
export const DRIFT_ACTION_LABEL: Record<string, string> = {
  none: "None (logged)",
  recalibrate_threshold: "Threshold recalibrated",
  queue_retraining_review: "Retraining review queued",
};

export const RUNG_NAME: Record<number, string> = {
  1: "Trivial",
  2: "Rules",
  3: "Tabular ML",
  4: "Tabular + behavioural features",
  5: "Graph structural features → XGBoost",
  6: "Homogeneous GNN",
  7: "Heterogeneous GNN",
  8: "Temporal heterogeneous GNN",
  9: "Fusion + ring detection",
};

// ---------------------------------------------------------------------------
// Colour (orbit theme). Risk is encoded by size and gold fill strength for
// nodes, and by a white -> black ramp for edges; numbers are always shown too.
// ---------------------------------------------------------------------------

export const GOLD = "#FFA800";
export const INK = "#0B0B0B";
export const CREAM = "#FFF4E6";

const clamp01 = (r: number) => Math.max(0, Math.min(1, r));

/** Node fill opacity: unscored accounts are hollow. */
export function riskFillOpacity(r: number | null | undefined): number {
  if (r === null || r === undefined || Number.isNaN(r)) return 0;
  return 0.3 + 0.7 * clamp01(r);
}

/** Edge colour on white glass: low risk light grey, high risk black; unscored translucent ink. */
export function edgeRiskColor(r: number | null | undefined): string {
  if (r === null || r === undefined || Number.isNaN(r)) return "rgba(11,11,11,0.35)";
  const v = Math.round(200 * (1 - clamp01(r)) ** 1.3);
  return `rgb(${v}, ${v}, ${v})`;
}
