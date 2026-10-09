/**
 * Mock feature attribution for fixture transactions.
 *
 * Feature names follow src/nexis/features/ and the rule names follow
 * RulesScorer in src/nexis/models/baselines/scorers.py, so the demo reads like
 * the real system. Every value is computed from records strictly before the
 * transaction (the same leakage guard the backend applies).
 */
import type { FeatureContribution, Reason } from "../types";
import { between, makeRng, round } from "./rng";
import type { WorldTx } from "./world";

const DAY_MS = 86_400_000;
const HOUR_MS = 3_600_000;

export interface TxFeatures {
  counterparty_is_new: number;
  src_has_history: number;
  src_secs_since_in: number | null;
  amt_over_src_in_24h: number | null;
  src_out_cnt_1h: number;
  src_out_cnt_24h: number;
  dst_in_cnt_24h: number;
  log_amount: number;
  is_cross_bank: number;
  is_cross_currency: number;
  amt_z_src: number | null;
  amt_vs_src_max: number | null;
  hour: number;
  src_n_counterparties: number;
  payment_format: string;
}

export function bankOf(account: string): string {
  return account.split("_")[0];
}

/** Records involving `account` strictly before `ms`. */
export function priorOf(byAccount: Map<string, WorldTx[]>, account: string, ms: number): WorldTx[] {
  return (byAccount.get(account) ?? []).filter((t) => t.ms < ms);
}

export function computeFeatures(byAccount: Map<string, WorldTx[]>, tx: WorldTx): TxFeatures {
  const srcPrior = priorOf(byAccount, tx.src_account, tx.ms);
  const dstPrior = priorOf(byAccount, tx.dst_account, tx.ms);

  const srcSends = srcPrior.filter((t) => t.src_account === tx.src_account);
  const srcReceipts = srcPrior.filter((t) => t.dst_account === tx.src_account);
  const dstReceipts = dstPrior.filter((t) => t.dst_account === tx.dst_account);

  const lastIn = srcReceipts.length ? srcReceipts[srcReceipts.length - 1] : null;
  const in24 = srcReceipts.filter((t) => t.ms > tx.ms - DAY_MS).reduce((s, t) => s + t.amount_usd, 0);

  const sendAmounts = srcSends.map((t) => t.amount_usd);
  let z: number | null = null;
  if (sendAmounts.length >= 2) {
    const mean = sendAmounts.reduce((s, a) => s + a, 0) / sendAmounts.length;
    const sd = Math.sqrt(sendAmounts.reduce((s, a) => s + (a - mean) ** 2, 0) / (sendAmounts.length - 1));
    z = sd > 0 ? (tx.amount_usd - mean) / sd : null;
  }
  const counterparties = new Set(
    srcPrior.map((t) => (t.src_account === tx.src_account ? t.dst_account : t.src_account)),
  );

  return {
    counterparty_is_new: srcSends.some((t) => t.dst_account === tx.dst_account) ? 0 : 1,
    src_has_history: srcPrior.length > 0 ? 1 : 0,
    src_secs_since_in: lastIn ? Math.round((tx.ms - lastIn.ms) / 1000) : null,
    amt_over_src_in_24h: in24 > 0 ? round(tx.amount_usd / in24, 3) : null,
    src_out_cnt_1h: srcSends.filter((t) => t.ms > tx.ms - HOUR_MS).length,
    src_out_cnt_24h: srcSends.filter((t) => t.ms > tx.ms - DAY_MS).length,
    dst_in_cnt_24h: dstReceipts.filter((t) => t.ms > tx.ms - DAY_MS).length,
    log_amount: round(Math.log1p(tx.amount_usd), 3),
    is_cross_bank: tx.is_cross_bank ? 1 : 0,
    is_cross_currency: tx.is_cross_currency ? 1 : 0,
    amt_z_src: z === null ? null : round(z, 2),
    amt_vs_src_max: sendAmounts.length ? round(tx.amount_usd / Math.max(...sendAmounts), 3) : null,
    hour: new Date(tx.ms).getUTCHours(),
    src_n_counterparties: counterparties.size,
    payment_format: tx.payment_format,
  };
}

/** SHAP-style contributions, deterministic per transaction. Sorted by |contribution|. */
export function contributions(tx: WorldTx, f: TxFeatures): FeatureContribution[] {
  const rng = makeRng(Number(tx.tx_id.split(":")[1]) ^ 0x5eed);
  const up = (lo: number, hi: number) => between(rng, lo, hi);
  const down = (lo: number, hi: number) => -between(rng, lo, hi);
  const out: FeatureContribution[] = [];
  const push = (feature: string, label: string, value: number | null, c: number) =>
    out.push({ feature, label, value, contribution: round(c, 2) });

  push(
    "counterparty_is_new",
    "First payment to this counterparty",
    f.counterparty_is_new,
    f.counterparty_is_new ? up(1.1, 1.9) : down(0.15, 0.4),
  );
  const s = f.src_secs_since_in;
  push(
    "src_secs_since_in",
    "Seconds since the sender last received funds",
    s,
    s === null ? down(0.05, 0.2) : s <= 3600 ? up(0.8, 1.3) : s <= 86_400 ? up(0.1, 0.35) : down(0.1, 0.3),
  );
  const r = f.amt_over_src_in_24h;
  push(
    "amt_over_src_in_24h",
    "Amount relative to the sender's receipts in the prior 24h",
    r,
    r !== null && r >= 0.5 && r <= 1.5 ? up(0.5, 1.0) : down(0.05, 0.25),
  );
  push(
    "src_out_cnt_24h",
    "Sender outgoing payments, prior 24h",
    f.src_out_cnt_24h,
    f.src_out_cnt_24h >= 3 ? up(0.35, 0.7) : down(0.05, 0.2),
  );
  push(
    "dst_in_cnt_24h",
    "Receiver incoming payments, prior 24h",
    f.dst_in_cnt_24h,
    f.dst_in_cnt_24h >= 3 ? up(0.4, 0.8) : down(0.05, 0.15),
  );
  push("log_amount", "Log amount (USD)", f.log_amount, tx.amount_usd > 8000 ? up(0.2, 0.5) : down(0.1, 0.35));
  push(
    "is_cross_bank",
    "Sender and receiver at different banks",
    f.is_cross_bank,
    f.is_cross_bank ? up(0.1, 0.3) : down(0.05, 0.15),
  );
  const fmt = f.payment_format;
  push(
    `payment_format=${fmt}`,
    `Payment format: ${fmt}`,
    1,
    fmt === "ACH"
      ? up(0.3, 0.6)
      : fmt === "Bitcoin" || fmt === "Wire"
        ? up(0.2, 0.45)
        : fmt === "Credit Card"
          ? down(0.3, 0.6)
          : between(rng, -0.1, 0.15),
  );
  push("hour", "Hour of day", f.hour, f.hour < 6 ? up(0.1, 0.25) : down(0.02, 0.12));
  push(
    "is_cross_currency",
    "Payment and receipt currencies differ",
    f.is_cross_currency,
    f.is_cross_currency ? up(0.3, 0.6) : down(0.01, 0.06),
  );
  const m = f.amt_vs_src_max;
  push(
    "amt_vs_src_max",
    "Amount relative to the sender's largest prior payment",
    m,
    m === null ? up(0.3, 0.7) : m > 1.5 ? up(0.6, 1.2) : down(0.1, 0.3),
  );
  push(
    "src_n_counterparties",
    "Sender distinct counterparties to date",
    f.src_n_counterparties,
    f.src_n_counterparties <= 2 ? up(0.1, 0.3) : down(0.05, 0.3),
  );

  // A high score with almost no positive attribution would be an incoherent
  // fixture; give it the timing-regularity feature the real model also uses.
  const positives = out.filter((c) => c.contribution > 0).length;
  if ((tx.risk_score ?? 0) >= 0.5 && positives < 3) {
    push("iat_cv", "Variability of the sender's time between payments", round(between(rng, 0.05, 0.3), 2), up(0.5, 0.9));
  }

  return out
    .filter((c) => c.contribution !== 0)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 10);
}

export function topReasons(contribs: FeatureContribution[]): Reason[] {
  return contribs
    .filter((c) => c.contribution > 0)
    .slice(0, 2)
    .map(({ feature, label, contribution }) => ({ feature, label, contribution }));
}

/** Rule hits with the same names and logic as RulesScorer.rule_hits. */
export function ruleHits(tx: WorldTx, f: TxFeatures): string[] {
  const hits: string[] = [];
  if (tx.amount_usd > 40_000) hits.push("large_amount");
  if ((f.amt_z_src ?? 0) > 3) hits.push("far_above_own_history");
  if (f.src_out_cnt_1h >= 2) hits.push("burst_1h");
  if (f.dst_in_cnt_24h >= 3) hits.push("fan_in_24h");
  if (
    f.src_secs_since_in !== null &&
    f.src_secs_since_in <= 3600 &&
    f.amt_over_src_in_24h !== null &&
    f.amt_over_src_in_24h >= 0.5 &&
    f.amt_over_src_in_24h <= 1.5
  ) {
    hits.push("pass_through_1h");
  }
  if (f.counterparty_is_new && f.is_cross_bank) hits.push("new_cross_bank_counterparty");
  if (f.is_cross_currency) hits.push("cross_currency");
  return hits;
}
