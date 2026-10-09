/**
 * A small, deterministic transaction world for demo mode.
 *
 * ~46 accounts, ~120 transactions (a few in the reference period, unscored),
 * three embedded candidate rings, and 40 alerts above the threshold. Every
 * endpoint mock derives its payload from this one world, so the alert queue,
 * evidence packets, graph explorer and ring view all agree with each other.
 */
import type { AccountId, Alert, AlertStatus, Ring, Transaction, TxId } from "../types";
import { bankOf, computeFeatures, contributions, topReasons } from "./features";
import { MINUTE_MS, between, intBetween, makeRng, normal, parseIso, pick, round, toIso, weighted } from "./rng";
import type { Rng } from "./rng";

export const REPLAY_START = "2022-09-09T16:10:00";
export const REPLAY_END = "2022-09-10T23:59:00";
export const REFERENCE = { start: "2022-09-01T00:00:00", end: "2022-09-06T13:34:00" };
export const THRESHOLD = 0.9731;
export const MODEL_VERSION = "xgb_behavioural@a1b2c3d";
export const DATASET = "IBM AML HI-Small";
export const LANGUAGE_NOTE =
  "Risk scores reflect model output on recorded data. They are not findings of wrongdoing.";

export interface WorldTx extends Transaction {
  risk_score: number | null;
  ms: number;
  ring_id: string | null;
}

export interface World {
  accounts: AccountId[];
  txs: WorldTx[];
  txById: Map<TxId, WorldTx>;
  byAccount: Map<AccountId, WorldTx[]>;
  rings: Ring[];
  alerts: Alert[];
}

const START_MS = parseIso(REPLAY_START);
const END_MS = parseIso(REPLAY_END);
const REF_START_MS = parseIso(REFERENCE.start);
const REF_END_MS = parseIso(REFERENCE.end);
const SPAN_MIN = Math.floor((END_MS - START_MS) / MINUTE_MS);

const BANKS = ["011", "0220", "0070", "001", "0012", "00213", "0118", "03208", "0410", "0025", "0153", "02841"];
const HEX = "0123456789ABCDEF";

/** USD -> currency units, approximate September 2022 rates. */
const RATES: Record<string, number> = {
  "US Dollar": 1,
  Euro: 1.0,
  "UK Pound": 0.87,
  "Swiss Franc": 0.97,
  Yuan: 6.93,
  Rupee: 79.6,
  Yen: 143.2,
  Bitcoin: 1 / 19800,
};

const FORMATS: readonly (readonly [string, number])[] = [
  ["ACH", 30],
  ["Cheque", 22],
  ["Credit Card", 22],
  ["Cash", 12],
  ["Wire", 10],
  ["Bitcoin", 4],
];

// The three accounts used in docs/api_contract.md examples.
const A_FEED = "0070_8000F0001";
const A_MID = "011_8000ABC10";
const A_DST = "0220_8001DD320";
/** The contract's example alert transaction. */
const PINNED_TX = { id: "HI-Small:0412337", ms: parseIso("2022-09-10T04:12:00") };

interface RawTx {
  ms: number;
  src: AccountId;
  dst: AccountId;
  amount: number;
  format: string;
  currency?: string;
  crossCurrency?: boolean;
  score: number | null;
  ring: string | null;
}

function accountId(rng: Rng): AccountId {
  let hex = "";
  for (let i = 0; i < 5; i++) hex += HEX[intBetween(rng, 0, 15)];
  return `${pick(rng, BANKS)}_800${pick(rng, ["0", "0", "1", "2"])}${hex}`;
}

function lowScore(rng: Rng): number {
  // Most transactions score near zero; a minority land in the middle.
  const r = rng();
  if (r < 0.12) return round(between(rng, 0.2, 0.9), 4);
  return round(0.0008 + 0.18 * rng() ** 4, 4);
}

function amountFor(rng: Rng, format: string): number {
  const mu = format === "Credit Card" ? 5.6 : format === "Wire" || format === "Bitcoin" ? 8.8 : 7.6;
  return round(Math.exp(mu + 1.3 * normal(rng)) + 5, 2);
}

export function buildWorld(): World {
  const rng = makeRng(20220909);

  // ---- accounts ---------------------------------------------------------
  const accounts = new Set<AccountId>([A_FEED, A_MID, A_DST]);
  while (accounts.size < 46) accounts.add(accountId(rng));
  const ids = [...accounts];
  const [, , , r3, r4, q0, q1, q2, q3, s0, m1, m2, m3, m4, s5, ...others] = ids;
  const hubs = others.slice(0, 3);

  const raw: RawTx[] = [];
  const t = (iso: string) => parseIso(iso);

  // ---- ring R-0031: gather, layer, partial return ------------------------
  const r31: [string, AccountId, AccountId, number, string, number][] = [
    ["2022-09-09T22:10:00", r3, A_FEED, 19120.0, "Wire", 0.9741],
    ["2022-09-09T23:45:00", r4, A_FEED, 9400.0, "ACH", 0.9788],
    ["2022-09-10T03:41:00", A_FEED, A_MID, 9050.0, "ACH", 0.9812],
    ["2022-09-10T03:58:00", A_FEED, A_MID, 18500.0, "ACH", 0.9876],
    ["2022-09-10T04:05:00", A_MID, r3, 8870.0, "ACH", 0.9805],
    ["2022-09-10T04:12:00", A_MID, A_DST, 18234.55, "ACH", 0.9934],
  ];
  // ---- ring R-0047: cycle across currencies ------------------------------
  const r47: [string, AccountId, AccountId, number, string, number, string][] = [
    ["2022-09-10T09:20:00", q0, q1, 42800.0, "Bitcoin", 0.9845, "Bitcoin"],
    ["2022-09-10T10:02:00", q1, q2, 41950.0, "Wire", 0.9762, "Euro"],
    ["2022-09-10T11:15:00", q2, q3, 41120.0, "Wire", 0.9791, "Euro"],
    ["2022-09-10T12:40:00", q3, q0, 40300.0, "Bitcoin", 0.9902, "Bitcoin"],
    ["2022-09-10T13:55:00", q0, q2, 12500.0, "Wire", 0.9739, "US Dollar"],
  ];
  // ---- ring R-0052: scatter then gather ----------------------------------
  const r52: [string, AccountId, AccountId, number, string, number][] = [
    ["2022-09-10T15:00:00", s0, m1, 9600.0, "Cash", 0.9771],
    ["2022-09-10T15:20:00", s0, m2, 9450.0, "Cheque", 0.9118],
    ["2022-09-10T15:42:00", s0, m3, 9700.0, "Cash", 0.9803],
    ["2022-09-10T16:05:00", s0, m4, 9300.0, "Cheque", 0.9536],
    ["2022-09-10T19:30:00", m1, s5, 9350.0, "ACH", 0.9758],
    ["2022-09-10T20:10:00", m2, s5, 9200.0, "ACH", 0.9824],
    ["2022-09-10T21:05:00", m3, s5, 9480.0, "ACH", 0.9867],
    ["2022-09-10T21:40:00", m4, s5, 9050.0, "ACH", 0.9893],
  ];
  for (const [iso, src, dst, amount, format, score] of r31) {
    raw.push({ ms: t(iso), src, dst, amount, format, score, ring: "R-0031" });
  }
  for (const [iso, src, dst, amount, format, score, currency] of r47) {
    raw.push({
      ms: t(iso), src, dst, amount, format, score, ring: "R-0047",
      currency, crossCurrency: currency !== "US Dollar",
    });
  }
  for (const [iso, src, dst, amount, format, score] of r52) {
    raw.push({ ms: t(iso), src, dst, amount, format, score, ring: "R-0052" });
  }

  // ---- background activity ---------------------------------------------
  const pickAccount = () => (rng() < 0.3 ? pick(rng, hubs) : pick(rng, ids));
  const background = (ms: number, src: AccountId, dst: AccountId, scored: boolean) => {
    if (src === dst) return;
    const format = weighted(rng, FORMATS);
    const cross = rng() < 0.08;
    raw.push({
      ms, src, dst, format,
      amount: amountFor(rng, format),
      currency: cross ? pick(rng, ["Euro", "UK Pound", "Yuan", "Rupee", "Swiss Franc", "Yen"]) : "US Dollar",
      crossCurrency: cross,
      score: scored ? lowScore(rng) : null,
      ring: null,
    });
  };
  const replayMs = () => START_MS + intBetween(rng, 1, SPAN_MIN - 1) * MINUTE_MS;

  for (let i = 0; i < 68; i++) background(replayMs(), pickAccount(), pickAccount(), true);
  // Give every ring member some ordinary activity so the rings are embedded, not isolated.
  for (const member of [A_FEED, A_MID, A_DST, r3, r4, q0, q1, q2, q3, s0, m1, m2, m3, m4, s5]) {
    const n = intBetween(rng, 1, 2);
    for (let i = 0; i < n; i++) {
      const other = pickAccount();
      if (rng() < 0.5) background(replayMs(), member, other, true);
      else background(replayMs(), other, member, true);
    }
  }
  // Reference-period records: before the replay, so they carry no score.
  const refMs = () => REF_START_MS + Math.floor(rng() * (REF_END_MS - REF_START_MS) / MINUTE_MS) * MINUTE_MS;
  for (const member of [A_FEED, A_MID, A_DST, r3, q0, s0]) background(refMs(), member, pickAccount(), false);
  for (let i = 0; i < 6; i++) background(refMs(), pickAccount(), pickAccount(), false);

  // ---- 23 background transactions above threshold (non-ring alerts) -------
  const ringHigh = raw.filter((r) => r.ring && (r.score ?? 0) >= THRESHOLD).length;
  const candidates = raw.filter((r) => !r.ring && r.score !== null);
  const need = 40 - ringHigh;
  const chosen = new Set<RawTx>();
  while (chosen.size < need) chosen.add(pick(rng, candidates));
  for (const r of chosen) r.score = round(between(rng, 0.9733, 0.9915), 4);

  // ---- ids, currencies, sort ------------------------------------------------
  raw.sort((a, b) => a.ms - b.ms);
  const used = new Set<number>();
  const txs: WorldTx[] = raw.map((r) => {
    let idx: number;
    if (r.ms === PINNED_TX.ms && r.src === A_MID && r.dst === A_DST) {
      idx = Number(PINNED_TX.id.split(":")[1]);
    } else if (r.ms >= START_MS) {
      // ~218 rows per minute: HI-Small row numbers grow with time.
      idx = 254_941 + 218 * Math.round((r.ms - START_MS) / MINUTE_MS) + intBetween(rng, 1, 200);
    } else {
      idx = 1_000 + 30 * Math.round((r.ms - REF_START_MS) / MINUTE_MS) + intBetween(rng, 0, 25);
    }
    while (used.has(idx)) idx += 1;
    used.add(idx);
    const currency = r.currency ?? "US Dollar";
    return {
      tx_id: `HI-Small:${String(idx).padStart(7, "0")}`,
      occurred_at: toIso(r.ms),
      src_account: r.src,
      dst_account: r.dst,
      amount_usd: r.amount,
      amount_paid: round(r.amount * (RATES[currency] ?? 1), currency === "Bitcoin" ? 6 : 2),
      pay_currency: currency,
      payment_format: r.format,
      is_cross_bank: bankOf(r.src) !== bankOf(r.dst),
      is_cross_currency: r.crossCurrency ?? false,
      risk_score: r.score,
      ms: r.ms,
      ring_id: r.ring,
    };
  });

  const txById = new Map(txs.map((x) => [x.tx_id, x] as const));
  const byAccount = new Map<AccountId, WorldTx[]>();
  for (const x of txs) {
    for (const a of [x.src_account, x.dst_account]) {
      const list = byAccount.get(a) ?? [];
      list.push(x);
      byAccount.set(a, list);
    }
  }

  // ---- rings --------------------------------------------------------------
  const ringMeta: Record<string, number> = { "R-0031": 0.91, "R-0047": 0.86, "R-0052": 0.78 };
  const rings: Ring[] = Object.entries(ringMeta).map(([ring_id, score]) => {
    const members = txs.filter((x) => x.ring_id === ring_id);
    const accountsInRing = [...new Set(members.flatMap((x) => [x.src_account, x.dst_account]))];
    return {
      ring_id,
      members: accountsInRing,
      n_transactions: members.length,
      score,
      first_seen: members[0].occurred_at,
      last_seen: members[members.length - 1].occurred_at,
      tx_ids: members.map((x) => x.tx_id),
    };
  });

  // ---- alerts -------------------------------------------------------------
  const alerted = txs.filter((x) => x.risk_score !== null && x.risk_score >= THRESHOLD);
  const byScore = [...alerted].sort((a, b) => (b.risk_score ?? 0) - (a.risk_score ?? 0) || a.ms - b.ms);
  const rankOf = new Map(byScore.map((x, i) => [x.tx_id, i + 1] as const));
  const statusRng = makeRng(7);
  const alerts: Alert[] = alerted.map((x, i) => {
    const rank = rankOf.get(x.tx_id) ?? 0;
    let status: AlertStatus = "open";
    if (rank > 1) {
      status = weighted(statusRng, [
        ["open", 31],
        ["escalated", 4],
        ["dismissed", 3],
        ["needs_info", 2],
      ] as const);
    }
    const feats = computeFeatures(byAccount, x);
    return {
      alert_id: i + 1,
      tx_id: x.tx_id,
      occurred_at: x.occurred_at,
      created_at: x.occurred_at,
      src_account: x.src_account,
      dst_account: x.dst_account,
      amount_usd: x.amount_usd,
      payment_format: x.payment_format,
      risk_score: x.risk_score ?? 0,
      rank,
      status,
      top_reasons: topReasons(contributions(x, feats)),
    };
  });

  return { accounts: ids, txs, txById, byAccount, rings, alerts };
}

export { END_MS, START_MS };
