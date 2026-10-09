/**
 * Builds contract payloads (evidence packet, investigator output, account
 * network, account transactions) from the mock world.
 *
 * The leakage guard is reproduced: account risk, context and network edges
 * only use records strictly before the cut-off time.
 */
import type {
  AccountContext,
  AccountTransaction,
  Alert,
  EvidencePacket,
  GraphEdge,
  GraphEvidenceEdge,
  GraphEvidenceNode,
  InvestigatorClaim,
  InvestigatorResult,
  Network,
  NetworkNode,
  TxId,
} from "../types";
import { computeFeatures, contributions, priorOf, ruleHits } from "./features";
import { parseIso } from "./rng";
import { LANGUAGE_NOTE, MODEL_VERSION, REPLAY_END, THRESHOLD, type World, type WorldTx } from "./world";

const HOUR_MS = 3_600_000;

/** Strip mock-internal fields so payloads carry exactly the contract shape. */
function publicTx(x: WorldTx): AccountTransaction {
  return {
    tx_id: x.tx_id,
    occurred_at: x.occurred_at,
    src_account: x.src_account,
    dst_account: x.dst_account,
    amount_usd: x.amount_usd,
    amount_paid: x.amount_paid,
    pay_currency: x.pay_currency,
    payment_format: x.payment_format,
    is_cross_bank: x.is_cross_bank,
    is_cross_currency: x.is_cross_currency,
    risk_score: x.risk_score,
  };
}

function edgeOf(x: WorldTx): GraphEdge {
  return {
    tx_id: x.tx_id,
    src: x.src_account,
    dst: x.dst_account,
    amount_usd: x.amount_usd,
    occurred_at: x.occurred_at,
    risk_score: x.risk_score,
    payment_format: x.payment_format,
  };
}

/** Highest score among the account's scored records at or before `ms` (strictly before when `strict`). */
export function accountRisk(world: World, account: string, ms: number, strict: boolean): number | null {
  let best: number | null = null;
  for (const x of world.byAccount.get(account) ?? []) {
    if (strict ? x.ms >= ms : x.ms > ms) break;
    if (x.risk_score !== null && (best === null || x.risk_score > best)) best = x.risk_score;
  }
  return best;
}

function context(world: World, account: string, ms: number, role: "src" | "dst"): AccountContext {
  const prior = priorOf(world.byAccount, account, ms);
  const sends = prior.filter((x) => x.src_account === account);
  const receipts = prior.filter((x) => x.dst_account === account);
  const counterparties = new Set(prior.map((x) => (x.src_account === account ? x.dst_account : x.src_account)));
  if (role === "dst") {
    return { account, prior_receipts: receipts.length, distinct_counterparties: counterparties.size };
  }
  const last = receipts.length ? receipts[receipts.length - 1] : null;
  return {
    account,
    prior_sends: sends.length,
    prior_receipts: receipts.length,
    distinct_counterparties: counterparties.size,
    secs_since_last_receipt: last ? Math.round((ms - last.ms) / 1000) : null,
  };
}

export function evidenceFor(world: World, alert: Alert): EvidencePacket {
  const tx = world.txById.get(alert.tx_id);
  if (!tx) throw new Error(`No transaction ${alert.tx_id}`);
  const feats = computeFeatures(world.byAccount, tx);

  // Graph evidence: the alerted payment plus each party's recent prior records.
  const prior = new Map<TxId, WorldTx>();
  for (const party of [tx.src_account, tx.dst_account]) {
    for (const x of priorOf(world.byAccount, party, tx.ms).slice(-7)) prior.set(x.tx_id, x);
  }
  const ring = tx.ring_id ? world.rings.find((r) => r.ring_id === tx.ring_id) ?? null : null;
  const ringTx = new Set(ring?.tx_ids ?? []);
  const edges: GraphEvidenceEdge[] = [...prior.values(), tx]
    .sort((a, b) => a.ms - b.ms)
    .map((x) => ({
      ...edgeOf(x),
      important:
        x.tx_id === tx.tx_id ||
        ringTx.has(x.tx_id) ||
        (x.dst_account === tx.src_account && tx.ms - x.ms <= 6 * HOUR_MS),
    }));
  const nodeIds = [...new Set(edges.flatMap((e) => [e.src, e.dst]))];
  const nodes: GraphEvidenceNode[] = nodeIds.map((id) => ({
    id,
    kind: "account",
    risk: accountRisk(world, id, tx.ms, false),
    is_focus: id === tx.src_account || id === tx.dst_account,
  }));

  const ringCandidate =
    world.rings.find((r) => r.members.includes(tx.src_account) && r.members.includes(tx.dst_account)) ??
    world.rings.find((r) => r.members.includes(tx.src_account) || r.members.includes(tx.dst_account)) ??
    null;

  return {
    alert_id: alert.alert_id,
    tx_id: tx.tx_id,
    model_version: MODEL_VERSION,
    risk_score: alert.risk_score,
    threshold: THRESHOLD,
    generated_at: alert.created_at,
    source_record_ids: [tx.tx_id, ...edges.map((e) => e.tx_id).filter((id) => id !== tx.tx_id)],
    transaction: publicTx(tx),
    feature_contributions: contributions(tx, feats),
    rule_hits: ruleHits(tx, feats),
    account_context: {
      src: context(world, tx.src_account, tx.ms, "src"),
      dst: context(world, tx.dst_account, tx.ms, "dst"),
    },
    graph_evidence: { nodes, edges },
    ring_candidate: ringCandidate
      ? { ring_id: ringCandidate.ring_id, members: ringCandidate.members, score: ringCandidate.score }
      : null,
    language_note: LANGUAGE_NOTE,
  };
}

const usd = (x: number) =>
  x.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });

export function investigate(world: World, alert: Alert, packet: EvidencePacket): InvestigatorResult {
  const tx = world.txById.get(alert.tx_id)!;
  const inPacket = new Set(packet.source_record_ids);
  const claims: InvestigatorClaim[] = [];
  const llm = alert.alert_id % 2 === 0;

  const fmtName = packet.transaction.payment_format;
  const article = /^[AEIOU]/i.test(fmtName) ? "An" : "A";
  claims.push({
    text: `${article} ${fmtName} payment of ${usd(tx.amount_usd)} from ${tx.src_account} to ${tx.dst_account} was recorded at ${tx.occurred_at.replace("T", " ")}.`,
    source_record_ids: [tx.tx_id],
    verified: true,
  });

  const receipts = priorOf(world.byAccount, tx.src_account, tx.ms).filter(
    (x) => x.dst_account === tx.src_account && inPacket.has(x.tx_id),
  );
  const lastIn = receipts[receipts.length - 1];
  if (lastIn) {
    const mins = Math.round((tx.ms - lastIn.ms) / 60000);
    claims.push({
      text: `The sender received ${usd(lastIn.amount_usd)} from ${lastIn.src_account} ${mins >= 120 ? `${Math.round(mins / 60)} hours` : `${mins} minutes`} before this payment.`,
      source_record_ids: [lastIn.tx_id],
      verified: true,
    });
  }

  const dstIn = priorOf(world.byAccount, tx.dst_account, tx.ms).filter(
    (x) => x.dst_account === tx.dst_account && inPacket.has(x.tx_id),
  );
  if (dstIn.length > 0) {
    claims.push({
      text: `The receiving account has ${dstIn.length} earlier incoming payment${dstIn.length === 1 ? "" : "s"} in the evidence packet.`,
      source_record_ids: dstIn.slice(-3).map((x) => x.tx_id),
      verified: true,
    });
  }

  const top = packet.feature_contributions.filter((c) => c.contribution > 0).slice(0, 2);

  if (llm && packet.ring_candidate) {
    // Cites the ring's records, several of which are outside this packet, so
    // the verifier must mark the claim unverified.
    const ring = world.rings.find((r) => r.ring_id === packet.ring_candidate!.ring_id)!;
    const parties = [tx.src_account, tx.dst_account].filter((a) => ring.members.includes(a));
    const who = parties.length === 2 ? "Both parties appear" : `${parties[0]} appears`;
    claims.push({
      text: `${who} in candidate ring ${ring.ring_id}, which the system grouped from ${ring.n_transactions} connected payments.`,
      source_record_ids: ring.tx_ids.slice(0, 4),
      verified: ring.tx_ids.slice(0, 4).every((id) => inPacket.has(id)),
    });
  } else if (llm && alert.alert_id % 4 === 0) {
    const outside = world.txs.find((x) => x.risk_score === null && !inPacket.has(x.tx_id));
    if (outside) {
      claims.push({
        text: "The sender's pattern resembles transfers recorded during the reference period.",
        source_record_ids: [outside.tx_id],
        verified: false,
      });
    }
  }

  const warnings = claims
    .map((c, i) => ({ c, i }))
    .filter(({ c }) => !c.verified)
    .map(({ c, i }) => {
      const missing = c.source_record_ids.filter((id) => !inPacket.has(id));
      return `Claim ${i + 1} cites ${missing.join(", ")}, which ${missing.length === 1 ? "is" : "are"} not in the evidence packet; marked unverified.`;
    });

  const summary =
    `The system observed a ${usd(tx.amount_usd)} ${packet.transaction.payment_format} payment from ${tx.src_account} to ${tx.dst_account}. ` +
    (top.length
      ? `The model weighted ${top.map((c) => c.label.charAt(0).toLowerCase() + c.label.slice(1)).join(" and ")} most heavily. `
      : "") +
    `The risk score of ${alert.risk_score.toFixed(4)} is above the review threshold of ${THRESHOLD.toFixed(4)}. ` +
    "This summary describes recorded activity for analyst review; it is not a finding of wrongdoing.";

  return {
    alert_id: alert.alert_id,
    mode: llm ? "llm" : "template",
    model: llm ? "claude-sonnet-5-5" : null,
    summary,
    claims,
    warnings,
  };
}

export function network(
  world: World,
  center: string,
  untilIso: string | undefined,
  hops: number,
  limit: number,
): Network {
  const until = untilIso ?? REPLAY_END;
  const untilMs = parseIso(until);
  // Leakage guard: only records strictly before `until`.
  const visible = world.txs.filter((x) => x.ms < untilMs);
  const adj = new Map<string, WorldTx[]>();
  for (const x of visible) {
    for (const a of [x.src_account, x.dst_account]) {
      const l = adj.get(a) ?? [];
      l.push(x);
      adj.set(a, l);
    }
  }
  const dist = new Map<string, number>([[center, 0]]);
  let frontier = [center];
  for (let h = 1; h <= hops; h++) {
    const next: string[] = [];
    for (const a of frontier) {
      for (const x of adj.get(a) ?? []) {
        const other = x.src_account === a ? x.dst_account : x.src_account;
        if (!dist.has(other)) {
          dist.set(other, h);
          next.push(other);
        }
      }
    }
    frontier = next;
  }
  const inScope = visible
    .filter((x) => dist.has(x.src_account) && dist.has(x.dst_account))
    .sort((a, b) => b.ms - a.ms);
  const kept = inScope.slice(0, limit);
  const nodeIds = new Set<string>([center]);
  for (const x of kept) {
    nodeIds.add(x.src_account);
    nodeIds.add(x.dst_account);
  }
  const nodes: NetworkNode[] = [...nodeIds].map((id) => ({
    id,
    kind: "account",
    risk: accountRisk(world, id, untilMs, true),
    is_center: id === center,
  }));
  return {
    center,
    until,
    truncated: inScope.length > kept.length,
    nodes,
    edges: kept.map(edgeOf),
  };
}

export function accountTransactions(
  world: World,
  account: string,
  untilIso: string | undefined,
  limit: number,
): AccountTransaction[] {
  const untilMs = untilIso ? parseIso(untilIso) : Number.POSITIVE_INFINITY;
  return (world.byAccount.get(account) ?? [])
    .filter((x) => x.ms < untilMs)
    .slice()
    .reverse()
    .slice(0, limit)
    .map(publicTx);
}
