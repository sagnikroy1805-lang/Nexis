import { motion } from "framer-motion";
import { useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { CytoGraph, GraphLegend, type GraphLayout } from "../components/CytoGraph";
import { buildElements } from "../components/graphElements";
import {
  AccountLink,
  Card,
  DEFAULT_LANGUAGE_NOTE,
  ErrorBox,
  Loading,
  PageTitle,
  RiskBar,
} from "../components/ui";
import { fmtInt, fmtScore, fmtTime, fmtUsd } from "../format";
import { useApi } from "../hooks/useApi";
import type { AccountTransaction, Ring } from "../types";

/**
 * Resolve a ring's tx_ids to records by reading each member's transactions and
 * keeping the ones the ring lists (the contract has no per-ring endpoint).
 */
async function ringTransactions(ring: Ring): Promise<{ txs: AccountTransaction[]; missing: string[] }> {
  const wanted = new Set(ring.tx_ids);
  const found = new Map<string, AccountTransaction>();
  const results = await Promise.all(ring.members.map((m) => api.accountTransactions(m, { limit: 500 })));
  for (const r of results) {
    for (const t of r.items) if (wanted.has(t.tx_id)) found.set(t.tx_id, t);
  }
  const txs = [...found.values()].sort((a, b) => a.occurred_at.localeCompare(b.occurred_at));
  return { txs, missing: ring.tx_ids.filter((id) => !found.has(id)) };
}

function RingDetail({ ring }: { ring: Ring }) {
  const res = useApi(() => ringTransactions(ring), [ring.ring_id]);

  const { elements, layout } = useMemo(() => {
    if (!res.data) return { elements: [], layout: { kind: "circle" } as GraphLayout };
    // Node risk: highest risk score among the ring's payments touching the account.
    const risk = new Map<string, number | null>(ring.members.map((m) => [m, null]));
    for (const t of res.data.txs) {
      for (const a of [t.src_account, t.dst_account]) {
        const cur = risk.get(a) ?? null;
        if (t.risk_score !== null && (cur === null || t.risk_score > cur)) risk.set(a, t.risk_score);
      }
    }
    const els = buildElements(
      [...risk.entries()].map(([id, r]) => ({ id, risk: r })),
      res.data.txs.map((t) => ({
        tx_id: t.tx_id,
        src: t.src_account,
        dst: t.dst_account,
        amount_usd: t.amount_usd,
        occurred_at: t.occurred_at,
        risk_score: t.risk_score,
        payment_format: t.payment_format,
      })),
      { labels: "all" },
    );
    return { elements: els, layout: { kind: "circle" } as GraphLayout };
  }, [res.data, ring.members]);

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="mt-6"
    >
      <div className="glass flex flex-wrap items-end justify-between gap-4">
        <h2 className="font-display text-[56px] leading-none">{ring.ring_id}</h2>
        <div className="num text-[12px]">
          score {fmtScore(ring.score, 2)} · {fmtTime(ring.first_seen)} → {fmtTime(ring.last_seen)}
        </div>
      </div>
      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[260px_minmax(0,1fr)]">
        <Card title={`Members · ${ring.members.length}`}>
          <ul className="flex flex-col gap-1">
            {ring.members.map((m) => (
              <li key={m}>
                <AccountLink id={m} />
              </li>
            ))}
          </ul>
        </Card>
        <div className="glass min-w-0">
          {res.error && <ErrorBox error={res.error} onRetry={res.reload} />}
          {res.loading && <Loading label="Resolving payments" />}
          {res.data && (
            <>
              <CytoGraph elements={elements} layout={layout} height={380} ariaLabel={`Payments within candidate ring ${ring.ring_id}`} />
              <GraphLegend />
              {res.data.missing.length > 0 && (
                <p className="mt-2 text-[12px]" role="alert">
                  {res.data.missing.length} of {ring.tx_ids.length} payments not found: {res.data.missing.join(", ")}
                </p>
              )}
            </>
          )}
        </div>
      </div>
      {res.data && res.data.txs.length > 0 && (
        <div className="glass mt-6 overflow-x-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th>Occurred</th>
                <th>Transaction</th>
                <th>Sender → receiver</th>
                <th className="r">Amount</th>
                <th>Format</th>
                <th>Risk</th>
              </tr>
            </thead>
            <tbody>
              {res.data.txs.map((t) => (
                <tr key={t.tx_id}>
                  <td className="num whitespace-nowrap">{fmtTime(t.occurred_at)}</td>
                  <td className="mono">{t.tx_id}</td>
                  <td className="mono whitespace-nowrap">
                    {t.src_account} → {t.dst_account}
                  </td>
                  <td className="r num">{fmtUsd(t.amount_usd)}</td>
                  <td>{t.payment_format}</td>
                  <td>
                    <RiskBar score={t.risk_score} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </motion.section>
  );
}

export default function Rings() {
  const { ringId } = useParams();
  const navigate = useNavigate();
  const rings = useApi(() => api.rings(50), []);
  const selected = rings.data?.items.find((r) => r.ring_id === ringId) ?? null;
  const go = (id: string) => navigate(`/rings/${encodeURIComponent(id)}`);

  return (
    <div>
      <PageTitle
        title="Rings"
        meta={rings.data ? <span className="num">{rings.data.items.length} candidates</span> : undefined}
        note={DEFAULT_LANGUAGE_NOTE}
      />
      {rings.error && <div className="mt-4"><ErrorBox error={rings.error} onRetry={rings.reload} /></div>}
      {rings.loading && <Loading />}
      {rings.data && (
        <div className="glass mt-6 overflow-x-auto">
          <table className="tbl tbl-click">
            <thead>
              <tr>
                <th>Ring</th>
                <th>Score</th>
                <th className="r">Accounts</th>
                <th className="r">Payments</th>
                <th>First seen</th>
                <th>Last seen</th>
              </tr>
            </thead>
            <tbody>
              {rings.data.items.map((r, i) => (
                <motion.tr
                  key={r.ring_id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35, delay: i * 0.04 }}
                  tabIndex={0}
                  className={r.ring_id === ringId ? "is-selected" : ""}
                  onClick={() => go(r.ring_id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      go(r.ring_id);
                    }
                  }}
                >
                  <td className="font-display text-[22px] leading-none">{r.ring_id}</td>
                  <td>
                    <RiskBar score={r.score} />
                  </td>
                  <td className="r num">{fmtInt(r.members.length)}</td>
                  <td className="r num">{fmtInt(r.n_transactions)}</td>
                  <td className="num whitespace-nowrap">{fmtTime(r.first_seen)}</td>
                  <td className="num whitespace-nowrap">{fmtTime(r.last_seen)}</td>
                </motion.tr>
              ))}
              {rings.data.items.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-8 text-center">
                    No candidate rings.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
      {ringId && rings.data && !selected && (
        <div className="mt-6">
          <ErrorBox error={`Ring ${ringId} is not in the list.`} />
        </div>
      )}
      {selected && <RingDetail key={selected.ring_id} ring={selected} />}
      {ringId && (
        <Link to="/rings" className="glass mt-6 inline-block !rounded-full !px-4 !py-1.5 text-[12px] hover:underline">
          ← All rings
        </Link>
      )}
    </div>
  );
}
