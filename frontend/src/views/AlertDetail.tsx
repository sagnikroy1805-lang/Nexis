import { motion } from "framer-motion";
import { useMemo, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { api, errorMessage } from "../api";
import { ContributionChart } from "../components/ContributionChart";
import { CytoGraph, GraphLegend, type GraphLayout } from "../components/CytoGraph";
import { buildElements, hopDistances } from "../components/graphElements";
import {
  AccountLink,
  Card,
  ErrorBox,
  Loading,
  PageTitle,
  StatusChip,
  TxChip,
} from "../components/ui";
import {
  CONTEXT_LABEL,
  RULE_LABEL,
  STATUS_LABEL,
  fmtDuration,
  fmtNum,
  fmtScore,
  fmtTime,
  fmtUsd,
  snakeToWords,
} from "../format";
import { useApi } from "../hooks/useApi";
import type {
  AccountContext,
  Alert,
  AlertDetail as AlertDetailT,
  Disposition,
  EvidencePacket,
  InvestigatorResult,
} from "../types";

function ScoreVsThreshold({ score, threshold }: { score: number; threshold: number }) {
  const lo = Math.max(0, Math.floor((Math.min(score, threshold) - 0.02) * 100) / 100);
  const pos = (v: number) => ((v - lo) / (1 - lo)) * 100;
  const diff = score - threshold;
  return (
    <Card>
      <div className="flex flex-wrap items-end gap-x-10 gap-y-4">
        <div>
          <div className="label">Risk score</div>
          <div className="num font-display text-[64px] leading-none md:text-[80px]">{fmtScore(score)}</div>
        </div>
        <div>
          <div className="label">Threshold</div>
          <div className="num font-display text-[32px] leading-none">{fmtScore(threshold)}</div>
        </div>
        <div>
          <div className="label">Margin</div>
          <div className="num font-display text-[32px] leading-none">
            {diff >= 0 ? "+" : "−"}
            {Math.abs(diff).toFixed(4)}
          </div>
        </div>
      </div>
      <div className="mt-5" aria-hidden="true">
        <div className="relative h-2 rounded-full bg-ink/20">
          <motion.span
            className="absolute inset-y-0 left-0 rounded-full bg-gold ring-1 ring-ink/60"
            initial={{ width: 0 }}
            animate={{ width: `${pos(score)}%` }}
            transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          />
          <span className="absolute -inset-y-1.5 w-[2px] bg-ink" style={{ left: `${pos(threshold)}%` }} />
        </div>
        <div className="num mt-1 flex justify-between text-[10.5px]">
          <span>{lo.toFixed(2)}</span>
          <span>1.00</span>
        </div>
      </div>
    </Card>
  );
}

function TransactionCard({ ev }: { ev: EvidencePacket }) {
  const t = ev.transaction;
  const rows: [string, ReactNode][] = [
    ["Transaction", <span className="mono">{t.tx_id}</span>],
    ["Occurred", <span className="num">{fmtTime(t.occurred_at)}</span>],
    ["Sender", <AccountLink id={t.src_account} until={t.occurred_at} />],
    ["Receiver", <AccountLink id={t.dst_account} until={t.occurred_at} />],
    ["Amount", <span className="num font-medium">{fmtUsd(t.amount_usd)}</span>],
    ["Paid", <span className="num">{fmtNum(t.amount_paid, 2)} {t.pay_currency}</span>],
    ["Format", t.payment_format],
    ["Cross-bank", t.is_cross_bank ? "Yes" : "No"],
    ["Cross-currency", t.is_cross_currency ? "Yes" : "No"],
    ["Model", <span className="mono">{ev.model_version}</span>],
  ];
  return (
    <Card title="Transaction">
      <dl className="grid grid-cols-1 gap-x-8 sm:grid-cols-2">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 border-b border-ink/20 py-1.5 text-[13px]">
            <dt className="label self-center">{k}</dt>
            <dd className="min-w-0 truncate text-right">{v}</dd>
          </div>
        ))}
      </dl>
    </Card>
  );
}

function ContextColumn({ title, ctx }: { title: string; ctx: AccountContext }) {
  const entries = Object.entries(ctx).filter(([k]) => k !== "account");
  return (
    <div>
      <div className="mb-1 flex flex-wrap items-baseline gap-2">
        <span className="font-display text-[22px] leading-none">{title}</span>
        <AccountLink id={ctx.account} />
      </div>
      <dl>
        {entries.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 border-b border-ink/20 py-1.5 text-[13px]">
            <dt>{CONTEXT_LABEL[k] ?? snakeToWords(k)}</dt>
            <dd className="num">
              {k.startsWith("secs_") ? fmtDuration(v as number | null) : typeof v === "number" ? fmtNum(v) : String(v ?? "—")}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function DispositionPanel({ alert, onUpdated }: { alert: Alert; onUpdated: (a: Alert) => void }) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState<Disposition | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);

  const submit = async (d: Disposition) => {
    setBusy(d);
    setError(null);
    setDone(null);
    try {
      const updated = await api.disposition(alert.alert_id, { disposition: d, note });
      onUpdated(updated);
      setDone(`Recorded: ${STATUS_LABEL[updated.status] ?? updated.status}`);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(null);
    }
  };

  const buttons: { d: Disposition; label: string }[] = [
    { d: "escalated", label: "Escalate" },
    { d: "dismissed", label: "Dismiss" },
    { d: "needs_info", label: "Needs info" },
  ];

  return (
    <Card title="Disposition">
      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        rows={3}
        maxLength={2000}
        aria-label="Analyst note"
        placeholder="Note"
        className="w-full resize-y rounded-xl border border-ink/40 bg-white/60 p-2.5 text-[13px] placeholder:text-ink/60 focus-visible:bg-white"
      />
      <div className="mt-3 flex flex-wrap gap-2">
        {buttons.map((b) => (
          <button
            key={b.d}
            type="button"
            className={`pill ${alert.status === b.d ? "is-current" : ""}`}
            disabled={busy !== null}
            onClick={() => submit(b.d)}
          >
            {busy === b.d ? "Saving…" : b.label}
          </button>
        ))}
      </div>
      {done && (
        <p className="mt-2 text-[12px]" role="status">
          {done}
        </p>
      )}
      {error && <div className="mt-2"><ErrorBox error={error} /></div>}
    </Card>
  );
}

function InvestigatorPanel({ alertId, ev }: { alertId: number; ev: EvidencePacket }) {
  const [result, setResult] = useState<InvestigatorResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const packet = useMemo(() => new Set(ev.source_record_ids), [ev.source_record_ids]);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.investigate(alertId));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card
      title="Investigator"
      aside={
        <button type="button" className="pill pill-sm pill-solid" onClick={run} disabled={busy}>
          {busy ? "Running…" : result ? "Run again" : "Run"}
        </button>
      }
    >
      {error && <ErrorBox error={error} onRetry={run} />}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="text-[13px]"
        >
          <div className="mb-2 flex items-center gap-2">
            <span
              className={`rounded-full px-2 py-px text-[10px] font-medium tracking-[0.12em] uppercase ${
                result.mode === "llm" ? "bg-ink text-gold" : "border border-ink"
              }`}
            >
              {result.mode === "llm" ? "LLM" : "Template"}
            </span>
            {result.model && <span className="mono text-cream-ink">{result.model}</span>}
          </div>
          <p className="mb-3 leading-relaxed">{result.summary}</p>
          <ol className="flex flex-col gap-2">
            {result.claims.map((c, i) => {
              // Shown as verified only if the API says so AND every cited id is in the packet.
              const outside = c.source_record_ids.filter((id) => !packet.has(id));
              const verified = c.verified && outside.length === 0 && c.source_record_ids.length > 0;
              return (
                <li
                  key={i}
                  className={`rounded-xl p-2.5 ${verified ? "border border-ink/15" : "border-2 border-ink bg-gold/25"}`}
                >
                  <div className="mb-1.5">
                    {!verified && (
                      <span className="mr-1.5 rounded-full bg-ink px-2 py-px text-[10px] font-medium tracking-[0.12em] text-gold uppercase">
                        Unverified
                      </span>
                    )}
                    {c.text}
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {c.source_record_ids.length === 0 && <TxChip id="no source cited" tone="bad" />}
                    {c.source_record_ids.map((id) => (
                      <TxChip key={id} id={id} tone={packet.has(id) ? "default" : "bad"} title={packet.has(id) ? "In packet" : "Not in packet"} />
                    ))}
                  </div>
                </li>
              );
            })}
          </ol>
          {result.warnings.length > 0 && (
            <ul className="mt-3 list-disc pl-5 text-[12px]" role="alert">
              {result.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
        </motion.div>
      )}
    </Card>
  );
}

function EvidenceGraph({ ev }: { ev: EvidencePacket }) {
  const { elements, layout } = useMemo(() => {
    const g = ev.graph_evidence;
    const focus = g.nodes.filter((n) => n.is_focus).map((n) => n.id);
    const roots = focus.length ? focus : [ev.transaction.src_account];
    const els = buildElements(
      g.nodes.map((n) => ({ id: n.id, risk: n.risk, focus: n.is_focus })),
      g.edges.map((e) => ({ ...e, alerted: e.tx_id === ev.tx_id })),
      { labels: "key", dimUnimportant: true },
    );
    const l: GraphLayout = { kind: "concentric", levels: hopDistances(roots, g.edges) };
    return { elements: els, layout: l };
  }, [ev]);

  const t = ev.transaction;
  return (
    <Card
      title={`Graph evidence · ${ev.graph_evidence.nodes.length} accounts · ${ev.graph_evidence.edges.length} payments`}
      aside={
        <Link className="pill pill-sm" to={`/graph/${encodeURIComponent(t.src_account)}?until=${encodeURIComponent(t.occurred_at)}`}>
          Open in graph →
        </Link>
      }
    >
      {ev.graph_evidence.edges.length === 0 ? (
        <p className="text-[12px]">No graph records.</p>
      ) : (
        <>
          <CytoGraph elements={elements} layout={layout} height={360} ariaLabel="Accounts and payments in the evidence packet" />
          <GraphLegend
            extra={
              <span className="inline-flex items-center gap-1.5">
                <span className="inline-block h-1.5 w-5 rounded-full bg-gold ring-1 ring-ink/50" aria-hidden="true" />
                important
              </span>
            }
          />
        </>
      )}
    </Card>
  );
}

function RingCard({ ev }: { ev: EvidencePacket }) {
  const r = ev.ring_candidate;
  return (
    <Card
      title="Candidate ring"
      aside={
        r && (
          <Link className="pill pill-sm" to={`/rings/${encodeURIComponent(r.ring_id)}`}>
            View →
          </Link>
        )
      }
    >
      {r ? (
        <>
          <div className="mb-2 flex items-baseline gap-3">
            <span className="font-display text-[30px] leading-none">{r.ring_id}</span>
            <span className="num text-[12px]">score {fmtScore(r.score, 2)}</span>
          </div>
          <ul className="flex flex-col gap-0.5">
            {r.members.map((m) => (
              <li key={m} className="flex items-center gap-2">
                <AccountLink id={m} />
                {(m === ev.transaction.src_account || m === ev.transaction.dst_account) && (
                  <span className="inline-block h-2 w-2 rounded-full bg-gold ring-1 ring-ink" title="Party to this payment" />
                )}
              </li>
            ))}
          </ul>
        </>
      ) : (
        <p className="text-[12px]">None.</p>
      )}
    </Card>
  );
}

function SourceRecords({ ev }: { ev: EvidencePacket }) {
  const ids = ev.source_record_ids;
  return (
    <Card title={`Source records · ${ids.length}`}>
      {ids.length === 0 ? (
        <div className="border border-ink px-3 py-2 text-[12px]" role="alert">
          No source records — this packet is defective.
        </div>
      ) : (
        <div className="flex flex-wrap gap-1">
          {ids.map((id) => (
            <TxChip key={id} id={id} title={id === ev.tx_id ? "Alerted transaction" : "Cited record"} />
          ))}
        </div>
      )}
    </Card>
  );
}

export default function AlertDetail() {
  const { id } = useParams();
  const alertId = Number(id);
  const valid = Number.isInteger(alertId) && alertId > 0;
  const detail = useApi<AlertDetailT>(
    () => (valid ? api.alert(alertId) : Promise.reject(new Error(`Invalid alert id "${id}"`))),
    [alertId],
  );

  const d = detail.data;
  return (
    <div>
      {detail.error && <div className="mt-4"><ErrorBox error={detail.error} onRetry={detail.reload} /></div>}
      {!d && detail.loading && <Loading label="Loading evidence" />}
      {d && (
        <>
          <PageTitle
            title={`Alert ${d.alert.alert_id}`}
            back={{ to: "/alerts", label: "Alerts" }}
            note={d.evidence.language_note}
            meta={
              <span className="num">
                Rank {d.alert.rank} · <span className="mono">{d.alert.tx_id}</span> · {fmtTime(d.alert.created_at)}
              </span>
            }
          >
            <StatusChip status={d.alert.status} />
          </PageTitle>

          <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,3fr)_minmax(330px,2fr)]">
            <div className="flex min-w-0 flex-col gap-6">
              <ScoreVsThreshold score={d.evidence.risk_score} threshold={d.evidence.threshold} />
              <Card title="Feature contributions">
                <ContributionChart items={d.evidence.feature_contributions} />
              </Card>
              <Card title="Rule hits">
                {d.evidence.rule_hits.length === 0 ? (
                  <p className="text-[12px]">None.</p>
                ) : (
                  <div className="flex flex-wrap gap-1.5">
                    {d.evidence.rule_hits.map((r) => (
                      <span
                        key={r}
                        className="mono rounded-full border border-ink px-2.5 py-0.5 text-[11.5px]"
                        title={RULE_LABEL[r] ?? snakeToWords(r)}
                      >
                        {r}
                      </span>
                    ))}
                  </div>
                )}
              </Card>
              <EvidenceGraph ev={d.evidence} />
              <TransactionCard ev={d.evidence} />
              <Card title="Account context">
                <div className="grid grid-cols-1 gap-8 sm:grid-cols-2">
                  <ContextColumn title="Sender" ctx={d.evidence.account_context.src} />
                  <ContextColumn title="Receiver" ctx={d.evidence.account_context.dst} />
                </div>
              </Card>
            </div>
            <div className="flex min-w-0 flex-col gap-6">
              <DispositionPanel
                key={`disp-${d.alert.alert_id}`}
                alert={d.alert}
                onUpdated={(a) => detail.setData({ ...d, alert: a })}
              />
              <InvestigatorPanel key={`inv-${d.alert.alert_id}`} alertId={d.alert.alert_id} ev={d.evidence} />
              <RingCard ev={d.evidence} />
              <SourceRecords ev={d.evidence} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
