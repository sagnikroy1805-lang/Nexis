import { motion } from "framer-motion";
import { useEffect, useMemo, useState, type FormEvent, type ReactNode } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { CytoGraph, GraphLegend, type GraphLayout } from "../components/CytoGraph";
import { buildElements, hopDistances } from "../components/graphElements";
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
import type { GraphEdge, Network } from "../types";

const LIMITS = [50, 100, 300, 500];

type Selection = { kind: "node"; id: string } | { kind: "edge"; id: string } | null;

function graphHref(account: string, search: URLSearchParams): string {
  const qs = search.toString();
  return `/graph/${encodeURIComponent(account)}${qs ? `?${qs}` : ""}`;
}

/** "2022-09-10T04:12:00" <-> "2022-09-10T04:12" for <input type="datetime-local">. */
const toInput = (iso: string | null) => (iso ? iso.slice(0, 16) : "");
const fromInput = (v: string) => (v.length === 16 ? `${v}:00` : v);

function Row({ k, children }: { k: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-ink/20 py-1.5 text-[13px]">
      <dt className="label">{k}</dt>
      <dd className="min-w-0 text-right">{children}</dd>
    </div>
  );
}

function NodeDetails({ id, net, onRecentre }: { id: string; net: Network; onRecentre: (id: string) => void }) {
  const node = net.nodes.find((n) => n.id === id);
  const out = net.edges.filter((e) => e.src === id);
  const inn = net.edges.filter((e) => e.dst === id);
  const sum = (es: GraphEdge[]) => es.reduce((s, e) => s + e.amount_usd, 0);
  return (
    <div>
      <div className="mono mb-2 text-[13px] font-medium break-all">{id}</div>
      <dl>
        <Row k="Risk">{node ? <RiskBar score={node.risk} /> : "—"}</Row>
        <Row k="In">
          <span className="num">
            {inn.length} · {fmtUsd(sum(inn))}
          </span>
        </Row>
        <Row k="Out">
          <span className="num">
            {out.length} · {fmtUsd(sum(out))}
          </span>
        </Row>
      </dl>
      {!node?.is_center && (
        <button type="button" className="pill pill-sm pill-solid mt-3" onClick={() => onRecentre(id)}>
          Recentre
        </button>
      )}
    </div>
  );
}

function EdgeDetails({ edge, linkSearch }: { edge: GraphEdge; linkSearch: URLSearchParams }) {
  return (
    <div>
      <div className="mono mb-2 text-[13px] font-medium">{edge.tx_id}</div>
      <dl>
        <Row k="From">
          <Link className="mono underline decoration-ink/40 underline-offset-2" to={graphHref(edge.src, linkSearch)}>
            {edge.src}
          </Link>
        </Row>
        <Row k="To">
          <Link className="mono underline decoration-ink/40 underline-offset-2" to={graphHref(edge.dst, linkSearch)}>
            {edge.dst}
          </Link>
        </Row>
        <Row k="Amount">
          <span className="num">{fmtUsd(edge.amount_usd)}</span>
        </Row>
        <Row k="Occurred">
          <span className="num">{fmtTime(edge.occurred_at)}</span>
        </Row>
        <Row k="Format">{edge.payment_format}</Row>
        <Row k="Risk">
          <RiskBar score={edge.risk_score} />
        </Row>
      </dl>
    </div>
  );
}

function AccountPicker() {
  const navigate = useNavigate();
  const [value, setValue] = useState("");
  const suggestions = useApi(() => api.alerts({ status: "all", sort: "risk_score", limit: 8 }), []);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    const v = value.trim();
    if (v) navigate(`/graph/${encodeURIComponent(v)}`);
  };
  return (
    <div>
      <PageTitle title="Graph" />
      <form className="glass mt-6 flex max-w-2xl flex-wrap items-center gap-3" onSubmit={submit}>
        <input
          className="field-line mono w-72 text-[14px]"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="011_8000ABC10"
          aria-label="Account id"
        />
        <button type="submit" className="pill pill-solid">
          Open
        </button>
      </form>
      <Card title="Highest-scored senders" className="mt-6 max-w-2xl">
        {suggestions.error && <ErrorBox error={suggestions.error} onRetry={suggestions.reload} />}
        {suggestions.loading && <Loading />}
        <ul className="flex flex-col">
          {suggestions.data?.items.map((a) => (
            <li key={a.alert_id} className="flex items-center justify-between gap-4 border-b border-ink/20 py-2 text-[13px]">
              <AccountLink id={a.src_account} until={a.occurred_at} />
              <span className="num">
                {fmtTime(a.occurred_at)} · {fmtScore(a.risk_score)}
              </span>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}

export default function GraphExplorer() {
  const { account } = useParams();
  if (!account) return <AccountPicker />;
  return <Explorer account={account} />;
}

function Explorer({ account }: { account: string }) {
  const navigate = useNavigate();
  const [search, setSearch] = useSearchParams();
  const until = search.get("until") || undefined;
  const hops: 1 | 2 = search.get("hops") === "1" ? 1 : 2;
  const limit = LIMITS.includes(Number(search.get("limit"))) ? Number(search.get("limit")) : 300;

  const [selection, setSelection] = useState<Selection>(null);
  const [untilDraft, setUntilDraft] = useState(toInput(until ?? null));
  const [accountDraft, setAccountDraft] = useState(account);
  useEffect(() => {
    setSelection(null);
    setAccountDraft(account);
  }, [account]);
  useEffect(() => setUntilDraft(toInput(until ?? null)), [until]);

  const net = useApi(() => api.network(account, { until, hops, limit }), [account, until, hops, limit], true);
  const txs = useApi(() => api.accountTransactions(account, { until, limit: 100 }), [account, until]);

  // Recentre keeps `until`, hops and limit: the cut-off must not move while exploring.
  const recentre = (id: string) => navigate(graphHref(id, search));
  const setParam = (k: string, v: string | null) => {
    const next = new URLSearchParams(search);
    if (v === null || v === "") next.delete(k);
    else next.set(k, v);
    setSearch(next, { replace: true });
  };

  // The previous network stays on screen while a recentre loads.
  const data = net.data;
  const { elements, layout } = useMemo(() => {
    if (!data) return { elements: [], layout: { kind: "cose" } as GraphLayout };
    const els = buildElements(
      data.nodes.map((n) => ({ id: n.id, risk: n.risk, center: n.is_center })),
      data.edges,
      { labels: data.nodes.length <= 15 ? "all" : "key" },
    );
    const l: GraphLayout = { kind: "concentric", levels: hopDistances([data.center], data.edges) };
    return { elements: els, layout: l };
  }, [data]);

  const selectedEdge = selection?.kind === "edge" ? data?.edges.find((e) => e.tx_id === selection.id) : undefined;
  const effectiveUntil = data?.until ?? until;

  return (
    <div>
      <PageTitle title="Graph" meta={<span className="mono text-[13px]">{account}</span>} note={DEFAULT_LANGUAGE_NOTE} />

      <div className="glass mt-6 !py-3">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 text-[12px]">
        <form
          className="flex max-w-full min-w-0 flex-wrap items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            const v = accountDraft.trim();
            if (v && v !== account) recentre(v);
          }}
        >
          <span className="label">Account</span>
          <input className="field-line mono w-44 max-w-full min-w-0" value={accountDraft} onChange={(e) => setAccountDraft(e.target.value)} aria-label="Account id" />
          <button type="submit" className="pill pill-sm">
            Go
          </button>
        </form>
        <div className="flex items-center gap-1.5" role="group" aria-label="Hops">
          <span className="label mr-1">Hops</span>
          {[1, 2].map((h) => (
            <button
              key={h}
              type="button"
              className="pill pill-sm"
              aria-pressed={hops === h}
              onClick={() => setParam("hops", String(h))}
            >
              {h}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2">
          <span className="label">Edges</span>
          <select className="field-line" value={limit} onChange={(e) => setParam("limit", e.target.value)}>
            {LIMITS.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <form
          className="flex max-w-full min-w-0 flex-wrap items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            setParam("until", untilDraft ? fromInput(untilDraft) : null);
          }}
        >
          <span className="label">Before</span>
          <input
            type="datetime-local"
            className="field-line num max-w-full min-w-0"
            value={untilDraft}
            onChange={(e) => setUntilDraft(e.target.value)}
            aria-label="Show records before"
          />
          <button type="submit" className="pill pill-sm">
            Apply
          </button>
          {until && (
            <button type="button" className="pill pill-sm" onClick={() => setParam("until", null)}>
              Reset
            </button>
          )}
        </form>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-ink/10 pt-2.5 text-[12px]" role="status">
        <span>
          Only transactions before <b className="num">{effectiveUntil ? fmtTime(effectiveUntil) : "the default cut-off"}</b> are shown
        </span>
        {data && (
          <span className="num">
            {fmtInt(data.nodes.length)} accounts · {fmtInt(data.edges.length)} payments
          </span>
        )}
        {data?.truncated && (
          <span className="rounded-full bg-ink px-2.5 py-0.5 text-[10.5px] font-medium tracking-[0.12em] text-gold uppercase" role="alert">
            Truncated at {limit} edges
          </span>
        )}
        {net.loading && data && <span>Refreshing…</span>}
      </div>
      </div>
      {net.error && <div className="mt-3"><ErrorBox error={net.error} onRetry={net.reload} /></div>}

      <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="glass min-w-0">
          {!data && net.loading && <Loading label="Loading network" />}
          {data && data.edges.length === 0 && <p className="py-10 text-[13px]">No payments before the cut-off.</p>}
          {data && data.edges.length > 0 && (
            <>
              <CytoGraph
                elements={elements}
                layout={layout}
                height={600}
                selectedId={selection?.id ?? null}
                ariaLabel={`Payment network around ${account}`}
                onNodeTap={(id) => setSelection({ kind: "node", id })}
                onNodeDblTap={(id) => id !== account && recentre(id)}
                onEdgeTap={(id) => setSelection({ kind: "edge", id })}
                onBackgroundTap={() => setSelection(null)}
              />
              <GraphLegend />
            </>
          )}
        </div>

        <div className="flex min-w-0 flex-col gap-6">
          <Card title={selection ? (selection.kind === "node" ? "Account" : "Payment") : "Centre"}>
            {data ? (
              <motion.div key={selection?.id ?? "centre"} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.25 }}>
                {selection?.kind === "edge" && selectedEdge ? (
                  <EdgeDetails edge={selectedEdge} linkSearch={search} />
                ) : (
                  <NodeDetails id={selection?.kind === "node" ? selection.id : data.center} net={data} onRecentre={recentre} />
                )}
              </motion.div>
            ) : (
              <Loading />
            )}
          </Card>

          <Card title="Transactions" aside={<span className="num">{txs.data ? txs.data.items.length : ""}</span>}>
            {txs.error && <ErrorBox error={txs.error} onRetry={txs.reload} />}
            {txs.loading && <Loading />}
            {txs.data && txs.data.items.length === 0 && <p className="text-[12px]">None before the cut-off.</p>}
            {txs.data && txs.data.items.length > 0 && (
              <ul className="max-h-[560px] overflow-y-auto pr-1">
                {txs.data.items.map((t) => {
                  const out = t.src_account === account;
                  const other = out ? t.dst_account : t.src_account;
                  const inGraph = data?.edges.some((e) => e.tx_id === t.tx_id);
                  const selected = selection?.kind === "edge" && selection.id === t.tx_id;
                  return (
                    <li key={t.tx_id} className={`border-b border-ink/20 px-1 py-2 text-[12.5px] ${selected ? "bg-ink/5" : ""}`}>
                      <div className="flex items-center gap-2">
                        <span
                          className={`rounded-full px-1.5 text-[9.5px] font-medium tracking-[0.12em] uppercase ${
                            out ? "border border-ink" : "bg-ink text-cream"
                          }`}
                        >
                          {out ? "Out" : "In"}
                        </span>
                        <Link className="mono underline decoration-ink/40 underline-offset-2" to={graphHref(other, search)}>
                          {other}
                        </Link>
                        <span className="num ml-auto font-medium">{fmtUsd(t.amount_usd)}</span>
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-3 text-[11.5px]">
                        <span className="num">{fmtTime(t.occurred_at)}</span>
                        <span>{t.payment_format}</span>
                        <RiskBar score={t.risk_score} />
                        {inGraph && (
                          <button
                            type="button"
                            className="ml-auto underline underline-offset-2"
                            onClick={() => setSelection({ kind: "edge", id: t.tx_id })}
                          >
                            show
                          </button>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
