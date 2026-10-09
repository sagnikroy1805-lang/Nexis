import { motion } from "framer-motion";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { ErrorBox, Loading, PageTitle, RiskBar, StatusChip } from "../components/ui";
import { STATUS_LABEL, fmtInt, fmtScore, fmtTime, fmtUsd } from "../format";
import { useApi } from "../hooks/useApi";
import type { AlertSort, AlertStatusFilter } from "../types";
import { ALERT_STATUSES } from "../types";

const PAGE_SIZES = [25, 50, 100];
const STATUS_OPTIONS: AlertStatusFilter[] = ["open", "escalated", "dismissed", "needs_info", "all"];

/** The full alert table. The dashboard (/) shows only the top five. */
export default function AlertQueue() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const status = (STATUS_OPTIONS.includes(params.get("status") as AlertStatusFilter)
    ? params.get("status")
    : "open") as AlertStatusFilter;
  const sort: AlertSort = params.get("sort") === "occurred_at" ? "occurred_at" : "risk_score";
  const size = PAGE_SIZES.includes(Number(params.get("size"))) ? Number(params.get("size")) : 25;
  const page = Math.max(1, Number(params.get("page")) || 1);

  // Only the threshold is needed here (marker on each risk bar).
  const summary = useApi(() => api.summary(), []);
  const alerts = useApi(
    () => api.alerts({ status, sort, limit: size, offset: (page - 1) * size }),
    [status, sort, size, page],
    true,
  );

  const update = (patch: Record<string, string | number>) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) next.set(k, String(v));
    setParams(next, { replace: true });
  };

  const total = alerts.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / size));
  const from = total === 0 ? 0 : (page - 1) * size + 1;
  const to = Math.min(total, page * size);
  const open = (id: number) => navigate(`/alerts/${id}`);

  return (
    <div>
      <PageTitle title="Alerts" />

      <section className="glass mt-6 min-w-0">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
          <div className="label">
            {fmtInt(total)} {status === "all" ? "total" : STATUS_LABEL[status as keyof typeof STATUS_LABEL]?.toLowerCase()}
          </div>
          <div className="flex flex-wrap items-center gap-5 text-[12px]">
            <label className="flex items-center gap-2">
              <span className="label">Status</span>
              <select className="field-line" value={status} onChange={(e) => update({ status: e.target.value, page: 1 })}>
                {ALERT_STATUSES.map((s) => (
                  <option key={s} value={s}>
                    {STATUS_LABEL[s]}
                  </option>
                ))}
                <option value="all">All</option>
              </select>
            </label>
            <label className="flex items-center gap-2">
              <span className="label">Sort</span>
              <select className="field-line" value={sort} onChange={(e) => update({ sort: e.target.value, page: 1 })}>
                <option value="risk_score">Risk score</option>
                <option value="occurred_at">Newest</option>
              </select>
            </label>
            <label className="flex items-center gap-2">
              <span className="label">Rows</span>
              <select className="field-line" value={size} onChange={(e) => update({ size: e.target.value, page: 1 })}>
                {PAGE_SIZES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>

        {alerts.error && <ErrorBox error={alerts.error} onRetry={alerts.reload} />}
        {/* Wide tables scroll inside the panel, never the page. */}
        <div className={`overflow-x-auto transition-opacity ${alerts.loading && alerts.data ? "opacity-60" : ""}`}>
          <table className="tbl tbl-click">
            <thead>
              <tr>
                <th className="r">#</th>
                <th>Risk</th>
                <th>Occurred</th>
                <th>Sender → receiver</th>
                <th className="r">Amount</th>
                <th>Format</th>
                <th>Top reasons</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {alerts.data?.items.map((a, i) => (
                <motion.tr
                  key={`${a.alert_id}-${status}-${sort}-${page}`}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35, delay: Math.min(i, 20) * 0.025, ease: [0.22, 1, 0.36, 1] }}
                  tabIndex={0}
                  onClick={() => open(a.alert_id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      open(a.alert_id);
                    }
                  }}
                  aria-label={`Alert ${a.alert_id}, rank ${a.rank}, risk score ${fmtScore(a.risk_score)}`}
                >
                  <td className="r num">{a.rank}</td>
                  <td>
                    <RiskBar score={a.risk_score} threshold={summary.data?.threshold} />
                  </td>
                  <td className="num whitespace-nowrap">{fmtTime(a.occurred_at)}</td>
                  <td className="mono leading-snug whitespace-nowrap">
                    <div>{a.src_account}</div>
                    <div>
                      <span aria-label="to">→ </span>
                      {a.dst_account}
                    </div>
                  </td>
                  <td className="r num whitespace-nowrap">{fmtUsd(a.amount_usd)}</td>
                  <td className="whitespace-nowrap">{a.payment_format}</td>
                  <td className="text-[12px] leading-snug">
                    {a.top_reasons.slice(0, 2).map((r) => (
                      <div
                        key={r.feature}
                        className="max-w-[clamp(140px,18vw,260px)] truncate"
                        title={`${r.label} (${r.feature}: ${r.contribution.toFixed(2)})`}
                      >
                        {r.label}
                      </div>
                    ))}
                  </td>
                  <td>
                    <StatusChip status={a.status} />
                  </td>
                </motion.tr>
              ))}
              {alerts.data && alerts.data.items.length === 0 && (
                <tr>
                  <td colSpan={8} className="py-8 text-center">
                    No alerts.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          {alerts.loading && !alerts.data && <Loading />}
        </div>
        <div className="mt-5 flex flex-wrap items-center justify-end gap-4 text-[12px]">
          <span className="num">
            {fmtInt(from)}–{fmtInt(to)} of {fmtInt(total)}
          </span>
          <button type="button" className="pill pill-sm" disabled={page <= 1} onClick={() => update({ page: page - 1 })}>
            ← Prev
          </button>
          <span className="num">
            {page} / {pages}
          </span>
          <button type="button" className="pill pill-sm" disabled={page >= pages} onClick={() => update({ page: page + 1 })}>
            Next →
          </button>
        </div>
      </section>
    </div>
  );
}
