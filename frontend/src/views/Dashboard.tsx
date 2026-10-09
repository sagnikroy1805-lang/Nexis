import { motion } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import { ErrorBox, Figure, Loading, MeanStd, RiskBar } from "../components/ui";
import { fmtInt, fmtPct, fmtScore, fmtTime, fmtUsd } from "../format";
import { useApi } from "../hooks/useApi";
import type { Summary } from "../types";

function Hero() {
  return (
    <section className="relative pt-2">
      <p className="font-display text-[17px] leading-snug text-white md:text-[19px]">
        Coordinated risk,
        <br />
        surfaced early.
      </p>
      {/* `isolate` keeps the circle (z-0) strictly behind the letters (z-10). */}
      <div className="relative isolate mt-8 select-none">
        <motion.span
          aria-hidden="true"
          className="absolute top-[2%] left-[41%] z-0 block aspect-square w-[clamp(104px,17vw,270px)] rounded-full bg-gold"
          animate={{ y: [0, -10, 0], scale: [1, 1.03, 1] }}
          transition={{ duration: 9, repeat: Infinity, ease: "easeInOut" }}
        />
        <h1 className="display-sturdy relative z-10 text-[clamp(64px,20vw,280px)] leading-[0.9] font-semibold tracking-[-0.02em] uppercase">
          Nexis
        </h1>
        <div className="display-sturdy relative z-10 pl-[5vw] text-[clamp(34px,7.2vw,124px)] leading-[1] font-medium tracking-[-0.015em]">
          Risk Intelligence
        </div>
      </div>
    </section>
  );
}

function Kpis({ s }: { s: Summary }) {
  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-x-8 gap-y-8">
      <Figure label="Transactions scored" value={fmtInt(s.transactions_scored)} />
      <Figure
        label="Open alerts"
        value={
          <>
            {fmtInt(s.alerts_open)}
            <span className="text-[0.5em]"> / {fmtInt(s.alerts_total)}</span>
          </>
        }
      />
      <Figure label="Threshold" value={fmtScore(s.threshold)} />
      <Figure
        label="PR-AUC"
        value={<MeanStd mean={s.metrics.pr_auc} std={s.metrics.pr_auc_std} />}
        sub={<>prevalence {fmtPct(s.metrics.prevalence, 3)}</>}
      />
    </div>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const summary = useApi(() => api.summary(), []);
  const top = useApi(() => api.alerts({ status: "open", sort: "risk_score", limit: 5 }), []);

  return (
    <div>
      <Hero />
      <div className="mt-14">
        {summary.error && <ErrorBox error={summary.error} onRetry={summary.reload} />}
        {summary.data ? <Kpis s={summary.data} /> : !summary.error && <Loading />}
      </div>

      <section className="mt-16 border-t border-ink pt-4">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-4">
          <h2 className="display-sturdy text-[34px] leading-none font-medium">Highest risk</h2>
          <Link to="/alerts" className="pill pill-solid">
            All alerts →
          </Link>
        </div>
        {top.error && <ErrorBox error={top.error} onRetry={top.reload} />}
        {top.loading && <Loading />}
        {top.data && (
          <ul>
            {top.data.items.map((a, i) => (
              <motion.li
                key={a.alert_id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, delay: i * 0.05, ease: [0.22, 1, 0.36, 1] }}
              >
                <button
                  type="button"
                  onClick={() => navigate(`/alerts/${a.alert_id}`)}
                  className="grid w-full grid-cols-[28px_minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 border-b border-ink/25 px-1 py-2.5 text-left text-[13px] transition-colors hover:bg-white/12 sm:grid-cols-[28px_150px_minmax(0,1fr)_120px_130px]"
                  aria-label={`Alert ${a.alert_id}, risk score ${fmtScore(a.risk_score)}`}
                >
                  <span className="num">{a.rank}</span>
                  <RiskBar score={a.risk_score} threshold={summary.data?.threshold} />
                  <span className="mono hidden truncate sm:block">
                    {a.src_account} → {a.dst_account}
                  </span>
                  <span className="num hidden text-right sm:block">{fmtUsd(a.amount_usd)}</span>
                  <span className="num text-right whitespace-nowrap">{fmtTime(a.occurred_at)}</span>
                </button>
              </motion.li>
            ))}
            {top.data.items.length === 0 && <li className="py-6 text-[13px]">No open alerts.</li>}
          </ul>
        )}
      </section>
    </div>
  );
}
