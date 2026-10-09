import { motion } from "framer-motion";
import { Fragment } from "react";
import { api } from "../api";
import { ErrorBox, Figure, Loading, MeanStd, PageTitle } from "../components/ui";
import { RUNG_NAME, fmtPct } from "../format";
import { useApi } from "../hooks/useApi";

const MIN_SEEDS = 5;

export default function Models() {
  const models = useApi(() => api.models(), []);
  const d = models.data;
  const rungs = d ? [...new Set(d.items.map((m) => m.rung))].sort((a, b) => a - b) : [];
  const best = d ? Math.max(...d.items.map((m) => m.pr_auc)) : null;
  const missingStd = d?.items.some(
    (m) => m.roc_auc_std === undefined || m.recall_at_fpr_1e3_std === undefined || m.precision_at_budget_std === undefined,
  );
  const fewSeeds = d?.items.some((m) => m.n_seeds < MIN_SEEDS);

  return (
    <div>
      <PageTitle title="Models" />
      {models.error && <ErrorBox error={models.error} onRetry={models.reload} />}
      {models.loading && <Loading />}
      {d && (
        <>
          <div className="glass mt-6 grid grid-cols-2 gap-8 md:grid-cols-4">
            <Figure label="Prevalence" value={fmtPct(d.prevalence, 3)} sub="random-scorer PR-AUC" />
            <Figure label="Models" value={d.items.length} />
            <Figure label="Rungs" value={rungs.length} />
          </div>

          <div className="glass mt-6">
          <div className="overflow-x-auto">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="r">PR-AUC</th>
                  <th className="r">÷ prev.</th>
                  <th className="r">ROC-AUC</th>
                  <th className="r">Recall @ FPR 1e-3</th>
                  <th className="r">Precision @ budget</th>
                  <th className="r">Seeds</th>
                </tr>
              </thead>
              <tbody>
                {rungs.map((rung) => (
                  <Fragment key={rung}>
                    <tr className="hover:!bg-transparent">
                      <th colSpan={7} scope="colgroup" className="!border-b-0 !px-0.5 !pt-7 !pb-1.5 text-left !tracking-normal !normal-case">
                        <span className="font-display text-[26px] font-normal">
                          {rung}. {RUNG_NAME[rung] ?? "Other"}
                        </span>
                      </th>
                    </tr>
                    {d.items
                      .filter((m) => m.rung === rung)
                      .map((m, i) => (
                        <motion.tr
                          key={`${rung}-${m.name}`}
                          initial={{ opacity: 0, y: 6 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.35, delay: (rung - 1) * 0.08 + i * 0.03 }}
                        >
                          <td className="mono whitespace-nowrap">
                            {m.pr_auc === best && (
                              <span className="mr-1.5 inline-block h-2 w-2 rounded-full bg-gold ring-1 ring-ink" title="Highest PR-AUC" />
                            )}
                            {m.name}
                          </td>
                          <td className="r">
                            {/* Near-prevalence values need more decimals to stay distinguishable. */}
                            <MeanStd mean={m.pr_auc} std={m.pr_auc_std} digits={m.pr_auc < 0.01 ? 5 : 3} />
                          </td>
                          <td className="r num">{d.prevalence > 0 ? `${(m.pr_auc / d.prevalence).toFixed(1)}×` : "—"}</td>
                          <td className="r">
                            <MeanStd mean={m.roc_auc} std={m.roc_auc_std} digits={3} />
                          </td>
                          <td className="r">
                            <MeanStd mean={m.recall_at_fpr_1e3} std={m.recall_at_fpr_1e3_std} digits={3} />
                          </td>
                          <td className="r">
                            <MeanStd mean={m.precision_at_budget} std={m.precision_at_budget_std} digits={3} />
                          </td>
                          <td className="r num">
                            {m.n_seeds < MIN_SEEDS && <span title={`Fewer than ${MIN_SEEDS} seeds`}>⚠ </span>}
                            {m.n_seeds}
                          </td>
                        </motion.tr>
                      ))}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 flex flex-col gap-0.5 border-t border-ink/10 pt-3 text-[11px]">
            <p>A difference smaller than the std is not a result.</p>
            {missingStd && <p>* std not provided for this metric.</p>}
            {fewSeeds && <p>⚠ fewer than {MIN_SEEDS} seeds.</p>}
          </div>
          </div>
        </>
      )}
    </div>
  );
}
