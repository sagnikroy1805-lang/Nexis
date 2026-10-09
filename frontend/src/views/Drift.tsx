import { api } from "../api";
import { DriftChart } from "../components/DriftChart";
import { Card, DriftChip, ErrorBox, Figure, Loading, PageTitle } from "../components/ui";
import { DRIFT_ACTION_LABEL, fmtPct, fmtTime, snakeToWords } from "../format";
import { useApi } from "../hooks/useApi";

const METRIC_LABEL: Record<string, string> = {
  psi_max: "Feature PSI",
  score_psi: "Score PSI",
};

export default function Drift() {
  const drift = useApi(() => api.drift(), []);
  const d = drift.data;
  const counts = d
    ? d.windows.reduce<Record<string, number>>((acc, w) => ((acc[w.status] = (acc[w.status] ?? 0) + 1), acc), {})
    : {};

  return (
    <div>
      <PageTitle title="Drift" />
      {drift.error && <ErrorBox error={drift.error} onRetry={drift.reload} />}
      {drift.loading && <Loading />}
      {d && (
        <>
          <div className="glass mt-6 grid grid-cols-2 gap-8 md:grid-cols-4">
            <Figure label="Windows" value={d.windows.length} />
            <Figure label="Warn" value={counts.warn ?? 0} />
            <Figure label="Drift" value={counts.drift ?? 0} />
            <Figure
              label="Reference"
              value={
                <span className="block text-[20px] leading-tight md:text-[22px]">
                  {fmtTime(d.reference.start).slice(0, 10)}
                  <br />→ {fmtTime(d.reference.end).slice(0, 10)}
                </span>
              }
            />
          </div>

          <Card title="PSI per window" className="mt-6">
            {d.windows.length === 0 ? <p className="text-[12px]">No windows.</p> : <DriftChart windows={d.windows} />}
          </Card>

          <Card title="Events" className="mt-6">
            <div className="overflow-x-auto">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Detected</th>
                    <th>Metric</th>
                    <th className="r">Value</th>
                    <th>Level</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {d.events.map((e, i) => (
                    <tr key={i}>
                      <td className="num whitespace-nowrap">{fmtTime(e.detected_at)}</td>
                      <td>{METRIC_LABEL[e.metric] ?? e.metric}</td>
                      <td className="r num">{e.value.toFixed(3)}</td>
                      <td>
                        <DriftChip status={e.level} />
                      </td>
                      <td title={e.action}>{DRIFT_ACTION_LABEL[e.action] ?? snakeToWords(e.action)}</td>
                    </tr>
                  ))}
                  {d.events.length === 0 && (
                    <tr>
                      <td colSpan={5} className="py-6 text-center">
                        No events.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Windows" className="mt-6">
            <div className="overflow-x-auto">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Window</th>
                    <th className="r">Feature PSI</th>
                    <th>Top features</th>
                    <th className="r">Score PSI</th>
                    <th className="r">Alert rate</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {d.windows.map((w) => (
                    <tr key={w.start}>
                      <td className="num whitespace-nowrap">
                        {fmtTime(w.start)} → {fmtTime(w.end).slice(11)}
                      </td>
                      <td className="r num">{w.psi_max.toFixed(3)}</td>
                      <td>
                        <div className="flex flex-wrap gap-x-3 gap-y-0.5">
                          {Object.entries(w.psi_top)
                            .sort((a, b) => b[1] - a[1])
                            .map(([f, v]) => (
                              <span key={f} className="whitespace-nowrap">
                                <span className="mono text-[11.5px]">{f}</span> <span className="num text-[11.5px]">{v.toFixed(3)}</span>
                              </span>
                            ))}
                        </div>
                      </td>
                      <td className="r num">{w.score_psi.toFixed(3)}</td>
                      <td className="r num">{fmtPct(w.alert_rate, 3)}</td>
                      <td>
                        <DriftChip status={w.status} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
