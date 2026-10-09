import { motion } from "framer-motion";
import { fmtNum, fmtSigned } from "../format";
import type { FeatureContribution } from "../types";

/**
 * Diverging bars, sorted by magnitude: gold = pushed the risk score up,
 * black = pushed it down. Each row also carries its signed value as text.
 */
export function ContributionChart({ items }: { items: FeatureContribution[] }) {
  if (items.length === 0) return <p className="text-[12px]">No contributions.</p>;
  const rows = [...items].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const max = Math.max(...rows.map((r) => Math.abs(r.contribution)), 1e-9);

  return (
    <div>
      <div className="mb-2 flex gap-5 text-[11px]">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-gold ring-1 ring-ink" aria-hidden="true" /> up
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-ink" aria-hidden="true" /> down
        </span>
      </div>
      <ul>
        {rows.map((r, i) => {
          const w = (Math.abs(r.contribution) / max) * 50;
          const up = r.contribution > 0;
          return (
            <li
              key={r.feature}
              className="grid grid-cols-[minmax(0,1.3fr)_minmax(90px,1fr)_48px] items-center gap-3 border-b border-ink/20 py-1.5"
              title={`${r.feature} = ${fmtNum(r.value)}`}
            >
              <div className="min-w-0">
                <div className="truncate text-[13px]">{r.label}</div>
                <div className="mono truncate text-[10.5px]">
                  {r.feature} = {fmtNum(r.value)}
                </div>
              </div>
              <div className="relative h-3" aria-hidden="true">
                <span className="absolute -inset-y-1 left-1/2 w-px bg-ink/50" />
                <motion.span
                  className={`absolute inset-y-0 ${up ? "rounded-r-full bg-gold ring-1 ring-ink/50" : "rounded-l-full bg-ink"}`}
                  style={up ? { left: "50%" } : { right: "50%" }}
                  initial={{ width: 0 }}
                  animate={{ width: `${w}%` }}
                  transition={{ duration: 0.6, delay: 0.03 * i, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
              <div className="num text-right text-[12.5px]">{fmtSigned(r.contribution)}</div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
