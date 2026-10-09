import { motion } from "framer-motion";
import { useState, type MouseEvent } from "react";
import { DRIFT_LABEL, fmtTime } from "../format";
import { useElementWidth } from "../hooks/useApi";
import type { DriftStatus, DriftWindow } from "../types";

export const WARN_LEVEL = 0.1;
export const DRIFT_LEVEL = 0.25;

// Primary series gold, secondary black (theme); both are PSI on one axis.
const SERIES = [
  { key: "psi_max" as const, label: "Feature PSI (max)", short: "feature", color: "#FFA800" },
  { key: "score_psi" as const, label: "Score PSI", short: "score", color: "#0B0B0B" },
];
const STATUS_ICON: Record<DriftStatus, string> = { ok: "●", warn: "▲", drift: "◆" };
const STATUS_FILL: Record<DriftStatus, string> = { ok: "#FFFFFF", warn: "#FFA800", drift: "#0B0B0B" }; // stroked black, readable on white

const H = 320;
const M = { top: 16, right: 92, bottom: 54, left: 40 };

function tickLabel(w: DriftWindow, i: number, all: DriftWindow[]): string {
  const t = fmtTime(w.end);
  const prev = i > 0 ? fmtTime(all[i - 1].end) : "";
  // Date when it changes, otherwise just the time.
  return i === 0 || prev.slice(0, 10) !== t.slice(0, 10) ? `${t.slice(5, 10)} ${t.slice(11)}` : t.slice(11);
}

export function DriftChart({ windows }: { windows: DriftWindow[] }) {
  const [ref, width] = useElementWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const W = Math.max(width, 320);
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;
  const maxVal = Math.max(...windows.flatMap((w) => [w.psi_max, w.score_psi]), DRIFT_LEVEL);
  const yMax = Math.ceil((maxVal * 1.15) / 0.05) * 0.05;
  const n = windows.length;
  const x = (i: number) => M.left + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v: number) => M.top + ih - (v / yMax) * ih;
  const ticks: number[] = [];
  for (let v = 0; v <= yMax + 1e-9; v += 0.05) ticks.push(Math.round(v * 100) / 100);
  const labelEvery = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(iw / 70))));

  const onMove = (e: MouseEvent<SVGRectElement>) => {
    const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    const i = Math.round(((e.clientX - rect.left - M.left) / iw) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, i)));
  };

  const hw = hover !== null ? windows[hover] : null;
  const last = windows[n - 1];

  return (
    <div className="drift-chart relative w-full" ref={ref}>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-1 text-[11px]">
        {SERIES.map((s) => (
          <span key={s.key} className="inline-flex items-center gap-1.5">
            <span className="inline-block h-[2px] w-5" style={{ background: s.color }} aria-hidden="true" /> {s.label}
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-5 border-t-[1.5px] border-dotted border-[#8a8a8a]" aria-hidden="true" /> warn{" "}
          {WARN_LEVEL.toFixed(2)}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-5 border-t-[1.5px] border-dashed border-ink" aria-hidden="true" /> drift{" "}
          {DRIFT_LEVEL.toFixed(2)}
        </span>
        <span>● ok ▲ warn ◆ drift</span>
      </div>
      {width > 0 && n > 0 && (
        <svg
          width={W}
          height={H}
          role="img"
          aria-label="PSI per window: feature PSI maximum and score PSI, with warn and drift reference lines"
          className="block"
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={M.left} x2={M.left + iw} y1={y(t)} y2={y(t)} className="grid" />
              <text x={M.left - 8} y={y(t)} className="axis-text" textAnchor="end" dominantBaseline="middle">
                {t.toFixed(2)}
              </text>
            </g>
          ))}
          <line x1={M.left} x2={M.left + iw} y1={y(0)} y2={y(0)} className="baseline" />

          <line x1={M.left} x2={M.left + iw} y1={y(WARN_LEVEL)} y2={y(WARN_LEVEL)} className="ref-warn" />
          <text x={M.left + iw + 8} y={y(WARN_LEVEL)} className="ref-text" dominantBaseline="middle">
            warn
          </text>
          <line x1={M.left} x2={M.left + iw} y1={y(DRIFT_LEVEL)} y2={y(DRIFT_LEVEL)} className="ref-drift" />
          <text x={M.left + iw + 8} y={y(DRIFT_LEVEL)} className="ref-text" dominantBaseline="middle">
            drift
          </text>

          {windows.map((w, i) =>
            i % labelEvery === 0 || i === n - 1 ? (
              <text key={w.end} x={x(i)} y={M.top + ih + 16} className="axis-text" textAnchor="middle">
                {tickLabel(w, i, windows)}
              </text>
            ) : null,
          )}

          {/* status per window: icon + fill */}
          {windows.map((w, i) => (
            <text
              key={`s-${w.end}`}
              x={x(i)}
              y={M.top + ih + 36}
              textAnchor="middle"
              fontSize={11}
              fill={STATUS_FILL[w.status] ?? "#0B0B0B"}
              stroke="#0B0B0B"
              strokeWidth={w.status === "drift" ? 0 : 0.6}
            >
              {STATUS_ICON[w.status] ?? "?"}
            </text>
          ))}

          {SERIES.map((s, si) => (
            <g key={s.key}>
              <motion.polyline
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeLinejoin="round"
                points={windows.map((w, i) => `${x(i)},${y(w[s.key])}`).join(" ")}
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 1.1, delay: 0.15 * si, ease: [0.22, 1, 0.36, 1] }}
              />
              {windows.map((w, i) => (
                <circle
                  key={i}
                  cx={x(i)}
                  cy={y(w[s.key])}
                  r={hover === i ? 5 : 3.5}
                  fill={s.color}
                  stroke="#0B0B0B"
                  strokeWidth={s.key === "psi_max" ? 1 : 0}
                />
              ))}
              <text x={x(n - 1) + 10} y={y(last[s.key]) + (si === 0 ? -7 : 7)} className="direct-label" dominantBaseline="middle">
                {s.short} {last[s.key].toFixed(2)}
              </text>
            </g>
          ))}

          {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={M.top} y2={M.top + ih} className="crosshair" />}
          <rect
            x={M.left - 10}
            y={M.top}
            width={iw + 20}
            height={ih}
            fill="transparent"
            onMouseMove={onMove}
            onMouseLeave={() => setHover(null)}
          />
        </svg>
      )}
      {hw && hover !== null && (
        <div
          className="pointer-events-none absolute min-w-[190px] rounded-xl bg-white p-3 text-[12px] shadow-[0_12px_30px_-12px_rgba(0,0,0,0.45)] ring-1 ring-black/10"
          style={{ left: Math.min(x(hover) + 14, W - 210), top: 40 }}
          role="status"
        >
          <div className="num mb-1 font-medium">
            {fmtTime(hw.start)} → {fmtTime(hw.end).slice(11)}
          </div>
          <div className="num">feature {hw.psi_max.toFixed(3)}</div>
          <div className="num">score {hw.score_psi.toFixed(3)}</div>
          <div>
            {STATUS_ICON[hw.status]} {DRIFT_LABEL[hw.status] ?? hw.status}
          </div>
        </div>
      )}
    </div>
  );
}
