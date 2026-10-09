import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { DRIFT_LABEL, STATUS_LABEL, decimalsFor, fmtScore } from "../format";
import type { AlertStatus, DriftStatus } from "../types";

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-6 text-[12px]" role="status" aria-live="polite">
      <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-gold ring-1 ring-ink" aria-hidden="true" />
      {label}…
    </div>
  );
}

export function ErrorBox({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div className="glass flex flex-wrap items-center gap-3 !rounded-2xl !px-4 !py-2.5 text-[13px]" role="alert">
      <span className="font-medium">Request failed.</span>
      <span>{error}</span>
      {onRetry && (
        <button type="button" className="pill pill-sm ml-auto" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

/**
 * A content block. Inner pages use frosted-glass panels ("glass", default) so
 * body text never sits directly on the vermilion; "plain" is a thin top rule.
 */
export function Card({
  title,
  aside,
  children,
  className = "",
  variant = "glass",
}: {
  title?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  variant?: "glass" | "plain" | "cream";
}) {
  const shell = variant === "glass" ? "glass" : variant === "cream" ? "panel-cream" : "border-t border-ink pt-3";
  return (
    <section className={`${shell} min-w-0 ${className}`}>
      {(title || aside) && (
        <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
          {title && <h2 className="label">{title}</h2>}
          {aside && <div className="flex items-center gap-2 text-[12px]">{aside}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

/**
 * Large didone page title on the vermilion. Any small text (back link, meta,
 * status, language note) goes on a slim glass bar beneath it.
 */
export function PageTitle({
  title,
  meta,
  children,
  note,
  back,
}: {
  title: ReactNode;
  meta?: ReactNode;
  children?: ReactNode;
  note?: string;
  back?: { to: string; label: string };
}) {
  const bar = meta || children || note || back;
  return (
    <div className="pb-2">
      <h1 className="display-sturdy text-[clamp(44px,7vw,88px)] leading-[1.02] font-medium tracking-tight">{title}</h1>
      {bar && (
        <div className="glass mt-4 !rounded-2xl !px-4 !py-2.5 text-[12px]">
          {(meta || children || back) && (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
              {back && (
                <Link to={back.to} className="font-medium hover:underline">
                  ← {back.label}
                </Link>
              )}
              {meta && <div className="min-w-0">{meta}</div>}
              {children && <div className="ml-auto flex items-center gap-2">{children}</div>}
            </div>
          )}
          {note && (
            <div className={meta || children || back ? "mt-1.5 border-t border-ink/10 pt-1.5" : ""}>
              <LanguageNote note={note} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const STATUS_STYLE: Record<AlertStatus, string> = {
  open: "border-ink",
  escalated: "border-ink bg-ink text-cream",
  dismissed: "border-ink border-dashed",
  needs_info: "border-ink bg-gold",
};

export function StatusChip({ status }: { status: AlertStatus }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-px text-[10.5px] font-medium tracking-wide uppercase whitespace-nowrap ${STATUS_STYLE[status] ?? "border-ink"}`}
    >
      {STATUS_LABEL[status] ?? status}
    </span>
  );
}

const DRIFT_STYLE: Record<DriftStatus, string> = {
  ok: "border-ink",
  warn: "border-ink bg-gold",
  drift: "border-ink bg-ink text-cream",
};
const DRIFT_ICON: Record<DriftStatus, string> = { ok: "●", warn: "▲", drift: "◆" };

/** Drift status: icon + label + fill, never colour alone. */
export function DriftChip({ status }: { status: string }) {
  const s = (status in DRIFT_LABEL ? status : "warn") as DriftStatus;
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-px text-[10.5px] font-medium tracking-wide uppercase ${DRIFT_STYLE[s]}`}
    >
      <span aria-hidden="true">{DRIFT_ICON[s]}</span>
      {DRIFT_LABEL[s] ?? status}
    </span>
  );
}

/** Golden risk bar with the number beside it (the bar repeats the number). */
export function RiskBar({ score, threshold }: { score: number | null; threshold?: number }) {
  const pct = score === null ? 0 : Math.max(0, Math.min(1, score)) * 100;
  return (
    <span className="inline-flex items-center gap-2" title={score === null ? "Not scored" : `Risk score ${fmtScore(score)}`}>
      <span className="relative h-1.5 w-16 rounded-full bg-ink/20" aria-hidden="true">
        <span className="absolute inset-y-0 left-0 rounded-full bg-gold ring-1 ring-ink/60" style={{ width: `${pct}%` }} />
        {threshold !== undefined && (
          <span className="absolute -inset-y-1 w-px bg-ink" style={{ left: `${threshold * 100}%` }} />
        )}
      </span>
      <span className="num text-[12.5px]">{fmtScore(score)}</span>
    </span>
  );
}

/** A large serif figure with a small label. */
export function Figure({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="label">{label}</div>
      <div className="num display-sturdy mt-1 text-[clamp(28px,3vw,46px)] leading-[1.05] font-medium">{value}</div>
      {sub && <div className="mt-1.5 text-[12px]">{sub}</div>}
    </div>
  );
}

/** One small line, on every evidence view (CLAUDE.md rule 5). */
export function LanguageNote({ note }: { note: string }) {
  return (
    <p className="text-[11px] leading-snug" role="note">
      <span aria-hidden="true">※ </span>
      {note}
    </p>
  );
}

/** Used where the payload carries no language_note of its own. */
export const DEFAULT_LANGUAGE_NOTE =
  "Risk scores reflect model output on recorded data. They are not findings of wrongdoing.";

export function AccountLink({ id, until }: { id: string; until?: string }) {
  const to = `/graph/${encodeURIComponent(id)}${until ? `?until=${encodeURIComponent(until)}` : ""}`;
  return (
    <Link
      className="mono underline decoration-ink/40 underline-offset-2 hover:decoration-ink"
      to={to}
      title="Open in graph"
      onClick={(e) => e.stopPropagation()}
    >
      {id}
    </Link>
  );
}

export function TxChip({ id, tone = "default", title }: { id: string; tone?: "default" | "bad"; title?: string }) {
  return (
    <span
      className={`mono inline-flex items-center gap-1 rounded-full border px-2 py-px text-[11px] whitespace-nowrap ${
        tone === "bad" ? "border-ink bg-ink text-gold" : "border-ink/50"
      }`}
      title={title}
    >
      {tone === "bad" && <span aria-hidden="true">⚠</span>}
      {id}
    </span>
  );
}

/** mean ± std. A missing std is flagged, not hidden (CLAUDE.md rule 3). */
export function MeanStd({ mean, std, digits }: { mean: number | null | undefined; std?: number | null; digits?: number }) {
  if (mean === null || mean === undefined) return <span>—</span>;
  const d = digits ?? decimalsFor(std ?? 0, 3);
  return (
    <span className="num whitespace-nowrap">
      {mean.toFixed(d)}
      <span className="text-[0.72em]">
        {" ± "}
        {std === null || std === undefined ? <span title="No std provided">n/a*</span> : std.toFixed(d)}
      </span>
    </span>
  );
}
