import { Suspense, lazy, type ReactNode } from "react";
import type { CytoGraphProps } from "./CytoGraphImpl";
import { Loading } from "./ui";

export type { CytoGraphProps, GraphLayout } from "./CytoGraphImpl";

// cytoscape is ~400 kB; load it only when a graph is actually on screen.
const Impl = lazy(() => import("./CytoGraphImpl"));

export function CytoGraph(props: CytoGraphProps) {
  return (
    <Suspense
      fallback={
        <div className="cy-wrap" style={{ height: props.height ?? 420 }}>
          <Loading label="Loading graph" />
        </div>
      }
    >
      <Impl {...props} />
    </Suspense>
  );
}

/** Compact legend shared by every graph view. */
export function GraphLegend({ extra }: { extra?: ReactNode }) {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1 text-[11px]">
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-2 w-2 rounded-full border border-ink bg-gold/40" aria-hidden="true" />
        <span className="inline-block h-3.5 w-3.5 rounded-full border border-ink bg-gold" aria-hidden="true" />
        size = risk
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-3 w-3 rounded-full border border-dashed border-ink" aria-hidden="true" />
        not scored
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-3 w-3 rounded-full border-[3px] border-ink" aria-hidden="true" />
        centre
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-px w-5 bg-[#c8c8c8]" aria-hidden="true" />
        <span className="inline-block h-[2px] w-5 bg-ink" aria-hidden="true" />
        edge low → high risk · width = amount
      </span>
      {extra}
    </div>
  );
}
