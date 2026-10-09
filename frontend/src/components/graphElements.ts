/**
 * Converts contract graph payloads into cytoscape element definitions.
 *
 * Encodings (the same in every graph view):
 *   node size + gold fill strength -> account risk (hollow = no scored activity)
 *   thick black outline            -> centre / focus account
 *   edge colour                    -> payment risk_score (white low -> black high)
 *   edge width                     -> amount (log scale)
 */
import type { ElementDefinition } from "cytoscape";
import { edgeRiskColor, fmtScore, fmtTime, fmtUsd, riskFillOpacity } from "../format";
import type { GraphEdge } from "../types";

export interface NodeInput {
  id: string;
  risk: number | null;
  center?: boolean;
  focus?: boolean;
}

export interface EdgeInput extends GraphEdge {
  important?: boolean;
  alerted?: boolean;
}

export function edgeWidth(amountUsd: number): number {
  return Math.max(0.8, Math.min(7, 0.6 + 1.3 * Math.log10(1 + amountUsd / 100)));
}

function nodeSize(risk: number | null): number {
  return 14 + 30 * (risk ?? 0);
}

export interface BuildOptions {
  /** "all" labels every node; "key" only the centre/focus nodes. */
  labels?: "all" | "key";
  /** De-emphasise edges not flagged important (evidence graph). */
  dimUnimportant?: boolean;
}

export function buildElements(nodes: NodeInput[], edges: EdgeInput[], opts: BuildOptions = {}): ElementDefinition[] {
  const byId = new Map<string, NodeInput>();
  for (const n of nodes) byId.set(n.id, n);
  // Edge endpoints must exist as nodes even if the payload omitted them.
  for (const e of edges) {
    for (const id of [e.src, e.dst]) if (!byId.has(id)) byId.set(id, { id, risk: null });
  }
  const labels = opts.labels ?? "key";
  const els: ElementDefinition[] = [];
  for (const n of byId.values()) {
    const key = !!(n.center || n.focus);
    els.push({
      group: "nodes",
      data: {
        id: n.id,
        label: labels === "all" || key ? n.id : "",
        fill: riskFillOpacity(n.risk),
        size: nodeSize(n.risk),
        risk: n.risk,
        unscored: n.risk === null,
        center: !!n.center,
        focus: !!n.focus,
        tip: `${n.id}\nrisk ${n.risk === null ? "not scored" : fmtScore(n.risk)}${n.center ? "\ncentre" : ""}`,
      },
    });
  }
  for (const e of edges) {
    els.push({
      group: "edges",
      data: {
        id: e.tx_id,
        source: e.src,
        target: e.dst,
        color: edgeRiskColor(e.risk_score),
        width: edgeWidth(e.amount_usd) + (e.important ? 1 : 0),
        risk: e.risk_score,
        unscored: e.risk_score === null,
        important: !!e.important,
        dim: !!opts.dimUnimportant && !e.important,
        alerted: !!e.alerted,
        label: e.alerted ? "alerted" : "",
        tip: `${e.tx_id}\n${fmtUsd(e.amount_usd)} · ${e.payment_format}\n${fmtTime(e.occurred_at)}\nrisk ${e.risk_score === null ? "not scored" : fmtScore(e.risk_score)}`,
      },
    });
  }
  return els;
}

/** Undirected hop distance from a set of root nodes. */
export function hopDistances(roots: string[], edges: { src: string; dst: string }[]): Map<string, number> {
  const adj = new Map<string, string[]>();
  for (const e of edges) {
    (adj.get(e.src) ?? adj.set(e.src, []).get(e.src)!).push(e.dst);
    (adj.get(e.dst) ?? adj.set(e.dst, []).get(e.dst)!).push(e.src);
  }
  const dist = new Map<string, number>();
  let frontier = [...roots];
  frontier.forEach((r) => dist.set(r, 0));
  let d = 0;
  while (frontier.length) {
    d += 1;
    const next: string[] = [];
    for (const a of frontier) {
      for (const b of adj.get(a) ?? []) {
        if (!dist.has(b)) {
          dist.set(b, d);
          next.push(b);
        }
      }
    }
    frontier = next;
  }
  return dist;
}
