import cytoscape, { type Core, type ElementDefinition, type EventObject, type LayoutOptions } from "cytoscape";
import { useEffect, useRef, useState } from "react";

export type GraphLayout =
  | { kind: "concentric"; levels: Map<string, number> }
  | { kind: "circle" }
  | { kind: "cose" };

export interface CytoGraphProps {
  elements: ElementDefinition[];
  layout: GraphLayout;
  height?: number | string;
  selectedId?: string | null;
  ariaLabel: string;
  onNodeTap?: (id: string) => void;
  onNodeDblTap?: (id: string) => void;
  onEdgeTap?: (id: string) => void;
  onBackgroundTap?: () => void;
}

// Typed loosely: cytoscape's CSS typings do not model "data(...)" mappers well.
const STYLE = [
  {
    selector: "node",
    style: {
      "background-color": "#FFA800",
      "background-opacity": "data(fill)",
      width: "data(size)",
      height: "data(size)",
      "border-width": 1,
      "border-color": "#0B0B0B",
      label: "data(label)",
      "font-size": 9,
      "font-family": "ui-monospace, SFMono-Regular, Consolas, monospace",
      color: "#0B0B0B",
      "text-valign": "bottom",
      "text-margin-y": 4,
    },
  },
  { selector: "node[?unscored]", style: { "border-style": "dashed" } },
  { selector: "node[?focus]", style: { "border-width": 3 } },
  { selector: "node[?center]", style: { "border-width": 4 } },
  {
    selector: "edge",
    style: {
      width: "data(width)",
      "line-color": "data(color)",
      "target-arrow-color": "data(color)",
      "target-arrow-shape": "triangle",
      "arrow-scale": 0.8,
      "curve-style": "bezier",
      opacity: 0.95,
      label: "data(label)",
      "font-size": 9,
      "font-family": "Inter, system-ui, sans-serif",
      color: "#0B0B0B",
      "text-rotation": "autorotate",
      "text-margin-y": -8,
    },
  },
  { selector: "edge[?unscored]", style: { "line-style": "dashed" } },
  { selector: "edge[?dim]", style: { opacity: 0.35 } },
  {
    selector: "edge[?important]",
    style: { "underlay-color": "#FFA800", "underlay-opacity": 0.75, "underlay-padding": 3, "z-index": 10 },
  },
  { selector: "edge[?alerted]", style: { "z-index": 20 } },
  {
    selector: ":selected",
    style: { "overlay-color": "#0B0B0B", "overlay-opacity": 0.12, "overlay-padding": 5 },
  },
] as unknown as cytoscape.StylesheetJson;

function layoutOptions(layout: GraphLayout): LayoutOptions {
  if (layout.kind === "circle") return { name: "circle", padding: 24, animate: false } as LayoutOptions;
  if (layout.kind === "cose") return { name: "cose", padding: 24, animate: false } as LayoutOptions;
  const maxLevel = Math.max(0, ...layout.levels.values());
  return {
    name: "concentric",
    padding: 24,
    animate: false,
    minNodeSpacing: 18,
    concentric: (n) => maxLevel + 1 - (layout.levels.get(n.id()) ?? maxLevel + 1),
    levelWidth: () => 1,
  } as LayoutOptions;
}

interface Tip {
  x: number;
  y: number;
  text: string;
}

export default function CytoGraphImpl(props: CytoGraphProps) {
  const { elements, layout, height = 420, selectedId, ariaLabel } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const handlers = useRef(props);
  handlers.current = props;
  const [tip, setTip] = useState<Tip | null>(null);

  // Create the instance once.
  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      style: STYLE,
      minZoom: 0.2,
      maxZoom: 3,
      wheelSensitivity: 0.3,
      boxSelectionEnabled: false,
    });
    cyRef.current = cy;
    cy.on("tap", "node", (e: EventObject) => handlers.current.onNodeTap?.(e.target.id()));
    cy.on("dbltap", "node", (e: EventObject) => handlers.current.onNodeDblTap?.(e.target.id()));
    cy.on("tap", "edge", (e: EventObject) => handlers.current.onEdgeTap?.(e.target.id()));
    cy.on("tap", (e: EventObject) => {
      if (e.target === cy) handlers.current.onBackgroundTap?.();
    });
    cy.on("mouseover", "node, edge", (e: EventObject) => {
      const pos = e.renderedPosition ?? e.target.renderedMidpoint?.() ?? { x: 0, y: 0 };
      setTip({ x: pos.x, y: pos.y, text: String(e.target.data("tip") ?? e.target.id()) });
    });
    cy.on("mouseout", "node, edge", () => setTip(null));
    cy.on("viewport", () => setTip(null));
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  // Replace elements and re-run the layout when the data changes.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.batch(() => {
      cy.elements().remove();
      cy.add(elements);
    });
    cy.layout(layoutOptions(layout)).run();
    cy.fit(undefined, 24);
    setTip(null);
  }, [elements, layout]);

  // Reflect the externally selected element.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements(":selected").unselect();
    if (selectedId) cy.getElementById(selectedId).select();
  }, [selectedId, elements]);

  // Keep the canvas sized to its container, and re-fit: the first layout can
  // run before the surrounding grid has settled on its final width.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    let frame = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const cy = cyRef.current;
        if (!cy) return;
        cy.resize();
        cy.fit(undefined, 24);
      });
    });
    ro.observe(el);
    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
    };
  }, []);

  return (
    <div className="cy-wrap" style={{ height }}>
      {/* Inline position: cytoscape injects an unlayered `position: relative` for
          its container, which would beat any layered (Tailwind) stylesheet rule. */}
      <div
        ref={containerRef}
        className="cy-canvas"
        style={{ position: "absolute", inset: 0 }}
        role="img"
        aria-label={ariaLabel}
      />
      <button
        type="button"
        className="pill pill-sm absolute top-2 right-2 z-[2]"
        onClick={() => cyRef.current?.fit(undefined, 24)}
        title="Fit graph to view"
      >
        Fit
      </button>
      {tip && (
        <div className="cy-tip" style={{ left: tip.x + 12, top: tip.y + 12 }}>
          {tip.text}
        </div>
      )}
    </div>
  );
}
