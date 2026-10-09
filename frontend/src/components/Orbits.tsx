import { useEffect, useState } from "react";

/** True when the viewer asked the OS for reduced motion. */
export function usePrefersReducedMotion(): boolean {
  const query = "(prefers-reduced-motion: reduce)";
  const [reduced, setReduced] = useState(() => typeof window !== "undefined" && window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setReduced(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}

interface Orbit {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  rot: number;
}

// Overlapping ellipses sweeping across a 1600x1000 canvas (sliced to cover).
const ORBITS: Orbit[] = [
  { cx: 820, cy: 520, rx: 780, ry: 300, rot: -12 },
  { cx: 820, cy: 520, rx: 620, ry: 225, rot: -12 },
  { cx: 700, cy: 470, rx: 920, ry: 430, rot: 7 },
  { cx: 1000, cy: 560, rx: 540, ry: 190, rot: -27 },
  { cx: 620, cy: 420, rx: 1040, ry: 540, rot: -4 },
  { cx: 900, cy: 600, rx: 360, ry: 140, rot: 18 },
];

interface Planet {
  orbit: number;
  r: number;
  /** Start position on the orbit, 0..1. */
  phase: number;
  /** Seconds per revolution (slow). */
  period: number;
}

const PLANETS: Planet[] = [
  { orbit: 0, r: 64, phase: 0.08, period: 260 },
  { orbit: 2, r: 30, phase: 0.55, period: 340 },
  { orbit: 3, r: 18, phase: 0.3, period: 200 },
  { orbit: 5, r: 11, phase: 0.75, period: 150 },
];

function ellipsePath(o: Orbit): string {
  // Two arcs starting at the rightmost point; SMIL motion follows it.
  return `M ${o.cx + o.rx} ${o.cy} A ${o.rx} ${o.ry} 0 1 1 ${o.cx - o.rx} ${o.cy} A ${o.rx} ${o.ry} 0 1 1 ${o.cx + o.rx} ${o.cy}`;
}

function pointOn(o: Orbit, phase: number): [number, number] {
  const t = phase * 2 * Math.PI;
  const x = o.rx * Math.cos(t);
  const y = -o.ry * Math.sin(t);
  const a = (o.rot * Math.PI) / 180;
  return [o.cx + x * Math.cos(a) - y * Math.sin(a), o.cy + x * Math.sin(a) + y * Math.cos(a)];
}

/** Fixed decorative background: thin white orbit lines and golden circles. */
export function Orbits() {
  const reduced = usePrefersReducedMotion();
  return (
    <svg
      className="pointer-events-none fixed inset-0 z-0 h-full w-full"
      viewBox="0 0 1600 1000"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      {ORBITS.map((o, i) => (
        <ellipse
          key={i}
          cx={o.cx}
          cy={o.cy}
          rx={o.rx}
          ry={o.ry}
          transform={`rotate(${o.rot} ${o.cx} ${o.cy})`}
          fill="none"
          stroke="#ffffff"
          strokeOpacity={0.25}
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
      ))}
      {PLANETS.map((p, i) => {
        const o = ORBITS[p.orbit];
        if (reduced) {
          const [x, y] = pointOn(o, p.phase);
          return <circle key={i} cx={x} cy={y} r={p.r} fill="#FFA800" />;
        }
        return (
          <g key={i} transform={`rotate(${o.rot} ${o.cx} ${o.cy})`}>
            <circle r={p.r} fill="#FFA800">
              <animateMotion
                dur={`${p.period}s`}
                begin={`-${(p.phase * p.period).toFixed(1)}s`}
                repeatCount="indefinite"
                path={ellipsePath(o)}
              />
              <animate
                attributeName="r"
                values={`${p.r};${(p.r * 1.07).toFixed(1)};${p.r}`}
                dur={`${12 + i * 3}s`}
                repeatCount="indefinite"
              />
            </circle>
          </g>
        );
      })}
    </svg>
  );
}
