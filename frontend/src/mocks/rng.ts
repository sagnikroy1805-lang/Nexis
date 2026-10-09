/** Deterministic helpers so the demo fixtures are identical on every load. */

export type Rng = () => number;

/** mulberry32: small, fast, seedable PRNG returning floats in [0, 1). */
export function makeRng(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function between(rng: Rng, lo: number, hi: number): number {
  return lo + (hi - lo) * rng();
}

export function intBetween(rng: Rng, lo: number, hiInclusive: number): number {
  return Math.floor(between(rng, lo, hiInclusive + 1));
}

export function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length)];
}

export function weighted<T>(rng: Rng, items: readonly (readonly [T, number])[]): T {
  const total = items.reduce((s, [, w]) => s + w, 0);
  let r = rng() * total;
  for (const [item, w] of items) {
    r -= w;
    if (r <= 0) return item;
  }
  return items[items.length - 1][0];
}

/** Standard normal via Box-Muller. */
export function normal(rng: Rng): number {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function round(x: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(x * f) / f;
}

// ---------------------------------------------------------------------------
// Time. Dataset timestamps are naive wall-clock strings; treat them as UTC
// internally so arithmetic never depends on the viewer's time zone.
// ---------------------------------------------------------------------------

export const MINUTE_MS = 60_000;

export function parseIso(iso: string): number {
  const hasZone = /(Z|[+-]\d\d:?\d\d)$/.test(iso);
  const ms = Date.parse(hasZone ? iso : `${iso}Z`);
  if (Number.isNaN(ms)) throw new Error(`Invalid timestamp: ${iso}`);
  return ms;
}

export function toIso(ms: number): string {
  return new Date(ms).toISOString().slice(0, 19);
}
