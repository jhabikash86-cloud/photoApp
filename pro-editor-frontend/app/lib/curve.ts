// Point tone curves. A TypeScript copy of the server's monotone-cubic (Fritsch-Carlson) interpolation, so the curve
// drawn here is the curve that is applied to the picture.
export type Pt = number[];

export const IDENTITY: Pt[] = [[0, 0], [1, 1]];

export function isIdentity(pts: Pt[]): boolean {
  return pts.length < 2 || pts.every(([x, y]) => Math.abs(x - y) < 1e-4);
}

function pchip(xs: number[], ys: number[], xq: number): number {
  const n = xs.length;
  if (n === 1) return ys[0];
  const h: number[] = [], d: number[] = [];
  for (let i = 0; i < n - 1; i++) { h.push(Math.max(xs[i + 1] - xs[i], 1e-9)); d.push((ys[i + 1] - ys[i]) / h[i]); }
  const m = new Array(n).fill(0);
  m[0] = d[0]; m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) {
    m[i] = d[i - 1] * d[i] <= 0 ? 0 : (3 * (h[i - 1] + h[i])) / ((2 * h[i] + h[i - 1]) / d[i - 1] + (h[i] + 2 * h[i - 1]) / d[i]);
  }
  let idx = 0;
  while (idx < n - 2 && xq > xs[idx + 1]) idx++;
  const t = (xq - xs[idx]) / h[idx], t2 = t * t, t3 = t2 * t;
  const v = (2 * t3 - 3 * t2 + 1) * ys[idx] + (t3 - 2 * t2 + t) * h[idx] * m[idx] + (-2 * t3 + 3 * t2) * ys[idx + 1] + (t3 - t2) * h[idx] * m[idx + 1];
  return Math.min(1, Math.max(0, v));
}

/** The curve sampled at `n` evenly spaced inputs (used for drawing). */
export function sampleCurve(points: Pt[], n = 96): number[] {
  const clean = [...points].map(([x, y]) => [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))]).sort((a, b) => a[0] - b[0]);
  const u: Pt[] = [];
  for (const p of clean) if (!u.length || Math.abs(p[0] - u[u.length - 1][0]) > 1e-4) u.push(p);
  if (u[0][0] > 0) u.unshift([0, u[0][1]]);
  if (u[u.length - 1][0] < 1) u.push([1, u[u.length - 1][1]]);
  const xs = u.map((p) => p[0]), ys = u.map((p) => p[1]);
  return Array.from({ length: n }, (_, i) => pchip(xs, ys, i / (n - 1)));
}

export const CURVE_PRESETS: { id: string; label: string; pts: Pt[] }[] = [
  { id: 'linear', label: 'Linear', pts: [[0, 0], [1, 1]] },
  { id: 'contrast', label: 'Medium contrast', pts: [[0, 0], [0.25, 0.2], [0.75, 0.8], [1, 1]] },
  { id: 'strong', label: 'Strong contrast', pts: [[0, 0], [0.25, 0.15], [0.75, 0.86], [1, 1]] },
  { id: 'fade', label: 'Faded blacks', pts: [[0, 0.07], [0.5, 0.52], [1, 0.98]] },
  { id: 'lift', label: 'Lift shadows', pts: [[0, 0], [0.2, 0.3], [1, 1]] },
  { id: 'matte', label: 'Matte film', pts: [[0, 0.1], [0.3, 0.27], [0.75, 0.82], [1, 0.95]] },
];
