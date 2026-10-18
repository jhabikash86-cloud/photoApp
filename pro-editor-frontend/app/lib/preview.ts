import { Params } from './params';

/**
 * Instant feedback while a slider moves. The server render takes a few hundred milliseconds, so for the three
 * sliders everyone reaches for first (exposure, contrast, colour) the picture on screen is nudged straight away with a
 * CSS filter that approximates the change since the last real render. The real render then replaces it, and the filter
 * disappears at the same moment, because it is always computed against the parameters of the picture currently shown.
 */
export function instantFilter(cur: Params, shown?: Params): string | undefined {
  if (!shown) return undefined;
  const b = cur.basic, s = shown.basic;
  const ev = ((b.exposure - s.exposure) / 100) * 3;
  const con = (b.contrast - s.contrast) / 100;
  const sat = (b.saturation - s.saturation + 0.6 * (b.vibrance - s.vibrance)) / 100;
  const parts: string[] = [];
  if (Math.abs(ev) > 0.01) parts.push(`brightness(${Math.pow(2, ev).toFixed(3)})`);
  if (Math.abs(con) > 0.01) parts.push(`contrast(${(1 + con * 0.8).toFixed(3)})`);
  if (Math.abs(sat) > 0.01) parts.push(`saturate(${Math.max(0, 1 + sat).toFixed(3)})`);
  return parts.length ? parts.join(' ') : undefined;
}
