// The simple "Auto-edit controls". Each one rescales a family of adjustments between neutral (0%) and the
// automatic edit (100%), up to 150%. The advanced panels edit the same values, so nothing is hidden.
import { Params } from './params';

type Path = string[];

export type GroupDef = {
  id: string; label: string; hint: string;
  match: (path: string[]) => boolean;
};

const last = (p: string[]) => p[p.length - 1];
const has = (p: string[], ...names: string[]) => names.includes(last(p));

export const GROUPS: GroupDef[] = [
  {
    id: 'grade', label: 'Overall colour grading', hint: 'Tinted shadows and highlights, colour balance, tone curve, warm/cool by region, vibrance',
    match: (p) =>
      (p[0] === 'grading' && has(p, 'sat', 'lum', 'balance')) ||
      p[0] === 'balance_shadows' || p[0] === 'balance_mids' || p[0] === 'balance_highs' || p[0] === 'curve' ||
      (p[0] === 'regions' && has(p, 'temperature', 'hue', 'vibrance', 'saturation')) ||
      (p[0] === 'basic' && has(p, 'vibrance', 'saturation')),
  },
  { id: 'hsl', label: 'HSL colour adjustments', hint: 'Hue, saturation and luminance per colour family', match: (p) => p[0] === 'mixer' },
  {
    id: 'tone', label: 'Tone & contrast', hint: 'Exposure, contrast, highlights, shadows, whites, blacks, clarity, dehaze',
    match: (p) =>
      (p[0] === 'basic' && has(p, 'exposure', 'contrast', 'highlights', 'shadows', 'whites', 'blacks', 'clarity', 'texture', 'dehaze')) ||
      (p[0] === 'regions' && has(p, 'exposure', 'contrast', 'clarity', 'dehaze')),
  },
  {
    id: 'lighting', label: 'Portrait lighting (dodge & burn)', hint: 'Face and skin dodge & burn, sculpting',
    match: (p) => p[0] === 'dodge_burn' && has(p, 'dodge', 'burn', 'form'),
  },
  {
    id: 'separation', label: 'Background separation', hint: 'Darker, softer background so the subject stands out',
    match: (p) => (p[0] === 'dodge_burn' && has(p, 'separation')) || (p[0] === 'effects' && has(p, 'bokeh')),
  },
  {
    id: 'people', label: 'Skin & face retouch', hint: 'Smoothing, tone, eyes, lips, backlight lift',
    match: (p) => p[0] === 'people' && has(p, 'smooth', 'brighten', 'warmth', 'tone_fix', 'eyes', 'lips', 'sculpt', 'lift'),
  },
];

function leaves(obj: unknown, prefix: string[] = [], out: Path[] = []): Path[] {
  if (typeof obj === 'number') out.push(prefix);
  else if (Array.isArray(obj)) obj.forEach((v, i) => leaves(v, [...prefix, String(i)], out));
  else if (obj && typeof obj === 'object') for (const [k, v] of Object.entries(obj)) leaves(v, [...prefix, k], out);
  return out;
}

const get = (o: unknown, path: string[]): number => path.reduce<unknown>((c, k) => (c as Record<string, unknown>)[k], o) as number;

function setDeep<T>(obj: T, path: string[], value: number): T {
  if (path.length === 0) return value as unknown as T;
  const [h, ...rest] = path;
  const src = obj as unknown as Record<string, unknown>;
  const copy: Record<string, unknown> | unknown[] = Array.isArray(obj) ? [...(obj as unknown[])] : { ...src };
  (copy as Record<string, unknown>)[h] = setDeep(src[h], rest, value);
  return copy as unknown as T;
}

/** The parameters with one group scaled to `k` (1 = the automatic edit, 0 = neutral, 1.5 = intensified). */
export function applyGroup(cur: Params, auto: Params, neutral: Params, group: GroupDef, k: number): Params {
  let out = cur;
  for (const path of leaves(auto)) {
    if (!group.match(path)) continue;
    const a = get(auto, path), n = get(neutral, path);
    out = setDeep(out, path, n + (a - n) * k);
  }
  return out;
}

/** The group's current strength as a percentage, estimated from the live parameters (null if it has no effect). */
export function groupStrength(cur: Params, auto: Params, neutral: Params, group: GroupDef): number | null {
  let num = 0, den = 0;
  for (const path of leaves(auto)) {
    if (!group.match(path)) continue;
    const a = get(auto, path), n = get(neutral, path), c = get(cur, path);
    const span = a - n;
    if (Math.abs(span) < 1e-6) continue;
    num += Math.abs(span) * ((c - n) / span);
    den += Math.abs(span);
  }
  return den > 0 ? (num / den) * 100 : null;
}

/** Does this group have anything to scale on this photo? (e.g. no portrait -> no lighting controls) */
export function groupActive(auto: Params, neutral: Params, group: GroupDef): boolean {
  return leaves(auto).some((path) => group.match(path) && Math.abs(get(auto, path) - get(neutral, path)) > 0.5);
}
