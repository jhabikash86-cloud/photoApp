// Mirrors params.py on the server. Every slider is a real adjustment (-100..+100), set by Auto and
// then moved by the user.

export const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export type Basic = {
  temperature: number; tint: number; exposure: number; contrast: number; highlights: number; shadows: number;
  whites: number; blacks: number; texture: number; clarity: number; dehaze: number; vibrance: number; saturation: number; keep_whites: number;
};
export type Region = {
  exposure: number; contrast: number; vibrance: number; saturation: number; clarity: number; dehaze: number; temperature: number; hue: number;
  tint: number; highlights: number; shadows: number;
  enabled: boolean; mask_strength: number; feather: number; refine: number; invert: boolean;
  strokes?: Stroke[];           // paint-to-fix: brush strokes that add to (or erase from) this automatic region
};
export type People = {
  smooth: number; brighten: number; warmth: number; tone_fix: number; eyes: number; lips: number; sculpt: number;
  lift: number; exposure: number; enabled: boolean;
};
export type Effects = {
  bokeh: number; bokeh_size: number; bokeh_highlights: number; glow: number; vignette: number; vignette_mid: number;
  grain: number; sharpen: number; denoise: number; denoise_color: number; bokeh_focus: number;
};
export type DodgeBurn = {
  enabled: boolean; strength: number; dodge: number; dodge_tones: number; burn: number; burn_tones: number;
  separation: number; form: number; feather: number; protect_face: boolean;
};
export type Wheel = { hue: number; sat: number; lum: number };
export type Grading = { shadows: Wheel; mids: Wheel; highs: Wheel; glob: Wheel; blending: number; balance: number };
export type Curve = { shadows: number; darks: number; lights: number; highlights: number };

export type Stroke = { pts: number[][]; size: number; feather: number; flow: number; erase: boolean };
export type LocalAdjust = {
  exposure: number; contrast: number; highlights: number; shadows: number; vibrance: number; saturation: number;
  clarity: number; dehaze: number; temperature: number; tint: number; hue: number;
};
export type MaskKind = 'linear' | 'radial' | 'brush' | 'luminosity' | 'color';
export type MaskLayer = {
  id: string; name: string; kind: MaskKind; enabled: boolean; invert: boolean; strength: number; feather: number; refine: number;
  limit: string | null; limit_invert: boolean; line: number[]; ellipse: number[]; strokes: Stroke[]; lum: number[]; lum_soft: number;
  pick: number[]; tolerance: number; adj: LocalAdjust;
};
export type ToneCurve = { rgb: number[][]; r: number[][]; g: number[][]; b: number[][] };

export const ZERO_ADJ: LocalAdjust = {
  exposure: 0, contrast: 0, highlights: 0, shadows: 0, vibrance: 0, saturation: 0, clarity: 0, dehaze: 0, temperature: 0, tint: 0, hue: 0,
};

const KIND_NAME: Record<MaskKind, string> = { linear: 'Gradient', radial: 'Radial', brush: 'Brush', luminosity: 'Luminosity', color: 'Colour range' };

/** A fresh mask layer with sensible geometry. `adj` pre-fills the adjustments (for the quick-start recipes). */
export function newLayer(kind: MaskKind, n: number, extra: Partial<MaskLayer> = {}): MaskLayer {
  return {
    id: Math.random().toString(36).slice(2, 9), name: `${KIND_NAME[kind]} ${n}`, kind, enabled: true, invert: false, strength: 100,
    feather: kind === 'brush' ? 60 : 55, refine: kind === 'brush' ? 55 : 0, limit: null, limit_invert: false,
    line: [0.5, 0.0, 0.5, 0.45], ellipse: [0.5, 0.5, 0.3, 0.3, 0], strokes: [], lum: [60, 100], lum_soft: 25, pick: [0.5, 0.5], tolerance: 35,
    adj: { ...ZERO_ADJ }, ...extra,
  };
}

export type Params = {
  look: string;
  intensity: number;
  wb_mode: string;
  basic: Basic;
  mixer: Record<string, number[]>;
  balance_shadows: number[];
  balance_mids: number[];
  balance_highs: number[];
  regions: Record<string, Region>;
  people: People;
  effects: Effects;
  dodge_burn: DodgeBurn;
  curve: Curve;
  tone: ToneCurve;
  masks: MaskLayer[];
  grading: Grading;
  grading_enabled: boolean;
  lut?: { id: string | null; intensity: number };
  sky_replace?: SkyReplace;
  drama?: DramaSettings;
  longexp?: LongExposureSettings;
  geometry?: GeometrySettings;
};
export type GeometrySettings = { enabled: boolean; angle: number; aspect: string; zoom: number; x: number; y: number };
export const GEOMETRY_OFF: GeometrySettings = { enabled: false, angle: 0, aspect: 'original', zoom: 1, x: 0, y: 0 };
export type LongExposureSettings = { enabled: boolean; water: number; water_dir: string; clouds: number; cloud_angle: number; grad_nd: number };
export const LONGEXP_OFF: LongExposureSettings = { enabled: false, water: 0, water_dir: 'auto', clouds: 0, cloud_angle: 0, grad_nd: 0 };
export type DramaSettings = {
  enabled: boolean; amount: number; mood: number; sh_hue: number; sh_chroma: number; mid_hue: number; mid_chroma: number; hi_hue: number; hi_chroma: number;
  protect: number; enrich: number; skin_safe: boolean; sky_burn: number; sky_model: number; sky_texture: number; sky_colour: number; sky_hue: number; sky_chroma: number;
  bloom: number; bloom_hue: number; mist: number; depth_haze: number; mist_y: number; glow: number; rays: number; sun_x: number; sun_y: number; white: number; detail: number; name: string;
};
export type SkyReplace = {
  id: string | null; blend: number; azimuth: number; lift: number; zoom: number; exposure: number; horizon: number; relight: number; reflect: number; edge: number;
};

export type MaskInfo = { score: number; passed: boolean; chosen: string; tries: string[]; note: string };
export type Analysis = {
  scene: string;
  faces: { focus: string; focus_score: number; size_pct: number }[];
  sky_pct: number;
  person_pct: number;
  haze: number;
  semantic?: { shares: Record<string, number>; indoor: boolean };
  masks: Record<string, MaskInfo>;
  natural_color: { confidence: number; warm_mood: boolean; note: string; shift: string; candidates: string[] };
  lighting?: string[];
  grade?: {
    light: string; light_label: string; light_conf: number; scene: string; scene_label: string; scene_conf: number;
    evidence: string[]; lights: string[]; scenes: string[]; notes: string[]; sun_elevation: number | null;
  };
  meta?: { when: string | null; lat: number | null; lon: number | null; sun_elevation: number | null; sun_source: string; camera: string };
  mask_names?: string[];
  coverage?: Record<string, number>;
  quality?: { blur: 'sharp' | 'soft' | 'blurry' | null; sharp_ratio: number | null; exposure: 'ok' | 'dark' | 'bright'; median: number };
  seconds: number;
};
export type Histogram = { r: number[]; g: number[]; b: number[]; l: number[]; clip_hi?: number; clip_lo?: number };

export type Item = {
  id: string;
  file: File;
  name: string;
  status: 'queued' | 'uploading' | 'analyzing' | 'ready' | 'error';
  progress: number;
  error?: string;
  sid?: string;
  thumb?: string;
  quick?: string;
  megapixels?: number;
  raw?: boolean;
  params?: Params;
  auto?: Params;
  neutral?: Params;
  autoOn?: boolean;
  strengths?: Record<string, number>;
  gradeNotes?: string[];
  lightingNotes?: string[];
  history: Params[];
  hIndex: number;
  analysis?: Analysis;
  regions: string[];
  originalUrl?: string;
  previewUrl?: string;
  histogram?: Histogram;
};

export type Path = (string | number)[];

export function getIn(obj: unknown, path: Path): number {
  let cur: unknown = obj;
  for (const k of path) cur = (cur as Record<string | number, unknown>)[k];
  return cur as number;
}

export function setIn<T>(obj: T, path: Path, value: number | boolean): T {
  if (path.length === 0) return value as unknown as T;
  const [head, ...rest] = path;
  const src = obj as unknown as Record<string | number, unknown>;
  const copy: Record<string | number, unknown> | unknown[] = Array.isArray(obj) ? [...(obj as unknown[])] : { ...src };
  (copy as Record<string | number, unknown>)[head] = setIn(src[head], rest, value);
  return copy as unknown as T;
}

export function getAny(obj: unknown, path: Path): unknown {
  let cur: unknown = obj;
  for (const k of path) cur = (cur as Record<string | number, unknown>)?.[k];
  return cur;
}

/** How many adjustments differ between two parameter sets (numbers by more than 0.5, switches at all). */
export function diffCount(a: unknown, b: unknown): number {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) > 0.5 ? 1 : 0;
  if (typeof a === 'boolean' && typeof b === 'boolean') return a !== b ? 1 : 0;
  if (Array.isArray(a) && Array.isArray(b)) return Math.abs(a.length - b.length) + a.reduce((n: number, v, i) => n + diffCount(v, b[i]), 0);
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    return Object.keys(a as object).reduce((n, k) => n + (k === 'look' ? 0 : diffCount((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k])), 0);
  }
  return 0;
}

/** Everything a panel needs to read and change the current edit. */
export type EditCtx = {
  p: Params; auto: Params; neutral: Params; analysis: Analysis; regions: string[]; sid: string; hasPerson: boolean;
  set: (path: Path, value: number | boolean, immediate?: boolean) => void;
  update: (fn: (p: Params) => Params, immediate?: boolean, commit?: boolean) => void;
  flash: (m: string) => void;
};

export type SliderDef = { key: string; label: string; min?: number; max?: number; fmt?: (v: number) => string };

const ev = (v: number) => `${(v / 100 * 3 >= 0 ? '+' : '')}${(v / 100 * 3).toFixed(2)}`;

export const TONE: SliderDef[] = [
  { key: 'exposure', label: 'Exposure', fmt: ev },
  { key: 'contrast', label: 'Contrast' },
  { key: 'highlights', label: 'Highlights' },
  { key: 'shadows', label: 'Shadows' },
  { key: 'whites', label: 'Whites' },
  { key: 'blacks', label: 'Blacks' },
];
export const PRESENCE: SliderDef[] = [
  { key: 'texture', label: 'Texture' },
  { key: 'clarity', label: 'Clarity' },
  { key: 'dehaze', label: 'Dehaze' },
  { key: 'vibrance', label: 'Vibrance' },
  { key: 'saturation', label: 'Saturation' },
];
export const REGION_SLIDERS: SliderDef[] = [
  { key: 'exposure', label: 'Exposure', fmt: ev },
  { key: 'contrast', label: 'Contrast' },
  { key: 'highlights', label: 'Highlights' },
  { key: 'shadows', label: 'Shadows' },
  { key: 'vibrance', label: 'Vibrance' },
  { key: 'saturation', label: 'Saturation' },
  { key: 'clarity', label: 'Clarity' },
  { key: 'dehaze', label: 'Dehaze' },
  { key: 'temperature', label: 'Temperature' },
  { key: 'tint', label: 'Tint' },
  { key: 'hue', label: 'Hue' },
];
export const PEOPLE_SLIDERS: SliderDef[] = [
  { key: 'smooth', label: 'Skin smoothing', min: 0 },
  { key: 'brighten', label: 'Face brightness' },
  { key: 'tone_fix', label: 'Natural skin tone', min: 0 },
  { key: 'warmth', label: 'Skin warmth' },
  { key: 'sculpt', label: 'Light & shadow', min: 0 },
  { key: 'eyes', label: 'Eyes', min: 0 },
  { key: 'lips', label: 'Lips', min: 0 },
  { key: 'lift', label: 'Backlight lift', min: 0 },
  { key: 'exposure', label: 'Person exposure', fmt: ev },
];

export const REGION_LABELS: Record<string, string> = {
  sky: 'Sky', mountains: 'Mountains', trees: 'Trees', grass: 'Grass', water: 'Water', buildings: 'Buildings',
  ground: 'Ground', person: 'People', foreground: 'Foreground', background: 'Background', skin: 'Skin', other: 'Miscellaneous',
};
export const REGION_COLORS: Record<string, string> = {
  sky: '#3b82f6', mountains: '#a16207', trees: '#15803d', grass: '#84cc16', water: '#06b6d4', buildings: '#a855f7',
  ground: '#d4a017', person: '#f97316', foreground: '#ef4444', background: '#64748b', skin: '#fb923c', other: '#ec4899',
};

export const MIXER_COLORS = ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta'];
export const MIXER_SWATCH: Record<string, string> = {
  red: '#ef4444', orange: '#f97316', yellow: '#eab308', green: '#22c55e', aqua: '#06b6d4', blue: '#3b82f6',
  purple: '#8b5cf6', magenta: '#ec4899',
};

export const CURVE_SLIDERS: SliderDef[] = [
  { key: 'highlights', label: 'Highlights' },
  { key: 'lights', label: 'Lights' },
  { key: 'darks', label: 'Darks' },
  { key: 'shadows', label: 'Shadows' },
];

export const LOOKS: { id: string; label: string; hint: string }[] = [
  { id: 'auto', label: 'Auto', hint: 'Best look for this photo' },
  { id: 'natural', label: 'Natural', hint: 'Clean, true to the scene' },
  { id: 'vivid', label: 'Vivid', hint: 'Punchier colour and contrast' },
  { id: 'dramatic', label: 'Dramatic', hint: 'Moody, rich, deep shadows' },
  { id: 'cinematic', label: 'Cinematic', hint: 'Teal and orange grade' },
  { id: 'dreamy', label: 'Warm Dreamy', hint: 'Soft glow, warm skin' },
  { id: 'soft_portrait', label: 'Soft Portrait', hint: 'Gentle, skin-friendly' },
];

export const WB_LABELS: Record<string, string> = {
  auto: 'Auto', asshot: 'As shot', neutral: 'Neutral', warm: 'Warmer', cool: 'Cooler', skin: 'Skin',
};

export const signed = (v: number) => (v > 0 ? `+${Math.round(v)}` : `${Math.round(v)}`);
