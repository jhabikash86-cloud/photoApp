// Ready-made looks for each automatic region. A preset is a full set of that region's adjustment sliders (anything not
// listed is 0). Values are on the same -100..+100 scale as the sliders, so after applying one you fine-tune it in the
// Masks tab. Faces also carry portrait-retouch values (the People sliders).
import { Params, People, Region } from './params';

export type AdjKey = 'exposure' | 'contrast' | 'highlights' | 'shadows' | 'vibrance' | 'saturation' | 'clarity' | 'dehaze' | 'temperature' | 'tint' | 'hue';
export type RegionPreset = { id: string; label: string; hint: string; adj: Partial<Record<AdjKey, number>>; people?: Partial<People> };

export const ADJ_KEYS: AdjKey[] = ['exposure', 'contrast', 'highlights', 'shadows', 'vibrance', 'saturation', 'clarity', 'dehaze', 'temperature', 'tint', 'hue'];

export const REGION_PRESETS: Record<string, RegionPreset[]> = {
  sky: [
    { id: 'deep_blue', label: 'Deep blue sky', hint: 'Rich clean blue, clouds kept bright and defined', adj: { exposure: -8, contrast: 14, highlights: -15, vibrance: 16, saturation: 4, clarity: 8, dehaze: 12, temperature: -4 } },
    { id: 'dramatic', label: 'Dramatic clouds', hint: 'Heavy contrast and texture in the clouds', adj: { exposure: -14, contrast: 24, highlights: -25, shadows: -6, clarity: 22, dehaze: 22, vibrance: 10 } },
    { id: 'golden', label: 'Golden glow', hint: 'Warm sunset / golden-hour sky', adj: { exposure: -6, contrast: 10, highlights: -22, vibrance: 20, saturation: 4, temperature: 16, tint: 4 } },
    { id: 'blue_hour', label: 'Blue hour', hint: 'Cool, calm, deep dusk blues', adj: { exposure: -6, contrast: 10, vibrance: 14, temperature: -14, tint: 4 } },
    { id: 'pastel', label: 'Soft pastel', hint: 'Light, airy, gently desaturated', adj: { exposure: 3, contrast: -6, highlights: -10, vibrance: 8, saturation: -8, temperature: 4 } },
    { id: 'moody', label: 'Moody overcast', hint: 'Dark, textured, quiet grey sky', adj: { exposure: -10, contrast: 18, clarity: 14, dehaze: 14, saturation: -10 } },
  ],
  mountains: [
    { id: 'crisp', label: 'Crisp & dramatic', hint: 'Sharp ridges, strong texture', adj: { exposure: 2, contrast: 16, shadows: 8, vibrance: 8, clarity: 24, dehaze: 14 } },
    { id: 'alpenglow', label: 'Warm alpenglow', hint: 'Sun-lit warm peaks', adj: { exposure: 4, contrast: 10, highlights: -8, vibrance: 16, clarity: 12, temperature: 18 } },
    { id: 'misty', label: 'Misty depth', hint: 'Soft atmospheric haze, layered distance', adj: { exposure: 3, contrast: -4, clarity: -8, dehaze: -10, temperature: -4 } },
    { id: 'cool_moody', label: 'Cool & moody', hint: 'Cold, dark, serious', adj: { exposure: -6, contrast: 14, clarity: 16, saturation: -6, temperature: -10 } },
  ],
  water: [
    { id: 'rich', label: 'Rich deep water', hint: 'Deeper tone, saturated, controlled highlights', adj: { exposure: -8, contrast: 12, highlights: -10, vibrance: 18, saturation: 4, clarity: 6, temperature: -6 } },
    { id: 'turquoise', label: 'Turquoise clear', hint: 'Tropical cyan-green glow', adj: { exposure: -2, contrast: 8, vibrance: 22, saturation: 8, temperature: -8, tint: -6 } },
    { id: 'glassy', label: 'Glassy calm', hint: 'Smooth mirror-like surface', adj: { exposure: -4, highlights: -14, vibrance: 10, clarity: -8 } },
    { id: 'sparkle', label: 'Sparkle', hint: 'Crisp ripples and highlights', adj: { exposure: 2, contrast: 14, highlights: 6, clarity: 14 } },
    { id: 'reflection', label: 'Golden reflection', hint: 'Warm light on the water', adj: { exposure: -6, vibrance: 14, temperature: 14 } },
  ],
  trees: [
    { id: 'forest', label: 'Rich forest green', hint: 'Deep, lush, lifted shadows', adj: { exposure: -6, contrast: 10, shadows: 8, vibrance: 18, saturation: -2, clarity: 8 } },
    { id: 'autumn', label: 'Autumn warmth', hint: 'Warm gold and amber foliage', adj: { exposure: -2, contrast: 8, vibrance: 16, temperature: 18 } },
    { id: 'moody_forest', label: 'Moody forest', hint: 'Dark, desaturated, cinematic', adj: { exposure: -12, contrast: 16, shadows: 10, saturation: -8, clarity: 10, temperature: -4 } },
    { id: 'spring', label: 'Fresh spring', hint: 'Bright, lively, light greens', adj: { exposure: 4, vibrance: 22, temperature: 4 } },
  ],
  grass: [
    { id: 'lush', label: 'Lush green', hint: 'Vivid healthy grass', adj: { exposure: -2, contrast: 8, vibrance: 24 } },
    { id: 'meadow', label: 'Golden meadow', hint: 'Warm sunlit grass', adj: { exposure: 2, vibrance: 14, temperature: 18 } },
    { id: 'natural', label: 'Muted natural', hint: 'Calm, true-to-life, less neon', adj: { exposure: -4, saturation: -12, vibrance: 6 } },
  ],
  buildings: [
    { id: 'crisp', label: 'Crisp architecture', hint: 'Clean lines, strong detail', adj: { contrast: 14, vibrance: 6, clarity: 22, dehaze: 8 } },
    { id: 'warm_stone', label: 'Warm stone', hint: 'Sun-baked warm walls', adj: { exposure: 2, temperature: 12, clarity: 12 } },
    { id: 'cool_modern', label: 'Cool modern', hint: 'Steel, glass, cold light', adj: { contrast: 12, saturation: -6, clarity: 16, temperature: -10 } },
  ],
  ground: [
    { id: 'earthy', label: 'Earthy rich', hint: 'Warm, textured earth and rock', adj: { contrast: 12, shadows: 8, vibrance: 12, clarity: 14, temperature: 10 } },
    { id: 'textured', label: 'Deep & textured', hint: 'Darker, gritty, strong texture', adj: { exposure: -8, contrast: 18, clarity: 24, dehaze: 8 } },
    { id: 'sandy', label: 'Sandy warm', hint: 'Bright warm sand', adj: { exposure: 3, vibrance: 10, temperature: 16 } },
  ],
  foreground: [
    { id: 'pop', label: 'Pop the foreground', hint: 'Brighter, punchier, more detail up front', adj: { exposure: 4, contrast: 12, shadows: 8, vibrance: 14, clarity: 16 } },
    { id: 'lift', label: 'Gentle lift', hint: 'Open up a dark or backlit foreground', adj: { exposure: 6, highlights: -6, shadows: 14, vibrance: 8 } },
    { id: 'depth', label: 'Darken for depth', hint: 'Frames the scene, leads the eye in', adj: { exposure: -8, contrast: 8, vibrance: 6 } },
  ],
  background: [
    { id: 'recede', label: 'Calm & recede', hint: 'Quieter, softer, less detail', adj: { exposure: -6, contrast: -6, saturation: -8, clarity: -14, dehaze: -4 } },
    { id: 'moody', label: 'Moody backdrop', hint: 'Darker and cooler behind the subject', adj: { exposure: -12, contrast: 8, saturation: -14, temperature: -4 } },
    { id: 'warm', label: 'Warm glow', hint: 'Warm, bright backdrop', adj: { exposure: 2, vibrance: 8, temperature: 12 } },
  ],
  other: [
    { id: 'balance', label: 'Neutral balance', hint: 'Lift shadows, rein in highlights', adj: { contrast: 4, highlights: -6, shadows: 10 } },
    { id: 'pop', label: 'Subtle pop', hint: 'A little colour and contrast', adj: { contrast: 8, vibrance: 10 } },
    { id: 'blend', label: 'Blend in', hint: 'Slightly darker and calmer', adj: { exposure: -3, contrast: -2, clarity: -4 } },
  ],
  skin: [
    { id: 'natural', label: 'Natural portrait', hint: 'Clean and honest: gentle smoothing, even tone', adj: { exposure: 3, contrast: -2, clarity: -6 },
      people: { smooth: 28, brighten: 18, tone_fix: 60, sculpt: 35, eyes: 30, lips: 20, warmth: 0 } },
    { id: 'dreamy', label: 'Dreamy glow', hint: 'Soft warm glow with a slight brightness on the face', adj: { exposure: 6, highlights: -6, vibrance: 4, clarity: -10, temperature: 6 },
      people: { smooth: 40, brighten: 28, tone_fix: 55, sculpt: 30, eyes: 28, lips: 18, warmth: 14 } },
    { id: 'sculpt', label: 'Studio sculpt', hint: 'More light and shadow shaping, brighter eyes', adj: { contrast: 8, clarity: 6 },
      people: { smooth: 22, brighten: 14, tone_fix: 55, sculpt: 60, eyes: 40, lips: 24, warmth: 0 } },
    { id: 'bright', label: 'Clean & bright', hint: 'Fresh, bright, even skin', adj: { exposure: 8, highlights: -8, clarity: -6 },
      people: { smooth: 34, brighten: 34, tone_fix: 70, sculpt: 25, eyes: 30, lips: 18, warmth: 4 } },
    { id: 'warm', label: 'Warm & healthy', hint: 'A little sun, tone kept natural', adj: { vibrance: 6, temperature: 6 },
      people: { smooth: 24, brighten: 16, tone_fix: 50, sculpt: 35, eyes: 28, lips: 22, warmth: 22 } },
    { id: 'matte', label: 'Matte soft', hint: 'Lower contrast, muted, editorial', adj: { contrast: -8, saturation: -8, clarity: -8 },
      people: { smooth: 36, brighten: 14, tone_fix: 55, sculpt: 20, eyes: 22, lips: 10, warmth: 0 } },
  ],
};

export const PRESET_REGION_ORDER = ['skin', 'sky', 'mountains', 'water', 'trees', 'grass', 'buildings', 'ground', 'foreground', 'background', 'other'];

/** The photo with `preset` applied to `region`: the region's adjustment sliders are set (the rest zeroed); mask controls stay. */
export function applyRegionPreset(p: Params, region: string, preset: RegionPreset): Params {
  const cur = p.regions[region];
  if (!cur) return p;
  const next: Region = { ...cur, enabled: true };
  for (const k of ADJ_KEYS) next[k] = preset.adj[k] ?? 0;
  const out: Params = { ...p, regions: { ...p.regions, [region]: next } };
  if (preset.people) out.people = { ...p.people, ...preset.people, enabled: true };
  return out;
}

export function presetIsActive(p: Params, region: string, preset: RegionPreset): boolean {
  const r = p.regions[region];
  return !!r && ADJ_KEYS.every((k) => Math.abs((r[k] ?? 0) - (preset.adj[k] ?? 0)) < 0.5);
}

/** Region sliders that are all zero: nothing is being adjusted in this region. */
export function regionIsNeutral(p: Params, region: string): boolean {
  const r = p.regions[region];
  return !r || ADJ_KEYS.every((k) => Math.abs(r[k] ?? 0) < 0.5);
}
