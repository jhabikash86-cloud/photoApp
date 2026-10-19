'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Slider from './Slider';
import Toggle from './Toggle';
import type { Suggest } from './SkyPanel';
import { Sub } from './panels';
import { API_URL, DramaSettings, EditCtx, Params, newLayer } from '../lib/params';

type LutItem = { id: string; name: string; builtin: boolean };
type Preset = { id: string; name: string; scopes: string[]; params: Partial<Params> };
type LookItem = { id: string; label: string; result: string; adjusts: string };
type LookState = { id: string; k: number; notes: string[]; regions: string[] };
type DramaItem = { id: string; label: string; blurb: string; fit: number; reason: string; recommended: boolean };
type DramaState = { id: string; k: number; notes: string[] };
const dramaStore = new Map<string, DramaState>();
const DRAMA_SLIDERS: { key: string; label: string; min: number; max: number; hint: string }[] = [
  { key: 'mood', label: 'Colour mood', min: 0, max: 100, hint: 'How strongly the look\'s colours replace the picture\'s own cast' },
  { key: 'enrich', label: 'Richer colours', min: -100, max: 100, hint: 'Make the colours that are there richer' },
  { key: 'sky_burn', label: 'Sky darkening', min: 0, max: 100, hint: 'Graduated darkening toward the top of the sky' },
  { key: 'sky_texture', label: 'Cloud contrast', min: 0, max: 100, hint: 'Keeps clouds from going flat' },
  { key: 'sky_colour', label: 'Sky colour', min: 0, max: 100, hint: 'Extra colour in the sky' },
  { key: 'bloom', label: 'Light bloom', min: 0, max: 100, hint: 'Glow around the brightest light' },
  { key: 'mist', label: 'Horizon mist', min: 0, max: 100, hint: 'Luminous haze on the horizon' },
  { key: 'depth_haze', label: 'Depth haze', min: 0, max: 100, hint: 'Far ground sinks into haze' },
];
type MatchState = { refId: string; thumb: string; snap: Params; notes: string[]; strength: number };

export const SCOPES: { id: string; label: string; hint: string; def: boolean }[] = [
  { id: 'tone', label: 'Tone & light', hint: 'Exposure, contrast, highlights, shadows, curves', def: true },
  { id: 'colour', label: 'Colour & grading', hint: 'White balance, saturation, vibrance, colour mixer, grading wheels', def: true },
  { id: 'lut', label: 'LUT', hint: 'The selected LUT and its intensity', def: true },
  { id: 'effects', label: 'Effects', hint: 'Bokeh, glow, vignette, grain, sharpen, denoise', def: true },
  { id: 'people', label: 'People & dodge/burn', hint: 'Portrait retouching and the dodge & burn engine', def: false },
  { id: 'regions', label: 'Region edits', hint: 'Sky, trees, water... adjustments (not the paint-to-fix strokes)', def: false },
  { id: 'masks', label: 'Gradients & range masks', hint: 'Gradient, radial, luminosity and colour-range masks (brush masks are photo-specific and never copied)', def: false },
];
const SCOPE_KEYS: Record<string, (keyof Params)[]> = {
  tone: ['basic', 'tone', 'curve', 'wb_mode'],
  colour: ['mixer', 'balance_shadows', 'balance_mids', 'balance_highs', 'grading', 'grading_enabled'],
  lut: ['lut'], effects: ['effects'], people: ['people', 'dodge_burn'], regions: ['regions'], masks: ['masks'],
};

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));

/** The part of the edit selected by `scopes`, safe to carry to another photo. */
export function extract(p: Params, scopes: string[]): Partial<Params> {
  const out: Record<string, unknown> = {};
  for (const sc of scopes) for (const k of SCOPE_KEYS[sc] ?? []) {
    if (!(k in p)) continue;
    let v = clone(p[k]);
    if (k === 'regions') for (const r of Object.values(v as Params['regions'])) delete r.strokes;
    if (k === 'masks') v = (v as Params['masks']).filter((l) => l.kind !== 'brush');
    out[k] = v;
  }
  return out as Partial<Params>;
}

/** Lay saved settings over the current edit. Brush masks and paint-to-fix strokes of THIS photo are kept. */
export function merge(cur: Params, saved: Partial<Params>): Params {
  const next: Params = { ...cur };
  for (const [k, v] of Object.entries(saved)) {
    if (k === 'regions') {
      const regs = { ...cur.regions };
      for (const [n, r] of Object.entries(v as Params['regions'])) if (n in regs) regs[n] = { ...clone(r), strokes: cur.regions[n].strokes ?? [] };
      next.regions = regs;
    } else if (k === 'masks') {
      const fresh = (v as Params['masks']).map((l, i) => ({ ...newLayer(l.kind, i + 1), ...clone(l), id: Math.random().toString(36).slice(2, 9) }));
      next.masks = [...cur.masks.filter((l) => l.kind === 'brush'), ...fresh];
    } else {
      (next as Record<string, unknown>)[k] = clone(v);
    }
  }
  return next;
}

let clipboard: Partial<Params> | null = null;                       // Copy / Paste settings between photos
const lookStore = new Map<string, LookState>();      // chosen look + strength + what it did, per photo
const matchStore = new Map<string, MatchState>();                   // survives switching tabs

async function getJSON<T>(path: string, body?: unknown, form?: FormData): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, form ? { method: 'POST', body: form } : body !== undefined
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Server error (${res.status})`);
  return data as T;
}

const chip = (on: boolean) =>
  `px-3 py-2 rounded-lg text-[13px] border transition-colors ${on ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-200 hover:border-zinc-500 hover:bg-zinc-800/60'}`;

export default function LooksPanel({ ctx, suggest }: { ctx: EditCtx; suggest: Suggest | null }) {
  const { p, sid } = ctx;
  const [luts, setLuts] = useState<LutItem[]>([]);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [loadingPrev, setLoadingPrev] = useState(false);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [scopes, setScopes] = useState<string[]>(SCOPES.filter((s) => s.def).map((s) => s.id));
  const [name, setName] = useState('');
  const [busyMatch, setBusyMatch] = useState(false);
  const [match, setMatch] = useState<MatchState | null>(matchStore.get(sid) ?? null);
  const [looks, setLooks] = useState<LookItem[]>([]);
  const [land, setLand] = useState<LookState | null>(lookStore.get(sid) ?? null);
  const [busyLook, setBusyLook] = useState(false);
  const [dramaList, setDramaList] = useState<{ looks: DramaItem[]; scene: Record<string, unknown> } | null>(null);
  const [dramaSel, setDramaSel] = useState<DramaState | null>(dramaStore.get(sid) ?? null);
  const [busyDrama, setBusyDrama] = useState(false);
  const dramaTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [smartSky, setSmartSky] = useState(true);
  const lookTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lutFile = useRef<HTMLInputElement>(null);
  const refFile = useRef<HTMLInputElement>(null);
  const matchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pRef = useRef(p);
  useEffect(() => { pRef.current = p; });

  const loadLuts = useCallback(() => getJSON<{ luts: LutItem[] }>('/api/luts').then((d) => setLuts(d.luts)).catch(() => {}), []);
  const loadPresets = useCallback(() => getJSON<{ presets: Preset[] }>('/api/presets').then((d) => setPresets(d.presets)).catch(() => {}), []);
  const refreshPreviews = useCallback(() => {
    setLoadingPrev(true);
    getJSON<{ previews: Record<string, string> }>('/api/lut_previews', { session_id: sid, params: pRef.current })
      .then((d) => setPreviews(d.previews)).catch(() => {}).finally(() => setLoadingPrev(false));
  }, [sid]);
  useEffect(() => { loadLuts(); loadPresets(); Promise.resolve().then(refreshPreviews); getJSON<{ looks: LookItem[] }>('/api/looks').then((d) => setLooks(d.looks)).catch(() => {}); fetch(`${API_URL}/api/drama/${sid}`).then((r) => r.json()).then((d) => { if (d.looks) setDramaList(d); }).catch(() => {}); }, [loadLuts, loadPresets, refreshPreviews, sid]);

  const setMatchState = (m: MatchState | null) => { if (m) matchStore.set(sid, m); else matchStore.delete(sid); setMatch(m); };
  const toggleScope = (id: string) => setScopes((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  const lut = p.lut ?? { id: null, intensity: 100 };

  // ---------------------------------------------------------------- scene-aware looks (the server reads the photo and builds the edit)
  const setLandState = (v: LookState | null) => { if (v) lookStore.set(sid, v); else lookStore.delete(sid); setLand(v); };
  const runLook = useCallback(async (id: string, k: number) => {
    setBusyLook(true);
    try {
      const gr = ctx.analysis.grade;
      const d = await getJSON<{ params: Params; notes: string[]; facts: { regions: string[] } }>('/api/look', { session_id: sid, look_id: id, params: pRef.current, strength: Math.round(k * 100), light: gr?.light, scene: gr?.scene, smart_sky: smartSky });
      ctx.update(() => d.params, true, true);
      setLandState({ id, k, notes: d.notes, regions: d.facts.regions });
    } catch (e) { ctx.flash(e instanceof Error ? e.message : 'Could not apply that look'); }
    finally { setBusyLook(false); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sid, ctx, smartSky]);
  const onLookStrength = (v: number) => {
    if (!land) return;
    setLandState({ ...land, k: v / 100 });
    if (lookTimer.current) clearTimeout(lookTimer.current);
    lookTimer.current = setTimeout(() => runLook(land.id, v / 100), 260);
  };
  const backToAuto = () => {
    ctx.update((c) => ({ ...c, basic: { ...ctx.auto.basic, temperature: c.basic.temperature, tint: c.basic.tint }, grading: ctx.auto.grading, mixer: ctx.auto.mixer,
      curve: ctx.auto.curve, dodge_burn: ctx.auto.dodge_burn, regions: Object.fromEntries(Object.entries(c.regions).map(([n, r]) => [n, { ...ctx.auto.regions[n], strokes: r.strokes }])) as Params['regions'],
      effects: { ...c.effects, vignette: ctx.auto.effects.vignette } }), true, true);
    setLandState(null);
  };

  // ---------------------------------------------------------------- Drama looks (generated for this photo)
  const saveDrama = (v: DramaState | null) => { if (v) dramaStore.set(sid, v); else dramaStore.delete(sid); setDramaSel(v); };
  const runDrama = useCallback(async (id: string, k: number) => {
    setBusyDrama(true);
    try {
      const d = await getJSON<{ params: Params; notes: string[] }>('/api/drama', { session_id: sid, look_id: id, params: pRef.current, strength: Math.round(k * 100) });
      ctx.update(() => d.params, true, true);
      saveDrama({ id, k, notes: d.notes });
    } catch (e) { ctx.flash(e instanceof Error ? e.message : 'Could not apply that look'); }
    finally { setBusyDrama(false); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sid, ctx]);
  const onDramaStrength = (v: number) => {
    if (!dramaSel) return;
    saveDrama({ ...dramaSel, k: v / 100 });
    if (dramaTimer.current) clearTimeout(dramaTimer.current);
    dramaTimer.current = setTimeout(() => runDrama(dramaSel.id, v / 100), 280);
  };
  const dramaOff = () => { ctx.update((c) => ({ ...c, drama: { ...(c.drama as DramaSettings), enabled: false } }), true, true); saveDrama(null); };
  const dm = p.drama;

  // ---------------------------------------------------------------- LUTs
  const pickLut = (id: string | null) => ctx.update((c) => ({ ...c, lut: { id, intensity: c.lut?.intensity ?? 100 } }), true, true);
  const importLut = async (f: File) => {
    const fd = new FormData(); fd.append('file', f);
    try {
      const d = await getJSON<{ id: string; luts: LutItem[] }>('/api/luts/upload', undefined, fd);
      setLuts(d.luts); pickLut(d.id); refreshPreviews(); ctx.flash(`Imported ${f.name}`);
    } catch (e) { ctx.flash(e instanceof Error ? e.message : 'Could not import that LUT'); }
  };
  const deleteLut = async (id: string) => {
    try {
      const d = await getJSON<{ luts: LutItem[] }>('/api/luts/delete', { id });
      setLuts(d.luts);
      if (lut.id === id) pickLut(null);
    } catch (e) { ctx.flash(e instanceof Error ? e.message : 'Could not delete'); }
  };

  // ---------------------------------------------------------------- presets
  const savePreset = async () => {
    if (!name.trim() || !scopes.length) { ctx.flash(!name.trim() ? 'Give the preset a name first.' : 'Tick at least one thing to save.'); return; }
    try {
      const d = await getJSON<{ presets: Preset[] }>('/api/presets', { name: name.trim(), scopes, params: extract(p, scopes) });
      setPresets(d.presets); setName(''); ctx.flash('Preset saved');
    } catch (e) { ctx.flash(e instanceof Error ? e.message : 'Could not save'); }
  };
  const applyPreset = (pr: Preset) => ctx.update((c) => merge(c, pr.params), true, true);
  const deletePreset = async (id: string) => {
    try { setPresets((await getJSON<{ presets: Preset[] }>('/api/presets/delete', { id })).presets); } catch { /* ignore */ }
  };

  // ---------------------------------------------------------------- match a reference
  const runMatch = useCallback(async (m: MatchState) => {
    setBusyMatch(true);
    try {
      const d = await getJSON<{ params: Params; notes: string[] }>('/api/match', { session_id: sid, ref_id: m.refId, params: m.snap, strength: m.strength });
      const r = d.params;
      ctx.update((c) => ({
        ...c, basic: { ...c.basic, saturation: r.basic.saturation, vibrance: r.basic.vibrance }, tone: { ...c.tone, rgb: r.tone.rgb },
        grading: { ...c.grading, shadows: r.grading.shadows, mids: r.grading.mids, highs: r.grading.highs }, mixer: r.mixer, grading_enabled: true,
      }), true, true);
      setMatchState({ ...m, notes: d.notes });
    } catch (e) { ctx.flash(e instanceof Error ? e.message : 'Matching failed'); }
    finally { setBusyMatch(false); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sid, ctx]);
  const uploadRef = async (f: File) => {
    setBusyMatch(true);
    try {
      const fd = new FormData(); fd.append('file', f);
      const up = await getJSON<{ ref_id: string; thumb_url: string }>('/api/reference', undefined, fd);
      const m: MatchState = { refId: up.ref_id, thumb: `${API_URL}${up.thumb_url}`, snap: clone(pRef.current), notes: [], strength: 60 };
      setMatchState(m);
      await runMatch(m);
    } catch (e) { ctx.flash(e instanceof Error ? e.message : 'Could not read that reference'); setBusyMatch(false); }
  };
  const onStrength = (v: number) => {
    if (!match) return;
    const m = { ...match, strength: v };
    setMatchState(m);
    if (matchTimer.current) clearTimeout(matchTimer.current);
    matchTimer.current = setTimeout(() => runMatch(m), 220);
  };
  const removeMatch = () => {
    if (!match) return;
    const s = match.snap;
    ctx.update((c) => ({ ...c, basic: { ...c.basic, saturation: s.basic.saturation, vibrance: s.basic.vibrance }, tone: { ...c.tone, rgb: s.tone.rgb },
      grading: { ...c.grading, shadows: s.grading.shadows, mids: s.grading.mids, highs: s.grading.highs }, mixer: s.mixer }), true, true);
    setMatchState(null);
  };

  return (
    <>
      <Sub title="Drama looks for this photo">
        <p className="text-[11px] leading-relaxed text-zinc-500">The scene is analysed, then looks are generated for it: colour poured over every tone, a sky that keeps its clouds, bloom and mist. The starred ones fit best, but it is your call.</p>
        {!dramaList && <p className="text-[11px] text-zinc-500">Analysing the scene…</p>}
        {dramaList && (
          <div className="grid grid-cols-2 gap-1.5">
            {dramaList.looks.map((l) => (
              <button key={l.id} disabled={busyDrama} onClick={() => runDrama(l.id, dramaSel?.k ?? 1)} title={`${l.blurb} · ${l.reason}`}
                className={`overflow-hidden rounded-lg border text-left transition-colors disabled:opacity-60 ${dramaSel?.id === l.id && dm?.enabled ? 'border-indigo-400 bg-indigo-950' : 'border-zinc-700 hover:border-zinc-500'}`}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`${API_URL}/api/drama_thumb/${sid}/${l.id}`} alt={l.label} loading="lazy" className="aspect-[3/2] w-full bg-zinc-800 object-cover" />
                <div className="px-2 py-1.5">
                  <div className="truncate text-[12px] font-medium text-zinc-100">{l.recommended ? '★ ' : ''}{l.label}</div>
                  <div className="truncate text-[10px] text-zinc-400">{l.reason}</div>
                </div>
              </button>
            ))}
          </div>
        )}
        {busyDrama && <p className="text-[11px] text-zinc-500">Building the look for this photo…</p>}
        {dm?.enabled && (
          <>
            <Slider label="Drama strength" value={dm.amount} min={0} max={150} def={100} fmt={(v) => `${Math.round(v)}%`}
              onChange={(v) => (dramaSel ? onDramaStrength(v) : ctx.set(['drama', 'amount'], v))} />
            {DRAMA_SLIDERS.map((d) => (
              <Slider key={d.key} label={d.label} hint={d.hint} value={(dm as unknown as Record<string, number>)[d.key]} min={d.min} max={d.max}
                fmt={(v) => `${Math.round(v)}`} onChange={(v) => ctx.set(['drama', d.key], v)} />
            ))}
            {dramaSel && <ul className="list-disc space-y-1 pl-4 text-[11px] leading-snug text-zinc-400">{dramaSel.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>}
            <button onClick={dramaOff} className={`${chip(false)} w-full`}>Turn Drama off</button>
          </>
        )}
      </Sub>

      {suggest && suggest.looks.length > 0 && (
        <Sub title="Suggested for this photo">
          <p className="text-[11px] leading-relaxed text-zinc-500">Picked after analysing this picture (light, scene, sky and tones).</p>
          {suggest.looks.map((l, i) => (
            <button key={l.id} disabled={busyLook} onClick={() => runLook(l.id, land?.k ?? 1)}
              className={`rounded-lg border px-3 py-2.5 text-left transition-colors disabled:opacity-60 ${land?.id === l.id ? 'border-indigo-400 bg-indigo-950' : 'border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800/60'}`}>
              <div className="flex items-center justify-between">
                <span className="text-[13px] font-medium text-zinc-100">{i === 0 ? '★ ' : ''}{l.label}</span>
                <span className="text-[10.5px] text-zinc-500">{Math.round(l.fit * 100)}% fit</span>
              </div>
              <div className="mt-0.5 text-[11px] text-zinc-400">{l.reason}</div>
            </button>
          ))}
        </Sub>
      )}

      <Sub title="Smart looks">
        {suggest?.sky && (
          <Toggle checked={smartSky} onChange={setSmartSky} small label={`Include a premium sky (${suggest.sky.label.toLowerCase()})`} title="Also applies the sky treatment that fits this sky best; adjust it in the Sky tab" />
        )}
        <p className="text-[11px] leading-relaxed text-zinc-500">One click reads the photo (light, scene, every region) and builds the edit: a colour grade, a different adjustment for each region (sky, mountains, water, trees, foreground; skin protected) and the lighting. Everything stays editable.</p>
        <div className="grid grid-cols-2 gap-1.5">
          {looks.map((pr) => (
            <button key={pr.id} disabled={busyLook} title={`${pr.result} · ${pr.adjusts}`} onClick={() => runLook(pr.id, land?.k ?? 1)}
              className={`rounded-lg border px-2.5 py-2.5 text-left transition-colors disabled:opacity-60 ${land?.id === pr.id ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-200 hover:border-zinc-500 hover:bg-zinc-800/60'}`}>
              <div className="text-[12.5px] font-medium leading-tight">{pr.label}</div>
              <div className="mt-0.5 text-[10.5px] leading-snug text-zinc-400">{pr.result}</div>
            </button>
          ))}
        </div>
        {busyLook && <p className="text-[11px] text-zinc-500">Reading the photo and building the edit…</p>}
        {land && (
          <>
            <Slider label="Look strength" value={land.k * 100} min={0} max={150} def={100} fmt={(v) => `${Math.round(v)}%`} onChange={onLookStrength} />
            {land.regions.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {land.regions.map((r) => (
                  <span key={r} className="rounded-full border border-zinc-700 px-2 py-0.5 text-[10.5px] text-zinc-300">{r === 'other' ? 'Miscellaneous' : r}</span>
                ))}
              </div>
            )}
            <ul className="list-disc space-y-1 pl-4 text-[11px] leading-snug text-zinc-400">{land.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
            <button onClick={backToAuto} className={`${chip(false)} w-full`}>Back to the automatic grade</button>
          </>
        )}
      </Sub>

      <Sub title="Match a reference">
        <p className="text-[11px] text-zinc-500 leading-relaxed">
          Pick a photo whose look you like (a film still, a Lightroom export, another of your edits). The app measures its brightness, colour cast and colour intensity and sets your sliders to match, so you can still fine-tune every one.
        </p>
        <input ref={refFile} type="file" accept="image/*,.heic,.heif,.nef,.dng,.cr2,.cr3,.arw" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) uploadRef(f); e.target.value = ''; }} />
        {!match ? (
          <button disabled={busyMatch} onClick={() => refFile.current?.click()} className={`${chip(false)} w-full disabled:opacity-40`}>
            {busyMatch ? 'Reading reference…' : 'Choose a reference photo…'}
          </button>
        ) : (
          <div className="flex flex-col gap-2.5">
            <div className="flex items-center gap-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={match.thumb} alt="Reference" className="h-14 w-20 rounded-md object-cover border border-zinc-700" />
              <div className="flex flex-1 flex-col gap-1.5">
                <button onClick={() => refFile.current?.click()} className={`${chip(false)} py-1.5 text-[12px]`}>Change reference</button>
                <button onClick={removeMatch} className={`${chip(false)} py-1.5 text-[12px]`}>Remove match</button>
              </div>
            </div>
            <Slider label="Match strength" value={match.strength} min={0} max={100} def={60} fmt={(v) => `${Math.round(v)}%`} onChange={onStrength} disabled={busyMatch} />
            {busyMatch && <p className="text-[11px] text-zinc-500">Matching…</p>}
            <ul className="space-y-1 text-[11px] text-zinc-400 list-disc pl-4">{match.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
          </div>
        )}
      </Sub>

      <Sub title="LUTs" right={<button onClick={refreshPreviews} className="text-[10px] text-indigo-400 hover:text-indigo-300">{loadingPrev ? 'Updating…' : 'Refresh previews'}</button>}>
        <input ref={lutFile} type="file" accept=".cube" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) importLut(f); e.target.value = ''; }} />
        <div className="grid grid-cols-3 gap-2">
          <button onClick={() => pickLut(null)} className={`rounded-lg border p-1.5 text-center text-[11px] ${!lut.id ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-300 hover:border-zinc-500'}`}>
            <div className="flex aspect-[4/3] items-center justify-center rounded-md bg-zinc-800 text-lg">∅</div>
            <div className="mt-1">None</div>
          </button>
          {luts.map((l) => (
            <div key={l.id} className="relative">
              <button onClick={() => pickLut(l.id)} title={l.name}
                className={`w-full rounded-lg border p-1.5 text-center text-[11px] ${lut.id === l.id ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-300 hover:border-zinc-500'}`}>
                {previews[l.id]
                  // eslint-disable-next-line @next/next/no-img-element
                  ? <img src={previews[l.id]} alt={l.name} className="aspect-[4/3] w-full rounded-md object-cover" />
                  : <div className="aspect-[4/3] w-full animate-pulse rounded-md bg-zinc-800" />}
                <div className="mt-1 truncate">{l.name}</div>
              </button>
              {!l.builtin && (
                <button onClick={() => deleteLut(l.id)} title="Delete this imported LUT" className="absolute right-1.5 top-1.5 rounded-full bg-black/70 px-1.5 text-[11px] leading-5 text-zinc-200 hover:bg-red-600">×</button>
              )}
            </div>
          ))}
        </div>
        {lut.id && <Slider label="LUT intensity" value={lut.intensity} min={0} max={100} def={100} fmt={(v) => `${Math.round(v)}%`}
          onChange={(v) => ctx.update((c) => ({ ...c, lut: { id: c.lut?.id ?? null, intensity: v } }))} />}
        <button onClick={() => lutFile.current?.click()} className={`${chip(false)} w-full`}>Import a .cube LUT…</button>
      </Sub>

      <Sub title="Presets & copy / paste">
        <p className="text-[11px] text-zinc-500">Choose what a preset (or Copy settings) includes:</p>
        <div className="flex flex-wrap gap-1.5">
          {SCOPES.map((s) => (
            <button key={s.id} title={s.hint} onClick={() => toggleScope(s.id)} className={`${chip(scopes.includes(s.id))} !py-1.5 !text-[12px]`}>{scopes.includes(s.id) ? '✓ ' : ''}{s.label}</button>
          ))}
        </div>
        <div className="flex gap-1.5">
          <input value={name} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') savePreset(); }} placeholder="Name this look…"
            className="min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-[13px] outline-none focus:border-indigo-400" />
          <button onClick={savePreset} className="rounded-lg bg-indigo-600 px-3.5 py-2 text-[13px] font-medium text-white hover:bg-indigo-500">Save</button>
        </div>
        <div className="flex gap-1.5">
          <button onClick={() => { clipboard = extract(p, scopes); ctx.flash('Settings copied'); }} className={`flex-1 ${chip(false)}`}>Copy settings</button>
          <button onClick={() => clipboard ? ctx.update((c) => merge(c, clipboard!), true, true) : ctx.flash('Nothing copied yet.')} className={`flex-1 ${chip(false)}`}>Paste settings</button>
        </div>
        {presets.length === 0 && <p className="text-[11px] text-zinc-500">No saved presets yet.</p>}
        <div className="flex flex-col gap-1.5">
          {presets.map((pr) => (
            <div key={pr.id} className="flex items-center gap-1.5">
              <button onClick={() => applyPreset(pr)} title={`Applies: ${pr.scopes.join(', ')}`} className={`${chip(false)} min-w-0 flex-1 truncate text-left`}>{pr.name}</button>
              <button onClick={() => deletePreset(pr.id)} title="Delete preset" className="rounded-lg px-2.5 py-2 text-zinc-500 hover:bg-zinc-800 hover:text-red-400">×</button>
            </div>
          ))}
        </div>
      </Sub>
    </>
  );
}
