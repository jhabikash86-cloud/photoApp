'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Slider from './Slider';
import { Sub } from './panels';
import { API_URL, EditCtx, Params, SkyReplace } from '../lib/params';

export type SkySummary = { kind: string; label: string; confidence: number; share: number; brightness: number; cloud: number; blue: number; warm: number; evidence: string[] };
export type SkyTreatment = { id: string; label: string; result: string; fit: number; recommended: boolean };
export type Suggest = {
  sky: SkySummary | null; sky_treatments: SkyTreatment[];
  looks: { id: string; label: string; result: string; fit: number; reason: string }[];
};

type LibSky = { id: string; name: string; label: string; fit: number; reason: string; thumb: string };
type SwapInfo = { available: boolean; needed: boolean; reason: string; warning?: string; skies: LibSky[] };
const swapCache = new Map<string, SwapInfo>();
const SWAP_CONTROLS: { key: keyof SkyReplace; label: string; min: number; max: number; hint: string }[] = [
  { key: 'blend', label: 'Amount', min: 0, max: 100, hint: '0 = original sky, 100 = fully replaced' },
  { key: 'azimuth', label: 'Turn the sky', min: -180, max: 180, hint: 'Moves the clouds (and the sun) left or right' },
  { key: 'lift', label: 'Look higher / lower', min: -30, max: 40, hint: 'Shows a higher or lower part of the sky' },
  { key: 'zoom', label: 'Cloud size', min: 50, max: 200, hint: 'Bigger or smaller clouds' },
  { key: 'exposure', label: 'Sky brightness', min: -100, max: 100, hint: 'Auto-matched to the original; adjust to taste' },
  { key: 'horizon', label: 'Horizon haze', min: 0, max: 100, hint: 'Carries the original horizon haze into the new sky' },
  { key: 'relight', label: 'Relight the landscape', min: 0, max: 100, hint: 'The landscape takes on the new sky\'s colour and light' },
  { key: 'reflect', label: 'Water reflections', min: 0, max: 100, hint: 'Water takes on the new sky\'s colour' },
  { key: 'edge', label: 'Edge cleanup', min: 0, max: 100, hint: 'Removes the old sky\'s colour from tree / hill edges' },
];

type SkyState = { id: string; k: number; notes: string[]; before: Params };
const skyStore = new Map<string, SkyState>();          // per photo: the treatment, its strength and the sliders from before it

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Server error (${res.status})`);
  return data as T;
}

const chip = (on: boolean) =>
  `rounded-lg border px-2.5 py-2.5 text-left transition-colors ${on ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-200 hover:border-zinc-500 hover:bg-zinc-800/60'}`;

/** Sky studio: reads the sky, recommends the treatments that fit it, and builds each one from editable sky masks. */
export default function SkyPanel({ ctx, suggest, onMasks }: { ctx: EditCtx; suggest: Suggest | null; onMasks: () => void }) {
  const { sid } = ctx;
  const [state, setState] = useState<SkyState | null>(skyStore.get(sid) ?? null);
  const [busy, setBusy] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [swap, setSwap] = useState<SwapInfo | null>(swapCache.get(sid) ?? null);
  const [showAll, setShowAll] = useState(false);
  const [busySwap, setBusySwap] = useState(false);
  const pRef = useRef(ctx.p);
  useEffect(() => { pRef.current = ctx.p; });

  useEffect(() => {
    if (swapCache.has(sid)) return;
    let live = true;
    fetch(`${API_URL}/api/skies/${sid}`).then((r) => r.json()).then((d) => { if (live && d.skies) { swapCache.set(sid, d); setSwap(d); } }).catch(() => {});
    return () => { live = false; };
  }, [sid]);
  const chooseSky = async (skyId: string) => {
    setBusySwap(true);
    try {
      const d = await post<{ sky_replace: SkyReplace }>('/api/sky_plan', { session_id: sid, sky_id: skyId });
      ctx.update((c) => ({ ...c, sky_replace: d.sky_replace }), true, true);
    } catch (e) { ctx.flash(e instanceof Error ? e.message : 'Could not use that sky'); }
    finally { setBusySwap(false); }
  };
  const removeSky = () => ctx.update((c) => ({ ...c, sky_replace: { ...(c.sky_replace as SkyReplace), id: null } }), true, true);
  const cur = ctx.p.sky_replace;

  const save = (v: SkyState | null) => { if (v) skyStore.set(sid, v); else skyStore.delete(sid); setState(v); };
  const run = useCallback(async (id: string, k: number, before: Params) => {
    setBusy(true);
    try {
      // the server replaces any earlier sky treatment, so always send the current sliders
      const d = await post<{ params: Params; notes: string[] }>('/api/sky', { session_id: sid, treatment: id, params: pRef.current, strength: Math.round(k * 100) });
      ctx.update(() => d.params, true, true);
      save({ id, k, notes: d.notes, before });
    } catch (e) { ctx.flash(e instanceof Error ? e.message : 'Could not apply the sky treatment'); }
    finally { setBusy(false); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sid, ctx]);

  const pick = (id: string) => run(id, state?.k ?? 1, state?.before ?? pRef.current);
  const onStrength = (v: number) => {
    if (!state) return;
    save({ ...state, k: v / 100 });
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => run(state.id, v / 100, state.before), 260);
  };
  const remove = () => {
    if (!state) return;
    const b = state.before;
    ctx.update((c) => ({ ...c, regions: { ...c.regions, sky: b.regions.sky }, masks: c.masks.filter((l) => !l.id.startsWith('sky_')) }), true, true);
    save(null);
  };

  if (!suggest) return <p className="p-4 text-[12px] text-zinc-500">Reading the sky…</p>;
  if (!suggest.sky) return <p className="p-4 text-[12px] leading-relaxed text-zinc-500">No sky was found in this photo, so there is nothing for the sky studio to do. Use the Regions and Masks tabs for the rest of the picture.</p>;
  const sky = suggest.sky;

  return (
    <>
      <Sub title="This sky">
        <div className="rounded-lg bg-zinc-800/60 p-3">
          <div className="flex items-baseline justify-between">
            <span className="text-[14px] font-semibold text-zinc-100">{sky.label}</span>
            <span className="text-[11px] text-zinc-400">{sky.share.toFixed(0)}% of the photo</span>
          </div>
          <div className="mt-2 grid grid-cols-4 gap-2 text-center text-[10.5px] text-zinc-400">
            {[['Blue', sky.blue], ['Cloud', sky.cloud], ['Warm', sky.warm]].map(([l, v]) => (
              <div key={l as string}><div className="text-[13px] font-semibold text-zinc-200">{v}%</div>{l}</div>
            ))}
            <div><div className="text-[13px] font-semibold text-zinc-200">{sky.brightness.toFixed(2)}</div>Bright</div>
          </div>
          <p className="mt-2 text-[11px] leading-snug text-zinc-500">{sky.evidence.join(' · ')}</p>
        </div>
      </Sub>

      <Sub title="Premium treatments for this sky">
        <p className="text-[11px] leading-relaxed text-zinc-500">Ranked for this sky. Each is built from editable sky-only masks (top gradient, clouds, blue, horizon glow), so you can fine-tune them in the Masks tab.</p>
        <div className="grid grid-cols-2 gap-1.5">
          {suggest.sky_treatments.map((t) => (
            <button key={t.id} disabled={busy} onClick={() => pick(t.id)} title={t.result}
              className={`${chip(state?.id === t.id)} disabled:opacity-60 ${t.fit < 0.4 && state?.id !== t.id ? 'opacity-60' : ''}`}>
              <div className="text-[12.5px] font-medium leading-tight">{t.label}</div>
              <div className="mt-0.5 text-[10.5px] leading-snug text-zinc-400">{t.recommended ? '★ recommended' : t.fit < 0.4 ? 'poor fit' : t.result}</div>
            </button>
          ))}
        </div>
        {busy && <p className="text-[11px] text-zinc-500">Building the sky edit…</p>}
        {state && (
          <>
            <Slider label="Sky strength" value={state.k * 100} min={0} max={150} def={100} fmt={(v) => `${Math.round(v)}%`} onChange={onStrength} />
            <ul className="list-disc space-y-1 pl-4 text-[11px] leading-snug text-zinc-400">{state.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
            <div className="flex gap-1.5">
              <button onClick={onMasks} className={`${chip(false)} flex-1 text-center text-[12px]`}>Fine-tune in Masks →</button>
              <button onClick={remove} className={`${chip(false)} flex-1 text-center text-[12px]`}>Remove sky edit</button>
            </div>
          </>
        )}
      </Sub>
    
      <Sub title="Replace the sky" open={false}>
        <p className="text-[11px] leading-relaxed text-zinc-500">Swap in a real sky from the library (free CC0 panoramas), chosen to suit the light this photo was taken in. Optional, never part of a look unless you choose it.</p>
        {!swap && <p className="text-[11px] text-zinc-500">Checking this photo…</p>}
        {swap && !swap.available && <p className="text-[11px] text-amber-300">{swap.reason}</p>}
        {swap && swap.available && (
          <>
            <p className={`text-[11px] leading-snug ${swap.needed ? 'text-emerald-300' : 'text-zinc-400'}`}>{swap.needed ? '★ Worth replacing: ' : 'Not needed: '}{swap.reason}</p>
            {swap.warning && <p className="text-[11px] leading-snug text-amber-300">{swap.warning}</p>}
            <div className="grid grid-cols-2 gap-1.5">
              {swap.skies.slice(0, showAll ? swap.skies.length : 8).map((k) => (
                <button key={k.id} disabled={busySwap} onClick={() => chooseSky(k.id)} title={`${k.name} · ${k.reason}`}
                  className={`overflow-hidden rounded-lg border text-left transition-colors disabled:opacity-60 ${cur?.id === k.id ? 'border-indigo-400 bg-indigo-950' : 'border-zinc-700 hover:border-zinc-500'} ${k.fit < 0.4 ? 'opacity-60' : ''}`}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={`${API_URL}${k.thumb}`} alt={k.name} className="aspect-[8/5] w-full object-cover" loading="lazy" />
                  <div className="px-2 py-1.5">
                    <div className="truncate text-[11.5px] font-medium text-zinc-100">{k.label}</div>
                    <div className="text-[10px] text-zinc-400">{Math.round(k.fit * 100)}% match{k.fit < 0.4 ? ' · poor fit' : ''}</div>
                  </div>
                </button>
              ))}
            </div>
            <button onClick={() => setShowAll((v) => !v)} className={`${chip(false)} w-full text-center text-[12px]`}>{showAll ? 'Show only the best matches' : `Show all ${swap.skies.length} skies`}</button>
            {busySwap && <p className="text-[11px] text-zinc-500">Fitting the sky to this photo…</p>}
            {cur?.id && (
              <>
                {SWAP_CONTROLS.map((c) => (
                  <Slider key={c.key} label={c.label} hint={c.hint} value={cur[c.key] as number} min={c.min} max={c.max}
                    def={c.key === 'blend' ? 100 : c.key === 'zoom' ? 100 : undefined}
                    fmt={(v) => (c.key === 'azimuth' || c.key === 'lift' ? `${Math.round(v)}°` : `${Math.round(v)}`)}
                    onChange={(v) => ctx.set(['sky_replace', c.key], v)} />
                ))}
                <button onClick={removeSky} className={`${chip(false)} w-full text-center text-[12px]`}>Remove the new sky</button>
              </>
            )}
          </>
        )}
      </Sub>
    </>
  );
}
