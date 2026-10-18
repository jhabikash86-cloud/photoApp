'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Slider from './Slider';
import { Sub } from './panels';
import { API_URL, EditCtx, Params } from '../lib/params';

export type SkySummary = { kind: string; label: string; confidence: number; share: number; brightness: number; cloud: number; blue: number; warm: number; evidence: string[] };
export type SkyTreatment = { id: string; label: string; result: string; fit: number; recommended: boolean };
export type Suggest = {
  sky: SkySummary | null; sky_treatments: SkyTreatment[];
  looks: { id: string; label: string; result: string; fit: number; reason: string }[];
};

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
  const pRef = useRef(ctx.p);
  useEffect(() => { pRef.current = ctx.p; });

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
    </>
  );
}
