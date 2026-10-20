'use client';

import { useState } from 'react';
import { API_URL, GEOMETRY_OFF, Item, LONGEXP_OFF, Params } from '../lib/params';

type Opts = { tone: boolean; drama: boolean; longexp: boolean; aspect: boolean; wb: boolean };

/** Copy the edit of `src` onto `tgt`: sliders and grades as they are, the Drama look re-built for that photo's own scene (done separately), nothing per-photo (masks, brush strokes, crop angle). */
export function mergeEdit(src: Params, tgt: Params, o: Opts): Params {
  const out: Params = JSON.parse(JSON.stringify(tgt));
  if (o.tone) {
    const keep = { temperature: out.basic.temperature, tint: out.basic.tint };
    out.basic = { ...src.basic, ...(o.wb ? {} : keep) };
    out.mixer = JSON.parse(JSON.stringify(src.mixer));
    out.grading = JSON.parse(JSON.stringify(src.grading));
    out.grading_enabled = src.grading_enabled;
    out.balance_shadows = [...src.balance_shadows]; out.balance_mids = [...src.balance_mids]; out.balance_highs = [...src.balance_highs];
    out.curve = JSON.parse(JSON.stringify(src.curve));
    out.tone = JSON.parse(JSON.stringify(src.tone));
    out.look = src.look;
    out.intensity = src.intensity;
    out.dodge_burn = JSON.parse(JSON.stringify(src.dodge_burn));
    out.lut = src.lut ? { ...src.lut } : out.lut;
    out.effects = { ...src.effects, denoise: out.effects.denoise, denoise_color: out.effects.denoise_color };      // noise is a property of each shot
    out.regions = Object.fromEntries(Object.entries(src.regions).map(([k, r]) => [k, { ...r, strokes: tgt.regions[k]?.strokes ?? [] }])) as Params['regions'];
  }
  if (o.wb) out.wb_mode = src.wb_mode;
  if (o.longexp) out.longexp = src.longexp ? { ...src.longexp } : { ...LONGEXP_OFF };
  if (o.aspect) {
    const a = src.geometry?.aspect ?? 'original';
    const g = { ...(tgt.geometry ?? GEOMETRY_OFF), aspect: a };
    out.geometry = { ...g, enabled: Math.abs(g.angle) > 0.001 || a !== 'original' || g.zoom > 1.001 };
  }
  return out;
}

export default function SyncDialog({ items, source, onApply, onClose, flash }: {
  items: Item[]; source: Item; onApply: (id: string, params: Params) => Promise<void>; onClose: () => void; flash: (m: string) => void;
}) {
  const targets = items.filter((i) => i.id !== source.id && i.status === 'ready' && i.sid && i.params);
  const hasDrama = !!source.params?.drama?.enabled && !!source.params.drama.name;
  const [o, setO] = useState<Opts>({ tone: true, drama: hasDrama, longexp: !!source.params?.longexp?.enabled, aspect: false, wb: false });
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState('');
  const set = (k: keyof Opts) => setO((c) => ({ ...c, [k]: !c[k] }));

  const run = async () => {
    if (!source.params || !source.sid) return;
    setBusy(true);
    try {
      let lookId: string | null = null;
      if (o.drama && hasDrama) {
        const list = await (await fetch(`${API_URL}/api/drama/${source.sid}`)).json();
        lookId = (list.looks ?? []).find((l: { id: string; label: string }) => l.label === source.params!.drama!.name)?.id ?? null;
        if (!lookId) flash('Could not tell which Drama look this is: copying the sliders only');
      }
      let n = 0;
      for (const t of targets) {
        setProgress(`${++n} / ${targets.length} · ${t.name}`);
        let next = mergeEdit(source.params, t.params!, o);
        if (lookId) {
          const res = await fetch(`${API_URL}/api/drama`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: t.sid, look_id: lookId, params: next, strength: Math.round(source.params.drama!.amount) }) });
          if (res.ok) next = (await res.json()).params as Params;
        }
        await onApply(t.id, next);
      }
      flash(`Applied to ${targets.length} photo${targets.length === 1 ? '' : 's'}`);
      onClose();
    } catch (e) { flash(e instanceof Error ? e.message : 'Could not sync'); }
    finally { setBusy(false); }
  };

  const row = (k: keyof Opts, title: string, hint: string, disabled = false) => (
    <label className={`flex items-start gap-3 rounded-lg border p-3 ${disabled ? 'opacity-40' : 'cursor-pointer hover:border-zinc-500'} ${o[k] && !disabled ? 'border-indigo-400 bg-indigo-950/60' : 'border-zinc-700'}`}>
      <input type="checkbox" className="mt-1" checked={o[k] && !disabled} disabled={disabled} onChange={() => set(k)} />
      <span><span className="block text-zinc-100">{title}</span><span className="block text-[11px] leading-snug text-zinc-400">{hint}</span></span>
    </label>
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-3" onClick={() => !busy && onClose()}>
      <div className="max-h-[92vh] w-[500px] max-w-full overflow-y-auto rounded-lg border border-zinc-700 bg-[#232327] p-5 text-sm" onClick={(e) => e.stopPropagation()}>
        <div className="mb-1 flex items-center justify-between">
          <h2 className="font-semibold text-zinc-100">Apply this edit to the other {targets.length} photo{targets.length === 1 ? '' : 's'}</h2>
          <button onClick={onClose} disabled={busy} className="text-zinc-400 hover:text-white disabled:opacity-30">✕</button>
        </div>
        <p className="mb-3 text-[11px] text-zinc-500">From {source.name}. Each photo keeps its own masks, brush strokes and crop angle.</p>
        <div className="space-y-2">
          {row('tone', 'Tone, colour and grades', 'Light and colour sliders, curves, grading wheels, dodge & burn, effects (noise reduction stays per photo)')}
          {row('drama', 'The Drama look, re-built for each photo', hasDrama ? `“${source.params!.drama!.name}” re-generated from each photo's own scene, so skies, light and colour suit each one` : 'Pick a Drama look first (Looks tab)', !hasDrama)}
          {row('longexp', 'Long exposure / ND look', 'Silky water, cloud streaks and graduated ND (only where the photo has water or sky)', !source.params?.longexp?.enabled)}
          {row('aspect', 'Crop shape', `Same aspect ratio (${source.params?.geometry?.aspect ?? 'original'}), each photo straightened on its own`, (source.params?.geometry?.aspect ?? 'original') === 'original')}
          {row('wb', 'White balance', 'Only if the shots were taken in the same light')}
        </div>
        {busy && <p className="mt-3 text-xs text-zinc-400">{progress}</p>}
        <div className="mt-4 flex gap-2">
          <button onClick={run} disabled={busy || !targets.length} className="flex-1 rounded-lg bg-indigo-600 py-2 font-semibold hover:bg-indigo-500 disabled:opacity-40">{busy ? 'Applying…' : `Apply to ${targets.length} photo${targets.length === 1 ? '' : 's'}`}</button>
        </div>
      </div>
    </div>
  );
}
