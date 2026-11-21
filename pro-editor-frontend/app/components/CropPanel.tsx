'use client';

import { useEffect, useState } from 'react';
import { API_URL, EditCtx, GEOMETRY_OFF, GeometrySettings } from '../lib/params';
import { Sub } from './panels';
import Slider from './Slider';

const ASPECTS: { id: string; label: string }[] = [
  { id: 'original', label: 'Original' }, { id: '3:2', label: '3 : 2' }, { id: '4:5', label: '4 : 5' }, { id: '1:1', label: '1 : 1' },
  { id: '16:9', label: '16 : 9' }, { id: '5:4', label: '5 : 4' }, { id: '2:3', label: '2 : 3' }, { id: '9:16', label: '9 : 16' }, { id: '3:1', label: '3 : 1 panorama' },
];
const chip = (on: boolean) => `px-3 py-1.5 rounded-lg text-[12.5px] border transition-colors ${on ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-200 hover:border-zinc-500 hover:bg-zinc-800/60'}`;

type Suggest = { angle: number; confidence: number; source: string };

export default function CropPanel({ ctx, sid }: { ctx: EditCtx; sid: string }) {
  const g: GeometrySettings = ctx.p.geometry ?? GEOMETRY_OFF;
  const [sug, setSug] = useState<Suggest | null>(null);
  useEffect(() => {
    let live = true;
    fetch(`${API_URL}/api/geometry/${sid}`).then((r) => r.json()).then((d) => { if (live && typeof d.angle === 'number') setSug(d); }).catch(() => {});
    return () => { live = false; };
  }, [sid]);
  const setG = (patch: Partial<GeometrySettings>, commit = true) => ctx.update((c) => {
    const n = { ...(c.geometry ?? GEOMETRY_OFF), ...patch };
    return { ...c, geometry: { ...n, enabled: Math.abs(n.angle) > 0.001 || n.aspect !== 'original' || n.zoom > 1.001 } };
  }, commit, commit);
  const worth = !!sug && Math.abs(sug.angle) >= 0.4 && sug.confidence >= 0.25;
  const moved = g.zoom > 1.001 || g.aspect !== 'original';
  return (
    <>
      <Sub title="Straighten">
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => sug && setG({ angle: Math.max(-45, Math.min(45, sug.angle)) })} disabled={!worth} className={`${chip(false)} disabled:opacity-40`}
            title={worth && sug ? `Found from the ${sug.source}` : 'No clear horizon or straight lines to level by'}>
            {worth && sug ? `Auto-straighten (${sug.angle > 0 ? '+' : ''}${sug.angle.toFixed(1)}°)` : sug ? 'Looks level already' : 'Looking for the horizon…'}
          </button>
          {worth && sug && <span className="text-[11px] text-zinc-500">{sug.source} · {Math.round(sug.confidence * 100)}% sure</span>}
        </div>
        <Slider label="Angle" value={g.angle} min={-15} max={15} step={0.1} def={0} fmt={(v) => `${v > 0 ? '+' : ''}${v.toFixed(1)}°`} hint="Positive turns the picture clockwise"
          onChange={(v) => setG({ angle: v }, false)} />
      </Sub>
      <Sub title="Crop">
        {ctx.crop && (
          ctx.crop.editing ? (
            <div className="space-y-2 rounded-lg border border-indigo-400/50 bg-indigo-950/40 p-3">
              <p className="text-[12px] leading-snug text-indigo-100">Drag a corner or an edge of the box on the photo to crop from that side. Drag inside it to move it, or drag outside to draw a new one.</p>
              <div className="flex gap-2">
                <button onClick={ctx.crop.apply} className="flex-1 rounded-lg bg-indigo-600 py-2 text-[13px] font-semibold hover:bg-indigo-500">Apply crop</button>
                <button onClick={ctx.crop.cancel} className={`${chip(false)} px-4`}>Cancel</button>
              </div>
              <p className="text-[10.5px] text-indigo-200/70">Enter applies · Esc cancels</p>
            </div>
          ) : (
            <button onClick={ctx.crop.start} className={`${chip(g.aspect === 'free' && g.enabled)} w-full`}>
              {g.aspect === 'free' && g.enabled ? `Free crop · ${Math.round(((g.cx1 ?? 1) - (g.cx0 ?? 0)) * 100)}% × ${Math.round(((g.cy1 ?? 1) - (g.cy0 ?? 0)) * 100)}% of the picture (edit the box)` : 'Free crop: draw the box yourself'}
            </button>
          )
        )}
        <div className="flex flex-wrap gap-1.5">
          {ASPECTS.map((a) => <button key={a.id} onClick={() => { if (ctx.crop?.editing) ctx.crop.cancel(); setG({ aspect: a.id, x: 0, y: 0, cx0: 0, cy0: 0, cx1: 1, cy1: 1 }); }} className={chip(g.aspect === a.id)}>{a.label}</button>)}
        </div>
        <Slider label="Zoom in" value={g.zoom * 100} min={100} max={400} def={100} fmt={(v) => `${Math.round(v)}%`} onChange={(v) => setG({ zoom: v / 100 }, false)} />
        {moved && (
          <>
            <Slider label="Move left / right" value={g.x * 100} min={-100} max={100} def={0} fmt={(v) => `${Math.round(v)}`} onChange={(v) => setG({ x: v / 100 }, false)} />
            <Slider label="Move up / down" value={g.y * 100} min={-100} max={100} def={0} fmt={(v) => `${Math.round(v)}`} onChange={(v) => setG({ y: v / 100 }, false)} />
          </>
        )}
        <p className="text-[11px] leading-relaxed text-zinc-500">The crop is the largest window that fits the straightened picture, so there are never empty corners. Painting masks works on the full frame: reset the crop first, paint, then crop again.</p>
        {!g.enabled && (Math.abs(g.angle) > 0.001 || g.aspect !== 'original' || g.zoom > 1.001) && <button onClick={() => ctx.update((c) => ({ ...c, geometry: { ...(c.geometry ?? GEOMETRY_OFF), enabled: true } }), true, true)} className={`${chip(true)} w-full`}>Turn the crop back on</button>}
        {(g.enabled || Math.abs(g.angle) > 0.001 || g.aspect !== 'original' || g.zoom > 1.001) && <button onClick={() => ctx.update((c) => ({ ...c, geometry: { ...GEOMETRY_OFF } }), true, true)} className={`${chip(false)} w-full`}>Reset crop & straighten</button>}
      </Sub>
    </>
  );
}
