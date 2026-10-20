'use client';

import { useEffect, useRef, useState } from 'react';
import { API_URL, Item } from '../lib/params';

type Preset = { id: string; label: string; format: string; max_edge: number | null };
type Job = { id: string; status: string; done: number; total: number; failed: number; current: string; errors: string[]; url: string | null };

const NOTES: Record<string, string> = {
  full: 'Everything you shot, at full resolution. For keeping and for printing at home.',
  web: 'For websites and email: sized to 2048 px and sharpened for the screen.',
  social: 'For Instagram and messages: 1350 px, sharpened.',
  tiny: 'Small and light: a WebP for quick previews.',
  print: '16-bit TIFF at full size: the print master, with the most room for further editing.',
  png16: '16-bit PNG at full size: a lossless archive copy.',
};

export default function ExportDialog({ items, activeId, onClose }: { items: Item[]; activeId: string | null; onClose: () => void }) {
  const ready = items.filter((i) => i.status === 'ready' && i.sid && i.params);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [preset, setPreset] = useState('web');
  const [scope, setScope] = useState<'this' | 'all'>('this');
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState('');
  const [done, setDone] = useState('');
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const active = ready.find((i) => i.id === activeId) ?? ready[0];

  useEffect(() => {
    fetch(`${API_URL}/api/export_presets`).then((r) => r.json()).then((d) => setPresets(d.presets ?? [])).catch(() => setError(`Can't reach the server at ${API_URL}.`));
    return () => { if (timer.current) clearInterval(timer.current); };
  }, []);

  const save = async (url: string, name: string) => {
    const blob = await (await fetch(`${API_URL}${url}`)).blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const run = async () => {
    setError(''); setDone(''); setBusy(true);
    try {
      if (scope === 'this' && active) {
        const res = await fetch(`${API_URL}/api/export_to`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: active.sid, params: active.params, preset, name: active.name, version: Date.now() % 100000 }) });
        const d = await res.json();
        if (!res.ok) throw new Error(d.detail || 'Export failed');
        await save(d.url, d.filename);
        setDone(`Saved ${d.filename} (${d.size[0]} × ${d.size[1]})`);
        setBusy(false);
      } else {
        const res = await fetch(`${API_URL}/api/export_batch`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ preset, items: ready.map((i) => ({ session_id: i.sid, params: i.params, name: i.name })) }) });
        const d = await res.json();
        if (!res.ok) throw new Error(d.detail || 'Export failed');
        timer.current = setInterval(async () => {
          const r = await fetch(`${API_URL}/api/export_jobs/${d.job_id}`);
          if (!r.ok) return;
          const j: Job = await r.json();
          setJob(j);
          if (['done', 'error', 'cancelled'].includes(j.status)) {
            if (timer.current) clearInterval(timer.current);
            setBusy(false);
            if (j.status === 'done' && j.url) { await save(j.url, `neural-edit-${preset}.zip`); setDone(`Saved a ZIP of ${j.done - j.failed} photos${j.failed ? ` (${j.failed} failed)` : ''}`); }
          }
        }, 1500);
      }
    } catch (e) { setError(e instanceof Error ? e.message : 'Export failed'); setBusy(false); }
  };

  const cancel = () => job && fetch(`${API_URL}/api/export_jobs/${job.id}/cancel`, { method: 'POST' });
  const pct = job && job.total ? Math.round((job.done / job.total) * 100) : 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-3" onClick={() => !busy && onClose()}>
      <div className="max-h-[92vh] w-[520px] max-w-full overflow-y-auto rounded-lg border border-zinc-700 bg-[#232327] p-5 text-sm" onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-semibold text-zinc-100">Export</h2>
          <button onClick={onClose} disabled={busy} className="text-zinc-400 hover:text-white disabled:opacity-30">✕</button>
        </div>
        <div className="space-y-2">
          {presets.map((p) => (
            <label key={p.id} className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 ${preset === p.id ? 'border-indigo-400 bg-indigo-950/60' : 'border-zinc-700 hover:border-zinc-500'}`}>
              <input type="radio" name="preset" checked={preset === p.id} onChange={() => setPreset(p.id)} className="mt-1" />
              <span><span className="block text-zinc-100">{p.label}</span><span className="block text-[11px] leading-snug text-zinc-400">{NOTES[p.id] ?? ''}</span></span>
            </label>
          ))}
        </div>
        <div className="mt-4 flex gap-2 text-[13px]">
          <button onClick={() => setScope('this')} className={`flex-1 rounded-lg border px-3 py-2 ${scope === 'this' ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-300'}`}>This photo{active ? ` · ${active.name}` : ''}</button>
          <button onClick={() => setScope('all')} disabled={ready.length < 2} className={`flex-1 rounded-lg border px-3 py-2 disabled:opacity-40 ${scope === 'all' ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-300'}`}>All {ready.length} photos (ZIP)</button>
        </div>
        {error && <p className="mt-3 text-xs text-red-400">{error}</p>}
        {job && busy && (
          <div className="mt-3 space-y-1">
            <div className="h-2 overflow-hidden rounded bg-zinc-800"><div className="h-full bg-indigo-500 transition-all" style={{ width: `${pct}%` }} /></div>
            <p className="text-[11px] text-zinc-400">{job.done} / {job.total} · {job.current}{job.failed ? ` · ${job.failed} failed` : ''}</p>
          </div>
        )}
        {job && job.errors.length > 0 && <ul className="mt-2 list-disc pl-4 text-[11px] text-amber-300">{job.errors.slice(0, 4).map((e, i) => <li key={i}>{e}</li>)}</ul>}
        {done && <p className="mt-3 text-xs text-emerald-400">{done}</p>}
        <div className="mt-4 flex gap-2">
          <button onClick={run} disabled={busy || !active} className="flex-1 rounded-lg bg-indigo-600 py-2 font-semibold hover:bg-indigo-500 disabled:opacity-40">{busy ? 'Rendering…' : 'Export'}</button>
          {busy && job && <button onClick={cancel} className="rounded-lg border border-zinc-600 px-4 py-2 hover:bg-zinc-800">Cancel</button>}
        </div>
        <p className="mt-3 text-[11px] leading-snug text-zinc-500">Each photo keeps its own edit. JPEG and WebP keep the original EXIF (camera, lens, GPS). Large TIFFs can take a while.</p>
      </div>
    </div>
  );
}
