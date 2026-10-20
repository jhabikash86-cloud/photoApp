'use client';

import { useEffect, useRef, useState } from 'react';
import { API_URL, LOOKS } from '../lib/params';

type Album = { id: string; name: string; count: number };
type Job = {
  id: string; album: string; look: string; status: string; done: number; total: number; failed: number;
  current: string; errors: string[]; edited_album_id: string | null;
};

export default function BatchDialog({ onClose }: { onClose: () => void }) {
  const [albums, setAlbums] = useState<Album[] | null>(null);
  const [configured, setConfigured] = useState(true);
  const [albumId, setAlbumId] = useState('');
  const [look, setLook] = useState('auto');
  const [limit, setLimit] = useState('');
  const [straighten, setStraighten] = useState(true);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState('');
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/api/immich/albums`).then((r) => r.json().then((d) => ({ ok: r.ok, d })))
      .then(({ ok, d }) => {
        if (!ok) { setError(d.detail || 'Could not load albums'); setAlbums([]); return; }
        setConfigured(d.configured); setAlbums(d.albums);
      }).catch(() => { setError(`Can't reach the server at ${API_URL}.`); setAlbums([]); });
    return () => { if (timer.current) clearInterval(timer.current); };
  }, []);

  const poll = (id: string) => {
    timer.current = setInterval(async () => {
      const res = await fetch(`${API_URL}/api/immich/jobs/${id}`);
      if (!res.ok) return;
      const j: Job = await res.json();
      setJob(j);
      if (['done', 'error', 'cancelled'].includes(j.status) && timer.current) clearInterval(timer.current);
    }, 2000);
  };

  const start = async () => {
    setError('');
    const res = await fetch(`${API_URL}/api/immich/batch`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ album_id: albumId, look, limit: limit ? parseInt(limit, 10) : null, straighten }),
    });
    const d = await res.json().catch(() => ({}));
    if (!res.ok) { setError(d.detail || 'Could not start'); return; }
    setJob({ id: d.job_id, album: '', look, status: 'starting', done: 0, total: 0, failed: 0, current: '', errors: [], edited_album_id: null });
    poll(d.job_id);
  };

  const cancel = () => job && fetch(`${API_URL}/api/immich/jobs/${job.id}/cancel`, { method: 'POST' });
  const running = job && ['starting', 'running'].includes(job.status);
  const pct = job && job.total ? Math.round((job.done / job.total) * 100) : 0;

  return (
    <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center" onClick={() => !running && onClose()}>
      <div className="w-[480px] rounded-lg bg-[#232327] border border-zinc-700 p-5 text-sm" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-zinc-100">Batch edit an Immich album</h2>
          <button onClick={onClose} disabled={!!running} className="text-zinc-400 hover:text-white disabled:opacity-30">✕</button>
        </div>
        {!configured && <p className="text-amber-300 text-xs">Immich isn&apos;t configured. Add IMMICH_URL and IMMICH_API_KEY to the .env file and restart the backend.</p>}
        {albums === null && <p className="text-zinc-400 text-xs">Loading albums…</p>}
        {error && <p className="text-red-400 text-xs mb-2">{error}</p>}
        {configured && albums && !job && (
          <div className="space-y-3">
            <label className="block text-xs text-zinc-400">Album
              <select value={albumId} onChange={(e) => setAlbumId(e.target.value)} className="mt-1 w-full bg-zinc-800 rounded px-2 py-1.5 text-zinc-100">
                <option value="">Choose an album…</option>
                {albums.map((a) => <option key={a.id} value={a.id}>{a.name} ({a.count})</option>)}
              </select>
            </label>
            <label className="block text-xs text-zinc-400">Look
              <select value={look} onChange={(e) => setLook(e.target.value)} className="mt-1 w-full bg-zinc-800 rounded px-2 py-1.5 text-zinc-100">
                <option value="drama_auto">✨ Scene-adaptive Drama — the best-fitting look for each photo</option>
                {LOOKS.map((l) => <option key={l.id} value={l.id}>{l.label} — {l.hint}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-2 text-xs text-zinc-300"><input type="checkbox" checked={straighten} onChange={(e) => setStraighten(e.target.checked)} /> Level crooked horizons automatically (only when the horizon is clear)</label>
            <label className="block text-xs text-zinc-400">Only the first N photos (optional, good for a trial)
              <input value={limit} onChange={(e) => setLimit(e.target.value.replace(/\D/g, ''))} placeholder="all" className="mt-1 w-full bg-zinc-800 rounded px-2 py-1.5 text-zinc-100" />
            </label>
            <p className="text-[11px] text-zinc-500 leading-relaxed">
              Each photo is edited at full resolution with the automatic edit and uploaded to Immich as a <b>new</b> photo.
              Your originals are never changed. The results are collected in a new album called “&lt;album&gt; - Edited”.
              Large photos take 10–30 seconds each.
            </p>
            <button onClick={start} disabled={!albumId} className="w-full py-2 rounded bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 font-semibold">Start batch</button>
          </div>
        )}
        {job && (
          <div className="space-y-3">
            <div className="h-2 rounded bg-zinc-700 overflow-hidden"><div className="h-full bg-indigo-500 transition-all" style={{ width: `${pct}%` }} /></div>
            <p className="text-xs text-zinc-300">
              {job.status === 'done' ? `Finished: ${job.done - job.failed} edited` : job.status === 'cancelled' ? 'Cancelled' :
                job.status === 'error' ? 'Stopped with an error' : `Editing ${job.done + 1} of ${job.total || '…'}`}
              {job.failed ? ` · ${job.failed} failed` : ''}
            </p>
            {running && <p className="text-[11px] text-zinc-500 truncate">{job.current}</p>}
            {job.status === 'done' && job.edited_album_id && <p className="text-xs text-emerald-400">New album “{job.album} - Edited” is in your Immich.</p>}
            {job.errors.length > 0 && <ul className="text-[11px] text-amber-300 max-h-24 overflow-y-auto list-disc pl-4">{job.errors.slice(0, 8).map((e, i) => <li key={i}>{e}</li>)}</ul>}
            <div className="flex gap-2">
              {running ? <button onClick={cancel} className="flex-1 py-2 rounded bg-zinc-700 hover:bg-zinc-600">Cancel after this photo</button>
                : <button onClick={onClose} className="flex-1 py-2 rounded bg-indigo-600 hover:bg-indigo-500">Close</button>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
