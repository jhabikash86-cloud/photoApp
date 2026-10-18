'use client';

import { useCallback, useEffect, useState } from 'react';
import { API_URL } from '../lib/params';

type Album = { id: string; title: string; count: number; raw: number; edited: number; group?: string };
type Group = { title: string; albums: Album[] };
type Photo = { id: number; path: string; kind: 'raw' | 'processed'; origin: string; country: string | null; place: string; taken: string; subject: string | null; tags: string | null; camera: string | null; sharp: number | null };
const SUBJECTS = ['Landscape', 'Beach & Coast', 'Lakes, Rivers & Waterfalls', 'Architecture', 'Street & City', 'Portrait', 'People', 'Wildlife & Animals', 'Food & Details', 'Night', 'Indoor', 'Other'];
const PAGE = 120;

const chip = (on: boolean) =>
  `rounded-lg border px-2.5 py-1.5 text-[12px] transition-colors ${on ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-300 hover:border-zinc-500 hover:bg-zinc-800/60'}`;

/** Browse the travel library (albums by country, subject, place; RAW and edited kept apart) and open photos straight from the drive. */
export default function LibraryDialog({ onClose, onOpen }: { onClose: () => void; onOpen: (files: File[]) => void }) {
  const [groups, setGroups] = useState<Group[] | null>(null);
  const [connected, setConnected] = useState(true);
  const [album, setAlbum] = useState('all');
  const [title, setTitle] = useState('All travel photos');
  const [kind, setKind] = useState<'' | 'raw' | 'processed'>('');
  const [subject, setSubject] = useState('');
  const [order, setOrder] = useState('taken');
  const [items, setItems] = useState<Photo[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [sel, setSel] = useState<Set<number>>(new Set());
  const [opening, setOpening] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API_URL}/api/library/albums`).then((r) => r.json()).then((d) => { setGroups(d.groups); setConnected(d.connected); })
      .catch(() => setError(`Can't reach the server at ${API_URL}.`));
  }, []);

  const query = useCallback((offset: number) => {
    const q = new URLSearchParams({ album, order, limit: String(PAGE), offset: String(offset) });
    if (kind) q.set('kind', kind);
    if (subject) q.set('subject', subject);
    return fetch(`${API_URL}/api/library/photos?${q}`).then((r) => r.json());
  }, [album, order, kind, subject]);

  useEffect(() => {
    let live = true;
    Promise.resolve().then(() => { if (live) setLoading(true); });
    query(0).then((d) => { if (live) { setItems(d.photos ?? []); setTotal(d.total ?? 0); setSel(new Set()); } })
      .catch(() => { if (live) setError('Could not load photos.'); })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [query]);

  const more = () => { setLoading(true); query(items.length).then((d) => { setItems((cur) => [...cur, ...(d.photos ?? [])]); }).finally(() => setLoading(false)); };
  const toggle = (id: number) => setSel((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  const open = async (ids: number[]) => {
    if (!ids.length) return;
    setOpening(true); setError('');
    try {
      const files: File[] = [];
      for (const id of ids) {
        const res = await fetch(`${API_URL}/api/library/file/${id}`);
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Could not read the file from the drive.');
        const name = items.find((p) => p.id === id)?.path.split('/').pop() ?? `photo-${id}`;
        files.push(new File([await res.blob()], name));
      }
      onOpen(files);
      onClose();
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not open that photo.'); }
    finally { setOpening(false); }
  };

  return (
    <div className="fixed inset-0 z-50 flex bg-zinc-950/97 backdrop-blur-sm" role="dialog" aria-label="Photo library">
      <aside className="w-[290px] shrink-0 overflow-y-auto border-r border-zinc-800 bg-[#18181b] p-3">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-[15px] font-semibold text-zinc-100">Travel library</h2>
          <button onClick={onClose} className="rounded-md px-2 py-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100" aria-label="Close">✕</button>
        </div>
        {!connected && <p className="mb-3 rounded-lg bg-amber-950 p-2 text-[11px] text-amber-200">The external drive is not connected: thumbnails already made still show, but photos can&apos;t be opened until it is plugged in.</p>}
        {!groups && <p className="text-[12px] text-zinc-500">Loading albums…</p>}
        {groups?.map((g) => (
          <div key={g.title} className="mb-4">
            <div className="mb-1 px-1 text-[10px] uppercase tracking-widest text-zinc-500">{g.title}</div>
            {g.albums.map((a) => (
              <button key={a.id} onClick={() => { setAlbum(a.id); setTitle(a.title); }}
                className={`mb-0.5 flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-left text-[13px] ${album === a.id ? 'bg-indigo-950 text-indigo-100' : 'text-zinc-300 hover:bg-zinc-800'}`}>
                <span className="truncate pr-2">{a.group && a.group !== 'Europe' && g.title === 'By country' ? `${a.title}` : a.title}</span>
                <span className="shrink-0 text-[11px] tabular-nums text-zinc-500">{a.count}</span>
              </button>
            ))}
          </div>
        ))}
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 px-4 py-3">
          <div className="mr-2">
            <div className="text-[16px] font-semibold text-zinc-100">{title}</div>
            <div className="text-[11px] text-zinc-500">{total.toLocaleString()} photos · in date order</div>
          </div>
          <div className="flex gap-1.5">
            {([['', 'All'], ['raw', 'RAW only'], ['processed', 'Edited & JPEG']] as const).map(([k, l]) => <button key={k} onClick={() => setKind(k)} className={chip(kind === k)}>{l}</button>)}
          </div>
          <select value={subject} onChange={(e) => setSubject(e.target.value)} className="rounded-lg border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-[12px] text-zinc-200">
            <option value="">Any subject</option>
            {SUBJECTS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <select value={order} onChange={(e) => setOrder(e.target.value)} className="rounded-lg border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-[12px] text-zinc-200">
            <option value="taken">Oldest first</option><option value="newest">Newest first</option><option value="sharp">Sharpest first</option>
          </select>
          <div className="ml-auto flex items-center gap-2">
            {error && <span className="text-[12px] text-amber-300">{error}</span>}
            <button disabled={!sel.size || opening} onClick={() => open([...sel])}
              className="rounded-lg bg-indigo-600 px-4 py-2 text-[13px] font-medium text-white hover:bg-indigo-500 disabled:opacity-40">
              {opening ? 'Opening…' : sel.size ? `Open ${sel.size} in editor` : 'Select photos'}
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <div className="grid gap-2.5" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))' }}>
            {items.map((p) => (
              <button key={p.id} onClick={() => toggle(p.id)} onDoubleClick={() => open([p.id])} title={`${p.path}${p.camera ? ` · ${p.camera}` : ''}`}
                className={`group relative overflow-hidden rounded-lg border bg-zinc-900 text-left ${sel.has(p.id) ? 'border-indigo-400 ring-2 ring-indigo-500/60' : 'border-zinc-800 hover:border-zinc-600'}`}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`${API_URL}/api/library/thumb/${p.id}`} alt="" loading="lazy" className="aspect-[3/2] w-full bg-zinc-800 object-cover" />
                <span className={`absolute left-1.5 top-1.5 rounded px-1.5 py-0.5 text-[10px] font-semibold ${p.kind === 'raw' ? 'bg-emerald-900/90 text-emerald-200' : 'bg-zinc-900/90 text-zinc-200'}`}>{p.kind === 'raw' ? 'RAW' : p.origin === 'edited' ? 'EDITED' : 'JPEG'}</span>
                {sel.has(p.id) && <span className="absolute right-1.5 top-1.5 rounded-full bg-indigo-500 px-1.5 text-[11px] text-white">✓</span>}
                <div className="px-2 py-1.5">
                  <div className="truncate text-[11px] text-zinc-300">{p.subject ?? '—'}{p.place && p.place !== '(by date)' ? ` · ${p.place}` : ''}</div>
                  <div className="text-[10px] text-zinc-500">{p.taken?.slice(0, 10)}{p.country ? ` · ${p.country}` : ''}</div>
                </div>
              </button>
            ))}
          </div>
          {loading && <p className="py-4 text-center text-[12px] text-zinc-500">Loading…</p>}
          {!loading && items.length < total && <div className="py-4 text-center"><button onClick={more} className={chip(false)}>Show more ({total - items.length} left)</button></div>}
          {!loading && total === 0 && <p className="py-10 text-center text-[13px] text-zinc-500">No photos match. The catalog may still be analysing: subjects fill in as it runs.</p>}
          <p className="pt-2 text-center text-[11px] text-zinc-600">Click to select · double-click to open · originals are never changed (the drive is read-only)</p>
        </div>
      </main>
    </div>
  );
}
