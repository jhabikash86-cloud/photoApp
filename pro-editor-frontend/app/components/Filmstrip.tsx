'use client';

import { API_URL, Item } from '../lib/params';

type Props = {
  items: Item[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
  onAdd: () => void;
};

const ham = (a: string, b: string) => {
  let x = BigInt('0x' + a) ^ BigInt('0x' + b), n = 0;
  while (x) { n += Number(x & BigInt(1)); x >>= BigInt(1); }
  return n;
};

/** Photos that are near-copies of an earlier, equally sharp or sharper shot: the loser of each pair gets a 'duplicate' flag. */
function duplicates(items: Item[]): Set<string> {
  const out = new Set<string>();
  const ready = items.filter((i) => i.status === 'ready' && i.analysis?.quality?.dhash);
  for (let i = 0; i < ready.length; i++) for (let j = i + 1; j < ready.length; j++) {
    const a = ready[i], b = ready[j];
    if (ham(a.analysis!.quality!.dhash!, b.analysis!.quality!.dhash!) <= 5) {
      const sa = a.analysis!.quality!.sharp_ratio ?? 0, sb = b.analysis!.quality!.sharp_ratio ?? 0;
      out.add(sa >= sb ? b.id : a.id);
    }
  }
  return out;
}

export default function Filmstrip({ items, activeId, onSelect, onRemove, onAdd }: Props) {
  const dups = duplicates(items);
  return (
    <div className="h-[92px] shrink-0 border-t border-zinc-800 bg-[#18181b] flex items-center gap-2 px-3 overflow-x-auto">
      {items.map((it) => (
        <div key={it.id} onClick={() => onSelect(it.id)}
          className={`relative h-[72px] w-[96px] shrink-0 rounded overflow-hidden cursor-pointer border-2 group
            ${it.id === activeId ? 'border-indigo-400' : 'border-transparent hover:border-zinc-600'}`}>
          {it.thumb ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={`${API_URL}${it.thumb}`} alt={it.name} className="h-full w-full object-cover" />
          ) : (
            <div className="h-full w-full bg-zinc-800 flex items-center justify-center text-[10px] text-zinc-500">{it.raw ? 'RAW' : '…'}</div>
          )}
          {(it.status === 'uploading' || it.status === 'analyzing' || it.status === 'queued') && (
            <div className="absolute inset-0 bg-black/55 flex flex-col items-center justify-center gap-1">
              <div className="h-3.5 w-3.5 rounded-full border-2 border-indigo-300 border-t-transparent animate-spin" />
              <span className="text-[9px] text-indigo-100">
                {it.status === 'uploading' ? `${Math.round(it.progress * 100)}%` : it.status === 'analyzing' ? 'analysing' : 'queued'}
              </span>
            </div>
          )}
          {it.status === 'error' && <div className="absolute inset-0 bg-red-900/70 flex items-center justify-center text-[10px] text-red-100 text-center p-1">failed</div>}
          {it.status === 'ready' && (() => {
            const q = it.analysis?.quality;
            if (!q) return null;
            const tags: { t: string; tip: string }[] = [];
            if (q.blur === 'blurry') tags.push({ t: 'blurry', tip: 'Looks out of focus or shaken: a candidate to reject' });
            else if (q.blur === 'soft') tags.push({ t: 'soft', tip: 'A little soft: check it at 100%' });
            if (dups.has(it.id)) tags.push({ t: 'duplicate', tip: 'Nearly the same picture as another open photo, and not the sharper one' });
            if (q.exposure === 'dark') tags.push({ t: 'dark', tip: 'Very dark overall' });
            if (q.exposure === 'bright') tags.push({ t: 'bright', tip: 'Very bright: highlights may be blown' });
            return tags.length ? <div className="absolute left-0.5 top-0.5 flex flex-col gap-0.5">{tags.map((g) => <span key={g.t} title={g.tip} className={`rounded px-1 text-[9px] leading-[13px] ${g.t === 'blurry' ? 'bg-red-600/90 text-white' : 'bg-amber-500/90 text-black'}`}>{g.t}</span>)}</div> : null;
          })()}
          <div className="absolute bottom-0 inset-x-0 bg-black/55 px-1 text-[9px] text-zinc-200 truncate">{it.name}</div>
          <button onClick={(e) => { e.stopPropagation(); onRemove(it.id); }} aria-label="Remove"
            className="absolute right-0.5 top-0.5 hidden group-hover:block h-4 w-4 rounded bg-black/70 text-[10px] leading-4 text-white">×</button>
        </div>
      ))}
      <button onClick={onAdd}
        className="h-[72px] w-[72px] shrink-0 rounded border border-dashed border-zinc-600 text-zinc-400 hover:text-white hover:border-zinc-400 text-2xl">+</button>
    </div>
  );
}
