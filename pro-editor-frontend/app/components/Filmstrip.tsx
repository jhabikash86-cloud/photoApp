'use client';

import { API_URL, Item } from '../lib/params';

type Props = {
  items: Item[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
  onAdd: () => void;
};

export default function Filmstrip({ items, activeId, onSelect, onRemove, onAdd }: Props) {
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
