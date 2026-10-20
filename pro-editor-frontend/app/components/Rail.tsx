'use client';

import { ReactNode } from 'react';

export type TabId = 'auto' | 'crop' | 'light' | 'colour' | 'looks' | 'sky' | 'presets' | 'masks' | 'curves' | 'dodge' | 'effects' | 'info';

const I = (d: ReactNode) => (
  <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">{d}</svg>
);
export const RAIL: { id: TabId; label: string; icon: ReactNode }[] = [
  { id: 'auto', label: 'Auto', icon: I(<><path d="M5 19 19 5" /><path d="M14 4v3M12.5 5.5h3M19 11v3M17.5 12.5h3M7 3v2M6 4h2" /></>) },
  { id: 'crop', label: 'Crop', icon: I(<><path d="M7 3v14h14" /><path d="M3 7h14v14" /></>) },
  { id: 'light', label: 'Light', icon: I(<><circle cx="12" cy="12" r="4" /><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6 7 7M17 17l1.4 1.4M18.4 5.6 17 7M7 17l-1.4 1.4" /></>) },
  { id: 'colour', label: 'Colour', icon: I(<><path d="M12 3c4.5 5 6.5 7.8 6.5 10.5a6.5 6.5 0 0 1-13 0C5.5 10.8 7.5 8 12 3Z" /></>) },
  { id: 'looks', label: 'Looks', icon: I(<><rect x="3.5" y="5" width="17" height="14" rx="2.5" /><path d="M3.5 15 9 10l4 4 2.5-2 5 4.5" /></>) },
  { id: 'sky', label: 'Sky', icon: I(<><path d="M7 17h10a3.5 3.5 0 0 0 .3-7A5 5 0 0 0 7.6 9.2 4 4 0 0 0 7 17Z" /><path d="M4 21h16" opacity=".45" /></>) },
  { id: 'presets', label: 'Regions', icon: I(<><path d="M4 18c3-6 5-6 8-2s5 2 8-6" /><path d="M4 7h6M14 5h6" /></>) },
  { id: 'masks', label: 'Masks', icon: I(<><circle cx="12" cy="12" r="8" strokeDasharray="3 2.4" /><circle cx="12" cy="12" r="3.2" /></>) },
  { id: 'curves', label: 'Curves', icon: I(<><path d="M4 20C9 19 9 5 20 4" /><path d="M4 20h16V4" opacity=".45" /></>) },
  { id: 'dodge', label: 'Dodge', icon: I(<><circle cx="12" cy="12" r="8" /><path d="M12 4a8 8 0 0 0 0 16Z" fill="currentColor" /></>) },
  { id: 'effects', label: 'Effects', icon: I(<><path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3Z" /><path d="m18.5 16 .7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7.7-1.8Z" /></>) },
  { id: 'info', label: 'Info', icon: I(<><circle cx="12" cy="12" r="8.5" /><path d="M12 11v5M12 7.8v.2" /></>) },
];

/** Vertical icon rail (like Lightroom): click an icon, its list of actions opens beside it. */
export default function Rail({ tab, onTab, dots }: { tab: TabId; onTab: (t: TabId) => void; dots?: Partial<Record<TabId, number>> }) {
  return (
    <nav className="flex md:w-[72px] shrink-0 flex-row md:flex-col items-center gap-1 overflow-x-auto md:overflow-x-visible md:overflow-y-auto border-t md:border-t-0 md:border-l border-zinc-800 bg-[#18181b] px-2 md:px-0 py-1.5 md:py-3" aria-label="Edit tools">
      {RAIL.map((r) => {
        const on = tab === r.id;
        return (
          <button key={r.id} onClick={() => onTab(r.id)} title={r.label} aria-label={r.label} aria-pressed={on}
            className={`relative flex w-14 shrink-0 flex-col items-center gap-1 rounded-xl py-2 transition-colors ${on ? 'bg-zinc-100 text-zinc-900' : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100'}`}>
            {r.icon}
            <span className="text-[10.5px] font-medium leading-none">{r.label}</span>
            {!!dots?.[r.id] && <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-amber-400" title="Changed by hand" />}
          </button>
        );
      })}
    </nav>
  );
}
