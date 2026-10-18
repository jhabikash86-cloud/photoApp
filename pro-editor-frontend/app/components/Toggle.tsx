'use client';

type Props = { checked: boolean; onChange: (v: boolean) => void; label?: string; title?: string; small?: boolean };

export default function Toggle({ checked, onChange, label, title, small }: Props) {
  const w = small ? 'h-4 w-7' : 'h-5 w-9';
  const k = small ? 'h-3 w-3' : 'h-4 w-4';
  return (
    <label className="inline-flex items-center gap-2 cursor-pointer select-none" title={title}>
      <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}
        className={`relative ${w} shrink-0 rounded-full transition-colors ${checked ? 'bg-indigo-500' : 'bg-zinc-600'}`}>
        <span className={`absolute top-0.5 ${k} rounded-full bg-white transition-all ${checked ? (small ? 'left-3.5' : 'left-[18px]') : 'left-0.5'}`} />
      </button>
      {label && <span className="text-[11px] text-zinc-300">{label}</span>}
    </label>
  );
}
