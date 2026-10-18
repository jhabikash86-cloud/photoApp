'use client';

import { useEffect, useRef, useState } from 'react';
import { signed } from '../lib/params';

type Props = {
  label: string;
  value: number;
  def?: number;               // the Auto value; double-click the label (or press ↺) to return to it
  min?: number;
  max?: number;
  step?: number;
  fmt?: (v: number) => string;
  onChange: (v: number) => void;
  disabled?: boolean;
  tint?: string;              // track fill colour
  track?: string;             // optional CSS gradient for the track (e.g. a hue rainbow)
  hint?: string;
};

const THUMB = 34;             // px, must match .pro-range thumb width in globals.css

/** − / + button: one tap = one step, hold to repeat (and accelerate), Shift = ×10. */
function Step({ dir, apply, disabled }: { dir: -1 | 1; apply: (dir: -1 | 1, big: boolean) => void; disabled?: boolean }) {
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stop = () => { if (timer.current) { clearTimeout(timer.current); timer.current = null; } };
  useEffect(() => stop, []);
  const start = (e: React.PointerEvent) => {
    e.preventDefault();
    const big = e.shiftKey;
    apply(dir, big);
    let delay = 340;
    const tick = () => { apply(dir, big); delay = Math.max(22, delay * 0.74); timer.current = setTimeout(tick, delay); };
    timer.current = setTimeout(tick, delay);
  };
  return (
    <button type="button" tabIndex={-1} disabled={disabled} aria-label={dir < 0 ? 'Decrease' : 'Increase'}
      onPointerDown={start} onPointerUp={stop} onPointerLeave={stop} onPointerCancel={stop}
      className="h-8 w-8 shrink-0 rounded-lg bg-zinc-800 text-lg leading-none text-zinc-200 hover:bg-zinc-700 active:bg-indigo-600 active:text-white
        disabled:opacity-30 select-none touch-none">
      {dir < 0 ? '−' : '+'}
    </button>
  );
}

export default function Slider({ label, value, def, min = -100, max = 100, step = 1, fmt, onChange, disabled, tint, track, hint }: Props) {
  const [text, setText] = useState<string | null>(null);
  const latest = useRef(value);
  useEffect(() => { latest.current = value; }, [value]);   // so a held + / − always builds on the newest value
  const bipolar = min < 0;
  const frac = (v: number) => (v - min) / (max - min);
  const pos = (v: number) => `calc(${THUMB / 2}px + ${frac(v)} * (100% - ${THUMB}px))`;
  const changed = def !== undefined && Math.abs(value - def) > 0.5;
  const accent = tint ?? '#818cf8';

  const bump = (dir: -1 | 1, big: boolean) => {
    const next = Math.min(max, Math.max(min, Math.round((latest.current + dir * step * (big ? 10 : 1)) / step) * step));
    latest.current = next;
    onChange(next);
  };

  return (
    <div className={`select-none ${disabled ? 'opacity-40 pointer-events-none' : ''}`}>
      <div className="flex items-center justify-between gap-2 text-[13px]">
        <span className="flex min-w-0 items-center gap-1.5">
          <span className={`truncate ${changed ? 'text-zinc-50' : 'text-zinc-300'}`}
            title={hint ?? (def !== undefined ? 'Double-click to reset to the Auto value' : undefined)}
            onDoubleClick={() => def !== undefined && onChange(def)}>
            {label}
          </span>
          {changed && (
            <button onClick={() => def !== undefined && onChange(def)} title="Edited by you · click to return to the Auto value"
              className="rounded px-1 text-sm leading-none text-amber-300 hover:bg-zinc-800 hover:text-amber-100">↺</button>
          )}
        </span>
        <span className="flex shrink-0 items-center gap-1">
          <Step dir={-1} apply={bump} />
          <input
            value={text ?? (fmt ? fmt(value) : bipolar ? signed(value) : String(Math.round(value)))}
            onFocus={(e) => { setText(String(Math.round(value))); e.currentTarget.select(); }}
            onChange={(e) => setText(e.target.value)}
            onBlur={() => {
              const n = parseFloat(text ?? '');
              if (!Number.isNaN(n)) onChange(Math.min(max, Math.max(min, n)));
              setText(null);
            }}
            onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
            className={`w-16 rounded-md bg-transparent px-1 text-center text-[15px] font-semibold tabular-nums outline-none focus:bg-zinc-800 ${changed ? 'text-zinc-50' : 'text-zinc-300'}`}
          />
          <Step dir={1} apply={bump} />
        </span>
      </div>
      <div className="relative h-9">
        <div className="absolute inset-x-0 top-[13px] h-[10px] rounded-full bg-zinc-800" style={track ? { background: track } : undefined} />
        {!track && (
          <div className="absolute top-[13px] h-[10px] rounded-full"
            style={{
              left: bipolar ? `min(${pos(0)}, ${pos(value)})` : 0,
              width: bipolar ? `max(${pos(0)}, ${pos(value)}) - min(${pos(0)}, ${pos(value)})` : pos(value),
              background: accent, opacity: 0.85,
            }} />
        )}
        {def !== undefined && (
          <div className="absolute top-[9px] h-[18px] w-[3px] -translate-x-1/2 rounded bg-zinc-300/60" style={{ left: pos(def) }} title="Auto value" />
        )}
        <input type="range" className="pro-range absolute inset-0" min={min} max={max} step={step} value={value}
          disabled={disabled} aria-label={label}
          onChange={(e) => onChange(Number(e.target.value))}
          onDoubleClick={() => def !== undefined && onChange(def)} />
      </div>
    </div>
  );
}
