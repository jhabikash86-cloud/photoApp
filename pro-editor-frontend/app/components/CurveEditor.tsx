'use client';

import { useRef, useState } from 'react';
import { CURVE_PRESETS, IDENTITY, Pt, isIdentity, sampleCurve } from '../lib/curve';
import { Histogram, ToneCurve } from '../lib/params';

type Channel = 'rgb' | 'r' | 'g' | 'b';
const CH: { id: Channel; label: string; color: string }[] = [
  { id: 'rgb', label: 'RGB', color: '#e4e4e7' },
  { id: 'r', label: 'R', color: '#f87171' },
  { id: 'g', label: 'G', color: '#4ade80' },
  { id: 'b', label: 'B', color: '#60a5fa' },
];
const SIZE = 256;       // viewBox units
const PAD = 14;         // room around the plot so the end points are easy to grab
const MAX_POINTS = 16;

type Props = { tone: ToneCurve; hist?: Histogram; onChange: (next: ToneCurve, commit?: boolean) => void };

/** Draggable point curve (like Lightroom's): click to add a point, drag to move, double-click or right-click to remove. */
export default function CurveEditor({ tone, hist, onChange }: Props) {
  const [ch, setCh] = useState<Channel>('rgb');
  const [hover, setHover] = useState<number | null>(null);
  const [readout, setReadout] = useState<string>('');
  const svg = useRef<SVGSVGElement>(null);
  const dragIdx = useRef<number | null>(null);
  const pts = tone[ch];
  const color = CH.find((c) => c.id === ch)!.color;

  const toX = (v: number) => PAD + v * (SIZE - 2 * PAD);
  const toY = (v: number) => SIZE - PAD - v * (SIZE - 2 * PAD);
  const fromEvent = (e: React.PointerEvent | PointerEvent): Pt => {
    const r = svg.current!.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * SIZE, y = ((e.clientY - r.top) / r.height) * SIZE;
    return [Math.min(1, Math.max(0, (x - PAD) / (SIZE - 2 * PAD))), Math.min(1, Math.max(0, (SIZE - PAD - y) / (SIZE - 2 * PAD)))];
  };
  const set = (next: Pt[], commit = false) => onChange({ ...tone, [ch]: next }, commit);

  const nearest = (p: Pt): number | null => {
    let best: number | null = null, bd = 0.07;     // grab radius, in curve units
    pts.forEach((q, i) => { const d = Math.hypot(q[0] - p[0], q[1] - p[1]); if (d < bd) { bd = d; best = i; } });
    return best;
  };

  const down = (e: React.PointerEvent) => {
    if (e.button === 2) return;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    const p = fromEvent(e);
    let i = nearest(p);
    if (i === null) {
      if (pts.length >= MAX_POINTS) return;
      const next = [...pts, p].sort((a, b) => a[0] - b[0]);
      i = next.findIndex((q) => q === p);
      set(next);
    }
    dragIdx.current = i;
  };
  const move = (e: React.PointerEvent) => {
    const p = fromEvent(e);
    const i = dragIdx.current;
    setReadout(`in ${Math.round(p[0] * 255)}  →  out ${Math.round((i !== null ? p[1] : sampleCurve(pts, 256)[Math.round(p[0] * 255)]) * 255)}`);
    if (i === null) { setHover(nearest(p)); return; }
    const next = pts.map((q) => [...q]);
    const lo = i > 0 ? next[i - 1][0] + 0.012 : 0, hi = i < next.length - 1 ? next[i + 1][0] - 0.012 : 1;
    const x = i === 0 ? 0 : i === next.length - 1 ? 1 : Math.min(hi, Math.max(lo, p[0]));      // the end points only slide up and down
    next[i] = [x, p[1]];
    set(next);
  };
  const up = () => { if (dragIdx.current !== null) { dragIdx.current = null; onChange(tone, true); } };
  const remove = (i: number | null) => {
    if (i === null || i === 0 || i === pts.length - 1) return;
    set(pts.filter((_, j) => j !== i), true);
  };

  const curve = sampleCurve(pts, 96);
  const path = curve.map((y, i) => `${i ? 'L' : 'M'}${toX(i / 95).toFixed(1)},${toY(y).toFixed(1)}`).join(' ');
  const bins = ch === 'rgb' ? hist?.l : hist?.[ch];
  const histPath = bins
    ? `M${toX(0)},${toY(0)} ` + bins.map((v, i) => `L${toX(i / (bins.length - 1)).toFixed(1)},${(toY(0) - v * (SIZE - 2 * PAD) * 0.9).toFixed(1)}`).join(' ') + ` L${toX(1)},${toY(0)} Z`
    : '';

  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-1.5">
        {CH.map((c) => (
          <button key={c.id} onClick={() => setCh(c.id)}
            className={`flex-1 rounded-lg border py-2 text-[13px] font-semibold transition-colors ${ch === c.id ? 'border-indigo-400 bg-indigo-950' : 'border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800/60'}`}
            style={{ color: c.color }}>
            {c.label}{!isIdentity(tone[c.id]) ? ' •' : ''}
          </button>
        ))}
      </div>

      <svg ref={svg} viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full aspect-square rounded-lg bg-zinc-900 touch-none select-none cursor-crosshair"
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerLeave={() => { setHover(null); setReadout(''); }}
        onContextMenu={(e) => { e.preventDefault(); remove(nearest(fromEvent(e as unknown as PointerEvent))); }}
        onDoubleClick={(e) => remove(nearest(fromEvent(e as unknown as PointerEvent)))}>
        {[0.25, 0.5, 0.75].map((g) => (
          <g key={g} stroke="#3f3f46" strokeWidth="0.6">
            <line x1={toX(g)} y1={toY(0)} x2={toX(g)} y2={toY(1)} /><line x1={toX(0)} y1={toY(g)} x2={toX(1)} y2={toY(g)} />
          </g>
        ))}
        <rect x={toX(0)} y={toY(1)} width={SIZE - 2 * PAD} height={SIZE - 2 * PAD} fill="none" stroke="#52525b" strokeWidth="0.8" />
        {histPath && <path d={histPath} fill={ch === 'rgb' ? '#71717a' : color} opacity="0.35" />}
        <line x1={toX(0)} y1={toY(0)} x2={toX(1)} y2={toY(1)} stroke="#52525b" strokeWidth="0.8" strokeDasharray="3 3" />
        <path d={path} fill="none" stroke={color} strokeWidth="2.2" strokeLinejoin="round" />
        {pts.map((q, i) => (
          <g key={i}>
            <circle cx={toX(q[0])} cy={toY(q[1])} r={hover === i ? 8.5 : 6.5} fill="#18181b" stroke={color} strokeWidth="2.4" />
            <circle cx={toX(q[0])} cy={toY(q[1])} r="2.2" fill={color} />
          </g>
        ))}
      </svg>

      <div className="flex items-center justify-between text-[12px] text-zinc-400">
        <span className="tabular-nums">{readout || 'click to add · drag to move · double-click to remove'}</span>
        <button onClick={() => onChange({ ...tone, [ch]: IDENTITY.map((p) => [...p]) }, true)}
          className="rounded-md px-2 py-1 text-indigo-300 hover:bg-zinc-800">Reset {ch === 'rgb' ? 'RGB' : ch.toUpperCase()}</button>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {CURVE_PRESETS.map((pr) => (
          <button key={pr.id} onClick={() => onChange({ ...tone, [ch]: pr.pts.map((p) => [...p]) }, true)}
            className="rounded-lg border border-zinc-700 px-2.5 py-1.5 text-[12px] text-zinc-300 hover:border-zinc-500 hover:bg-zinc-800/60">{pr.label}</button>
        ))}
      </div>
    </div>
  );
}
