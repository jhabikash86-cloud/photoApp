'use client';

import { useEffect, useRef } from 'react';
import type { Histogram as H } from '../lib/params';

type Props = { data?: H; clipOn?: boolean; onToggleClip?: () => void };

export default function Histogram({ data, clipOn, onToggleClip }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const w = c.width, h = c.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#18181b';
    ctx.fillRect(0, 0, w, h);
    if (!data) return;
    ctx.globalCompositeOperation = 'screen';
    const draw = (bins: number[], color: string) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(0, h);
      bins.forEach((v, i) => ctx.lineTo((i / (bins.length - 1)) * w, h - v * (h - 4)));
      ctx.lineTo(w, h);
      ctx.fill();
    };
    draw(data.r, 'rgba(239,68,68,0.75)');
    draw(data.g, 'rgba(34,197,94,0.75)');
    draw(data.b, 'rgba(59,130,246,0.75)');
    ctx.globalCompositeOperation = 'source-over';
  }, [data]);

  const hi = (data?.clip_hi ?? 0), lo = (data?.clip_lo ?? 0);
  const tri = (on: boolean, color: string, flip: boolean) => ({
    width: 0, height: 0, borderLeft: '9px solid transparent', borderRight: '9px solid transparent',
    [flip ? 'borderTop' : 'borderBottom']: `13px solid ${on ? color : '#52525b'}`,
  } as React.CSSProperties);
  return (
    <div className="relative">
      <canvas ref={ref} width={380} height={84} className="w-full h-[84px] rounded-lg" />
      <button title={`Shadows clipped: ${(lo * 100).toFixed(1)}% of the picture (J shows where)`} onClick={onToggleClip}
        className="absolute left-2 top-2 p-1.5 rounded hover:bg-zinc-700/60"><span style={tri(lo > 0.001, '#3b82f6', false)} className="block" /></button>
      <button title={`Highlights clipped: ${(hi * 100).toFixed(1)}% of the picture (J shows where)`} onClick={onToggleClip}
        className="absolute right-2 top-2 p-1.5 rounded hover:bg-zinc-700/60"><span style={tri(hi > 0.001, '#ef4444', false)} className="block" /></button>
      {onToggleClip && (
        <button onClick={onToggleClip} title="Show clipped areas on the picture (J)"
          className={`absolute left-1/2 top-2 -translate-x-1/2 rounded-md px-2 py-0.5 text-[11px] ${clipOn ? 'bg-red-600 text-white' : 'bg-zinc-800/80 text-zinc-300 hover:bg-zinc-700'}`}>
          {clipOn ? 'Clipping on' : 'Clipping'}
        </button>
      )}
    </div>
  );
}
