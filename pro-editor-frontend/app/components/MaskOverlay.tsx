'use client';

import { useRef, useState } from 'react';
import { MaskLayer, Stroke } from '../lib/params';

export type BrushCfg = { size: number; feather: number; flow: number; erase: boolean };
export const DEFAULT_BRUSH: BrushCfg = { size: 22, feather: 60, flow: 100, erase: false };

type Props = {
  layer: MaskLayer;
  geo: { w: number; h: number };
  brush: BrushCfg;
  picking: boolean;
  onLayer: (fn: (l: MaskLayer) => MaskLayer, commit?: boolean) => void;
  onPicked: () => void;
};

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const HANDLE = 11;

/** The interactive layer drawn on top of the picture while a mask is being edited. Everything is in picture-relative
 *  coordinates, so it lines up at any zoom and any export size. */
export default function MaskOverlay({ layer, geo, brush, picking, onLayer, onPicked }: Props) {
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<string | null>(null);
  const [live, setLive] = useState<number[][] | null>(null);       // the stroke being painted right now
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  const { w, h } = geo;

  const norm = (e: React.PointerEvent | PointerEvent): [number, number] => {
    const r = svg.current!.getBoundingClientRect();
    return [clamp01((e.clientX - r.left) / r.width), clamp01((e.clientY - r.top) / r.height)];
  };
  const grab = (what: string, e: React.PointerEvent) => {
    e.stopPropagation();
    svg.current!.setPointerCapture(e.pointerId);
    drag.current = what;
  };

  const brushRadiusPx = (0.004 + (0.12 * brush.size) / 100) * Math.max(w, h);

  // ---------------------------------------------------------------- pointer routing
  const onDown = (e: React.PointerEvent) => {
    if (layer.kind === 'brush') {
      svg.current!.setPointerCapture(e.pointerId);
      drag.current = 'paint';
      setLive([norm(e)]);
    } else if (layer.kind === 'color' && picking) {
      const p = norm(e);
      onLayer((l) => ({ ...l, pick: p }), true);
      onPicked();
    }
  };
  const onMove = (e: React.PointerEvent) => {
    const p = norm(e);
    setCursor(p);
    const d = drag.current;
    if (!d) return;
    if (d === 'paint') {
      setLive((cur) => {
        if (!cur) return [p];
        const last = cur[cur.length - 1];
        return Math.hypot((p[0] - last[0]) * w, (p[1] - last[1]) * h) > 2.5 ? [...cur, p] : cur;
      });
    } else if (d === 'p1' || d === 'p2') {
      onLayer((l) => {
        const line = [...l.line];
        if (d === 'p1') { line[0] = p[0]; line[1] = p[1]; } else { line[2] = p[0]; line[3] = p[1]; }
        return { ...l, line };
      });
    } else if (d === 'mid') {
      onLayer((l) => {
        const cx = (l.line[0] + l.line[2]) / 2, cy = (l.line[1] + l.line[3]) / 2, dx = p[0] - cx, dy = p[1] - cy;
        return { ...l, line: [clamp01(l.line[0] + dx), clamp01(l.line[1] + dy), clamp01(l.line[2] + dx), clamp01(l.line[3] + dy)] };
      });
    } else if (d === 'c') {
      onLayer((l) => ({ ...l, ellipse: [p[0], p[1], l.ellipse[2], l.ellipse[3], l.ellipse[4]] }));
    } else if (d === 'rx' || d === 'ry') {
      onLayer((l) => {
        const [cx, cy, rx, ry, ang] = l.ellipse, a = (ang * Math.PI) / 180;
        const vx = p[0] * w - cx * w, vy = p[1] * h - cy * h;
        if (d === 'rx') return { ...l, ellipse: [cx, cy, Math.min(1.5, Math.max(0.02, (vx * Math.cos(a) + vy * Math.sin(a)) / w)), ry, ang] };
        return { ...l, ellipse: [cx, cy, rx, Math.min(1.5, Math.max(0.02, (-vx * Math.sin(a) + vy * Math.cos(a)) / h)), ang] };
      });
    }
  };
  const onUp = (e: React.PointerEvent) => {
    const d = drag.current;
    drag.current = null;
    try { svg.current?.releasePointerCapture(e.pointerId); } catch { /* not captured */ }
    if (d === 'paint' && live && live.length) {
      const stroke: Stroke = { pts: live.length === 1 ? [live[0], [live[0][0] + 0.0005, live[0][1]]] : live, size: brush.size, feather: brush.feather, flow: brush.flow, erase: brush.erase || e.altKey };
      onLayer((l) => ({ ...l, strokes: [...l.strokes, stroke] }), true);
      setLive(null);
    } else if (d) {
      onLayer((l) => l, true);
    }
  };

  // ---------------------------------------------------------------- drawing
  const handle = (x: number, y: number, what: string, fill = '#fbbf24', r = HANDLE) => (
    <g key={what} onPointerDown={(e) => grab(what, e)} style={{ cursor: 'grab' }}>
      <circle cx={x} cy={y} r={r + 9} fill="transparent" />
      <circle cx={x} cy={y} r={r} fill={fill} stroke="#18181b" strokeWidth="2.5" />
    </g>
  );

  let body: React.ReactNode = null;
  if (layer.kind === 'linear') {
    const [x1, y1, x2, y2] = layer.line, p1 = [x1 * w, y1 * h], p2 = [x2 * w, y2 * h];
    const dx = p2[0] - p1[0], dy = p2[1] - p1[1], len = Math.max(Math.hypot(dx, dy), 1);
    const nx = -dy / len, ny = dx / len, ext = Math.hypot(w, h);
    const bar = (p: number[], dash?: string, col = '#fbbf24') => (
      <line x1={p[0] - nx * ext} y1={p[1] - ny * ext} x2={p[0] + nx * ext} y2={p[1] + ny * ext} stroke={col} strokeWidth="2" strokeDasharray={dash} />
    );
    const mid = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
    body = (
      <>
        {bar(p1)}{bar(p2, '7 6', '#e4e4e7')}
        <line x1={p1[0]} y1={p1[1]} x2={p2[0]} y2={p2[1]} stroke="#fbbf24" strokeWidth="1.5" opacity="0.8" />
        <g onPointerDown={(e) => grab('mid', e)} style={{ cursor: 'move' }}>
          <line x1={mid[0] - nx * 60} y1={mid[1] - ny * 60} x2={mid[0] + nx * 60} y2={mid[1] + ny * 60} stroke="transparent" strokeWidth="26" />
          <rect x={mid[0] - 13} y={mid[1] - 13} width="26" height="26" rx="7" fill="#27272a" stroke="#fbbf24" strokeWidth="2" />
          <text x={mid[0]} y={mid[1] + 5} textAnchor="middle" fontSize="15" fill="#fbbf24">✥</text>
        </g>
        {handle(p1[0], p1[1], 'p1')}
        {handle(p2[0], p2[1], 'p2', '#e4e4e7')}
        <text x={p1[0] + 16} y={p1[1] - 12} fontSize="12" fill="#fbbf24" style={{ pointerEvents: 'none' }}>full effect</text>
        <text x={p2[0] + 16} y={p2[1] - 12} fontSize="12" fill="#e4e4e7" style={{ pointerEvents: 'none' }}>none</text>
      </>
    );
  } else if (layer.kind === 'radial') {
    const [cx, cy, rx, ry, ang] = layer.ellipse, a = (ang * Math.PI) / 180;
    const C = [cx * w, cy * h], ax = rx * w, ay = ry * h;
    const hx = [C[0] + ax * Math.cos(a), C[1] + ax * Math.sin(a)], hy = [C[0] - ay * Math.sin(a), C[1] + ay * Math.cos(a)];
    body = (
      <>
        <ellipse cx={C[0]} cy={C[1]} rx={ax} ry={ay} transform={`rotate(${ang} ${C[0]} ${C[1]})`} fill="none" stroke="#fbbf24" strokeWidth="2" />
        <ellipse cx={C[0]} cy={C[1]} rx={ax * 1.0} ry={ay * 1.0} transform={`rotate(${ang} ${C[0]} ${C[1]})`} fill="none" stroke="#e4e4e7" strokeWidth="1" strokeDasharray="6 6" opacity="0.6" />
        {handle(C[0], C[1], 'c')}
        {handle(hx[0], hx[1], 'rx', '#e4e4e7', 9)}
        {handle(hy[0], hy[1], 'ry', '#e4e4e7', 9)}
      </>
    );
  } else if (layer.kind === 'brush') {
    const r = brushRadiusPx;
    const sw = 2 * r * (1 - 0.45 * brush.feather / 100);
    body = (
      <>
        {live && live.length > 0 && (
          <polyline points={live.map((q) => `${q[0] * w},${q[1] * h}`).join(' ')} fill="none" stroke={brush.erase ? 'rgba(96,165,250,0.5)' : 'rgba(255,70,50,0.5)'}
            strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" />
        )}
        {cursor && (
          <g style={{ pointerEvents: 'none' }}>
            <circle cx={cursor[0] * w} cy={cursor[1] * h} r={r} fill="none" stroke="#fff" strokeWidth="1.5" opacity="0.9" />
            <circle cx={cursor[0] * w} cy={cursor[1] * h} r={r * (1 - 0.45 * brush.feather / 100)} fill="none" stroke="#fff" strokeWidth="1" strokeDasharray="4 4" opacity="0.7" />
          </g>
        )}
      </>
    );
  } else if (layer.kind === 'color') {
    const px = layer.pick[0] * w, py = layer.pick[1] * h;
    body = (
      <g style={{ pointerEvents: 'none' }}>
        <circle cx={px} cy={py} r="13" fill="none" stroke="#fff" strokeWidth="2.5" />
        <circle cx={px} cy={py} r="13" fill="none" stroke="#18181b" strokeWidth="1" />
        <line x1={px - 22} y1={py} x2={px - 8} y2={py} stroke="#fff" strokeWidth="2" /><line x1={px + 8} y1={py} x2={px + 22} y2={py} stroke="#fff" strokeWidth="2" />
        <line x1={px} y1={py - 22} x2={px} y2={py - 8} stroke="#fff" strokeWidth="2" /><line x1={px} y1={py + 8} x2={px} y2={py + 22} stroke="#fff" strokeWidth="2" />
      </g>
    );
  }

  return (
    <svg ref={svg} width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="absolute inset-0 touch-none select-none overflow-hidden"
      style={{ cursor: layer.kind === 'brush' ? 'none' : layer.kind === 'color' && picking ? 'crosshair' : 'default' }}
      onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={() => setCursor(null)}>
      {body}
    </svg>
  );
}
