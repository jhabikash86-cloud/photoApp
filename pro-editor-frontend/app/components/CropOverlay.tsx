'use client';

import { useEffect, useRef } from 'react';

export type CropRect = [number, number, number, number];      // x0, y0, x1, y1 as fractions of the picture
type Props = { geo: { w: number; h: number }; rect: CropRect; onRect: (r: CropRect) => void; onApply: () => void; onCancel: () => void };

const MIN = 0.03;
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
type Hit = 'move' | 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w' | 'new' | null;

/** The free-crop box drawn on the photo: drag a corner or an edge to crop from that side, drag inside to move the box, or drag on the dim area to draw a new box.
 *  Enter applies, Escape cancels. Everything is in picture fractions, so it lines up at any zoom and the export crops exactly the same area. */
export default function CropOverlay({ geo, rect, onRect, onApply, onCancel }: Props) {
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<{ hit: Hit; start: [number, number]; r0: CropRect } | null>(null);
  const { w, h } = geo;
  const [x0, y0, x1, y1] = rect;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Enter') { e.preventDefault(); onApply(); }
      else if (e.key === 'Escape') { e.preventDefault(); onCancel(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onApply, onCancel]);

  const norm = (e: React.PointerEvent): [number, number] => {
    const r = svg.current!.getBoundingClientRect();
    return [clamp((e.clientX - r.left) / r.width, 0, 1), clamp((e.clientY - r.top) / r.height, 0, 1)];
  };
  const hitTest = (p: [number, number]): Hit => {
    const tx = 16 / w, ty = 16 / h;                                   // grab zone: 16 screen pixels
    const near = (a: number, b: number, t: number) => Math.abs(a - b) <= t;
    const inX = p[0] > x0 - tx && p[0] < x1 + tx, inY = p[1] > y0 - ty && p[1] < y1 + ty;
    const l = near(p[0], x0, tx), r = near(p[0], x1, tx), t = near(p[1], y0, ty), b = near(p[1], y1, ty);
    if (inY && inX) {
      if (t && l) return 'nw'; if (t && r) return 'ne'; if (b && l) return 'sw'; if (b && r) return 'se';
      if (t) return 'n'; if (b) return 's'; if (l) return 'w'; if (r) return 'e';
      if (p[0] > x0 && p[0] < x1 && p[1] > y0 && p[1] < y1) return 'move';
    }
    return 'new';
  };
  const onDown = (e: React.PointerEvent) => {
    svg.current!.setPointerCapture(e.pointerId);
    const p = norm(e);
    drag.current = { hit: hitTest(p), start: p, r0: rect };
    if (drag.current.hit === 'new') onRect([p[0], p[1], p[0], p[1]]);
  };
  const onMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const p = norm(e);
    let [a, b, c, f] = d.r0;
    const dx = p[0] - d.start[0], dy = p[1] - d.start[1];
    switch (d.hit) {
      case 'move': {
        const ww = c - a, hh = f - b;
        a = clamp(a + dx, 0, 1 - ww); b = clamp(b + dy, 0, 1 - hh); c = a + ww; f = b + hh; break;
      }
      case 'new': a = d.start[0]; b = d.start[1]; c = p[0]; f = p[1]; break;
      default: {
        const hit = d.hit as string;
        if (hit.includes('w')) a = clamp(a + dx, 0, c - MIN);
        if (hit.includes('e')) c = clamp(c + dx, a + MIN, 1);
        if (hit.includes('n')) b = clamp(b + dy, 0, f - MIN);
        if (hit.includes('s')) f = clamp(f + dy, b + MIN, 1);
      }
    }
    onRect([Math.min(a, c), Math.min(b, f), Math.max(a, c), Math.max(b, f)]);
  };
  const onUp = () => {
    const d = drag.current;
    drag.current = null;
    if (d?.hit === 'new') {                                           // a stray click must not leave a zero-size box
      const [a, b, c, f] = rect;
      if (c - a < MIN || f - b < MIN) onRect(d.r0);
    }
  };

  const X0 = x0 * w, Y0 = y0 * h, X1 = x1 * w, Y1 = y1 * h;
  const handles: [number, number, string][] = [
    [X0, Y0, 'nwse-resize'], [(X0 + X1) / 2, Y0, 'ns-resize'], [X1, Y0, 'nesw-resize'], [X1, (Y0 + Y1) / 2, 'ew-resize'],
    [X1, Y1, 'nwse-resize'], [(X0 + X1) / 2, Y1, 'ns-resize'], [X0, Y1, 'nesw-resize'], [X0, (Y0 + Y1) / 2, 'ew-resize'],
  ];
  return (
    <svg ref={svg} width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="absolute inset-0 touch-none select-none overflow-hidden" style={{ cursor: 'crosshair' }}
      onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}>
      <path fillRule="evenodd" fill="rgba(0,0,0,0.55)" d={`M0 0H${w}V${h}H0Z M${X0} ${Y0}H${X1}V${Y1}H${X0}Z`} />
      <rect x={X0} y={Y0} width={Math.max(0, X1 - X0)} height={Math.max(0, Y1 - Y0)} fill="none" stroke="#fff" strokeWidth="1.5" />
      <g stroke="rgba(255,255,255,0.45)" strokeWidth="1">
        <line x1={X0 + (X1 - X0) / 3} y1={Y0} x2={X0 + (X1 - X0) / 3} y2={Y1} /><line x1={X0 + (2 * (X1 - X0)) / 3} y1={Y0} x2={X0 + (2 * (X1 - X0)) / 3} y2={Y1} />
        <line x1={X0} y1={Y0 + (Y1 - Y0) / 3} x2={X1} y2={Y0 + (Y1 - Y0) / 3} /><line x1={X0} y1={Y0 + (2 * (Y1 - Y0)) / 3} x2={X1} y2={Y0 + (2 * (Y1 - Y0)) / 3} />
      </g>
      {handles.map(([hx, hy, cur], i) => <rect key={i} x={hx - 6} y={hy - 6} width="12" height="12" rx="2" fill="#fff" stroke="#18181b" strokeWidth="1" style={{ cursor: cur }} />)}
    </svg>
  );
}
