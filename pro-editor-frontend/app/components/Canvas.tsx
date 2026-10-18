'use client';

import { ReactNode, useCallback, useEffect, useRef, useState } from 'react';

export type CompareMode = 'after' | 'before' | 'split';

type Props = {
  beforeUrl?: string;
  afterUrl?: string;
  overlayUrl?: string;
  mode: CompareMode;
  busy?: boolean;
  empty?: ReactNode;
  onZoomChange?: (z: number) => void;
  afterFilter?: string;                      // instant CSS approximation while the real render is on its way
  onAfterShown?: (url: string | undefined) => void;
  clipUrl?: string;                          // highlight / shadow clipping overlay
  tool?: (geo: { w: number; h: number }) => ReactNode;   // an interactive editing overlay (mask handles, brush, eyedropper)
  toolActive?: boolean;                      // while true the overlay owns the mouse, so dragging does not pan the picture
};

// Loads a URL off-screen and only swaps when ready, so slider moves never flash a blank image.
function useStable(url?: string) {
  const [loaded, setLoaded] = useState<{ url: string; w: number; h: number } | null>(null);
  useEffect(() => {
    if (!url) return;
    let live = true;
    const img = new Image();
    img.onload = () => { if (live) setLoaded({ url, w: img.naturalWidth, h: img.naturalHeight }); };
    img.src = url;
    return () => { live = false; };
  }, [url]);
  return { shown: url ? loaded?.url : undefined, size: url && loaded ? { w: loaded.w, h: loaded.h } : null };
}

export default function Canvas({ beforeUrl, afterUrl, overlayUrl, mode, busy, empty, onZoomChange, afterFilter, onAfterShown, clipUrl, tool, toolActive }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [cs, setCs] = useState({ w: 800, h: 600 });
  const [z, setZ] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [split, setSplit] = useState(0.5);
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const after = useStable(afterUrl);
  const before = useStable(beforeUrl);
  const nat = after.size ?? before.size;

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setCs({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setCs({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  useEffect(() => { onZoomChange?.(z); }, [z, onZoomChange]);
  const afterShownUrl = after.shown;
  useEffect(() => { onAfterShown?.(afterShownUrl); }, [afterShownUrl, onAfterShown]);

  const fit = nat ? Math.min(cs.w / nat.w, cs.h / nat.h) : 1;
  const s = fit * z;
  const iw = nat ? nat.w * s : 0;
  const ih = nat ? nat.h * s : 0;
  const left = cs.w / 2 - iw / 2 + pan.x;
  const top = cs.h / 2 - ih / 2 + pan.y;

  const zoomAt = useCallback((mx: number, my: number, nz: number) => {
    if (!nat) return;
    const clamped = Math.min(8, Math.max(1, nz));
    const px = (mx - left) / s, py = (my - top) / s;
    const ns = fit * clamped;
    const nl = mx - px * ns, nt = my - py * ns;
    setZ(clamped);
    setPan(clamped === 1 ? { x: 0, y: 0 } : { x: nl - (cs.w / 2 - (nat.w * ns) / 2), y: nt - (cs.h / 2 - (nat.h * ns) / 2) });
  }, [nat, left, top, s, fit, cs]);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(e.clientX - r.left, e.clientY - r.top, z * Math.exp(-e.deltaY * 0.0018));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [zoomAt, z]);

  const down = (e: React.PointerEvent) => {
    if (z <= 1 || toolActive) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
  };
  const move = (e: React.PointerEvent) => {
    if (!drag.current) return;
    setPan({ x: drag.current.px + e.clientX - drag.current.x, y: drag.current.py + e.clientY - drag.current.y });
  };
  const up = () => { drag.current = null; };
  const dbl = (e: React.MouseEvent) => {
    if (toolActive) return;
    const r = box.current!.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, z > 1 ? 1 : 2.5);
  };

  const dragSplit = (e: React.PointerEvent) => {
    e.stopPropagation();
    const el = e.currentTarget as HTMLElement;
    el.setPointerCapture(e.pointerId);
    const move2 = (ev: PointerEvent) => {
      const r = box.current!.getBoundingClientRect();
      setSplit(Math.min(1, Math.max(0, (ev.clientX - r.left - left) / iw)));
    };
    const up2 = () => { el.removeEventListener('pointermove', move2); el.removeEventListener('pointerup', up2); };
    el.addEventListener('pointermove', move2);
    el.addEventListener('pointerup', up2);
  };

  const hasImage = !!(after.shown || before.shown);
  return (
    <div
      ref={box}
      className={`relative flex-1 min-h-0 overflow-hidden bg-[#111113] ${toolActive ? 'cursor-crosshair' : z > 1 ? 'cursor-grab active:cursor-grabbing' : 'cursor-zoom-in'}`}
      onPointerDown={down} onPointerMove={move} onPointerUp={up} onDoubleClick={dbl}
    >
      {!hasImage && <div className="absolute inset-0 flex items-center justify-center">{empty}</div>}
      {hasImage && nat && (
        <div className="absolute" style={{ left, top, width: iw, height: ih }}>
          {(mode === 'before' || mode === 'split') && before.shown && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={before.shown} alt="Before" draggable={false} className="absolute inset-0 w-full h-full select-none" />
          )}
          {(mode === 'after' || mode === 'split') && after.shown && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={after.shown} alt="After" draggable={false} className="absolute inset-0 w-full h-full select-none"
              style={{ ...(mode === 'split' ? { clipPath: `inset(0 0 0 ${split * 100}%)` } : {}), filter: afterFilter }} />
          )}
          {overlayUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={overlayUrl} alt="Regions" draggable={false} className="absolute inset-0 w-full h-full select-none opacity-70 pointer-events-none" />
          )}
          {clipUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={clipUrl} alt="Clipping" draggable={false} className="absolute inset-0 w-full h-full select-none pointer-events-none" style={{ imageRendering: 'pixelated' }} />
          )}
          {tool && (
            <div className="absolute inset-0" style={{ pointerEvents: toolActive ? 'auto' : 'none' }}>{tool({ w: iw, h: ih })}</div>
          )}
          {mode === 'split' && (
            <div className="absolute top-0 bottom-0 w-0.5 bg-white/80" style={{ left: `${split * 100}%` }}>
              <div onPointerDown={dragSplit}
                className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 h-8 w-8 rounded-full bg-white/90 text-slate-800 flex items-center justify-center text-xs shadow cursor-ew-resize">
                ↔
              </div>
            </div>
          )}
        </div>
      )}
      {hasImage && (
        <span className="absolute left-3 top-3 text-[10px] uppercase tracking-widest text-white/70 bg-black/40 px-2 py-0.5 rounded pointer-events-none">
          {mode === 'before' ? 'Before' : mode === 'split' ? 'Before | After' : 'After'}
        </span>
      )}
      {busy && <span className="absolute right-3 top-3 text-[10px] text-indigo-200 bg-black/50 px-2 py-0.5 rounded animate-pulse pointer-events-none">updating…</span>}
      {hasImage && z > 1 && (
        <span className="absolute left-3 bottom-3 text-[10px] text-white/70 bg-black/40 px-2 py-0.5 rounded pointer-events-none">{Math.round(z * 100)}%</span>
      )}
    </div>
  );
}
