'use client';

/* The edit callbacks handed to the panels read refs (the live item list), but they are only ever invoked from event
   handlers, never while rendering, so the React-compiler ref rule does not apply to this file. */
/* eslint-disable react-hooks/refs */

import { useCallback, useEffect, useRef, useState } from 'react';
import Canvas, { CompareMode } from './components/Canvas';
import BatchDialog from './components/BatchDialog';
import Filmstrip from './components/Filmstrip';
import Histogram from './components/Histogram';
import Slider from './components/Slider';
import Toggle from './components/Toggle';
import LooksPanel from './components/LooksPanel';
import RegionPresetsPanel from './components/RegionPresetsPanel';
import Rail, { RAIL, TabId } from './components/Rail';
import { ColourGradingPanel, CurvesPanel, DodgeBurnPanel, EffectsPanel, MaskUI, MaskingPanel, SimplePanel, Sub } from './components/panels';
import MaskOverlay, { BrushCfg, DEFAULT_BRUSH } from './components/MaskOverlay';
import { makeClipOverlay } from './lib/clip';
import { GROUPS, applyGroup } from './lib/groups';
import { instantFilter } from './lib/preview';
import {
  API_URL, Analysis, EditCtx, Histogram as HistData, Item, LOOKS, Params, diffCount, newLayer, setIn,
} from './lib/params';

const ACCEPT = '.jpg,.jpeg,.png,.webp,.heic,.heif,.tif,.tiff,.nef,.nrw,.dng,.cr2,.cr3,.crw,.arw,.sr2,.raf,.orf,.rw2,.pef,.srw,.3fr,.iiq';
const OK_EXT = new Set(ACCEPT.split(',').map((e) => e.slice(1)));
const uid = () => Math.random().toString(36).slice(2, 10);

type UploadResp = { session_id: string; quick_url: string; thumb_url: string; megapixels: number; raw: boolean };

function uploadFile(file: File, onProgress: (f: number) => void): Promise<UploadResp> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_URL}/api/upload`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let data: { detail?: string } = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data as unknown as UploadResp);
      else reject(new Error(data.detail || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error(`Can't reach the server at ${API_URL}. Is the Python backend running?`));
    const fd = new FormData();
    fd.append('file', file);
    xhr.send(fd);
  });
}

async function api<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(data.detail || `Server error (${res.status})`), { status: res.status });
  return data as T;
}

export default function Editor() {
  const [items, setItems] = useState<Item[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [mode, setMode] = useState<CompareMode>('after');
  const [busy, setBusy] = useState(false);
  const [showMasks, setShowMasks] = useState(false);
  const [overlay, setOverlay] = useState<{ sid: string; url: string } | null>(null);
  const [exporting, setExporting] = useState(false);
  const [message, setMessage] = useState('');
  const [selRegion, setSelRegion] = useState('sky');
  const [tab, setTab] = useState<TabId>('auto');
  const [selMask, setSelMask] = useState<string | null>(null);
  const [maskEditing, setMaskEditing] = useState(true);
  const [picking, setPicking] = useState(false);
  const [fixing, setFixing] = useState(false);          // painting corrections onto the selected automatic region
  const [showCustomMask, setShowCustomMask] = useState(true);
  const [brush, setBrush] = useState<BrushCfg>(DEFAULT_BRUSH);
  const [customOverlay, setCustomOverlay] = useState<{ key: string; url: string } | null>(null);
  const [clipOn, setClipOn] = useState(false);
  const [clip, setClip] = useState<{ src: string; url: string } | null>(null);
  const [afterShown, setAfterShown] = useState<string | undefined>();
  const renderMap = useRef(new Map<string, Params>());      // preview URL -> the parameters that produced it
  const [maskOverlayOn, setMaskOverlayOn] = useState(false);
  const [maskOverlay, setMaskOverlay] = useState<{ key: string; url: string } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [batchOpen, setBatchOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const itemsRef = useRef<Item[]>([]);
  const activeRef = useRef<string | null>(null);
  const processRef = useRef<(id: string, keep?: boolean) => void>(() => {});
  const exportVersion = useRef(0);
  useEffect(() => { itemsRef.current = items; activeRef.current = activeId; });
  const queue = useRef<Promise<void>>(Promise.resolve());
  const render = useRef<{
    settleTimer: ReturnType<typeof setTimeout> | null; settleCtl: AbortController | null; v: number; applied: number;
    draftBusy: boolean; latest: { id: string; params: Params; dirty: boolean } | null; pending: number;
  }>({ settleTimer: null, settleCtl: null, v: 0, applied: 0, draftBusy: false, latest: null, pending: 0 });
  const commitTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const edgeRef = useRef(1200);
  const DRAFT_EDGE = 576;

  const active = items.find((i) => i.id === activeId) ?? null;
  const patch = useCallback((id: string, p: Partial<Item> | ((i: Item) => Partial<Item>)) => {
    setItems((all) => all.map((i) => (i.id === id ? { ...i, ...(typeof p === 'function' ? p(i) : p) } : i)));
  }, []);
  const flash = useCallback((m: string) => { setMessage(m); setTimeout(() => setMessage(''), 4000); }, []);

  // ---------------------------------------------------------------- rendering
  // Two-stage preview. While a slider is moving, small DRAFT renders are requested back to back (as fast as the server
  // answers, never piling up), so the effect shows almost at once. When the slider rests, one full-quality render
  // replaces the draft. A newer change always cancels an older one.
  const post = useCallback(async (id: string, params: Params, draft: boolean, signal?: AbortSignal) => {
    const item = itemsRef.current.find((i) => i.id === id);
    if (!item?.sid) return null;
    const v = ++render.current.v;
    const res = await fetch(`${API_URL}/api/render`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal,
      body: JSON.stringify({ session_id: item.sid, params, version: v, edge: draft ? DRAFT_EDGE : edgeRef.current, draft }),
    });
    const data = await res.json().catch(() => ({}));
    if (res.status === 404) { flash('Session expired — reopening the photo…'); processRef.current(id, true); return null; }
    if (!res.ok) throw new Error(data.detail || `Server error (${res.status})`);
    if (v > render.current.applied) {
      render.current.applied = v;
      const url = `${API_URL}${data.preview_url}`;
      renderMap.current.set(url, params);
      if (renderMap.current.size > 40) renderMap.current.delete(renderMap.current.keys().next().value as string);
      patch(id, { previewUrl: url, histogram: data.histogram });
    }
    return data;
  }, [flash, patch]);

  const runDrafts = useCallback(async () => {
    const r = render.current;
    if (r.draftBusy) return;
    r.draftBusy = true;
    setBusy(true);
    try {
      while (r.latest?.dirty) {
        const { id, params } = r.latest;
        r.latest.dirty = false;
        await post(id, params, true).catch(() => null);
      }
    } finally {
      r.draftBusy = false;
      if (!r.settleTimer && !r.settleCtl) setBusy(false);
    }
  }, [post]);

  const scheduleRender = useCallback((id: string, params: Params, immediate = false) => {
    const r = render.current;
    r.latest = { id, params, dirty: true };
    if (r.settleTimer) clearTimeout(r.settleTimer);
    r.settleCtl?.abort();
    r.settleCtl = null;
    const settle = async () => {
      r.settleTimer = null;
      const ctl = new AbortController();
      r.settleCtl = ctl;
      setBusy(true);
      try {
        await post(id, params, false, ctl.signal);
      } catch (err) {
        if ((err as Error).name !== 'AbortError') flash(err instanceof Error ? err.message : 'Render failed');
      } finally {
        if (r.settleCtl === ctl) { r.settleCtl = null; if (!r.draftBusy) setBusy(false); }
      }
    };
    if (immediate) {
      r.latest.dirty = false;
      settle();                       // a look or reset: one full render straight away
    } else {
      runDrafts();                    // dragging: instant drafts...
      r.settleTimer = setTimeout(settle, 380);   // ...then the quality render once the slider rests
    }
  }, [flash, post, runDrafts]);

  // ---------------------------------------------------------------- loading photos
  const processItem = useCallback(async (id: string, keepParams = false) => {
    const get = () => itemsRef.current.find((i) => i.id === id);
    const item = get();
    if (!item) return;
    try {
      patch(id, { status: 'uploading', progress: 0, error: undefined });
      const up = await uploadFile(item.file, (f) => patch(id, { progress: f }));
      patch(id, { sid: up.session_id, quick: up.quick_url, thumb: up.thumb_url, megapixels: up.megapixels, raw: up.raw, status: 'analyzing' });
      const an = await api<{
        params: Params; neutral: Params; analysis: Analysis; regions: string[]; original_url: string; preview_url: string;
        histogram: HistData; lighting_notes: string[];
      }>('/api/analyze', { session_id: up.session_id, look: 'auto' });
      if (keepParams && get()?.params) {
        patch(id, { status: 'ready', analysis: an.analysis, regions: an.regions, auto: an.params, neutral: an.neutral, originalUrl: an.original_url });
        scheduleRender(id, get()!.params!, true);
      } else {
        patch(id, {
          status: 'ready', params: an.params, auto: an.params, neutral: an.neutral, autoOn: true, history: [an.params], hIndex: 0, analysis: an.analysis,
          regions: an.regions, originalUrl: an.original_url, previewUrl: (renderMap.current.set(`${API_URL}${an.preview_url}`, an.params), `${API_URL}${an.preview_url}`), histogram: an.histogram,
        });
        setSelRegion((r) => (an.regions.includes(r) ? r : an.regions[0] ?? 'sky'));
      }
    } catch (e) {
      patch(id, { status: 'error', error: e instanceof Error ? e.message : 'Failed' });
    }
  }, [patch, scheduleRender]);
  useEffect(() => { processRef.current = (id, keep) => { processItem(id, keep); }; }, [processItem]);

  const openFiles = useCallback((files: FileList | File[]) => {
    const list = Array.from(files).filter((f) => OK_EXT.has((f.name.split('.').pop() || '').toLowerCase()));
    if (!list.length) { flash('Those files are not supported photos.'); return; }
    const fresh: Item[] = list.map((f) => ({ id: uid(), file: f, name: f.name, status: 'queued', progress: 0, history: [], hIndex: 0, regions: [] }));
    setItems((all) => [...all, ...fresh]);
    setActiveId((cur) => cur ?? fresh[0].id);
    itemsRef.current = [...itemsRef.current, ...fresh];
    fresh.forEach((it) => { queue.current = queue.current.then(() => processItem(it.id)); });
  }, [flash, processItem]);

  const commitHistory = useCallback((id: string) => {
    setItems((all) => all.map((i) => {
      if (i.id !== id || !i.params) return i;
      if (JSON.stringify(i.history[i.hIndex]) === JSON.stringify(i.params)) return i;
      const h = [...i.history.slice(0, i.hIndex + 1), i.params].slice(-80);
      return { ...i, history: h, hIndex: h.length - 1 };
    }));
  }, []);

  const updateParams = useCallback((fn: (p: Params) => Params, immediate = false, commitNow = false, keepStrengths = false) => {
    const id = activeRef.current;
    const item = itemsRef.current.find((i) => i.id === id);
    if (!id || !item?.params) return;
    const next = fn(item.params);
    patch(id, keepStrengths ? { params: next } : { params: next, strengths: undefined });
    itemsRef.current = itemsRef.current.map((i) => (i.id === id ? { ...i, params: next, ...(keepStrengths ? {} : { strengths: undefined }) } : i));
    scheduleRender(id, next, immediate);
    if (commitTimer.current) clearTimeout(commitTimer.current);
    if (commitNow) commitHistory(id); else commitTimer.current = setTimeout(() => commitHistory(id), 600);
  }, [commitHistory, patch, scheduleRender]);

  const step = useCallback((dir: -1 | 1) => {
    const id = activeRef.current;
    const item = itemsRef.current.find((i) => i.id === id);
    if (!id || !item) return;
    if (commitTimer.current) { clearTimeout(commitTimer.current); commitHistory(id); }
    setTimeout(() => {
      const it = itemsRef.current.find((i) => i.id === id);
      if (!it) return;
      const idx = it.hIndex + dir;
      if (idx < 0 || idx >= it.history.length) return;
      patch(id, { hIndex: idx, params: it.history[idx] });
      itemsRef.current = itemsRef.current.map((i) => (i.id === id ? { ...i, hIndex: idx, params: it.history[idx] } : i));
      scheduleRender(id, it.history[idx], true);
    }, 0);
  }, [commitHistory, patch, scheduleRender]);

  const applyLook = async (look: string) => {
    if (!active?.sid || !active.params) return;
    try {
      const res = await api<{ params: Params; read: Analysis['grade'] | null; grade_notes: string[] }>('/api/auto', { session_id: active.sid, look });
      patch(active.id, (it) => ({ auto: res.params, analysis: it.analysis && res.read ? { ...it.analysis, grade: { ...it.analysis.grade!, ...res.read, notes: res.grade_notes } } : it.analysis }));
      updateParams(() => res.params, true, true);
    } catch (e) {
      const err = e as Error & { status?: number };
      flash(err.status === 404 ? 'Session expired — reopening the photo…' : err.message);
      if (err.status === 404) processRef.current(active.id, true);
    }
  };

  const setLightScene = async (light?: string, scene?: string) => {
    if (!active?.sid || !active.params) return;
    try {
      const res = await api<{ params: Params; read: Analysis['grade'] | null; grade_notes: string[] }>('/api/auto',
        { session_id: active.sid, look: active.params.look, light: light ?? active.analysis?.grade?.light, scene: scene ?? active.analysis?.grade?.scene });
      patch(active.id, (it) => ({ auto: res.params, analysis: it.analysis && res.read ? { ...it.analysis, grade: { ...it.analysis.grade!, ...res.read, notes: res.grade_notes } } : it.analysis }));
      updateParams(() => res.params, true, true);
    } catch (e) { flash(e instanceof Error ? e.message : 'Could not regrade'); }
  };

  const setStrength = (id: string, k: number) => {
    const g = GROUPS.find((x) => x.id === id);
    if (!g || !active?.auto || !active.neutral) return;
    const auto = active.auto, neutral = active.neutral;
    patch(active.id, (it) => ({ strengths: { ...(it.strengths ?? {}), [id]: k } }));
    itemsRef.current = itemsRef.current.map((i) => (i.id === active.id ? { ...i, strengths: { ...(i.strengths ?? {}), [id]: k } } : i));
    updateParams((cur) => applyGroup(cur, auto, neutral, g, k), false, false, true);
  };

  const resetAll = () => active?.auto && updateParams(() => active.auto!, true, true);

  const download = async () => {
    if (!active?.sid || !active.params) return;
    setExporting(true);
    try {
      const data = await api<{ url: string }>('/api/export', { session_id: active.sid, params: active.params, version: ++exportVersion.current });
      const blob = await (await fetch(`${API_URL}${data.url}`)).blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${active.name.replace(/\.[^.]+$/, '')}-edit.jpg`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      flash(e instanceof Error ? e.message : 'Export failed');
    } finally {
      setExporting(false);
    }
  };

  const remove = (id: string) => {
    setItems((all) => all.filter((i) => i.id !== id));
    setActiveId((cur) => (cur === id ? itemsRef.current.find((i) => i.id !== id)?.id ?? null : cur));
  };

  // masks overlay (fetched only while the toggle is on)
  const activeSid = active?.sid;
  const activeReady = active?.status === 'ready';
  useEffect(() => {
    if (!showMasks || !activeSid || !activeReady) return;
    let live = true;
    fetch(`${API_URL}/api/regions/${activeSid}`).then((r) => r.json())
      .then((d) => { if (live && d.url) setOverlay({ sid: activeSid, url: `${API_URL}${d.url}?t=${Date.now()}` }); })
      .catch(() => {});
    return () => { live = false; };
  }, [showMasks, activeSid, activeReady]);
  const overlayUrl = showMasks && overlay && overlay.sid === activeSid ? overlay.url : undefined;

  // sharper preview when zoomed in
  const onZoom = useCallback((z: number) => {
    const want = z > 1.3 ? 2000 : 1200;
    if (want !== edgeRef.current) {
      edgeRef.current = want;
      const it = itemsRef.current.find((i) => i.id === activeRef.current);
      if (it?.params && it.status === 'ready') scheduleRender(it.id, it.params, true);
    }
  }, [scheduleRender]);

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && (t.tagName === 'INPUT' && (t as HTMLInputElement).type !== 'range' || t.tagName === 'TEXTAREA')) return;
      const meta = e.metaKey || e.ctrlKey;
      if (meta && e.key.toLowerCase() === 'z') { e.preventDefault(); step(e.shiftKey ? 1 : -1); }
      else if (meta && e.key.toLowerCase() === 'y') { e.preventDefault(); step(1); }
      else if (e.key === '\\') setMode((m) => (m === 'before' ? 'after' : 'before'));
      else if (e.key.toLowerCase() === 'm') setShowMasks((s) => !s);
      else if (e.key.toLowerCase() === 'j') setClipOn((c) => !c);
      else if (e.key.toLowerCase() === 's' && !meta) setMode((m) => (m === 'split' ? 'after' : 'split'));
      else if (e.key.toLowerCase() === 'r' && !meta) resetAll();
      else if (e.key.toLowerCase() === 'e' && !meta) download();
      else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        const list = itemsRef.current; const idx = list.findIndex((i) => i.id === activeRef.current);
        const nxt = list[idx + (e.key === 'ArrowRight' ? 1 : -1)];
        if (nxt) setActiveId(nxt.id);
      } else if (e.key === 'o' && meta) { e.preventDefault(); fileInput.current?.click(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, active?.sid, active?.params]);

  // ---------------------------------------------------------------- UI helpers
  const p = active?.params;
  const auto = active?.auto;
  const ready = active?.status === 'ready' && !!p && !!auto;
  const hasPerson = !!active?.analysis?.masks.person;
  const ctx: EditCtx | null = ready && active?.neutral ? {
    p: p!, auto: auto!, neutral: active.neutral, analysis: active.analysis!, regions: active.regions, sid: active.sid!, hasPerson,
    set: (path, v, immediate) => updateParams((cur) => setIn(cur, path, v), immediate),
    update: updateParams, flash,
  } : null;
  const manual = ready ? diffCount(p, auto) : 0;
  const toggleAuto = (v: boolean) => {
    if (!active?.auto || !active.neutral || !active.params) return;
    updateParams(() => (v ? active.auto! : { ...active.neutral!, look: active.params!.look }), true, true);
    patch(active.id, { autoOn: v });
  };

  // overlay of the ONE selected mask, reflecting its feather / refine / invert / strength as the user moves them
  const selR = p?.regions[selRegion];
  const strokeSig = (selR?.strokes ?? []).map((st) => `${st.pts.length}${st.erase ? 'e' : 'a'}${st.size}`).join(',').replace(/\|/g, '');
  const maskKey = ready && maskOverlayOn && tab === 'masks' && selR
    ? `${active?.sid}|${selRegion}|${selR.mask_strength}|${selR.feather}|${selR.refine}|${selR.invert}|${strokeSig}` : '';
  useEffect(() => {
    if (!maskKey) return;
    const [sid, name] = maskKey.split('|');
    const [, , str, fe, re, inv] = maskKey.split('|');
    const strokes = selR?.strokes ?? [];
    let live = true;
    const t = setTimeout(() => {
      fetch(`${API_URL}/api/mask_overlay`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, name, mask_strength: +str, feather: +fe, refine: +re, invert: inv === 'true', strokes, version: Date.now() % 100000 }),
      }).then((r) => r.json()).then((d) => { if (live && d.url) setMaskOverlay({ key: maskKey, url: `${API_URL}${d.url}` }); }).catch(() => {});
    }, 120);
    return () => { live = false; clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [maskKey]);
  const maskOverlayUrl = maskKey && maskOverlay?.key === maskKey ? maskOverlay.url : undefined;
  const selLayer = ready && tab === 'masks' ? p!.masks.find((l) => l.id === selMask) ?? null : null;
  const maskUI: MaskUI = {
    selected: selMask, setSelected: (id) => { setSelMask(id); if (id) setFixing(false); }, editing: maskEditing, setEditing: setMaskEditing, picking, setPicking,
    showMask: showCustomMask, setShowMask: setShowCustomMask, brush, setBrush, fixing, setFixing,
  };
  const fixLayer = ready && tab === 'masks' && fixing && selR ? newLayer('brush', 0, { id: '__fix', strokes: selR.strokes ?? [] }) : null;
  const toolLayer = fixLayer ?? (selLayer && maskEditing ? selLayer : null);
  const toolActive = !!fixLayer || !!selLayer && selLayer.enabled && maskEditing && ['linear', 'radial', 'brush'].includes(selLayer.kind) || (!!selLayer && maskEditing && selLayer.kind === 'color' && picking);

  // overlay of the selected custom mask (geometry only: adjustments do not change what is selected)
  const geoKey = selLayer ? JSON.stringify({ ...selLayer, adj: 0, name: '', enabled: 0 }) : '';
  const cKey = selLayer && showCustomMask && active?.sid ? `${active.sid}|${geoKey}` : '';
  useEffect(() => {
    if (!cKey) return;
    let live = true;
    const layer = selLayer;
    const sid = active?.sid;
    const t = setTimeout(() => {
      fetch(`${API_URL}/api/custom_overlay`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, layer, version: Date.now() % 100000 }),
      }).then((r) => r.json()).then((d) => { if (live && d.url) setCustomOverlay({ key: cKey, url: `${API_URL}${d.url}` }); }).catch(() => {});
    }, 140);
    return () => { live = false; clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cKey]);
  const customOverlayUrl = cKey && customOverlay?.key === cKey ? customOverlay.url : undefined;

  // clipping overlay, computed from whatever preview is on screen
  const clipSrc = clipOn ? active?.previewUrl : undefined;
  useEffect(() => {
    if (!clipSrc) return;
    let live = true;
    makeClipOverlay(clipSrc).then((url) => { if (live && url) setClip({ src: clipSrc, url }); });
    return () => { live = false; };
  }, [clipSrc]);
  const clipUrl = clipOn && clip && clip.src === clipSrc ? clip.url : undefined;
  const afterUrl = active?.previewUrl ?? undefined;
  const beforeUrl = active?.originalUrl ? `${API_URL}${active.originalUrl}` : active?.quick ? `${API_URL}${active.quick}` : undefined;
  const shownAfter = ready ? afterUrl : active?.quick ? `${API_URL}${active.quick}` : undefined;

  return (
    <div className="h-screen flex flex-col bg-[#1c1c1f] text-zinc-200 text-sm overflow-hidden"
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setDragging(false); }}
      onDrop={(e) => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files.length) openFiles(e.dataTransfer.files); }}>
      <input ref={fileInput} type="file" multiple accept={ACCEPT} className="hidden"
        onChange={(e) => { if (e.target.files) openFiles(e.target.files); e.target.value = ''; }} />

      {/* top bar */}
      <header className="h-14 shrink-0 flex items-center gap-3 px-4 border-b border-zinc-800 bg-[#18181b]">
        <span className="font-black tracking-tight text-indigo-400">⚡ NEURAL EDIT PRO</span>
        <button onClick={() => fileInput.current?.click()} className="px-4 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-[13px]">Open photos…</button>
        <button onClick={() => setBatchOpen(true)} title="Edit a whole Immich album" className="px-4 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-[13px]">Batch (Immich)…</button>
        <span className="text-xs text-zinc-500 truncate max-w-[260px]">{active ? `${active.name}${active.megapixels ? ` · ${active.megapixels} MP` : ''}${active.raw ? ' · RAW' : ''}` : ''}</span>
        <div className="flex-1" />
        {ready && (
          <>
            <div className="flex rounded overflow-hidden border border-zinc-700 text-xs">
              {([['before', 'Before'], ['split', 'Split'], ['after', 'After']] as const).map(([m, l]) => (
                <button key={m} onClick={() => setMode(m)} className={`px-4 py-2 text-[13px] ${mode === m ? 'bg-indigo-600 text-white' : 'bg-zinc-800 hover:bg-zinc-700'}`}>{l}</button>
              ))}
            </div>
            <button onClick={() => step(-1)} disabled={!active || active.hIndex <= 0} title="Undo (Ctrl/⌘ Z)" className="px-3.5 py-2 rounded-lg text-base bg-zinc-800 hover:bg-zinc-700 disabled:opacity-30">↶</button>
            <button onClick={() => step(1)} disabled={!active || active.hIndex >= active.history.length - 1} title="Redo (Shift Ctrl/⌘ Z)" className="px-3.5 py-2 rounded-lg text-base bg-zinc-800 hover:bg-zinc-700 disabled:opacity-30">↷</button>
            <button onClick={download} disabled={exporting} title="Export full-resolution JPEG (E)" className="px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-[13px] font-semibold disabled:opacity-50">
              {exporting ? 'Rendering full size…' : 'Export'}
            </button>
          </>
        )}
      </header>

      {batchOpen && <BatchDialog onClose={() => setBatchOpen(false)} />}
      <div className="flex flex-1 min-h-0">
        {/* canvas + filmstrip */}
        <div className="flex-1 min-w-0 flex flex-col relative">
          <Canvas key={active?.id ?? 'none'} beforeUrl={beforeUrl} afterUrl={shownAfter} overlayUrl={customOverlayUrl ?? maskOverlayUrl ?? overlayUrl} clipUrl={clipUrl} toolActive={toolActive}
            tool={toolLayer ? (geo) => (
              <MaskOverlay layer={toolLayer} geo={geo} brush={brush} picking={picking} onPicked={() => setPicking(false)}
                onLayer={(fn, commit) => (toolLayer.id === '__fix'
                  ? updateParams((cur) => ({ ...cur, regions: { ...cur.regions, [selRegion]: { ...cur.regions[selRegion], strokes: fn({ ...toolLayer, strokes: cur.regions[selRegion].strokes ?? [] }).strokes } } }), !!commit, !!commit)
                  : updateParams((cur) => ({ ...cur, masks: cur.masks.map((l) => (l.id === toolLayer.id ? fn(l) : l)) }), !!commit, !!commit))} />
            ) : undefined}
            mode={ready ? mode : 'after'} busy={busy || (!!active && active.status !== 'ready' && active.status !== 'error')}
            onZoomChange={onZoom} onAfterShown={setAfterShown}
            afterFilter={ready && p ? instantFilter(p, afterShown ? renderMap.current.get(afterShown) : undefined) : undefined}
            empty={
              <div className="text-center text-zinc-500 space-y-3">
                <div className="text-5xl">📷</div>
                <p className="text-zinc-300">Drop photos here, or</p>
                <button onClick={() => fileInput.current?.click()} className="px-4 py-2 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-sm">Open photos…</button>
                <p className="text-xs">RAW (NEF, DNG, CR2, ARW…), HEIC, JPEG, PNG, TIFF · up to 400 MB each</p>
              </div>
            } />
          {active && active.status !== 'ready' && (
            <div className="absolute inset-x-0 bottom-24 flex justify-center pointer-events-none">
              <div className="bg-black/70 text-xs px-3 py-1.5 rounded-full text-zinc-200">
                {active.status === 'error' ? `⚠ ${active.error}` :
                  active.status === 'uploading' ? `Uploading ${Math.round(active.progress * 100)}%…` :
                  active.status === 'analyzing' ? 'Analysing scene, masks and colour…' : 'Waiting in queue…'}
              </div>
            </div>
          )}
          {message && <div className="absolute left-1/2 -translate-x-1/2 top-3 bg-amber-900/90 text-amber-100 text-xs px-3 py-1.5 rounded">{message}</div>}
          {dragging && <div className="absolute inset-0 bg-indigo-500/20 border-2 border-dashed border-indigo-400 flex items-center justify-center text-lg pointer-events-none">Drop to open</div>}
          <Filmstrip items={items} activeId={activeId} onSelect={setActiveId} onRemove={remove} onAdd={() => fileInput.current?.click()} />
        </div>

        {/* right: tool panel + icon rail */}
        <div className="flex shrink-0">
        <aside className="w-[372px] shrink-0 bg-[#1f1f23] flex flex-col min-h-0">
          {!ctx && <p className="p-5 text-xs text-zinc-500">Open a photo to start. Every slider is a real adjustment, set by Auto and ready to move.</p>}
          {ctx && p && auto && active && (
            <>
              <div className="p-3 border-b border-zinc-800 shrink-0"><Histogram data={active.histogram} clipOn={clipOn} onToggleClip={() => setClipOn((v) => !v)} /></div>
              <div className="flex items-center justify-between px-4 py-2.5 border-b border-zinc-800 shrink-0">
                <span className="text-[15px] font-semibold text-zinc-100">{RAIL.find((r) => r.id === tab)?.label === 'Regions' ? 'Region presets' : tab === 'dodge' ? 'Dodge & burn' : RAIL.find((r) => r.id === tab)?.label}</span>
                <span className="text-[10px] text-zinc-500" title="Adjustments you changed by hand from the automatic start">
                  {manual > 0 ? `${manual} manual change${manual > 1 ? 's' : ''}` : 'automatic start'}
                </span>
              </div>
              <div className="flex-1 min-h-0 overflow-y-auto">
              {tab === 'auto' && (<>
              <div className="px-4 py-3 border-b border-zinc-800 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <Toggle checked={active.autoOn !== false} onChange={toggleAuto} label="Auto Edit"
                    title="On: the editor's automatic starting edit. Off: neutral, with no adjustments." />
                  <button onClick={() => { resetAll(); patch(active.id, { autoOn: true }); }} className="text-[11px] text-indigo-400 hover:text-indigo-300" title="Reset all adjustments (R)">Reset all</button>
                </div>
                <Slider label="Edit strength" value={p.intensity * 100} def={100} min={0} max={150} fmt={(v) => `${Math.round(v)}%`}
                  onChange={(v) => updateParams((c) => ({ ...c, intensity: v / 100 }))} />
                <div className="flex flex-wrap gap-1.5">
                  {LOOKS.map((l) => (
                    <button key={l.id} title={l.hint} onClick={() => applyLook(l.id)}
                      className={`px-3.5 py-2 rounded-lg text-[13px] border transition-colors ${(l.id === 'auto' ? false : p.look === l.id) ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-200 hover:border-zinc-500 hover:bg-zinc-800/60'}`}>
                      {l.label}
                    </button>
                  ))}
                </div>
              </div>

              <SimplePanel ctx={ctx} ctl={{ strengths: active.strengths ?? {}, setStrength, setLightScene, busy }} />
              </>)}

              {tab === 'looks' && <LooksPanel key={active.sid} ctx={ctx} />}
              {tab === 'light' && <ColourGradingPanel ctx={ctx} part="light" />}
              {tab === 'colour' && <ColourGradingPanel ctx={ctx} part="colour" />}
              {tab === 'presets' && <RegionPresetsPanel ctx={ctx} onTune={(r) => { setSelRegion(r); setTab('masks'); }} />}
              {tab === 'masks' && <MaskingPanel ctx={ctx} overlay={{ on: maskOverlayOn, setOn: setMaskOverlayOn, selected: selRegion, setSelected: setSelRegion }} maskUI={maskUI} />}
              {tab === 'curves' && <CurvesPanel ctx={ctx} hist={active.histogram} />}
              {tab === 'dodge' && <DodgeBurnPanel ctx={ctx} />}
              {tab === 'effects' && <EffectsPanel ctx={ctx} />}
              {tab === 'info' && (
                <Sub title="Analysis">
                  <div className="text-[11px] space-y-1 text-zinc-400">
                    <p className="flex justify-between"><span>Scene</span><span className="text-zinc-200 capitalize">{active.analysis!.semantic?.indoor ? `${active.analysis!.scene} · indoor` : active.analysis!.scene}</span></p>
                    {active.analysis!.faces.map((f, i) => (
                      <p key={i} className="flex justify-between"><span>Face {i + 1} focus</span>
                        <span className={f.focus === 'sharp' ? 'text-emerald-400' : f.focus === 'soft' ? 'text-amber-400' : f.focus === 'blurry' ? 'text-red-400' : 'text-zinc-500'}>{f.focus}</span></p>
                    ))}
                    {Object.entries(active.analysis!.masks).map(([k, m]) => (
                      <p key={k} className="flex justify-between"><span className="capitalize">{k} mask</span>
                        <span className={m.passed ? 'text-emerald-400' : 'text-amber-400'}>{m.passed ? '✓' : '⚠'} {Math.round(m.score * 100)}%</span></p>
                    ))}
                    <p className="flex justify-between"><span>Natural colour confidence</span><span className="text-zinc-200">{Math.round(active.analysis!.natural_color.confidence * 100)}%{active.analysis!.natural_color.warm_mood ? ' · warm kept' : ''}</span></p>
                    <p className="flex justify-between"><span>Analysis time</span><span className="text-zinc-200">{active.analysis!.seconds}s</span></p>
                    <label className="flex items-center gap-2 pt-1 cursor-pointer text-zinc-300">
                      <input type="checkbox" checked={showMasks} onChange={(e) => setShowMasks(e.target.checked)} /> Show all detected regions (M)
                    </label>
                  </div>
                  <p className="pt-2 text-[10px] text-zinc-600 leading-relaxed">
                    Shortcuts: \ before/after · S split · M regions · R reset · ← → next photo · Ctrl/⌘ Z undo · E export · scroll to zoom, double-click for 250%.
                    Amber ↺ marks a slider you changed; click it (or double-click the label) to return to the Auto value.
                  </p>
                </Sub>
              )}
              </div>
            </>
          )}
        </aside>
        <Rail tab={tab} onTab={setTab} dots={{ masks: ready ? diffCount(p!.regions, auto!.regions) + diffCount(p!.masks, auto!.masks) : 0, curves: ready ? diffCount(p!.tone, auto!.tone) : 0 }} />
        </div>
      </div>
    </div>
  );
}
