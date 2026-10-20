'use client';

import { ReactNode, useEffect, useState } from 'react';
import PSlider from './PSlider';
import Slider from './Slider';
import Toggle from './Toggle';
import { GROUPS, groupActive, groupStrength } from '../lib/groups';
import CurveEditor from './CurveEditor';
import { BrushCfg } from './MaskOverlay';
import { API_URL,
  CURVE_SLIDERS, EditCtx, Histogram as HistData, MaskKind, MaskLayer, ToneCurve, newLayer, ZERO_ADJ, MIXER_COLORS, MIXER_SWATCH, PEOPLE_SLIDERS, PRESENCE, REGION_COLORS, REGION_LABELS,
  REGION_SLIDERS, TONE, WB_LABELS, diffCount, SliderDef,
} from '../lib/params';

export function Sub({ title, children, count, right, open: initial = true }: {
  title: string; children: ReactNode; count?: number; right?: ReactNode; open?: boolean;
}) {
  const [open, setOpen] = useState(initial);
  return (
    <section className="border-b border-zinc-800">
      <div className="flex items-center justify-between px-4 py-2">
        <button onClick={() => setOpen(!open)} className="flex items-center gap-2 py-1 text-[12px] font-semibold uppercase tracking-widest text-zinc-200 hover:text-white">
          <span>{open ? '▾' : '▸'}</span>{title}
          {!!count && <span title="Adjustments you changed from the automatic start" className="rounded bg-amber-500/20 px-1.5 text-[10px] normal-case tracking-normal text-amber-300">{count} edited</span>}
        </button>
        {right}
      </div>
      {open && <div className="px-4 pb-4 flex flex-col gap-3">{children}</div>}
    </section>
  );
}

const HUE_TRACK = 'linear-gradient(90deg, hsl(0 80% 55%), hsl(60 80% 55%), hsl(120 80% 50%), hsl(180 80% 50%), hsl(240 80% 60%), hsl(300 80% 55%), hsl(360 80% 55%))';

const chip = (on: boolean) =>
  `px-3 py-1.5 rounded-lg text-[13px] border transition-colors ${on ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-300 hover:border-zinc-500 hover:bg-zinc-800/60'}`;

// ------------------------------------------------------------------------------------------------ colour grading
export function ColourGradingPanel({ ctx, part = 'all' }: { ctx: EditCtx; part?: 'all' | 'light' | 'colour' }) {
  const { p, auto } = ctx;
  const [mixColor, setMixColor] = useState('green');
  const on = p.grading_enabled;
  return (
    <>
      {part !== 'colour' && <Sub title="White balance & tone" count={diffCount(p.basic, auto.basic) + (p.wb_mode !== auto.wb_mode ? 1 : 0)}>
        <div className="flex flex-wrap gap-1.5 mb-1">
          {ctx.analysis.natural_color.candidates.map((c) => (
            <button key={c} onClick={() => ctx.update((cur) => ({ ...cur, wb_mode: c }), true, true)} className={chip(p.wb_mode === c)}>{WB_LABELS[c] ?? c}</button>
          ))}
        </div>
        <PSlider ctx={ctx} path={['basic', 'temperature']} def={{ key: 'temperature', label: 'Temperature' }} tint="#f59e0b" />
        <PSlider ctx={ctx} path={['basic', 'tint']} def={{ key: 'tint', label: 'Tint' }} tint="#d946ef" />
        <div className="text-[10px] uppercase tracking-widest text-zinc-500 mt-2">Tone</div>
        {TONE.map((d) => <PSlider key={d.key} ctx={ctx} path={['basic', d.key]} def={d} />)}
        <div className="text-[10px] uppercase tracking-widest text-zinc-500 mt-2">Presence</div>
        {PRESENCE.map((d) => <PSlider key={d.key} ctx={ctx} path={['basic', d.key]} def={d} />)}
      </Sub>}

      {part !== 'light' && <>
      <div className="px-4 py-2 border-b border-zinc-800 flex items-center justify-between">
        <Toggle checked={on} onChange={(v) => ctx.set(['grading_enabled'], v, true)} label="HSL, tone curve & colour balance" title="Switch the three tools below on or off without losing their settings" />
      </div>
      <div className={on ? '' : 'opacity-40 pointer-events-none'}>
        <Sub title="Colour grading wheels" count={diffCount(p.grading, auto.grading)}>
          {([['shadows', 'Shadows'], ['mids', 'Midtones'], ['highs', 'Highlights'], ['glob', 'Global']] as const).map(([key, name]) => (
            <div key={key} className="flex flex-col gap-1.5 border-l pl-3 mb-1" style={{ borderColor: `hsl(${p.grading[key].hue} 70% 55%)` }}>
              <span className="text-[11px] font-semibold text-zinc-300">{name}</span>
              <Slider label="Hue" value={p.grading[key].hue} def={auto.grading[key].hue} min={0} max={360} fmt={(v) => `${Math.round(v)}°`}
                track={HUE_TRACK} onChange={(v) => ctx.set(['grading', key, 'hue'], v)} />
              <PSlider ctx={ctx} path={['grading', key, 'sat']} def={{ key: 'sat', label: 'Saturation', min: 0, max: 100 }} tint={`hsl(${p.grading[key].hue} 70% 55%)`} />
              <PSlider ctx={ctx} path={['grading', key, 'lum']} def={{ key: 'lum', label: 'Luminance' }} />
            </div>
          ))}
          <PSlider ctx={ctx} path={['grading', 'blending']} def={{ key: 'blending', label: 'Blending', min: 0, max: 100 }} />
          <PSlider ctx={ctx} path={['grading', 'balance']} def={{ key: 'balance', label: 'Balance' }} />
        </Sub>
        <Sub title="HSL (colour mixer)" count={diffCount(p.mixer, auto.mixer)} open={false}>
          <div className="flex gap-1.5 mb-1">
            {MIXER_COLORS.map((c) => (
              <button key={c} onClick={() => setMixColor(c)} title={c}
                className={`h-5 w-5 rounded-full border-2 ${mixColor === c ? 'border-white' : 'border-transparent'}`} style={{ background: MIXER_SWATCH[c] }} />
            ))}
          </div>
          {(['Hue', 'Saturation', 'Luminance'] as const).map((l, i) => (
            <Slider key={l} label={l} value={p.mixer[mixColor][i]} def={auto.mixer[mixColor][i]} tint={MIXER_SWATCH[mixColor]}
              onChange={(v) => ctx.update((cur) => ({ ...cur, mixer: { ...cur.mixer, [mixColor]: cur.mixer[mixColor].map((x, j) => (j === i ? v : x)) } }))} />
          ))}
        </Sub>
        <Sub title="Colour balance" count={diffCount([p.balance_shadows, p.balance_mids, p.balance_highs], [auto.balance_shadows, auto.balance_mids, auto.balance_highs])} open={false}>
          {([['balance_shadows', 'Shadows'], ['balance_mids', 'Midtones'], ['balance_highs', 'Highlights']] as const).map(([key, name]) => (
            <div key={key} className="flex flex-col gap-1.5 border-l border-zinc-700 pl-3 mb-1">
              <span className="text-[11px] font-semibold text-zinc-300">{name}</span>
              {['Cyan ↔ Red', 'Magenta ↔ Green', 'Yellow ↔ Blue'].map((axis, i) => (
                <Slider key={axis} label={axis} value={p[key][i]} def={auto[key][i]}
                  onChange={(v) => ctx.update((cur) => ({ ...cur, [key]: cur[key].map((x, j) => (j === i ? v : x)) }))} />
              ))}
            </div>
          ))}
        </Sub>
      </div>
      </>}
    </>
  );
}

// ------------------------------------------------------------------------------------------------ selective masking
function PaintSlider({ label, value, onChange, tint }: { label: string; value: number; onChange: (v: number) => void; tint?: string }) {
  return <Slider label={label} value={value} min={1} max={100} fmt={(v) => String(Math.round(v))} onChange={onChange} tint={tint} />;
}

export type MaskOverlayCtl = { on: boolean; setOn: (v: boolean) => void; selected: string; setSelected: (n: string) => void };

export function MaskingPanel({ ctx, overlay, maskUI }: { ctx: EditCtx; overlay: MaskOverlayCtl; maskUI: MaskUI }) {
  const { p, auto } = ctx;
  const names = ctx.regions.filter((r) => r in p.regions);
  const sel = names.includes(overlay.selected) ? overlay.selected : names[0];
  const r = sel ? p.regions[sel] : null;
  const rep = sel ? ctx.analysis.masks[sel] : undefined;
  const tint = sel ? REGION_COLORS[sel] : undefined;
  const cov = ctx.analysis.coverage;
  const misc = cov?.other ?? 0;
  const covered = Math.max(0, 100 - misc);
  const nStrokes = r?.strokes?.length ?? 0;
  const resetMask = () => sel && ctx.update((cur) => ({ ...cur, regions: { ...cur.regions, [sel]: { ...auto.regions[sel] } } }), true, true);
  const ctrl = (key: string, label: string): SliderDef => ({ key, label, min: 0, max: 100 });
  return (
    <>
      <CustomMasksPanel ctx={ctx} ui={maskUI} />
      <div className="px-4 pt-3 pb-1 text-[11px] uppercase tracking-widest text-zinc-500 border-b border-zinc-800/60">Automatic regions</div>
      <Sub title="Select a mask" count={diffCount(p.regions, auto.regions)}
        right={<span className="text-[10px] text-zinc-500">{names.length} found</span>}>
        {cov && (
          <div className="mb-1">
            <div className="flex h-2 overflow-hidden rounded-full bg-zinc-800" title="How much of the picture the automatic regions cover">
              <div style={{ width: `${covered}%`, background: '#6366f1' }} />
              <div style={{ width: `${misc}%`, background: REGION_COLORS.other }} />
            </div>
            <p className={`mt-1 text-[11px] ${misc > 15 ? 'text-amber-400' : 'text-zinc-400'}`}>
              Recognised {covered.toFixed(0)}% · Miscellaneous {misc.toFixed(0)}% — anything the scene analysis could not name lands in Miscellaneous, so every pixel is covered.
              {misc > 15 && ' That is a lot: use Paint to fix, or the Global tools, to cover it.'}
            </p>
          </div>
        )}
        {names.length === 0 && <p className="text-[11px] text-zinc-500">No regions were detected in this photo.</p>}
        <div className="flex flex-wrap gap-1.5">
          {names.map((n) => (
            <button key={n} onClick={() => overlay.setSelected(n)} className={`flex items-center gap-1.5 ${chip(sel === n)} ${p.regions[n].enabled ? '' : 'opacity-50 line-through'}`}>
              <span className="h-2 w-2 rounded-full" style={{ background: REGION_COLORS[n] }} />{REGION_LABELS[n] ?? n}
            </button>
          ))}
        </div>
      </Sub>

      {sel && r && (
        <>
          <Sub title={`${REGION_LABELS[sel] ?? sel} mask`} count={diffCount(r, auto.regions[sel])}
            right={<button onClick={resetMask} className="text-[10px] text-indigo-400 hover:text-indigo-300">Reset mask</button>}>
            <div className="flex items-center justify-between">
              <Toggle checked={r.enabled} onChange={(v) => ctx.set(['regions', sel, 'enabled'], v, true)} label="Adjustment on" />
              <Toggle checked={overlay.on} onChange={overlay.setOn} label="Show mask" small />
            </div>
            {rep && (
              <p className={`text-[11px] ${rep.passed ? 'text-emerald-400' : 'text-amber-400'}`}>
                {rep.passed ? '✓' : '⚠'} mask accuracy {Math.round(rep.score * 100)}%{rep.chosen !== 'base' ? ` · refined (${rep.chosen})` : ''}
                {!rep.passed && ' · edited more gently here'}
              </p>
            )}
            <div className="rounded-lg border border-zinc-800 p-2.5 flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <Toggle checked={maskUI.fixing} onChange={(v) => { maskUI.setFixing(v); if (v) { maskUI.setSelected(null); overlay.setOn(true); } }} label="Paint to fix this mask" small />
                {nStrokes > 0 && <span className="text-[10px] text-zinc-500">{nStrokes} stroke{nStrokes > 1 ? 's' : ''}</span>}
              </div>
              {maskUI.fixing && (
                <>
                  <div className="flex gap-1.5">
                    <button onClick={() => maskUI.setBrush({ ...maskUI.brush, erase: false })} className={`flex-1 ${chip(!maskUI.brush.erase)}`}>＋ Add to {REGION_LABELS[sel] ?? sel}</button>
                    <button onClick={() => maskUI.setBrush({ ...maskUI.brush, erase: true })} className={`flex-1 ${chip(maskUI.brush.erase)}`}>− Remove</button>
                  </div>
                  <PaintSlider label="Brush size" value={maskUI.brush.size} onChange={(v) => maskUI.setBrush({ ...maskUI.brush, size: v })} tint={tint} />
                  <PaintSlider label="Feather" value={maskUI.brush.feather} onChange={(v) => maskUI.setBrush({ ...maskUI.brush, feather: v })} tint={tint} />
                  <div className="flex gap-1.5">
                    <button disabled={!nStrokes} onClick={() => ctx.update((cur) => ({ ...cur, regions: { ...cur.regions, [sel]: { ...cur.regions[sel], strokes: (cur.regions[sel].strokes ?? []).slice(0, -1) } } }), true, true)}
                      className={`flex-1 ${chip(false)} disabled:opacity-30`}>Undo stroke</button>
                    <button disabled={!nStrokes} onClick={() => ctx.update((cur) => ({ ...cur, regions: { ...cur.regions, [sel]: { ...cur.regions[sel], strokes: [] } } }), true, true)}
                      className={`flex-1 ${chip(false)} disabled:opacity-30`}>Clear</button>
                  </div>
                  <p className="text-[10px] text-zinc-500">Paint over anything the automatic mask missed (＋) or wrongly included (−). Strokes follow the picture&apos;s edges. The mask overlay shows the result.</p>
                </>
              )}
              {!maskUI.fixing && <p className="text-[10px] text-zinc-500">Mask missed something, or caught too much? Switch this on and paint the fix.</p>}
            </div>
            <div className={r.enabled ? 'flex flex-col gap-2' : 'flex flex-col gap-2 opacity-40 pointer-events-none'}>
              <div className="text-[10px] uppercase tracking-widest text-zinc-500 mt-1">Mask</div>
              <PSlider ctx={ctx} path={['regions', sel, 'mask_strength']} def={ctrl('mask_strength', 'Strength')} tint={tint} />
              <PSlider ctx={ctx} path={['regions', sel, 'feather']} def={ctrl('feather', 'Feathering')} tint={tint} />
              <PSlider ctx={ctx} path={['regions', sel, 'refine']} def={ctrl('refine', 'Edge refinement')} tint={tint} />
              <Toggle checked={r.invert} onChange={(v) => ctx.set(['regions', sel, 'invert'], v, true)} label="Invert (everything except this)" small />
              <div className="text-[10px] uppercase tracking-widest text-zinc-500 mt-2">Adjustments inside the mask</div>
              {REGION_SLIDERS.map((d) => <PSlider key={d.key} ctx={ctx} path={['regions', sel, d.key]} def={d} tint={tint} />)}
            </div>
          </Sub>
        </>
      )}

      {ctx.hasPerson && (
        <Sub title="People" count={diffCount(p.people, auto.people)}
          right={<Toggle checked={p.people.enabled} onChange={(v) => ctx.set(['people', 'enabled'], v, true)} small title="Switch people retouching on or off" />}>
          <div className={p.people.enabled ? 'flex flex-col gap-2' : 'flex flex-col gap-2 opacity-40 pointer-events-none'}>
            {PEOPLE_SLIDERS.map((d) => <PSlider key={d.key} ctx={ctx} path={['people', d.key]} def={d} tint="#f97316" />)}
          </div>
          <p className="text-[10px] text-zinc-500">A person mask is not a skin mask: skin tone and smoothing use the skin mask, separation uses the person.</p>
        </Sub>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------------ dodge & burn
export function DodgeBurnPanel({ ctx }: { ctx: EditCtx }) {
  const { p, auto, neutral } = ctx;
  const db = p.dodge_burn;
  const isAuto = diffCount(db, auto.dodge_burn) === 0;
  const notes = ctx.analysis.lighting ?? [];
  const noSubject = !ctx.hasPerson;
  const setAuto = (v: boolean) =>
    ctx.update((cur) => ({ ...cur, dodge_burn: v ? { ...auto.dodge_burn } : { ...neutral.dodge_burn, enabled: cur.dodge_burn.enabled } }), true, true);
  const d = (key: string, label: string, min = 0, max = 100): SliderDef => ({ key, label, min, max });
  const off = !db.enabled;
  return (
    <>
      <Sub title="Dodge & burn" count={diffCount(db, auto.dodge_burn)}
        right={<button onClick={() => ctx.update((cur) => ({ ...cur, dodge_burn: { ...auto.dodge_burn } }), true, true)} className="text-[10px] text-indigo-400 hover:text-indigo-300">Reset</button>}>
        <div className="flex items-center justify-between">
          <Toggle checked={db.enabled} onChange={(v) => ctx.set(['dodge_burn', 'enabled'], v, true)} label="Enabled" />
          <Toggle checked={isAuto} onChange={setAuto} label="Automatic lighting" small title="On: the editor's own suggestion. Off: start from zero and do it by hand." />
        </div>
        <div className="rounded bg-zinc-800/60 p-2 text-[11px] text-zinc-400 leading-relaxed">
          {notes.length ? notes.map((n, i) => <p key={i}>• {n}</p>) : <p>• No lighting suggestion for this photo.</p>}
          {noSubject && <p className="text-amber-300/80">• No reliable subject found, so only gentle global corrections are suggested.</p>}
        </div>
        <div className={off ? 'flex flex-col gap-2 opacity-40 pointer-events-none' : 'flex flex-col gap-2'}>
          <div className="text-[10px] uppercase tracking-widest text-zinc-500 mt-1">Dodge (brighten)</div>
          <PSlider ctx={ctx} path={['dodge_burn', 'dodge']} def={d('dodge', 'Strength')} tint="#fbbf24" />
          <PSlider ctx={ctx} path={['dodge_burn', 'dodge_tones']} def={{ ...d('dodge_tones', 'Tones: shadows → highlights'), min: 0, max: 100 }} tint="#fbbf24" />
          <div className="text-[10px] uppercase tracking-widest text-zinc-500 mt-2">Burn (darken)</div>
          <PSlider ctx={ctx} path={['dodge_burn', 'burn']} def={d('burn', 'Strength')} tint="#6366f1" />
          <PSlider ctx={ctx} path={['dodge_burn', 'burn_tones']} def={d('burn_tones', 'Tones: shadows → highlights')} tint="#6366f1" />
          <div className="text-[10px] uppercase tracking-widest text-zinc-500 mt-2">Subject</div>
          <PSlider ctx={ctx} path={['dodge_burn', 'separation']} def={d('separation', 'Background separation')} tint="#f97316" disabled={noSubject} />
          <PSlider ctx={ctx} path={['dodge_burn', 'form']} def={d('form', 'Face & skin form')} tint="#f97316" disabled={noSubject} />
          <Toggle checked={db.protect_face} onChange={(v) => ctx.set(['dodge_burn', 'protect_face'], v, true)} label="Protect face & skin (never burned, bright skin held back)" small />
          <div className="text-[10px] uppercase tracking-widest text-zinc-500 mt-2">Result</div>
          <PSlider ctx={ctx} path={['dodge_burn', 'feather']} def={d('feather', 'Softness of transitions')} />
          <PSlider ctx={ctx} path={['dodge_burn', 'strength']} def={d('strength', 'Overall strength', 0, 150)} />
        </div>
        <p className="text-[10px] text-zinc-500 leading-relaxed">
          Smooth exposure maps applied in linear light: skin texture is untouched, highlights are protected from clipping and
          shadows from crushing. It strengthens the light that is already there; it never invents lighting.
        </p>
      </Sub>
    </>
  );
}

// ------------------------------------------------------------------------------------------------ effects, detail, noise
export function EffectsPanel({ ctx, sid }: { ctx: EditCtx; sid?: string }) {
  const { p, auto } = ctx;
  const [dust, setDust] = useState<number | null>(null);
  useEffect(() => {
    if (!sid) return;
    let live = true;
    fetch(`${API_URL}/api/dust/${sid}`).then((r) => r.json()).then((d) => { if (live && typeof d.count === 'number') setDust(d.count); }).catch(() => {});
    return () => { live = false; };
  }, [sid]);
  const e = (key: string, label: string, min = 0, max = 100): SliderDef => ({ key, label, min, max });
  const needs = !ctx.hasPerson;
  return (
    <>
      <Sub title={`Background blur (bokeh)${needs ? ' — needs a person' : ''}`} count={diffCount(p.effects, auto.effects)}>
        <PSlider ctx={ctx} path={['effects', 'bokeh']} def={e('bokeh', 'Amount')} tint="#f97316" disabled={needs} />
        <PSlider ctx={ctx} path={['effects', 'bokeh_size']} def={e('bokeh_size', 'Disc size')} tint="#f97316" disabled={needs} />
        <PSlider ctx={ctx} path={['effects', 'bokeh_highlights']} def={e('bokeh_highlights', 'Light discs')} tint="#f97316" disabled={needs} />
        <PSlider ctx={ctx} path={['effects', 'bokeh_focus']} def={e('bokeh_focus', 'Focus range')} tint="#f97316" disabled={needs} />
      </Sub>
      <Sub title="Finish">
        <PSlider ctx={ctx} path={['effects', 'glow']} def={e('glow', 'Glow')} />
        <PSlider ctx={ctx} path={['effects', 'vignette']} def={e('vignette', 'Vignette', -100, 100)} />
        <PSlider ctx={ctx} path={['effects', 'vignette_mid']} def={e('vignette_mid', 'Vignette midpoint')} />
        <PSlider ctx={ctx} path={['effects', 'grain']} def={e('grain', 'Grain')} />
      </Sub>
      <Sub title="Sensor dust (sky)">
        <PSlider ctx={ctx} path={['effects', 'dust']} def={e('dust', 'Clean dust spots')} disabled={dust === 0} />
        <p className="text-[11px] leading-snug text-zinc-500">{dust === null ? 'Looking for dust spots in the sky…' : dust === 0 ? 'No dust spots found in the sky.' : `${dust} faint round spot${dust === 1 ? '' : 's'} found in the sky. Birds, planes and clouds are left alone.`}</p>
      </Sub>
      <Sub title="Detail & noise">
        <PSlider ctx={ctx} path={['effects', 'sharpen']} def={e('sharpen', 'Sharpening', 0, 150)} />
        <PSlider ctx={ctx} path={['effects', 'denoise']} def={e('denoise', 'Noise reduction')} />
        <PSlider ctx={ctx} path={['effects', 'denoise_color']} def={e('denoise_color', 'Colour noise')} />
      </Sub>
    </>
  );
}


// ------------------------------------------------------------------------------------------------ curves
export function CurvesPanel({ ctx, hist }: { ctx: EditCtx; hist?: HistData }) {
  const { p, auto } = ctx;
  return (
    <>
      <Sub title="Point curve" count={diffCount(p.tone, auto.tone)}
        right={<button onClick={() => ctx.update((cur) => ({ ...cur, tone: auto.tone }), true, true)} className="rounded-md px-2 py-1 text-[12px] text-indigo-300 hover:bg-zinc-800">Reset all</button>}>
        <CurveEditor tone={p.tone} hist={hist} onChange={(next: ToneCurve, commit) => ctx.update((cur) => ({ ...cur, tone: next }), !!commit, !!commit)} />
      </Sub>
      <Sub title="Parametric curve" count={diffCount(p.curve, auto.curve)} open={false}>
        {CURVE_SLIDERS.map((d) => <PSlider key={d.key} ctx={ctx} path={['curve', d.key]} def={d} />)}
        <p className="text-[12px] text-zinc-500">Each slider bends the curve inside its own tonal range; tones never swap order.</p>
      </Sub>
    </>
  );
}

// ------------------------------------------------------------------------------------------------ custom masks
export type MaskUI = {
  selected: string | null; setSelected: (id: string | null) => void;
  editing: boolean; setEditing: (v: boolean) => void;
  picking: boolean; setPicking: (v: boolean) => void;
  showMask: boolean; setShowMask: (v: boolean) => void;
  brush: BrushCfg; setBrush: (b: BrushCfg) => void;
  fixing: boolean; setFixing: (v: boolean) => void;      // painting corrections onto the selected automatic region
};

const KIND_BUTTONS: { kind: MaskKind; label: string; icon: string; hint: string }[] = [
  { kind: 'linear', label: 'Gradient', icon: '◐', hint: 'Graduated filter: full effect at one line, fading to none at another' },
  { kind: 'radial', label: 'Radial', icon: '◎', hint: 'Elliptical mask: lighten a subject or darken the edges' },
  { kind: 'brush', label: 'Brush', icon: '✎', hint: 'Paint the effect exactly where you want it (dodge and burn by hand)' },
  { kind: 'luminosity', label: 'Luminosity', icon: '◑', hint: 'Select by brightness: only the highlights, or only the shadows' },
  { kind: 'color', label: 'Colour', icon: '◍', hint: 'Select every area that has the colour you click on' },
];

const ADJ_GROUPS: { title: string; items: { key: keyof typeof ZERO_ADJ; label: string; ev?: boolean }[] }[] = [
  { title: 'Light', items: [{ key: 'exposure', label: 'Exposure', ev: true }, { key: 'contrast', label: 'Contrast' }, { key: 'highlights', label: 'Highlights' }, { key: 'shadows', label: 'Shadows' }] },
  { title: 'Colour', items: [{ key: 'temperature', label: 'Temperature' }, { key: 'tint', label: 'Tint' }, { key: 'vibrance', label: 'Vibrance' }, { key: 'saturation', label: 'Saturation' }, { key: 'hue', label: 'Hue' }] },
  { title: 'Detail', items: [{ key: 'clarity', label: 'Clarity' }, { key: 'dehaze', label: 'Dehaze' }] },
];

export function CustomMasksPanel({ ctx, ui }: { ctx: EditCtx; ui: MaskUI }) {
  const { p } = ctx;
  const layers = p.masks ?? [];
  const sel = layers.find((l) => l.id === ui.selected) ?? null;
  const evFmt = (v: number) => `${v >= 0 ? '+' : ''}${((v / 100) * 3).toFixed(2)}`;
  const has = (n: string) => ctx.analysis.mask_names?.includes(n);

  const addLayer = (kind: MaskKind, extra: Partial<MaskLayer> = {}) => {
    const layer = newLayer(kind, layers.filter((l) => l.kind === kind).length + 1, extra);
    ctx.update((cur) => ({ ...cur, masks: [...cur.masks, layer] }), true, true);
    ui.setSelected(layer.id);
    ui.setEditing(true);
    ui.setShowMask(true);
    ui.setPicking(kind === 'color');
  };
  const patchLayer = (id: string, fn: (l: MaskLayer) => MaskLayer, immediate = false, commit = false) =>
    ctx.update((cur) => ({ ...cur, masks: cur.masks.map((l) => (l.id === id ? fn(l) : l)) }), immediate, commit);
  const remove = (id: string) => {
    ctx.update((cur) => ({ ...cur, masks: cur.masks.filter((l) => l.id !== id) }), true, true);
    if (ui.selected === id) ui.setSelected(null);
  };
  const duplicate = (l: MaskLayer) => {
    const c = { ...JSON.parse(JSON.stringify(l)), id: Math.random().toString(36).slice(2, 9), name: `${l.name} copy` } as MaskLayer;
    ctx.update((cur) => ({ ...cur, masks: [...cur.masks, c] }), true, true);
    ui.setSelected(c.id);
  };

  const recipes: { label: string; make: () => void }[] = [
    { label: 'Darken sky', make: () => addLayer('linear', { name: 'Darken sky', line: [0.5, 0.0, 0.5, 0.5], feather: 80, limit: has('sky') ? 'sky' : null, adj: { ...ZERO_ADJ, exposure: -30, contrast: 12, vibrance: 14 } }) },
    { label: 'Vignette', make: () => addLayer('radial', { name: 'Vignette', ellipse: [0.5, 0.5, 0.62, 0.62, 0], feather: 85, invert: true, adj: { ...ZERO_ADJ, exposure: -34 } }) },
    { label: 'Lift foreground', make: () => addLayer('linear', { name: 'Lift foreground', line: [0.5, 1.0, 0.5, 0.55], feather: 80, adj: { ...ZERO_ADJ, exposure: 10, shadows: 35, vibrance: 8 } }) },
    ...(ctx.hasPerson ? [{ label: 'Light the subject', make: () => addLayer('radial', { name: 'Light the subject', ellipse: [0.5, 0.45, 0.35, 0.42, 0], feather: 90, limit: 'person', adj: { ...ZERO_ADJ, exposure: 14, contrast: 6 } }) }] : []),
  ];

  const ctl = (key: string, label: string, value: number, min: number, max: number, set: (v: number) => void, tint?: string) => (
    <Slider key={key} label={label} value={value} min={min} max={max} def={undefined} tint={tint} onChange={set} />
  );

  return (
    <>
      <Sub title="Custom masks" count={layers.length}
        right={layers.length ? <span className="text-[12px] text-zinc-500">{layers.length} mask{layers.length > 1 ? 's' : ''}</span> : undefined}>
        <div className="grid grid-cols-5 gap-1.5">
          {KIND_BUTTONS.map((k) => (
            <button key={k.kind} title={k.hint} onClick={() => addLayer(k.kind)}
              className="flex flex-col items-center gap-0.5 rounded-lg border border-zinc-700 py-2 text-zinc-200 transition-colors hover:border-indigo-400 hover:bg-indigo-950/50 active:bg-indigo-900">
              <span className="text-xl leading-none">{k.icon}</span><span className="text-[11px]">{k.label}</span>
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] uppercase tracking-widest text-zinc-500">Quick start</span>
          {recipes.map((r) => (
            <button key={r.label} onClick={r.make} className="rounded-lg border border-zinc-700 px-2.5 py-1.5 text-[12px] text-zinc-300 hover:border-zinc-500 hover:bg-zinc-800/60">{r.label}</button>
          ))}
        </div>
        {layers.length === 0 && <p className="text-[12px] leading-relaxed text-zinc-500">No custom masks yet. Add a gradient to darken a sky, a brush to dodge and burn by hand, or pick a colour to change only that colour.</p>}
        <div className="flex flex-col gap-1">
          {layers.map((l) => (
            <div key={l.id} onClick={() => { ui.setSelected(l.id); ui.setEditing(true); }}
              className={`flex cursor-pointer items-center gap-2 rounded-lg border px-2 py-1.5 ${ui.selected === l.id ? 'border-indigo-400 bg-indigo-950/60' : 'border-zinc-800 hover:border-zinc-600'}`}>
              <span onClick={(e) => e.stopPropagation()}>
                <Toggle checked={l.enabled} small onChange={(v) => patchLayer(l.id, (x) => ({ ...x, enabled: v }), true, true)} />
              </span>
              <span className="text-lg leading-none text-zinc-300">{KIND_BUTTONS.find((k) => k.kind === l.kind)?.icon}</span>
              <input value={l.name} onClick={(e) => e.stopPropagation()} onChange={(e) => patchLayer(l.id, (x) => ({ ...x, name: e.target.value }))}
                className={`min-w-0 flex-1 bg-transparent text-[13px] outline-none ${l.enabled ? 'text-zinc-100' : 'text-zinc-500 line-through'}`} />
              <button title="Duplicate" onClick={(e) => { e.stopPropagation(); duplicate(l); }} className="rounded px-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-white">⧉</button>
              <button title="Delete" onClick={(e) => { e.stopPropagation(); remove(l.id); }} className="rounded px-1.5 text-zinc-400 hover:bg-red-900/60 hover:text-white">✕</button>
            </div>
          ))}
        </div>
      </Sub>

      {sel && (
        <Sub title={sel.name || 'Mask'} count={0}
          right={<button onClick={() => patchLayer(sel.id, (l) => ({ ...l, adj: { ...ZERO_ADJ } }), true, true)} className="rounded-md px-2 py-1 text-[12px] text-indigo-300 hover:bg-zinc-800">Reset adjustments</button>}>
          <div className="flex items-center justify-between gap-2">
            <Toggle checked={ui.editing} onChange={ui.setEditing} label="Edit on picture" small />
            <Toggle checked={ui.showMask} onChange={ui.setShowMask} label="Show mask" small />
          </div>

          <div className="text-[11px] uppercase tracking-widest text-zinc-500">Mask shape</div>
          {sel.kind === 'linear' && (
            <>
              {ctl('lf', 'Transition softness', sel.feather, 0, 100, (v) => patchLayer(sel.id, (l) => ({ ...l, feather: v })))}
              <button onClick={() => patchLayer(sel.id, (l) => ({ ...l, line: [l.line[2], l.line[3], l.line[0], l.line[1]] }), true, true)}
                className="rounded-lg border border-zinc-700 py-2 text-[13px] text-zinc-200 hover:border-zinc-500 hover:bg-zinc-800/60">⇅ Flip direction</button>
              <p className="text-[12px] text-zinc-500">Drag the two lines on the picture: full effect at the amber line, none at the white one.</p>
            </>
          )}
          {sel.kind === 'radial' && (
            <>
              {ctl('rf', 'Feather', sel.feather, 0, 100, (v) => patchLayer(sel.id, (l) => ({ ...l, feather: v })))}
              {ctl('rr', 'Rotation', sel.ellipse[4], -90, 90, (v) => patchLayer(sel.id, (l) => ({ ...l, ellipse: [l.ellipse[0], l.ellipse[1], l.ellipse[2], l.ellipse[3], v] })))}
              <p className="text-[12px] text-zinc-500">Drag the centre to move it, the side handles to resize. Use “Invert” to affect everything outside.</p>
            </>
          )}
          {sel.kind === 'brush' && (
            <>
              {ctl('bs', 'Brush size', ui.brush.size, 1, 100, (v) => ui.setBrush({ ...ui.brush, size: v }), '#f87171')}
              {ctl('bf', 'Feather', ui.brush.feather, 0, 100, (v) => ui.setBrush({ ...ui.brush, feather: v }), '#f87171')}
              {ctl('bl', 'Flow', ui.brush.flow, 1, 100, (v) => ui.setBrush({ ...ui.brush, flow: v }), '#f87171')}
              <div className="flex items-center justify-between">
                <Toggle checked={ui.brush.erase} onChange={(v) => ui.setBrush({ ...ui.brush, erase: v })} label="Erase (or hold Alt)" small />
                <span className="flex gap-1.5">
                  <button onClick={() => patchLayer(sel.id, (l) => ({ ...l, strokes: l.strokes.slice(0, -1) }), true, true)} disabled={!sel.strokes.length}
                    className="rounded-lg border border-zinc-700 px-2.5 py-1.5 text-[12px] text-zinc-200 hover:bg-zinc-800 disabled:opacity-30">Undo stroke</button>
                  <button onClick={() => patchLayer(sel.id, (l) => ({ ...l, strokes: [] }), true, true)} disabled={!sel.strokes.length}
                    className="rounded-lg border border-zinc-700 px-2.5 py-1.5 text-[12px] text-zinc-200 hover:bg-zinc-800 disabled:opacity-30">Clear</button>
                </span>
              </div>
              <p className="text-[12px] text-zinc-500">Paint on the picture. “Follow edges” below makes the brush stop at object edges.</p>
            </>
          )}
          {sel.kind === 'luminosity' && (
            <>
              {ctl('l0', 'From brightness', sel.lum[0], 0, 100, (v) => patchLayer(sel.id, (l) => ({ ...l, lum: [v, l.lum[1]] })), '#fbbf24')}
              {ctl('l1', 'To brightness', sel.lum[1], 0, 100, (v) => patchLayer(sel.id, (l) => ({ ...l, lum: [l.lum[0], v] })), '#fbbf24')}
              {ctl('ls', 'Softness', sel.lum_soft, 0, 100, (v) => patchLayer(sel.id, (l) => ({ ...l, lum_soft: v })))}
              {ctl('lf', 'Spatial feather', sel.feather, 0, 100, (v) => patchLayer(sel.id, (l) => ({ ...l, feather: v })))}
            </>
          )}
          {sel.kind === 'color' && (
            <>
              <button onClick={() => { ui.setPicking(true); ui.setEditing(true); }}
                className={`rounded-lg border py-2.5 text-[13px] transition-colors ${ui.picking ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-200 hover:border-zinc-500 hover:bg-zinc-800/60'}`}>
                {ui.picking ? '◍ Click a colour on the picture…' : '◍ Pick a different colour'}
              </button>
              {ctl('ct', 'Tolerance', sel.tolerance, 1, 100, (v) => patchLayer(sel.id, (l) => ({ ...l, tolerance: v })))}
              {ctl('cf', 'Spatial feather', sel.feather, 0, 100, (v) => patchLayer(sel.id, (l) => ({ ...l, feather: v })))}
            </>
          )}

          <div className="text-[11px] uppercase tracking-widest text-zinc-500 mt-1">Mask controls</div>
          {ctl('st', 'Strength', sel.strength, 0, 100, (v) => patchLayer(sel.id, (l) => ({ ...l, strength: v })))}
          {ctl('rf2', 'Follow edges (auto mask)', sel.refine, 0, 100, (v) => patchLayer(sel.id, (l) => ({ ...l, refine: v })))}
          <label className="flex flex-col gap-1 text-[12px] text-zinc-400">Only inside
            <select value={sel.limit ?? ''} onChange={(e) => patchLayer(sel.id, (l) => ({ ...l, limit: e.target.value || null }), true, true)}
              className="rounded-lg bg-zinc-900 px-2 py-2 text-[13px] text-zinc-100">
              <option value="">Everywhere</option>
              {(ctx.analysis.mask_names ?? []).filter((n) => n !== 'hair').map((n) => <option key={n} value={n}>{REGION_LABELS[n] ?? n}</option>)}
            </select>
          </label>
          <div className="flex flex-wrap gap-x-4 gap-y-2">
            {sel.limit && <Toggle checked={sel.limit_invert} onChange={(v) => patchLayer(sel.id, (l) => ({ ...l, limit_invert: v }), true, true)} label={`Everywhere except ${REGION_LABELS[sel.limit] ?? sel.limit}`} small />}
            <Toggle checked={sel.invert} onChange={(v) => patchLayer(sel.id, (l) => ({ ...l, invert: v }), true, true)} label="Invert mask" small />
          </div>

          {ADJ_GROUPS.map((g) => (
            <div key={g.title} className="flex flex-col gap-3">
              <div className="text-[11px] uppercase tracking-widest text-zinc-500 mt-1">{g.title}</div>
              {g.items.map((it) => (
                <Slider key={it.key} label={it.label} value={sel.adj[it.key]} def={0} fmt={it.ev ? evFmt : undefined}
                  onChange={(v) => patchLayer(sel.id, (l) => ({ ...l, adj: { ...l.adj, [it.key]: v } }))} />
              ))}
            </div>
          ))}
        </Sub>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------------ simple controls
export type SimpleCtl = {
  strengths: Record<string, number>;
  setStrength: (id: string, k: number) => void;
  setLightScene: (light?: string, scene?: string) => void;
  busy: boolean;
};

export function SimplePanel({ ctx, ctl }: { ctx: EditCtx; ctl: SimpleCtl }) {
  const { p, auto, neutral, analysis } = ctx;
  const gr = analysis.grade;
  const labels: Record<string, string> = {
    golden_hour: 'Golden hour', sunset: 'Sunset glow', blue_hour: 'Blue hour', sunny: 'Midday sun', overcast: 'Overcast', night: 'Night', indoor: 'Indoor',
    mountain_lake: 'Mountain lake', mountains: 'Mountains', coast: 'Coast / water', forest: 'Forest', countryside: 'Countryside', urban: 'Urban',
    snow: 'Snow', desert: 'Desert / sand', general: 'General',
  };
  return (
    <div className="px-4 py-3 border-b border-zinc-800 flex flex-col gap-3">
      {gr && (
        <div className="rounded-lg bg-zinc-800/60 p-2.5 flex flex-col gap-2">
          <div className="text-[10px] uppercase tracking-widest text-zinc-500">Detected scene</div>
          <div className="grid grid-cols-2 gap-2">
            <label className="text-[11px] text-zinc-400">Light
              <select value={gr.light} disabled={ctl.busy} onChange={(e) => ctl.setLightScene(e.target.value, undefined)}
                className="mt-1 w-full rounded-lg bg-zinc-900 px-2 py-2 text-[13px] text-zinc-100">
                {gr.lights.map((l) => <option key={l} value={l}>{labels[l] ?? l}</option>)}
              </select>
            </label>
            <label className="text-[11px] text-zinc-400">Scene
              <select value={gr.scene} disabled={ctl.busy} onChange={(e) => ctl.setLightScene(undefined, e.target.value)}
                className="mt-1 w-full rounded-lg bg-zinc-900 px-2 py-2 text-[13px] text-zinc-100">
                {gr.scenes.map((l) => <option key={l} value={l}>{labels[l] ?? l}</option>)}
              </select>
            </label>
          </div>
          <p className="text-[11px] leading-relaxed text-zinc-400">
            {gr.evidence.slice(0, 3).join(' · ')}{gr.light_conf < 0.6 ? ' · unsure, change it if it looks wrong' : ''}
          </p>
          {gr.notes.length > 0 && <p className="text-[11px] leading-relaxed text-zinc-300">{gr.notes.slice(1, 4).join(' · ')}</p>}
        </div>
      )}
      <div className="text-[11px] uppercase tracking-widest text-zinc-400 -mb-1">Auto-edit strength</div>
      {GROUPS.filter((g) => groupActive(auto, neutral, g) && (ctx.hasPerson || !['lighting', 'separation', 'people'].includes(g.id))).map((g) => {
        const k = ctl.strengths[g.id];                 // stored as a fraction (1 = the automatic edit)
        const live = groupStrength(p, auto, neutral, g);   // already a percentage
        const shown = k !== undefined ? k * 100 : (live ?? 100);
        const touched = k === undefined && live !== null && Math.abs(live - 100) > 2;
        return (
          <div key={g.id}>
            <Slider label={g.label} value={shown} def={100} min={0} max={150} fmt={(v) => `${Math.round(v)}%`} hint={g.hint}
              onChange={(v) => ctl.setStrength(g.id, v / 100)} />
            {touched && <p className="text-[10px] text-amber-300/80">adjusted by hand in the advanced controls</p>}
          </div>
        );
      })}
      <p className="text-[10px] text-zinc-600 leading-relaxed">0% = untouched · 100% = the automatic edit · up to 150% = intensified. Every value is also in the advanced controls below.</p>
    </div>
  );
}
