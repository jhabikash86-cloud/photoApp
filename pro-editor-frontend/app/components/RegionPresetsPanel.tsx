'use client';

import { Sub } from './panels';
import { EditCtx, REGION_COLORS, REGION_LABELS } from '../lib/params';
import { PRESET_REGION_ORDER, REGION_PRESETS, applyRegionPreset, presetIsActive, regionIsNeutral } from '../lib/regionPresets';

/** Top-notch starting looks for every region found in this photo. One click sets that region's sliders; fine-tune in Masks. */
export default function RegionPresetsPanel({ ctx, onTune }: { ctx: EditCtx; onTune: (region: string) => void }) {
  const { p } = ctx;
  const found = PRESET_REGION_ORDER.filter((r) => ctx.regions.includes(r) && r in p.regions && REGION_PRESETS[r]);
  const cov = ctx.analysis.coverage ?? {};
  return (
    <>
      <div className="px-4 py-3 border-b border-zinc-800 text-[11px] leading-relaxed text-zinc-400">
        One-click looks for each part of this photo. Pick one, then fine-tune the sliders in the region&apos;s mask. Each preset only touches its own region, so sky presets never change the water.
      </div>
      {found.length === 0 && <p className="p-4 text-[12px] text-zinc-500">No regions were detected in this photo.</p>}
      {found.map((name) => {
        const list = REGION_PRESETS[name];
        const neutral = regionIsNeutral(p, name);
        const pct = cov[name];
        return (
          <Sub key={name} title={name === 'skin' ? 'Face & skin' : REGION_LABELS[name] ?? name}
            right={<span className="flex items-center gap-2 text-[10px] text-zinc-500">
              {pct !== undefined && <span>{pct.toFixed(0)}% of photo</span>}
              <span className="h-2 w-2 rounded-full" style={{ background: REGION_COLORS[name] }} />
            </span>}>
            <div className="grid grid-cols-2 gap-1.5">
              {list.map((pr) => {
                const on = presetIsActive(p, name, pr);
                return (
                  <button key={pr.id} title={pr.hint} onClick={() => ctx.update((c) => applyRegionPreset(c, name, pr), true, true)}
                    className={`rounded-lg border px-2.5 py-2.5 text-left text-[12.5px] leading-tight transition-colors ${on ? 'border-indigo-400 bg-indigo-950 text-indigo-100' : 'border-zinc-700 text-zinc-200 hover:border-zinc-500 hover:bg-zinc-800/60'}`}>
                    {pr.label}
                  </button>
                );
              })}
            </div>
            <div className="flex items-center justify-between pt-1">
              <button onClick={() => onTune(name)} className="text-[11px] text-indigo-300 hover:text-indigo-200">Fine-tune in Masks →</button>
              <button disabled={neutral} onClick={() => ctx.update((c) => ({ ...c, regions: { ...c.regions, [name]: { ...c.regions[name], exposure: 0, contrast: 0, highlights: 0, shadows: 0, vibrance: 0, saturation: 0, clarity: 0, dehaze: 0, temperature: 0, tint: 0, hue: 0 } } }), true, true)}
                className="text-[11px] text-zinc-400 hover:text-zinc-200 disabled:opacity-30">Clear region edits</button>
            </div>
            {name === 'skin' && !ctx.hasPerson && <p className="text-[10px] text-zinc-500">Face retouch values need a detected person.</p>}
          </Sub>
        );
      })}
    </>
  );
}
