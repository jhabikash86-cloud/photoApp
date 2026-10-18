'use client';

import Slider from './Slider';
import { EditCtx, Path, SliderDef, getIn } from '../lib/params';

type Props = { ctx: EditCtx; path: Path; def: SliderDef; tint?: string; disabled?: boolean };

/** A slider bound to one parameter: shows the Auto value as a tick and a reset arrow once you've changed it. */
export default function PSlider({ ctx, path, def, tint, disabled }: Props) {
  return (
    <Slider label={def.label} value={getIn(ctx.p, path)} def={getIn(ctx.auto, path)} min={def.min} max={def.max}
      fmt={def.fmt} tint={tint} disabled={disabled} onChange={(v) => ctx.set(path, v)} />
  );
}
