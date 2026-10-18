/** Clipping overlay: red where the picture has lost its highlight detail, blue where it has lost its shadow detail.
 *  Computed in the browser from the preview that is on screen, so it is instant and always matches what you see. */
export async function makeClipOverlay(url: string): Promise<string | undefined> {
  try {
    const blob = await (await fetch(url)).blob();
    const bmp = await createImageBitmap(blob);
    const w = Math.min(900, bmp.width), h = Math.round((bmp.height * w) / bmp.width);
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    if (!ctx) return undefined;
    ctx.drawImage(bmp, 0, 0, w, h);
    const img = ctx.getImageData(0, 0, w, h), d = img.data;
    const out = ctx.createImageData(w, h), o = out.data;
    for (let i = 0; i < d.length; i += 4) {
      const mx = Math.max(d[i], d[i + 1], d[i + 2]), mn = Math.min(d[i], d[i + 1], d[i + 2]);
      if (mx >= 253) { o[i] = 255; o[i + 1] = 30; o[i + 2] = 30; o[i + 3] = 215; }
      else if (mx <= 3 && mn <= 3) { o[i] = 30; o[i + 1] = 110; o[i + 2] = 255; o[i + 3] = 215; }
    }
    ctx.putImageData(out, 0, 0);
    return c.toDataURL('image/png');
  } catch {
    return undefined;
  }
}
