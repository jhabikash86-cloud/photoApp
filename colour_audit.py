"""Colour audit: measures how truthful the colour and tone tools are. Run: ./venv/bin/python colour_audit.py"""
import sys, glob
sys.path.insert(0, '.')
import numpy as np, cv2
import grade, colorcal, adjust
from params import Basic, Grading, Wheel

L = np.array([.2126, .7152, .0722], np.float32)
def lab(a): return cv2.cvtColor(np.clip(a, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
def hue(a, b): return np.degrees(np.arctan2(b, a)) % 360
def dh(h1, h0): return (h1 - h0 + 180) % 360 - 180

def patches():
    """A colour chart: 12 hues at three saturations and three lightnesses, plus a grey ramp, as a picture."""
    out = []
    for l in (35, 55, 75):
        for c in (25, 50, 75):
            for h in range(0, 360, 30):
                a, b = c * np.cos(np.radians(h)), c * np.sin(np.radians(h))
                rgb = cv2.cvtColor(np.array([[[l, a, b]]], np.float32), cv2.COLOR_Lab2RGB)[0, 0]
                if (rgb > 0.999).any() or (rgb < 0.001).any(): continue
                out.append(rgb)
    return np.array(out, np.float32).reshape(-1, 1, 3)

P = patches()
Lab0 = lab(P)[:, 0]
H0, C0 = hue(Lab0[:, 1], Lab0[:, 2]), np.hypot(Lab0[:, 1], Lab0[:, 2])
print(f"chart: {len(P)} in-gamut colour patches\n")

def tone_ctx(a):
    return grade.Ctx(a, None)

print("=== A. Do the tone sliders change colour? (Lab hue shift in degrees, chroma change; lower hue shift is better)")
for name, fn in [
    ("exposure +40", lambda a: grade.exposure(a, 40)),
    ("exposure -40", lambda a: grade.exposure(a, -40)),
    ("contrast +50", lambda a: grade.contrast(a, 50)),
    ("contrast -50", lambda a: grade.contrast(a, -50)),
    ("highlights -60", lambda a: grade.highlights(a, -60, tone_ctx(a))),
    ("shadows +60", lambda a: grade.shadows(a, 60, tone_ctx(a))),
    ("whites +40", lambda a: grade.whites(a, 40, tone_ctx(a))),
    ("blacks -40", lambda a: grade.blacks(a, -40, tone_ctx(a))),
]:
    o = lab(fn(P.copy()))[:, 0]
    d = np.abs(dh(hue(o[:, 1], o[:, 2]), H0)); mask = C0 > 20
    cr = np.hypot(o[:, 1], o[:, 2]) / np.maximum(C0, 1e-3)
    print(f"  {name:16s} mean hue shift {d[mask].mean():5.2f}°  worst {d[mask].max():5.1f}°   chroma x{np.median(cr[mask]):.2f}   L {np.median(o[:,0]-Lab0[:,0]):+5.1f}")

print("\n=== B. Saturation / vibrance: hue drift and lightness drift (should be ~0 / ~0)")
for name, fn in [("saturation +50", lambda a: grade.saturation(a, 50)), ("saturation -50", lambda a: grade.saturation(a, -50)), ("vibrance +50", lambda a: grade.vibrance(a, 50))]:
    o = lab(fn(P.copy()))[:, 0]; d = np.abs(dh(hue(o[:, 1], o[:, 2]), H0)); mask = C0 > 20
    cr = np.hypot(o[:, 1], o[:, 2]) / np.maximum(C0, 1e-3)
    print(f"  {name:16s} hue drift {d[mask].mean():5.2f}° (worst {d[mask].max():4.1f}°)  lightness drift {np.abs(o[:,0]-Lab0[:,0])[mask].mean():4.2f}  chroma x{np.median(cr[mask]):.2f} (lowest {cr[mask].min():.2f} highest {cr[mask].max():.2f})")

print("\n=== C. HSL mixer, one family at a time: does +50 hue move that family by ~15° and leave the others alone?")
names = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
for ch, label in ((0, "hue +50 (expect +15°)"), (1, "saturation +50 (expect chroma up)"), (2, "luminance +50 (expect L up)")):
    print("  " + label)
    for nm in names:
        mix = {n: [0.0, 0.0, 0.0] for n in names}; mix[nm][ch] = 50.0
        o = lab(grade.mixer(P.copy(), mix))[:, 0]
        hh = hue(o[:, 1], o[:, 2]); centre = {"red": 10, "orange": 45, "yellow": 95, "green": 140, "aqua": 195, "blue": 265, "purple": 305, "magenta": 345}[nm]
        near = (np.abs(dh(H0, centre)) < 18) & (C0 > 25); far = (np.abs(dh(H0, centre)) > 60) & (C0 > 25)
        if ch == 0: mv, lk = np.median(dh(hh, H0)[near]) if near.any() else float('nan'), np.abs(dh(hh, H0)[far]).max()
        elif ch == 1: mv, lk = np.median(np.hypot(o[:, 1], o[:, 2])[near] / C0[near]) if near.any() else float('nan'), np.abs(np.hypot(o[:, 1], o[:, 2]) / C0 - 1)[far].max()
        else: mv, lk = np.median((o[:, 0] - Lab0[:, 0])[near]) if near.any() else float('nan'), np.abs(o[:, 0] - Lab0[:, 0])[far].max()
        side = ""
        if ch == 0 and near.any(): side = f"  lightness drift {np.median(o[:,0]-Lab0[:,0])[()] if False else np.median((o[:,0]-Lab0[:,0])[near]):+.1f}"
        print(f"    {nm:8s} moves {mv:+7.2f}   leaks into far colours up to {lk:5.2f}{side}")

print("\n=== D. Temperature / tint: neutral stays neutral in lightness? warm = more yellow-orange?")
grey = np.linspace(0.05, 0.95, 19, dtype=np.float32).reshape(-1, 1, 1) * np.ones((1, 1, 3), np.float32)
for t in (-40, 40):
    o = colorcal.apply_gains(grey.copy(), np.ones(3, np.float32), t, 0.0); lo = lab(o)[:, 0]; l0 = lab(grey)[:, 0]
    print(f"  temperature {t:+d}: grey ramp b* {np.median(lo[:,2]):+5.1f} (warm should be +), lightness change {np.median(lo[:,0]-l0[:,0]):+5.2f}, worst {np.abs(lo[:,0]-l0[:,0]).max():.2f}")

print("\n=== E. Colour-grading wheels: does the tint land on the chosen hue, and only in its tonal range?")
for rng, hh in (("shadows", 215), ("highs", 40), ("mids", 130)):
    g = Grading(); setattr(g, rng, Wheel(hue=hh, sat=40, lum=0)); g.blending = 50
    ramp = np.linspace(0.04, 0.96, 24, dtype=np.float32).reshape(-1, 1, 1) * np.ones((1, 1, 3), np.float32)
    o = lab(grade.apply_wheels(ramp.copy(), g))[:, 0]; hres = hue(o[:, 1], o[:, 2]); ch = np.hypot(o[:, 1], o[:, 2])
    k = int(np.argmax(ch)); lt = {"shadows": (0, 8), "highs": (16, 24), "mids": (8, 16)}[rng]
    print(f"  {rng:8s} wheel hue {hh}°: strongest tint at ramp step {k}/24 with hue {hres[k]:6.1f}° (chroma {ch[k]:.1f}); chroma in its own third {ch[lt[0]:lt[1]].mean():.1f} vs the other thirds {np.delete(ch, range(lt[0], lt[1])).mean():.1f}")

print("\n=== F. Banding: a smooth gradient through typical edits, quantised to 8 bit (fraction of levels used; 1.0 = no gaps)")
grad = np.tile(np.linspace(0, 1, 4096, dtype=np.float32)[None, :, None], (8, 1, 3))
def used(a):
    q = np.round(np.clip(a, 0, 1)[0, :, 1] * 255).astype(int); return len(np.unique(q)) / 256.0
for name, fn in [("none", lambda a: a), ("contrast +30, shadows +40, highlights -40", lambda a: grade.highlights(grade.shadows(grade.contrast(a, 30), 40, tone_ctx(a)), -40, tone_ctx(a))),
                 ("exposure +50", lambda a: grade.exposure(a, 50)), ("strong S-curve (contrast +80)", lambda a: grade.contrast(a, 80))]:
    print(f"  {name:44s} levels used {used(fn(grad.copy())):.2f}")
