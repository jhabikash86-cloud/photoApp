"""Tune the Drama style knobs against real pro before/after pairs: minimise the mean Lab distance to the pro edit.
    python tune_drama.py <scratchpad dir with ref/ images>"""
import sys, itertools, json
import numpy as np, cv2
from PIL import Image
import engine, drama_looks as dl
R = sys.argv[1] + '/ref'
load = lambda p: np.asarray(Image.open(p).convert('RGB'), np.float32) / 255
def lab(a): return cv2.cvtColor(a, cv2.COLOR_RGB2Lab)
def de(a, b): return float(np.sqrt(((lab(a) - lab(b)) ** 2).sum(-1)).mean())
E = 560
pairs = []
for n in ('lake', 'cloud', 'terrace', 'roots'):
    b = load(f'{R}/{n}_before.jpg'); t = load(f'{R}/{n}_after.jpg')
    an = engine.analyze(f'{R}/{n}_before.jpg'); auto = engine.auto_params(an, 'auto')
    ranked = dl.generate(an)
    h, w = t.shape[:2]; s = E / max(h, w)
    pairs.append((n, an, auto, ranked, cv2.resize(t, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)))
def evaluate(show=False):
    tot_top, tot_best, rows = [], [], []
    for n, an, auto, ranked, t in pairs:
        sc = {}
        for r in ranked[:5] if not show else ranked:
            p, _, _ = dl.build(an, auto, r['id'], 1.0)
            im = np.asarray(engine.render(an, p, E), np.float32) / 255
            if im.shape != t.shape: im = cv2.resize(im, (t.shape[1], t.shape[0]))
            sc[r['id']] = de(im, t)
        top = sc[ranked[0]['id']]; best = min(sc.values()); tot_top.append(top); tot_best.append(best)
        rows.append((n, round(top, 1), round(best, 1), min(sc, key=sc.get)))
    return float(np.mean(tot_best)), float(np.mean(tot_top)), rows
if __name__ == '__main__':
    grid = dict(white=[1.0, 0.0], hl=[1.0, 0.5, 0.0], expo=[1.0, 0.4, 0.0], vib=[0, 15, 30], con=[0, 10, 20], chroma=[1.0, 1.5, 2.0, 2.6], enrich=[1.0, 1.8, 2.6],
                burn=[1.0, 0.5, 0.0], mist=[1.0, 0.4, 0.0], bloom=[1.0, 1.8, 2.6], lift=[0, -8, 8])
    best, bt, rows = evaluate(); print('start', round(best, 2), round(bt, 2), rows, flush=True)
    for sweep in range(2):
        for k, vals in grid.items():
            keep = dl.TUNE[k]
            for v in vals:
                if v == keep: continue
                dl.TUNE[k] = v
                sc, st, rows = evaluate()
                if sc < best - 0.05:
                    best, keep = sc, v; print(f'  {k}={v}: best-of {sc:.2f} top-ranked {st:.2f}', flush=True)
            dl.TUNE[k] = keep
        print('sweep', sweep, json.dumps(dl.TUNE), round(best, 2), flush=True)
    print('FINAL', json.dumps(dl.TUNE), evaluate())
