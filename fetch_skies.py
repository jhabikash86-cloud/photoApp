"""Download the Poly Haven 'Pure Skies' set (CC0) into skies/. Official API, one file at a time, polite pause. Safe to re-run: finished files are skipped."""
import json, os, sys, time, urllib.parse, urllib.request
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skies")
UA = {"User-Agent": "NeuralEditPro-local-sky-library/1.0 (personal use; CC0 assets)"}
API = "https://api.polyhaven.com/assets?t=hdris&c=" + urllib.parse.quote("pure skies")
os.makedirs(OUT, exist_ok=True)

def get(url, tries=4):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
        except Exception as e:
            print("  retry", i + 1, e, flush=True); time.sleep(3 * (i + 1))
    raise RuntimeError("failed: " + url)

meta = json.loads(get(API))
json.dump(meta, open(os.path.join(OUT, "index.json"), "w"), indent=1)
open(os.path.join(OUT, "LICENSE.txt"), "w").write("Skies from Poly Haven (https://polyhaven.com), licensed CC0 (public domain). Downloaded via the public API for personal use.\n")
ids = sorted(meta)
print(len(ids), "skies", flush=True)
done = 0
for k, i in enumerate(ids, 1):
    path = os.path.join(OUT, i + ".jpg")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        done += 1; continue
    url = f"https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/{i}.jpg"
    data = get(url)
    if len(data) < 1_000_000 or data[:2] != b"\xff\xd8":
        print("  skipped (not a JPEG):", i, len(data), flush=True); continue
    open(path + ".part", "wb").write(data); os.replace(path + ".part", path)
    done += 1
    print(f"[{k}/{len(ids)}] {i} {len(data)/1e6:.1f} MB", flush=True)
    time.sleep(1.0)
print("done", done, "of", len(ids), flush=True)
