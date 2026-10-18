"""Row-band processing for per-pixel operations.

A per-pixel operation (exposure, contrast, saturation, colour grading wheels ...) gives each output pixel from the same
input pixel only, so it can run on a horizontal band of the picture at a time and give exactly the same result as on
the whole picture. For a 48-megapixel export this avoids allocating a dozen full-size temporary arrays per operation:
peak memory drops and the working set fits the CPU cache, which is also faster.

`banded` slices every per-pixel argument alongside the picture: arrays with the picture's height (masks, weights) and
context objects that offer `.band(y0, y1)`. Small pictures (previews) are processed in one piece, as before.
"""
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

BAND_PIXELS = 400_000            # pixels per band: its temporaries stay small and CPU-cache friendly
WORKERS = max(1, min(4, (os.cpu_count() or 2) - 2))
_pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="band") if WORKERS > 1 else None
MIN_PIXELS = 3_000_000          # below ~3 MP (every preview) a single pass is already fast and cheap


def banded(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(arr, *args, **kw):
        if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[0] * arr.shape[1] < MIN_PIXELS:
            return fn(arr, *args, **kw)
        h = arr.shape[0]

        def cut(a, y0, y1):
            if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[0] == h:
                return a[y0:y1]
            if hasattr(a, "band"):
                return a.band(y0, y1)
            return a

        rows = max(32, BAND_PIXELS // max(arr.shape[1], 1))
        spans = [(y0, min(h, y0 + rows)) for y0 in range(0, h, rows)]
        out = np.empty(arr.shape, np.float32)

        def run(span):
            y0, y1 = span
            part = arr[y0:y1]
            res = fn(part, *[cut(a, y0, y1) for a in args], **{k: cut(v, y0, y1) for k, v in kw.items()})
            if res is part:
                return False                      # the operation is switched off
            out[y0:y1] = res
            return True

        if not run(spans[0]):                     # decided by the settings, never by the pixels: hand the picture back untouched
            return arr
        rest = spans[1:]
        if _pool is not None and len(rest) > 1:
            list(_pool.map(run, rest))           # numpy releases the GIL on big arrays, so the bands really run side by side
        else:
            for sp in rest:
                run(sp)
        return out
    return wrapper
