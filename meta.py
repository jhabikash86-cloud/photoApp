"""Photo metadata that matters for grading: when and where it was taken, and therefore where the sun was.

Reads the capture time and GPS position from EXIF (HEIC, JPEG, DNG, NEF...) and computes the sun's elevation with
the NOAA solar-position formulas. A low sun means golden hour, a sun below the horizon means twilight, and so on:
a far better cue for grading than guessing from pixels alone. Everything is optional; with no metadata the grader
falls back on the picture itself.
"""
import datetime as dt
import io
import math
import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image

import adjust


@dataclass
class Meta:
    when: Optional[dt.datetime] = None      # local capture time (naive)
    utc_offset_h: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    camera: str = ""
    sun_elevation: Optional[float] = None   # degrees above the horizon (negative = below)
    sun_source: str = ""

    def summary(self) -> dict:
        return {"when": self.when.isoformat(timespec="minutes") if self.when else None,
                "lat": None if self.lat is None else round(self.lat, 3), "lon": None if self.lon is None else round(self.lon, 3),
                "sun_elevation": None if self.sun_elevation is None else round(self.sun_elevation, 1),
                "sun_source": self.sun_source, "camera": self.camera}


def _rat(v) -> float:
    try:
        return float(v)
    except Exception:
        return float(v[0]) / float(v[1])


def _dms(vals, ref) -> Optional[float]:
    try:
        d = _rat(vals[0]) + _rat(vals[1]) / 60 + _rat(vals[2]) / 3600
        return -d if str(ref).upper().startswith(("S", "W")) else d
    except Exception:
        return None


def _exif_of(path: str):
    if adjust.is_raw(path):
        import rawpy
        with rawpy.imread(path) as r:
            th = r.extract_thumb()
            if th.format == rawpy.ThumbFormat.JPEG:
                return Image.open(io.BytesIO(th.data)).getexif()
        return None
    with Image.open(path) as im:
        return im.getexif()


def solar_elevation(lat: float, lon: float, utc: dt.datetime) -> float:
    """Sun elevation in degrees (NOAA approximation, good to about 0.5 degrees)."""
    n = utc.timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1 + (utc.hour - 12) / 24)
    eot = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g) \
        - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g)
    minutes = utc.hour * 60 + utc.minute + utc.second / 60
    true_solar = (minutes + eot + 4 * lon) % 1440
    ha = math.radians(true_solar / 4 - 180)
    la = math.radians(lat)
    return math.degrees(math.asin(math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)))


def read(path: str) -> Meta:
    m = Meta()
    try:
        ex = _exif_of(path)
        if ex is None:
            return m
        sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
        raw_t = sub.get(0x9003) or ex.get(0x0132)
        if raw_t:
            try:
                m.when = dt.datetime.strptime(str(raw_t)[:19], "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
        off = sub.get(0x9011)                      # OffsetTimeOriginal, e.g. "+01:00"
        if off:
            try:
                sign = -1 if str(off).startswith("-") else 1
                hh, mm = str(off).lstrip("+-").split(":")[:2]
                m.utc_offset_h = sign * (int(hh) + int(mm) / 60)
            except Exception:
                pass
        m.camera = f"{ex.get(0x010F, '')} {ex.get(0x0110, '')}".strip()
        gps = ex.get_ifd(0x8825) if hasattr(ex, "get_ifd") else {}
        if gps.get(2) and gps.get(4):
            m.lat, m.lon = _dms(gps[2], gps.get(1, "N")), _dms(gps[4], gps.get(3, "E"))
        if m.when and m.lat is not None and m.lon is not None:
            off_h = m.utc_offset_h if m.utc_offset_h is not None else round(m.lon / 15)
            m.sun_source = "EXIF time + GPS" + ("" if m.utc_offset_h is not None else " (time zone estimated)")
            m.sun_elevation = solar_elevation(m.lat, m.lon, m.when - dt.timedelta(hours=off_h))
    except Exception:
        pass
    return m
