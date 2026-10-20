"""Edit parameters. Every slider is a real Lightroom-style adjustment (bipolar, -100..+100), not a
percentage of an automatic suggestion. The automatic edit simply sets these values, and the user moves
them from there."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Local-adjustment regions the segmentation can find. People are handled separately.
REGIONS = ["sky", "mountains", "trees", "grass", "water", "buildings", "ground", "foreground", "background", "skin", "other"]
MIXER_COLORS = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]

S = dict(ge=-100, le=100)


class Basic(BaseModel):
    """Global adjustments, applied to the whole picture."""
    temperature: float = Field(0, **S)
    tint: float = Field(0, **S)
    exposure: float = Field(0, **S)      # +-100 = +-3 EV
    contrast: float = Field(0, **S)
    highlights: float = Field(0, **S)
    shadows: float = Field(0, **S)
    whites: float = Field(0, **S)
    blacks: float = Field(0, **S)
    texture: float = Field(0, **S)       # fine detail
    clarity: float = Field(0, **S)       # midtone local contrast
    dehaze: float = Field(0, **S)
    vibrance: float = Field(0, **S)
    saturation: float = Field(0, **S)
    keep_whites: float = Field(75, ge=0, le=100)   # how much of the picture's own white level is restored after highlight recovery (0 = off)


class Region(BaseModel):
    """Local adjustments applied only inside one detected region (like a Lightroom mask)."""
    exposure: float = Field(0, **S)
    contrast: float = Field(0, **S)
    vibrance: float = Field(0, **S)
    saturation: float = Field(0, **S)
    clarity: float = Field(0, **S)
    dehaze: float = Field(0, **S)
    temperature: float = Field(0, **S)
    tint: float = Field(0, **S)
    hue: float = Field(0, **S)           # +-100 = +-30 degrees
    highlights: float = Field(0, **S)
    shadows: float = Field(0, **S)
    # mask controls (Selective Masking)
    enabled: bool = True                           # switch this adjustment off without losing its settings
    mask_strength: float = Field(100, ge=0, le=100)  # opacity of the mask
    feather: float = Field(0, ge=0, le=100)        # soft transition
    refine: float = Field(0, ge=0, le=100)         # edge-aware snap to the picture's real edges
    invert: bool = False                           # apply to everything EXCEPT this region
    strokes: List['Stroke'] = []                   # paint to correct this region: add area, or (erase) remove area


class LocalAdjust(BaseModel):
    """The adjustments a custom mask applies inside itself (same set as an AI region)."""
    exposure: float = Field(0, **S)
    contrast: float = Field(0, **S)
    highlights: float = Field(0, **S)
    shadows: float = Field(0, **S)
    vibrance: float = Field(0, **S)
    saturation: float = Field(0, **S)
    clarity: float = Field(0, **S)
    dehaze: float = Field(0, **S)
    temperature: float = Field(0, **S)
    tint: float = Field(0, **S)
    hue: float = Field(0, **S)


class Stroke(BaseModel):
    """One brush stroke. Points are normalised to the picture (0..1), so a stroke fits at every resolution."""
    pts: List[List[float]] = []
    size: float = Field(25, ge=1, le=100)
    feather: float = Field(60, ge=0, le=100)
    flow: float = Field(100, ge=1, le=100)
    erase: bool = False


class MaskLayer(BaseModel):
    """A mask the user creates: gradient, radial, brush, luminosity range or colour range, with its own adjustments."""
    id: str = ""
    name: str = ""
    kind: str = "linear"                         # linear | radial | brush | luminosity | color
    enabled: bool = True
    invert: bool = False
    strength: float = Field(100, ge=0, le=100)   # opacity of the whole mask
    feather: float = Field(50, ge=0, le=100)     # softness of the edge / transition
    refine: float = Field(0, ge=0, le=100)       # "auto mask": snap the edge to the picture's real edges
    limit: Optional[str] = None                  # only inside this AI region (sky, person, trees ...)
    limit_invert: bool = False                   # ... or everywhere EXCEPT that region
    line: List[float] = [0.5, 0.0, 0.5, 0.45]    # linear: x1,y1 (full effect) -> x2,y2 (none)
    ellipse: List[float] = [0.5, 0.5, 0.3, 0.3, 0.0]   # radial: cx, cy, rx, ry (fractions of width/height), angle (deg)
    strokes: List[Stroke] = []                   # brush
    lum: List[float] = [60.0, 100.0]             # luminosity range (0-100)
    lum_soft: float = Field(25, ge=0, le=100)
    pick: List[float] = [0.5, 0.5]               # colour range: where the colour was picked (0..1)
    tolerance: float = Field(35, ge=1, le=100)
    adj: LocalAdjust = LocalAdjust()


class ToneCurve(BaseModel):
    """Point tone curves (like Lightroom's): the master RGB curve and one per channel. Points are (input, output) in 0..1."""
    rgb: List[List[float]] = [[0.0, 0.0], [1.0, 1.0]]
    r: List[List[float]] = [[0.0, 0.0], [1.0, 1.0]]
    g: List[List[float]] = [[0.0, 0.0], [1.0, 1.0]]
    b: List[List[float]] = [[0.0, 0.0], [1.0, 1.0]]


class Drama(BaseModel):
    """The 'Drama' look: what separates a pro landscape from a correct one. Colour is poured over ALL tones (including greys), the sky is
    darkened but keeps its cloud texture, light gets bloom and mist, and the detail is pushed. Every number is a normal editable slider."""
    enabled: bool = False
    amount: float = Field(100, ge=0, le=150)           # master strength
    mood: float = Field(80, ge=0, le=100)              # how strongly the three colours below replace the picture's own cast
    sh_hue: float = Field(250, ge=0, le=360)           # colour of the shadows (Lab hue angle: 15 pink-red, 60 orange, 95 yellow, 140 green, 215 teal, 272 blue, 295 violet, 340 magenta)
    sh_chroma: float = Field(12, ge=0, le=50)
    mid_hue: float = Field(250, ge=0, le=360)
    mid_chroma: float = Field(20, ge=0, le=50)
    hi_hue: float = Field(250, ge=0, le=360)
    hi_chroma: float = Field(18, ge=0, le=50)
    protect: float = Field(60, ge=0, le=100)           # leave already-colourful pixels (foliage, flowers) alone
    enrich: float = Field(25, ge=-100, le=100)         # make the colours that are there richer (gamut-limited)
    skin_safe: bool = True                             # never tint skin
    sky_burn: float = Field(60, ge=0, le=100)          # graduated darkening of the sky toward the top
    sky_texture: float = Field(50, ge=0, le=100)       # contrast inside the clouds (keeps them from going flat)
    sky_model: float = Field(0, ge=0, le=100)          # cloud modelling: a flat bright veil gets real light and shadow (contrast curve inside the sky)
    sky_colour: float = Field(40, ge=0, le=100)        # extra colour in the sky
    sky_hue: float = Field(60, ge=0, le=360)           # the sky's own colour (an orange sky over green land, a blue sky over a golden foreground)
    sky_chroma: float = Field(0, ge=0, le=50)          # 0 = the sky simply takes the picture's colours
    bloom: float = Field(30, ge=0, le=100)             # glow around the brightest light
    bloom_hue: float = Field(70, ge=0, le=360)
    mist: float = Field(30, ge=0, le=100)              # luminous haze on the horizon + atmospheric depth
    depth_haze: float = Field(25, ge=0, le=100)        # atmospheric depth: far ground sinks into haze (turned down automatically in already-hazy scenes)
    glow: float = Field(0, ge=0, le=100)               # warm glow radiating from a sun / lamp (only if one is found)
    rays: float = Field(0, ge=0, le=100)               # soft light rays from the same source
    sun_x: float = Field(-1, ge=-1, le=1)              # where the light is (fraction of width / height); -1 = none found
    sun_y: float = Field(-1, ge=-1, le=1)
    white: float = Field(0, ge=0, le=1)                # light anchor: the brightest tones are restored toward this level (the photo's own brightness), 0 = off
    mist_y: float = Field(-1, ge=-1, le=1)             # where the haze lies (fraction of height); -1 = find the horizon automatically
    detail: float = Field(50, ge=0, le=100)            # recorded here so the look can be re-applied; the sliders live in Basic
    name: str = ""                                     # which generated look this is


class LongExposure(BaseModel):
    """Digital long exposure / ND-filter look, region by region (see longexp.py)."""
    enabled: bool = False
    water: float = Field(0, ge=0, le=100)               # silky water: ripples averaged away, reflections and rocks kept
    water_dir: str = "auto"                              # auto (follow the flow of a stream, else glass) | horizontal | vertical
    clouds: float = Field(0, ge=0, le=100)              # clouds stretched into streaks along the wind
    cloud_angle: float = Field(0, ge=-90, le=90)        # wind direction in degrees (0 = horizontal, + = tilted clockwise)
    grad_nd: float = Field(0, ge=0, le=100)             # graduated ND filter: the sky held down toward the top, the land untouched


class SkyReplace(BaseModel):
    """Replace the sky with one from the sky library, matched to the photo. Everything adaptive is decided when the sky is chosen
    (see sky_replace.plan) and stored here, so the render itself is cheap and every number stays editable."""
    id: Optional[str] = None                          # library sky (file name without .jpg); None = off
    blend: float = Field(100, ge=0, le=100)           # 0 = original sky, 100 = fully replaced
    azimuth: float = Field(0, ge=-180, le=180)        # turn the panorama left / right (degrees): moves the clouds and the sun
    lift: float = Field(0, ge=-30, le=40)             # look higher / lower in the panorama (degrees of camera pitch)
    zoom: float = Field(100, ge=50, le=200)           # 100 = the photo's field of view; larger = bigger clouds
    exposure: float = Field(0, ge=-100, le=100)       # brightness of the new sky (auto-matched when chosen)
    horizon: float = Field(40, ge=0, le=100)          # blend of the original horizon haze into the new sky
    relight: float = Field(50, ge=0, le=100)          # how much the landscape takes on the new sky's colour and light
    reflect: float = Field(60, ge=0, le=100)          # how much water takes on the new sky's colour (reflections)
    edge: float = Field(50, ge=0, le=100)             # edge cleanup against trees / hills (decontamination)


class LutSetting(BaseModel):
    """A creative 3D LUT (built-in or imported .cube) and how strongly it is applied."""
    id: Optional[str] = None                      # "builtin:teal_orange", "user:my_look", or None for no LUT
    intensity: float = Field(100, ge=0, le=100)


class People(BaseModel):
    smooth: float = Field(0, ge=0, le=100)        # skin softening (frequency separation)
    brighten: float = Field(0, **S)               # face brightness
    warmth: float = Field(0, **S)                 # skin warmth
    tone_fix: float = Field(0, ge=0, le=100)      # pull orange/red casts back to natural skin
    eyes: float = Field(0, ge=0, le=100)          # eye pop
    lips: float = Field(0, ge=0, le=100)
    sculpt: float = Field(0, ge=0, le=100)        # light and shadow on the face (dodge & burn)
    lift: float = Field(0, ge=0, le=100)          # lift a backlit/dark person
    exposure: float = Field(0, **S)               # whole person
    enabled: bool = True


class Effects(BaseModel):
    bokeh: float = Field(0, ge=0, le=100)          # background blur amount (portraits)
    bokeh_size: float = Field(50, ge=0, le=100)    # how large the blur discs are
    bokeh_highlights: float = Field(40, ge=0, le=100)  # light-point discs in the blur
    glow: float = Field(0, ge=0, le=100)           # soft dreamy bloom
    vignette: float = Field(0, **S)                # - darken, + lighten edges
    vignette_mid: float = Field(50, ge=0, le=100)
    grain: float = Field(0, ge=0, le=100)
    sharpen: float = Field(0, ge=0, le=150)
    denoise: float = Field(0, ge=0, le=100)          # luminance noise reduction
    denoise_color: float = Field(0, ge=0, le=100)    # colour (chroma) noise reduction
    bokeh_focus: float = Field(50, ge=0, le=100)     # depth of the in-focus zone around the subject


class DodgeBurn(BaseModel):
    """Mask-aware lighting: smooth, low-frequency exposure maps (dodge = brighten, burn = darken)."""
    enabled: bool = True
    strength: float = Field(100, ge=0, le=150)       # master scale for all of the below
    dodge: float = Field(0, ge=0, le=100)            # brighten (up to about +0.6 EV)
    dodge_tones: float = Field(40, ge=0, le=100)     # where dodge acts: 0 shadows .. 50 midtones .. 100 highlights
    burn: float = Field(0, ge=0, le=100)             # darken (up to about -0.9 EV)
    burn_tones: float = Field(70, ge=0, le=100)      # where burn acts
    separation: float = Field(0, ge=0, le=100)       # darker background + slightly lifted subject
    form: float = Field(0, ge=0, le=100)             # emphasise the existing light and shadow on face/skin
    feather: float = Field(50, ge=0, le=100)         # smoothness of the maps (more = softer, no visible edges)
    protect_face: bool = True                        # burn never darkens skin; dodge is held back on bright skin


class Wheel(BaseModel):
    """One colour-grading wheel: the tint (hue in degrees, saturation) and a luminance nudge for a tonal range."""
    hue: float = Field(0, ge=0, le=360)
    sat: float = Field(0, ge=0, le=100)
    lum: float = Field(0, **S)


class Grading(BaseModel):
    """Colour grading like Lightroom's: tint the shadows, midtones and highlights separately."""
    shadows: Wheel = Wheel()
    mids: Wheel = Wheel()
    highs: Wheel = Wheel()
    glob: Wheel = Wheel()                          # tints everything
    blending: float = Field(50, ge=0, le=100)      # how widely the ranges overlap
    balance: float = Field(0, **S)                 # - favours shadows, + favours highlights


class Curve(BaseModel):
    """Parametric tone curve (like Lightroom's): each slider bends the curve inside one tonal range."""
    shadows: float = Field(0, **S)
    darks: float = Field(0, **S)
    lights: float = Field(0, **S)
    highlights: float = Field(0, **S)


def _zero_mixer() -> Dict[str, List[float]]:
    return {c: [0.0, 0.0, 0.0] for c in MIXER_COLORS}      # hue, saturation, luminance


def _regions() -> Dict[str, Region]:
    return {r: Region() for r in REGIONS}


class EditParams(BaseModel):
    look: str = "dreamy"
    intensity: float = Field(1.0, ge=0, le=1.5)   # overall strength of the whole edit
    wb_mode: str = "asshot"                       # asshot | auto | neutral | warm | cool | skin ("asshot" = untouched)
    basic: Basic = Basic()
    mixer: Dict[str, List[float]] = Field(default_factory=_zero_mixer)
    balance_shadows: List[float] = [0, 0, 0]      # (cyan<->red, magenta<->green, yellow<->blue)
    balance_mids: List[float] = [0, 0, 0]
    balance_highs: List[float] = [0, 0, 0]
    regions: Dict[str, Region] = Field(default_factory=_regions)
    people: People = People()
    effects: Effects = Effects()
    dodge_burn: DodgeBurn = DodgeBurn()
    curve: Curve = Curve()
    tone: ToneCurve = ToneCurve()
    masks: List[MaskLayer] = []
    lut: LutSetting = LutSetting()
    sky_replace: SkyReplace = SkyReplace()
    drama: Drama = Drama()
    longexp: LongExposure = LongExposure()
    grading: Grading = Grading()
    grading_enabled: bool = True                  # HSL mixer + colour balance + curve


Region.model_rebuild()
