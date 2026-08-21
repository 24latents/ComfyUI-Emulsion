"""
grading.py — pure color grading operations for ComfyUI-Emulsion.

Every function operates on ComfyUI IMAGE tensors: (B, H, W, C), float 0-1,
device and dtype preserved. No dependency besides torch.

Fixed pipeline (apply_preset):
  levels -> tone_curve -> brightness -> contrast -> temperature/tint -> hue
  -> saturation -> color_balance -> hsl -> lut -> clarity -> sharpness
  -> bloom -> halation -> chromatic_aberration -> vignette -> grain
  -> lerp(original, result, strength)

All preset fields are optional; absent = neutral.

Full JSON schema:
{
  "name": str, "description": str,
  "levels": {  # 0-255 values; either flat (RGB) or per channel {"R": {...}, "G": {...}, "B": {...}}
    "input_black": 0, "input_white": 255, "gamma": 1.0,
    "output_black": 0, "output_white": 255
  },
  "tone_curve": [[0,0],[128,128],[255,255]],   # (in,out) points 0-255, Catmull-Rom;
                                                # either flat (RGB) or per channel
                                                # {"R": [...], "G": [...], "B": [...]};
                                                # out values may go below 0 / above 255
                                                # to crush or clip a channel
  "brightness": 0.0,                            # additive, -1..1
  "contrast": 1.0, "contrast_pivot": 0.5,
  "temperature": 0.0, "tint": 0.0,              # -100..100
  "hue": 0.0,                                   # degrees -180..180
  "saturation": 1.0,
  "color_balance": {                            # RGB offsets -1..1 per zone
    "shadows": [0,0,0], "midtones": [0,0,0], "highlights": [0,0,0]
  },
  "hsl": {                                      # per hue band mixer
    "reds":     {"hue": 0, "sat": 0, "lum": 0},   # hue in degrees, sat/lum -1..1
    "oranges": {...}, "yellows": {...}, "greens": {...},
    "cyans": {...}, "blues": {...}, "purples": {...}, "magentas": {...}
  },
  "lut": "file_name.cube",                      # looked up in presets/luts/
  "clarity": 0.0,                               # -1..1, local contrast
  "sharpness": 0.0,                             # 0..2
  "bloom":    {"threshold": 0.8, "amount": 0.0, "size": 0.05},
  "halation": {"threshold": 0.85, "amount": 0.0, "size": 0.03,
               "color": [1.0, 0.25, 0.1]},
  "chromatic_aberration": 0.0,                  # 0..0.02 typ.
  "vignette": {"amount": 0.0, "shape": "circle"|"ellipse",
               "radius": 0.7, "feather": 0.5, "curve": 1.0},
  "grain": {"amount": 0.0, "size": 1.0, "mono": true},
  "strength": 1.0
}
"""

from __future__ import annotations
import math
import os
import torch
import torch.nn.functional as F

_LUM = (0.2126, 0.7152, 0.0722)  # Rec.709

# ------------------------------------------------------------------ helpers

def _luminance(img: torch.Tensor) -> torch.Tensor:
    r, g, b = _LUM
    return (img[..., 0:1] * r + img[..., 1:2] * g + img[..., 2:3] * b)

def _to_bchw(img: torch.Tensor) -> torch.Tensor:
    return img.permute(0, 3, 1, 2)

def _to_bhwc(img: torch.Tensor) -> torch.Tensor:
    return img.permute(0, 2, 3, 1)

def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()

def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur on (B,H,W,C). sigma in pixels."""
    if sigma <= 0:
        return img
    x = _to_bchw(img)
    c = x.shape[1]
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    kh = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    kv = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, kh, groups=c)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, kv, groups=c)
    return _to_bhwc(x)

def _rgb_to_hsv(img: torch.Tensor):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc, _ = img.max(dim=-1)
    minc, _ = img.min(dim=-1)
    v = maxc
    delta = maxc - minc
    s = torch.where(maxc > 0, delta / maxc.clamp(min=1e-8), torch.zeros_like(maxc))
    dz = delta.clamp(min=1e-8)
    h = torch.zeros_like(maxc)
    h = torch.where(maxc == r, ((g - b) / dz) % 6.0, h)
    h = torch.where(maxc == g, (b - r) / dz + 2.0, h)
    h = torch.where(maxc == b, (r - g) / dz + 4.0, h)
    h = torch.where(delta > 1e-8, h / 6.0, torch.zeros_like(h))  # 0..1
    return h, s, v

def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h6 = (h % 1.0) * 6.0
    i = torch.floor(h6)
    f = h6 - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i.long() % 6
    r = torch.stack([v, q, p, p, t, v], dim=-1).gather(-1, i.unsqueeze(-1)).squeeze(-1)
    g = torch.stack([t, v, v, q, p, p], dim=-1).gather(-1, i.unsqueeze(-1)).squeeze(-1)
    b = torch.stack([p, p, t, v, v, q], dim=-1).gather(-1, i.unsqueeze(-1)).squeeze(-1)
    return torch.stack([r, g, b], dim=-1)

# ------------------------------------------------------------------- levels

def _levels_one(x: torch.Tensor, p: dict) -> torch.Tensor:
    ib = p.get("input_black", 0) / 255.0
    iw = p.get("input_white", 255) / 255.0
    g = max(p.get("gamma", 1.0), 1e-3)
    ob = p.get("output_black", 0) / 255.0
    ow = p.get("output_white", 255) / 255.0
    t = ((x - ib) / max(iw - ib, 1e-4)).clamp(0, 1)
    t = t ** (1.0 / g)
    return ob + (ow - ob) * t

def apply_levels(img: torch.Tensor, params: dict) -> torch.Tensor:
    """Shader-style levels: flat (RGB) or per channel via R/G/B keys."""
    if any(k in params for k in ("R", "G", "B")):
        out = img.clone()
        for i, ch in enumerate("RGB"):
            if ch in params:
                out[..., i] = _levels_one(img[..., i], params[ch])
        return out
    return _levels_one(img, params)

# --------------------------------------------------------------- tone curve

def _catmull_rom_lut(points, n=1024, device=None, dtype=None) -> torch.Tensor:
    """Builds a 1D LUT from (in,out) points 0-255, Catmull-Rom spline."""
    pts = sorted((float(a) / 255.0, float(b) / 255.0) for a, b in points)
    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < 1:
        pts.append((1.0, pts[-1][1]))
    if len(pts) < 2:
        pts = [(0.0, pts[0][1]), (1.0, pts[0][1])] if pts else [(0.0, 0.0), (1.0, 1.0)]
    ghost0 = (2 * pts[0][0] - pts[1][0], 2 * pts[0][1] - pts[1][1])
    ghost1 = (2 * pts[-1][0] - pts[-2][0], 2 * pts[-1][1] - pts[-2][1])
    ext = [ghost0] + pts + [ghost1]  # ghost points by reflection
    xs = torch.linspace(0, 1, n)
    ys = torch.empty(n)
    seg = 0
    for j, x in enumerate(xs.tolist()):
        while seg < len(pts) - 2 and x > pts[seg + 1][0]:
            seg += 1
        p0, p1, p2, p3 = ext[seg], ext[seg + 1], ext[seg + 2], ext[seg + 3]
        span = max(p2[0] - p1[0], 1e-6)
        t = (x - p1[0]) / span
        t2, t3 = t * t, t * t * t
        ys[j] = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                       + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
    return ys.clamp(0, 1).to(device=device, dtype=dtype)

def apply_tone_curve(img: torch.Tensor, points) -> torch.Tensor:
    """Shared curve (list of points) or per channel via R/G/B keys."""
    if isinstance(points, dict):
        out = img.clone()
        for i, ch in enumerate("RGB"):
            if ch in points:
                lut = _catmull_rom_lut(points[ch], device=img.device, dtype=img.dtype)
                idx = (img[..., i].clamp(0, 1) * (lut.numel() - 1)).round().long()
                out[..., i] = lut[idx]
        return out
    lut = _catmull_rom_lut(points, device=img.device, dtype=img.dtype)
    idx = (img.clamp(0, 1) * (lut.numel() - 1)).round().long()
    return lut[idx]

# ------------------------------------------------- brightness / contrast

def apply_brightness(img: torch.Tensor, amount: float) -> torch.Tensor:
    return img + amount

def apply_contrast(img: torch.Tensor, amount: float, pivot: float = 0.5) -> torch.Tensor:
    return (img - pivot) * amount + pivot

# ------------------------------------------------- white balance / tint

def apply_temperature_tint(img: torch.Tensor, temperature: float, tint: float) -> torch.Tensor:
    """temperature > 0 = warmer (R+, B-); tint > 0 = magenta (G-). Scale -100..100."""
    t = temperature / 100.0 * 0.3
    g = tint / 100.0 * 0.3
    gains = torch.tensor([1.0 + t, 1.0 - g * 0.5, 1.0 - t],
                         device=img.device, dtype=img.dtype)
    return img * gains

def apply_hue(img: torch.Tensor, degrees: float) -> torch.Tensor:
    if abs(degrees) < 1e-4:
        return img
    h, s, v = _rgb_to_hsv(img.clamp(0, 1))
    return _hsv_to_rgb(h + degrees / 360.0, s, v)

def apply_saturation(img: torch.Tensor, amount: float) -> torch.Tensor:
    l = _luminance(img)
    return l + (img - l) * amount

# ------------------------------------------------------------ color balance

def _tonal_masks(l: torch.Tensor):
    """Soft shadows/midtones/highlights masks, summing to 1."""
    sh = (1.0 - l).clamp(0, 1) ** 2
    hi = l.clamp(0, 1) ** 2
    mid = (1.0 - sh - hi).clamp(0, 1)
    return sh, mid, hi

def apply_color_balance(img: torch.Tensor, params: dict) -> torch.Tensor:
    l = _luminance(img)
    sh, mid, hi = _tonal_masks(l)
    out = img
    for zone, mask in (("shadows", sh), ("midtones", mid), ("highlights", hi)):
        off = params.get(zone)
        if off:
            off_t = torch.tensor(off, device=img.device, dtype=img.dtype)
            out = out + mask * off_t
    return out

# -------------------------------------------------------------- HSL mixer

_HSL_BANDS = {  # band center in degrees
    "reds": 0, "oranges": 30, "yellows": 60, "greens": 120,
    "cyans": 180, "blues": 240, "purples": 280, "magentas": 320,
}

def apply_hsl(img: torch.Tensor, params: dict) -> torch.Tensor:
    h, s, v = _rgb_to_hsv(img.clamp(0, 1))
    dh = torch.zeros_like(h)
    ds = torch.zeros_like(s)
    dv = torch.zeros_like(v)
    sigma = 30.0 / 360.0  # band width
    for band, center in _HSL_BANDS.items():
        p = params.get(band)
        if not p:
            continue
        d = (h - center / 360.0 + 0.5) % 1.0 - 0.5  # circular distance
        w = torch.exp(-(d ** 2) / (2 * sigma ** 2)) * s.clamp(0, 1)  # saturation-weighted
        dh = dh + w * (p.get("hue", 0) / 360.0)
        ds = ds + w * p.get("sat", 0)
        dv = dv + w * p.get("lum", 0)
    return _hsv_to_rgb(h + dh, (s * (1 + ds)).clamp(0, 1), (v * (1 + dv)).clamp(0, 1))

# --------------------------------------------------------------------- LUT

def load_cube_lut(path: str, device=None, dtype=None) -> torch.Tensor:
    """Loads a 3D .cube -> (S, S, S, 3) tensor indexed [b][g][r]."""
    size = None
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            up = line.upper()
            if up.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
            elif up.startswith(("TITLE", "DOMAIN_", "LUT_1D")):
                continue
            else:
                parts = line.split()
                if len(parts) == 3:
                    data.append([float(x) for x in parts])
    if size is None or len(data) != size ** 3:
        raise ValueError(f"Invalid LUT: {path}")
    lut = torch.tensor(data, device=device, dtype=dtype).view(size, size, size, 3)
    return lut  # .cube order: r varies fastest -> lut[b][g][r]

def apply_lut(img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Trilinear application via 3D grid_sample."""
    b, h, w, _ = img.shape
    vol = lut.permute(3, 0, 1, 2).unsqueeze(0)          # (1, 3, S, S, S) = (N,C,D,H,W) with D=b,H=g,W=r
    vol = vol.expand(b, -1, -1, -1, -1)
    grid = img.clamp(0, 1) * 2.0 - 1.0                   # x=r, y=g, z=b
    grid = torch.stack([grid[..., 0], grid[..., 1], grid[..., 2]], dim=-1)
    grid = grid.view(b, 1, h, w, 3)                      # (N, D=1, H, W, 3)
    out = F.grid_sample(vol, grid, mode="bilinear", align_corners=True)
    return out.view(b, 3, h, w).permute(0, 2, 3, 1)

# --------------------------------------------------- clarity / sharpness

def apply_clarity(img: torch.Tensor, amount: float) -> torch.Tensor:
    """Local contrast: unsharp mask on luminance, wide radius."""
    sigma = max(img.shape[1], img.shape[2]) * 0.02
    l = _luminance(img)
    blur = gaussian_blur(l, sigma)
    return img + (l - blur) * amount

def apply_sharpness(img: torch.Tensor, amount: float) -> torch.Tensor:
    blur = gaussian_blur(img, 1.0)
    return img + (img - blur) * amount

# ------------------------------------------------------ bloom / halation

def _screen(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 - a) * (1.0 - b)

def apply_bloom(img: torch.Tensor, params: dict) -> torch.Tensor:
    amount = params.get("amount", 0.0)
    if amount <= 0:
        return img
    thr = params.get("threshold", 0.8)
    sigma = params.get("size", 0.05) * max(img.shape[1], img.shape[2])
    hi = ((img - thr) / max(1.0 - thr, 1e-4)).clamp(0, 1)
    glow = gaussian_blur(hi, sigma) * amount
    return _screen(img.clamp(0, 1), glow.clamp(0, 1))

def apply_halation(img: torch.Tensor, params: dict) -> torch.Tensor:
    amount = params.get("amount", 0.0)
    if amount <= 0:
        return img
    thr = params.get("threshold", 0.85)
    sigma = params.get("size", 0.03) * max(img.shape[1], img.shape[2])
    color = torch.tensor(params.get("color", [1.0, 0.25, 0.1]),
                         device=img.device, dtype=img.dtype)
    l = _luminance(img)
    hi = ((l - thr) / max(1.0 - thr, 1e-4)).clamp(0, 1)
    glow = gaussian_blur(hi, sigma) * amount * color
    return _screen(img.clamp(0, 1), glow.clamp(0, 1))

# ------------------------------------------------- chromatic aberration

def apply_chromatic_aberration(img: torch.Tensor, amount: float) -> torch.Tensor:
    """Shifts R outward and B inward, radially."""
    if abs(amount) < 1e-6:
        return img
    b, h, w, _ = img.shape
    ys = torch.linspace(-1, 1, h, device=img.device, dtype=img.dtype)
    xs = torch.linspace(-1, 1, w, device=img.device, dtype=img.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    x = _to_bchw(img)
    out = img.clone()
    for c, scale in ((0, 1.0 + amount), (2, 1.0 - amount)):
        grid = base * scale
        s = F.grid_sample(x[:, c:c + 1], grid, mode="bilinear",
                          padding_mode="border", align_corners=True)
        out[..., c] = s[:, 0]
    return out

# ----------------------------------------------------------------- vignette

def apply_vignette(img: torch.Tensor, params: dict) -> torch.Tensor:
    amount = params.get("amount", 0.0)
    if abs(amount) < 1e-6:
        return img
    b, h, w, _ = img.shape
    ys = torch.linspace(-1, 1, h, device=img.device, dtype=img.dtype)
    xs = torch.linspace(-1, 1, w, device=img.device, dtype=img.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    if params.get("shape", "circle") == "circle":
        aspect = w / h
        r = torch.sqrt((gx * aspect) ** 2 + gy ** 2) / math.sqrt(aspect ** 2 + 1)
    else:
        r = torch.sqrt(gx ** 2 + gy ** 2) / math.sqrt(2.0)
    radius = params.get("radius", 0.7)
    feather = max(params.get("feather", 0.5), 1e-3)
    curve = max(params.get("curve", 1.0), 1e-3)
    m = ((r - radius) / feather).clamp(0, 1) ** curve
    mask = 1.0 - m * amount  # amount > 0 darkens, < 0 brightens
    return img * mask.unsqueeze(0).unsqueeze(-1)

# -------------------------------------------------------------------- grain

def apply_grain(img: torch.Tensor, params: dict,
                generator: torch.Generator | None = None) -> torch.Tensor:
    amount = params.get("amount", 0.0)
    if amount <= 0:
        return img
    size = max(params.get("size", 1.0), 0.0)
    mono = params.get("mono", True)
    b, h, w, c = img.shape
    shape = (b, h, w, 1) if mono else (b, h, w, c)
    noise = torch.randn(shape, device=img.device, dtype=img.dtype, generator=generator)
    if size > 1.0:
        noise = gaussian_blur(noise.expand(b, h, w, max(1, noise.shape[-1])), (size - 1.0))
        noise = noise / noise.std().clamp(min=1e-6)
    # modulated grain: most visible in midtones, attenuated at the extremes
    l = _luminance(img).clamp(0, 1)
    mod = 4.0 * l * (1.0 - l)
    return img + noise * amount * (0.4 + 0.6 * mod)

# ---------------------------------------------------------------- pipeline

def apply_preset(img: torch.Tensor, preset: dict,
                 luts_dir: str | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Applies a full preset. img: (B,H,W,C) float 0-1."""
    original = img
    x = img

    if "levels" in preset:
        x = apply_levels(x, preset["levels"])
    if "tone_curve" in preset:
        x = apply_tone_curve(x, preset["tone_curve"])
    if preset.get("brightness"):
        x = apply_brightness(x, preset["brightness"])
    if preset.get("contrast", 1.0) != 1.0:
        x = apply_contrast(x, preset["contrast"], preset.get("contrast_pivot", 0.5))
    if preset.get("temperature") or preset.get("tint"):
        x = apply_temperature_tint(x, preset.get("temperature", 0.0), preset.get("tint", 0.0))
    if preset.get("hue"):
        x = apply_hue(x.clamp(0, 1), preset["hue"])
    if preset.get("saturation", 1.0) != 1.0:
        x = apply_saturation(x, preset["saturation"])
    if "color_balance" in preset:
        x = apply_color_balance(x, preset["color_balance"])
    if "hsl" in preset:
        x = apply_hsl(x.clamp(0, 1), preset["hsl"])
    if preset.get("lut"):
        path = preset["lut"]
        if luts_dir and not os.path.isabs(path):
            path = os.path.join(luts_dir, path)
        lut = load_cube_lut(path, device=x.device, dtype=x.dtype)
        x = apply_lut(x.clamp(0, 1), lut)
    if preset.get("clarity"):
        x = apply_clarity(x, preset["clarity"])
    if preset.get("sharpness"):
        x = apply_sharpness(x, preset["sharpness"])
    if "bloom" in preset:
        x = apply_bloom(x, preset["bloom"])
    if "halation" in preset:
        x = apply_halation(x, preset["halation"])
    if preset.get("chromatic_aberration"):
        x = apply_chromatic_aberration(x, preset["chromatic_aberration"])
    if "vignette" in preset:
        x = apply_vignette(x, preset["vignette"])
    if "grain" in preset:
        x = apply_grain(x, preset["grain"], generator=generator)

    x = x.clamp(0, 1)
    strength = preset.get("strength", 1.0)
    if strength != 1.0:
        # strength > 1 extrapolates the grade; clamping after the lerp bounds
        # the extrapolation without changing behavior for strength <= 1.
        x = (original + (x - original) * strength).clamp(0, 1)
    return x
