"""fitting.py — grading preset extraction by gradient descent.

Pure functions, stdlib + torch only (no scipy). Everything runs on the
device of the input tensors (CPU or GPU).

The fit optimizes (Adam, lr ~0.02 with decay), in the order of the
grading.py pipeline:
  1. tone_curve: one 6-point curve PER RGB CHANNEL, with FIXED x positions
     [0, 51, 102, 153, 204, 255]; only the y values move, reparametrized as
     a cumulative sum of softplus to guarantee monotonicity (no point
     crossing possible). Per-channel curves capture color casts — e.g. a
     vintage look that lifts red blacks while crushing the blue channel —
     that a single shared curve cannot represent. When the three fitted
     curves agree, they are collapsed into a single shared curve in the
     written preset. Global temperature/tint gains are not fitted
     separately: they are multiplicative per-channel maps, which the
     per-channel curves represent exactly.
  2. saturation (init 1) — mixes channels, not representable by curves.
  3. optional vignette: amount, radius, feather (shape "circle" and
     curve 1.0 fixed; init amount 0) — spatial, not representable by curves.

During the fit, curves are evaluated with piecewise linear interpolation
(torch.searchsorted, differentiable): the Catmull-Rom in grading.py rebuilds
its LUT in pure Python and is not differentiable. The written preset contains
the points, and grading.py's spline applies them at render time; the
linear-fit / spline-render gap is on the order of the fit noise (documented
in the README). Fitted y values may go below 0 or above 255 — that is how a
crushed or clipped channel is represented (the render clamps).

The vignette is a differentiable mirror of the grading.py formula:
apply_vignette short-circuits at amount == 0 (zero gradient at init).
apply_saturation is reused as is.

Grain is NOT fitted (stochastic): after convergence, estimate_grain compares
the residual high-frequency noise (std of the residual after a gaussian blur
of sigma 1.5) between the graded source and the target.
"""

from __future__ import annotations
import logging
import math

import torch
import torch.nn.functional as F

try:
    from . import grading
except ImportError:  # direct import outside the package (tests)
    import grading

logger = logging.getLogger("ComfyUI-Emulsion")

CURVE_XS = (0.0, 51.0, 102.0, 153.0, 204.0, 255.0)  # fixed x positions, 0-255
CHANNELS = "RGB"

# Thresholds below which a parameter is considered neutral and omitted
TOL_CURVE = 1.0        # max deviation from identity, in 0-255 units
CURVE_CHANNEL_TOL = 2.0  # max spread between channels to collapse to one curve
TOL_SAT = 0.01
TOL_VIGNETTE = 0.01
GRAIN_THRESHOLD = 0.002  # high-frequency std gap that triggers grain

_LUM = (0.2126, 0.7152, 0.0722)


def _softplus_inv(y: float) -> float:
    return math.log(math.expm1(y))


def curve_ys_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """Monotonically increasing 0-1 y values: y0 + cumsum of softplus(deltas)."""
    deltas = F.softplus(raw[1:])
    return torch.cat([raw[0:1], raw[0:1] + torch.cumsum(deltas, dim=0)])


def _curve_raw_identity(device, dtype) -> torch.Tensor:
    """Raw parametrization for which curve_ys_from_raw yields the identity curve."""
    step = 1.0 / (len(CURVE_XS) - 1)
    raw = [0.0] + [_softplus_inv(step)] * (len(CURVE_XS) - 1)
    return torch.tensor(raw, device=device, dtype=dtype)


def apply_linear_curve(img: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Piecewise linear interpolation, differentiable with respect to ys."""
    x = img.clamp(0, 1)
    idx = torch.searchsorted(xs, x.contiguous()).clamp(1, xs.numel() - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    t = (x - x0) / (x1 - x0)
    return y0 + (y1 - y0) * t


def _vignette_radius_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    """Normalized radial grid, "circle" shape of grading.apply_vignette."""
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    aspect = w / h
    return torch.sqrt((gx * aspect) ** 2 + gy ** 2) / math.sqrt(aspect ** 2 + 1)


def _vignette(img: torch.Tensor, r: torch.Tensor, amount: torch.Tensor,
              radius: torch.Tensor, feather: torch.Tensor) -> torch.Tensor:
    """Same formula as grading.apply_vignette (curve 1.0), gradient preserved."""
    m = ((r - radius) / feather.clamp(min=1e-3)).clamp(0, 1)
    mask = 1.0 - m * amount
    return img * mask.unsqueeze(0).unsqueeze(-1)


def _downsample(img: torch.Tensor, max_side: int = 512) -> torch.Tensor:
    """Bilinear down to ~max_side px on the long side; unchanged if smaller."""
    _, h, w, _ = img.shape
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / m
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    x = F.interpolate(img.permute(0, 3, 1, 2), size=(nh, nw),
                      mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)


def _rmse255(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b) ** 2).mean().sqrt().item() * 255.0


def estimate_grain(graded: torch.Tensor, target: torch.Tensor,
                   sigma: float = 1.5, threshold: float = GRAIN_THRESHOLD):
    """Grain estimated on the residual high-frequency noise, or None below threshold."""
    hf_g = graded - grading.gaussian_blur(graded, sigma)
    hf_t = target - grading.gaussian_blur(target, sigma)
    sg, st = hf_g.std().item(), hf_t.std().item()
    delta = math.sqrt(max(st * st - sg * sg, 0.0))
    if delta <= threshold:
        return None
    r, g, b = _LUM
    l = (graded[..., 0] * r + graded[..., 1] * g + graded[..., 2] * b).clamp(0, 1)
    mod = (0.4 + 0.6 * 4.0 * l * (1.0 - l)).mean().item()  # apply_grain modulation
    amount = min(delta / max(mod, 1e-3), 0.05)
    return {"amount": round(amount, 4), "mono": True}


def fit_preset(source: torch.Tensor, target: torch.Tensor,
               iterations: int = 400, fit_vignette: bool = True,
               lr: float = 0.02, max_side: int = 512):
    """Recovers a grading preset that transforms source into target.

    source, target: (B,H,W,C) float 0-1. Resolutions may differ (e.g. an
    upscaled target) as long as the aspect ratio matches: the target is then
    resampled to the source resolution.
    Returns (preset, info): preset only contains the significantly
    non-neutral fields; info = rmse_init / rmse_final (/255, measured at the
    source's native resolution), iterations_run, curve_ys (0-255 per
    channel), etc.
    """
    # ComfyUI runs nodes under torch.inference_mode(), where autograd is
    # disabled: exit it explicitly, and clone the inputs in _fit because
    # inference tensors remain unusable by autograd even outside the context.
    with torch.inference_mode(False), torch.enable_grad():
        return _fit(source, target, iterations, fit_vignette, lr, max_side)


def _fit(source, target, iterations, fit_vignette, lr, max_side):
    if source.ndim != 4 or target.ndim != 4:
        raise ValueError("source and target must be IMAGE tensors (B,H,W,C)")
    if source.shape[0] != target.shape[0] or source.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"source {tuple(source.shape)} and target {tuple(target.shape)}: "
            "incompatible batch size or channel count"
        )
    device = source.device
    dtype = torch.float32
    src = source.clone().detach().to(dtype)
    tgt = target.clone().detach().to(device=device, dtype=dtype)

    target_resized = False
    if src.shape[1:3] != tgt.shape[1:3]:
        hs, ws = src.shape[1], src.shape[2]
        ht, wt = tgt.shape[1], tgt.shape[2]
        if abs(ws / hs - wt / ht) > 0.02 * (ws / hs):
            raise ValueError(
                f"source {ws}x{hs} and target {wt}x{ht}: incompatible aspect "
                "ratios — this is probably not the same image"
            )
        tgt = F.interpolate(tgt.permute(0, 3, 1, 2), size=(hs, ws),
                            mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        target_resized = True
        logger.warning(
            "Emulsion Extract: different resolutions (source %dx%d, target "
            "%dx%d) — target resampled to the source; the grain estimate may "
            "be attenuated by the resampling", ws, hs, wt, ht,
        )

    src_s = _downsample(src, max_side)
    tgt_s = _downsample(tgt, max_side)
    _, h, w, _ = src_s.shape

    xs = torch.tensor([x / 255.0 for x in CURVE_XS], device=device, dtype=dtype)
    r_grid = _vignette_radius_grid(h, w, device, dtype)

    curve_raws = [_curve_raw_identity(device, dtype).requires_grad_(True)
                  for _ in CHANNELS]
    sat = torch.ones((), device=device, dtype=dtype, requires_grad=True)
    params = curve_raws + [sat]
    if fit_vignette:
        vig_amount = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
        vig_radius = torch.full((), 0.7, device=device, dtype=dtype, requires_grad=True)
        vig_feather_raw = torch.full((), _softplus_inv(0.5), device=device,
                                     dtype=dtype, requires_grad=True)
        params += [vig_amount, vig_radius, vig_feather_raw]

    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)

    best = math.inf
    last_improve = 0
    iterations_run = 0
    for i in range(iterations):
        opt.zero_grad()
        x = torch.stack(
            [apply_linear_curve(src_s[..., c], xs, curve_ys_from_raw(curve_raws[c]))
             for c in range(len(CHANNELS))], dim=-1)
        x = grading.apply_saturation(x, sat)
        if fit_vignette:
            x = _vignette(x, r_grid, vig_amount, vig_radius,
                          F.softplus(vig_feather_raw))
        loss = F.mse_loss(x, tgt_s)
        loss.backward()
        opt.step()
        sched.step()
        iterations_run = i + 1
        l = loss.item()
        if l < best - 1e-7:
            best = l
            last_improve = i
        elif i - last_improve >= 50:  # early stopping
            break

    with torch.no_grad():
        ys_by_ch = {ch: [y * 255.0 for y in curve_ys_from_raw(raw).tolist()]
                    for ch, raw in zip(CHANNELS, curve_raws)}

        preset = {}
        spread = max(
            max(ys_by_ch[ch][i] for ch in CHANNELS)
            - min(ys_by_ch[ch][i] for ch in CHANNELS)
            for i in range(len(CURVE_XS))
        )
        if spread <= CURVE_CHANNEL_TOL:
            # channels agree: collapse to a single shared curve
            mean_ys = [sum(ys_by_ch[ch][i] for ch in CHANNELS) / len(CHANNELS)
                       for i in range(len(CURVE_XS))]
            if max(abs(y - x) for x, y in zip(CURVE_XS, mean_ys)) > TOL_CURVE:
                preset["tone_curve"] = [[x, round(y, 2)]
                                        for x, y in zip(CURVE_XS, mean_ys)]
        else:
            preset["tone_curve"] = {
                ch: [[x, round(y, 2)] for x, y in zip(CURVE_XS, ys_by_ch[ch])]
                for ch in CHANNELS
            }
        if abs(sat.item() - 1.0) > TOL_SAT:
            preset["saturation"] = round(sat.item(), 3)
        if fit_vignette and abs(vig_amount.item()) > TOL_VIGNETTE:
            preset["vignette"] = {
                "amount": round(vig_amount.item(), 3),
                "shape": "circle",
                "radius": round(vig_radius.item(), 3),
                "feather": round(F.softplus(vig_feather_raw).item(), 3),
                "curve": 1.0,
            }

        # RMSE reported at native resolution, rendered via grading.apply_preset
        # (Catmull-Rom), before the (stochastic) grain estimation.
        graded = grading.apply_preset(src, preset)
        rmse_init = _rmse255(src, tgt)
        rmse_final = _rmse255(graded, tgt)
        grain = estimate_grain(graded, tgt)
        if grain:
            preset["grain"] = grain

    info = {
        "rmse_init": rmse_init,
        "rmse_final": rmse_final,
        "target_resized": target_resized,
        "iterations_run": iterations_run,
        "curve_ys": {ch: [round(y, 3) for y in ys] for ch, ys in ys_by_ch.items()},
        "saturation": sat.item(),
    }
    return preset, info
