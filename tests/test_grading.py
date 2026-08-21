"""Test suite for grading.py, with repo-relative paths.

The regression test depends on two reference images (A.png, B.PNG) expected
in tests/; it is skipped when they are absent.
"""

import json
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import grading as G

DATA_DIR = os.path.join(REPO, "tests")
IMG_A = os.path.join(DATA_DIR, "A.png")
IMG_B = os.path.join(DATA_DIR, "B.PNG")
PRESET_MOODY = os.path.join(REPO, "presets", "moody_garden.json")


@pytest.fixture
def img():
    torch.manual_seed(0)
    return torch.rand(2, 64, 80, 3)


def test_identity_empty_preset(img):
    out = G.apply_preset(img, {})
    assert torch.allclose(out, img), "empty preset is not neutral"


def test_identity_explicit_neutral(img):
    neutral = {
        "levels": {"input_black": 0, "input_white": 255, "gamma": 1.0,
                   "output_black": 0, "output_white": 255},
        "tone_curve": [[0, 0], [255, 255]],
        "brightness": 0.0, "contrast": 1.0, "temperature": 0.0, "tint": 0.0,
        "hue": 0.0, "saturation": 1.0,
        "color_balance": {"shadows": [0, 0, 0], "midtones": [0, 0, 0], "highlights": [0, 0, 0]},
        "hsl": {}, "clarity": 0.0, "sharpness": 0.0,
        "bloom": {"amount": 0.0}, "halation": {"amount": 0.0},
        "chromatic_aberration": 0.0, "vignette": {"amount": 0.0}, "grain": {"amount": 0.0},
        "strength": 1.0,
    }
    out = G.apply_preset(img, neutral)
    err = (out - img).abs().max().item()
    assert err < 2e-3, f"neutral preset is not the identity, err={err}"


OPS = {
    "levels": {"levels": {"input_black": 30, "output_white": 220}},
    "tone_curve": {"tone_curve": [[0, 20], [128, 110], [255, 235]]},
    "contrast": {"contrast": 1.3},
    "temp": {"temperature": 40},
    "hue": {"hue": 30},
    "sat": {"saturation": 0.5},
    "cb": {"color_balance": {"shadows": [0.05, 0, -0.05]}},
    "hsl": {"hsl": {"greens": {"hue": 20, "sat": -0.3}}},
    "clarity": {"clarity": 0.5},
    "sharp": {"sharpness": 1.0},
    "bloom": {"bloom": {"amount": 0.5, "threshold": 0.6}},
    "halation": {"halation": {"amount": 0.5, "threshold": 0.5}},
    "ca": {"chromatic_aberration": 0.01},
    "vignette": {"vignette": {"amount": 0.5, "radius": 0.3}},
    "grain": {"grain": {"amount": 0.05}},
}


@pytest.mark.parametrize("name", sorted(OPS))
def test_operation_active_and_bounded(img, name):
    out = G.apply_preset(img, OPS[name])
    assert out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 1
    assert not torch.allclose(out, img), f"{name} has no effect"


def test_strength_lerp(img):
    p = {"contrast": 1.5}
    full = G.apply_preset(img, p)
    half = G.apply_preset(img, {**p, "strength": 0.5})
    assert torch.allclose(half, (img + full) / 2, atol=1e-5)


def test_strength_extrapolation(img):
    p = {"contrast": 1.3}
    full = G.apply_preset(img, p)
    ext = G.apply_preset(img, {**p, "strength": 2.0})
    expected = (img + (full - img) * 2.0).clamp(0, 1)
    assert torch.allclose(ext, expected, atol=1e-6)
    assert ext.min() >= 0 and ext.max() <= 1
    assert not torch.allclose(ext, full), "strength 2 must go beyond the full grade"


@pytest.mark.skipif(
    not (os.path.exists(IMG_A) and os.path.exists(IMG_B)),
    reason="reference images tests/A.png and tests/B.PNG are absent",
)
def test_regression_moody_garden():
    np = pytest.importorskip("numpy")
    PILImage = pytest.importorskip("PIL.Image")
    with open(PRESET_MOODY, "r", encoding="utf-8") as f:
        preset = json.load(f)
    A = torch.from_numpy(
        np.asarray(PILImage.open(IMG_A).convert("RGB"), dtype=np.float32) / 255
    ).unsqueeze(0)
    B = torch.from_numpy(
        np.asarray(PILImage.open(IMG_B).convert("RGB"), dtype=np.float32) / 255
    ).unsqueeze(0)
    # drop grain (stochastic) and vignette for the measurement
    p2 = dict(preset)
    p2.pop("grain", None)
    p2.pop("vignette", None)
    rec = G.apply_preset(A, p2)
    rmse = ((rec - B) ** 2).mean().sqrt().item() * 255
    assert rmse < 6.0, f"RMSE A->B with the preset: {rmse:.2f}/255 (original fit: 5.42)"
