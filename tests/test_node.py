"""Tests for the Emulsion Grade node, without launching ComfyUI.

emulsion.py depends only on the stdlib and torch: it imports directly,
nothing from ComfyUI is required (hence nothing to mock).
"""

import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import emulsion
import grading


def _run(img, preset="moody_garden", strength=1.0, seed=0):
    (out,) = emulsion.EmulsionGrade().grade(img, preset, strength, seed)
    return out


def test_preset_scan_contains_moody_garden():
    presets = emulsion.list_presets()
    assert "moody_garden" in presets
    assert presets == sorted(presets)
    widget = emulsion.EmulsionGrade.INPUT_TYPES()["required"]["preset"]
    assert "moody_garden" in widget[0]


def test_execution_shape_and_bounds():
    img = torch.rand(1, 64, 64, 3)
    out = _run(img)
    assert out.shape == (1, 64, 64, 3)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
    assert not torch.equal(out, img), "the preset must modify the image"


def test_unknown_preset_returns_image_unchanged():
    img = torch.rand(1, 64, 64, 3)
    out = _run(img, preset="does_not_exist")
    assert torch.equal(out, img)


def test_seed_reproducibility():
    img = torch.rand(1, 64, 64, 3)
    a = _run(img, seed=123)
    b = _run(img, seed=123)
    c = _run(img, seed=124)
    assert torch.equal(a, b), "the same seed must give a bit-identical output"
    assert not torch.equal(a, c), "different seeds must differ (grain)"


def test_strength_2_consistent_with_apply_preset():
    # The node passes strength directly into the preset: the strength > 1
    # extrapolation must be bit-identical to the library's.
    img = torch.rand(1, 64, 64, 3)
    out = _run(img, strength=2.0, seed=7)
    with open(os.path.join(REPO, "presets", "moody_garden.json"), encoding="utf-8") as f:
        data = json.load(f)
    data["strength"] = 2.0
    gen = torch.Generator(device=img.device)
    gen.manual_seed(7)
    ref = grading.apply_preset(img, data, generator=gen)
    assert torch.equal(out, ref)
    assert out.min() >= 0 and out.max() <= 1
