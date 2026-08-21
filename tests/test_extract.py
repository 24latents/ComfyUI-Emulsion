"""Tests for Emulsion Extract: fitting.py (pure functions) and the node.

The real-pair round-trip depends on tests/A.png and tests/B.PNG (skipped
when absent), like the regression test in test_grading.py.
"""

import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import extract
import fitting
import grading

DATA_DIR = os.path.join(REPO, "tests")
IMG_A = os.path.join(DATA_DIR, "A.png")
IMG_B = os.path.join(DATA_DIR, "B.PNG")


def _load_png(path):
    np = pytest.importorskip("numpy")
    PILImage = pytest.importorskip("PIL.Image")
    return torch.from_numpy(
        np.asarray(PILImage.open(path).convert("RGB"), dtype=np.float32) / 255
    ).unsqueeze(0)


def test_synthetic_round_trip():
    torch.manual_seed(0)
    img = torch.rand(1, 128, 128, 3)
    preset = {
        "tone_curve": [[0, 12], [128, 122], [255, 242]],  # gentle S
        "saturation": 0.8,
        "vignette": {"amount": 0.3},
    }
    target = grading.apply_preset(img, preset)
    fitted, info = fitting.fit_preset(img, target, iterations=600, fit_vignette=True)
    assert info["rmse_final"] < 2.0, f"RMSE {info['rmse_final']:.2f}/255"
    assert info["rmse_final"] < info["rmse_init"]
    assert "saturation" in fitted and abs(fitted["saturation"] - 0.8) < 0.1
    assert "vignette" in fitted and abs(fitted["vignette"]["amount"] - 0.3) < 0.1


@pytest.mark.skipif(
    not (os.path.exists(IMG_A) and os.path.exists(IMG_B)),
    reason="reference images tests/A.png and tests/B.PNG are absent",
)
def test_real_round_trip_A_to_B():
    A = _load_png(IMG_A)
    B = _load_png(IMG_B)
    fitted, info = fitting.fit_preset(A, B, iterations=800, fit_vignette=True)
    assert info["rmse_final"] < 6.0, (
        f"RMSE {info['rmse_final']:.2f}/255 (original scipy fit: 5.4)"
    )


def test_identity_yields_near_empty_preset():
    torch.manual_seed(1)
    img = torch.rand(1, 96, 96, 3)
    fitted, info = fitting.fit_preset(img, img, iterations=400, fit_vignette=True)
    assert info["rmse_final"] < 1.0, f"RMSE {info['rmse_final']:.2f}/255"
    # every field below the thresholds -> empty preset
    assert fitted == {}, f"unexpected non-neutral fields: {sorted(fitted)}"


def test_tone_curve_monotonicity():
    # Adversarial target (inverted image): the curve wants to decrease,
    # the cumulative-softplus reparametrization must prevent it.
    torch.manual_seed(2)
    img = torch.rand(1, 96, 96, 3)
    _, info = fitting.fit_preset(img, 1.0 - img, iterations=300, fit_vignette=False)
    ys = info["curve_ys"]
    assert all(b >= a - 1e-6 for a, b in zip(ys, ys[1:])), f"non-monotonic curve: {ys}"


def test_different_resolutions_target_resampled():
    # Real-world case: the target is the same image passed through a 2x upscale.
    # Smoothed image: on pure white noise, the bilinear up/down round trip
    # would destroy the high frequencies and set an artificial RMSE floor.
    torch.manual_seed(4)
    img = grading.gaussian_blur(torch.rand(1, 64, 96, 3), 2.0)
    graded = grading.apply_preset(img, {"saturation": 0.7})
    target_hi = F.interpolate(graded.permute(0, 3, 1, 2), scale_factor=2,
                              mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    fitted, info = fitting.fit_preset(img, target_hi, iterations=200, fit_vignette=False)
    assert info["target_resized"] is True
    assert "saturation" in fitted and abs(fitted["saturation"] - 0.7) < 0.05
    assert info["rmse_final"] < 3.0, f"RMSE {info['rmse_final']:.2f}/255"


def test_incompatible_aspect_ratio_raises():
    img = torch.rand(1, 64, 64, 3)
    tgt = torch.rand(1, 64, 128, 3)
    with pytest.raises(ValueError, match="aspect"):
        fitting.fit_preset(img, tgt, iterations=50)


def test_fit_under_inference_mode():
    # ComfyUI runs nodes under torch.inference_mode(): images are inference
    # tensors there and autograd is disabled. The fit must still work
    # (reproduces the real execution context).
    torch.manual_seed(5)
    with torch.inference_mode():
        img = grading.gaussian_blur(torch.rand(1, 48, 48, 3), 2.0)
        target = grading.apply_preset(img, {"saturation": 0.7})
        fitted, info = fitting.fit_preset(img, target, iterations=100, fit_vignette=False)
    assert "saturation" in fitted and abs(fitted["saturation"] - 0.7) < 0.1
    assert info["rmse_final"] < info["rmse_init"]


def test_node_writes_file_handles_collisions_returns_json(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "PRESETS_DIR", str(tmp_path))
    torch.manual_seed(3)
    img = torch.rand(1, 48, 48, 3)
    target = grading.apply_preset(img, {"saturation": 0.7})
    node = extract.EmulsionExtract()

    (text1,) = node.extract(img, target, "My Preset !", 50, True)
    data = json.loads(text1)
    assert data["name"] == "My Preset !"
    assert "description" in data
    path1 = tmp_path / "my_preset.json"
    assert path1.exists()
    assert json.loads(path1.read_text(encoding="utf-8")) == data

    # name collision -> _2 suffix
    (text2,) = node.extract(img, target, "My Preset !", 50, True)
    assert (tmp_path / "my_preset_2.json").exists()
    assert json.loads(text2)["name"] == "My Preset !"

    # the extracted preset is usable by apply_preset
    out = grading.apply_preset(img, json.loads(text1))
    assert out.shape == img.shape
