"""Emulsion Grade — ComfyUI node for JSON-preset color grading.

Depends only on the stdlib and torch: importable and testable without ComfyUI.
"""

import json
import logging
import os

import torch

try:
    from . import grading
except ImportError:  # direct import outside the package (tests)
    import grading

logger = logging.getLogger("ComfyUI-Emulsion")

_HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS_DIR = os.path.join(_HERE, "presets")
LUTS_DIR = os.path.join(PRESETS_DIR, "luts")


def list_presets():
    """Sorted names (without extension) of presets/*.json; ["(none)"] if empty."""
    try:
        names = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(PRESETS_DIR)
            if f.lower().endswith(".json")
        )
    except OSError:
        names = []
    return names if names else ["(none)"]


class EmulsionGrade:
    CATEGORY = "image/postprocessing"
    FUNCTION = "grade"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        # The presets folder is rescanned on every call: ComfyUI calls this
        # method again on browser refresh, so a newly added preset shows up
        # without a restart.
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (list_presets(),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            }
        }

    def grade(self, image, preset, strength, seed):
        path = os.path.join(PRESETS_DIR, f"{preset}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            logger.error(
                "Emulsion Grade: preset '%s' not found or invalid JSON (%s) "
                "— returning the image unchanged", preset, e,
            )
            return (image,)

        data["strength"] = strength
        generator = torch.Generator(device=image.device)
        generator.manual_seed(int(seed))
        try:
            out = grading.apply_preset(image, data, luts_dir=LUTS_DIR, generator=generator)
        except Exception as e:
            logger.error(
                "Emulsion Grade: preset '%s' failed (%s) — returning the image unchanged",
                preset, e,
            )
            return (image,)

        return (out,)
